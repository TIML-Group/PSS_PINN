"""
refine_dp.py
============
Second-stage refinement: take a trained checkpoint and push the control effort
||dP|| down as far as it will go WITHOUT giving up rescue rate.

    python refine_dp.py --ckpt receding_output/best.pt --data data_mm

Why this is a constrained problem
---------------------------------
Minimising ||dP|| on its own has a trivial optimum: dP = 0, rescue = 0. So the
objective is

    minimise   ||dP||^2
    subject to rescue_rate >= floor

implemented with an adaptive multiplier (dual ascent) rather than a fixed
--w_dp_norm:

    loss = ||dP||^2 + lambda * (failure surrogate + violation)

    lambda <- lambda * exp(eta * (floor - rescue))     each epoch

If rescue slips below the floor, lambda climbs and the envelope term dominates
until rescue recovers; if rescue sits comfortably above, lambda decays and the
norm term is free to keep shrinking. lambda settles at the exchange rate where
the constraint is exactly tight, which is the point of minimum effort -- and it
finds it without a manual sweep of --w_dp_norm.

The floor defaults to the loaded model's own measured rescue rate (minus a
small tolerance), so the refinement can never end up worse than what it started
from.

Stopping
--------
Selection is lexicographic, not a weighted sum: a checkpoint counts as better
only if it MEETS the rescue floor and has a lower ||dP|| than the best feasible
one so far (by at least --min_delta). Patience counts epochs with no such
improvement. This is the stopping condition the main script could not provide,
because there rescue and norm were mixed into one score and the norm term was
always dominated.

By default only the dP head trains; the identification stack is frozen, so the
window, hypothesis, alpha and trajectory results of the loaded checkpoint are
preserved exactly. --unfreeze_all lifts that.

Every epoch's (rescue, ||dP||) pair is written to refine_pareto.json -- that
trace is the effort/reliability Pareto front.
"""

import argparse, json, os, sys, time

import numpy as np
import torch
import torch.nn as nn

from ieee9_physics import IEEE9, HYPOTHESES
from pinn_model import FaultEncoder, RolloutController

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--data', required=True)
ap.add_argument('--base', default='ieee9_base.mat')
ap.add_argument('--sys', default='ieee9_gov_simulink_data.mat')
ap.add_argument('--out', default=None, help='default: <ckpt dir>/refined')
ap.add_argument('--epochs', type=int, default=200)
ap.add_argument('--batch_size', type=int, default=512)
ap.add_argument('--instants_per_scenario', type=int, default=2)
ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--seed', type=int, default=0)
# constraint
ap.add_argument('--rescue_floor', type=float, default=None,
                help='Rescue rate to hold. Default: the loaded model measured '
                     'on validation, minus --floor_tol.')
ap.add_argument('--floor_tol', type=float, default=0.005,
                help='Slack below the measured baseline rescue rate.')
ap.add_argument('--lam0', type=float, default=10.0, help='initial multiplier')
ap.add_argument('--lam_eta', type=float, default=5.0, help='dual step size')
ap.add_argument('--lam_min', type=float, default=0.1)
ap.add_argument('--lam_max', type=float, default=1e4)
# stopping
ap.add_argument('--patience', type=int, default=30)
ap.add_argument('--min_delta', type=float, default=1e-4,
                help='||dP|| reduction (pu) that counts as an improvement.')
ap.add_argument('--val_instants', type=int, default=4096)
ap.add_argument('--unfreeze_all', action='store_true')
ap.add_argument('--print_every', type=int, default=1)
ap.add_argument('--no_progress', action='store_true',
                help='Disable the live progress bar (use when piping to a log).')
args = ap.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

ck = torch.load(args.ckpt, map_location=device, weights_only=False)
A = ck['args']
out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(args.ckpt)),
                                   'refined')
os.makedirs(out_dir, exist_ok=True)
print(f"Device: {device}")
print(f"Loaded {args.ckpt}: epoch {ck['epoch']}, window {ck['window_s']*1000:.1f} ms, "
      f"dP head '{A.get('dp_head','linear')}'")

# ── progress bar ─────────────────────────────────────────────────────────────
def _hms(sec):
    sec = int(max(0, sec))
    d, sec = divmod(sec, 86400)
    h, sec = divmod(sec, 3600)
    m, sec = divmod(sec, 60)
    return f"{d}d {h:02d}:{m:02d}:{sec:02d}" if d else f"{h:02d}:{m:02d}:{sec:02d}"


class Progress:
    """Dependency-free \r bar on stderr, so stdout stays a clean log."""

    def __init__(self, n_epochs, n_batches, width=22, enabled=True):
        self.ne, self.nb, self.w = n_epochs, n_batches, width
        self.enabled = enabled and sys.stderr.isatty()
        self.t0 = time.time()
        self.ep_times = []
        self.ep_start = self.t0

    def batch(self, epoch, bi, stats, extra=""):
        if not self.enabled:
            return
        frac = (bi + 1) / self.nb
        fill = int(self.w * frac)
        bar = "#" * fill + "." * (self.w - fill)
        el = time.time() - self.ep_start
        rate = (bi + 1) / max(el, 1e-9)
        eta_ep = (self.nb - bi - 1) / max(rate, 1e-9)
        per_ep = (np.mean(self.ep_times) if self.ep_times
                  else el / max(frac, 1e-9))
        eta_run = eta_ep + per_ep * (self.ne - epoch)
        txt = "  ".join(f"{k}={v:.4f}" for k, v in stats.items())
        if extra:
            txt += "  " + extra
        sys.stderr.write(
            f"\rep {epoch}/{self.ne} [{bar}] {bi+1}/{self.nb}  {txt}  "
            f"{rate:.1f} b/s  ep {_hms(eta_ep)}  run {_hms(eta_run)}   ")
        sys.stderr.flush()

    def epoch_start(self):
        self.ep_start = time.time()

    def epoch_end(self):
        self.ep_times.append(time.time() - self.ep_start)
        if len(self.ep_times) > 20:
            self.ep_times.pop(0)
        if self.enabled:
            sys.stderr.write("\r" + " " * 140 + "\r")
            sys.stderr.flush()

    def note(self, msg):
        """Write a one-off line without the bar overwriting it."""
        if self.enabled:
            sys.stderr.write("\r" + " " * 140 + "\r")
            sys.stderr.flush()
        print(msg)

    def elapsed(self):
        return time.time() - self.t0


# ── data (same construction as train_receding.py) ───────────────────────────
X = np.load(os.path.join(args.data, 'X.npy'), mmap_mode='r')
md = np.load(os.path.join(args.data, 'meta.npz'))
DT = float(md['dt_solver']); t_grid = md['t']; lines = md['lines'].astype(int)
y_type = md['y_type'].ravel().astype(int)
meta_loc = md['meta_loc'].ravel().astype(int)
meta_alpha = md['meta_alpha'].ravel()
t_fault = md['y_t_fault'].ravel(); t_clear = md['meta_t_clear'].ravel()
dp_status = md['dp_status'].ravel().astype(int)
y_dp = md['y_delta_p'] if 'y_delta_p' in md else np.zeros((len(y_type), 3))
OMEGA_LIMIT = float(md['omega_limit'])
idx_tr = np.asarray(ck['split']['train']); idx_va = np.asarray(ck['split']['val'])
idx_te = np.asarray(ck['split']['test'])
print(f"Split from checkpoint: {len(idx_tr)} train / {len(idx_va)} val / {len(idx_te)} test")

D_IDX, W_IDX, PM_IDX = [0, 2, 4], [1, 3, 5], [6, 7, 8]
ID_IDX, IQ_IDX = [9, 11, 13], [10, 12, 14]
VM_IDX = list(range(15, 33, 2)); TH_IDX = list(range(16, 33, 2))
FUT_IDX = [1, 3, 5, 6, 7, 8]
W_STEPS = ck['w_steps']
W_STRIDE = int(round(A['dt_window'] / DT))
ROLL_STRIDE = int(round(A['dt_roll'] / DT))
N_ROLL = int(round(A['horizon'] / A['dt_roll']))
TYPE_NAMES = ['gen', 'load', 'line']
AGE_BINS = [(0.0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1e9)]


def make_batch(ks, rng):
    ks = np.asarray(ks); bs = len(ks)
    lo, hi = A['w_max'], t_grid[-1] - A['horizon']
    tk = rng.uniform(lo, hi, bs)
    if A.get('p_active', 0) > 0:
        pick = rng.random(bs) < A['p_active']
        a = np.clip(t_fault[ks], lo, hi); b = np.clip(t_clear[ks], lo, hi)
        ow = A.get('onset_window', 0.0)
        if ow and ow > 0:
            b = np.minimum(b, a + ow)
        ok = pick & (b > a)
        tk[ok] = rng.uniform(a[ok], b[ok])
    i_end = np.round(tk / DT).astype(int)

    win = np.empty((bs, W_STEPS, 33), np.float32)
    fut = np.empty((bs, N_ROLL + 1, 6), np.float32)
    for j, (k, ie) in enumerate(zip(ks, i_end)):
        i0 = ie - (W_STEPS - 1) * W_STRIDE
        blk = np.asarray(X[k, i0:ie + N_ROLL * ROLL_STRIDE + 1])
        o = ie - i0
        win[j] = blk[0:o + 1:W_STRIDE]
        fut[j] = blk[o::ROLL_STRIDE][:N_ROLL + 1][:, FUT_IDX]

    d = win[..., D_IDX]; w = win[..., W_IDX]; pm = win[..., PM_IDX]
    th = win[..., TH_IDX]
    feat = np.concatenate([d[..., 1:] - d[..., :1], (w - 1.0) / 0.01, pm,
                           win[..., ID_IDX], win[..., IQ_IDX],
                           win[..., VM_IDX], th[..., 1:] - th[..., :1]], -1)
    re = win[:, -1, :]; dd = re[:, D_IDX]
    x_meas = np.concatenate([np.stack([np.zeros(bs), dd[:, 1] - dd[:, 0],
                                       dd[:, 2] - dd[:, 0]], 1),
                             re[:, W_IDX], re[:, PM_IDX]], 1)
    active = (tk >= t_fault[ks]) & (tk < t_clear[ks])
    t_change = np.where(active, t_clear[ks],
                        np.where(tk < t_fault[ks], t_fault[ks], np.inf))
    valid_t = np.clip(t_change - tk, 0.0, A['horizon'])
    step_t = np.arange(N_ROLL + 1) * A['dt_roll']
    vmask = (step_t[None, :] <= valid_t[:, None]).astype(np.float32)
    age = (tk - t_fault[ks]).astype(np.float32)
    return (torch.tensor(feat), torch.tensor(x_meas, dtype=torch.float32),
            torch.tensor(fut), torch.tensor(vmask), torch.tensor(age),
            torch.tensor(active), ks)


def to_dev(b):
    f_, x_, fu, vm, ag, ac, ks = b
    return (f_.to(device), x_.to(device), fu.to(device), vm.to(device),
            ag.to(device), ac.to(device), ks)


# ── model ────────────────────────────────────────────────────────────────────
S = IEEE9(args.base, args.sys, dtype=torch.float32, device=device, lines=lines)
enc = FaultEncoder(ck['n_feat'], w_max_steps=W_STEPS, dt_window=A['dt_window'],
                   lstm_layers=A.get('lstm_layers', 10),
                   init_window=A['init_window'] / A['w_max'],
                   max_window=A['w_max'], beta0=A['beta0'])
model = RolloutController(S, enc, horizon=A['horizon'], dt_roll=A['dt_roll'],
                          omega_limit=OMEGA_LIMIT, margin=A.get('margin', 1e-4),
                          dp_head=A.get('dp_head', 'linear'),
                          dp_hidden=A.get('dp_hidden', 128),
                          dp_fc_layers=A.get('dp_fc_layers', 2),
                          n_refine=A.get('n_refine', 3)).to(device)
model.load_state_dict(ck['model'])
enc.beta.fill_(A['beta_min'])          # keep the trained mask sharpness

# freeze everything but the dP head, so the identification results of the
# loaded checkpoint are preserved exactly
if not args.unfreeze_all:
    for q in model.parameters():
        q.requires_grad_(False)
    if A.get('dp_head', 'linear') == 'cascade':
        train_mods = [model.dp_cell, model.dp_fc]
    else:
        train_mods = [model.enc.head_dP]
    for m_ in train_mods:
        for q in m_.parameters():
            q.requires_grad_(True)
trainable = [q for q in model.parameters() if q.requires_grad]
print(f"Trainable parameters: {sum(q.numel() for q in trainable):,} "
      f"of {sum(q.numel() for q in model.parameters()):,}"
      f"{'' if not args.unfreeze_all else '  (all unfrozen)'}")

opt = torch.optim.Adam(trainable, lr=args.lr)
val_bs = min(args.batch_size, args.val_instants)
n_val_batches = max(1, args.val_instants // val_bs)


@torch.no_grad()
def evaluate(pool):
    model.eval()
    rng_v = np.random.default_rng(12345)
    hit = n_need = 0; norm_s = norm_n = 0.0
    ok_hit = ok_n = 0
    age_hit = [0] * len(AGE_BINS); age_n = [0] * len(AGE_BINS)
    for _ in range(n_val_batches):
        ks_b = rng_v.choice(pool, val_bs, replace=True)
        feat, x_meas, fut, vmask, age, active, ks = to_dev(make_batch(ks_b, rng_v))
        o = model(feat, x_meas, hard=True, ctrl=True, vmask=vmask)
        pk_t, has = model.judged(fut[:, :, 0:3], vmask)
        pk_c, has_c = model.judged(o['w_ctrl'], vmask)
        dev_now = (x_meas[:, 3:6] - 1.0).abs().amax(dim=1)
        # actionable = fault active now AND omega still inside the envelope.
        # MUST match train_receding.py / verify_true.py, or the floor is set on
        # a different metric than the one the model was trained and reported on.
        reach = has_c & (dev_now <= model.limit) & active
        need = (pk_t > model.limit) & reach
        won = (pk_c <= model.limit)
        hit += int(won[need].sum()); n_need += int(need.sum())
        ok_hit += int(won[reach].sum()); ok_n += int(reach.sum())
        nrm = o['dP'].pow(2).sum(1).sqrt()
        norm_s += float(nrm.sum()); norm_n += nrm.numel()
        ag = age.cpu().numpy(); nd = need.cpu().numpy(); wn = won.cpu().numpy()
        for i, (b0, b1) in enumerate(AGE_BINS):
            m = nd & (ag >= b0) & (ag < b1)
            if m.sum():
                age_hit[i] += int(wn[m].sum()); age_n[i] += int(m.sum())
    model.train()
    return {'rescue': hit / max(n_need, 1), 'n_need': n_need,
            'ok': ok_hit / max(ok_n, 1), 'norm': norm_s / max(norm_n, 1),
            'by_age': [(n, h / n if n else float('nan'))
                       for h, n in zip(age_hit, age_n)]}


base = evaluate(idx_va)
floor = args.rescue_floor if args.rescue_floor is not None \
    else max(0.0, base['rescue'] - args.floor_tol)
print(f"\nBaseline (validation): rescue {base['rescue']:.4f} on {base['n_need']} "
      f"cases, ok {base['ok']:.4f}, ||dP|| {base['norm']:.4f} pu")
print(f"Constraint: hold rescue >= {floor:.4f}; minimise ||dP|| under it.\n")

lam = args.lam0
best_norm = float('inf'); best_epoch = 0; since = 0
epoch_units = np.repeat(idx_tr, args.instants_per_scenario)
n_batches = max(1, int(np.ceil(len(epoch_units) / args.batch_size)))
rng_b = np.random.default_rng(args.seed + 1)
trace = []
t0 = time.time()
prog = Progress(args.epochs, n_batches, enabled=not args.no_progress)
print(f"Epoch = {len(epoch_units)} decision instants "
      f"({len(idx_tr)} scenarios x {args.instants_per_scenario}) "
      f"= {n_batches} batches   |   validation: {n_val_batches * val_bs} "
      f"fixed instants each epoch")

for epoch in range(1, args.epochs + 1):
    prog.epoch_start()
    order = rng_b.permutation(epoch_units)
    agg = dict(norm=0., fail=0., n=0)
    for bi in range(n_batches):
        ks_b = order[bi * args.batch_size:(bi + 1) * args.batch_size]
        if not len(ks_b):
            continue
        feat, x_meas, fut, vmask, age, active, ks = to_dev(make_batch(ks_b, rng_b))
        opt.zero_grad(set_to_none=True)
        o = model(feat, x_meas, ctrl=True, vmask=vmask)
        pk_c, has_c = model.judged(o['w_ctrl'], vmask)
        viol = model.violation(o['w_ctrl'], vmask)
        dev_now = (x_meas[:, 3:6] - 1.0).abs().amax(dim=1)
        # same gate in the loss: post-clearing / pre-fault instants have nothing
        # to correct, and leaving them in pushes dP UP -- directly against the
        # norm reduction this script exists to achieve
        reach = has_c & (dev_now <= model.limit) & active
        rm = reach.float()
        tgt = model.limit - model.margin
        p_fail = torch.sigmoid((pk_c - tgt) / A['dp_tau'])
        envelope = (((p_fail + A['w_viol'] * viol / model.limit) * rm).sum()
                    / rm.sum().clamp(min=1))
        norm = o['dP'].pow(2).sum(1).mean()
        (norm + lam * envelope).backward()
        nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        agg['norm'] += float(norm.detach()); agg['fail'] += float(envelope.detach())
        agg['n'] += 1
        nd = agg['n']
        prog.batch(epoch, bi,
                   {'|dP|^2': agg['norm'] / nd, 'env': agg['fail'] / nd},
                   extra=f"lam={lam:8.2f}  best|dP|="
                         f"{(best_norm if best_norm < 1e9 else float('nan')):.4f}pu")

    prog.epoch_end()
    v = evaluate(idx_va)
    trace.append({'epoch': epoch, 'rescue': v['rescue'], 'norm': v['norm'],
                  'ok': v['ok'], 'lam': lam})

    # dual ascent: lambda rises when the constraint is violated, decays when slack
    lam = float(np.clip(lam * np.exp(args.lam_eta * (floor - v['rescue'])),
                        args.lam_min, args.lam_max))

    feasible = v['rescue'] >= floor
    improved = feasible and v['norm'] < best_norm - args.min_delta
    if improved:
        best_norm, best_epoch, since = v['norm'], epoch, 0
        torch.save({**ck, 'model': model.state_dict(), 'epoch': epoch,
                    'refined_from': args.ckpt, 'refine_args': vars(args),
                    'val': v, 'rescue_floor': floor, 'lam': lam},
                   os.path.join(out_dir, 'best.pt'))
    else:
        since += 1

    if epoch % args.print_every == 0:
        flag = '  <-- best' if improved else (
            '  INFEASIBLE' if not feasible else f'  ({since}/{args.patience})')
        prog.note(f"  ep {epoch:4d}  rescue={v['rescue']:.4f}  "
                  f"||dP||={v['norm']:.4f}pu "
                  f"({100*v['norm']/base['norm']:.1f}% of baseline)  "
                  f"ok={v['ok']:.4f}  lam={lam:8.2f}  "
                  f"[{_hms(prog.elapsed())}]{flag}")
    if args.patience and since >= args.patience:
        print(f"\nStopped: no feasible ||dP|| reduction for {args.patience} epochs.")
        break

json.dump(trace, open(os.path.join(out_dir, 'refine_pareto.json'), 'w'), indent=1)

print(f"\n{'='*68}")
print(f"REFINEMENT  ({time.time()-t0:.0f}s, best epoch {best_epoch})")
print(f"{'='*68}")
if best_epoch:
    bk = torch.load(os.path.join(out_dir, 'best.pt'), map_location=device,
                    weights_only=False)
    model.load_state_dict(bk['model'])
    te = evaluate(idx_te)
    red = 100 * (1 - te['norm'] / base['norm'])
    print(f"  ||dP||       {base['norm']:.4f} -> {te['norm']:.4f} pu   "
          f"({red:.1f}% {'reduction' if red >= 0 else 'INCREASE'})")
    print(f"  rescue       {base['rescue']:.4f} -> {te['rescue']:.4f}   "
          f"(floor {floor:.4f}, test split)")
    print(f"  overall ok   {base['ok']:.4f} -> {te['ok']:.4f}")
    print(f"  by fault age " + "  ".join(
        f"{lo*1000:.0f}-{(hi*1000 if hi < 1e8 else 9999):.0f}ms:{r:.2f}(n={n})"
        for (lo, hi), (n, r) in zip(AGE_BINS, te['by_age'])))
    json.dump({'baseline_val': base, 'test': te, 'floor': floor,
               'best_epoch': best_epoch},
              open(os.path.join(out_dir, 'refine_results.json'), 'w'), indent=1)
    print(f"\n  -> {out_dir}/best.pt, refine_results.json, refine_pareto.json")
else:
    print("  No feasible improvement found. The starting point is already at "
          "minimum effort for this rescue floor, or --lam0 is too low to hold "
          "the constraint (rescue stayed below the floor throughout).")
