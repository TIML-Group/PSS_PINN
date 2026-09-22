"""
train_receding.py
=================
Train the receding-horizon identifier/controller.

    python train_receding.py --data pinn_dataset_small.mat --epochs 200

Losses
------
identification (trains the encoder):
    CE on the 13-way hypothesis  +  alpha regression (masked)
    + trajectory match: rolled-out omega/Pm vs the MEASURED future.
    X is the uncorrected simulation, so the future IS the free rollout's
    target. This is dense per-timestep supervision, not one label.

control (trains the dP head only -- physics parameters are detached):
    ||dP||^2  +  hinge(max|omega-1| over the judged window - (limit-margin))
    evaluated on the rollout WITH dP applied. This is the definition of
    y_delta_p, so the labels are kept as held-out validation rather than a
    training target.

window penalty:
    lambda * L / L_max, so the learnable window finds the SHORTEST length that
    still identifies the disturbance. Without it L just grows to the cap.

Decision instants are sampled uniformly over the run (receding horizon), so the
model sees pre-fault, during-fault and post-clearing windows. The hypothesis
target is the active fault at the window end, or class 0 if none is active.

An epoch is one shuffled pass over every training scenario -- each is visited
exactly `--instants_per_scenario` times, at freshly drawn decision instants.
So 500 epochs means each training scenario is seen 500 times (at 500 different
instants), not an arbitrary number of random draws.
"""

import argparse, json, os, queue, sys, threading, time

import h5py
import numpy as np
import torch
import torch.nn as nn

from ieee9_physics import IEEE9, HYPOTHESES
from pinn_model import FaultEncoder, RolloutController, N_CLASS

p = argparse.ArgumentParser()
p.add_argument('--data', type=str, required=True)
p.add_argument('--base', type=str, default='ieee9_base.mat')
p.add_argument('--sys',  type=str, default='ieee9_gov_simulink_data.mat')
p.add_argument('--out',  type=str, default='receding_output')
p.add_argument('--epochs', type=int, default=200)
p.add_argument('--batch_size', type=int, default=64)
p.add_argument('--instants_per_scenario', type=int, default=1,
               help='Decision instants drawn per scenario per epoch. An epoch '
                    'is one shuffled pass over all training scenarios.')
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--lr_min', type=float, default=1e-5,
               help='Cosine-decayed floor. A flat 1e-3 makes the loss spike '
                    'once it gets small.')
p.add_argument('--lstm_layers', type=int, default=10,
               help='Depth of the recurrent stack. Worth ablating: 3 and 5 are '
                    'plausible for identification over ~60 steps.')
p.add_argument('--print_every', type=int, default=1)
p.add_argument('--val_every', type=int, default=1,
               help='Run validation every N epochs (default: every epoch). '
                    'best.pt is written whenever the validation score improves.')
p.add_argument('--val_instants', type=int, default=4096,
               help='Decision instants scored per validation pass. Sized in '
                    'instants rather than batches so the validation cost does '
                    'not change when --batch_size does. Drawn from a FIXED '
                    'seed, so every pass scores exactly the same points and '
                    'the curve reflects the model, not sampling noise.')
p.add_argument('--patience', type=int, default=40,
               help='Stop after this many validations with no improvement '
                    '(= epochs, at the default --val_every 1). 0 disables '
                    'early stopping.')
p.add_argument('--no_progress', action='store_true',
               help='Disable the live progress bar (use when piping to a log).')
p.add_argument('--seed', type=int, default=0)
# windows / rollout
p.add_argument('--w_max', type=float, default=0.30, help='max window [s]')
p.add_argument('--init_window', type=float, default=0.15)
p.add_argument('--dt_window', type=float, default=0.005, help='window stride [s]')
p.add_argument('--horizon', type=float, default=1.5)
p.add_argument('--dt_roll', type=float, default=0.01,
               help='Rollout RK4 step. 10 ms matches the 1 ms reference to the '
                    'measurement noise floor (5.6e-5 vs 5.0e-5) with identical '
                    'OOL agreement, at half the kernel launches of 5 ms.')
p.add_argument('--prefetch', type=int, default=3,
               help='Batches prepared ahead on a worker thread. Batch building '
                    'is numpy on the main thread otherwise, which stalls the '
                    'GPU between steps.')
# loss weights
p.add_argument('--w_traj',  type=float, default=1.0)
p.add_argument('--w_cls',   type=float, default=1.0)
p.add_argument('--w_alpha', type=float, default=1.0)
p.add_argument('--w_ctrl',  type=float, default=0.0,
               help='control-branch weight. 0 until identification plateaus; '
                    '--ctrl_start turns it on automatically.')
p.add_argument('--ctrl_start', type=int, default=50)
p.add_argument('--p_active', type=float, default=0.5,
               help='Fraction of decision instants drawn from inside a fault '
                    'window. Uniform sampling puts only ~25%% there, and '
                    'pre-fault instants carry almost no trajectory signal (the '
                    'system is at equilibrium and the valid horizon ends at '
                    'fault onset). 0 restores uniform sampling.')
p.add_argument('--dp_head', type=str, default='linear',
               choices=['linear', 'cascade'],
               help="'linear': one Linear(128,3) on the shared trunk, emitted "
                    "beside the identification heads. 'cascade': identification "
                    "-> free rollout -> LSTM+FC head conditioned on the "
                    "predicted excursion, refined --n_refine times with the "
                    "simulator in the loop. Context is detached, so only the "
                    "head trains from the control objective.")
p.add_argument('--dp_hidden', type=int, default=128)
p.add_argument('--dp_fc_layers', type=int, default=2)
p.add_argument('--n_refine', type=int, default=3,
               help='Refinement steps in the cascade head. Each costs one '
                    'extra rollout (forward and backward).')
p.add_argument('--onset_window', type=float, default=0.2,
               help='Active-fault instants are drawn from t_fault + U(0, this). '
                    'A receding-horizon controller watches continuously and '
                    'acts within ~50 ms of onset (the labels use exactly 50 ms); '
                    'it never meets a fault that has run unattended for a '
                    'second. Uniform sampling over the whole fault window put '
                    'half the control gradient on cases already outside the '
                    'envelope, where no dP can help. 0 restores uniform.')
p.add_argument('--w_dp_norm', type=float, default=1.0,
               help='Weight on ||dP||^2. The old hardcoded 0.05 was 30-700x '
                    'weaker than the envelope term, so the optimiser bought '
                    'margin with arbitrarily large dP (155%% of ||y_delta_p||, '
                    'worse than emitting zero).')
p.add_argument('--w_viol', type=float, default=10.0,
               help='Weight on the dense envelope-violation term.')
p.add_argument('--margin', type=float, default=1e-4,
               help='Envelope safety margin in pu. Worth raising toward the '
                    'trajectory RMSE: a dP that clears the SIMULATED envelope '
                    'by less than the simulation error can still violate the '
                    'real one.')
p.add_argument('--dp_tau', type=float, default=2e-4,
               help='Softness of the failure-probability surrogate, in pu of '
                    'omega. Smaller = closer to a hard success count.')
p.add_argument('--select', type=str, default='auto',
               choices=['auto', 'traj', 'dp'],
               help="Early-stopping metric. 'auto' selects on identification "
                    "(traj+cls) until the control head switches on, then on dP "
                    "performance -- dP is the deliverable, and traj plateaus "
                    "long before dP is done. 'traj'/'dp' force one throughout.")
p.add_argument('--lam_window', type=float, default=0.05,
               help='penalty on window length. Sweep this for the Pareto curve.')
p.add_argument('--beta0', type=float, default=0.04)
p.add_argument('--beta_min', type=float, default=0.005)
args = p.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
print(f"Device: {device}")
os.makedirs(args.out, exist_ok=True)

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

    def batch(self, epoch, bi, stats, extra=""):
        if not self.enabled:
            return
        frac = (bi + 1) / self.nb
        fill = int(self.w * frac)
        bar = "#" * fill + "." * (self.w - fill)
        el = time.time() - self.ep_start
        rate = (bi + 1) / max(el, 1e-9)
        eta_ep = (self.nb - bi - 1) / max(rate, 1e-9)
        # whole-run estimate from completed epochs, falling back to this one
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

    def elapsed(self):
        return time.time() - self.t0


# ── data ─────────────────────────────────────────────────────────────────────
# Expects the memmap produced by convert_dataset.py: X is (N, T, 33) in C order
# so one scenario's time-block is contiguous. Reading windows straight out of
# the .mat is not viable -- it stores (33, T, N) with the scenario index on the
# fastest axis, so every window read strides across the whole file.
if os.path.isdir(args.data):
    X = np.load(os.path.join(args.data, 'X.npy'), mmap_mode='r')   # (N,T,33)
    md = np.load(os.path.join(args.data, 'meta.npz'))
    N, T = X.shape[0], X.shape[1]
    t_grid = md['t']; DT = float(md['dt_solver'])
    lines = md['lines'].astype(int)
    OMEGA_LIMIT = float(md['omega_limit'])
    g = lambda k: md[k].ravel()
else:
    raise SystemExit(
        f"--data must be the directory made by convert_dataset.py.\n"
        f"Run:  python convert_dataset.py --mat {args.data} --out data_mm\n"
        f"then: python train_receding.py --data data_mm ...")

y_type = g('y_type').astype(int)
meta_loc = g('meta_loc').astype(int)
meta_alpha = g('meta_alpha')
t_fault = g('y_t_fault')
t_clear = g('meta_t_clear')
y_ool = g('y_ool').astype(bool)
dp_status = g('dp_status').astype(int)
failed = g('failed').astype(bool)
y_dp = md['y_delta_p'] if 'y_delta_p' in md else np.zeros((N, 3))
print(f"Dataset: {N} scenarios x {T} steps @ {DT*1000:.0f} ms  (memmap)")

# class index of each scenario's fault (1..12), used when a fault is active
TYPE_NAMES = ['gen', 'load', 'line']
sc_class = np.array([HYPOTHESES.index((TYPE_NAMES[y_type[k]-1], meta_loc[k])) + 1
                     for k in range(N)])

# class -> "does alpha mean anything here" lookup. Built once; doing this per
# batch as a Python comprehension over cls.cpu() forced a GPU sync every step.
ALPHA_CLASS = np.array([False] + [HYPOTHESES[c][0] in ('gen', 'load')
                                  for c in range(len(HYPOTHESES))])
# class -> fault-type id (0 none, 1 gen, 2 load, 3 line), for type-only accuracy
TYPE_OF = np.array([0] + [TYPE_NAMES.index(HYPOTHESES[c][0]) + 1
                          for c in range(len(HYPOTHESES))])

# ── 90/5/5 split (the stored `split` is 80/10/10; regenerating as requested) ──
rng = np.random.default_rng(args.seed)
ok = np.where(~failed)[0]
perm = rng.permutation(ok)
n_tr = int(0.90 * len(perm)); n_va = int(0.05 * len(perm))
idx_tr, idx_va, idx_te = perm[:n_tr], perm[n_tr:n_tr+n_va], perm[n_tr+n_va:]
np.savez(os.path.join(args.out, 'split.npz'),
         train=idx_tr, val=idx_va, test=idx_te, seed=args.seed)
print(f"Split: {len(idx_tr)} train / {len(idx_va)} val / {len(idx_te)} test")

# ── features ─────────────────────────────────────────────────────────────────
# 33 channels -> 31 features. delta is unwrapped and drifts with system
# frequency, so only angle DIFFERENCES are used (Pe depends on differences
# alone, so the dynamics are invariant to a common shift anyway).
D_IDX, W_IDX, PM_IDX = [0, 2, 4], [1, 3, 5], [6, 7, 8]
FUT_IDX = [1, 3, 5, 6, 7, 8]          # omega(3), Pm(3) -- rollout targets
ID_IDX, IQ_IDX = [9, 11, 13], [10, 12, 14]
VM_IDX = list(range(15, 33, 2))
TH_IDX = list(range(16, 33, 2))

def featurize(raw):
    """raw: (B, W, 33) -> (B, W, 31) and the measured state (B, 9)."""
    d = raw[..., D_IDX]; w = raw[..., W_IDX]; pm = raw[..., PM_IDX]
    th = raw[..., TH_IDX]
    d_rel = d[..., 1:] - d[..., :1]
    th_rel = th[..., 1:] - th[..., :1]
    feat = np.concatenate([d_rel, (w - 1.0) / 0.01, pm,
                           raw[..., ID_IDX], raw[..., IQ_IDX],
                           raw[..., VM_IDX], th_rel], axis=-1)
    return feat

N_FEAT = 2 + 3 + 3 + 3 + 3 + 9 + 8

W_STEPS = int(round(args.w_max / args.dt_window))
W_STRIDE = int(round(args.dt_window / DT))
ROLL_STRIDE = int(round(args.dt_roll / DT))
N_ROLL = int(round(args.horizon / args.dt_roll))

def make_batch(ks, rng):
    """Build a batch for the given scenario ids at fresh decision instants."""
    ks = np.asarray(ks); bs = len(ks)
    # decision instant must leave room for the window behind and horizon ahead
    lo = args.w_max
    hi = t_grid[-1] - args.horizon
    tk = rng.uniform(lo, hi, bs)
    if args.p_active > 0:
        # draw a share of instants from inside the fault window, concentrated
        # near onset where a correction can still act in time
        pick = rng.random(bs) < args.p_active
        a = np.clip(t_fault[ks], lo, hi)
        b = np.clip(t_clear[ks], lo, hi)
        if args.onset_window > 0:
            b = np.minimum(b, a + args.onset_window)
        ok = pick & (b > a)
        tk[ok] = rng.uniform(a[ok], b[ok])
    i_end = np.round(tk / DT).astype(int)

    win = np.empty((bs, W_STEPS, 33), dtype=np.float32)
    fut = np.empty((bs, N_ROLL + 1, 6), dtype=np.float32)   # omega(3), Pm(3)
    for j, (k, ie) in enumerate(zip(ks, i_end)):
        i0 = ie - (W_STEPS - 1) * W_STRIDE
        i1 = ie + N_ROLL * ROLL_STRIDE + 1
        blk = np.asarray(X[k, i0:i1])                      # (span, 33) one read
        o = ie - i0
        win[j] = blk[0:o + 1:W_STRIDE]
        fut[j] = blk[o::ROLL_STRIDE][:N_ROLL + 1][:, FUT_IDX]

    feat = featurize(win)
    raw_end = win[:, -1, :]
    d = raw_end[:, D_IDX]
    x_meas = np.concatenate([np.stack([np.zeros(bs), d[:, 1]-d[:, 0],
                                       d[:, 2]-d[:, 0]], 1),
                             raw_end[:, W_IDX], raw_end[:, PM_IDX]], 1)

    active = (tk >= t_fault[ks]) & (tk < t_clear[ks])
    cls = np.where(active, sc_class[ks], 0)
    alpha_t = meta_alpha[ks]

    # ── validity horizon ────────────────────────────────────────────────────
    # The free rollout holds the identified network for the whole horizon,
    # which is right for a receding-horizon controller but WRONG as a target
    # once the real network changes: the fault clears, or a new one starts.
    # Supervising past that point penalises the model for failing to predict
    # an unknowable event and puts a floor under l_traj well above the noise
    # floor. So the trajectory loss is masked to the interval where the
    # identified network actually holds.
    #   fault active -> valid until t_clear
    #   pre-fault    -> valid until t_fault
    #   post-clear   -> valid for the whole horizon (network is pre-fault again)
    t_change = np.where(active, t_clear[ks],
                        np.where(tk < t_fault[ks], t_fault[ks], np.inf))
    valid_t = np.clip(t_change - tk, 0.0, args.horizon)
    step_t = np.arange(N_ROLL + 1) * args.dt_roll
    vmask = (step_t[None, :] <= valid_t[:, None]).astype(np.float32)

    age = (tk - t_fault[ks]).astype(np.float32)   # <0 = pre-fault
    return (torch.tensor(feat), torch.tensor(x_meas, dtype=torch.float32),
            torch.tensor(fut), torch.tensor(cls),
            torch.tensor(alpha_t, dtype=torch.float32),
            torch.tensor(active), torch.tensor(vmask),
            torch.tensor(age), ks, tk)

# ── model ────────────────────────────────────────────────────────────────────
S = IEEE9(args.base, args.sys, dtype=torch.float32, device=device, lines=lines)
enc = FaultEncoder(N_FEAT, w_max_steps=W_STEPS, dt_window=args.dt_window,
                   lstm_layers=args.lstm_layers,
                   init_window=args.init_window, max_window=args.w_max,
                   beta0=args.beta0)
model = RolloutController(S, enc, horizon=args.horizon, dt_roll=args.dt_roll,
                          omega_limit=OMEGA_LIMIT, margin=args.margin,
                          dp_head=args.dp_head, dp_hidden=args.dp_hidden,
                          dp_fc_layers=args.dp_fc_layers,
                          n_refine=args.n_refine).to(device)
print(f"Parameters: {sum(q.numel() for q in model.parameters()):,}")
print(f"Window: {W_STEPS} steps @ {args.dt_window*1000:.0f} ms "
      f"(max {args.w_max}s, init {args.init_window}s)")
print(f"Rollout: {N_ROLL} steps @ {args.dt_roll*1000:.0f} ms = {args.horizon}s")
print(f"dP head: {args.dp_head}" + (f" (LSTM{args.dp_hidden} + {args.dp_fc_layers} FC, "
      f"{args.n_refine} refinement steps, +{args.n_refine-1} rollouts/step)"
      if args.dp_head == 'cascade' else " (Linear on shared trunk)"))

ALPHA_CLASS_T = torch.tensor(ALPHA_CLASS, device=device)
TYPE_OF_T = torch.tensor(TYPE_OF, device=device)
AGE_BINS = [(0.0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1e9)]

opt = torch.optim.Adam(model.parameters(), lr=args.lr)
ce = nn.CrossEntropyLoss()
sched = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt, T_max=args.epochs, eta_min=args.lr_min)
rng_b = np.random.default_rng(args.seed + 1)
epoch_units = np.repeat(idx_tr, args.instants_per_scenario)
n_batches = max(1, int(np.ceil(len(epoch_units) / args.batch_size)))
print(f"Epoch = {len(epoch_units)} decision instants "
      f"({len(idx_tr)} scenarios x {args.instants_per_scenario}) "
      f"= {n_batches} batches")
val_bs = min(args.batch_size, args.val_instants)
n_val_batches = max(1, args.val_instants // val_bs)
print(f"Validation: every {args.val_every} epoch(s), "
      f"{n_val_batches * val_bs} fixed instants over {len(idx_va)} scenarios "
      f"(~{100 * n_val_batches * val_bs / (3 * len(epoch_units)):.0f}% "
      f"of an epoch's cost; forward only)")
def to_device(b):
    feat, x_meas, fut, cls, alpha_t, active, vmask, age, ks, tk = b
    nb = args.prefetch > 0 and device.type == 'cuda'
    if nb:
        feat, x_meas, fut = feat.pin_memory(), x_meas.pin_memory(), fut.pin_memory()
    return (feat.to(device, non_blocking=nb), x_meas.to(device, non_blocking=nb),
            fut.to(device, non_blocking=nb), cls.to(device, non_blocking=nb),
            alpha_t.to(device, non_blocking=nb), active.to(device, non_blocking=nb),
            vmask.to(device, non_blocking=nb), age.to(device, non_blocking=nb))


def epoch_batches(order, rng):
    """Yield ready-to-use batches, built on a worker thread so the numpy work
    overlaps the GPU step instead of stalling it."""
    if args.prefetch <= 0:
        for bi in range(n_batches):
            ks_b = order[bi * args.batch_size:(bi + 1) * args.batch_size]
            if len(ks_b):
                yield bi, to_device(make_batch(ks_b, rng))
        return

    q = queue.Queue(maxsize=args.prefetch)

    def produce():
        for bi in range(n_batches):
            ks_b = order[bi * args.batch_size:(bi + 1) * args.batch_size]
            if len(ks_b):
                q.put((bi, make_batch(ks_b, rng)))
        q.put(None)

    th = threading.Thread(target=produce, daemon=True)
    th.start()
    while True:
        item = q.get()
        if item is None:
            break
        bi, b = item
        yield bi, to_device(b)


@torch.no_grad()
def evaluate(pool, ctrl_on):
    """Validation pass. The instants come from a FIXED rng, so successive
    evaluations score the same decision points and the curve reflects the
    model rather than sampling noise."""
    model.eval()
    rng_v = np.random.default_rng(12345)
    keys = ['traj', 'cls', 'acc', 'acc_active', 'acc_type', 'alpha',
            'ool_acc', 'ool_tpr', 'ool_tnr', 'dp_err', 'dp_rel',
            'dp_success', 'dp_rescue', 'dp_rescue_all', 'dp_reachable',
            'dp_norm']
    age_hit = [0] * len(AGE_BINS); age_n = [0] * len(AGE_BINS)
    tot = {k: 0. for k in keys}
    cnt = {k: 0 for k in keys}

    def add(k, v, n=1):
        if n:
            tot[k] += float(v) * n; cnt[k] += n

    for _ in range(n_val_batches):
        ks_b = rng_v.choice(pool, val_bs, replace=True)
        feat, x_meas, fut, cls, alpha_t, active, vmask, age, ks, tk = make_batch(ks_b, rng_v)
        feat, x_meas, fut = feat.to(device), x_meas.to(device), fut.to(device)
        cls, alpha_t = cls.to(device), alpha_t.to(device)
        active, vmask = active.to(device), vmask.to(device)
        o = model(feat, x_meas, hard=True, ctrl=ctrl_on, vmask=vmask)

        se = ((o['w_free'] - fut[:, :, 0:3]) / 0.01) ** 2
        m3 = vmask.unsqueeze(-1)
        add('traj', (se * m3).sum() / m3.expand_as(se).sum().clamp(min=1))
        add('cls', ce(o['hyp_logits'], cls))

        pred = o['hyp_logits'].argmax(1)
        add('acc', (pred == cls).float().mean())
        na = int(active.sum())
        if na:
            add('acc_active', (pred[active] == cls[active]).float().mean(), na)
            # type-only accuracy: gen/load/line, ignoring which unit or line
            add('acc_type', (TYPE_OF_T[pred[active]] == TYPE_OF_T[cls[active]])
                .float().mean(), na)
        m = active & ALPHA_CLASS_T[cls]
        if int(m.sum()):
            add('alpha', (((o['alpha'] - alpha_t) ** 2) * m).sum() / m.sum(),
                int(m.sum()))

        # ── OOL, scored against the MEASURED future over the SAME judged
        # window. y_ool is a whole-run label; the rollout only sees `horizon`
        # seconds from the decision instant, so comparing to it measures a
        # definitional mismatch rather than the model.
        pk_f, has_f = model.judged(o['w_free'], vmask)
        pk_t, _ = model.judged(fut[:, :, 0:3], vmask)
        pred_ool = (pk_f > model.limit) & has_f
        true_ool = (pk_t > model.limit) & has_f
        nv = int(has_f.sum())
        if nv:
            add('ool_acc', ((pred_ool == true_ool) & has_f).float().sum() / nv, nv)
        if int(true_ool.sum()):
            add('ool_tpr', pred_ool[true_ool].float().mean(), int(true_ool.sum()))
        neg = (~true_ool) & has_f
        if int(neg.sum()):
            add('ool_tnr', (~pred_ool[neg]).float().mean(), int(neg.sum()))

        if ctrl_on:
            # Primary dP metric: does the controlled rollout stay inside the
            # envelope? y_delta_p answers a DIFFERENT question (minimum norm,
            # withdrawn at clearing), so distance to it is a reference scale,
            # not a target.
            pk_c, has_c = model.judged(o['w_ctrl'], vmask)
            # only count cases where a correction could still act: omega still
            # inside the envelope at the decision instant
            dev_now = (x_meas[:, 3:6] - 1.0).abs().amax(dim=1)
            # actionable: fault active now AND omega still inside the envelope
            reach = has_c & (dev_now <= model.limit) & active
            need_all = true_ool & has_c
            need = true_ool & reach
            if int(reach.sum()):
                add('dp_success', (pk_c[reach] <= model.limit).float().mean(),
                    int(reach.sum()))
            if int(need.sum()):
                add('dp_rescue', (pk_c[need] <= model.limit).float().mean(),
                    int(need.sum()))
            if int(need_all.sum()):
                add('dp_rescue_all', (pk_c[need_all] <= model.limit).float().mean(),
                    int(need_all.sum()))
                add('dp_reachable', (need.sum() / need_all.sum().clamp(min=1)),
                    int(need_all.sum()))
            add('dp_norm', float(o['dP'].pow(2).sum(1).sqrt().mean()))
            # rescue rate stratified by how old the fault is at decision time
            ag = age.cpu().numpy(); okn = need.cpu().numpy()
            won = (pk_c <= model.limit).cpu().numpy()
            for bi_, (b0, b1) in enumerate(AGE_BINS):
                m = okn & (ag >= b0) & (ag < b1)
                if m.sum():
                    age_hit[bi_] += int(won[m].sum()); age_n[bi_] += int(m.sum())
            sel = (dp_status[ks] == 1)
            if sel.any():
                d = o['dP'].cpu().numpy()[sel] - y_dp[ks][sel]
                e = np.sqrt((d ** 2).sum(1))
                ref = np.linalg.norm(y_dp[ks][sel], axis=1)
                add('dp_err', e.mean(), int(sel.sum()))
                add('dp_rel', (e / np.maximum(ref, 1e-6)).mean(), int(sel.sum()))
    model.train()
    out = {k: (tot[k] / cnt[k] if cnt[k] else float('nan')) for k in keys}
    out['rescue_by_age'] = [(n, (h / n if n else float('nan')))
                            for h, n in zip(age_hit, age_n)]
    return out


hist = []
best_score = float('inf'); best_epoch = 0; since_improve = 0
selecting_on_dp = False
prog = Progress(args.epochs, n_batches, enabled=not args.no_progress)
print(f"Training {args.epochs} epochs ... (Ctrl-C to stop; "
      f"best checkpoint is written every --print_every epochs)")

for epoch in range(1, args.epochs + 1):
    model.train()
    prog.epoch_start()
    # anneal the mask edge so the window boundary sharpens over training
    frac = min(1.0, epoch / max(1, args.epochs * 0.6))
    enc.beta.fill_(args.beta0 + (args.beta_min - args.beta0) * frac)
    use_ctrl = (epoch >= args.ctrl_start)
    w_ctrl = args.w_ctrl if args.w_ctrl > 0 else (1.0 if use_ctrl else 0.0)

    agg = dict(cls=0., alpha=0., traj=0., ctrl=0., acc=0.)
    order = rng_b.permutation(epoch_units)
    for bi, (feat, x_meas, fut, cls, alpha_t, active, vmask, age) in epoch_batches(order, rng_b):
        opt.zero_grad(set_to_none=True)
        o = model(feat, x_meas, ctrl=(w_ctrl > 0), vmask=vmask)

        # 1. hypothesis
        l_cls = ce(o['hyp_logits'], cls)
        # 2. alpha, only where it is meaningful (gen/load faults, fault active)
        is_la = active & ALPHA_CLASS_T[cls]
        l_alpha = (((o['alpha'] - alpha_t) ** 2) * is_la).sum() / is_la.sum().clamp(min=1)
        # 3. trajectory match against the measured (uncorrected) future,
        #    masked to the interval where the identified network still holds
        w_meas = fut[:, :, 0:3]
        se = ((o['w_free'] - w_meas) / 0.01) ** 2          # (B, T, 3)
        m3 = vmask.unsqueeze(-1)
        l_traj = (se * m3).sum() / m3.expand_as(se).sum().clamp(min=1)
        # 4. control branch (detached physics).
        #    The objective is the SUCCESS RATE of keeping omega inside the
        #    envelope. A hard success count has no gradient, so:
        #      p_fail  - a bounded sigmoid surrogate for "this case fails",
        #                whose mean IS the expected failure rate. Bounded in
        #                [0,1], so it cannot be traded against ||dP|| without
        #                limit the way the old unbounded squared hinge could.
        #      viol    - mean excess over the envelope across every valid
        #                timestep, giving dense gradient far from the boundary
        #                where the sigmoid has saturated.
        if w_ctrl > 0:
            tgt = model.limit - model.margin
            peak_c, has_c = model.judged(o['w_ctrl'], vmask)
            viol = model.violation(o['w_ctrl'], vmask)
            p_fail = torch.sigmoid((peak_c - tgt) / args.dp_tau)
            # REACHABILITY: if omega is already outside the envelope at the
            # decision instant, the excursion has happened and no set-point
            # change can undo it -- the governor needs Tg=0.2 s and the peak is
            # already in the judged window. Training on those cases only
            # teaches the head to emit large useless corrections.
            dev_now = (x_meas[:, 3:6] - 1.0).abs().amax(dim=1)
            # Gate on an ACTIVE fault too: post-clearing and pre-fault instants
            # have nothing for a set-point shift to correct, and including them
            # pushes the head to emit large useless corrections.
            reach = has_c & (dev_now <= model.limit) & active
            hm = reach.float()
            l_ctrl = ((p_fail * hm).sum() / hm.sum().clamp(min=1)
                      + args.w_viol * (viol * hm).sum() / hm.sum().clamp(min=1)
                        / model.limit
                      + args.w_dp_norm * (o['dP'] ** 2).sum(1).mean())
        else:
            l_ctrl = torch.zeros((), device=device)

        loss = (args.w_cls * l_cls + args.w_alpha * l_alpha
                + args.w_traj * l_traj + w_ctrl * l_ctrl
                + args.lam_window * enc.window_len() / args.w_max)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        agg['cls'] += l_cls.item(); agg['alpha'] += l_alpha.item()
        agg['traj'] += l_traj.item(); agg['ctrl'] += float(l_ctrl.detach())
        agg['acc'] += (o['hyp_logits'].argmax(1) == cls).float().mean().item()
        n_done = bi + 1
        prog.batch(epoch, bi, {'traj': agg['traj'] / n_done,
                               'cls': agg['cls'] / n_done,
                               'acc': agg['acc'] / n_done,
                               'ctrl': agg['ctrl'] / n_done},
                   extra=f"L={float(enc.window_len().detach())*1000:5.1f}ms")

    sched.step()
    prog.epoch_end()
    for k in agg: agg[k] /= n_batches
    L = float(enc.window_len().detach())
    hist.append({'epoch': epoch, **agg, 'window_s': L})
    if epoch % args.print_every == 0 or epoch == 1:
        print(f"  ep {epoch:4d}  cls={agg['cls']:.4f} acc={agg['acc']:.3f}  "
              f"alpha={agg['alpha']:.4f}  traj={agg['traj']:.4f}  "
              f"ctrl={agg['ctrl']:.4f}  L={L*1000:.0f} ms  "
              f"lr={sched.get_last_lr()[0]:.2e}  [{_hms(prog.elapsed())}]")

    def _ckpt(path, **extra):
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                    'args': vars(args), 'epoch': epoch, 'window_s': L,
                    'hist': hist, 'n_feat': N_FEAT, 'w_steps': W_STEPS,
                    'split': {'train': idx_tr, 'val': idx_va, 'test': idx_te},
                    **extra}, os.path.join(args.out, path))

    if epoch % args.val_every == 0:
        v = evaluate(idx_va, ctrl_on=(w_ctrl > 0))
        # Selection metric. dP is the deliverable, and traj plateaus long
        # before dP is finished -- selecting on traj stopped the last run at
        # epoch 176 with dP still unusable. So once the control head is on,
        # select on dP: failure rate first, norm as a tie-break.
        ctrl_live = (w_ctrl > 0) and v['dp_success'] == v['dp_success']
        use_dp = (args.select == 'dp') or (args.select == 'auto' and ctrl_live)
        if use_dp:
            score = ((1.0 - v['dp_rescue']) + 0.5 * (1.0 - v['dp_success'])
                     + 0.05 * v['dp_norm'])
        else:
            score = v['traj'] + v['cls']
        # the metric changes when control switches on; restart patience so the
        # run is not stopped by a stale identification-phase best
        if use_dp and not selecting_on_dp:
            selecting_on_dp = True
            best_score, since_improve = float('inf'), 0
            print(f"    [selection metric -> dP (rescue rate); patience reset]")
        hist[-1]['val'] = v
        improved = score < best_score - 1e-6
        if improved:
            best_score, best_epoch, since_improve = score, epoch, 0
            _ckpt('best.pt', val=v, score=score)
        else:
            since_improve += 1
        _ckpt('last.pt', val=v, score=score)
        dp_s = (f"  dP: rescue={v['dp_rescue']:.3f}"
                f"(all {v['dp_rescue_all']:.3f}, reach {v['dp_reachable']*100:.0f}%) "
                f"ok={v['dp_success']:.3f} |dP|={v['dp_norm']:.3f}pu"
                if w_ctrl > 0 else "")
        print(f"    val: traj={v['traj']:.5f}  acc={v['acc']:.3f}"
              f"(act {v['acc_active']:.3f}/type {v['acc_type']:.3f})  "
              f"ool={v['ool_acc']:.3f}(tpr {v['ool_tpr']:.2f} tnr {v['ool_tnr']:.2f})  "
              f"alpha={v['alpha']:.4f}{dp_s}"
              f"   window={L*1000:.1f}ms ({L*1000-args.init_window*1000:+.1f})"
              f"   score={score:.4f}"
              f"{'  <-- best' if improved else f'  ({since_improve}/{args.patience})'}")
        if args.patience and since_improve >= args.patience:
            print(f"\nEarly stop: no improvement for {args.patience} validations. "
                  f"Best was epoch {best_epoch} (score {best_score:.4f}).")
            break
        pass

json.dump(hist, open(os.path.join(args.out, 'history.json'), 'w'), indent=1)
print(f"\nFinal learned window: {float(enc.window_len().detach())*1000:.1f} ms")
print(f"Best epoch {best_epoch} (val score {best_score:.4f}) -> {args.out}/best.pt")
print(f"Latest state -> {args.out}/last.pt")

# ── test: used exactly once, here, on the best checkpoint ────────────────────
# Deliberately NOT touched during training -- scoring it alongside validation
# would turn it into a second validation set and the number would stop meaning
# anything.
bp = os.path.join(args.out, 'best.pt')
if os.path.exists(bp) and len(idx_te):
    ck = torch.load(bp, map_location=device, weights_only=False)
    model.load_state_dict(ck['model'])
    L_best = ck['window_s']
    te = evaluate(idx_te, ctrl_on=(args.w_ctrl > 0 or args.epochs >= args.ctrl_start))
    print(f"\n{'='*66}")
    print(f"TEST SET ({len(idx_te)} held-out scenarios, best.pt from epoch {ck['epoch']})")
    print(f"{'='*66}")
    print(f"  trajectory loss   {te['traj']:.5f}   "
          f"(omega RMSE {np.sqrt(te['traj'])*0.01:.2e}; noise floor "
          f"{(5e-5/0.01)**2:.2e} / 5.0e-05)")
    print(f"  hypothesis acc    {te['acc']:.4f}   "
          f"(fault-active only {te['acc_active']:.4f}, "
          f"type-only {te['acc_type']:.4f})")
    print(f"  OOL acc           {te['ool_acc']:.4f}   "
          f"(TPR {te['ool_tpr']:.3f}, TNR {te['ool_tnr']:.3f}; "
          f"vs measured future over the same horizon)")
    print(f"  alpha MSE         {te['alpha']:.5f}   "
          f"(RMSE {np.sqrt(te['alpha']):.4f} on a 0.1-1.0 range)")
    if te['dp_success'] == te['dp_success']:
        print(f"  dP rescue rate    {te['dp_rescue']:.4f}   "
              f"(REACHABLE cases: omega still inside the envelope at decision "
              f"time)")
        print(f"     unfiltered     {te['dp_rescue_all']:.4f}   "
              f"({te['dp_reachable']*100:.0f}% of needy cases were reachable; "
              f"the rest had already left the envelope)")
        print(f"     by fault age   " + "  ".join(
            f"{lo*1000:.0f}-{(hi*1000 if hi<1e8 else 9999):.0f}ms:"
            f"{r:.2f}(n={n})" for (lo, hi), (n, r) in
            zip(AGE_BINS, te['rescue_by_age'])))
        print(f"  dP overall ok     {te['dp_success']:.4f}   "
              f"(all judged cases inside the envelope after control)")
        print(f"  mean ||dP||       {te['dp_norm']:.4f} pu  "
              f"(label distance {te['dp_err']:.4f} pu = "
              f"{te['dp_rel']*100:.1f}% of ||y_delta_p||, reference only)")
    print(f"  learned window    {L_best*1000:.1f} ms")
    json.dump({'test': te, 'val_best': ck.get('val'), 'epoch': int(ck['epoch']),
               'window_s': float(L_best), 'n_test': int(len(idx_te))},
              open(os.path.join(args.out, 'test_results.json'), 'w'), indent=1)
    print(f"\n  -> {args.out}/test_results.json")
