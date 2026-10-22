"""
compare_models.py
=================
Architecture ablation, scored on IDENTICAL test cases.

Each training run reports its test numbers on its own draw of decision
instants, and the selection metric, the beta annealing and the split all leave
small differences between runs. This script removes those: every model is
scored on the same scenarios, at the same decision instants, with the same
true-dynamics simulation of the uncontrolled system. The only thing that
changes from column to column is the network.

Works for both systems; the system is read from the checkpoint (31 input
features = 9-bus, 88 = 30-bus). Run it from that system's folder, the one with
the MLP-capable pinn_model.py / pinn_model30.py (which still loads the old LSTM
checkpoints).

    # 9-bus (from bus_9/)
    python compare_models.py --data data_mm ^
        --ckpt "LSTM+LSTM=..\\PINN5\\receding_cascade\\best.pt" ^
        --ckpt "MLP+LSTM=receding_mlp\\best.pt" ^
        --ckpt "MLP+MLP=receding_fullmlp\\best.pt"

    # 30-bus (from bus_30/)
    python compare_models.py --data data30_mm ^
        --ckpt "LSTM+LSTM=case30_ctrl\\best.pt" ^
        --ckpt "MLP+LSTM=case30_mlp_ctrl\\best.pt" ^
        --ckpt "MLP+MLP=case30_fullmlp_ctrl\\best.pt"

Label format: "label=path" (label = encoder + dP cell). Without a label, one
is built from the checkpoint. The FIRST checkpoint is the reference in the
difference table.

What is measured, per model
---------------------------
  identification   decision accuracy (the hard rule the physics uses: declare
                   a fault if sum p_fault > p_none, then argmax), fault accuracy
                   given a fault is active, detection, alpha RMSE,
                   free-rollout omega RMSE, OOL TPR / TNR
  control          rescue rate in the model's own rollout, ||dP||
  TRUE dynamics    rescue rate when dP is applied to the real faulted network
                   (as verify_true), by fault type and by fault age, rescue
                   judged to the end of the run, HARMED (healthy cases pushed
                   out by dP)
  cost             parameters (encoder / dP head), learned window, latency

Uncertainty
-----------
95% intervals from a CLUSTER bootstrap over scenarios (instants from the same
scenario are correlated, so resampling instants would make the intervals too
narrow). Differences are bootstrapped PAIRED: the same resampled scenarios
for both models. These intervals cover test-set sampling only. They do NOT
cover training-seed noise (two runs of the same architecture differed by ~4
TPR points on the 30-bus); only a second seed per architecture measures that.

Trajectory figures
------------------
For each fault type, --plots_per_type cases (default 1) are drawn over the
whole run: ONE generator's speed, uncontrolled (dashed) and with each
architecture's dP applied in the true dynamics. The decision is placed just
after fault onset. The generator shown is the one with the largest
uncontrolled swing (--plot_gen N to choose). --plot_pick rescued | disagree |
random chooses which cases; --plot_scenario K [K ...] draws specific ones.
These are illustrations picked on purpose, not measurements.

Outputs (in --out): compare_results.json, compare_summary.csv, compare.png +
compare.svg, trajectories/<type>_sc<id>_G<n>.png + .svg, trajectories/trajectories.json
(every figure is written in both raster PNG and vector SVG)
"""

def save_fig(fig, base, dpi=150, bbox=None):
    """Save figure as both PNG (raster) and SVG (vector). base = path WITHOUT extension."""
    errs = []
    for ext in ('.png', '.svg'):
        try:
            fig.savefig(base + ext, dpi=dpi, **({'bbox_inches': bbox} if bbox else {}))
        except OSError:
            # Windows file lock / OneDrive sync race: retry once with a unique name
            try:
                fig.savefig(base + '_' + str(int(time.time())) + ext, dpi=dpi,
                            **({'bbox_inches': bbox} if bbox else {}))
            except OSError as e2:
                errs.append(f'{ext}: {e2}')
    if errs:
        print('  [save_fig] failed:', '; '.join(errs))

import argparse, csv, json, math, os, sys, time

import numpy as np
import torch

ap = argparse.ArgumentParser(description=__doc__.split('\n')[1],
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--ckpt', action='append', required=True,
                help='"label=path" or "path"; repeat. First one is the reference.')
ap.add_argument('--data', required=True)
ap.add_argument('--base', default=None, help='default: ieee9_base.mat / case30_base.mat')
ap.add_argument('--sys', default=None,
                help='default: ieee9_gov_simulink_data.mat / case30_simulink_data.mat')
ap.add_argument('--split', default='test', choices=['test', 'val'])
ap.add_argument('--n', type=int, default=4096, help='decision instants (all metrics)')
ap.add_argument('--n_true', type=int, default=2048,
                help='of those, how many also get the true-dynamics check '
                     '(the slow part: two 8 s simulations per instant per model)')
ap.add_argument('--no_true', action='store_true', help='skip true dynamics')
ap.add_argument('--batch', type=int, default=256)
ap.add_argument('--dt_true', type=float, default=0.005)
ap.add_argument('--dp_hold', default='clear', choices=['clear', 'window', 'run'])
ap.add_argument('--beta', default='ckpt', choices=['ckpt', 'min'],
                help="mask sharpness: 'ckpt' (default) = the value stored in the "
                     "checkpoint, i.e. what training validated and tested with; "
                     "'min' = force beta_min, as the 9-bus verify/refine scripts do")
ap.add_argument('--boot', type=int, default=2000, help='bootstrap resamples')
ap.add_argument('--latency_reps', type=int, default=30)
ap.add_argument('--seed', type=int, default=11)
ap.add_argument('--out', default='compare_out')
ap.add_argument('--instants-file', default=None,
                help='load KS/TK draw from a json (shared with '
                     'verify_true); with no file, the draw is made '
                     'and saved to <out>/instants.json')
ap.add_argument('--no_plot', action='store_true')
ap.add_argument('--plots_per_type', type=int, default=1,
                help='trajectory figures per fault type (gen / load / line): one '
                     'generator, whole run, uncontrolled vs each architecture\'s '
                     'dP in the TRUE dynamics. 0 = none.')
ap.add_argument('--plot_pick', default='rescued',
                choices=['rescued', 'disagree', 'random'],
                help="which cases to draw. 'rescued' (default): cases that need "
                     "rescue and that every model keeps inside the envelope for "
                     "the whole run, if any exist. 'disagree': cases where the "
                     "models' outcomes differ. 'random': any case needing rescue.")
ap.add_argument('--plot_gen', type=int, default=0,
                help='generator to draw (1-based). 0 = the machine with the '
                     'largest uncontrolled excursion in that case.')
ap.add_argument('--plot_scenario', type=int, nargs='*', default=None,
                help='draw these scenario ids instead of searching for cases')
ap.add_argument('--plot_candidates', type=int, default=64,
                help='scenarios examined per fault type when picking cases')
args = ap.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(args.out, exist_ok=True)
TYPE_NAMES = ['gen', 'load', 'line']
AGE_BINS = [(0.0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1e9)]
AGE_LABELS = ['0-50', '50-100', '100-200', '200-500', '>500']


def parse_ckpt(s):
    if '=' in s and not os.path.exists(s):
        lab, p = s.split('=', 1)
        return lab.strip(), p.strip()
    return None, s


specs = [parse_ckpt(s) for s in args.ckpt]
for _, p in specs:
    if not os.path.exists(p):
        sys.exit(f"not found: {p}")
first = torch.load(specs[0][1], map_location='cpu', weights_only=False)
SYSTEM = 30 if int(first['n_feat']) == 88 else 9
del first


# ═════════════════════════════════════════════════════════════════════════════
# System adapters. Each loaded model exposes:
#   model, A, ck, split dict, make_batch(ks, tk) -> dict on device
# and the system object exposes the ground truth and the true dynamics.
# Both batch builders mirror their training script line for line.
# ═════════════════════════════════════════════════════════════════════════════
if SYSTEM == 30:
    import case30_common as C30
    from case30_physics import NG, N_STATE

    class Loaded:
        def __init__(self, path):
            su = C30.Setup(path, args.data, args.base or 'case30_base.mat',
                           args.sys or 'case30_simulink_data.mat',
                           device=device, verbose=False)
            self.su, self.model, self.A, self.ck = su, su.model, su.A, su.ck
            self.split = {'train': su.idx_tr, 'val': su.idx_va, 'test': su.idx_te}
            self.S = su.S

        def make_batch(self, ks, tk):
            b = self.su.make_batch(ks, np.random.default_rng(0), tk_fixed=tk)
            return self.su.to_dev(b)

    class System:
        def __init__(self, L):
            su = L.su
            self.S, self.su = su.S, su
            self.t_fault, self.t_clear, self.t_grid = su.t_fault, su.t_clear, su.t_grid
            self.y_type, self.meta_loc, self.meta_alpha = su.y_type, su.meta_loc, su.meta_alpha
            self.limit = su.OMEGA_LIMIT
            H = su.HYPOTHESES
            self.HYP = H
            self.sc_class = np.array([H.index((TYPE_NAMES[self.y_type[k] - 1],
                                               int(self.meta_loc[k]))) + 1
                                      for k in range(len(self.y_type))])
            self.x0 = su.S.x0.to(torch.float32)
            self.act_idx = L.model.act_idx

        def true_network(self, ks):
            return self.su.true_network(ks)

else:
    from ieee9_physics import IEEE9, HYPOTHESES
    from pinn_model import FaultEncoder, RolloutController
    NG, N_STATE = 3, 9
    _X = np.load(os.path.join(args.data, 'X.npy'), mmap_mode='r')
    _md = np.load(os.path.join(args.data, 'meta.npz'))
    _lines = (_md['lines'] if 'lines' in _md.files else _md['branches']).astype(int)
    _S9 = IEEE9(args.base or 'ieee9_base.mat', args.sys or 'ieee9_gov_simulink_data.mat',
                dtype=torch.float32, device=device, lines=_lines)
    D_IDX, W_IDX, PM_IDX = [0, 2, 4], [1, 3, 5], [6, 7, 8]
    ID_IDX, IQ_IDX = [9, 11, 13], [10, 12, 14]
    VM_IDX = list(range(15, 33, 2)); TH_IDX = list(range(16, 33, 2))
    FUT_IDX = [1, 3, 5, 6, 7, 8]

    class Loaded:
        def __init__(self, path):
            ck = torch.load(path, map_location=device, weights_only=False)
            A = ck['args']
            self.ck, self.A, self.S = ck, A, _S9
            self.DT = float(_md['dt_solver'])
            self.W_STEPS = int(ck['w_steps'])
            self.W_STRIDE = int(round(A['dt_window'] / self.DT))
            self.ROLL_STRIDE = int(round(A['dt_roll'] / self.DT))
            self.N_ROLL = int(round(A['horizon'] / A['dt_roll']))
            enc = FaultEncoder(int(ck['n_feat']), w_max_steps=self.W_STEPS,
                               dt_window=A['dt_window'],
                               lstm_layers=A.get('lstm_layers', 10),
                               init_window=A['init_window'], max_window=A['w_max'],
                               beta0=A['beta0'], encoder=A.get('encoder', 'lstm'),
                               mlp_proj=A.get('mlp_proj'))
            self.model = RolloutController(
                _S9, enc, horizon=A['horizon'], dt_roll=A['dt_roll'],
                omega_limit=float(_md['omega_limit']), margin=A.get('margin', 1e-4),
                dp_head=A.get('dp_head', 'linear'), dp_hidden=A.get('dp_hidden', 128),
                dp_fc_layers=A.get('dp_fc_layers', 2), n_refine=A.get('n_refine', 3),
                dp_cell=A.get('dp_cell', 'lstm'),
                alpha_lo=A.get('alpha_lo', 0.10),
                alpha_hi=A.get('alpha_hi', 1.00)).to(device)
            self.model.load_state_dict(ck['model'])
            sp = ck['split']
            self.split = {k: np.asarray(sp[k]) for k in ('train', 'val', 'test')}

        def make_batch(self, ks, tk):
            """verify_true.py / train_receding.py make_batch, at fixed instants."""
            ks = np.asarray(ks); bs = len(ks); A = self.A
            i_end = np.round(np.asarray(tk) / self.DT).astype(int)
            win = np.empty((bs, self.W_STEPS, _X.shape[2]), np.float32)
            fut = np.empty((bs, self.N_ROLL + 1, 6), np.float32)
            for j, (k, ie) in enumerate(zip(ks, i_end)):
                i0 = ie - (self.W_STEPS - 1) * self.W_STRIDE
                blk = np.asarray(_X[k, i0:ie + self.N_ROLL * self.ROLL_STRIDE + 1])
                o = ie - i0
                win[j] = blk[0:o + 1:self.W_STRIDE]
                fut[j] = blk[o::self.ROLL_STRIDE][:self.N_ROLL + 1][:, FUT_IDX]
            d = win[..., D_IDX]; w = win[..., W_IDX]; pm = win[..., PM_IDX]
            th = win[..., TH_IDX]
            feat = np.concatenate([d[..., 1:] - d[..., :1], (w - 1.0) / 0.01, pm,
                                   win[..., ID_IDX], win[..., IQ_IDX],
                                   win[..., VM_IDX], th[..., 1:] - th[..., :1]], -1)
            re = win[:, -1, :]; dd = re[:, D_IDX]
            x_meas = np.concatenate([np.stack([np.zeros(bs), dd[:, 1] - dd[:, 0],
                                               dd[:, 2] - dd[:, 0]], 1),
                                     re[:, W_IDX], re[:, PM_IDX]], 1)
            tf, tc = SYS.t_fault[ks], SYS.t_clear[ks]
            active = (tk >= tf) & (tk < tc)
            t_change = np.where(active, tc, np.where(tk < tf, tf, np.inf))
            valid_t = np.clip(t_change - tk, 0.0, A['horizon'])
            step_t = np.arange(self.N_ROLL + 1) * A['dt_roll']
            vmask = (step_t[None, :] <= valid_t[:, None]).astype(np.float32)
            T = lambda a, dt=None: torch.tensor(a, dtype=dt).to(device)
            return dict(feat=T(feat), x_meas=T(x_meas, torch.float32), fut=T(fut),
                        vmask=T(vmask), active=T(active), ks=ks, tk=np.asarray(tk),
                        valid_t=valid_t)

    class System:
        def __init__(self, L):
            g = lambda k: _md[k].ravel()
            self.S = _S9
            self.t_fault, self.t_clear, self.t_grid = g('y_t_fault'), g('meta_t_clear'), _md['t'].ravel()
            self.y_type = g('y_type').astype(int)
            self.meta_loc = g('meta_loc').astype(int)
            self.meta_alpha = g('meta_alpha')
            self.limit = float(_md['omega_limit'])
            self.HYP = HYPOTHESES
            self.sc_class = np.array([HYPOTHESES.index((TYPE_NAMES[self.y_type[k] - 1],
                                                        int(self.meta_loc[k]))) + 1
                                      for k in range(len(self.y_type))])
            S = _S9
            self.x0 = torch.cat([S.delta0, torch.ones(3, device=device), S.Pm0])
            self.act_idx = None

        def true_network(self, ks):
            n = len(ks); S = self.S
            a = torch.tensor(self.meta_alpha[ks], dtype=torch.float32, device=device)
            Yf = torch.zeros(n, 3, 3, dtype=S.cdtype, device=device)
            dPm = torch.zeros(n, 3, dtype=torch.float32, device=device)
            for j, k in enumerate(ks):
                h = int(self.sc_class[k]) - 1
                Yf[j], _ = S.kron_reduce(S.faulted_Y(h, a[j:j + 1]))
                dPm[j] = S.dPm0(h, a[j:j + 1])[0]
            return Yf, dPm


# ── load everything ──────────────────────────────────────────────────────────
print(f"Device: {device}   system: {SYSTEM}-bus   split: {args.split}")
models = []
for lab, path in specs:
    L = Loaded(path)
    A = L.A
    L.path = path
    L.label = lab or (f"{A.get('encoder', 'lstm').upper()}+"
                      f"{A.get('dp_cell', 'lstm').upper()}")
    if args.beta == 'min':
        L.model.enc.beta.fill_(A['beta_min'])
    L.model.eval()
    models.append(L)
if len({m.label for m in models}) < len(models):
    sys.exit("two checkpoints have the same label; give them explicit 'label=path'")
SYS = System(models[0])

# the models must share the geometry that defines a decision and its judgement
for key in ('w_max', 'horizon', 'dt_roll', 'dt_window'):
    vals = {m.label: m.A[key] for m in models}
    if len(set(vals.values())) > 1:
        sys.exit(f"'{key}' differs between checkpoints {vals}: the judged windows "
                 f"would differ, so the numbers would not be comparable")
for m in models[1:]:
    if abs(m.model.limit - models[0].model.limit) > 1e-12 or \
       abs(m.model.grace - models[0].model.grace) > 1e-12:
        sys.exit("omega limit or grace period differs between checkpoints")
LIMIT, GRACE = models[0].model.limit, models[0].model.grace

# common evaluation pool: in every model's test (or val) split, in nobody's train
seen = set()
for m in models:
    seen |= set(m.split['train'].tolist())
    if args.split == 'test':
        seen |= set(m.split['val'].tolist())
pool = set(models[0].split[args.split].tolist())
same = all(set(m.split[args.split].tolist()) == pool for m in models[1:])
for m in models[1:]:
    pool &= set(m.split[args.split].tolist())
pool = np.array(sorted(pool - seen))
print(f"Splits identical across models: {'yes' if same else 'NO'}  ->  "
      f"{len(pool)} {args.split} scenarios usable by every model")
if not same:
    print("  (scenarios that any model trained or validated on are excluded)")
if len(pool) == 0:
    sys.exit("no held-out scenario is shared by every model. Were they trained "
             "with different --seed values or on different datasets?")
if len(pool) < 50:
    print(f"  WARNING: only {len(pool)} shared held-out scenarios -- intervals "
          f"will be wide and the comparison weak")


# ── parameters, window ───────────────────────────────────────────────────────
def count(mod, pfx=None):
    return sum(p.numel() for n, p in mod.named_parameters()
               if pfx is None or n.startswith(pfx))

for m in models:
    mm = m.model
    m.params = dict(total=count(mm), encoder=count(mm, 'enc.'),
                    dp_head=count(mm, ('dp_cell.', 'dp_fc.')))
    m.window_ms = float(mm.enc.window_len().detach()) * 1000
    m.has_lstm = any('LSTM' in type(x).__name__ for x in mm.modules())
    print(f"  {m.label:10s} {m.path}\n"
          f"             epoch {m.ck.get('epoch')}, {m.params['total']:,} params "
          f"(encoder {m.params['encoder']:,}, dP head {m.params['dp_head']:,}), "
          f"window {m.window_ms:.0f} ms, beta {float(mm.enc.beta)*1000:.1f} ms, "
          f"LSTM layers anywhere: {'yes' if m.has_lstm else 'NO'}")


# ── decision instants, drawn once, as training draws them ───────────────────
A0 = models[0].A
if args.instants_file:
    _inst = json.load(open(args.instants_file))
    KS = np.asarray(_inst['ks'], dtype=int)
    TK = np.asarray(_inst['tk'], dtype=float)
    assert len(KS) == args.n, f"--instants-file has {len(KS)} instants, --n is {args.n}"
    print(f"instants loaded from {args.instants_file} (shared draw)")
else:
    rng = np.random.default_rng(args.seed)
    KS = rng.choice(pool, args.n)
    lo, hi = A0['w_max'], float(SYS.t_grid[-1]) - A0['horizon']
    TK = rng.uniform(lo, hi, args.n)
    if A0.get('p_active', 0) > 0:
        pick = rng.random(args.n) < A0['p_active']
        a = np.clip(SYS.t_fault[KS], lo, hi); b = np.clip(SYS.t_clear[KS], lo, hi)
        if A0.get('onset_window', 0) and A0['onset_window'] > 0:
            b = np.minimum(b, a + A0['onset_window'])
        ok = pick & (b > a)
        TK[ok] = rng.uniform(a[ok], b[ok])
    os.makedirs(args.out, exist_ok=True)
    json.dump({'ks': KS.tolist(), 'tk': TK.tolist(), 'seed': args.seed,
               'split': args.split, 'pool_note': 'test minus train/val'},
              open(os.path.join(args.out, 'instants.json'), 'w'), indent=1)
    print(f"instant draw saved -> {os.path.join(args.out, 'instants.json')}")
N_TRUE = 0 if args.no_true else min(args.n_true, args.n)
ACTIVE = (TK >= SYS.t_fault[KS]) & (TK < SYS.t_clear[KS])
AGE = TK - SYS.t_fault[KS]
print(f"\n{args.n} decision instants over {len(np.unique(KS))} scenarios "
      f"({ACTIVE.mean()*100:.0f}% with a fault active); true dynamics on the "
      f"first {N_TRUE}")


# ── true dynamics (one integrator for both systems) ─────────────────────────
@torch.no_grad()
def true_sim(ks, tk, valid_t, dP_full=None, record=False):
    """
    Whole run from the pre-fault equilibrium in the TRUE network:
    pre-fault -> faulted on [t_fault, t_clear) -> pre-fault. dP (NG-wide) is
    held from tk until --dp_hold removes it. Returns the peak |omega-1| after
    the grace period over the judged window and over the rest of the run (-1
    where empty), and the clean state at tk. Same masks as verify_true.
    """
    S = SYS.S; n = len(ks)
    Yf, dPm = SYS.true_network(ks)
    Mp, Np = S.pe_consts(S.Yred_pre.expand(n, NG, NG))
    Mf, Nf = S.pe_consts(Yf)
    f32 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
    tf, tc, tkt, vt = f32(SYS.t_fault[ks]), f32(SYS.t_clear[ks]), f32(tk), f32(valid_t)
    t_off = {'clear': tc, 'run': torch.full_like(tkt, 1e9), 'window': tkt + vt}[args.dp_hold]
    x = SYS.x0.unsqueeze(0).expand(n, N_STATE).clone()
    h = args.dt_true
    n_steps = int(round(float(SYS.t_grid[-1]) / h))
    i_tk = torch.as_tensor(np.round(np.asarray(tk) / h).astype(int), device=device)
    x_at = torch.zeros_like(x)
    pk_win = torch.full((n,), -1.0, device=device); pk_run = pk_win.clone()
    traj = torch.zeros(n, n_steps + 1, NG, device=device) if record else None
    if record:
        traj[:, 0] = x[:, NG:2 * NG]
    for i in range(n_steps):
        tm = i * h + h / 2
        act = (tm >= tf) & (tm < tc)
        M = torch.where(act.view(n, 1, 1), Mf, Mp)
        N_ = torch.where(act.view(n, 1, 1), Nf, Np)
        Ps = S.Pm0.unsqueeze(0) + dPm * act.view(n, 1).float()
        if dP_full is not None:
            on = (tm >= tkt) & (tm < t_off)
            Ps = Ps + dP_full * on.view(n, 1).float()
        x = S.rk4_real(x, M, N_, Ps, h)
        t_now = (i + 1) * h
        d = (x[:, NG:2 * NG] - 1.0).abs().amax(1)
        in_run = t_now >= tkt + GRACE
        in_win = in_run & (t_now <= tkt + vt)
        pk_run = torch.where(in_run, torch.maximum(pk_run, d), pk_run)
        pk_win = torch.where(in_win, torch.maximum(pk_win, d), pk_win)
        hit = i_tk == (i + 1)
        if hit.any():
            x_at[hit] = x[hit]
        if record:
            traj[:, i + 1] = x[:, NG:2 * NG]
    if record:
        return pk_win, pk_run, x_at, traj
    return pk_win, pk_run, x_at


def full_dp(dP):
    if SYS.act_idx is None or dP.shape[1] == NG:
        return dP
    out = torch.zeros(dP.shape[0], NG, dtype=dP.dtype, device=dP.device)
    return out.index_copy(1, SYS.act_idx, dP)


# ── per-instant records ─────────────────────────────────────────────────────
def zeros(dt=np.float64):
    return np.zeros(args.n, dt)

shared = dict(need_true=zeros(bool), healthy=zeros(bool), free_run_ok=zeros(bool),
              true_eval=zeros(bool))
for m in models:
    m.r = {k: zeros() for k in (
        'dec_ok', 'det_ok', 'fault_ok', 'type_ok', 'alpha_se', 'alpha_sd', 'alpha_m',
        'traj_se', 'traj_n', 'pred_ool', 'true_ool', 'has', 'need_m', 'won_m',
        'dp_norm', 'won_t', 'run_ok', 'harmed')}

ALPHA_CLASS = np.array([False] + [h[0] in ('gen', 'load') for h in SYS.HYP])
TYPE_OF = np.array([0] + [TYPE_NAMES.index(h[0]) + 1 for h in SYS.HYP])
CLS = np.where(ACTIVE, SYS.sc_class[KS], 0)

t0 = time.time()
for s in range(0, args.n, args.batch):
    sl = slice(s, min(s + args.batch, args.n)); n_b = sl.stop - sl.start
    ks, tk = KS[sl], TK[sl]
    do_true = sl.start < N_TRUE
    sub = np.arange(n_b) < (N_TRUE - sl.start)            # within the true subset
    vt_ref = None
    for mi, m in enumerate(models):
        b = m.make_batch(ks, tk)
        if vt_ref is None:
            vt_ref = b['valid_t']
            if do_true:
                pk_free, pk_free_run, x_clean = true_sim(ks, tk, vt_ref)
                active_t = b['active']
                reach_t = ((pk_free >= 0)
                                   & ((x_clean[:, NG:2 * NG] - 1).abs().amax(1) <= LIMIT)
                                   & active_t)
                need_t = (pk_free > LIMIT) & reach_t
                healthy = (pk_free >= 0) & (pk_free <= LIMIT)
                idx = np.arange(sl.start, sl.stop)[sub]
                st = torch.as_tensor(sub, device=device)
                shared['need_true'][idx] = need_t[st].cpu().numpy()
                shared['healthy'][idx] = healthy[st].cpu().numpy()
                shared['free_run_ok'][idx] = (pk_free_run <= LIMIT)[st].cpu().numpy()
                shared['true_eval'][idx] = True
        with torch.no_grad():
            o = m.model(b['feat'], b['x_meas'], hard=True, ctrl=True, vmask=b['vmask'])
        lg = o['hyp_logits']
        cls = torch.as_tensor(CLS[sl], device=device)
        act = b['active']
        det = lg[:, 1:].logsumexp(1) > lg[:, 0]
        pred_f = lg[:, 1:].argmax(1) + 1
        pred_hard = torch.where(det, pred_f, torch.zeros_like(pred_f))
        R = m.r; cpu = lambda t: t.detach().float().cpu().numpy()
        R['dec_ok'][sl] = cpu(pred_hard == cls)
        R['det_ok'][sl] = cpu(det == act)
        R['fault_ok'][sl] = cpu((pred_f == cls) & act)
        to = torch.as_tensor(TYPE_OF, device=device)
        R['type_ok'][sl] = cpu((to[pred_f] == to[cls]) & act)
        am = act & torch.as_tensor(ALPHA_CLASS, device=device)[cls]
        at = torch.as_tensor(SYS.meta_alpha[ks], dtype=torch.float32, device=device)
        R['alpha_se'][sl] = cpu((o['alpha'] - at) ** 2 * am)
        # Signed residual, same mask as the RMSE. RMSE cannot distinguish a model
        # that scatters symmetrically from one that is consistently low: both are
        # unsafe in different ways, and only the sign says which. Masked to
        # ALPHA_CLASS (gen + load) because line-fault alpha is neither trained
        # (masked out of the loss) nor used (line faults are binary in the network
        # build), so pooling it would average in a free parameter.
        R['alpha_sd'][sl] = cpu((o['alpha'] - at) * am)
        R['alpha_m'][sl] = cpu(am)
        se = (o['w_free'] - b['fut'][:, :, 0:NG]) ** 2 * b['vmask'].unsqueeze(-1)
        R['traj_se'][sl] = cpu(se.sum((1, 2)))
        R['traj_n'][sl] = cpu(b['vmask'].sum(1) * NG)
        pk_f, has = m.model.judged(o['w_free'], b['vmask'])
        pk_t, _ = m.model.judged(b['fut'][:, :, 0:NG], b['vmask'])
        R['pred_ool'][sl] = cpu((pk_f > LIMIT) & has)
        R['true_ool'][sl] = cpu((pk_t > LIMIT) & has)
        R['has'][sl] = cpu(has)
        pk_c, has_c = m.model.judged(o['w_ctrl'], b['vmask'])
        reach = has_c & ((b['x_meas'][:, NG:2 * NG] - 1).abs().amax(1) <= LIMIT) & act
        R['need_m'][sl] = cpu((pk_t > LIMIT) & reach)
        R['won_m'][sl] = cpu(pk_c <= LIMIT)
        R['dp_norm'][sl] = cpu(o['dP'].pow(2).sum(1).sqrt())
        if do_true:
            pk_c_t, pk_c_run, _ = true_sim(ks, tk, vt_ref, full_dp(o['dP']))
            st = torch.as_tensor(sub, device=device)
            idx = np.arange(sl.start, sl.stop)[sub]
            R['won_t'][idx] = cpu(pk_c_t <= LIMIT)[sub]
            R['run_ok'][idx] = cpu(pk_c_run <= LIMIT)[sub]
            R['harmed'][idx] = cpu(healthy & (pk_c_t > LIMIT))[sub]
    el = time.time() - t0
    print(f"\r  {sl.stop}/{args.n} instants  [{el/60:.1f} min, "
          f"~{el/sl.stop*(args.n-sl.stop)/60:.1f} min left]", end='', flush=True)
print()


# ── latency ─────────────────────────────────────────────────────────────────
def sync():
    if device.type == 'cuda':
        torch.cuda.synchronize()

@torch.no_grad()
def latency(m, bs):
    b = m.make_batch(KS[:bs], TK[:bs])
    f = lambda: m.model(b['feat'], b['x_meas'], hard=True, ctrl=True, vmask=b['vmask'])
    g = lambda: m.model.enc(b['feat'])
    out = {}
    for name, fn in (('decision', f), ('encoder', g)):
        for _ in range(3):
            fn()
        sync(); t = time.perf_counter()
        for _ in range(args.latency_reps):
            fn()
        sync(); out[name] = (time.perf_counter() - t) / args.latency_reps * 1000
    return out

for m in models:
    m.lat1 = latency(m, 1)
    m.latB = latency(m, min(args.batch, args.n))


# ── metrics as ratio estimators, cluster bootstrap over scenarios ───────────
need_t = shared['need_true']; teval = shared['true_eval']
TYP = SYS.y_type[KS]

def metric_defs(R):
    """name -> (numerator, denominator, transform, higher_is_better)"""
    one = np.ones(args.n)
    act = ACTIVE.astype(float)
    sq = lambda v: math.sqrt(v) if v >= 0 else float('nan')
    D = {
        'decision_acc':  (R['dec_ok'], one, None, True),
        'fault_acc':     (R['fault_ok'], act, None, True),
        'type_acc':      (R['type_ok'], act, None, True),
        'detect_acc':    (R['det_ok'], one, None, True),
        'alpha_rmse':    (R['alpha_se'], R['alpha_m'], sq, False),
        'alpha_bias':    (R['alpha_sd'], R['alpha_m'], None, None),
        'omega_rmse':    (R['traj_se'], R['traj_n'], sq, False),
        'ool_tpr':       (R['pred_ool'] * R['true_ool'], R['true_ool'], None, True),
        'ool_tnr':       ((1 - R['pred_ool']) * (1 - R['true_ool']) * R['has'],
                          (1 - R['true_ool']) * R['has'], None, True),
        'rescue_model':  (R['won_m'] * R['need_m'], R['need_m'], None, True),
        'dP_norm':       (R['dp_norm'], one, None, False),
    }
    if N_TRUE:
        nt = need_t.astype(float)
        D.update({
            'rescue_true':    (R['won_t'] * nt, nt, None, True),
            'rescue_true_end_of_run': (R['run_ok'] * nt, nt, None, True),
            'harmed_pct':     (R['harmed'] * 100, shared['healthy'].astype(float), None, False),
            'dP_norm_need':   (R['dp_norm'] * nt, nt, None, False),
        })
        for ti, tn in enumerate(TYPE_NAMES):
            mk = nt * (TYP == ti + 1)
            D[f'rescue_true_{tn}'] = (R['won_t'] * mk, mk, None, True)
        for (b0, b1), lab in zip(AGE_BINS, AGE_LABELS):
            mk = nt * ((AGE >= b0) & (AGE < b1))
            D[f'rescue_true_age_{lab}ms'] = (R['won_t'] * mk, mk, None, True)
    for (b0, b1), lab in zip(AGE_BINS, AGE_LABELS):
        mk = act * ((AGE >= b0) & (AGE < b1))
        D[f'fault_acc_age_{lab}ms'] = (R['fault_ok'] * mk, mk, None, True)
    return D

# per-scenario sums -> bootstrap by multinomial scenario weights
uniq, inv = np.unique(KS, return_inverse=True)
n_sc = len(uniq)
brng = np.random.default_rng(args.seed + 7)
W = brng.multinomial(n_sc, np.full(n_sc, 1.0 / n_sc), size=args.boot).astype(np.float64)

def per_sc(v):
    out = np.zeros(n_sc); np.add.at(out, inv, v); return out

def est(num, den, tf):
    with np.errstate(invalid='ignore', divide='ignore'):
        r = num / den
    if tf is not None:
        r = np.vectorize(tf)(r) if np.ndim(r) else tf(r)
    return r

for m in models:
    m.D = metric_defs(m.r)
    m.point, m.boot = {}, {}
    for k, (num, den, tf, _) in m.D.items():
        ns, ds = per_sc(num), per_sc(den)
        m.point[k] = float(est(ns.sum(), ds.sum(), tf)) if ds.sum() > 0 else float('nan')
        m.boot[k] = est(W @ ns, W @ ds, tf)
    m.n_need = {k: int(round(m.D[k][1].sum())) for k in m.D}
    m.n_sc = {k: int((per_sc(m.D[k][1]) > 0).sum()) for k in m.D}

METRICS = list(models[0].D.keys())
HIB = {k: models[0].D[k][3] for k in METRICS}


def ci(a):
    a = a[np.isfinite(a)]
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if len(a) else (np.nan, np.nan)


# ── report ──────────────────────────────────────────────────────────────────
def fmt(v, k):
    if not np.isfinite(v):
        return '   n/a'
    if k == 'omega_rmse':
        return f"{v:.2e}"
    if k == 'harmed_pct':
        return f"{v:.2f}%"
    return f"{v:.4f}"

CW = 26
print(f"\n{'=' * (28 + CW * len(models))}")
print(f"ARCHITECTURE COMPARISON  ({SYSTEM}-bus, {args.split}, {args.n} instants, "
      f"{len(uniq)} scenarios, same instants for every model)")
print(f"{'=' * (28 + CW * len(models))}")
hdr = f"{'metric':28s}" + "".join(f"{m.label:>{CW}s}" for m in models)
GROUPS = [
    ('IDENTIFICATION', ['decision_acc', 'fault_acc', 'type_acc', 'detect_acc',
                        'alpha_rmse', 'alpha_bias', 'omega_rmse', 'ool_tpr',
                        'ool_tnr']),
    ('CONTROL (model rollout)', ['rescue_model', 'dP_norm']),
    ('CONTROL (TRUE dynamics)', ['rescue_true', 'rescue_true_gen', 'rescue_true_load',
                                 'rescue_true_line', 'rescue_true_end_of_run',
                                 'harmed_pct', 'dP_norm_need']),
    ('TRUE RESCUE BY FAULT AGE', [k for k in METRICS if k.startswith('rescue_true_age')]),
    ('FAULT ACCURACY BY FAULT AGE', [k for k in METRICS if k.startswith('fault_acc_age')]),
]
print(hdr)
for gname, keys in GROUPS:
    keys = [k for k in keys if k in METRICS]
    if not keys:
        continue
    print(f"-- {gname}")
    for k in keys:
        row = f"  {k:26s}"
        for m in models:
            lo_, hi_ = ci(m.boot[k])
            cell = f"{fmt(m.point[k], k)} [{fmt(lo_, k).strip()}-{fmt(hi_, k).strip()}]" \
                if k != 'omega_rmse' else f"{fmt(m.point[k], k)}"
            row += f"{cell:>{CW}s}"
        print(row)
if 'alpha_bias' in METRICS:
    print("-- SEVERITY BIAS  (signed, gen+load only; negative = UNDER-estimates alpha)")
    for m in models:
        lo_, hi_ = ci(m.boot['alpha_bias'])
        nz = np.isfinite(lo_) and (lo_ > 0 or hi_ < 0)
        verdict = (('UNDER-estimates severity' if m.point['alpha_bias'] < 0
                    else 'OVER-estimates severity') if nz else 'no detectable bias')
        print(f"  {m.label:26s}{m.point['alpha_bias']:+.4f} "
              f"[{lo_:+.4f}, {hi_:+.4f}]   {verdict}")
    print("     RMSE mixes bias and scatter; this row separates them. A model that "
          "scatters\n     symmetrically and one that is consistently low can share an "
          "RMSE while failing\n     in different ways. UNDER-estimation is the unsafe "
          "direction: dP is then sized\n     for a milder fault than the one actually "
          "occurring.")
print("-- COST")
for name, f_ in (('parameters', lambda m: f"{m.params['total']:,}"),
                 ('  encoder', lambda m: f"{m.params['encoder']:,}"),
                 ('  dP head', lambda m: f"{m.params['dp_head']:,}"),
                 ('learned window', lambda m: f"{m.window_ms:.0f} ms"),
                 ('latency, 1 decision', lambda m: f"{m.lat1['decision']:.1f} ms"),
                 ('  of which encoder', lambda m: f"{m.lat1['encoder']:.2f} ms"),
                 (f'latency/instant @{min(args.batch, args.n)}',
                  lambda m: f"{m.latB['decision'] / min(args.batch, args.n):.3f} ms")):
    print(f"  {name:26s}" + "".join(f"{f_(m):>{CW}s}" for m in models))
if N_TRUE:
    print(f"\n  cases needing rescue (true dynamics): {int(need_t.sum())} of {N_TRUE};"
          f" without dP {shared['free_run_ok'][need_t].mean()*100:.1f}% of them "
          f"would end the run inside the envelope anyway")

# paired differences vs every other model
pairs = [(i, j) for i in range(len(models)) for j in range(i + 1, len(models))]
KEY = ['decision_acc', 'fault_acc', 'alpha_rmse', 'alpha_bias', 'omega_rmse',
       'ool_tpr', 'ool_tnr',
       'rescue_model', 'dP_norm'] + (['rescue_true', 'rescue_true_gen',
                                      'rescue_true_end_of_run', 'harmed_pct'] if N_TRUE else [])
diffs = {}
print(f"\nPAIRED DIFFERENCES (B - A, 95% cluster-bootstrap CI; * = CI excludes 0, "
      f"needs >= 10 scenarios behind the metric)")
for i, j in pairs:
    a_, b_ = models[i], models[j]
    print(f"-- A = {a_.label}, B = {b_.label}")
    dd = {}
    for k in KEY:
        d = b_.boot[k] - a_.boot[k]
        lo_, hi_ = ci(d)
        pt = b_.point[k] - a_.point[k]
        n_sup = min(a_.n_sc[k], b_.n_sc[k])     # scenarios the metric rests on
        sc = 1e4 if k == 'omega_rmse' else 1
        unit = ' (x1e-4)' if k == 'omega_rmse' else ''
        if not np.isfinite(pt):
            print(f"  {k + unit:30s}      n/a  (no cases)")
            dd[k] = dict(diff=None, lo=None, hi=None, significant=False, n_scenarios=n_sup)
            continue
        few = n_sup < 10
        sig = (not few) and np.isfinite(lo_) and (lo_ > 0 or hi_ < 0)
        # HIB None means neither sign is better and zero is (alpha_bias): the
        # meaningful comparison is which model sits closer to no bias at all.
        better = (abs(b_.point[k]) < abs(a_.point[k])) if HIB[k] is None \
            else (pt > 0) == HIB[k]
        dd[k] = dict(diff=pt, lo=lo_, hi=hi_, significant=bool(sig), n_scenarios=n_sup)
        tag = (f'too few scenarios ({n_sup}) to tell' if few else
               ((('B closer to 0' if better else 'A closer to 0') if HIB[k] is None
                 else ('B better' if better else 'A better')) if sig
                else 'no clear difference'))
        print(f"  {k + unit:30s} {pt*sc:+.4f}  [{lo_*sc:+.4f}, {hi_*sc:+.4f}]"
              f"{' *' if sig else '  '}  {tag}")
    if N_TRUE:
        nd = need_t
        wa, wb = a_.r['won_t'][nd] > 0, b_.r['won_t'][nd] > 0
        print(f"  true rescue, case by case: both {int((wa & wb).sum())}, only A "
              f"{int((wa & ~wb).sum())}, only B {int((~wa & wb).sum())}, neither "
              f"{int((~wa & ~wb).sum())}")
        dd['case_by_case'] = dict(both=int((wa & wb).sum()), only_A=int((wa & ~wb).sum()),
                                  only_B=int((~wa & wb).sum()), neither=int((~wa & ~wb).sum()))
    diffs[f"{b_.label} - {a_.label}"] = dd

print("\nIntervals cover test-set sampling only, not training-seed variation.")


# ── save ────────────────────────────────────────────────────────────────────
res = dict(system=SYSTEM, split=args.split, n_instants=args.n, n_true=N_TRUE,
           n_scenarios=int(len(uniq)), splits_identical=bool(same), beta=args.beta,
           dp_hold=args.dp_hold, seed=args.seed,
           models={m.label: dict(path=m.path, epoch=m.ck.get('epoch'),
                                 encoder=m.A.get('encoder', 'lstm'),
                                 dp_cell=m.A.get('dp_cell', 'lstm'),
                                 has_lstm=m.has_lstm, params=m.params,
                                 window_ms=m.window_ms,
                                 latency_ms=dict(single=m.lat1, batch=m.latB),
                                 metrics={k: dict(value=m.point[k], ci=ci(m.boot[k]),
                                                  n=m.n_need[k], n_scenarios=m.n_sc[k])
                                          for k in METRICS})
                   for m in models},
           differences=diffs)
json.dump(res, open(os.path.join(args.out, 'compare_results.json'), 'w'), indent=1,
          default=float)
with open(os.path.join(args.out, 'compare_summary.csv'), 'w', newline='') as fh:
    wr = csv.writer(fh)
    wr.writerow(['metric'] + sum([[m.label, f'{m.label} lo', f'{m.label} hi']
                                  for m in models], []))
    for k in METRICS:
        wr.writerow([k] + sum([[m.point[k], *ci(m.boot[k])] for m in models], []))
    for name, f_ in (('params_total', lambda m: m.params['total']),
                     ('params_encoder', lambda m: m.params['encoder']),
                     ('params_dp_head', lambda m: m.params['dp_head']),
                     ('window_ms', lambda m: m.window_ms),
                     ('latency_ms_single', lambda m: m.lat1['decision'])):
        wr.writerow([name] + sum([[f_(m), '', ''] for m in models], []))


# ── figure: small multiples, one metric per panel, one axis each ────────────
if not args.no_plot:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    COL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
    INK, INK2, GRID = '#2c2c2a', '#6b6a64', '#e4e3dd'
    BIAS = '#7b2d8e'          # overlay colour: off the bar palette on purpose
    panels = [('decision_acc', 'Decision accuracy', None),
              ('alpha_rmse', 'Severity (α) RMSE ↓', None),
              ('ool_tpr', 'OOL detection rate (TPR)', None),
              (('rescue_true' if N_TRUE else 'rescue_model'),
               ('Rescue rate TRUE ↓' if N_TRUE else 'model-internal rescue'), None),
              ('dP_norm', '‖ΔP‖ [pu] ↓', None),
              ('window', 'Learned window [ms]', None)]
    fig, axs = plt.subplots(3, 2, figsize=(3.5, 4.5))
    x = np.arange(len(models))
    for ax, (k, title, _) in zip(axs.ravel(), panels):
        if k == 'window':
            vals = np.array([m.window_ms for m in models]); err = None
        else:
            vals = np.array([m.point[k] for m in models])
            cis = np.array([ci(m.boot[k]) for m in models])
            err = np.vstack([vals - cis[:, 0], cis[:, 1] - vals])
        ax.bar(x, vals, width=0.62, color=COL[:len(models)], edgecolor='white',
               linewidth=1.5, zorder=2)
        if err is not None:
            ax.errorbar(x, vals, yerr=err, fmt='none', ecolor=INK, elinewidth=0.68,
                        capsize=3.0, zorder=3)
        top = (vals + (err[1] if err is not None else 0)).max()
        bot = (vals - (err[0] if err is not None else 0)).min()
        # zoom on rates near 1 so differences are visible; say so on the axis
        zoomed = False
        if k in ('decision_acc', 'ool_tpr', 'rescue_true', 'rescue_model') and bot > 0.5:
            zoomed = True
            span = max(top - bot, 0.01)
            ylo = max(0.0, bot - 1.2 * span); ax.set_ylim(ylo, min(1.0, top + 0.8 * span) + 1e-9)
            if ylo > 0:
                ax.text(0.99, 0.03, f'y≥{ylo:.2f}', transform=ax.transAxes,
                        fontsize=5.0, color=INK2, va='bottom', ha='right')
        else:
            ax.set_ylim(0, top * 1.25 if top > 0 else 1.0)
        ytop = vals + (err[1] if err is not None else 0)
        # On a zoomed rate axis, 3 d.p. labels can round two visibly different
        # bars to the same text (e.g. 0.9893 and 0.9898 both print "0.990"),
        # which reads as a plotting error. Label those panels at 4 d.p. so the
        # printed number matches the bar height actually drawn.
        for xi, v, yt in zip(x, vals, ytop):
            lab = (f"{v:.0f}" if k == 'window'
                   else f"{v:.4f}" if zoomed else f"{v:.3f}")
            ax.text(xi, yt, lab, ha='center', va='bottom', fontsize=6.0, color=INK,
                    zorder=4, bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none'))
        ax.set_title(title, fontsize=7.8, color=INK, loc='left')
        ax.set_xticks(x, [m.label for m in models], fontsize=5.8, color=INK,
                  rotation=0)
        ax.tick_params(axis='x', pad=1.2)
        ax.tick_params(axis='y', labelsize=6.2, colors=INK2)
        ax.grid(axis='y', color=GRID, lw=0.6, zorder=0)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'):
            ax.spines[sp].set_color(GRID)

        # Signed alpha bias overlaid on the RMSE panel. Separate right-hand axis
        # because the two quantities have different ranges and, more importantly,
        # different zeros: RMSE is bounded below by 0 and bias is centred on it.
        # The axis is forced symmetric so "above or below the dashed line" reads
        # directly as over- or under-estimation.
        if k == 'alpha_rmse' and 'alpha_bias' in METRICS:
            bv = np.array([m.point['alpha_bias'] for m in models])
            bc = np.array([ci(m.boot['alpha_bias']) for m in models])
            ax2 = ax.twinx()
            lim = max(float(np.nanmax(np.abs(np.r_[bc.ravel(), bv]))) * 2.0, 1e-3)
            ax2.set_ylim(-lim, lim)
            ax2.axhline(0, color=BIAS, lw=0.68, ls=(0, (4, 3)), zorder=1, alpha=0.45)
            for xi, v, (clo, chi) in zip(x, bv, bc):
                nz = np.isfinite(clo) and (clo > 0 or chi < 0)   # CI excludes zero
                ax2.errorbar(xi, v, yerr=[[v - clo], [chi - v]], fmt='D', ms=5.2,
                             mfc=(BIAS if nz else 'white'), mec=BIAS, mew=1.6,
                             ecolor=BIAS, elinewidth=0.79, capsize=3.0, zorder=5)
                ax2.annotate(f"{v:+.3f}", (xi, v), textcoords='offset points',
                             xytext=(6, -2), fontsize=5.4, color=BIAS, zorder=6,
                             bbox=dict(boxstyle='square,pad=0.12', fc='white',
                                       ec='none', alpha=0.85))
            pass  # diamond legend -> paper caption
            ax2.set_ylabel('signed \u03b1 bias', fontsize=6.2, color=BIAS)
            ax2.tick_params(axis='y', labelsize=6.2, colors=BIAS)
            ax2.spines['top'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax2.spines['right'].set_color(BIAS)
    fig.suptitle(f"{SYSTEM}-bus: encoder + ΔP cell ablation\n"
                 f"({args.split}, {args.n} instants, {len(uniq)} scenarios; "
                 f"95% cluster-bootstrap CI)", fontsize=7.2, color=INK,
                 x=0.02, ha='left', va='top', linespacing=1.35)
    fig.tight_layout(rect=(0, 0, 1, 0.90), h_pad=0.9, w_pad=0.9)
    p = os.path.join(args.out, 'compare')
    save_fig(fig, p, dpi=600)
    plt.close(fig)
    print(f"-> {p}")
# ── trajectory figures: one generator, whole run, every architecture ────────
# The decision is placed just after fault onset, as a deployed controller
# would act. Every model decides at the SAME instant from its own measurement
# window; its dP is then applied in the TRUE dynamics and held per --dp_hold.
# 'rescued' and 'disagree' pick cases ON PURPOSE: these figures illustrate,
# the rescue rates in the table are what measure.
def model_dp(m, ks, tk):
    b = m.make_batch(ks, tk)
    with torch.no_grad():
        o = m.model(b['feat'], b['x_meas'], hard=True, ctrl=True, vmask=b['vmask'])
    return o['dP'], b['valid_t']


def pick_cases(ft, n_want, rng_):
    lo_, hi_ = A0['w_max'], float(SYS.t_grid[-1]) - A0['horizon']
    ow = A0.get('onset_window', 0.2) or 0.2
    src = np.asarray(args.plot_scenario) if args.plot_scenario else pool
    cand = src[(SYS.y_type[src] == ft + 1) & (SYS.t_fault[src] >= lo_)
               & (SYS.t_fault[src] < hi_)]
    if len(cand) == 0:
        return []
    if not args.plot_scenario:
        cand = rng_.permutation(cand)[:args.plot_candidates]
    span = np.minimum(ow, SYS.t_clear[cand] - SYS.t_fault[cand]) * 0.999
    tk = np.clip(SYS.t_fault[cand] + rng_.uniform(0, 1, len(cand)) * span, lo_, hi_)
    dPs = []
    for m in models:
        dP, vt = model_dp(m, cand, tk)
        dPs.append(dP)
    pf, _, x_clean = true_sim(cand, tk, vt)
    dev0 = (x_clean[:, NG:2 * NG] - 1).abs().amax(1)
    needed = ((pf > LIMIT) & (dev0 <= LIMIT)).cpu().numpy()
    okw, okr = [], []
    for dP in dPs:
        pc, pcr, _ = true_sim(cand, tk, vt, full_dp(dP))
        okw.append((pc <= LIMIT).cpu().numpy()); okr.append((pcr <= LIMIT).cpu().numpy())
    okw, okr = np.array(okw), np.array(okr)
    all_r, any_r = okr.all(0), okr.any(0)
    all_w = okw.all(0)
    differ = (okr.any(0) & ~okr.all(0)) | (okw.any(0) & ~okw.all(0))
    if args.plot_scenario:
        tiers = [np.ones(len(cand), bool)]
    elif args.plot_pick == 'rescued':
        tiers = [needed & all_r, needed & all_w & ~all_r, needed & any_r & ~all_w,
                 needed & ~any_r & ~all_w, ~needed]
    elif args.plot_pick == 'disagree':
        tiers = [needed & differ, needed & all_r, needed & ~differ & ~all_r, ~needed]
    else:
        tiers = [needed, ~needed]
    seen_, order = set(), []
    for t_ in tiers:
        for j in np.where(t_)[0]:
            if j not in seen_:
                seen_.add(j); order.append(j)
    return [(int(cand[j]), float(tk[j]), bool(needed[j])) for j in order[:n_want]]


if not args.no_plot and args.plots_per_type > 0:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    COL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
    INK, INK2, GRID = '#2c2c2a', '#6b6a64', '#e4e3dd'
    tdir = os.path.join(args.out, 'trajectories'); os.makedirs(tdir, exist_ok=True)
    rng_p = np.random.default_rng(args.seed + 99)
    print(f"\nTrajectory figures ({args.plot_pick} cases, decision just after onset, "
          f"dP held until: {args.dp_hold})")
    traj_log = []
    for ft, tname in enumerate(TYPE_NAMES):
        cases = pick_cases(ft, args.plots_per_type, rng_p)
        if not cases:
            print(f"  {tname}: no usable scenario of this type"); continue
        for k, tkj, needed_j in cases:
            ksa, tka = np.array([k]), np.array([tkj])
            runs = []
            for m in models:
                dP, vt = model_dp(m, ksa, tka)
                pc, pcr, _, tr = true_sim(ksa, tka, vt, full_dp(dP), record=True)
                runs.append(dict(dP=dP[0].cpu().numpy(), tr=tr[0].cpu().numpy(),
                                 pk_win=float(pc[0]), pk_run=float(pcr[0])))
            pf, pfr, _, tr_free = true_sim(ksa, tka, vt, record=True)
            tr_free = tr_free[0].cpu().numpy()
            tt = np.arange(tr_free.shape[0]) * args.dt_true
            after = tt >= tkj
            g = (args.plot_gen - 1 if args.plot_gen > 0
                 else int(np.abs(tr_free[after] - 1).max(0).argmax()))
            g = min(max(g, 0), NG - 1)

            fig, ax = plt.subplots(figsize=(3.5, 2.6))
            ax.axvspan(SYS.t_fault[k], SYS.t_clear[k], color='#f3ead2', zorder=0)
            for sgn in (+1, -1):
                ax.axhline(1 + sgn * LIMIT, color=INK, lw=0.75, ls=':', zorder=1)
            ax.axvline(tkj, color=INK2, lw=0.75, zorder=1)
            ax.plot(tt, tr_free[:, g], color='#2c2c2a', lw=1.05, ls='--', zorder=20)
            handles = [Line2D([], [], color='#2c2c2a', lw=1.05, ls='--',
                              label=f"no ΔP   (peak |ω−1| {float(pfr[0]):.4f})")]
            log = dict(scenario=k, type=tname, t_decision=tkj, generator=g + 1,
                       needed_rescue=needed_j, peak_free_run=float(pfr[0]), models={})
            for mi, (m, r) in enumerate(zip(models, runs)):
                ax.plot(tt, r['tr'][:, g], color=COL[mi], lw=1.5, zorder=3 + mi)
                st = ('inside all run' if r['pk_run'] <= LIMIT else
                      'inside judged window only' if r['pk_win'] <= LIMIT else
                      'leaves envelope')
                dps = ", ".join(f"{v:+.3f}" for v in r['dP'])
                handles.append(Line2D([], [], color=COL[mi], lw=1.5,
                    label=f"{m.label}  peak {r['pk_run']:.4f}  {st}"))
                log['models'][m.label] = dict(dP=[float(v) for v in r['dP']],
                                              peak_run=r['pk_run'],
                                              peak_window=r['pk_win'], status=st)
            handles += [Line2D([], [], color=INK, lw=0.75, ls=':',
                               label=f'envelope ±{LIMIT:g}'),
                        plt.Rectangle((0, 0), 1, 1, color='#f3ead2', label='fault active'),
                        Line2D([], [], color=INK2, lw=0.75, label='decision instant')]
            allv = np.concatenate([tr_free[:, g]] + [r['tr'][:, g] for r in runs])
            pad = max(1.6 * LIMIT, 1.08 * np.abs(allv - 1).max())
            ax.set_ylim(1 - pad, 1 + pad)
            ax.set_xlim(0, float(tt[-1]))
            ax.set_xlabel('t [s]', color=INK); ax.set_ylabel(f'ω of G{g+1} [pu]', color=INK)
            ax.tick_params(colors=INK2, labelsize=5.8)
            ax.grid(color=GRID, lw=0.6, zorder=0)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            for sp in ('left', 'bottom'):
                ax.spines[sp].set_color(GRID)
            ax.legend(handles=handles, fontsize=5.6, loc='upper center',
                      bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False,
                      handlelength=1.6, columnspacing=1.2)
            ax.set_title(
                f"{SYSTEM}-bus, {tname} fault, scenario {k}  (loc {SYS.meta_loc[k]}, "
                f"α {SYS.meta_alpha[k]:.2f}, fault {SYS.t_fault[k]:.2f}–"
                f"{SYS.t_clear[k]:.2f} s)   decision at {tkj:.2f} s "
                f"(+{1000*(tkj-SYS.t_fault[k]):.0f} ms)\n"
                f"G{g+1}" + ("" if args.plot_gen else " (largest uncontrolled swing)")
                + "; true dynamics; ΔP order: "
                + (", ".join(f"G{int(i)+1}" for i in SYS.act_idx.tolist())
                   if SYS.act_idx is not None else ", ".join(f"G{i+1}" for i in range(NG)))
                + ("" if needed_j else "   [this case did not need rescue]"),
                fontsize=6.2, color=INK, loc='left')
            fig.tight_layout()
            p = os.path.join(tdir, f"{tname}_sc{k}_G{g+1}")
            save_fig(fig, p, dpi=600, bbox='tight')
            plt.close(fig)
            log['file'] = p; traj_log.append(log)
            print(f"  {p}   " + "  ".join(
                f"{lab}: {v['status']}" for lab, v in log['models'].items()))
    json.dump(traj_log, open(os.path.join(tdir, 'trajectories.json'), 'w'), indent=1,
              default=float)

print(f"-> {os.path.join(args.out, 'compare_results.json')}")
print(f"-> {os.path.join(args.out, 'compare_summary.csv')}")
