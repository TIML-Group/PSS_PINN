"""
verify_true.py
==============
Independent check: does the emitted dP actually work in the REAL system?

Everything reported so far -- rescue rate, "overall ok", the 26% effort
reduction -- was measured inside the model's own identified dynamics. That
simulation carries an omega RMSE of ~6e-4 against an envelope of 5e-3 and a
safety margin of 1e-4, so a correction that clears the SIMULATED limit by the
margin is a coin flip in reality. Refinement makes this sharper, not softer:
it deliberately shrinks dP until the simulated peak only just clears.

This script trusts none of that. It takes the dP the model emits from the
noisy measurement window, then applies it in the TRUE dynamics:

  * true faulted network, rebuilt from meta_loc / meta_alpha (not the
    model's hypothesis),
  * the verified integrator, which reproduces the dataset trajectories to the
    5e-5 measurement-noise floor,
  * clean initial state taken from a true simulation of the scenario, so the
    measurement noise on delta (0.01 rad) does not contaminate the test.

Judged over the same masked window the model is scored on, so the numbers are
directly comparable: fault active, dP held, peak |omega-1| after the governor
grace period.

    python verify_true.py --ckpt receding_cascade/refined/best.pt --data data_mm

Plots a few random validation cases per fault type to <out>/plots/.
"""

import argparse, json, os

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ieee9_physics import IEEE9, HYPOTHESES
from pinn_model import FaultEncoder, RolloutController

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--data', required=True)
ap.add_argument('--base', default='ieee9_base.mat')
ap.add_argument('--sys', default='ieee9_gov_simulink_data.mat')
ap.add_argument('--out', default=None, help='default: <ckpt dir>/verify')
ap.add_argument('--n', type=int, default=2048, help='validation instants')
ap.add_argument('--batch', type=int, default=256)
ap.add_argument('--dt_true', type=float, default=0.005,
                help='True-dynamics integration step. 5 ms matches the 1 ms '
                     'reference at the noise floor.')
ap.add_argument('--dp_hold', default='clear', choices=['clear', 'window', 'run'],
                help="How long dP stays applied. 'clear' (default): until the "
                     "fault clears -- the y_delta_p convention, and what a "
                     "receding-horizon controller does by re-deciding every "
                     "~20 ms. 'window': until the model's judged window ends, "
                     "which on faults longer than --horizon drops the "
                     "correction MID-FAULT at a hyperparameter boundary. "
                     "'run': never withdrawn (a surplus after clearing).")
ap.add_argument('--plots_per_type', type=int, default=3)
ap.add_argument('--split', default='val', choices=['val', 'test', 'train'],
                help='Split to verify and plot from (default: validation).')
ap.add_argument('--seed', type=int, default=11)
args = ap.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ck = torch.load(args.ckpt, map_location=device, weights_only=False)
A = ck['args']
out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(args.ckpt)),
                                   'verify')
os.makedirs(os.path.join(out_dir, 'plots'), exist_ok=True)
print(f"Device: {device}")
print(f"Loaded {args.ckpt} (epoch {ck['epoch']}, dP head "
      f"'{A.get('dp_head','linear')}')")

# ── data ─────────────────────────────────────────────────────────────────────
X = np.load(os.path.join(args.data, 'X.npy'), mmap_mode='r')
md = np.load(os.path.join(args.data, 'meta.npz'))
DT = float(md['dt_solver']); t_grid = md['t']; lines = md['lines'].astype(int)
y_type = md['y_type'].ravel().astype(int)
meta_loc = md['meta_loc'].ravel().astype(int)
meta_alpha = md['meta_alpha'].ravel()
t_fault = md['y_t_fault'].ravel(); t_clear = md['meta_t_clear'].ravel()
OMEGA_LIMIT = float(md['omega_limit'])
idx_va = np.asarray(ck['split'][args.split])
print(f"Split '{args.split}': {len(idx_va)} scenarios  "
      f"(gen {int((y_type[idx_va]==1).sum())}, load {int((y_type[idx_va]==2).sum())}, "
      f"line {int((y_type[idx_va]==3).sum())})")

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

S = IEEE9(args.base, args.sys, dtype=torch.float32, device=device, lines=lines)
enc = FaultEncoder(ck['n_feat'], w_max_steps=W_STEPS, dt_window=A['dt_window'],
                   lstm_layers=A.get('lstm_layers', 10),
                   init_window=A['init_window'] / A['w_max'],
                   max_window=A['w_max'], beta0=A['beta0'],
                   encoder=A.get('encoder', 'lstm'),
                   mlp_proj=A.get('mlp_proj'))
model = RolloutController(S, enc, horizon=A['horizon'], dt_roll=A['dt_roll'],
                          omega_limit=OMEGA_LIMIT, margin=A.get('margin', 1e-4),
                          dp_head=A.get('dp_head', 'linear'),
                          dp_hidden=A.get('dp_hidden', 128),
                          dp_fc_layers=A.get('dp_fc_layers', 2),
                          n_refine=A.get('n_refine', 3),
                          dp_cell=A.get('dp_cell', 'lstm')).to(device)
model.load_state_dict(ck['model']); model.eval()
# beta is restored by load_state_dict above. Forcing A['beta_min'] here would
# evaluate the model with a sharper mask edge than it was trained and validated
# with: beta anneals over the first 60% of the planned epochs, so a checkpoint
# selected before that point holds a softer edge. case30_common.py keeps the
# stored value for exactly this reason.
GRACE_STEPS = int(round(model.grace / args.dt_true))


def make_batch(ks, rng, tk_fixed=None):
    ks = np.asarray(ks); bs = len(ks)
    lo, hi = A['w_max'], t_grid[-1] - A['horizon']
    tk = rng.uniform(lo, hi, bs)
    if tk_fixed is not None:
        tk = np.asarray(tk_fixed, dtype=float).copy()
    elif A.get('p_active', 0) > 0:
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
    x_meas = np.concatenate([np.stack([np.zeros(bs), dd[:, 1]-dd[:, 0],
                                       dd[:, 2]-dd[:, 0]], 1),
                             re[:, W_IDX], re[:, PM_IDX]], 1)
    active = (tk >= t_fault[ks]) & (tk < t_clear[ks])
    t_change = np.where(active, t_clear[ks],
                        np.where(tk < t_fault[ks], t_fault[ks], np.inf))
    valid_t = np.clip(t_change - tk, 0.0, A['horizon'])
    step_t = np.arange(N_ROLL + 1) * A['dt_roll']
    vmask = (step_t[None, :] <= valid_t[:, None]).astype(np.float32)
    return (torch.tensor(feat), torch.tensor(x_meas, dtype=torch.float32),
            torch.tensor(fut), torch.tensor(vmask), ks, tk, valid_t)


# ── true dynamics ────────────────────────────────────────────────────────────
def true_network(ks):
    """Exact faulted network and set-point offset from the METADATA."""
    a = torch.tensor(meta_alpha[ks], dtype=torch.float32, device=device)
    n = len(ks)
    Yf = torch.zeros(n, 3, 3, dtype=S.cdtype, device=device)
    dPm = torch.zeros(n, 3, dtype=torch.float32, device=device)
    for j, k in enumerate(ks):
        h = HYPOTHESES.index((TYPE_NAMES[y_type[k]-1], int(meta_loc[k])))
        Yf[j], _ = S.kron_reduce(S.faulted_Y(h, a[j:j+1]))
        dPm[j] = S.dPm0(h, a[j:j+1])[0]
    return Yf, dPm


@torch.no_grad()
def clean_state(ks, tk):
    """
    Clean state at each decision instant, from a true simulation of the whole
    scenario. Using the measured state instead would inject the 0.01 rad delta
    noise into the initial condition of the verification.
    """
    n = len(ks)
    Yf, dPm = true_network(ks)
    Ypre = S.Yred_pre.expand(n, 3, 3)
    Mp, Np = S.pe_consts(Ypre); Mf, Nf = S.pe_consts(Yf)
    tf = torch.tensor(t_fault[ks], dtype=torch.float32, device=device)
    tc = torch.tensor(t_clear[ks], dtype=torch.float32, device=device)
    x = torch.cat([S.delta0, torch.ones(3, device=device), S.Pm0]) \
        .unsqueeze(0).expand(n, 9).clone()
    h = args.dt_true
    n_steps = int(round(float(t_grid[-1]) / h))
    want = np.round(tk / h).astype(int)
    out = torch.zeros(n, 9, device=device)
    traj = torch.zeros(n, n_steps + 1, 3, device=device)   # omega only
    traj[:, 0] = x[:, 3:6]
    for i in range(n_steps):
        tm = i * h + h / 2
        act = (tm >= tf) & (tm < tc)
        M = torch.where(act.view(n, 1, 1), Mf, Mp)
        N = torch.where(act.view(n, 1, 1), Nf, Np)
        Ps = S.Pm0.unsqueeze(0) + dPm * act.view(n, 1).float()
        x = S.rk4_real(x, M, N, Ps, h)
        traj[:, i + 1] = x[:, 3:6]
        hit = torch.tensor(want == (i + 1), device=device)
        if hit.any():
            out[hit] = x[hit]
    return out, traj, Yf, dPm


@torch.no_grad()
def full_sim(ks, tk_arr, dP=None, hold='clear', vt=None):
    """
    Whole 8 s of the scenario in TRUE dynamics, from the pre-fault equilibrium.
    Network: pre-fault -> faulted on [t_fault, t_clear) -> pre-fault again.

    dP is applied from the decision instant until `hold` says to remove it:
      'clear'  -> at t_clear (the y_delta_p convention; a receding-horizon
                  controller keeps re-emitting while the fault persists)
      'window' -> at tk + valid_t, i.e. the judged-window edge, which is capped
                  by --horizon and so cuts the correction mid-fault whenever
                  the fault outlasts the horizon
      'run'    -> never
    """
    n = len(ks)
    Yf, dPm = true_network(ks)
    Ypre = S.Yred_pre.expand(n, 3, 3)
    Mp, Np = S.pe_consts(Ypre); Mf, Nf = S.pe_consts(Yf)
    tf = torch.tensor(t_fault[ks], dtype=torch.float32, device=device)
    tc = torch.tensor(t_clear[ks], dtype=torch.float32, device=device)
    tk = torch.tensor(np.asarray(tk_arr), dtype=torch.float32, device=device)
    if hold == 'clear':
        t_off = tc
    elif hold == 'run':
        t_off = torch.full_like(tk, float(t_grid[-1]) + 1.0)
    else:
        t_off = tk + torch.tensor(np.asarray(vt), dtype=torch.float32,
                                  device=device)
    x = torch.cat([S.delta0, torch.ones(3, device=device), S.Pm0]) \
        .unsqueeze(0).expand(n, 9).clone()
    h = args.dt_true
    n_steps = int(round(float(t_grid[-1]) / h))
    traj = torch.zeros(n, n_steps + 1, 3, device=device)
    states = torch.zeros(n, n_steps + 1, 9, device=device)
    traj[:, 0] = x[:, 3:6]; states[:, 0] = x
    for i in range(n_steps):
        tm = i * h + h / 2
        act = (tm >= tf) & (tm < tc)
        M = torch.where(act.view(n, 1, 1), Mf, Mp)
        N = torch.where(act.view(n, 1, 1), Nf, Np)
        Ps = S.Pm0.unsqueeze(0) + dPm * act.view(n, 1).float()
        if dP is not None:
            on = (tm >= tk) & (tm < t_off)
            Ps = Ps + dP * on.view(n, 1).float()
        x = S.rk4_real(x, M, N, Ps, h)
        traj[:, i + 1] = x[:, 3:6]; states[:, i + 1] = x
    return traj, states


@torch.no_grad()
def peaks_from(traj, tk, valid_t):
    """Peak |omega-1| after the grace period, over (a) the model's judged
    window and (b) the rest of the run."""
    h = args.dt_true
    T = traj.shape[1]
    ti = torch.arange(T, device=device).unsqueeze(0).float() * h
    tk_t = torch.as_tensor(tk, dtype=torch.float32, device=device).unsqueeze(1)
    vt_t = torch.as_tensor(valid_t, dtype=torch.float32, device=device).unsqueeze(1)
    d = (traj - 1.0).abs().amax(dim=2)
    m_run = ti >= (tk_t + model.grace)
    m_win = m_run & (ti <= tk_t + vt_t)
    return (torch.where(m_win, d, torch.full_like(d, -1.0)).amax(1),
            torch.where(m_run, d, torch.full_like(d, -1.0)).amax(1))


@torch.no_grad()
def true_rollout(x0, Yf, dPm, dP, valid_t, active, record=False):
    """
    Integrate the TRUE dynamics forward from the decision instant with dP held,
    over the valid window. The network is constant across that window by
    construction (the mask is cut at the next change), but WHICH network
    depends on the instant: faulted only if a fault is active at tk, otherwise
    pre-fault. Simulating a post-clearing instant with the fault still applied
    would invent an excursion that never happens.
    """
    n = x0.shape[0]
    act = torch.as_tensor(active, device=device).view(n, 1, 1).float()
    Ypre = S.Yred_pre.expand(n, 3, 3)
    Mf, Nf = S.pe_consts(Yf); Mp, Np = S.pe_consts(Ypre)
    M = act * Mf + (1 - act) * Mp
    N = act * Nf + (1 - act) * Np
    Ps = (S.Pm0.unsqueeze(0) + dPm * act.view(n, 1)
          + (dP if dP is not None else 0.0))
    h = args.dt_true
    n_steps = int(round(A['horizon'] / h))
    nv = torch.tensor(np.floor(valid_t / h).astype(int), device=device)
    x = x0.clone()
    peak = torch.full((n,), -1.0, device=device)
    tr = torch.zeros(n, n_steps + 1, 3, device=device) if record else None
    if record:
        tr[:, 0] = x[:, 3:6]
    for i in range(n_steps):
        x = S.rk4_real(x, M, N, Ps, h)
        if record:
            tr[:, i + 1] = x[:, 3:6]
        live = (i + 1 <= nv) & (i + 1 >= GRACE_STEPS)
        if live.any():
            d = (x[:, 3:6] - 1.0).abs().amax(dim=1)
            peak = torch.where(live, torch.maximum(peak, d), peak)
    return peak, tr


# ── run ──────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(args.seed)
tot = dict(n=0, need=0, resc_true=0, resc_model=0, ok_true=0, ok_n=0,
           agree=0, norm=0.0, resc_run=0, free_run=0, excluded=0)
age_hit_t = [0]*len(AGE_BINS); age_hit_m = [0]*len(AGE_BINS); age_n = [0]*len(AGE_BINS)
plot_pool = {0: [], 1: [], 2: []}

n_done = 0
while n_done < args.n:
    bs = min(args.batch, args.n - n_done)
    ks = rng.choice(idx_va, bs)
    feat, x_meas, fut, vmask, ks, tk, valid_t = make_batch(ks, rng)
    feat, x_meas = feat.to(device), x_meas.to(device)
    fut, vmask = fut.to(device), vmask.to(device)

    with torch.no_grad():
        o = model(feat, x_meas, hard=True, ctrl=True, vmask=vmask)
    dP = o['dP']

    tr_free, st_free = full_sim(ks, tk, None)
    tr_ctrl, _ = full_sim(ks, tk, dP, hold=args.dp_hold, vt=valid_t)
    pk_free_t, pk_free_run = peaks_from(tr_free, tk, valid_t)
    pk_ctrl_t, pk_ctrl_run = peaks_from(tr_ctrl, tk, valid_t)
    pk_ctrl_m, has_m = model.judged(o['w_ctrl'], vmask)

    i_tk = torch.as_tensor(np.round(tk / args.dt_true).astype(int), device=device)
    x0 = st_free[torch.arange(len(ks), device=device), i_tk]
    dev_now = (x0[:, 3:6] - 1.0).abs().amax(dim=1)
    # ACTIONABLE: a fault must be active at the decision instant. With the
    # fault already cleared there is nothing for a set-point change to correct,
    # and dP is applied for zero duration (t_off = t_clear < t_k), so the case
    # is a guaranteed failure that the controller could never have won. Line
    # faults are hit hardest -- their excursion often peaks AFTER reconnection,
    # so 27% of their needy instants land post-clearing versus ~5% for load.
    act_now = torch.tensor((tk >= t_fault[ks]) & (tk < t_clear[ks]),
                           device=device)
    reach = (pk_free_t >= 0) & (dev_now <= model.limit) & act_now
    need = (pk_free_t > model.limit) & reach
    n_excl = int(((pk_free_t > model.limit) & (pk_free_t >= 0)
                  & (dev_now <= model.limit) & ~act_now).sum())
    tot['excluded'] = tot.get('excluded', 0) + n_excl
    win_t = (pk_ctrl_t <= model.limit)
    win_m = (pk_ctrl_m <= model.limit)
    win_run = (pk_ctrl_run <= model.limit)
    tot['resc_run'] = tot.get('resc_run', 0) + int(win_run[need].sum())
    tot['free_run'] = tot.get('free_run', 0) + int(
        (pk_free_run[need] <= model.limit).sum())

    tot['n'] += bs; tot['need'] += int(need.sum())
    tot['resc_true'] += int(win_t[need].sum())
    tot['resc_model'] += int(win_m[need].sum())
    tot['ok_true'] += int(win_t[reach].sum()); tot['ok_n'] += int(reach.sum())
    tot['agree'] += int((win_t[need] == win_m[need]).sum())
    tot['norm'] += float(dP.pow(2).sum(1).sqrt().sum())

    ag = tk - t_fault[ks]; nd = need.cpu().numpy()
    wt = win_t.cpu().numpy(); wm = win_m.cpu().numpy()
    for i, (b0, b1) in enumerate(AGE_BINS):
        m = nd & (ag >= b0) & (ag < b1)
        if m.sum():
            age_hit_t[i] += int(wt[m].sum()); age_hit_m[i] += int(wm[m].sum())
            age_n[i] += int(m.sum())

    n_done += bs
    print(f"\r  {n_done}/{args.n}", end='', flush=True)
print()

R = lambda a, b: (a / b if b else float('nan'))
print(f"\n{'='*70}")
print(f"TRUE-DYNAMICS VERIFICATION  (validation split, {tot['n']} instants)")
print(f"{'='*70}")
print(f"  cases needing rescue (true dynamics) : {tot['need']}")
print(f"  excluded, no fault active at decision: {tot['excluded']}  "
      f"({100*tot['excluded']/max(tot['need']+tot['excluded'],1):.1f}% of "
      f"otherwise-needy cases -- nothing for dP to act on)")
print(f"  rescue rate, TRUE dynamics           : {R(tot['resc_true'], tot['need']):.4f}")
print(f"  rescue rate, model's own rollout     : {R(tot['resc_model'], tot['need']):.4f}")
print(f"  the two agree on                     : {R(tot['agree'], tot['need'])*100:.1f}% of cases")
print(f"  overall ok (true, reachable)         : {R(tot['ok_true'], tot['ok_n']):.4f}")
print(f"\n  judged to END OF RUN (through clearing and ring-down):")
print(f"    with dP                            : {R(tot['resc_run'], tot['need']):.4f}")
print(f"    without dP                         : {R(tot['free_run'], tot['need']):.4f}")
print(f"  dP held until: {args.dp_hold}")
print(f"  mean ||dP||                          : {tot['norm']/tot['n']:.4f} pu")
print(f"\n  rescue by fault age   " + "  ".join(
    f"{lo*1000:.0f}-{(hi*1000 if hi<1e8 else 9999):.0f}ms:"
    f"{R(h,n):.2f}/{R(m,n):.2f}(n={n})"
    for (lo, hi), h, m, n in zip(AGE_BINS, age_hit_t, age_hit_m, age_n)))
print(f"  (true / model-internal)")

gap = R(tot['resc_model'], tot['need']) - R(tot['resc_true'], tot['need'])
print(f"\n  model-internal minus true = {gap*+100:+.1f} points")
if abs(gap) < 0.02:
    print("  -> the identified dynamics are trustworthy for control; the "
          "reported\n     rescue rate survives contact with the real system.")
else:
    print("  -> the model-internal number is OPTIMISTIC. Raise --margin toward "
          "the\n     trajectory RMSE and re-run refinement against a realistic "
          "buffer.")

json.dump({'n_instants': tot['n'], 'n_need': tot['need'],
           'rescue_true': R(tot['resc_true'], tot['need']),
           'rescue_model': R(tot['resc_model'], tot['need']),
           'agreement': R(tot['agree'], tot['need']),
           'ok_true': R(tot['ok_true'], tot['ok_n']),
           'mean_dP': tot['norm'] / tot['n'], 'dp_hold': args.dp_hold,
           'excluded_no_active_fault': tot['excluded'],
           'rescue_true_full_run': R(tot['resc_run'], tot['need']),
           'no_dp_full_run': R(tot['free_run'], tot['need']),
           'by_age': [{'lo': lo, 'hi': hi, 'n': n, 'true': R(h, n),
                       'model': R(m, n)}
                      for (lo, hi), h, m, n in
                      zip(AGE_BINS, age_hit_t, age_hit_m, age_n)]},
          open(os.path.join(out_dir, 'verify_results.json'), 'w'), indent=1)

# ── plots ────────────────────────────────────────────────────────────────────
# ── pick plot cases PER FAULT TYPE ──────────────────────────────────────────
# Plot cases used to be taken only from instants that needed rescue AND were
# actionable. Line faults almost never qualify: their excursions mostly peak
# after reconnection, which the active-fault gate excludes -- so the line slot
# was always empty. Now each type is drawn deliberately: decision placed just
# after onset (as a deployed controller would act), cases that genuinely need
# rescue preferred, and otherwise the best available case of that type shown,
# marked as such in the title.
def select_plot_cases(ft, n_want, rng):
    lo, hi = A['w_max'], float(t_grid[-1]) - A['horizon']
    ow = A.get('onset_window', 0.2) or 0.2
    cand = idx_va[(y_type[idx_va] == ft + 1) & (t_fault[idx_va] >= lo)
                  & (t_fault[idx_va] < hi)]
    if len(cand) == 0:
        return []
    cand = rng.permutation(cand)[:48]
    span = np.minimum(ow, t_clear[cand] - t_fault[cand]) * 0.999
    tkc = np.clip(t_fault[cand] + rng.uniform(0, 1, len(cand)) * span, lo, hi)
    feat, x_meas, fut, vmask, ks_, tk_, vt_ = make_batch(cand, rng, tk_fixed=tkc)
    with torch.no_grad():
        o_ = model(feat.to(device), x_meas.to(device), hard=True, ctrl=True,
                   vmask=vmask.to(device))
    dP_ = o_['dP']
    trf, stf = full_sim(ks_, tk_, None)
    trc, _ = full_sim(ks_, tk_, dP_, hold=args.dp_hold, vt=vt_)
    pf, _ = peaks_from(trf, tk_, vt_)
    pc, _ = peaks_from(trc, tk_, vt_)
    pm, _ = model.judged(o_['w_ctrl'], vmask.to(device))
    i_tk = torch.as_tensor(np.round(tk_ / args.dt_true).astype(int), device=device)
    dev0 = (stf[torch.arange(len(ks_), device=device), i_tk][:, 3:6] - 1).abs().amax(1)
    needed = ((pf > model.limit) & (dev0 <= model.limit)).cpu().numpy()
    order = list(np.where(needed)[0]) + list(np.where(~needed)[0])
    out = []
    for j in order[:n_want]:
        out.append((int(ks_[j]), float(tk_[j]), float(vt_[j]), dP_[j].cpu().numpy(),
                    float(pf[j]), float(pc[j]), float(pm[j]), bool(needed[j])))
    return out


rng_p = np.random.default_rng(args.seed + 99)
plot_pool = {}
for ft in range(3):
    plot_pool[ft] = select_plot_cases(ft, args.plots_per_type, rng_p)
    nn_ = sum(c[7] for c in plot_pool[ft])
    if not plot_pool[ft]:
        print(f"  {TYPE_NAMES[ft]}: no scenarios of this type in split "
              f"'{args.split}' with a usable fault time -- nothing to plot")
    else:
        print(f"  {TYPE_NAMES[ft]}: {len(plot_pool[ft])} case(s), "
              f"{nn_} needing rescue")

print(f"\nplotting {sum(len(v) for v in plot_pool.values())} cases ...")
h = args.dt_true
from matplotlib.lines import Line2D
GEN_COLORS = ['#1f77b4', '#d62728', '#2ca02c']
for ft, cases in plot_pool.items():
    for ci, (k, tkj, vt, dPj, pf, pc, pm_, needed_) in enumerate(cases):
        ksa = np.array([k]); tka = np.array([tkj])
        dPt = torch.tensor(dPj, dtype=torch.float32, device=device).unsqueeze(0)
        vta = np.array([vt])
        tr_free = full_sim(ksa, tka, None)[0][0].cpu().numpy()
        tr_ctrl = full_sim(ksa, tka, dPt, hold=args.dp_hold, vt=vta)[0][0].cpu().numpy()
        tt = np.arange(tr_free.shape[0]) * h

        fig, ax = plt.subplots(figsize=(13, 5.5))
        # only the two things being compared: without dP (dashed), with dP (solid)
        for g, c in enumerate(GEN_COLORS):
            ax.plot(tt, tr_free[:, g], color=c, lw=1.5, ls='--')
            ax.plot(tt, tr_ctrl[:, g], color=c, lw=2.0)
        # minimal context: envelope, fault interval, decision instant
        for sgn in (+1, -1):
            ax.axhline(1 + sgn * OMEGA_LIMIT, color='k', lw=1.0, ls=':')
        ax.axvspan(t_fault[k], t_clear[k], color='#ffe9b0', alpha=0.55, zorder=0)
        ax.axvline(tkj, color='0.25', lw=1.2)
        # grey decision window: decision -> decision + governor grace period.
        # Peaks inside it are not scored (the governor needs time to respond).
        ax.axvspan(tkj, tkj + model.grace, color='0.82', zorder=1)

        handles = [Line2D([], [], color='0.2', lw=1.5, ls='--', label='without dP'),
                   Line2D([], [], color='0.2', lw=2.0, label='with dP')]
        handles += [Line2D([], [], color=c, lw=2.0, label=f'G{g+1}')
                    for g, c in enumerate(GEN_COLORS)]
        handles += [Line2D([], [], color='k', lw=1.0, ls=':', label='envelope'),
                    plt.Rectangle((0, 0), 1, 1, color='#ffe9b0', alpha=0.55,
                                  label='fault active'),
                    Line2D([], [], color='0.25', lw=1.2, label='decision'),
                    plt.Rectangle((0, 0), 1, 1, color='0.82',
                                  label=f'decision window ({1000*model.grace:.0f} ms grace)')]
        ax.legend(handles=handles, fontsize=8, loc='lower right', ncol=2)

        ax.set_xlim(0, float(t_grid[-1]))
        ax.set_xlabel('t [s]')
        ax.set_ylabel(r'$\omega$ [pu]'); ax.grid(alpha=0.3)
        ax.set_title(
            f"{TYPE_NAMES[ft]} fault, scenario {k}  "
            f"[{'needed rescue' if needed_ else 'no rescue needed in judged window'}]"
            f"  |  loc={meta_loc[k]} alpha={meta_alpha[k]:.2f}  "
            f"fault {t_fault[k]:.2f}-{t_clear[k]:.2f}s  "
            f"decision t={tkj:.2f}s (+{1000*(tkj-t_fault[k]):.0f} ms)\n"
            f"dP = [{dPj[0]:+.3f} {dPj[1]:+.3f} {dPj[2]:+.3f}] pu   |   "
            f"peak |w-1|: without dP {pf:.5f} -> with dP {pc:.5f}  "
            f"(limit {OMEGA_LIMIT})",
            fontsize=9)
        fig.tight_layout()
        p_ = os.path.join(out_dir, 'plots',
                          f"{TYPE_NAMES[ft]}_sc{k}_t{tkj:.2f}.png")
        fig.savefig(p_, dpi=130); plt.close(fig)
        print(f"  {p_}")

print(f"\n-> {out_dir}/verify_results.json")
