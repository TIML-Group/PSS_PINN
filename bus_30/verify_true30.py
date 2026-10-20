"""
verify_true30.py
================
30-bus port of verify_true.py. Does the emitted dP actually work in the REAL
system?

Every rescue rate reported by training, refinement and the diagnostic is
measured inside the model's own identified dynamics, which carry an omega
RMSE of ~9.5e-4 (30-bus test set) against an envelope of 5e-3 and a margin of
1e-3. Refinement makes this sharper: it shrinks dP until the SIMULATED peak
only just clears. This script trusts none of that. It takes the dP the model
emits from the noisy measurement window and applies it in the TRUE dynamics:

  * true faulted network, rebuilt from meta_loc / meta_alpha (not the model's
    hypothesis), with exact branch removal for line faults,
  * the verified integrator (reproduces the dataset to the 5e-5 noise floor),
  * a clean full-run simulation from the pre-fault equilibrium, so the 0.01 rad
    measurement noise on delta does not contaminate the test.

Judged over the same masked window the model is scored on. The actionability
gate is case30_common.actionable -- the same function refinement uses.

Added for the 30-bus system:
  * ||dP|| split into cases needing rescue and cases that don't,
  * HARMED: instants that were inside the envelope WITHOUT dP and are pushed
    out BY it. Rescue rate cannot see these (it only scores cases that needed
    help), and they are the failure a false alarm causes.

    python verify_true30.py --ckpt case30_ctrl/refined/best.pt --data data30_mm

Plots one figure per case, per fault type, to <out>/plots/.
"""

import argparse, json, os

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from case30_common import (Setup, actionable, AGE_BINS, TYPE_NAMES, fmt_age,
                           NG, N_STATE)

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--data', required=True)
ap.add_argument('--base', default='case30_base.mat')
ap.add_argument('--sys', default='case30_simulink_data.mat')
ap.add_argument('--out', default=None, help='default: <ckpt dir>/verify')
ap.add_argument('--n', type=int, default=2048, help='decision instants')
ap.add_argument('--batch', type=int, default=256)
ap.add_argument('--dt_true', type=float, default=0.005,
                help='True-dynamics step. 5 ms matches the 1 ms reference at '
                     'the noise floor.')
ap.add_argument('--dp_hold', default='clear', choices=['clear', 'window', 'run'],
                help="How long dP stays applied: 'clear' (default, the "
                     "y_delta_p convention) until the fault clears; 'window' "
                     "until the judged window ends; 'run' never withdrawn.")
ap.add_argument('--plots_per_type', type=int, default=3)
ap.add_argument('--split', default='val', choices=['val', 'test', 'train'])
ap.add_argument('--seed', type=int, default=11)
args = ap.parse_args()

su = Setup(args.ckpt, args.data, args.base, args.sys)
device, model, A, S = su.device, su.model, su.A, su.S
model.eval()
t_fault, t_clear, t_grid = su.t_fault, su.t_clear, su.t_grid
y_type, meta_loc, meta_alpha = su.y_type, su.meta_loc, su.meta_alpha
OMEGA_LIMIT = su.OMEGA_LIMIT
out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(args.ckpt)),
                                   'verify')
os.makedirs(os.path.join(out_dir, 'plots'), exist_ok=True)
idx = {'train': su.idx_tr, 'val': su.idx_va, 'test': su.idx_te}[args.split]
print(f"Split '{args.split}': {len(idx)} scenarios  "
      f"(gen {int((y_type[idx]==1).sum())}, load {int((y_type[idx]==2).sum())}, "
      f"line {int((y_type[idx]==3).sum())})")


# ── true dynamics ────────────────────────────────────────────────────────────
@torch.no_grad()
def full_sim(ks, tk_arr, dP=None, hold='clear', vt=None):
    """
    Whole run in TRUE dynamics from the pre-fault equilibrium:
    pre-fault -> faulted on [t_fault, t_clear) -> pre-fault again.
    dP (B, 2) is applied from the decision instant until `hold` removes it.
    Returns omega (B, T, 6) and the full state (B, T, 18).
    """
    n = len(ks)
    Yf, dPm = su.true_network(ks)
    Mp, Np = S.pe_consts(S.Yred_pre.expand(n, NG, NG))
    Mf, Nf = S.pe_consts(Yf)
    tf = torch.tensor(t_fault[ks], dtype=torch.float32, device=device)
    tc = torch.tensor(t_clear[ks], dtype=torch.float32, device=device)
    tk = torch.tensor(np.asarray(tk_arr), dtype=torch.float32, device=device)
    if hold == 'clear':
        t_off = tc
    elif hold == 'run':
        t_off = torch.full_like(tk, float(t_grid[-1]) + 1.0)
    else:
        t_off = tk + torch.tensor(np.asarray(vt), dtype=torch.float32, device=device)
    dP6 = model.scatter_dp(dP) if dP is not None else None
    x = S.x0.to(torch.float32).unsqueeze(0).expand(n, N_STATE).clone()
    h = args.dt_true
    n_steps = int(round(float(t_grid[-1]) / h))
    traj = torch.zeros(n, n_steps + 1, NG, device=device)
    states = torch.zeros(n, n_steps + 1, N_STATE, device=device)
    traj[:, 0] = x[:, NG:2 * NG]; states[:, 0] = x
    for i in range(n_steps):
        tm = i * h + h / 2
        act = (tm >= tf) & (tm < tc)
        M = torch.where(act.view(n, 1, 1), Mf, Mp)
        N = torch.where(act.view(n, 1, 1), Nf, Np)
        Ps = S.Pm0.unsqueeze(0) + dPm * act.view(n, 1).float()
        if dP6 is not None:
            on = (tm >= tk) & (tm < t_off)
            Ps = Ps + dP6 * on.view(n, 1).float()
        x = S.rk4_real(x, M, N, Ps, h)
        traj[:, i + 1] = x[:, NG:2 * NG]; states[:, i + 1] = x
    return traj, states


@torch.no_grad()
def peaks_from(traj, tk, valid_t):
    """Peak |omega-1| after the grace period over (a) the judged window and
    (b) the rest of the run. -1 where the window is empty."""
    T = traj.shape[1]
    ti = torch.arange(T, device=device).unsqueeze(0).float() * args.dt_true
    tk_t = torch.as_tensor(tk, dtype=torch.float32, device=device).unsqueeze(1)
    vt_t = torch.as_tensor(valid_t, dtype=torch.float32, device=device).unsqueeze(1)
    d = (traj - 1.0).abs().amax(dim=2)
    m_run = ti >= (tk_t + model.grace)
    m_win = m_run & (ti <= tk_t + vt_t)
    neg = torch.full_like(d, -1.0)
    return torch.where(m_win, d, neg).amax(1), torch.where(m_run, d, neg).amax(1)


# ── run ──────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(args.seed)
tot = dict(n=0, need=0, resc_true=0, resc_model=0, ok_true=0, ok_n=0, agree=0,
           resc_run=0, free_run=0, excluded=0, healthy=0, harmed=0)
norm_all, norm_need, norm_quiet = [], [], []
age_t = [0] * len(AGE_BINS); age_m = [0] * len(AGE_BINS); age_n = [0] * len(AGE_BINS)
by_type = {t: [0, 0] for t in TYPE_NAMES}
# Q13: per-case record for the needy cases of every type, so the misses can be
# cross-tabulated by true severity and by how much actuator authority was left
# unused. Columns: [type, alpha, rescued, ||dP||, max|dP|/dp_max].
case_rows = []
# Q5: calibration of p_fail, the bounded surrogate the control loss minimises.
# Its mean is only the expected failure rate UNDER THE MODEL'S OWN ROLLOUT; this
# records it against the TRUE-dynamics outcome as well, so the gap between the
# two is measured rather than assumed.
# Columns: [p_fail, fail_true, fail_model_tgt, fail_model_limit, needy].
calib_rows = []

n_done = 0
while n_done < args.n:
    bs = min(args.batch, args.n - n_done)
    b = su.to_dev(su.make_batch(rng.choice(idx, bs), rng))
    ks, tk, valid_t = b['ks'], b['tk'], b['valid_t']
    with torch.no_grad():
        o = model(b['feat'], b['x_meas'], hard=True, ctrl=True, vmask=b['vmask'])
    dP = o['dP']

    tr_free, st_free = full_sim(ks, tk, None)
    tr_ctrl, _ = full_sim(ks, tk, dP, hold=args.dp_hold, vt=valid_t)
    pk_free_t, pk_free_run = peaks_from(tr_free, tk, valid_t)
    pk_ctrl_t, pk_ctrl_run = peaks_from(tr_ctrl, tk, valid_t)
    pk_ctrl_m, _ = model.judged(o['w_ctrl'], b['vmask'])

    # the gate, on the CLEAN state at the decision instant
    i_tk = torch.as_tensor(np.round(tk / args.dt_true).astype(int), device=device)
    x_clean = st_free[torch.arange(bs, device=device), i_tk]
    reach = actionable(pk_free_t >= 0, x_clean, b['active'], model.limit)
    need = (pk_free_t > model.limit) & reach
    reach_ina = actionable(pk_free_t >= 0, x_clean,
                           torch.ones_like(b['active']), model.limit)
    tot['excluded'] += int(((pk_free_t > model.limit) & reach_ina & ~b['active']).sum())
    win_t = pk_ctrl_t <= model.limit
    win_m = pk_ctrl_m <= model.limit

    # harm: healthy without dP (judged window inside the envelope) but pushed
    # out by the correction -- over EVERY instant, fault or not
    healthy = (pk_free_t >= 0) & (pk_free_t <= model.limit)
    tot['healthy'] += int(healthy.sum())
    tot['harmed'] += int((healthy & (pk_ctrl_t > model.limit)).sum())

    tot['n'] += bs; tot['need'] += int(need.sum())
    tot['resc_true'] += int(win_t[need].sum())
    tot['resc_model'] += int(win_m[need].sum())
    tot['ok_true'] += int(win_t[reach].sum()); tot['ok_n'] += int(reach.sum())
    tot['agree'] += int((win_t[need] == win_m[need]).sum())
    tot['resc_run'] += int((pk_ctrl_run[need] <= model.limit).sum())
    tot['free_run'] += int((pk_free_run[need] <= model.limit).sum())
    nrm = dP.pow(2).sum(1).sqrt()
    norm_all.append(nrm.cpu()); norm_need.append(nrm[need].cpu())
    norm_quiet.append(nrm[~need].cpu())

    ag = tk - t_fault[ks]; nd = need.cpu().numpy()
    wt = win_t.cpu().numpy(); wm = win_m.cpu().numpy()
    for i, (b0, b1) in enumerate(AGE_BINS):
        m = nd & (ag >= b0) & (ag < b1)
        if m.sum():
            age_t[i] += int(wt[m].sum()); age_m[i] += int(wm[m].sum())
            age_n[i] += int(m.sum())
    for ti_, tname in enumerate(TYPE_NAMES):
        m = nd & (y_type[ks] == ti_ + 1)
        by_type[tname][0] += int(wt[m].sum()); by_type[tname][1] += int(m.sum())

    # Q5 record, over every REACHABLE instant -- the population the control loss
    # is actually averaged over. p_fail uses the same expression as the loss:
    # a sigmoid on the model's own judged peak against (limit - margin).
    rch = reach.cpu().numpy()
    if rch.any():
        tgt_c = model.limit - model.margin
        p_fail_c = torch.sigmoid((pk_ctrl_m - tgt_c) / A['dp_tau']).cpu().numpy()
        calib_rows.append(np.stack([
            p_fail_c[rch],
            (~win_t.cpu().numpy())[rch].astype(float),          # true failure
            (pk_ctrl_m.cpu().numpy() > tgt_c)[rch].astype(float),   # model, at tgt
            (~win_m.cpu().numpy())[rch].astype(float),          # model, at limit
            nd[rch].astype(float),
        ], axis=1))

    # Q13 record (needy cases only -- the ones a rescue rate is computed over)
    if nd.any():
        sat = (dP.abs().amax(dim=1) / model.dp_max).cpu().numpy()
        case_rows.append(np.stack([
            y_type[ks][nd].astype(float),
            meta_alpha[ks][nd],
            wt[nd].astype(float),
            nrm.cpu().numpy()[nd],
            sat[nd],
        ], axis=1))

    n_done += bs
    print(f"\r  {n_done}/{args.n}", end='', flush=True)
print()

R = lambda a, b: (a / b if b else float('nan'))
mean = lambda L: float(torch.cat(L).mean()) if sum(len(x) for x in L) else float('nan')
r_true, r_model = R(tot['resc_true'], tot['need']), R(tot['resc_model'], tot['need'])
print(f"\n{'='*70}")
print(f"TRUE-DYNAMICS VERIFICATION  ({args.split} split, {tot['n']} instants)")
print(f"{'='*70}")
print(f"  cases needing rescue (true dynamics) : {tot['need']}")
print(f"  excluded, no fault active at decision: {tot['excluded']}  "
      f"({100*R(tot['excluded'], tot['need']+tot['excluded']):.1f}% of "
      f"otherwise-needy cases -- nothing for dP to act on)")
print(f"  rescue rate, TRUE dynamics           : {r_true:.4f}")
print(f"  rescue rate, model's own rollout     : {r_model:.4f}")
print(f"  the two agree on                     : {100*R(tot['agree'], tot['need']):.1f}% of cases")
print(f"  overall ok (true, actionable)        : {R(tot['ok_true'], tot['ok_n']):.4f}")
print(f"  rescue by type (true)                : " + "  ".join(
    f"{t} {R(h, n):.3f}(n={n})" for t, (h, n) in by_type.items()))
print(f"\n  HARMED: healthy without dP, pushed out by it: {tot['harmed']} of "
      f"{tot['healthy']} ({100*R(tot['harmed'], tot['healthy']):.2f}%)")
print(f"\n  judged to END OF RUN (through clearing and ring-down):")
print(f"    with dP                            : {R(tot['resc_run'], tot['need']):.4f}")
print(f"    without dP                         : {R(tot['free_run'], tot['need']):.4f}")
print(f"  dP held until: {args.dp_hold}")
print(f"  mean ||dP||                          : {mean(norm_all):.4f} pu  "
      f"(needing rescue {mean(norm_need):.4f}, not needing {mean(norm_quiet):.4f})")
print(f"\n  rescue by fault age (true)  {fmt_age(AGE_BINS, age_t, age_n)}")
print(f"  rescue by fault age (model) {fmt_age(AGE_BINS, age_m, age_n)}")

# ── Q13: where do the misses live? ──────────────────────────────────────────
# A failure with dP at the +/-dp_max rail is actuator-limited: no set-point
# change inside the limits could have saved it. A failure with authority to
# spare is the head's -- a solution existed and it did not emit one. Splitting
# by true alpha separates "severe faults are simply beyond reach" from "the
# head under-corrects when it matters".
ALPHA_BINS = [(0.10, 0.30), (0.30, 0.50), (0.50, 0.70), (0.70, 0.90), (0.90, 1.01)]
RAIL = 0.95

if case_rows:
    C = np.concatenate(case_rows, axis=0)
    print(f"\n{'-'*70}")
    print(f"WHERE DO THE MISSES LIVE?  ({len(C)} needy cases, true dynamics)")
    print(f"{'-'*70}")
    print(f"  {'type':5s} {'alpha':12s} {'n':>5s} {'rescued':>8s} "
          f"{'mean|dP|':>9s} {'at rail':>8s} {'miss@rail':>10s} {'miss spare':>11s}")
    for ti_, tname in enumerate(TYPE_NAMES):
        T = C[C[:, 0] == ti_ + 1]
        if not len(T):
            continue
        for lo_, hi_ in ALPHA_BINS:
            B = T[(T[:, 1] >= lo_) & (T[:, 1] < hi_)]
            if not len(B):
                continue
            miss = B[B[:, 2] == 0]
            m_rail = int((miss[:, 4] >= RAIL).sum()) if len(miss) else 0
            print(f"  {tname:5s} {lo_:.2f}-{hi_:<7.2f} {len(B):5d} "
                  f"{B[:, 2].mean():8.3f} {B[:, 3].mean():9.4f} "
                  f"{100*(B[:, 4] >= RAIL).mean():7.1f}% "
                  f"{m_rail:10d} {len(miss)-m_rail:11d}")
        miss = T[T[:, 2] == 0]
        m_rail = int((miss[:, 4] >= RAIL).sum()) if len(miss) else 0
        print(f"  {tname:5s} {'ALL':12s} {len(T):5d} {T[:, 2].mean():8.3f} "
              f"{T[:, 3].mean():9.4f} {100*(T[:, 4] >= RAIL).mean():7.1f}% "
              f"{m_rail:10d} {len(miss)-m_rail:11d}")

    G = C[C[:, 0] == 1]                      # generator trips
    gmiss = G[G[:, 2] == 0]
    if len(gmiss):
        n_rail = int((gmiss[:, 4] >= RAIL).sum())
        n_spare = len(gmiss) - n_rail
        print(f"\n  GENERATOR TRIPS: {len(gmiss)} of {len(G)} needy cases missed "
              f"(rescue {G[:, 2].mean():.4f})")
        print(f"    dP at the rail (>={RAIL:.2f} of dp_max): {n_rail:4d}  "
              f"({100*n_rail/len(gmiss):.1f}%)  -> actuator-limited")
        print(f"    authority to spare                   : {n_spare:4d}  "
              f"({100*n_spare/len(gmiss):.1f}%)  -> head under-corrects")
        if (G[:, 2] == 1).any():
            print(f"    mean true alpha: misses {gmiss[:, 1].mean():.3f}  vs  "
                  f"rescued {G[G[:, 2] == 1][:, 1].mean():.3f}")
        if n_rail > 2 * n_spare:
            print("    -> MARGINAL: most misses had no authority left. The gap is a\n"
                  "       feasibility limit of G1/G2, not a head failure.")
        elif n_spare > 2 * n_rail:
            print("    -> HEAD: most misses left authority unused. A larger correction\n"
                  "       existed and was not emitted -- an amortisation gap.")
        else:
            print("    -> MIXED: neither cause dominates; both are live.")
    elif len(G):
        print(f"\n  GENERATOR TRIPS: all {len(G)} needy cases rescued.")

# ── Q5: is p_fail calibrated? ───────────────────────────────────────────────
# The control loss minimises mean(p_fail). That mean is the expected failure
# rate of the MODEL'S rollout, which is only useful if (a) p_fail tracks the
# model's own outcome, and (b) the model's outcome tracks the true one. This
# separates the two so a miscalibration can be attributed.
#
# One deliberate offset to keep in mind: p_fail is a surrogate for
# "peak > limit - margin", while the true outcome is "peak > limit". The margin
# makes p_fail conservative ON PURPOSE, so a positive bias against the true
# outcome is expected and is not by itself a fault.
if calib_rows:
    K = np.concatenate(calib_rows, axis=0)
    pf_, ft_, fm_tgt, fm_lim, ndy = (K[:, 0], K[:, 1], K[:, 2], K[:, 3], K[:, 4])
    EDGES = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0001]

    def reliability(mask, label):
        p, t, m, g = pf_[mask], ft_[mask], fm_lim[mask], fm_tgt[mask]
        if not len(p):
            return
        print(f"\n  {label}  (n = {len(p)})")
        print(f"    {'p_fail bin':14s} {'n':>6s} {'mean p':>8s} "
              f"{'model@tgt':>10s} {'model@limit':>12s} {'actual TRUE':>12s}")
        ece = 0.0
        for lo_, hi_ in zip(EDGES[:-1], EDGES[1:]):
            b_ = (p >= lo_) & (p < hi_)
            if not b_.sum():
                continue
            ece += b_.sum() / len(p) * abs(p[b_].mean() - t[b_].mean())
            print(f"    {lo_:.2f}-{hi_ if hi_ <= 1 else 1.0:<9.2f} {int(b_.sum()):6d} "
                  f"{p[b_].mean():8.3f} {g[b_].mean():10.3f} {m[b_].mean():12.3f} "
                  f"{t[b_].mean():12.3f}")
        print(f"    {'OVERALL':14s} {len(p):6d} {p.mean():8.3f} "
              f"{g.mean():10.3f} {m.mean():12.3f} {t.mean():12.3f}")
        print(f"    expected calibration error (vs TRUE): {ece:.4f}")
        # The honest test of the surrogate is against the event it is a
        # surrogate FOR (peak > tgt). Comparing it to peak > limit folds in the
        # margin, which is a deliberate offset and not a calibration failure.
        ece_tgt = 0.0
        for lo_, hi_ in zip(EDGES[:-1], EDGES[1:]):
            b_ = (p >= lo_) & (p < hi_)
            if b_.sum():
                ece_tgt += b_.sum() / len(p) * abs(p[b_].mean() - g[b_].mean())
        print(f"    expected calibration error (vs its OWN event, peak>tgt): "
              f"{ece_tgt:.4f}")
        # overconfidence where it matters: confident-safe cases that still fail
        lowm = p < 0.05
        if lowm.sum():
            print(f"    confident-safe bin (p<0.05): {int(lowm.sum())} cases, "
                  f"p={p[lowm].mean():.3f}, TRUE failures {t[lowm].mean():.3f} "
                  f"({int(t[lowm].sum())} cases)")
        return p.mean(), t.mean(), m.mean(), g.mean()

    print(f"\n{'-'*70}")
    print("IS p_fail CALIBRATED?  (reachable instants; p_fail is what the control")
    print("loss minimises, so its reliability is the loss's reliability)")
    print(f"{'-'*70}")
    res_all = reliability(np.ones(len(pf_), bool), 'ALL REACHABLE')
    res_need = reliability(ndy > 0, 'NEEDING RESCUE')

    if res_all:
        mp, mt, mm, mg = res_all
        print(f"\n  predicted failure rate (mean p_fail)   : {mp:.4f}")
        print(f"  model's own rate at its OWN event (tgt): {mg:.4f}   "
              f"[surrogate error {mp-mg:+.4f}]")
        print(f"  model's own rate at the LIMIT          : {mm:.4f}   "
              f"[margin offset {mg-mm:+.4f}, by design]")
        print(f"  TRUE failure rate                      : {mt:.4f}   "
              f"[rollout error {mm-mt:+.4f}]")
        print(f"  net: p_fail vs reality                 : {mp-mt:+.4f}")
        if abs(mp - mg) <= 0.02:
            print("  -> the SURROGATE is sound: p_fail tracks the event it stands in\n"
                  "     for. dp_tau needs no change.")
        else:
            print("  -> the SURROGATE itself is off against its own event; dp_tau is\n"
                  "     the lever.")
        # The rollout verdict must be read on the NEEDY cases. Over all reachable
        # instants most are quiet and both the model and reality call them safe,
        # which dilutes the error toward zero and can read as "trustworthy" while
        # the rescue rate is several points optimistic.
        if res_need:
            _, nt, nm, _ = res_need
            print(f"\n  on cases NEEDING RESCUE (the population that matters):")
            print(f"    model's own failure rate : {nm:.4f}")
            print(f"    TRUE failure rate        : {nt:.4f}   "
                  f"[rollout error {nm-nt:+.4f}]")
            if nm < nt - 0.02:
                print("  -> the ROLLOUT is optimistic where it counts: the model\n"
                      "     under-predicts failures on the cases it is meant to save.\n"
                      "     --margin is the lever; the surrogate is not at fault.")
            else:
                print("  -> the ROLLOUT holds up on the needy cases.")
        if mp > mt:
            print(f"  -> net effect is CONSERVATIVE ({mp:.3f} predicted vs {mt:.3f}\n"
                  f"     actual overall), so the objective errs toward over-correcting.")
    calib_summary = {'n': int(len(pf_)), 'mean_p_fail': float(pf_.mean()),
                     'true_failure_rate': float(ft_.mean()),
                     'model_failure_rate': float(fm_lim.mean()),
                     'model_failure_rate_at_tgt': float(fm_tgt.mean())}
else:
    calib_summary = None

gap = r_model - r_true
print(f"\n  model-internal minus true = {gap*100:+.1f} points")
if abs(gap) < 0.02:
    print("  -> the identified dynamics are trustworthy for control; the reported\n"
          "     rescue rate survives contact with the real system.")
elif gap > 0:
    print("  -> the model-internal number is OPTIMISTIC. Raise --margin toward the\n"
          "     trajectory RMSE and re-run refinement against a realistic buffer.")
else:
    print("  -> the model-internal number is PESSIMISTIC: the true system fares\n"
          "     better than the model predicts. Conservative, not unsafe.")

json.dump({'split': args.split, 'n_instants': tot['n'], 'n_need': tot['need'],
           'rescue_true': r_true, 'rescue_model': r_model,
           'agreement': R(tot['agree'], tot['need']),
           'ok_true': R(tot['ok_true'], tot['ok_n']),
           'harmed': tot['harmed'], 'healthy': tot['healthy'],
           'mean_dP': mean(norm_all), 'mean_dP_need': mean(norm_need),
           'mean_dP_quiet': mean(norm_quiet), 'dp_hold': args.dp_hold,
           'excluded_no_active_fault': tot['excluded'],
           'rescue_true_full_run': R(tot['resc_run'], tot['need']),
           'no_dp_full_run': R(tot['free_run'], tot['need']),
           'by_type': {t: {'n': n, 'true': R(h, n)} for t, (h, n) in by_type.items()},
           'by_age': [{'lo': lo, 'hi': hi, 'n': n, 'true': R(h, n), 'model': R(m, n)}
                      for (lo, hi), h, m, n in zip(AGE_BINS, age_t, age_m, age_n)],
           'p_fail_calibration': calib_summary},
          open(os.path.join(out_dir, 'verify_results.json'), 'w'), indent=1)


# ── plots: cases chosen PER FAULT TYPE ───────────────────────────────────────
# Decision placed just after onset, as a deployed controller would act. Cases
# that genuinely need rescue are preferred; otherwise the best available case
# of that type is shown and marked as such in the title.
def select_plot_cases(ft, n_want, rng):
    lo, hi = A['w_max'], float(t_grid[-1]) - A['horizon']
    ow = A.get('onset_window', 0.2) or 0.2
    cand = idx[(y_type[idx] == ft + 1) & (t_fault[idx] >= lo) & (t_fault[idx] < hi)]
    if len(cand) == 0:
        return []
    cand = rng.permutation(cand)[:48]
    span = np.minimum(ow, t_clear[cand] - t_fault[cand]) * 0.999
    tkc = np.clip(t_fault[cand] + rng.uniform(0, 1, len(cand)) * span, lo, hi)
    b = su.to_dev(su.make_batch(cand, rng, tk_fixed=tkc))
    with torch.no_grad():
        o_ = model(b['feat'], b['x_meas'], hard=True, ctrl=True, vmask=b['vmask'])
    dP_ = o_['dP']
    trf, stf = full_sim(b['ks'], b['tk'], None)
    trc, _ = full_sim(b['ks'], b['tk'], dP_, hold=args.dp_hold, vt=b['valid_t'])
    pf, _ = peaks_from(trf, b['tk'], b['valid_t'])
    pc, pc_run = peaks_from(trc, b['tk'], b['valid_t'])
    i_tk = torch.as_tensor(np.round(b['tk'] / args.dt_true).astype(int), device=device)
    dev0 = (stf[torch.arange(len(cand), device=device), i_tk][:, NG:2 * NG]
            - 1).abs().amax(1)
    needed = ((pf > model.limit) & (dev0 <= model.limit)).cpu().numpy()
    # Preference order, most convincing first:
    #   1. needed rescue, and dP keeps omega inside the envelope for the WHOLE
    #      run after the grace period -- what the plot actually shows, since
    #      it spans the full 8 s, not just the judged window
    #   2. needed rescue, inside the envelope over the judged window only
    #      (may leave it later, after the window or after clearing)
    #   3. needed rescue, not rescued
    #   4. did not need rescue
    # This is a SHOWCASE choice: it picks successes on purpose. The rescue
    # rates above are computed on random instants and are not affected.
    ok_run = (pc_run <= model.limit).cpu().numpy()
    ok_win = (pc <= model.limit).cpu().numpy()
    tiers = [needed & ok_run, needed & ok_win & ~ok_run,
             needed & ~ok_win, ~needed]
    order = [j for t in tiers for j in np.where(t)[0]]
    status = {}
    for name, t in zip(('rescued (whole run)', 'rescued in judged window only',
                        'NOT rescued', 'no rescue needed in judged window'), tiers):
        for j in np.where(t)[0]:
            status[j] = name
    return [(int(b['ks'][j]), float(b['tk'][j]), float(b['valid_t'][j]),
             dP_[j].cpu().numpy(), float(pf[j]), float(pc[j]), status[j])
            for j in order[:n_want]]


rng_p = np.random.default_rng(args.seed + 99)
plot_pool = {}
for ft in range(3):
    plot_pool[ft] = select_plot_cases(ft, args.plots_per_type, rng_p)
    if not plot_pool[ft]:
        print(f"  {TYPE_NAMES[ft]}: no scenarios of this type in split "
              f"'{args.split}' with a usable fault time -- nothing to plot")
    else:
        print(f"  {TYPE_NAMES[ft]}: {len(plot_pool[ft])} case(s), "
              + ", ".join(f"{sum(c[6] == st for c in plot_pool[ft])} {st}"
                          for st in dict.fromkeys(c[6] for c in plot_pool[ft])))

print(f"\nplotting {sum(len(v) for v in plot_pool.values())} cases ...")
h = args.dt_true
# G1, G2 carry governors and receive dP; G3-G6 are synchronous condensers
COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#8c564b', '#e377c2']
LW = [2.0, 2.0, 1.2, 1.2, 1.2, 1.2]
for ft, cases in plot_pool.items():
    for k, tkj, vtj, dPj, pf, pc, status_ in cases:
        ksa = np.array([k]); tka = np.array([tkj])
        dPt = torch.tensor(dPj, dtype=torch.float32, device=device).unsqueeze(0)
        tr_free = full_sim(ksa, tka, None)[0][0].cpu().numpy()
        tr_ctrl = full_sim(ksa, tka, dPt, hold=args.dp_hold,
                           vt=np.array([vtj]))[0][0].cpu().numpy()
        tt = np.arange(tr_free.shape[0]) * h

        fig, ax = plt.subplots(figsize=(13, 5.5))
        for g in range(NG):
            ax.plot(tt, tr_free[:, g], color=COLORS[g], lw=LW[g] * 0.75, ls='--')
            ax.plot(tt, tr_ctrl[:, g], color=COLORS[g], lw=LW[g])
        for sgn in (+1, -1):
            ax.axhline(1 + sgn * OMEGA_LIMIT, color='k', lw=1.0, ls=':')
        ax.axvspan(t_fault[k], t_clear[k], color='#ffe9b0', alpha=0.55, zorder=0)
        ax.axvline(tkj, color='0.25', lw=1.2)
        ax.axvspan(tkj, tkj + model.grace, color='0.82', zorder=1)

        handles = [Line2D([], [], color='0.2', lw=1.5, ls='--', label='without dP'),
                   Line2D([], [], color='0.2', lw=2.0, label='with dP')]
        handles += [Line2D([], [], color=COLORS[g], lw=LW[g],
                           label=f'G{g+1}' + (' (gov)' if g < 2 else ''))
                    for g in range(NG)]
        handles += [Line2D([], [], color='k', lw=1.0, ls=':', label='envelope'),
                    plt.Rectangle((0, 0), 1, 1, color='#ffe9b0', alpha=0.55,
                                  label='fault active'),
                    Line2D([], [], color='0.25', lw=1.2, label='decision'),
                    plt.Rectangle((0, 0), 1, 1, color='0.82',
                                  label=f'decision window ({1000*model.grace:.0f} ms grace)')]
        ax.legend(handles=handles, fontsize=8, loc='lower right', ncol=3)
        ax.set_xlim(0, float(t_grid[-1]))
        ax.set_xlabel('t [s]'); ax.set_ylabel(r'$\omega$ [pu]'); ax.grid(alpha=0.3)
        ax.set_title(
            f"{TYPE_NAMES[ft]} fault, scenario {k}  "
            f"[{status_}]"
            f"  |  loc={meta_loc[k]} alpha={meta_alpha[k]:.2f}  "
            f"fault {t_fault[k]:.2f}-{t_clear[k]:.2f}s  "
            f"decision t={tkj:.2f}s (+{1000*(tkj-t_fault[k]):.0f} ms)\n"
            f"dP = [G1 {dPj[0]:+.3f}, G2 {dPj[1]:+.3f}] pu   |   "
            f"peak |w-1|: without dP {pf:.5f} -> with dP {pc:.5f}  "
            f"(limit {OMEGA_LIMIT})", fontsize=9)
        fig.tight_layout()
        p_ = os.path.join(out_dir, 'plots', f"{TYPE_NAMES[ft]}_sc{k}_t{tkj:.2f}.png")
        fig.savefig(p_, dpi=130); plt.close(fig)
        print(f"  {p_}")

print(f"\n-> {out_dir}/verify_results.json")
