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
                      for (lo, hi), h, m, n in zip(AGE_BINS, age_t, age_m, age_n)]},
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
