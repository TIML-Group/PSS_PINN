"""
end_mech_cpu.py  (untracked diagnostic — CPU only)
==================================================
Why do needy cases that are rescued IN-WINDOW still fail at END OF RUN
under --dp_hold clear (0.837 -> 0.668 on the frozen run)?

Two candidate mechanisms, separated here by trajectory shape, not assertion:
  KICK  — the withdrawal step at t_clear excites a post-clearing transient
          that briefly exits the band and RECOVERS (settles back inside).
  BASIN — support evaporates: the state at t_clear sits outside/at the edge
          of the post-fault basin; |omega-1| GROWS and never comes back.

Replicates verify_true30.py's full_sim / peaks_from / gate line-for-line,
but forced to CPU (no GPU contention) and recording per-case exit times.
Draw: compare_out_own_s42/instants.json (frozen seed-42 shared draw).

Output: verify_end_mech/ PNGs + end_mech_records.json
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''   # force CPU before torch loads
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from case30_common import Setup, actionable, NG, N_STATE

CKPT = 'case30_ctrl/refined_full/best.pt'
DATA = 'data30_mm'
INSTANTS = 'compare_out_own_s42/instants.json'
OUT = 'verify_end_mech'
DT = 0.005
BATCH = 128
os.makedirs(OUT, exist_ok=True)

device = torch.device('cpu')
su = Setup(CKPT, DATA, device=device)
model = su.model
model.eval()
S = su.S
t_fault, t_clear, t_grid = su.t_fault, su.t_clear, su.t_grid
y_type = su.y_type
meta_alpha = su.meta_alpha
LIMIT = model.limit
GRACE = float(model.grace)
print(f"limit {LIMIT}, grace {GRACE} s")

inst = json.load(open(INSTANTS))
print("pool_note:", repr(inst.get('pool_note')), "| split:", repr(inst.get('split')),
      "| seed:", inst.get('seed'))
KS = np.asarray(inst['ks'], dtype=int)[:2048]
TK = np.asarray(inst['tk'], dtype=float)[:2048]

n_steps = int(round(float(t_grid[-1]) / DT))
ti = torch.arange(n_steps + 1, dtype=torch.float32) * DT


@torch.no_grad()
def full_sim(ks, tk_arr, dP=None, hold='clear', vt=None):
    """verify_true30.full_sim, eager CPU copy."""
    n = len(ks)
    Yf, dPm = su.true_network(ks)
    tf = torch.tensor(t_fault[ks], dtype=torch.float32)
    tc = torch.tensor(t_clear[ks], dtype=torch.float32)
    tk = torch.tensor(np.asarray(tk_arr), dtype=torch.float32)
    if hold == 'clear':
        t_off = tc
    elif hold == 'run':
        t_off = torch.full_like(tk, float(t_grid[-1]) + 1.0)
    else:
        t_off = tk + torch.tensor(np.asarray(vt), dtype=torch.float32)
    dP6 = model.scatter_dp(dP) if dP is not None else None
    b = dict(
        x=S.x0.to(torch.float32).unsqueeze(0).expand(n, N_STATE).clone(),
        tf=tf, tc=tc, tk=tk, t_off=t_off, dPm=dPm,
        dP=dP6 if dP6 is not None else torch.zeros(n, NG),
        traj=torch.zeros(n, n_steps + 1, NG),
        states=torch.zeros(n, n_steps + 1, N_STATE),
    )
    Mp, Np = S.pe_consts(S.Yred_pre.expand(n, NG, NG))
    b['Mf'], b['Nf'] = S.pe_consts(Yf)
    b['Mp'], b['Np'] = Mp, Np
    x = b['x']
    traj, states = b['traj'], b['states']
    traj[:, 0] = x[:, NG:2 * NG]
    states[:, 0] = x
    tf_, tc_, tk_, t_off_ = b['tf'], b['tc'], b['tk'], b['t_off']
    Mf, Nf, Mp_, Np_ = b['Mf'], b['Nf'], b['Mp'], b['Np']
    dP, dPm = b['dP'], b['dPm']
    act_hist = []
    for i in range(n_steps):
        tm = i * DT + DT / 2
        act = (tm >= tf_) & (tm < tc_)
        act_hist.append(act)
        M = torch.where(act.view(n, 1, 1), Mf, Mp_)
        N = torch.where(act.view(n, 1, 1), Nf, Np_)
        Ps = S.Pm0.unsqueeze(0) + dPm * act.view(n, 1).float()
        on = (tm >= tk_) & (tm < t_off_)
        Ps = Ps + dP * on.view(n, 1).float()
        x = S.rk4_real(x, M, N, Ps, DT)
        traj[:, i + 1] = x[:, NG:2 * NG]
        states[:, i + 1] = x
    return traj, states, torch.stack(act_hist, 1)


def analyze(traj, tk_a, vt_a, tc_a):
    """Peak over judged window / full run + first band-exit time + final-level."""
    B, T, _ = traj.shape
    d = (traj - 1.0).abs().amax(dim=2)                      # (B, T)
    tk_t = torch.as_tensor(tk_a, dtype=torch.float32).unsqueeze(1)
    vt_t = torch.as_tensor(np.asarray(vt_a), dtype=torch.float32).unsqueeze(1)
    m_run = ti.unsqueeze(0) >= (tk_t + GRACE)
    m_win = m_run & (ti.unsqueeze(0) <= tk_t + vt_t)
    neg = torch.full_like(d, -1.0)
    pk_win = torch.where(m_win, d, neg).amax(1)
    pk_run = torch.where(m_run, d, neg).amax(1)
    out = (d > LIMIT) & m_run
    has = out.any(dim=1)
    i_exit = out.float().argmax(dim=1)                      # first True index
    t_exit = torch.where(has, ti[i_exit],
                         torch.full_like(ti[i_exit], float('nan')))
    final = d[:, -int(round(0.5 / DT)):].mean(dim=1)        # last 0.5 s mean
    return pk_win, pk_run, t_exit, final, d


recs = []
TYPE_N = lambda y: ['gen', 'load', 'line'][y - 1]   # noqa: E731  (helper first use)
tot = dict(n=0, need=0, resc_win=0, resc_run=0, free_run=0)
buf_by_type = {t: [] for t in ['gen', 'load', 'line']}
n_done = 0
while n_done < len(KS):
    bs = min(BATCH, len(KS) - n_done)
    ks = KS[n_done:n_done + bs]
    tk_a = TK[n_done:n_done + bs]
    rng = np.random.default_rng(0)
    b = su.to_dev(su.make_batch(ks, rng, tk_fixed=tk_a))
    with torch.no_grad():
        o = model(b['feat'], b['x_meas'], hard=True, ctrl=True, vmask=b['vmask'])
        dP = o['dP']
    tr_free, st_free, _ = full_sim(ks, tk_a, None)
    tr_ctrl, _, _ = full_sim(ks, tk_a, dP, hold='clear', vt=b['valid_t'])

    i_tk = torch.as_tensor(np.round(tk_a / DT).astype(int))
    x_clean = st_free[torch.arange(bs), i_tk]
    pkf_w, pkf_r, _, _, d_free = analyze(tr_free, tk_a, b['valid_t'],
                                         torch.tensor(t_clear[ks], dtype=torch.float32))
    pkc_w, pkc_r, t_exit, final_lvl, d_ctrl = analyze(
        tr_ctrl, tk_a, b['valid_t'], torch.tensor(t_clear[ks], dtype=torch.float32))
    reach = actionable(pkf_w >= 0, x_clean, b['active'], model.limit)
    need = (pkf_w > model.limit) & reach

    win_t = (pkc_w <= model.limit)
    run_t = (pkc_r <= model.limit)
    tot['n'] += bs
    tot['need'] += int(need.sum())
    tot['resc_win'] += int(win_t[need].sum())
    tot['resc_run'] += int(run_t[need].sum())
    tot['free_run'] += int((pkf_r[need] <= model.limit).sum())

    nd = need.numpy()
    nrm = dP.pow(2).sum(1).sqrt().numpy()
    t_ex = t_exit.numpy()
    tc_a = t_clear[ks].astype(float)
    for j in np.nonzero(nd)[0]:
        post = t_ex[j] - tc_a[j]
        recs.append(dict(
            k=int(ks[j]), tk=float(tk_a[j]), tf=float(t_fault[ks[j]]),
            tc=float(tc_a[j]),
            typ=int(y_type[ks[j]]), alpha=float(meta_alpha[ks[j]]),
            dp_norm=float(nrm[j]), nsteps=int(bs),
            resc_win=bool(win_t[j]), resc_run=bool(run_t[j]),
            pk_win=float(pkc_w[j]), pk_run=float(pkc_r[j]),
            exit_after_clear=float(post) if np.isfinite(post) else None,
            final_lvl=float(final_lvl[j]),
            pk_win_free=float(pkf_w[j])))
        # stash the worst-case trajectory for plotting candidates
        if win_t[j] and not run_t[j]:
            buf_by_type[TYPE_N(y_type[ks[j]])].append(
                dict(rec=recs[-1], ctrl=tr_ctrl[j].numpy(),
                     free=d_free[j].numpy(),
                     act=None, tk=float(tk_a[j]), vt=float(b['valid_t'][j]),
                     dP=dP[j].numpy()))
    print(f"\r  {n_done+bs}/{len(KS)}", end='', flush=True)
    n_done += bs

print()
print(f"\nneed {tot['need']}  | in-window {tot['resc_win']/tot['need']:.4f}"
      f"  | end-of-run {tot['resc_run']/tot['need']:.4f}"
      f"  | free-run {tot['free_run']/tot['need']:.4f}   (frozen s42 draw, CPU)")

# lost-in-ring-down population ---------------------------------------------
lost = [r for r in recs if r['resc_win'] and not r['resc_run']]
exit_times = np.array([r['exit_after_clear'] for r in lost
                       if r['exit_after_clear'] is not None])
print(f"lost in ring-down: {len(lost)}/{tot['need']}"
      f" ({tot['resc_win']/tot['need']:.3f} -> {tot['resc_run']/tot['need']:.3f})")
if len(exit_times):
    recovered = np.array([r['final_lvl'] <= LIMIT for r in lost
                          if r['exit_after_clear'] is not None])
    print(f"exit-after-clear: min {exit_times.min():.3f}  "
          f"p25 {np.percentile(exit_times,25):.3f}  med {np.median(exit_times):.3f}"
          f"  p75 {np.percentile(exit_times,75):.3f}  max {exit_times.max():.3f} s")
    kick = [r for r in lost if r['exit_after_clear'] is not None
            and r['final_lvl'] <= LIMIT]
    base = [r for r in lost if r['exit_after_clear'] is None
            or r['final_lvl'] > LIMIT]
    print(f"KICK-type (exits band post-clear, settles back in): {len(kick)}")
    print(f"BASIN-type (stays out / diverges at end of run): {len(base)}")

json.dump(dict(summary=tot, records=recs), open(f'{OUT}/end_mech_records.json', 'w'),
          indent=1)
print(f"records -> {OUT}/end_mech_records.json")
