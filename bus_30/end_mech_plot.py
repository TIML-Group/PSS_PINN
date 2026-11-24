"""
end_mech_plot.py  (untracked diagnostic — CPU only; reads end_mech_records.json)
===============================================================================
1. Picks the clearest KICK-type case (post-clearing band excursion that
   recovers) and the boundary case (exit after the judged window ends but
   before clearing), re-simulates exactly, plots omega deviation vs policies.
2. Counterfactual: ramped withdrawal (dP decayed linearly to zero over
   0.3 s / 0.6 s after t_clear instead of a step at t_clear) on ALL lost
   cases — quantifies how much of the ring-down loss the withdrawal step causes.
3. Histogram of exit-after-clear times for the 52 lost cases.

Output: verify_end_mech/end_mech_kick.png, end_mech_boundary.png,
        end_mech_exit_hist.png + JSON summary appended to end_mech_records.json
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''   # force CPU before torch loads
import json
import numpy as np
import torch

from case30_common import Setup, NG, N_STATE

CKPT = 'case30_ctrl/refined_full/best.pt'
DATA = 'data30_mm'
OUT = 'verify_end_mech'
DT = 0.005
LIMIT = 0.005
GRACE = 0.2

device = torch.device('cpu')
su = Setup(CKPT, DATA, device=device)
model = su.model
model.eval()
S = su.S
t_fault, t_clear, t_grid = su.t_fault, su.t_clear, su.t_grid
n_steps = int(round(float(t_grid[-1]) / DT))
ti = torch.arange(n_steps + 1, dtype=torch.float32) * DT

recs = json.load(open(f'{OUT}/end_mech_records.json'))
lost = [r for r in recs['records'] if r['resc_win'] and not r['resc_run']]
print(f"records: {len(recs['records'])} needy, lost {len(lost)}")
pre_clear = [r for r in lost if r['exit_after_clear'] < 0]
print(f"exit BEFORE clearing (judged window ends at tk+horizon): {len(pre_clear)}")


@torch.no_grad()
def sim(ks, tk_arr, dP=None, ramp_dur=None):
    """verify_true30 full_sim, eager CPU; ramp_dur=None -> step at t_clear."""
    n = len(ks)
    Yf, dPm = su.true_network(ks)
    tf = torch.tensor(t_fault[ks], dtype=torch.float32)
    tc = torch.tensor(t_clear[ks], dtype=torch.float32)
    tk = torch.tensor(np.asarray(tk_arr), dtype=torch.float32)
    dP6 = model.scatter_dp(dP) if dP is not None else None
    n_steps = int(round(float(t_grid[-1]) / DT))
    b = dict(
        x=S.x0.to(torch.float32).unsqueeze(0).expand(n, N_STATE).clone(),
        traj=torch.zeros(n, n_steps + 1, NG),
    )
    Mp, Np = S.pe_consts(S.Yred_pre.expand(n, NG, NG))
    Mf, Nf = S.pe_consts(Yf)
    x = b['x']
    traj = b['traj']
    traj[:, 0] = x[:, NG:2 * NG]
    tk_ = tk.unsqueeze(1)          # dP on from decision instant
    tc_ = tc.unsqueeze(1)
    for i in range(n_steps):
        tm = (i * DT + DT / 2)
        T = torch.full((n, 1), tm)
        act = (T >= tf.view(n, 1)) & (T < tc.view(n, 1))
        M = torch.where(act.view(n, 1, 1), Mf, Mp)
        N = torch.where(act.view(n, 1, 1), Nf, Np)
        Ps = S.Pm0.unsqueeze(0) + dPm * act.view(n, 1).float()
        on = (T >= tk_) & (T < tc_)
        if dP6 is not None:
            if ramp_dur is None:
                f = on.float()
            else:
                # 1 until t_clear, then linear decay to 0 over ramp_dur
                raw = (tc_ + ramp_dur - T) / ramp_dur
                f = on.float() + torch.clamp(raw, min=0.0, max=1.0) * (~on).float()
                f = torch.where(T >= tk_, f, torch.zeros_like(f))
            Ps = Ps + dP6 * f
        x = S.rk4_real(x, M, N, Ps, DT)
        traj[:, i + 1] = x[:, NG:2 * NG]
    return traj


@torch.no_grad()
def d_series(traj):
    return (traj - 1.0).abs().amax(dim=2)


@torch.no_grad()
def peaks(d2, tk_a, vt_a):
    tk_t = torch.as_tensor(tk_a, dtype=torch.float32).unsqueeze(1)
    vt_t = torch.as_tensor(np.asarray(vt_a), dtype=torch.float32).unsqueeze(1)
    m_run = ti.unsqueeze(0) >= (tk_t + GRACE)
    m_win = m_run & (ti.unsqueeze(0) <= tk_t + vt_t)
    neg = torch.full_like(d2, -1.0)
    return (torch.where(m_win, d2, neg).amax(1),
            torch.where(m_run, d2, neg).amax(1))


def pick_case(rec):
    import numpy as _np
    r = np.random.default_rng(0)
    b = su.to_dev(su.make_batch(np.array([rec['k']]), r, tk_fixed=np.array([rec['tk']])))
    o = model(b['feat'], b['x_meas'], hard=True, ctrl=True, vmask=b['vmask'])
    return b, o['dP'], b['valid_t']


# ── 1. counterfactual on ALL lost cases ─────────────────────────────────────
res = {}
for dur in (0.3, 0.6):
    rescued = 0
    for i in range(0, len(lost), 32):
        chunk = lost[i:i + 32]
        ks = np.array([c['k'] for c in chunk])
        tk_a = np.array([c['tk'] for c in chunk])
        rng = np.random.default_rng(0)
        b = su.to_dev(su.make_batch(ks, rng, tk_fixed=tk_a))
        with torch.no_grad():
            o = model(b['feat'], b['x_meas'], hard=True, ctrl=True,
                      vmask=b['vmask'])
        dP = o['dP']
        tr = sim(ks, tk_a, dP, ramp_dur=dur)
        d2 = d_series(tr)
        _, pk_run = peaks(d2, tk_a, b['valid_t'])
        rescued += int((pk_run <= LIMIT).sum())
    res[dur] = rescued
    print(f"ramped withdrawal {dur:.1f} s: end-of-run rescue {rescued}/{len(lost)}"
          f" lost cases recovered (step-withdrawal baseline: 0/{len(lost)})")

# ── 2. example figures ───────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_case(rec, tag, title):
    b, dP, vt = pick_case(rec)
    k = np.array([rec['k']])
    tka = np.array([rec['tk']])
    tr_c = sim(k, tka, dP)                       # step withdrawal at clear
    tr_r = sim(k, tka, dP, ramp_dur=0.3)         # ramped withdrawal
    tr_r6 = sim(k, tka, dP, ramp_dur=0.6)
    tr_f = sim(k, tka, None)                     # no dP at all
    dc, dr, dr6, df = d_series(tr_c)[0].numpy(), d_series(tr_r)[0].numpy(), \
        d_series(tr_r6)[0].numpy(), d_series(tr_f)[0].numpy()
    t = ti.numpy()
    tc, tf = rec['tc'], rec['tf']
    typ = ['gen', 'load', 'line'][rec['typ'] - 1]

    # post-clearing excursion drives the scale; clip runaway free-run trace
    i_tc = int(round(tc / DT))
    post_pk = float(dc[i_tc:].max())
    post_at = float(ti[i_tc + dc[i_tc:].argmax()])
    cap = max(post_pk, float(dc.max())) * 1.4 + 0.02
    df_p = np.minimum(df, cap)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.axvspan(tf, tc, color='#ffe9b0', alpha=0.5, zorder=0)
    ax.axhline(LIMIT, color='k', ls=':', lw=1)
    ax.plot(t, df_p, color='#888888', lw=0.7, ls='--',
            label='no dP (uncontrolled; clipped)')
    ax.plot(t, dc, color='#c0392b', lw=1.6, label='dP held to clear, step-off (judged)')
    ax.plot(t, dr, color='#2e86de', lw=1.3, label='dP ramped off over 0.3 s')
    ax.plot(t, dr6, color='#27ae60', lw=1.0, alpha=0.9, label='dP ramped off over 0.6 s')
    pk = rec['pk_run']
    ax.set_ylim(0, cap)
    ax.axhline(-LIMIT, color='k', ls=':', lw=1)
    ax.axvline(tc, color='#c0392b', lw=1.0, ls='-.')
    ax.annotate('t_clear', xy=(tc, ax.get_ylim()[1] * 0.92), fontsize=8,
                color='#c0392b', ha='left', va='top')
    ax.annotate(f"withdrawal transient peak {post_pk:.3f} pu\n"
                f"at t_clear +{post_at - tc:.2f} s",
                xy=(post_at, post_pk * 0.7), fontsize=8, ha='left')
    ax.text(0.02, 0.97,
            f"{title}\nfault {typ}, alpha {rec['alpha']:.2f}, "
            f"||dP|| {rec['dp_norm']:.2f} pu\n"
            f"in-window peak {rec['pk_win']:.4f} pu (limit 0.005)",
            transform=ax.transAxes, va='top', fontsize=8)
    ax.set_xlabel('t (s)')
    ax.set_ylabel(r'max$_j$ |\omega_j - 1| (pu)')
    ax.legend(loc='upper right', fontsize=7)
    ax.set_xlim(0, float(t_grid[-1]))
    fig.tight_layout()
    path = f'{OUT}/{tag}.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")
    return dc, dr, dr6, df

# clearest kick: soon exit, big overshoot, full recovery, CLEAN onset
# (reject cases whose dP-onset step spikes during the grace window and
#  dominates the scale -- choose on the POST-CLEARING excursion instead)
def case_probe(rec):
    b, dP, vt = pick_case(rec)
    k = np.array([rec['k']])
    tka = np.array([rec['tk']])
    tr_c = sim(k, tka, dP)
    d = d_series(tr_c)[0].numpy()
    i_tk, i_tf, i_tc = [int(round(x / DT)) for x in
                        (rec['tk'], rec['tf'], rec['tc'])]
    i_g = min(i_tk + int(GRACE / DT), len(d) - 1)
    onset_max = float(d[i_tk:i_g].max())            # dP-onset spike (grace-masked)
    post = d[max(i_tc, i_tk + int(GRACE / DT)):]
    i_start = max(i_tc, i_tk + int(GRACE / DT))
    post_pk = float(post.max())
    post_at = float(ti[i_start + post.argmax()]) if len(post) else float('nan')
    return onset_max, post_pk, post_at - rec['tc']

kick = sorted([r for r in lost if 0.05 <= r['exit_after_clear'] <= 0.30],
              key=lambda r: -r['pk_run'])[:10] or sorted(lost, key=lambda r: -r['pk_run'])
scored = []
for cand in kick:
    onset_max, post_pk, post_at = case_probe(cand)
    scored.append((cand, onset_max, post_pk, post_at))
    print(f"  cand k={cand['k']} type={['gen','load','line'][cand['typ']-1]}"
          f" alpha={cand['alpha']:.2f} ||dP||={cand['dp_norm']:.2f}"
          f" onset_max={onset_max:.4f} post_pk={post_pk:.4f} at +{post_at:.3f} s"
          f" exit={cand['exit_after_clear']:.3f} s")
kick[0] = sorted(scored, key=lambda s: (s[1] > 0.02, -s[2]))[0][0]
print(f"chosen kick case: k={kick[0]['k']} (largest post-clearing excursion,"
      f" clean onset)")
bound = min(lost, key=lambda r: r['exit_after_clear'])
plot_case(kick[0], 'end_mech_kick',
          'KICK: rescued in-window, exits band right after the step-off, rings back in')
plot_case(bound, 'end_mech_boundary',
          'BOUNDARY: exit after judged window ends but before clearing')

# ── 3. exit-time histogram ───────────────────────────────────────────────────
et = np.array([r['exit_after_clear'] for r in lost])
fig, ax = plt.subplots(figsize=(6.5, 3.6))
bins = np.arange(-0.45, 0.45, 0.03)
ax.hist(et[et >= 0], bins=bins, color='#c0392b', alpha=0.75,
        label='post-clearing exits (52)')
ax.hist(et[et < 0], bins=bins, color='#8e44ad', alpha=0.75,
        label='exits after window ends, pre-clear')
ax.axvline(0, color='k', lw=1.2)
ax.annotate('t_clear', xy=(0, ax.get_ylim()[1] * 0.95), fontsize=8,
            ha='right', va='top')
ax.axvline(np.median(et[et >= 0]), color='#c0392b', ls=':', lw=1.5)
ax.annotate(f"median +{np.median(et[et >= 0]):.2f} s", fontsize=8,
            xy=(np.median(et[et >= 0]), ax.get_ylim()[1] * 0.8), ha='left')
ax.set_xlabel('first band-exit time after t_clear (s)')
ax.set_ylabel('cases')
ax.set_title('All 52 ring-down losses exit within +-0.4 s of clearing — '
             'withdrawal-step transient', fontsize=9)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f'{OUT}/end_mech_exit_hist.png', dpi=150)
print(f"  wrote {OUT}/end_mech_exit_hist.png")

recs['ramp_result'] = {f'{"%.1f" % k}': v for k, v in res.items()}
recs['n_lost'] = len(lost)
recs['n_pre_clear'] = len(pre_clear)
json.dump(recs, open(f'{OUT}/end_mech_records.json', 'w'), indent=1)
print("summary:", recs['ramp_result'], "lost", len(lost),
      "pre-clear exits", len(pre_clear))
