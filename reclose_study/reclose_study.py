"""
reclose_study.py
================
Fixed-time vs synchronism-check reclosing of a temporarily opened line, on the
9-bus and 30-bus classical models already used for the dataset (no dP, no other
fault). Standalone: it reuses the physics ports and changes nothing else.

Event: from pre-fault equilibrium, open one faultable line at T_OPEN (exactly as
the dataset's 'line' event does: bus admittance with that branch removed, Kron
reduced). The line is always fully open or fully closed.

  A  (fixed)  close at T_OPEN + d, whatever the grid state  (= current simulator)
  B  (check)  from T_OPEN + d on, close at the first step where, across the open
              line,  |angle(V_f) - angle(V_t)| <= ang_max  AND
                     ||V_f| - |V_t||           <= dv_max.
              If not met within max_wait, the line stays open for the run.

Bus voltages during the outage: V_bus = Vrec_open @ E(delta), the same recovery
matrix the Kron reduction returns. Thresholds are study parameters, not values
from the repo.

Per case: angle / magnitude difference and slip at closing, per-machine Pe jump
at the switching instant (same delta, closed minus open network), peak |w-1|
in the POST window after closing (max machine, COI, inter-machine), band
(0.005 pu) violations, wait beyond the timer, never-closed share.
"""
import argparse, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ap = argparse.ArgumentParser()
ap.add_argument('--grid', choices=['9', '30'], required=True)
ap.add_argument('--durations', default='0.1,0.25,0.5,1,2,3')
ap.add_argument('--ang_max', default='10,20,30', help='degrees, swept')
ap.add_argument('--dv_max', type=float, default=0.05, help='pu')
ap.add_argument('--slip_max', type=float, default=float('inf'), help='Hz, |d(dtheta)/dt|/2pi')
ap.add_argument('--max_wait', type=float, default=5.0, help='s after the timer')
ap.add_argument('--post', type=float, default=3.0, help='s after closing')
ap.add_argument('--t_open', type=float, default=0.5)
ap.add_argument('--h', type=float, default=0.001)
ap.add_argument('--limit', type=float, default=0.005)
ap.add_argument('--out', default=None)
args = ap.parse_args()

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
f64 = torch.float64

# ── grid ─────────────────────────────────────────────────────────────────────
if args.grid == '30':
    sys.path.insert(0, os.path.join(ROOT, 'bus_30'))
    from case30_physics import CASE30
    S = CASE30(os.path.join(ROOT, 'bus_30', 'case30_base.mat'),
               os.path.join(ROOT, 'bus_30', 'case30_simulink_data.mat'),
               device=dev, dtype=f64)
    br = [int(i) for i in S.faultable]                        # 0-based rows
    ends = [(S.br_from[i] - 1, S.br_to[i] - 1) for i in br]
    names = [f'br{i+1} ({S.br_from[i]}-{S.br_to[i]})' for i in br]
    Yopen = torch.stack([S._open_branch(S.Y_pre, i) for i in br])
    Yred_pre, Vrec_pre = S.kron(S.Y_pre)
    Yred_o, Vrec_o = S.kron(Yopen)
    x0, Pm0 = S.x0, S.Pm0
    H = S.H
    NG = 6
    step = lambda x, Yr, Ps, h: S.rk4(x, Yr, Ps, h)
    Ec = S.Ec
else:
    sys.path.insert(0, os.path.join(ROOT, 'bus_9'))
    from ieee9_physics import IEEE9
    md = np.load(os.path.join(ROOT, 'bus_9', 'data_mm', 'meta.npz'))
    lines = md['lines'].astype(int)
    S = IEEE9(os.path.join(ROOT, 'bus_9', 'ieee9_base.mat'),
              os.path.join(ROOT, 'bus_9', 'ieee9_gov_simulink_data.mat'),
              dtype=f64, device=dev, lines=lines)
    ends = [(r - 1, s - 1) for r, s in lines]
    names = [f'line{k+1} ({r}-{s})' for k, (r, s) in enumerate(lines)]
    idx = 6 + torch.arange(len(lines))                         # HYPOTHESES rows
    Yopen = torch.cat([S.faulted_Y(int(i), torch.ones(1, dtype=f64, device=dev))
                       for i in idx])
    Yred_pre, Vrec_pre = S.kron_reduce(S.Y_pre.unsqueeze(0))
    Yred_pre, Vrec_pre = Yred_pre[0], Vrec_pre[0]
    Yred_o, Vrec_o = S.kron_reduce(Yopen)
    x0 = torch.cat([S.delta0, torch.ones(3, dtype=f64, device=dev), S.Pm0])
    Pm0, H, NG = S.Pm0, S.H, 3
    step = lambda x, Yr, Ps, h: S.rk4(x, Yr, Ps, h)
    Ec = S.Ec

NL = len(ends)
durs = [float(v) for v in args.durations.split(',')]
angs = [float(v) for v in args.ang_max.split(',')]
pols = [('A', None)] + [(f'B{a:g}', a) for a in angs] + [('open', -1.0)]  # open = never reclose (reference)

# case table: (line, duration, policy)
cases = [(l, d, p) for p in range(len(pols)) for d in durs for l in range(NL)]
n = len(cases)
L = torch.tensor([c[0] for c in cases], device=dev)
Dd = torch.tensor([c[1] for c in cases], dtype=f64, device=dev)
amax = torch.tensor([np.inf if pols[c[2]][1] is None else (-1.0 if pols[c[2]][1] < 0 else np.deg2rad(pols[c[2]][1]))
                     for c in cases], dtype=f64, device=dev)
is_A = torch.tensor([pols[c[2]][1] is None for c in cases], device=dev)
fr = torch.tensor([ends[c[0]][0] for c in cases], device=dev)
to = torch.tensor([ends[c[0]][1] for c in cases], device=dev)

Yo_c, Vo_c = Yred_o[L], Vrec_o[L]                              # (n,NG,NG),(n,NB,NG)
Ypre_c = Yred_pre.unsqueeze(0).expand(n, NG, NG)
Hs = H / H.sum()

h = args.h
T_end = args.t_open + max(durs) + args.max_wait + args.post + 0.01
N = int(round(T_end / h))
t_timer = args.t_open + Dd

x = x0.unsqueeze(0).repeat(n, 1)
Ps = Pm0.unsqueeze(0).expand(n, NG)
closed = torch.ones(n, dtype=torch.bool, device=dev)          # before opening
opened = False
t_close = torch.full((n,), np.nan, dtype=f64, device=dev)
nwin = torch.zeros(n, dtype=f64, device=dev)
fail = {k: torch.zeros(n, dtype=f64, device=dev) for k in ('ang', 'mag', 'freq')}
sole = {k: torch.zeros(n, dtype=f64, device=dev) for k in ('ang', 'mag', 'freq')}
ever = {k: torch.zeros(n, dtype=torch.bool, device=dev) for k in ('ang', 'mag', 'freq')}
rec = {k: torch.full((n,), np.nan, dtype=f64, device=dev)
       for k in ['dang', 'dv', 'slip', 'dpe_max', 'dpe_sum', 'vf', 'vt']}
dpe_vec = torch.full((n, NG), np.nan, dtype=f64, device=dev)
post_pk = torch.zeros(n, dtype=f64, device=dev)               # max machine |w-1|
post_coi = torch.zeros(n, dtype=f64, device=dev)
post_rel = torch.zeros(n, dtype=f64, device=dev)              # max |w_i - w_coi|
pre_pk = torch.zeros(n, dtype=f64, device=dev)                # during the outage
pre_rel = torch.zeros(n, dtype=f64, device=dev)
open_pk = torch.zeros(n, dtype=f64, device=dev)               # whole run, never-closed
ar = torch.arange(n, device=dev)


def Evec(x):
    return Ec.unsqueeze(0) * torch.exp(1j * x[:, :NG].to(Ec.dtype))


def Pe(x, Yr):
    E = Evec(x)
    return torch.real(E * torch.conj(torch.einsum('bij,bj->bi', Yr, E)))


def across(x):
    V = torch.einsum('bij,bj->bi', Vo_c, Evec(x))              # bus voltages, open net
    vf, vt = V[ar, fr], V[ar, to]
    da = torch.angle(vf * torch.conj(vt))                      # wrapped to (-pi,pi]
    return da, (vf.abs() - vt.abs()), vf.abs(), vt.abs()


da_prev = None
for k in range(N):
    t = k * h
    if not opened and t >= args.t_open - 1e-12:
        closed[:] = False
        opened = True
    if opened:
        da, dv, vf, vt = across(x)
        if da_prev is None:
            da_prev = da.clone()
        slip = torch.remainder(da - da_prev + np.pi, 2 * np.pi) - np.pi
        slip = slip / h                                         # rad/s, wrap-safe
        due = (~closed) & torch.isnan(t_close) & (t >= t_timer - 1e-12) \
            & (t <= t_timer + args.max_wait + 1e-12)
        ok_a = da.abs() <= amax
        ok_v = dv.abs() <= args.dv_max
        ok_f = slip.abs() <= 2 * np.pi * args.slip_max
        ok = is_A | (ok_a & ok_v & ok_f)
        win = due & ~is_A
        nwin += win.double()
        for key, okk, o1, o2 in (('ang', ok_a, ok_v, ok_f), ('mag', ok_v, ok_a, ok_f),
                                 ('freq', ok_f, ok_a, ok_v)):
            fail[key] += (win & ~okk).double()
            sole[key] += (win & ~okk & o1 & o2).double()
            ever[key] |= (win & okk)
        go = due & ok
        if go.any():
            jump = Pe(x, Ypre_c) - Pe(x, Yo_c)
            rec['dang'][go] = da[go]
            rec['dv'][go] = dv[go]
            rec['slip'][go] = slip[go]
            rec['vf'][go], rec['vt'][go] = vf[go], vt[go]
            dpe_vec[go] = jump[go]
            rec['dpe_max'][go] = jump[go].abs().max(1).values
            rec['dpe_sum'][go] = jump[go].abs().sum(1)
            t_close[go] = t
            closed[go] = True
        da_prev = da
    # frequency bookkeeping
    w = x[:, NG:2 * NG] - 1.0
    wc = (w * Hs).sum(1)
    pk = w.abs().max(1).values
    rel = (w - wc.unsqueeze(1)).abs().max(1).values
    in_post = (~torch.isnan(t_close)) & (t >= t_close) & (t <= t_close + args.post + 1e-12)
    post_pk = torch.where(in_post, torch.maximum(post_pk, pk), post_pk)
    post_coi = torch.where(in_post, torch.maximum(post_coi, wc.abs()), post_coi)
    post_rel = torch.where(in_post, torch.maximum(post_rel, rel), post_rel)
    if opened:
        out_now = ~closed
        pre_pk = torch.where(out_now, torch.maximum(pre_pk, pk), pre_pk)
        pre_rel = torch.where(out_now, torch.maximum(pre_rel, rel), pre_rel)
    open_pk = torch.maximum(open_pk, pk)
    Yr = torch.where(closed.view(n, 1, 1), Ypre_c, Yo_c)
    x = step(x, Yr, Ps, h)

da_end, dv_end, _, _ = across(x)
E0 = Ec * torch.exp(1j * x0[:NG].to(Ec.dtype))
V0 = Vrec_pre @ E0
ref_closed = [(float(np.rad2deg(torch.angle(V0[f] * torch.conj(V0[t_])).item())),
               float((V0[f].abs() - V0[t_].abs()).item())) for f, t_ in ends]
# ── tables ───────────────────────────────────────────────────────────────────
c = lambda v: v.detach().cpu().numpy()
tc, tt = c(t_close), c(t_timer)
res = dict(grid=args.grid, h=h, t_open=args.t_open, dv_max=args.dv_max, slip_max=args.slip_max,
           max_wait=args.max_wait, post=args.post, limit=args.limit,
           durations=durs, policies=[p for p, _ in pols], lines=names, ref_closed=ref_closed, cases=[])
for i, (l, d, p) in enumerate(cases):
    res['cases'].append(dict(
        line=names[l], dur=d, policy=pols[p][0],
        closed=bool(np.isfinite(tc[i])),
        wait=float(tc[i] - tt[i]) if np.isfinite(tc[i]) else None,
        dang_deg=float(np.rad2deg(c(rec['dang'])[i])),
        dv=float(c(rec['dv'])[i]), slip_deg_s=float(np.rad2deg(c(rec['slip'])[i])),
        dpe=c(dpe_vec)[i].tolist(),
        dpe_max=float(c(rec['dpe_max'])[i]), dpe_sum=float(c(rec['dpe_sum'])[i]),
        post_pk=float(c(post_pk)[i]), post_coi=float(c(post_coi)[i]),
        post_rel=float(c(post_rel)[i]), pre_pk=float(c(pre_pk)[i]),
        pre_rel=float(c(pre_rel)[i]), run_pk=float(c(open_pk)[i]),
        dang_end_deg=float(np.rad2deg(c(da_end)[i])), dv_end=float(c(dv_end)[i]),
        win_steps=int(c(nwin)[i]),
        fail_frac={k: float(c(fail[k])[i] / max(c(nwin)[i], 1)) for k in fail},
        sole_frac={k: float(c(sole[k])[i] / max(c(nwin)[i], 1)) for k in sole},
        ever_pass={k: bool(c(ever[k])[i]) for k in ever}))

out = args.out or os.path.join(HERE, f'out_{args.grid}bus')
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, 'results.json'), 'w') as f:
    json.dump(res, f, indent=1)

# summary
R = res['cases']
lim = args.limit


def q(v, p):
    return float(np.percentile(v, p)) if len(v) else float('nan')


print(f'{args.grid}-bus: {NL} lines x {len(durs)} durations, dv_max {args.dv_max}, slip_max {args.slip_max} Hz, '
      f'max_wait {args.max_wait}s, post {args.post}s, h {h}')
print('across-line reference: closed pre-fault (dang, dV) | open steady state (dang, dV) | open-only run peak |w-1|')
for l in range(NL):
    ro = [r for r in R if r['policy'] == 'open' and r['line'] == names[l]][0]
    print(f'  {names[l]:16} {ref_closed[l][0]:7.1f} {ref_closed[l][1]:7.3f} | '
          f'{ro["dang_end_deg"]:7.1f} {ro["dv_end"]:7.3f} | {ro["run_pk"]:.4f}')
hdr = f'{"policy":7} {"closed":>8} {"wait med/max":>13} {"|dang| med/max":>15} {"dPe max med/max":>16} {"post pk med/max":>16} {"post>lim":>8} {"rel med/max":>14}'
print(hdr)
fmt = lambda v, f: (f % v) if np.isfinite(v) else '   -  '
for pn, _ in pols[:-1]:
    rr = [r for r in R if r['policy'] == pn]
    cl = [r for r in rr if r['closed']]
    a = lambda k: np.array([r[k] for r in cl], dtype=float)
    mx = lambda v: float(v.max()) if len(v) else float('nan')
    da = np.abs(a('dang_deg'))
    viol = sum(r['post_pk'] > lim for r in cl)
    print(f'{pn:7} {len(cl):3}/{len(rr):<4} {fmt(q(a("wait"),50),"%6.2f")}/{fmt(mx(a("wait")),"%5.2f")} '
          f'{fmt(q(da,50),"%7.1f")}/{fmt(mx(da),"%6.1f")} {fmt(q(a("dpe_max"),50),"%8.3f")}/{fmt(mx(a("dpe_max")),"%6.3f")} '
          f'{fmt(q(a("post_pk"),50),"%8.4f")}/{fmt(mx(a("post_pk")),"%6.4f")} {viol:8} '
          f'{fmt(q(a("post_rel"),50),"%7.4f")}/{fmt(mx(a("post_rel")),"%6.4f")}')
print(f'saved {out}')
