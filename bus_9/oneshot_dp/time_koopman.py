"""
time_koopman.py -- end-to-end per-decision latency of the governor-channel
Koopman MPC (koopman_baseline.py gov_fullchan), one scenario at a time.

Closed loop identical to koopman_closed_loop_batch with n = 1. Timed span per
decision (every 20 ms): Kalman-filter update on the newest noisy measurement
-> lift phi(xhat) -> 20-step free omega prediction -> QP bound update ->
warm OSQP solve -> first-cell action extracted. The three KF updates between
decisions run every 5 ms but are not on the decision path; their cost is
reported separately. CPU/numpy, as the baseline runs.
"""
import json, os, sys, time
import numpy as np
import torch

BUS9 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BUS9)
import koopman_baseline as kb                                     # noqa: E402
import osqp                                                       # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'koopman_rerun')
N_SC = int(sys.argv[1]) if len(sys.argv) > 1 else 40
WARM_DEC = 10                     # decisions per scenario discarded as warm-up

d, S = kb.load_all()
lift = kb.GovLift(S, OUT); m = 9
K = kb.make_kf(lift, m, S)['K']
qp = kb.build_qp_layout(lift, lift.A, lift.B)
A, B = lift.A, lift.B
H, DEC, NDEC = kb.H, kb.DEC_EVERY, kb.N_DEC
settings = dict(verbose=False, eps_abs=1e-6, eps_rel=1e-6, max_iter=20000, polishing=True)
l0, u0 = qp['l0'], qp['u0']
rp = np.zeros((H, 3), int); rm = np.zeros((H, 3), int)
for (kk, mm, a, b) in qp['hinge_rows']:
    rp[kk, mm] = a; rm[kk, mm] = b
rp, rm = rp.reshape(-1), rm.reshape(-1)
tp = np.array([p for (p, _) in qp['term_rows']]); tmi = np.array([q for (_, q) in qp['term_rows']])
r0 = [r for (c, mm, r) in qp['rate_rows'] if c == 0]
wsl = lift.omega_rows()

inst = json.load(open(kb.INSTANTS_DEFAULT))
uk = np.unique(np.asarray(inst['ks'][:2048], int))
sc = np.random.default_rng(0).choice(uk, N_SC, replace=False)
dec_ms, kf_ms = [], []
for k in sc:
    Yf, dPm = kb.fault_arrays_row(S, d, np.array([k]))
    Mp, Np = S.pe_consts(S.Yred_pre.expand(1, 3, 3)); Mf, Nf = S.pe_consts(Yf)
    tf, tc = float(d['t_fault'][k]), float(d['t_clear'][k])
    x = torch.cat([S.delta0, torch.ones(3), S.Pm0]).unsqueeze(0).clone()
    xhat = x.numpy()[:, :m].copy(); u_prev = np.zeros(3); u_cur = np.zeros(3)
    rng = np.random.default_rng(42)
    so = osqp.OSQP(); so.setup(P=qp['P'], q=qp['q'], A=qp['A'], l=l0, u=u0, **settings)
    T = len(d['t_grid']) - 1; nd = 0
    for t in range(T):
        tm = t * kb.DT + kb.DT / 2
        act = tf <= tm < tc
        Uapp = u_cur * (tm < tc)
        Ps = S.Pm0.unsqueeze(0) + dPm * float(act) + torch.as_tensor(Uapp[None], dtype=S.rdtype)
        x = S.rk4_real(x, Mf if act else Mp, Nf if act else Np, Ps, kb.DT)
        y = x.numpy()[:, :m] + rng.normal(0.0, kb.NOISE_STD[:m], (1, m))
        is_dec = (t + 1) % DEC == 0 and (t + 1) <= T - DEC
        t0 = time.perf_counter()
        Zp = lift.phi(xhat) @ A.T + Uapp[None] @ B.T          # KF predict
        xp = lift.phys_rows(Zp)
        xhat = xp + (y - xp) @ K.T                             # KF correct
        if not is_dec:
            kf_ms.append((time.perf_counter() - t0) * 1e3)
            continue
        z = lift.phi(xhat)[0]
        of = np.empty((H, 3))
        for i in range(H):
            z = A @ z
            of[i] = z[wsl] - 1.0
        off = of.reshape(-1)
        lu = l0.copy(); uu = u0.copy()
        lu[rp] = off - kb.QP_DB; lu[rm] = -off - kb.QP_DB
        lu[tp] = of[-1]; lu[tmi] = -of[-1]
        for mm in range(3):
            lu[r0[mm]] = -kb.QP_RATE + u_prev[mm]; uu[r0[mm]] = kb.QP_RATE + u_prev[mm]
        so.update(l=lu, u=uu)
        res = so.solve()
        ucell = np.clip(res.x[:3 * NDEC].reshape(NDEC, 3)[0], -kb.QP_UMAX, kb.QP_UMAX)
        t1 = time.perf_counter()
        if nd >= WARM_DEC:
            dec_ms.append((t1 - t0) * 1e3)
        u_prev = ucell.copy(); u_cur = ucell.copy(); nd += 1
    print(f'sc {k}: {nd} decisions', flush=True)

dec_ms, kf_ms = np.array(dec_ms), np.array(kf_ms)
res = dict(n_scenarios=int(N_SC), n_decisions=int(len(dec_ms)),
           decision_median_ms=float(np.median(dec_ms)), decision_p95_ms=float(np.percentile(dec_ms, 95)),
           decision_mean_ms=float(dec_ms.mean()), decision_max_ms=float(dec_ms.max()),
           kf_update_median_ms=float(np.median(kf_ms)), kf_update_p95_ms=float(np.percentile(kf_ms, 95)),
           osqp_version=osqp.__version__)
json.dump(res, open(os.path.join(os.path.dirname(OUT), 'latency_koopman.json'), 'w'), indent=1)
print(json.dumps(res, indent=1))
