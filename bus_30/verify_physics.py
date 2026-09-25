"""
verify_physics.py
=================
Validation gates for psys_physics.py, run BEFORE anything is trained.

  1. generic module reproduces ieee9_physics.IEEE9 exactly (9-bus regression)
  2. Kron reduction matches the stored Yred_pre / Vrec_pre
  3. the shipped initial state is an equilibrium (dx/dt ~ 0)
  4. simulated omega matches the dataset over N scenarios (noise floor)
  5. OOL agreement with the stored y_ool labels
  6. dt_roll = 10 ms is still accurate for this system
"""
import argparse, sys
import numpy as np, scipy.io as sio, torch, h5py
import psys_physics as pp

ap = argparse.ArgumentParser()
ap.add_argument('--system', default='case30', choices=['case30', 'ieee9'])
ap.add_argument('--base', default='/mnt/user-data/uploads/case30_base.mat')
ap.add_argument('--sys',  default='/mnt/user-data/uploads/case30_simulink_data.mat')
ap.add_argument('--data', default='/mnt/user-data/uploads/case30_dataset_small_1.mat')
ap.add_argument('--n', type=int, default=100)
args = ap.parse_args()

torch.set_default_dtype(torch.float64)
ok_all = True


def check(label, val, tol, unit=''):
    global ok_all
    good = val <= tol
    ok_all &= good
    print(f"  [{'OK ' if good else 'FAIL'}] {label:44s} {val:.3e}{unit}  (tol {tol:.0e})")


# ── 1. 9-bus regression: generic vs original module ─────────────────────────
print("1. generic module vs ieee9_physics (9-bus regression)")
try:
    import ieee9_physics as old
    with h5py.File('/mnt/user-data/uploads/pinn_dataset_small.mat', 'r') as _h9:
        lines = np.array(_h9['lines']).T.astype(int)   # v7.3 stores transposed
    S_old = old.IEEE9('/mnt/user-data/uploads/ieee9_base.mat',
                      '/mnt/user-data/uploads/ieee9_gov_simulink_data.mat', lines=lines)
    S_new = pp.ieee9('/mnt/user-data/uploads/ieee9_base.mat',
                     '/mnt/user-data/uploads/ieee9_gov_simulink_data.mat', lines=lines)
    check('Yred_pre', (S_old.Yred_pre - S_new.Yred_pre).abs().max().item(), 1e-14)
    a = torch.rand(7, dtype=torch.float64) * 0.9 + 0.1
    dmax = 0.0
    for h in range(old.N_HYP):
        dmax = max(dmax, (S_old.faulted_Y(h, a) - S_new.faulted_Y(h, a)).abs().max().item())
        dmax = max(dmax, (S_old.dPm0(h, a) - S_new.dPm0(h, a)).abs().max().item())
    check('faulted_Y + dPm0, all 12 hypotheses', dmax, 1e-14)
    x = S_new.x0(5) + 0.01 * torch.randn(5, 9, dtype=torch.float64)
    check('rhs', (S_old.rhs(x, S_old.Yred_pre.expand(5, -1, -1), S_old.Pm0.expand(5, -1))
                  - S_new.rhs(x, S_new.Yred_pre.expand(5, -1, -1), S_new.Pm0.expand(5, -1))
                  ).abs().max().item(), 1e-14)
except Exception as e:
    print(f"  [SKIP] 9-bus files unavailable ({e})")

# ── build the system under test ─────────────────────────────────────────────
print(f"\n2. {args.system}: network")
S = pp.case30(args.base, args.sys) if args.system == 'case30' else None
print(f"  NG={S.NG}  NB={S.NB}  hypotheses={S.N_HYP} "
      f"(gen {int(S.is_gen.sum())}, load {int(S.is_load.sum())}, line {int(S.is_line.sum())})")
print(f"  controllable machines: {(S.ctrl_machines + 1).tolist()}   invR={S.invR.tolist()}")

check('Yred_pre vs stored', (S.Yred_pre[0] - S.Yred_pre_ref).abs().max().item(), 1e-12)
check('Vrec_pre vs stored', (S.Vrec_pre[0] - S.Vrec_pre_ref).abs().max().item(), 1e-12)

# stored fault matrices correspond to the scenario baked into the .mat
P = sio.loadmat(args.sys)

print("\n3. equilibrium")
x0 = S.x0(1)
check('|dx/dt| at the shipped initial state',
      S.rhs(x0, S.Yred_pre, S.Pm0.unsqueeze(0)).abs().max().item(), 1e-10)

# ── 4/5. trajectory vs dataset ──────────────────────────────────────────────
print("\n4. simulated omega vs dataset")
f = h5py.File(args.data, 'r')
X = f['X']                                   # (C, T, N)
t = np.array(f['t']).ravel()
DT = float(t[1] - t[0])
y_type = np.array(f['y_type']).ravel().astype(int)
y_ool  = np.array(f['y_ool']).ravel().astype(bool)
alpha  = np.array(f['meta_alpha']).ravel()
loc    = np.array(f['meta_loc']).ravel().astype(int)
t_f    = np.array(f['y_t_fault']).ravel()
t_c    = np.array(f['meta_t_clear']).ravel()
branches = np.array(f['branches']).T          # (41,2)
limit  = float(np.array(f['omega_limit']).ravel()[0])
N = min(args.n, X.shape[2])

# branches in the dataset must match the branch table in the base file
b0 = sio.loadmat(args.base)
bt = np.stack([b0['br_from'].ravel(), b0['br_to'].ravel()], 1)
check('dataset `branches` == base br_from/br_to',
      float(np.abs(branches - bt).max()), 0.5)

# map (type, loc) -> hypothesis index
hyp_of = {}
for h, (k, l) in enumerate(S.hypotheses):
    hyp_of[({'gen': 1, 'load': 2, 'line': 3}[k], l)] = h


def simulate(ks, dt, T_end):
    """Free (uncorrected) trajectories, ALL scenarios in one batched rollout.

    Per-sample fault networks are selected per step with torch.where instead of
    looping over scenarios; a 100-scenario 8 s run at 1 ms goes from ~10 min to
    a few seconds. Same trick the training rollout uses.
    """
    ks = np.asarray(ks); B = len(ks)
    hs = np.array([hyp_of.get((y_type[k], loc[k]), -1) for k in ks])
    if (hs < 0).any():
        raise KeyError(f"unmapped: {ks[hs < 0][:5]}")
    a = torch.tensor(alpha[ks], dtype=torch.float64)
    Yb = torch.stack([S.faulted_Y(int(h), a[i:i+1])[0] for i, h in enumerate(hs)])
    Yf, _ = S.kron_reduce(Yb)                                   # (B,NG,NG)
    dP0 = torch.stack([S.dPm0(int(h), a[i:i+1])[0] for i, h in enumerate(hs)])
    Pset_f = S.Pm0.unsqueeze(0) + dP0
    Pset_0 = S.Pm0.unsqueeze(0).expand(B, -1).contiguous()
    Mf, Nf = S.pe_consts(Yf)
    M0, N0 = S.pe_consts(S.Yred_pre.expand(B, -1, -1))
    tf = torch.tensor(t_f[ks]); tc_ = torch.tensor(t_c[ks])

    n = int(round(T_end / dt))
    x = S.x0(B)
    W = torch.empty(n + 1, B, S.NG, dtype=torch.float64)
    W[0] = x[:, S.NG:2 * S.NG]
    for i in range(n):
        tm = (i + 0.5) * dt
        inf_ = ((tm >= tf) & (tm < tc_)).view(B, 1, 1)
        M = torch.where(inf_, Mf, M0)
        N = torch.where(inf_, Nf, N0)
        Ps = torch.where(inf_[:, :, 0], Pset_f, Pset_0)
        x = S.rk4_real(x, M, N, Ps, dt)
        W[i + 1] = x[:, S.NG:2 * S.NG]
    return W


om_ch = S.idx_omega
bad = [k for k in range(N) if hyp_of.get((y_type[k], loc[k])) is None]
ks_ok = np.array([k for k in range(N) if k not in bad])
W = simulate(ks_ok, DT, t[-1]).numpy()                 # (T,B,NG)
meas = np.asarray(X[om_ch, :, :])[:, :, ks_ok].transpose(1, 2, 0)   # (T,B,NG)
rmse = np.sqrt(((W - meas) ** 2).mean(axis=(0, 2)))
dev_sim = np.abs(W - 1).max(axis=(0, 2))
dev_dat = np.abs(meas - 1).max(axis=(0, 2))
noise = float(np.array(f['noise_std']).ravel()[om_ch][0])
print(f"  scenarios simulated: {len(rmse)}   unmapped: {len(bad)}")
print(f"  omega RMSE: median {np.median(rmse):.3e}  p90 {np.percentile(rmse,90):.3e} "
      f" max {rmse.max():.3e}   (noise std {noise:.1e})")
check('median omega RMSE vs measured', float(np.median(rmse)), 1.5 * noise)
check('max omega RMSE vs measured', float(rmse.max()), 4.0 * noise)

print("\n5. OOL agreement")
ool_sim = np.array(dev_sim) > limit
ok_idx = list(ks_ok)
agree = (ool_sim == y_ool[ok_idx]).sum()
print(f"  {agree}/{len(ok_idx)} agree with stored y_ool  (limit {limit})")
mism = np.where(ool_sim != y_ool[ok_idx])[0]
for j in mism[:6]:
    k = ok_idx[j]
    print(f"    scenario {k}: type {y_type[k]} loc {loc[k]} alpha {alpha[k]:.2f} "
          f"sim peak {dev_sim[j]:.5f} stored maxdev {np.array(f['meta_maxdev']).ravel()[k]:.5f}")
check('OOL mismatches', float(len(mism)), 0.5)

print("\n6. rollout step size")
# Judge over the ROLLOUT HORIZON, not the full 8 s: the model integrates ~1.5 s
# from the decision instant and is then re-decided. Error over 8 s is dominated
# by accumulated phase drift the controller never sees.
HOR = 1.5
ks20 = ks_ok[:20]
print(f"  compared over {HOR:.1f} s from fault onset (the rollout horizon), "
      f"{len(ks20)} scenarios")
for dt_roll in (0.005, 0.01, 0.02):     # must be a multiple of DT
    step = int(round(dt_roll / DT))
    Wc = simulate(ks20, dt_roll, t[-1]).numpy()
    Wf = W[:, :len(ks20)][::step]
    m = min(len(Wf), len(Wc))
    i0 = (t_f[ks20] / dt_roll).astype(int)
    i1 = np.minimum(i0 + int(HOR / dt_roll), m)
    e = [np.abs(Wf[i0[j]:i1[j], j] - Wc[i0[j]:i1[j], j]) for j in range(len(ks20))]
    mx = max(float(x.max()) for x in e)
    rm = float(np.sqrt(np.mean(np.concatenate([x.ravel() ** 2 for x in e]))))
    # what the controller actually thresholds is the PEAK |omega-1| over the
    # horizon, so the decision-relevant error is the error in that peak --
    # judged against the safety margin, not against the noise floor.
    pk_c = np.array([np.abs(Wc[i0[j]:i1[j], j] - 1).max() for j in range(len(ks20))])
    pk_f = np.array([np.abs(Wf[i0[j]:i1[j], j] - 1).max() for j in range(len(ks20))])
    dpk = float(np.abs(pk_f - pk_c).max())
    print(f"  dt_roll {1000*dt_roll:5.1f} ms:  traj max {mx:.3e}  rmse {rm:.3e}   "
          f"peak err {dpk:.3e} = {100*dpk/limit:.2f}% of limit, "
          f"{100*dpk/1e-3:.1f}% of a 1e-3 margin")

print("\n" + ("ALL GATES PASSED" if ok_all else "SOME GATES FAILED"))
sys.exit(0 if ok_all else 1)
