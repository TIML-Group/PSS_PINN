"""
oracle_case30.py
================
What trajectory error would a PERFECT identifier get on this dataset?

    python oracle_case30.py --data data30_mm --n 512

The network is removed entirely: the true hypothesis and the true alpha are
read from the labels, the exact faulted network is built from them, and the
rollout starts from the MEASURED state at the decision instant. Everything
else -- horizon, validity mask, dt_roll -- matches training.

That makes this the floor on `l_traj`. If the oracle lands near the noise
floor, the 0.17 that training shows is the identifier failing and the physics
is fine. If the oracle is also ~0.17, no identifier could do better and the
problem is upstream: the initial state, the feature convention, or the
rollout, not the head.

Three starting states are compared, because delta carries 0.01 rad of noise
and Pe depends on angle differences:

    measured   x0 straight from the noisy channels (what training uses)
    smoothed   x0 averaged over the last `--smooth` window samples
    clean      x0 from the true equilibrium + the true fault (no measurement
               noise at all) -- the absolute floor
"""
import argparse, os
import numpy as np
import torch

from case30_physics import CASE30, NG, N_STATE

ap = argparse.ArgumentParser()
ap.add_argument('--data', default='data30_mm')
ap.add_argument('--base', default='case30_base.mat')
ap.add_argument('--sys', default='case30_simulink_data.mat')
ap.add_argument('--n', type=int, default=512, help='decision instants')
ap.add_argument('--horizon', type=float, default=1.5)
ap.add_argument('--dt_roll', type=float, default=0.01)
ap.add_argument('--w_max', type=float, default=1.0)
ap.add_argument('--onset_window', type=float, default=0.2)
ap.add_argument('--p_active', type=float, default=0.7)
ap.add_argument('--smooth', type=int, default=5, help='samples to average for x0')
ap.add_argument('--seed', type=int, default=0)
args = ap.parse_args()

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
S = CASE30(args.base, args.sys, dtype=torch.float32, device=dev)

X = np.load(os.path.join(args.data, 'X.npy'), mmap_mode='r')
md = np.load(os.path.join(args.data, 'meta.npz'), allow_pickle=True)
g = lambda k: md[k].ravel()
t_grid = g('t'); DT = float(md['dt_solver'])
y_type = g('y_type').astype(int); meta_loc = g('meta_loc').astype(int)
alpha_l = g('meta_alpha'); t_fault = g('y_t_fault'); t_clear = g('meta_t_clear')
TYPE_NAMES = ['gen', 'load', 'line']

D_IDX = list(range(0, 2 * NG, 2))
W_IDX = list(range(1, 2 * NG, 2))
PM_IDX = list(range(2 * NG, 3 * NG))
N_ROLL = int(round(args.horizon / args.dt_roll))
ROLL_STRIDE = int(round(args.dt_roll / DT))

rng = np.random.default_rng(args.seed)
N = X.shape[0]
lo, hi = args.w_max, t_grid[-1] - args.horizon
ks = rng.choice(N, size=args.n, replace=False)
tk = rng.uniform(lo, hi, args.n)
pick = rng.random(args.n) < args.p_active
a = np.clip(t_fault[ks], lo, hi)
b = np.clip(np.minimum(t_clear[ks], t_fault[ks] + args.onset_window), lo, hi)
ok = pick & (b > a)
tk[ok] = rng.uniform(a[ok], b[ok])
i_end = np.round(tk / DT).astype(int)

res = {k: [] for k in ('measured', 'smoothed', 'clean')}
for j, (k, ie) in enumerate(zip(ks, i_end)):
    typ = TYPE_NAMES[y_type[k] - 1]
    al = float(alpha_l[k])
    active = (tk[j] >= t_fault[k]) and (tk[j] < t_clear[k])

    blk = np.asarray(X[k, ie - args.smooth + 1:ie + N_ROLL * ROLL_STRIDE + 1])
    o = args.smooth - 1
    fut = blk[o::ROLL_STRIDE][:N_ROLL + 1]
    if len(fut) < N_ROLL + 1:
        continue
    w_meas = torch.tensor(fut[:, W_IDX], device=dev)

    # exact faulted network from the TRUE label
    if active:
        Yred, dPm0 = S.scenario(typ, int(meta_loc[k]), al)
    else:
        Yred, dPm0 = S.Yred_pre, torch.zeros(NG, device=dev)
    Pset = (S.Pm0 + dPm0).unsqueeze(0)
    Yr = Yred.unsqueeze(0)

    # validity mask: the rollout holds the fault, the real system does not
    t_change = (t_clear[k] if active
                else (t_fault[k] if tk[j] < t_fault[k] else np.inf))
    valid = np.clip(t_change - tk[j], 0.0, args.horizon)
    vm = torch.tensor((np.arange(N_ROLL + 1) * args.dt_roll <= valid),
                      device=dev).unsqueeze(1)

    starts = {}
    raw_end = blk[o]
    d = raw_end[D_IDX]
    starts['measured'] = np.concatenate([d - d[0], raw_end[W_IDX], raw_end[PM_IDX]])
    sm = blk[:args.smooth].mean(0)
    ds = sm[D_IDX]
    starts['smoothed'] = np.concatenate([ds - ds[0], sm[W_IDX], sm[PM_IDX]])
    x_clean = S.x0.clone()
    x_clean[:NG] = x_clean[:NG] - x_clean[0]
    starts['clean'] = x_clean.cpu().numpy()

    M, Nc = S.pe_consts(Yr)
    for name, x0 in starts.items():
        if name == 'clean' and active:
            continue                     # only meaningful pre-fault
        x = torch.tensor(np.asarray(x0, np.float32), device=dev).unsqueeze(0)
        tr = [x[:, NG:2 * NG]]
        for _ in range(N_ROLL):
            x = S.rk4_real(x, M, Nc, Pset, args.dt_roll)
            tr.append(x[:, NG:2 * NG])
        w = torch.cat(tr, 0)
        se = ((w - w_meas) / 0.01) ** 2
        res[name].append(float((se * vm).sum() / vm.expand_as(se).sum().clamp(min=1)))

print(f"oracle over {args.n} instants "
      f"({int(100*np.mean([(tk[j] >= t_fault[ks[j]]) and (tk[j] < t_clear[ks[j]]) for j in range(args.n)]))}% fault-active)")
print(f"{'start state':10s} {'n':>5s} {'l_traj':>10s} {'omega RMSE':>12s}")
for name, v in res.items():
    if not v:
        continue
    v = np.array(v)
    print(f"{name:10s} {len(v):5d} {np.median(v):10.5f} "
          f"{0.01*np.sqrt(np.median(v)):12.3e}")
print(f"\nnoise floor: omega noise_std 5.0e-05 -> l_traj 2.5e-05")
print("training currently reports l_traj ~0.17 (omega RMSE ~4e-3).")
print("If the oracle is far below that, identification is the problem.")
print("If the oracle is near it, no identifier could do better.")
