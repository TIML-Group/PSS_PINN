"""
verify_case30.py
================
Same gate the 9-bus port had to pass: simulate every scenario of the small
dataset with the ported physics and compare omega against the stored (noisy)
channels. If the RMSE sits at the noise floor (5e-5) and the out-of-limit flag
agrees, the differentiable model IS the system that generated the data.
"""
import numpy as np, h5py, torch, argparse, os
from case30_physics import CASE30, NG

ap = argparse.ArgumentParser()
ap.add_argument('--mat', default='case30_dataset_small_2.mat',
                help='dataset file holding X and the labels')
ap.add_argument('--base', default='case30_base.mat')
ap.add_argument('--sim', default='case30_simulink_data.mat')
ap.add_argument('--n', type=int, default=100)
ap.add_argument('--h', type=float, default=0.001)
args = ap.parse_args()

for _f in (args.base, args.sim, args.mat):
    if not os.path.exists(_f):
        raise SystemExit(f"not found: {_f}\n"
                         f"Run from the folder holding the case30 .mat files, "
                         f"or pass --base/--sim/--mat.")
S = CASE30('case30_base.mat', 'case30_simulink_data.mat')

f = h5py.File(args.mat, 'r')
g = lambda k: np.array(f[k]).ravel()
y_type, meta_loc = g('y_type').astype(int), g('meta_loc').astype(int)
alpha, t_fault, t_clear = g('meta_alpha'), g('y_t_fault'), g('meta_t_clear')
y_ool = g('y_ool').astype(bool)
t = g('t')
W_IDX = [1, 3, 5, 7, 9, 11]          # omega channels, 0-based
TYPE = {1: 'gen', 2: 'load', 3: 'line'}

rms, ool_ok, bad = [], 0, []
n = min(args.n, len(y_type))
for k in range(n):
    typ = TYPE[y_type[k]]
    _, x = S.simulate(typ, int(meta_loc[k]), float(alpha[k]),
                      float(t_fault[k]), float(t_clear[k]),
                      T=float(t[-1]), h=args.h)
    w_sim = x[:, NG:2 * NG].numpy()
    w_meas = np.array(f['X'][W_IDX, :, k]).T          # (T, 6)
    r = float(np.sqrt(((w_sim - w_meas) ** 2).mean()))
    rms.append(r)
    ool_sim = bool((np.abs(w_sim - 1) > S.limit).any())
    ool_ok += int(ool_sim == y_ool[k])
    if r > 5e-4:
        bad.append((k, typ, int(meta_loc[k]), round(r, 6)))
    if (k + 1) % 20 == 0:
        print(f"  {k+1}/{n}  median RMSE so far {np.median(rms):.3e}")

rms = np.array(rms)
print(f"\nomega RMSE vs stored channels over {n} scenarios")
print(f"  median {np.median(rms):.4e}   mean {rms.mean():.4e}   max {rms.max():.4e}")
print(f"  noise floor (omega noise_std) 5.0e-05")
print(f"  OOL agreement: {ool_ok}/{n}")
for typ_id, name in TYPE.items():
    m = y_type[:n] == typ_id
    if m.any():
        print(f"  {name:5s} n={m.sum():3d}  median RMSE {np.median(rms[m]):.4e}")
if bad:
    print("\nmismatches (RMSE > 5e-4):")
    for b in bad:
        print("  ", b)
else:
    print("\nno mismatches")
