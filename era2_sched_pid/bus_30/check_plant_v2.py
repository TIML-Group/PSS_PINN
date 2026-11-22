"""Cross-check plant_v2.py against case30_pidv2.slx (run case30_pidv2_check.m first)."""
import numpy as np, scipy.io as sio, torch
from plant_v2 import PlantV2, const_gains, T_END, H_STEP

S = sio.loadmat('pidv2_check_simulink.mat', squeeze_me=True)
p = PlantV2()
rows = [('none', 0, 1, 1e9, 1e9), ('load', 7, .5, 1.0, 1.5), ('line', 5, 1, 1.2, 2.0),
        ('gen', 1, .6, 1.0, 2.5), ('load', 21, .8, 1.5, 1.65)]
sc = [dict(type=r[0] if r[0] != 'none' else 'line', loc=r[1] or 1, alpha=r[2], t_fault=r[3], t_clear=r[4]) for r in rows]
G = const_gains(len(sc), p.n_lbl, [[20, 5, 2]] * 6, p.device, p.dtype).clone()
G[1, 40:60, :, 0] = 40; G[1, 40:60, :, 2] = 4
o = p.simulate(p.batch(sc), G, t_start=0.0, record=True)
wp = o['rec']['w'].cpu().numpy()                       # (B, n, NG) at t = h..T
for i, r in enumerate(rows):
    ws = np.asarray(S['W'][i])[1:]                     # drop t = 0
    n = min(len(ws), wp.shape[1])
    err = np.abs(ws[:n] - wp[i, :n]).max()
    print(f"{r[0]:5s} loc {r[1]:2d}: peak py {o['peak'][i].item():.4e}  max|Simulink - python| = {err:.2e}")
