"""Simulink v1.3 screen (v13sl_*.mat) vs the python screen (screen_v13_raw.npz).
Prints Simulink rates for both scorings and the max |peak| difference to python. Writes screen_v13_simulink.json."""
import glob
import json

import numpy as np
import scipy.io as sio

LIM = 0.005
M = sio.loadmat('matrix_scen.mat', squeeze_me=True); ne = int(M['n_eval'])
R = np.load('screen_v13_raw.npz')
ty_all = np.array([{1: 'gen', 2: 'load', 3: 'line'}[int(t)] for t in M['typ']]); loc_all = M['loc'].astype(int)
sets = {'ev': np.arange(ne), 'reach': np.arange(ne, len(ty_all)), 'disc': np.where(M['typ'][:ne] == 1)[0]}
pyk = {'ev': 'ev', 'reach': 'reach', 'disc': 'disc'}
out = {}
for name, sel in sets.items():
    fs = sorted(glob.glob(f'v13sl_{name}_*.mat'))
    if not fs:
        continue
    pos = {k: i for i, k in enumerate(sel)}
    pk = np.full(len(sel), np.nan); px = pk.copy(); ts = pk.copy(); tsx = pk.copy(); tv = pk.copy()
    for f in fs:
        m = sio.loadmat(f, squeeze_me=True)
        ii = [pos[int(k) - 1] for k in np.atleast_1d(m['idx'])]
        pk[ii], px[ii], ts[ii], tsx[ii], tv[ii] = m['peak'], m['peakx'], m['ts'], m['tsx'], m['tv']
    done = ~np.isnan(pk)
    ty, loc = ty_all[sel], loc_all[sel]
    d_pk = np.abs(pk - R[f'{pyk[name]}_peak'])[done].max()
    d_px = np.abs(px - R[f'{pyk[name]}_peak_x'])[done].max()
    flips = int(((pk <= LIM) != (R[f'{pyk[name]}_peak'] <= LIM))[done].sum() + ((px <= LIM) != (R[f'{pyk[name]}_peak_x'] <= LIM))[done].sum())
    row = dict(n_done=int(done.sum()), n=len(sel), max_abs_dpeak_vs_python=float(d_pk), max_abs_dpeak_excl_vs_python=float(d_px),
               verdict_flips_vs_python=flips)
    for key, p in (('counted', pk), ('excluded', px)):
        ok = p[done] <= LIM; t_ = ty[done]; l_ = loc[done]
        row[key] = dict(in_envelope=float(ok.mean()),
                        by_type={t: float(ok[t_ == t].mean()) for t in ('gen', 'load', 'line') if (t_ == t).any()},
                        gen_by_machine={f'G{j}': float(ok[(t_ == 'gen') & (l_ == j)].mean()) for j in (1, 2) if ((t_ == 'gen') & (l_ == j)).any()},
                        max_dev_median={t: float(np.median(p[done][t_ == t])) for t in ('gen', 'load', 'line') if (t_ == t).any()})
    row['tv_median_gen'] = float(np.median(tv[done][ty[done] == 'gen'])) if (ty[done] == 'gen').any() else None
    out[name] = row
    print(name, json.dumps(row))
json.dump(out, open('screen_v13_simulink.json', 'w'), indent=1)
