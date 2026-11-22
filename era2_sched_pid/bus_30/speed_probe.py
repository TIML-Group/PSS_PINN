"""
speed_probe.py -- wiki S16: Python runner timing (pair: speed_probe_sl.m).
Same cases on both runners: the first 64 of the 1,200 seed-7 eval scenarios (matrix_scen.mat),
v1.3 semantics, rail 1.0, no rate limit, pole-placement generalist, 8 s at 1 ms.
Times: single scenario (B=1) and the 64 batch, on cuda and cpu, best of REPS after one warm-up,
plus cold (first call, incl. CUDA init / build). Writes speed_probe_py.json (+ peaks for the cross-check).
"""
import json
import time

import numpy as np
import scipy.io as sio
import torch

from plant_v2 import PlantV2, const_gains

REPS = 3
M = sio.loadmat('matrix_scen.mat', squeeze_me=True)
T = {1: 'gen', 2: 'load', 3: 'line'}
S = [dict(type=T[int(M['typ'][k])], loc=int(M['loc'][k]), alpha=float(M['alpha'][k]),
          t_fault=float(M['tf'][k]), t_clear=float(M['tc'][k])) for k in range(64)]
g = np.array(M['Gpole'])
out = dict(cases='matrix_scen.mat eval idx 1..64 (seed 7)', mix={t: sum(s['type'] == t for s in S) for t in T.values()})


def sync(dev):
    if dev == 'cuda':
        torch.cuda.synchronize()


for dev in ('cuda', 'cpu'):
    t0 = time.perf_counter()
    p = PlantV2(device=dev, rail=1.0, rate=float('inf'), gen_mode='v13')
    t_build = time.perf_counter() - t0
    res = dict(plant_build_s=t_build)
    for name, sc in (('single', S[:1]), ('batch64', S)):
        times, pk = [], None
        for r in range(REPS + 1):
            sync(dev); t0 = time.perf_counter()
            bt = p.batch(sc)
            # t_start=0: full 8 s from t=0, same as Simulink (no skip of the pre-fault rest)
            o = p.simulate(bt, const_gains(len(sc), p.n_lbl, g, p.device, p.dtype), t_start=0.0)
            sync(dev); times.append(time.perf_counter() - t0)
            pk = o['peak'].cpu().numpy()
        res[name] = dict(cold_s=times[0], warm_best_s=min(times[1:]), n=len(sc),
                         cases_per_s=len(sc) / min(times[1:]))
        if name == 'batch64':
            res['peaks'] = pk.tolist()
    out[dev] = res
    print(dev, {k: v for k, v in res.items() if k != 'peaks'}, flush=True)
json.dump(out, open('speed_probe_py.json', 'w'), indent=1)
