"""
eval_ceiling_v13x.py -- wiki S15 ceiling re-tune vs the pole-placement generalist, same 1,200 seed-7
eval cases (matrix_scen.mat), v1.3 semantics, rail 1.0, no rate limit. Primary = EXCLUDED scoring;
the faulted unit's own excursion (counted peak) is auxiliary. Python here; Simulink check via
run_v13_sl('ev', part, n, 'v13x') + this script's --sl flag. Writes eval_ceiling_v13x.json.
"""
import glob
import json
import sys

import numpy as np
import scipy.io as sio

from plant_v2 import PlantV2, const_gains, LIMIT

BAND = 1e-3
M = sio.loadmat('matrix_scen.mat', squeeze_me=True); ne = int(M['n_eval'])
T = {1: 'gen', 2: 'load', 3: 'line'}
S = [dict(type=T[int(M['typ'][k])], loc=int(M['loc'][k]), alpha=float(M['alpha'][k]),
          t_fault=float(M['tf'][k]), t_clear=float(M['tc'][k])) for k in range(ne)]
ty = np.array([s['type'] for s in S]); loc = np.array([s['loc'] for s in S]); al = np.array([s['alpha'] for s in S])
tf = np.array([s['t_fault'] for s in S])
gains = {'pole': np.array(M['Gpole']), 'v13x': np.array(json.load(open('baseline_ceiling_v13x.json'))['gains'])}


def summarize(pkx, pkc, tsx, tv):
    ok = pkx <= LIMIT; g1 = (ty == 'gen') & (loc == 1)
    return dict(in_envelope=float(ok.mean()),
                by_type={t: float(ok[ty == t].mean()) for t in ('gen', 'load', 'line')},
                gen_by_machine={f'G{j}': float(ok[(ty == 'gen') & (loc == j)].mean()) for j in (1, 2)},
                G1_by_alpha={b: float(ok[g1 & m].mean()) for b, m in
                             (('a<0.39', al < 0.39), ('0.39-0.7', (al >= 0.39) & (al < 0.7)), ('a>=0.7', al >= 0.7))},
                max_dev_median={t: float(np.median(pkx[ty == t])) for t in ('gen', 'load', 'line')},
                max_dev_p95={t: float(np.percentile(pkx[ty == t], 95)) for t in ('gen', 'load', 'line')},
                ts_median={t: float(np.median(tsx[ty == t])) for t in ('gen', 'load', 'line')},
                tv_median={t: float(np.median(tv[ty == t])) for t in ('gen', 'load', 'line')},
                aux_faulted_unit_peak_median_gen=float(np.median(pkc[ty == 'gen'])),
                aux_counted_in_envelope=float((pkc <= LIMIT).mean()))


out = {}
if '--sl' in sys.argv:
    out = json.load(open('eval_ceiling_v13x.json'))
    for des, pat in (('pole', 'v13sl_ev_[0-9].mat'), ('v13x', 'v13sl_ev_v13x_*.mat')):
        pk = np.full(ne, np.nan); px = pk.copy(); tsx = pk.copy(); tv = pk.copy()
        for f in glob.glob(pat):
            m = sio.loadmat(f, squeeze_me=True); i = np.atleast_1d(m['idx']) - 1
            pk[i], px[i], tsx[i], tv[i] = m['peak'], m['peakx'], m['tsx'], m['tv']
        assert not np.isnan(pk).any(), des
        out[f'simulink_{des}'] = summarize(px, pk, tsx, tv)
        py = np.load(f'eval_v13x_{des}_py.npz')
        out[f'simulink_{des}']['max_abs_dpeak_vs_python'] = float(np.abs(px - py['px']).max())
        out[f'simulink_{des}']['verdict_flips_vs_python'] = int(((px <= LIMIT) != (py['px'] <= LIMIT)).sum())
else:
    p = PlantV2(device='cuda', rail=1.0, rate=float('inf'), gen_mode='v13'); p.bands = (BAND,)
    for des, g in gains.items():
        o = p.simulate(p.batch(S), const_gains(ne, p.n_lbl, g, p.device, p.dtype))
        px, pc, tv = (o[k].cpu().numpy() for k in ('peak_excl', 'peak', 'tv'))
        tsx = np.maximum(o['last_exit_excl'][BAND].cpu().numpy() - tf, 0)
        np.savez(f'eval_v13x_{des}_py.npz', px=px, pc=pc, tsx=tsx, tv=tv)
        out[f'python_{des}'] = summarize(px, pc, tsx, tv)
    out['gains'] = {k: np.round(v, 4).tolist() for k, v in gains.items()}
    out['gain_ratio_v13x_over_pole'] = np.round(gains['v13x'] / gains['pole'], 3).tolist()
json.dump(out, open('eval_ceiling_v13x.json', 'w'), indent=1)
for k, v in out.items():
    if k.startswith(('python', 'simulink')):
        print(k, round(v['in_envelope'], 4), v['by_type'], v['gen_by_machine'], v['G1_by_alpha'],
              'mdev', v['max_dev_median'], 'ts', v['ts_median'], 'tv', v['tv_median'], 'aux', v['aux_faulted_unit_peak_median_gen'])
