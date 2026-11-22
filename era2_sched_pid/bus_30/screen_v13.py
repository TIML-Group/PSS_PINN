"""
screen_v13.py -- wiki S13 (v1.3, C-hybrid gen faults) screen on the S10 cases.

Plant: v1.2 actuator (rail |P_set - Pm0| <= 1.0, no rate limit), pole-placement generalist.
Cases (matrix_scen.mat, identical to the S10 Simulink matrix):
  * eval  : the 1,200 seed-7 scenarios, same split / fault mix / alpha values.
            All gen alphas are < 1, so every gen case is a connected runback (PID active).
  * reach : the 57-point G1 alpha grid (alpha 0.10..1.00 x durations 0.1/0.5/3 s);
            its alpha = 1.0 points are disconnections.
  * disc  : separate alpha = 1 group -- every gen case of the eval set re-run as a full
            disconnection of the same machine at the same t_fault (duration irrelevant:
            the loss is permanent). Reported separately; the eval split is untouched.
Both scorings are reported: 'counted' (all machines in the network) and 'excluded'
(the faulted machine excluded from t_fault onward). For disconnections they coincide.
Also re-runs the eval set under the legacy Q8 semantics as the S10 reference.
Writes screen_v13.json.
"""
import json
import sys
import time

import numpy as np
import scipy.io as sio

from plant_v2 import PlantV2, const_gains, LIMIT, actuator_hash

dev = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
TYPES = {1: 'gen', 2: 'load', 3: 'line'}
BAND = 1e-3

M = sio.loadmat('matrix_scen.mat', squeeze_me=True)
ne = int(M['n_eval'])
g = np.array(M['Gpole'])
allS = [dict(type=TYPES[int(t)], loc=int(l), alpha=float(a), t_fault=float(f), t_clear=float(c))
        for t, l, a, f, c in zip(M['typ'], M['loc'], M['alpha'], M['tf'], M['tc'])]
ev, reach = allS[:ne], allS[ne:]
disc = [dict(s, alpha=1.0) for s in ev if s['type'] == 'gen']


def run(p, S):
    p.bands = (BAND,)
    o = p.simulate(p.batch(S), const_gains(len(S), p.n_lbl, g, p.device, p.dtype))
    tf = np.array([s['t_fault'] for s in S])
    f = lambda k: o[k].cpu().numpy()
    ts = lambda le: np.maximum(le[BAND].cpu().numpy() - tf, 0)
    return dict(peak=f('peak'), peak_x=f('peak_excl'), tv=f('tv'),
                ts=ts(o['last_exit']), ts_x=ts(o['last_exit_excl']),
                w_end=(o['w_end'] - 1).abs().amax(1).cpu().numpy())


def stats(r, S, key):
    pk = r['peak' if key == 'counted' else 'peak_x']
    ts = r['ts' if key == 'counted' else 'ts_x']
    ty = np.array([s['type'] for s in S]); loc = np.array([s['loc'] for s in S])
    al = np.array([s['alpha'] for s in S]); ok = pk <= LIMIT
    out = dict(n=len(S), in_envelope=float(ok.mean()), by_type={}, max_dev_median={}, ts_median={}, tv_median={})
    for t in ('gen', 'load', 'line'):
        m = ty == t
        if m.any():
            out['by_type'][t] = float(ok[m].mean())
            out['max_dev_median'][t] = float(np.median(pk[m]))
            out['ts_median'][t] = float(np.median(ts[m]))
            out['tv_median'][t] = float(np.median(r['tv'][m]))
    gm = ty == 'gen'
    if gm.any():
        out['gen_by_machine'] = {f'G{j}': float(ok[gm & (loc == j)].mean()) for j in (1, 2) if (gm & (loc == j)).any()}
        out['gen_G1_by_alpha'] = {b: float(ok[gm & (loc == 1) & m].mean()) for b, m in
                                  (('a<0.39', al < 0.39), ('0.39-0.7', (al >= 0.39) & (al < 0.7)), ('a>=0.7', al >= 0.7))
                                  if (gm & (loc == 1) & m).any()}
        out['gen_max_dev_by_machine_median'] = {f'G{j}': float(np.median(pk[gm & (loc == j)])) for j in (1, 2) if (gm & (loc == j)).any()}
    return out


def reach_tab(r, key):
    pk = r['peak' if key == 'counted' else 'peak_x']
    a = np.array([s['alpha'] for s in reach]); d = np.array([s['t_clear'] - s['t_fault'] for s in reach])
    out = {}
    for dd in (0.1, 0.5, 3.0):
        m = np.isclose(d, dd)
        okm = m & (pk <= LIMIT)
        out[f'{dd:.1f}s'] = dict(largest_alpha_held=(float(a[okm].max()) if okm.any() else None),
                                 all_held_up_to=float(max([x for x in sorted(a[m]) if (pk[m][a[m] <= x] <= LIMIT).all()], default=0.0)),
                                 peak_by_alpha={f'{x:.2f}': float(pk[m][a[m] == x][0]) for x in sorted(a[m])})
    return out


t0 = time.time()
res = dict(note=__doc__.strip().splitlines()[1], limit=LIMIT, settle_band=BAND, gains='seeds_v2 generalist (pole placement)')
p13 = PlantV2(device=dev, rail=1.0, rate=float('inf'), gen_mode='v13')
pq8 = PlantV2(device=dev, rail=1.0, rate=float('inf'), gen_mode='q8')
res['actuator_hash'] = dict(v13=actuator_hash(1.0, float('inf'), 'v13'), q8=actuator_hash(1.0, float('inf'), 'q8'))

r_q8 = run(pq8, ev)
res['eval_q8_reference'] = stats(r_q8, ev, 'counted')
r_ev = run(p13, ev)
res['eval_v13'] = {k: stats(r_ev, ev, k) for k in ('counted', 'excluded')}
r_re = run(p13, reach)
res['G1_reach_v13'] = {k: reach_tab(r_re, k) for k in ('counted', 'excluded')}
r_di = run(p13, disc)
res['disc_alpha1'] = stats(r_di, disc, 'excluded')
loc_d = np.array([s['loc'] for s in disc])
res['disc_alpha1']['per_machine'] = {
    f'G{j}': dict(n=int((loc_d == j).sum()),
                  peak_min=float(r_di['peak_x'][loc_d == j].min()), peak_max=float(r_di['peak_x'][loc_d == j].max()),
                  ts_median=float(np.median(r_di['ts_x'][loc_d == j])),
                  dev_at_8s_max=float(r_di['w_end'][loc_d == j].max()))
    for j in (1, 2)}
res['runtime_s'] = time.time() - t0
np.savez('screen_v13_raw.npz', **{f'{n}_{k}': v for n, r in (('q8', r_q8), ('ev', r_ev), ('reach', r_re), ('disc', r_di)) for k, v in r.items()})
json.dump(res, open('screen_v13.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
