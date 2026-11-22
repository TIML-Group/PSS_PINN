"""Q6: effort-weight sweep on 20 anchors (before pinning LAM). Writes lam_sweep.json."""
import json, time
import numpy as np
from plant_v2 import PlantV2, LIMIT, actuator_hash
from scenarios_v2 import draw, candidates
from tune_anchors import tune, GENS

LAMS = [0.0, 0.03, 0.1, 0.3, 1.0]
p = PlantV2(); seeds = json.load(open('seeds_v2.json'))['seeds']
rng = np.random.default_rng(11)
anc = []
def add(t, loc, a, dur):
    tf = float(rng.uniform(0.5, 4.0)); anc.append(dict(id=len(anc), type=t, loc=int(loc), alpha=float(a), t_fault=tf, t_clear=tf + float(dur)))
for dur in (0.15, 0.6, 1.5): add('line', 1, 1, dur)
for dur in (0.4, 1.2): add('line', 5, 1, dur)
other = [l for l in candidates(p)['line'] if l not in (1, 5)]
for l in rng.choice(other, 3, replace=False): add('line', l, 1, rng.uniform(0.05, 3.0))
for a in (0.2, 0.5, 0.9): add('load', rng.choice(candidates(p)['load']), a, rng.uniform(0.05, 3.0)); add('load', rng.choice(candidates(p)['load']), a, rng.uniform(0.05, 3.0))
for m in (1, 2):
    for a in (0.3, 0.6, 0.95): add('gen', m, a, rng.uniform(0.05, 3.0))
A = len(anc)
probs = [dict(a, id=a['id']) for _ in LAMS for a in anc]
lams = np.repeat(LAMS, A)
t0 = time.time()
th, J, info, info0 = tune(p, probs, seeds, lams)
res = {'actuator_hash': actuator_hash(), 'anchors': anc, 'lams': LAMS, 'gens': GENS, 'rows': []}
for i, lam in enumerate(LAMS):
    s = slice(i * A, (i + 1) * A)
    f = info['feas'][s]; pk = info['peak'][s]; marg = (LIMIT - pk) / LIMIT
    row = dict(lam=lam, feasible=int(f.sum()), ts_median=float(np.median(info['ts'][s][f])) if f.any() else None,
               tv_median=float(np.median(info['tv'][s])), edge_riders=int((f & (marg < 0.05)).sum()),
               margin_median=float(np.median(marg[f])) if f.any() else None,
               gmag_median=float(np.median(info['gmag'][s])),
               max_mult_median=float(np.median(np.exp(np.abs(th[s]).max(1)))),
               ts_seed_median=float(np.median(info0['ts'][s][info0['feas'][s]])) if info0['feas'][s].any() else None,
               feasible_seed=int(info0['feas'][s].sum()))
    res['rows'].append(row); print(row)
res['per_anchor'] = dict(peak=info['peak'].tolist(), ts=info['ts'].tolist(), tv=info['tv'].tolist(), J=J.tolist(),
                         peak_seed=info0['peak'].tolist(), ts_seed=info0['ts'].tolist())
json.dump(res, open('lam_sweep_v2.json', 'w'), indent=1)
print('done in %.0f s' % (time.time() - t0))
