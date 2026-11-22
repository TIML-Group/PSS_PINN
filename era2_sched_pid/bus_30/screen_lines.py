"""Line pre-screen (answers §3): every faultable line x fault duration on the v2 plant
under the frozen generalist baseline. Writes line_screen.json."""
import json, numpy as np, torch
from plant_v2 import PlantV2, const_gains, LIMIT, actuator_hash
from scenarios_v2 import candidates
import sys
SRC = sys.argv[1] if len(sys.argv) > 1 else 'pole'
p = PlantV2(); g = np.array(json.load(open('seeds_v2.json'))['generalist'] if SRC == 'pole' else json.load(open('baseline_ceiling.json'))['gains'])
lines = candidates(p)['line']; durs = np.round(np.arange(0.05, 3.0001, 0.05), 3)
sc = [dict(type='line', loc=l, alpha=1.0, t_fault=1.0, t_clear=1.0 + d) for l in lines for d in durs]
o = p.simulate(p.batch(sc), const_gains(len(sc), p.n_lbl, g, p.device, p.dtype))
pk = o['peak'].cpu().numpy().reshape(len(lines), len(durs))
res = {'actuator_hash': actuator_hash(), 'durations': durs.tolist(), 'lines': lines, 'peak': pk.tolist()}
res['gains_source'] = SRC
json.dump(res, open('line_screen.json' if SRC == 'pole' else 'line_screen_ceiling.json', 'w'))
bad = pk > LIMIT
print('needy (line,duration) cells: %d / %d  (%.1f%%)' % (bad.sum(), bad.size, 100 * bad.mean()))
for i, l in enumerate(lines):
    if bad[i].any():
        print(f'  line {l:2d}: needy for duration >= {durs[bad[i]].min():.2f}s  ({bad[i].sum()} of {len(durs)} cells), worst peak {pk[i].max():.4f}')
print('line max peak overall: %.4f; lines never needy: %d' % (pk.max(), (~bad.any(1)).sum()))
