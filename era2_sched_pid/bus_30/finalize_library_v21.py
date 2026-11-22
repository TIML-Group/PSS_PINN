"""finalize_library_v21.py -- S26 'matching passes' fallback: entries whose BOTH seeds fail the peak floor on the
full verification grid take the generalist gains (gap 0). Re-verifies those entries and records the reason.
R1_runback_G1 additionally carries a structural rail flag: all rail-binding is the runback unit's own PID on deep
runbacks (alpha >= 0.92), the generalist binds 6.3 % there too, no other machine reaches the band -> for Ashhar."""
import json, numpy as np
import build_library as BL
from library import Lin, enumerate_configs, post_loss_angles
from sched_sim import make_plant
r = json.load(open('library_v21.json')); E = r['entries']; g18 = BL.g18
fb = [n for n, v in E.items() if n != 'R0' and v['verify']['fails'] and all(vv['fails'] for vv in (v['verify_by_seed'] or {}).values())]
lin = Lin(); post_loss_angles(lin, g18); C = {c['name']: c for c in enumerate_configs(lin)}
lib = {n: np.array(v['gains']) for n, v in E.items()}
for n in fb:
    lib[n] = g18.copy()
v = BL.verify(make_plant(), C, lib, fb)
for n in fb:
    E[n]['gains'] = g18.tolist(); E[n]['verify_before_fallback'] = E[n]['verify']; E[n]['verify'] = v[n]
    E[n]['seed_chosen'] = 'fallback: generalist (both seeds failed the peak floor on the full grid; S26 matching passes)'
    print(n, 'fallback ->', v[n]['fails'], 'gap', v[n]['gap_rel'], 'rail', v[n]['rail_max'])
E['R1_runback_G1']['flag'] = ('structural rail: binding is entirely the runback unit (G1) own PID on deep runbacks '
                              '(alpha >= 0.92); generalist 6.3 %, entry 6.6 %, other machines 0 % -- rule interpretation for Ashhar')
r['fallback'] = fb
json.dump(r, open('library_v21.json', 'w'), indent=1)
