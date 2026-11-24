"""
repair_focused.py -- wiki S36(b): focused, wider CMA-ES repair of R2_runback_G2 on library v2.2.

Same mechanism and objective as repair_delayed.py (S35): from the current v2.2 entry (log-multipliers), hard terms at
both acceptance states L = 0 / 28 ms (median t_s floor vs [R1 entry, generalist], envelope, needy rule, survivors'
rail <= 5 %, own R1 -> R2 bump <= 0.05 pu), main term mean t_s at 28 ms; no slew, no softer-gains fallback.
Wider than S35: tuned on ALL cases of the entry (S35's <= 60-case stride subset missed the 1 % of cases that still
bumped 0.104 pu), sigma0 0.6, population 24, 60 generations. Accepted only if the entry's full-grid failure count
at the acceptance states drops. R1d entries are exempt from the bump criterion (S36(a)); their transfer jump is
reported as a disclosed-transient column. Final: full re-verification of all entries at L = 0 / 28 / 57 ms.
Writes library_v23.json, repair_focused.json; stdout -> repair_focused.log.
"""
import json
import time

import numpy as np

import repair_delayed as RD
from reverify_delayed import verify_at, load_lib, context

NAMES = ['R2_runback_G2']
RD.POP, RD.GENS, RD.SIGMA0, RD.N_TUNE = 24, 60, 0.6, 10 ** 6
RD.T0 = T0 = time.time()


def fails_ex(C, n, v):
    """S36(a): R1d entries exempt from the bump criterion."""
    f = v['fails']
    return [x for x in f if not x.startswith('bump')] if C[n]['regime'] == 'R1d' else f


if __name__ == '__main__':
    C, p = context(); lib = load_lib('library_v22.json')
    RD.before_gains = {n: g.copy() for n, g in lib.items()}
    names = [n for n in C if n != 'R0']
    report = RD.stage(p, C, lib, NAMES, 'focused (S36b)')
    final = {f'{L * 1000:.0f}': verify_at(p, C, lib, names, L) for L in (0.0, 0.028, 0.057)}
    for Lk, v in final.items():
        ff = {n: v[n]['fails'] for n in names if v[n]['fails']}
        fx = {n: fails_ex(C, n, v[n]) for n in names if fails_ex(C, n, v[n])}
        bmax = max(v[n]['bump_max'] for n in names if C[n]['regime'] != 'R1d')
        r1d = {n: round(v[n]['bump_max'], 3) for n in names if C[n]['regime'] == 'R1d'}
        RD.log(f'FINAL L = {Lk} ms: failing {len(ff)} {ff}')
        RD.log(f'    with S36(a) R1d exemption: failing {len(fx)} {fx}; bump max (non-R1d) {bmax:.3f}; R1d transfer (disclosed) {r1d}')
    r = json.load(open('library_v22.json'))
    for n in names:
        r['entries'][n]['gains'] = lib[n].tolist()                      # full precision (6-dp rounding moved the g18 fallbacks off g18)
        r['entries'][n]['verify_delayed'] = {Lk: final[Lk][n] for Lk in final}
        if C[n]['regime'] == 'R1d':
            r['entries'][n]['bump_exempt'] = 'S36(a): disclosed transfer transient'
        if n in report:
            r['entries'][n]['repair_focused'] = dict(accepted=report[n]['accepted'])
    r['spec'] = 'v2.3 / S36: library v2.2 + focused repair of R2_runback_G2; R1d bump-exempt (disclosed transient)'
    r['repair_focused'] = dict(budget=dict(pop=RD.POP, gens=RD.GENS, sigma0=RD.SIGMA0, cases='all'), accept_L_ms=[0, 28],
                               repaired=NAMES, runtime_s=time.time() - T0)
    json.dump(r, open('library_v23.json', 'w'), indent=1)
    json.dump(dict(report=report, final=final), open('repair_focused.json', 'w'), indent=1)
    RD.log(f'done ({time.time() - T0:.0f}s)')
