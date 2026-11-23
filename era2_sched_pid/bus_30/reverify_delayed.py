"""
reverify_delayed.py -- wiki S34(c): re-verify ALL R1/R1d and R2 library entries under DELAYED-ENTRY switch states.

The S24 verification grid (build_library.ver_cases, same cases and references) is run with the identification
switch at t1 = t_fault + L instead of t_fault (oracle entry, L = 0). R2 switches at t2 = max(t_clear, t1)
(straight to R2 at t1 when the fault cleared before the decision); t2 = inf for alpha = 1 (S20).
Schedules (as S23): R1/R1d entry e: [g18, e, g18] vs reference [g18, g18, g18];
                    R2 entry r:     [g18, R1 entry of the case, r] vs [g18, R1 entry, g18].
Both entry and reference use the same switch times. Scoring = build_library.score with S28 rail-binding
(survivors only, rail_frac_surv) and the bump of the entry's own transfer (R1/R1d entries: the switch into R1;
R2 entries: the switch into R2): fails = envelope loss vs reference | median objective worse than reference
(not inactive) | survivors' rail > 5 % | transition bump > 0.05 pu.
L grid: 0 (oracle entry, the S27 verification) and 28 ms (operating variant cls_learned_A, S34) are the acceptance
states; 32 / 57 / 64 ms are reported as diagnostics (the other trained windows).
Usage:  python reverify_delayed.py [--lib library_v21.json] [--L 0 0.028 0.032 0.057 0.064] [--out reverify_v21.json]
"""
import argparse
import json
import time

import numpy as np

import build_library as BL
from library import Lin, enumerate_configs, post_loss_angles
from sched_sim import make_plant, simulate_sched

ACCEPT_L = (0.0, 0.028)


def switch_arrays(S, L):
    tf = np.array([s['t_fault'] for s in S]); tc = np.array([s['t_clear'] for s in S])
    a1 = np.array([s['type'] == 'gen' and s['alpha'] >= 1 - 1e-12 for s in S])
    t1 = tf + L
    return t1, np.where(a1, np.inf, np.maximum(tc, t1))


def verify_at(p, C, lib, names, L, override=None):
    S, G, R, owner = BL.schedules(C, lib, names, override=override)
    o = simulate_sched(p, S, G, t_sw=switch_arrays(S, L))
    o = dict(o); o['rail_frac'] = o['rail_frac_surv']                       # S28: survivors only
    isr2 = np.array([C[n]['regime'] == 'R2' for n in owner])
    o['bump'] = np.where(isr2, o['bump_r2'], o['bump_r1'])                  # the entry's OWN transfer
    return BL.score(C, names, S, R, owner, o)


def load_lib(path):
    E = json.load(open(path))['entries']
    return {n: np.array(v['gains']) for n, v in E.items()}


def context():
    lin = Lin(); post_loss_angles(lin, BL.g18)
    return {c['name']: c for c in enumerate_configs(lin)}, make_plant()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--lib', default='library_v21.json')
    ap.add_argument('--L', type=float, nargs='+', default=[0.0, 0.028, 0.032, 0.057, 0.064])
    ap.add_argument('--out', default='reverify_v21.json')
    args = ap.parse_args()
    C, p = context(); lib = load_lib(args.lib)
    names = [n for n in C if n != 'R0']
    res = {}
    for L in args.L:
        t0 = time.time()
        v = verify_at(p, C, lib, names, L)
        res[f'{L * 1000:.0f}'] = v
        fails = {n: v[n]['fails'] for n in names if v[n]['fails']}
        kinds = {}
        for f in sum(fails.values(), []):
            k = f.split(' ')[0]; kinds[k] = kinds.get(k, 0) + 1
        print(f'L = {L * 1000:3.0f} ms: failing {len(fails)}/{len(names)} {kinds}; bump max {max(v[n]["bump_max"] for n in names):.3f} '
              f'({time.time() - t0:.0f}s)', flush=True)
        for n, f in fails.items():
            print(f'    {n:18s} {f}  bump {v[n]["bump_max"]:.3f}  med {v[n]["entry_median"]:.3e} vs ref {v[n]["ref_median"]:.3e}  '
                  f'rail {v[n]["rail_max"]:.3f}', flush=True)
    json.dump(dict(lib=args.lib, L_ms=[L * 1000 for L in args.L], accept_L_ms=[L * 1000 for L in ACCEPT_L], verify=res),
              open(args.out, 'w'), indent=1)
