"""
build_library_v21.py -- spec v2.1 / wiki S26: peak-first per-configuration library (the S6a oracle).

Every entry (62; R0 = S18 baseline) is CMA-ES-tuned from TWO seeds -- the S25 pole-placed entry
(library_v2.json) and the S18 generalist -- with the frozen budget pop 16 x 40 generations per seed,
global cap 3 h CUDA. Parameters: log-multipliers on the seed gains.
Lexicographic objective (S26), as a penalty-ordered scalar:
  hard  (x HARD): peak floor  -- median excluded peak not worse than the generalist's on the same cases;
                  needy rule  -- where the generalist leaves cases out of envelope, every case must be in;
                  no envelope loss vs the generalist; rail-binding <= 5 %; transition bump <= 0.05 pu
  then  min mean t_s (0.001 band, from t_fault) [s]
  then  effort: EFF_W x mean TV(Pm)
Stage A tunes R1 / R1d (schedule [g18, entry, g18]); stage B tunes R2 (schedule [g18, chosen R1
entry of the case, entry]). Tuning uses a deterministic stride subset (<= N_TUNE cases per entry);
the per-entry winner of the two seeds is picked on the FULL S24 verification grid; the final
library is re-verified in full (gap column vs the generalist, bumpless check).
Writes library_v21.json and build_library_v21.log (stdout).
"""
import json
import time

import numpy as np

import build_library as BL
from library import Lin, enumerate_configs, post_loss_angles
from sched_sim import make_plant, simulate_sched
from plant_v2 import LIMIT, NG
from tune_anchors import CMA

POP, GENS, SIGMA0, SEED0 = 16, 40, 0.3, 20261126
N_TUNE, HARD, EFF_W = 60, 100.0, 1e-3
CAP_S = 3 * 3600
g18 = BL.g18
T0 = time.time()


def log(*a):
    print(*a, flush=True)


def stride(lst, n):
    if len(lst) <= n:
        return lst
    st = int(np.ceil(len(lst) / n))
    return lst[::st]


def entry_cases(C, nm):
    cf = C[nm]
    if cf['regime'] in ('R1', 'R1d'):
        return BL.ver_cases(cf)
    fam = cf['family']
    S = [s for c2 in C.values() if c2['regime'] == 'R1' and c2['family'] == fam for s in BL.ver_cases(c2)]
    return [s for s in S if s['type'] != 'gen' or s['alpha'] < 1]


def g3_for(C, lib, nm, s, g):
    if C[nm]['regime'] in ('R1', 'R1d'):
        return [g18, g, g18]
    return [g18, lib[BL.r1_name(s)], g]


def objective(o, idx, ref):
    """o: sim output; idx: slice of this member's cases; ref: dict of generalist arrays on the same cases."""
    pk = o['peak_excl'][idx]; ts = o['ts'][idx]; tv = o['tv'][idx]
    viol = max(0.0, np.median(pk) / max(np.median(ref['pk']), 1e-12) - 1.0)
    viol += ((pk > LIMIT) & (ref['pk'] <= LIMIT)).mean()
    if (ref['pk'] > LIMIT).any():                      # needy configuration: must beat the generalist
        viol += np.maximum(0.0, pk / LIMIT - 1.0).mean() * 10
    viol += max(0.0, o['rail_frac'][idx].max() - BL.RAIL_MAX) + max(0.0, o['bump'][idx].max() - BL.BUMP_MAX)
    return HARD * viol + ts.mean() + EFF_W * tv.mean(), viol


def tune_stage(p, C, lib, names, seeds, t_cap):
    """Stacked CMA-ES over names x seeds. Returns {(nm, seed): gains}, history."""
    cases = {nm: stride(entry_cases(C, nm), N_TUNE) for nm in names}
    # generalist reference on the tuning cases (computed once)
    S_ref, G_ref, own = [], [], []
    for nm in names:
        for s in cases[nm]:
            S_ref.append(s); G_ref.append([g18, g18, g18] if C[nm]['regime'] != 'R2' else [g18, lib[BL.r1_name(s)], g18]); own.append(nm)
    o = simulate_sched(p, S_ref, np.array(G_ref)); own = np.array(own)
    ref = {nm: dict(pk=o['peak_excl'][own == nm]) for nm in names}
    keys = [(nm, sd) for nm in names for sd in seeds]
    es = {k: CMA(NG * 3, SEED0 + i, sigma0=SIGMA0, pop=POP) for i, k in enumerate(keys)}
    best = {k: (np.inf, np.zeros(NG * 3), np.inf) for k in keys}
    hist = []; t0 = time.time()
    for gen in range(GENS):
        X = {k: es[k].ask() for k in keys}
        S_all, G_all, slices = [], [], {}
        for k in keys:
            nm, sd = k; base = seeds[sd][nm]
            for m in range(POP):
                gk = base * np.exp(X[k][m].reshape(NG, 3))
                a = len(S_all)
                for s in cases[nm]:
                    S_all.append(s); G_all.append(g3_for(C, lib, nm, s, gk))
                slices[(k, m)] = slice(a, len(S_all))
        o = simulate_sched(p, S_all, np.array(G_all))
        for k in keys:
            J = np.zeros(POP); V = np.zeros(POP)
            for m in range(POP):
                J[m], V[m] = objective(o, slices[(k, m)], ref[k[0]])
            es[k].tell(X[k], J)
            i = int(np.argmin(J))
            if J[i] < best[k][0]:
                best[k] = (float(J[i]), X[k][i].copy(), float(V[i]))
        nviol = sum(best[k][2] > 0 for k in keys)
        hist.append(dict(gen=gen, t=time.time() - t0, n_sims=len(S_all), n_keys_violating=int(nviol),
                         best={f'{k[0]}|{k[1]}': best[k][0] for k in keys}))
        log(f'  gen {gen:2d}: {len(S_all)} sims, keys still violating hard constraints {nviol}/{len(keys)}, '
            f'mean best J {np.mean([best[k][0] for k in keys]):.3f}  ({time.time() - t0:.0f}s, total {time.time() - T0:.0f}s)')
        if time.time() - T0 > t_cap:
            log('  GLOBAL TIME CAP reached -- stopping this stage early'); break
    return {k: seeds[k[1]][k[0]] * np.exp(best[k][1].reshape(NG, 3)) for k in keys}, hist


def key_full(v):
    """Lexicographic key on the full verification score of one entry."""
    hard = len(v['fails'])
    return (hard, round(v['entry_ts_median'], 4), -v['gap_rel'])


def pick(p, C, lib, names, cand, seeds):
    """Verify each seed's candidate on the full grid; keep the better one per entry."""
    ver = {}
    for sd in seeds:
        trial = dict(lib); trial.update({nm: cand[(nm, sd)] for nm in names})
        ver[sd] = BL.verify(p, C, trial, names)
    chosen = {}
    for nm in names:
        sd = min(seeds, key=lambda s: key_full(ver[s][nm]))
        chosen[nm] = (sd, cand[(nm, sd)], ver[sd][nm], {s: ver[s][nm] for s in seeds})
    return chosen


def main():
    lin = Lin(); post_loss_angles(lin, g18)
    C = {c['name']: c for c in enumerate_configs(lin)}
    p = make_plant()
    pole = {nm: np.array(v['gains']) for nm, v in json.load(open('library_v2.json'))['entries'].items()}
    seeds = {'pole': pole, 'generalist': {nm: g18.copy() for nm in C}}
    lib = {'R0': g18.copy()}
    A = [nm for nm in C if C[nm]['regime'] in ('R1', 'R1d')]
    Bn = [nm for nm in C if C[nm]['regime'] == 'R2']
    log(f'stage A: {len(A)} entries x 2 seeds')
    candA, histA = tune_stage(p, C, lib, A, seeds, CAP_S * 0.80)
    chA = pick(p, C, lib, A, candA, seeds)
    for nm in A:
        lib[nm] = chA[nm][1]
    log(f'stage A picked: pole {sum(chA[n][0] == "pole" for n in A)}, generalist {sum(chA[n][0] == "generalist" for n in A)}; '
        f'failing {[n for n in A if chA[n][2]["fails"]]}  ({time.time() - T0:.0f}s)')
    log(f'stage B: {len(Bn)} entries x 2 seeds')
    candB, histB = tune_stage(p, C, lib, Bn, seeds, CAP_S)
    chB = pick(p, C, lib, Bn, candB, seeds)
    for nm in Bn:
        lib[nm] = chB[nm][1]
    names = A + Bn
    vf = BL.verify(p, C, lib, names)
    ch = {**chA, **chB}
    res = dict(spec='v2.1 / S26 peak-first per-configuration library', budget=dict(pop=POP, gens=GENS, sigma0=SIGMA0, n_tune=N_TUNE,
               cap_s=CAP_S, eff_w=EFF_W), runtime_s=time.time() - T0,
               entries={nm: dict(regime=C[nm]['regime'], family=C[nm]['family'], gains=np.round(lib[nm], 6).tolist(),
                                 seed_chosen=(ch[nm][0] if nm in ch else 'R0 = S18 generalist'),
                                 verify=vf.get(nm), verify_by_seed=(ch[nm][3] if nm in ch else None)) for nm in C},
               history=dict(A=histA, B=histB))
    json.dump(res, open('library_v21.json', 'w'), indent=1)
    ff = [n for n in names if vf[n]['fails']]
    act = [n for n in names if not vf[n]['inactive']]
    log(f'FINAL: failing {len(ff)} {ff}; active {len(act)}; gap (peak/ts) median over active '
        f'{np.median([vf[n]["gap_rel"] for n in act]) * 100:+.1f}%; min gap {min(vf[n]["gap_rel"] for n in names) * 100:+.1f}%; '
        f'bump max {max(vf[n]["bump_max"] for n in names):.1e}; runtime {time.time() - T0:.0f}s')


if __name__ == '__main__':
    main()
