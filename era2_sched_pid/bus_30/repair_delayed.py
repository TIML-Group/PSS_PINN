"""
repair_delayed.py -- wiki S34(c): CMA-ES repair of the library entries that fail the delayed-entry verification
(reverify_delayed.py) at the ACCEPTANCE states L = 0 (oracle entry) and L = 28 ms (operating variant cls_learned_A).

Repair is from the CURRENT entry (log-multipliers on its gains), frozen S26 budget (pop 16 x 40 generations,
sigma0 0.3), stacked over all repaired entries, on the deterministic stride subset (<= 60 cases per entry) of
the S24 verification grid; every case is simulated at BOTH acceptance states. No softer-gains fallback and no
switch slew (S34): the mechanism is unchanged, only the entry's gains move.
Objective (penalty-ordered scalar, as S26):
  hard (x 100), at each acceptance state: median-objective floor vs the reference (R1/R1d: excluded peak vs the
        generalist; R2: t_s vs [R1, generalist]), no envelope loss vs the reference, needy rule (reference out of
        envelope -> every case must be in), survivors' rail-binding <= 5 % (S28), the entry's own transfer
        bump <= 0.05 pu;
  then  R1/R1d: median excluded peak at L = 28 ms relative to the reference (the operating-state headline),
        + 0.1 x the same at L = 0;  R2: mean t_s at L = 28 ms;
  then  0.01 x mean t_s + 1e-3 x mean TV(Pm) (tie-breakers).
R1/R1d entries are repaired first; R2 entries that still fail are then repaired against the repaired R1 entries.
Accept per entry only if its full-grid verification at the acceptance states has fewer failures than before
(a repaired entry that still fails is disclosed with its numbers, S34). Final: full re-verification of ALL entries
at L = 0 / 28 / 57 ms. Writes library_v22.json, repair_delayed.json; stdout -> repair_delayed.log.
"""
import json
import time

import numpy as np

import build_library as BL
import build_library_v21 as V21
from reverify_delayed import verify_at, switch_arrays, load_lib, context, ACCEPT_L
from sched_sim import simulate_sched
from plant_v2 import LIMIT, NG
from tune_anchors import CMA

POP, GENS, SIGMA0, SEED0, N_TUNE, HARD = 16, 40, 0.3, 20261123, 60, 100.0
g18 = BL.g18
T0 = time.time()


def log(*a):
    print(*a, flush=True)


def run(p, S, G):
    """simulate every case at every acceptance state; returns {L: output} (same case order)."""
    out = {}
    for L in ACCEPT_L:
        o = simulate_sched(p, S, np.asarray(G), t_sw=switch_arrays(S, L))
        out[L] = o
    return out


def objective(reg, o, idx, ref):
    viol = 0.0; key = 'ts' if reg == 'R2' else 'peak_excl'
    bump = 'bump_r2' if reg == 'R2' else 'bump_r1'
    for L in ACCEPT_L:
        v, r = o[L][key][idx], ref[L][key]
        pk, rpk = o[L]['peak_excl'][idx], ref[L]['peak_excl']
        viol += max(0.0, np.median(v) / max(np.median(r), 1e-12) - 1.0)
        viol += ((pk > LIMIT) & (rpk <= LIMIT)).mean()
        if (rpk > LIMIT).any():
            viol += np.maximum(0.0, pk / LIMIT - 1.0).mean() * 10
        viol += max(0.0, o[L]['rail_frac_surv'][idx].max() - BL.RAIL_MAX)
        viol += max(0.0, o[L][bump][idx].max() - BL.BUMP_MAX)
    L1 = ACCEPT_L[-1]
    if reg == 'R2':
        main = o[L1]['ts'][idx].mean()
    else:
        main = (np.median(o[L1]['peak_excl'][idx]) / max(np.median(ref[L1]['peak_excl']), 1e-12)
                + 0.1 * np.median(o[0.0]['peak_excl'][idx]) / max(np.median(ref[0.0]['peak_excl']), 1e-12))
    tie = 0.01 * o[L1]['ts'][idx].mean() + 1e-3 * o[L1]['tv'][idx].mean()
    return HARD * viol + main + tie, viol


def repair(p, C, lib, names):
    cases = {nm: V21.stride(V21.entry_cases(C, nm), N_TUNE) for nm in names}
    S_ref, G_ref, own = [], [], []
    for nm in names:
        for s in cases[nm]:
            S_ref.append(s); own.append(nm)
            G_ref.append([g18, g18, g18] if C[nm]['regime'] != 'R2' else [g18, lib[BL.r1_name(s)], g18])
    o = run(p, S_ref, G_ref); own = np.array(own)
    ref = {nm: {L: {k: o[L][k][own == nm] for k in ('peak_excl', 'ts')} for L in ACCEPT_L} for nm in names}
    es = {nm: CMA(NG * 3, SEED0 + i, sigma0=SIGMA0, pop=POP) for i, nm in enumerate(names)}
    best = {nm: (np.inf, np.zeros(NG * 3), np.inf) for nm in names}
    for gen in range(GENS):
        X = {nm: es[nm].ask() for nm in names}
        S_all, G_all, sl = [], [], {}
        for nm in names:
            for m in range(POP):
                gk = lib[nm] * np.exp(X[nm][m].reshape(NG, 3)); a = len(S_all)
                for s in cases[nm]:
                    S_all.append(s); G_all.append(V21.g3_for(C, lib, nm, s, gk))
                sl[(nm, m)] = slice(a, len(S_all))
        o = run(p, S_all, G_all)
        for nm in names:
            J = np.zeros(POP); V = np.zeros(POP)
            for m in range(POP):
                J[m], V[m] = objective(C[nm]['regime'], o, sl[(nm, m)], ref[nm])
            es[nm].tell(X[nm], J); i = int(np.argmin(J))
            if J[i] < best[nm][0]:
                best[nm] = (float(J[i]), X[nm][i].copy(), float(V[i]))
        if gen % 5 == 0 or gen == GENS - 1:
            log(f'  gen {gen:2d}: ' + '  '.join(f'{nm} J {best[nm][0]:.3f} viol {best[nm][2]:.3f}' for nm in names)
                + f'  ({time.time() - T0:.0f}s)')
    return {nm: lib[nm] * np.exp(best[nm][1].reshape(NG, 3)) for nm in names}


def nfail(v):
    return sum(len(v[L]['fails']) for L in v)


def stage(p, C, lib, names, tag):
    before = {nm: {L: verify_at(p, C, lib, [nm], L)[nm] for L in ACCEPT_L} for nm in names}
    log(f'{tag}: repairing {names}')
    cand = repair(p, C, lib, names)
    trial = dict(lib); trial.update(cand)
    after = {L: verify_at(p, C, trial, names, L) for L in ACCEPT_L}
    rep = {}
    for nm in names:
        a = {L: after[L][nm] for L in ACCEPT_L}
        ok = nfail(a) < nfail(before[nm])
        if ok:
            lib[nm] = cand[nm]
        rep[nm] = dict(accepted=bool(ok), gains_before=np.round(before_gains[nm], 6).tolist(), gains_after=np.round(cand[nm], 6).tolist(),
                       before={f'{L * 1000:.0f}': before[nm][L] for L in ACCEPT_L}, after={f'{L * 1000:.0f}': a[L] for L in ACCEPT_L})
        log(f'  {nm:18s} {"ACCEPTED" if ok else "rejected"}: ' + ' | '.join(
            f'L{L * 1000:.0f}: fails {before[nm][L]["fails"]} -> {a[L]["fails"]}, bump {before[nm][L]["bump_max"]:.3f} -> {a[L]["bump_max"]:.3f}, '
            f'median {before[nm][L]["entry_median"]:.3e} -> {a[L]["entry_median"]:.3e} (ref {a[L]["ref_median"]:.3e}), '
            f'in-env {a[L]["entry_in_env"]:.2f} (ref {a[L]["ref_in_env"]:.2f})' for L in ACCEPT_L))
    return rep


if __name__ == '__main__':
    C, p = context(); lib = load_lib('library_v21.json'); before_gains = {n: g.copy() for n, g in lib.items()}
    names = [n for n in C if n != 'R0']
    v0 = {L: verify_at(p, C, lib, names, L) for L in ACCEPT_L}
    failing = [n for n in names if any(v0[L][n]['fails'] for L in ACCEPT_L)]
    A = [n for n in failing if C[n]['regime'] in ('R1', 'R1d')]
    log(f'failing at the acceptance states: {failing}')
    report = {}
    if A:
        report.update(stage(p, C, lib, A, 'stage A (R1/R1d)'))
    vB = {L: verify_at(p, C, lib, [n for n in names if C[n]['regime'] == 'R2'], L) for L in ACCEPT_L}
    B = [n for n in vB[0.0] if any(vB[L][n]['fails'] for L in ACCEPT_L)]
    log(f'R2 entries failing after stage A: {B}')
    if B:
        report.update(stage(p, C, lib, B, 'stage B (R2)'))
    final = {f'{L * 1000:.0f}': verify_at(p, C, lib, names, L) for L in (0.0, 0.028, 0.057)}
    for Lk, v in final.items():
        ff = {n: v[n]['fails'] for n in names if v[n]['fails']}
        log(f'FINAL L = {Lk} ms: failing {len(ff)} {ff}; bump max {max(v[n]["bump_max"] for n in names):.3f}')
    r = json.load(open('library_v21.json'))
    for n in names:
        r['entries'][n]['gains'] = lib[n].tolist()                      # full precision (6-dp rounding moved the g18 fallbacks off g18)
        r['entries'][n]['verify_delayed'] = {Lk: final[Lk][n] for Lk in final}
        if n in report:
            r['entries'][n]['repair_delayed'] = dict(accepted=report[n]['accepted'])
    r['spec'] = 'v2.2 / S34(c): library v2.1 + delayed-entry re-verification and repair (acceptance L = 0, 28 ms)'
    r['repair_delayed'] = dict(budget=dict(pop=POP, gens=GENS, sigma0=SIGMA0, n_tune=N_TUNE), accept_L_ms=[0, 28],
                               repaired=list(report), runtime_s=time.time() - T0)
    json.dump(r, open('library_v22.json', 'w'), indent=1)
    json.dump(dict(report=report, final=final), open('repair_delayed.json', 'w'), indent=1)
    log(f'done ({time.time() - T0:.0f}s)')
