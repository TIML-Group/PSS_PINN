"""
build_library.py -- spec v2.0 / S23-S24 gain-library build: design -> verification (+ generalist->entry
gap) -> repair (<= 20 % of entries, <= 3 h) -> final verification. CUDA batched.

Targets: targets_v2.json (frozen from lamc_sweep.json, S24). R0 = S18 baseline (not designed).
Verification grid per entry (S23 item 4): 6 durations x 5 alpha (where alpha applies) x 2 fault
instants + the matching seed-7 cases. Schedules (oracle switching at t_fault):
  R1 / R1d entry e : [g18, e, g18]  vs  baseline [g18, g18, g18]   objective: peak (excluded)
  R2 entry r       : [g18, R1 entry of the case, r]  vs  [g18, R1 entry, g18]   objective: t_s
Fail (S23): out of envelope where the reference is in | median objective worse than the reference |
max rail-binding > 5 % | max transition bump > 0.05 pu.
No improvement (S24 addition): relative gap on the median objective < 1 %, unless the configuration
is inactive (reference median peak < 5e-4 pu = 10 % of the limit, or R2 reference median t_s < 0.05 s):
inactive entries are logged, not repaired (nothing to improve).
Repair: frozen CMA-ES (tune_anchors.CMA, pop 16, 40 generations, seed 20261124), stacked over all
repaired entries in one batch; parameters = log-multipliers on the pole-placed entry; objective =
mean objective (peak/LIMIT or t_s) + 10 x (envelope losses vs reference) + 10 x rail excess
+ LAMBDA x mean(theta^2) (LAMBDA = 0.03, S4 provisional; logged).
Writes library_v2.json (+ build log).
"""
import json
import sys
import time

import numpy as np
import scipy.io as sio

from library import Lin, enumerate_configs, design_best, post_loss_angles
from sched_sim import make_plant, simulate_sched
from plant_v2 import LIMIT, NG
from tune_anchors import CMA

DUR6 = (0.05, 0.1, 0.5, 1.0, 2.0, 3.0)
A_LOAD = (0.1, 0.3, 0.5, 0.7, 1.0)
A_RUN = (0.1, 0.3, 0.5, 0.7, 0.95)
TF2 = (1.0, 2.5)
INACTIVE_PEAK, INACTIVE_TS, GAP_MIN = 5e-4, 0.05, 0.01
RAIL_MAX, BUMP_MAX = 0.05, 0.05
POP, GENS, CMA_SEED, SIGMA0, LAMBDA = 16, 40, 20261124, 0.3, 0.03
CAP_FRAC, CAP_S = 0.20, 3 * 3600
MAX_REP_CASES = 120          # per entry per CMA member (stride subset of its verification cases)

g18 = np.array(json.load(open('baseline_ceiling_v13x.json'))['gains'])
TG = json.load(open('targets_v2.json'))
M7 = sio.loadmat('matrix_scen.mat', squeeze_me=True); NE = int(M7['n_eval'])
T7 = {1: 'gen', 2: 'load', 3: 'line'}
S7 = [dict(type=T7[int(M7['typ'][k])], loc=int(M7['loc'][k]), alpha=float(M7['alpha'][k]),
           t_fault=float(M7['tf'][k]), t_clear=float(M7['tc'][k])) for k in range(NE)]


def log(*a):
    print(*a, flush=True)


def ver_cases(cf):
    reg, fam = cf['regime'], cf['family']
    if reg == 'R1d':
        return [dict(type='gen', loc=cf['loc'], alpha=1.0, t_fault=tf, t_clear=tf + 1.0) for tf in (0.5, 1.0, 2.5, 4.0)]
    if fam == 'line':
        g = [dict(type='line', loc=cf['loc'], alpha=1.0, t_fault=tf, t_clear=tf + d) for tf in TF2 for d in DUR6]
        return g + [s for s in S7 if s['type'] == 'line' and s['loc'] == cf['loc']]
    if fam == 'load':
        g = [dict(type='load', loc=cf['loc'], alpha=a, t_fault=tf, t_clear=tf + d) for tf in TF2 for a in A_LOAD for d in DUR6]
        return g + [s for s in S7 if s['type'] == 'load' and s['loc'] == cf['loc']]
    j = cf['loc']
    g = [dict(type='gen', loc=j, alpha=a, t_fault=tf, t_clear=tf + d) for tf in TF2 for a in A_RUN for d in DUR6]
    return g + [s for s in S7 if s['type'] == 'gen' and s['loc'] == j and s['alpha'] < 1]


def r1_name(s):
    if s['type'] == 'line':
        return f"R1_line_{s['loc']}"
    if s['type'] == 'load':
        return f"R1_load_{s['loc']}"
    return f"R1d_G{s['loc']}" if s['alpha'] >= 1 - 1e-12 else f"R1_runback_G{s['loc']}"


def r2_name(s):
    return 'R2_' + ({'line': 'line', 'load': 'load'}.get(s['type']) or f"runback_G{s['loc']}")


def schedules(C, lib, names, override=None):
    """Build the case list + G3 for verifying `names`; override: {name: gains} replaces lib[name]."""
    G = dict(lib); G.update(override or {})
    S_all, G_all, R_all, owner = [], [], [], []
    for nm in names:
        cf = C[nm]
        if cf['regime'] in ('R1', 'R1d'):
            S = ver_cases(cf)
            for s in S:
                S_all += [s, s]; G_all += [[g18, G[nm], g18], [g18, g18, g18]]; R_all += ['e', 'r']; owner += [nm, nm]
        else:
            fam = cf['family']
            S = [s for c2 in C.values() if c2['regime'] == 'R1' and c2['family'] == fam for s in ver_cases(c2)]
            S = [s for s in S if s['type'] != 'gen' or s['alpha'] < 1]
            for s in S:
                g1 = G[r1_name(s)]
                S_all += [s, s]; G_all += [[g18, g1, G[nm]], [g18, g1, g18]]; R_all += ['e', 'r']; owner += [nm, nm]
    return S_all, np.array(G_all), np.array(R_all), np.array(owner)


def score(C, names, S, R, owner, o):
    out = {}
    for nm in names:
        m = owner == nm; e = m & (R == 'e'); r = m & (R == 'r')
        reg = C[nm]['regime']
        obj = 'ts' if reg == 'R2' else 'peak_excl'
        oe, orf = o[obj][e], o[obj][r]
        pe, pr = o['peak_excl'][e], o['peak_excl'][r]
        loss = int(((pe > LIMIT) & (pr <= LIMIT)).sum())
        med_e, med_r = float(np.median(oe)), float(np.median(orf))
        inactive = (med_r < INACTIVE_TS) if reg == 'R2' else (float(np.median(pr)) < INACTIVE_PEAK)
        gap = (med_r - med_e) / max(med_r, 1e-12)
        fails = []
        if loss: fails.append(f'envelope_loss x{loss}')
        if med_e > med_r * (1 + 1e-9) and not inactive: fails.append('worse_than_reference')
        if o['rail_frac'][e].max() > RAIL_MAX: fails.append('rail>5%')
        if o['bump'][e].max() > BUMP_MAX: fails.append('bump>0.05')
        out[nm] = dict(n=int(e.sum()), objective=obj, entry_median=med_e, ref_median=med_r, gap_rel=float(gap),
                       entry_in_env=float((pe <= LIMIT).mean()), ref_in_env=float((pr <= LIMIT).mean()),
                       entry_peak_max=float(pe.max()), ref_peak_max=float(pr.max()),
                       entry_ts_median=float(np.median(o['ts'][e])), ref_ts_median=float(np.median(o['ts'][r])),
                       rail_max=float(o['rail_frac'][e].max()), bump_max=float(o['bump'][e].max()),
                       dps_ordinary_max=float(o['dps_ordinary'][e].max()),
                       inactive=bool(inactive), fails=fails,
                       no_improvement=bool((not inactive) and gap < GAP_MIN and not fails))
    return out


def verify(p, C, lib, names):
    S, G, R, owner = schedules(C, lib, names)
    o = simulate_sched(p, S, G)
    return score(C, names, S, R, owner, o)


def repair(p, C, lib, names, t_budget):
    """Stacked CMA-ES over all `names`; returns {name: gains}, history."""
    es = {nm: CMA(NG * 3, CMA_SEED + i, sigma0=SIGMA0, pop=POP) for i, nm in enumerate(names)}
    base = {nm: lib[nm].copy() for nm in names}
    best = {nm: (np.inf, np.zeros(NG * 3)) for nm in names}
    hist = []; t0 = time.time()
    for gen in range(GENS):
        X = {nm: es[nm].ask() for nm in names}
        S_all, G_all, R_all, own_all, mem_all = [], [], [], [], []
        for nm in names:
            for k in range(POP):
                gk = base[nm] * np.exp(X[nm][k].reshape(NG, 3))
                S, G, R, owner = schedules(C, lib, [nm], override={nm: gk})
                if len(S) > 2 * MAX_REP_CASES:        # deterministic stride subset of (entry, ref) pairs
                    st = int(np.ceil(len(S) / (2 * MAX_REP_CASES)))
                    keep = np.concatenate([[i, i + 1] for i in range(0, len(S), 2 * st)])
                    S = [S[i] for i in keep]; G, R, owner = G[keep], R[keep], owner[keep]
                S_all += S; G_all.append(G); R_all.append(R); own_all.append(owner); mem_all.append(np.full(len(S), k))
        G_all = np.concatenate(G_all); R_all = np.concatenate(R_all); own_all = np.concatenate(own_all); mem_all = np.concatenate(mem_all)
        o = simulate_sched(p, S_all, G_all)
        for nm in names:
            reg = C[nm]['regime']; obj = 'ts' if reg == 'R2' else 'peak_excl'
            J = np.zeros(POP)
            for k in range(POP):
                e = (own_all == nm) & (mem_all == k) & (R_all == 'e'); r = (own_all == nm) & (mem_all == k) & (R_all == 'r')
                val = o[obj][e].mean() / (LIMIT if obj == 'peak_excl' else 1.0)
                loss = ((o['peak_excl'][e] > LIMIT) & (o['peak_excl'][r] <= LIMIT)).mean()
                rail = max(0.0, o['rail_frac'][e].max() - RAIL_MAX)
                J[k] = val + 10 * loss + 10 * rail + LAMBDA * np.mean(X[nm][k] ** 2)
            es[nm].tell(X[nm], J)
            i = int(np.argmin(J))
            if J[i] < best[nm][0]:
                best[nm] = (float(J[i]), X[nm][i].copy())
        hist.append(dict(gen=gen, best={nm: best[nm][0] for nm in names}, t=time.time() - t0))
        log(f'  repair gen {gen:2d}  ' + '  '.join(f'{nm}:{best[nm][0]:.4f}' for nm in names) + f'  ({time.time() - t0:.0f}s)')
        if time.time() - t0 > t_budget:
            log('  repair time cap reached'); break
    return {nm: base[nm] * np.exp(best[nm][1].reshape(NG, 3)) for nm in names}, hist


def main():
    t_start = time.time()
    lin = Lin(); post_loss_angles(lin, g18)
    C = {c['name']: c for c in enumerate_configs(lin)}
    p = make_plant()
    # ---- 1. design ----
    lib, design_rep = {'R0': g18.copy()}, {'R0': 'S18 baseline (not designed)'}
    for nm, cf in C.items():
        if cf['regime'] == 'R0':
            continue
        tg = TG[cf['regime']]
        g, rep = design_best(lin, cf['Ks'], dict(sigma_em=TG['sigma_em'], zeta_em=tg['zeta'], lam_c=tg['lam_c']), starts=(g18,))
        lib[nm], design_rep[nm] = g, rep
    log(f'designed {len(lib) - 1} entries in {time.time() - t_start:.0f}s; '
        f"max J {max(r['J'] for k, r in design_rep.items() if k != 'R0'):.2e}")
    names = [nm for nm in C if nm != 'R0']
    # ---- 2. verification ----
    t1 = time.time()
    v0 = verify(p, C, lib, names)
    log(f'verification pass 1: {time.time() - t1:.0f}s')
    fail = [nm for nm in names if v0[nm]['fails']]
    noimp = [nm for nm in names if v0[nm]['no_improvement']]
    inact = [nm for nm in names if v0[nm]['inactive']]
    log(f'fail {len(fail)}: {fail}\nno-improvement {len(noimp)}: {noimp}\ninactive {len(inact)}')
    cap = int(np.floor(CAP_FRAC * len(C)))
    rep_list = fail + noimp
    status = 'ok'
    if len(fail) > cap:
        status = f'STOP: {len(fail)} failing entries > cap {cap} -- revisit targets (S23 stop rule)'
        log(status)
    elif len(rep_list) > cap:
        status = f'repair candidates {len(rep_list)} > cap {cap}: failures first, then largest-gap-deficit no-improvement entries'
        noimp_sorted = sorted(noimp, key=lambda nm: v0[nm]['gap_rel'])
        rep_list = fail + noimp_sorted[:cap - len(fail)]
        log(status)
    repaired, hist = {}, []
    if not status.startswith('STOP') and rep_list:
        log(f'repairing {len(rep_list)}: {rep_list}')
        repaired, hist = repair(p, C, lib, rep_list, CAP_S - (time.time() - t_start))
        trial = dict(lib); trial.update(repaired)
        vr = verify(p, C, trial, rep_list)
        for nm in rep_list:   # keep the repair only if it does not make the entry worse
            if (len(vr[nm]['fails']) <= len(v0[nm]['fails'])) and vr[nm]['entry_median'] <= v0[nm]['entry_median']:
                lib[nm] = repaired[nm]
            else:
                repaired.pop(nm)
    # ---- 3. final verification (all) ----
    vf = verify(p, C, lib, names)
    res = dict(status=status, targets=TG, baseline='S18 ceiling re-tune (R0)', lambda_effort=LAMBDA,
               cap_entries=cap, cap_seconds=CAP_S, runtime_s=time.time() - t_start,
               entries={nm: dict(regime=C[nm]['regime'], family=C[nm]['family'],
                                 gains=np.round(lib[nm], 6).tolist(), design=design_rep[nm],
                                 verify_initial=v0.get(nm), verify_final=vf.get(nm),
                                 repaired=nm in repaired) for nm in C},
               repair_history=hist)
    json.dump(res, open('library_v2.json', 'w'), indent=1)
    ff = [nm for nm in names if vf[nm]['fails']]; ni = [nm for nm in names if vf[nm]['no_improvement']]
    log(f'FINAL: fails {len(ff)} {ff} | no-improvement {len(ni)} {ni} | repaired {list(repaired)} | {time.time() - t_start:.0f}s')


if __name__ == '__main__':
    main()
