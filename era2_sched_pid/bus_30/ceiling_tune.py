"""
ceiling_tune.py -- Q7 rev 2: tune the generalist baseline to its no-schedule ceiling.

ONE constant gain set (NG x 3), no schedule. Parameters: theta = log-multipliers on the
pure pole-placement generalist (seeds_v2.json 'generalist'); theta0 = 0.
Deterministic protocol, as for the anchor tuner: CMA-ES (tune_anchors.CMA), fixed seed,
sigma0, population, generation budget, no early stopping.

Objective (minimise), judged over the whole run:
    J = -min_c rate_c                          worst-class in-envelope rate (gen/load/line)
        + EXC_W * mean_c mean_s max(0, log(peak_s / LIMIT))     smooth tie-break on failures
        + EFF_W * mean_s TV_s                  tiny actuator-activity tie-break
The tie-break weights are small enough that a single extra rescue in the worst class
(1/n_c >= 1/200) always outweighs them.

Tuning sample: its own era-1-mix draw, class-balanced (TUNE_N per class, seed TUNE_SEED),
disjoint from the 1,200-scenario acceptance sample (seed 7), where the result is measured.
Writes baseline_ceiling.json.

Round 'v13x' (wiki S15 / spec v2.0): v1.3 gen semantics (runback alpha<1, own PID active),
rail 1.0, no rate limit, EXCLUDED scoring (faulted machine's speed dropped from t_fault on).
Tuning sample has only partial losses (alpha ~ U[0.1,1.0], asserted < 1); alpha = 1
disconnections are held pending the reconnection ruling. Writes baseline_ceiling_v13x.json.
"""
import json
import sys
import time

import numpy as np
import torch

from plant_v2 import PlantV2, const_gains, NG, LIMIT, actuator_hash, PLANT_TAG
from scenarios_v2 import draw
from tune_anchors import CMA

TUNE_SEED, TUNE_N = 101, 200          # per class
SEED, SIGMA0, POP, GENS = 20261122, 0.5, 16, 80
EXC_W, EFF_W = 1e-3, 1e-4
TYPES = ['gen', 'load', 'line']


def sample(p):
    out = []
    for i, t in enumerate(TYPES):
        oh = [0.0, 0.0, 0.0]; oh[i] = 1.0
        out += draw(p, TUNE_N, TUNE_SEED + i, type_p=oh)
    return out


def main():
    rnd = sys.argv[1] if len(sys.argv) > 1 else 'rate2.5'      # 'A' no rail | 'B' rail 1.0 | 'v13x' | 'rate2.5' (S7, superseded)
    kw = {'A': dict(rail=float('inf'), rate=float('inf')), 'B': dict(rail=1.0, rate=float('inf')),
          'v13x': dict(rail=1.0, rate=float('inf'), gen_mode='v13')}.get(rnd, {})
    p = PlantV2(**kw)
    pk_key = 'peak_excl' if rnd == 'v13x' else 'peak'
    g0 = np.array(json.load(open('seeds_v2.json'))['generalist'])
    S = sample(p); n = len(S)
    assert all(s['alpha'] < 1.0 for s in S if s['type'] == 'gen')
    ty = np.array([s['type'] for s in S])
    bt_one = p.batch(S)
    big = {k: (torch.cat([v] * POP) if torch.is_tensor(v) else v) for k, v in bt_one.items()}
    big['B'] = n * POP

    def score(Th):
        G = torch.cat([const_gains(n, p.n_lbl, g0 * np.exp(th.reshape(NG, 3)), p.device, p.dtype) for th in Th])
        bt = big if len(Th) == POP else {k: (torch.cat([v] * len(Th)) if torch.is_tensor(v) else v)
                                          for k, v in bt_one.items()} | {'B': n * len(Th)}
        o = p.simulate(bt, G)
        pk = o[pk_key].cpu().numpy().reshape(len(Th), n)
        tv = o['tv'].cpu().numpy().reshape(len(Th), n)
        rates = np.stack([(pk[:, ty == t] <= LIMIT).mean(1) for t in TYPES], 1)
        exc = np.stack([np.maximum(0, np.log(pk[:, ty == t] / LIMIT)).mean(1) for t in TYPES], 1).mean(1)
        J = -rates.min(1) + EXC_W * exc + EFF_W * tv.mean(1)
        return J, rates, pk

    t0 = time.time()
    J0, r0, _ = score(np.zeros((1, NG * 3)))
    print(f'start (pole placement): rates gen/load/line {r0[0].round(3)}  J {J0[0]:.4f}')
    es = CMA(NG * 3, SEED, sigma0=SIGMA0, pop=POP)
    es.best_f, es.best_x = float(J0[0]), np.zeros(NG * 3)
    hist = []
    for g in range(GENS):
        X = es.ask()
        J, R, _ = score(X)
        es.tell(X, J)
        hist.append(dict(gen=g, best_J=es.best_f, gen_best_rates=R[np.argmin(J)].tolist()))
        if g % 5 == 0 or g == GENS - 1:
            print(f'gen {g:3d}  best J {es.best_f:.4f}  gen-best rates {R[np.argmin(J)].round(3)}  sigma {es.sigma:.3f}  ({time.time() - t0:.0f} s)', flush=True)
    J1, r1, _ = score(es.best_x[None])
    g1 = g0 * np.exp(es.best_x.reshape(NG, 3))
    res = dict(plant=PLANT_TAG, round=rnd, rail=p.rail, rate=p.rate, gen_mode=p.gen_mode, scoring=pk_key,
               actuator_hash=actuator_hash(p.rail, p.rate, p.gen_mode),
               protocol=dict(tune_seed=TUNE_SEED, tune_n_per_class=TUNE_N, cma_seed=SEED, sigma0=SIGMA0,
                             pop=POP, gens=GENS, exc_w=EXC_W, eff_w=EFF_W, init='pole-placement generalist'),
               gains=g1.tolist(), theta=es.best_x.tolist(),
               tune_rates_start=dict(zip(TYPES, r0[0].tolist())), tune_rates_end=dict(zip(TYPES, r1[0].tolist())),
               history=hist)
    json.dump(res, open('baseline_ceiling.json' if rnd == 'rate2.5' else f'baseline_ceiling_{rnd}.json', 'w'), indent=1)
    if rnd in ('A', 'B', 'v13x'):
        import scipy.io as sio
        sio.savemat(f'ceiling_{rnd}.mat', dict(G=g1))
    print('final tune-sample rates', dict(zip(TYPES, r1[0].round(3))))
    print('gains\n', np.round(g1, 2))


if __name__ == '__main__':
    main()
