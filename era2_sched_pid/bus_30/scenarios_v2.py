"""
scenarios_v2.py -- era-1 scenario generators (gen / load / line), same ranges as
bus_30/case30_generate_dataset.m, plus the era-2 line-oversampling hook (spec §4, §3 plan).

    gen  : machine in {G1, G2}, alpha ~ U[0.10, 1.00]
    load : one of the 21 load buses, alpha ~ U[0.10, 1.00]
    line : one of the 33 faultable branches, alpha = 1
    t_fault ~ U[0.50, 4.00],  duration ~ U[0.05, 3.00],  t_clear = t_fault + duration
"""
import numpy as np

T_FAULT = (0.50, 4.00)
DURATION = (0.05, 3.00)
ALPHA = {'gen': (0.10, 1.00), 'load': (0.10, 1.00), 'line': (1.0, 1.0)}


def candidates(plant):
    c = plant.c
    return {'gen': [1, 2], 'load': [int(b) for b in c.load_bus],
            'line': [int(i) + 1 for i in c.faultable]}


def draw(plant, n, seed, type_p=(1 / 3, 1 / 3, 1 / 3)):
    rng = np.random.default_rng(seed)
    cand = candidates(plant)
    types = rng.choice(['gen', 'load', 'line'], size=n, p=type_p)
    out = []
    for t in types:
        a0, a1 = ALPHA[t]
        tf = rng.uniform(*T_FAULT)
        out.append(dict(type=str(t), loc=int(rng.choice(cand[t])), alpha=float(rng.uniform(a0, a1)),
                        t_fault=float(tf), t_clear=float(tf + rng.uniform(*DURATION))))
    return out
