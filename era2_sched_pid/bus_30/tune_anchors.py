"""
tune_anchors.py -- Stage 2 of Q9: deterministic, frozen anchor tuner (spec §4).

Schedule parameterisation (37 numbers per anchor, all log-multipliers on the seeds):
    g_f  = seed[cell] * exp(theta_f)        (NG x 3)  fault-phase gains
    g_p  = seed['post'] * exp(theta_p)      (NG x 3)  post-clear / steady gains
    tau  = TAU0 * exp(theta_tau)                      taper time constant
On the global 20 Hz grid (tick k starts at t_k = k*T_LBL):
    t_k <  t_fault            : g_p   (pre-fault: plant at rest, e = 0, gains inert)
    t_fault <= t_k < t_clear  : g_f
    t_k >= t_clear            : g_p + (g_f - g_p) * exp(-(t_k - t_clear)/tau)
Labels = the resulting (n_lbl, NG, 3) array, i.e. exactly what the sim ran.

Objective (lexicographic via an offset):
    feasible   (peak |w-1| <= LIMIT over the run):  J = t_s + LAM * E
    infeasible                                   :  J = T_END + 1 + peak/LIMIT + LAM * E
    E   = TV + GAIN_W * mean(theta^2)   (Q6: actuator rate activity + gain magnitude;
          gain magnitude measured as log-deviation from the pole-placed seed)
    t_s = last instant max_j |w_j - 1| > SETTLE_BAND, measured from t_fault
    TV  = total variation of Pm over the run, summed over machines [pu]
Optimizer: (mu/mu_w, lambda)-CMA-ES, theta0 = 0 (= the pole-placed seed), sigma0,
population, generations and RNG seed (BASE_SEED + anchor id) fixed. No early stopping.
All anchors x population are simulated as one batch per generation.
"""
import argparse
import json
import time

import numpy as np
import torch

from plant_v2 import PlantV2, NG, T_LBL, T_END, LIMIT, actuator_hash, PLANT_TAG

SETTLE_BAND = 1e-3     # settling band for t_s [pu]   (pinned with LAM, see datagen notes)
GAIN_W = 1.0           # weight of the gain-magnitude part of the effort term
TAU0 = 0.5             # taper time-constant scale [s]
SIGMA0 = 0.5           # CMA-ES initial step (log-multiplier units)
POP = 16
GENS = 60
BASE_SEED = 20261121
DIM = 2 * NG * 3 + 1


def cell_of(s):
    if s['type'] == 'load':
        a = s['alpha']
        return 'load_a1' if a < 0.4 else ('load_a2' if a < 0.7 else 'load_a3')
    return s['type']


def schedule(theta, scen, seeds, n_lbl):
    """theta (B, DIM) numpy, scen list (B) -> gains (B, n_lbl, NG, 3) numpy."""
    B = theta.shape[0]
    tk = np.arange(n_lbl) * T_LBL
    G = np.empty((B, n_lbl, NG, 3))
    for b in range(B):
        s = scen[b]
        gf = np.asarray(seeds[cell_of(s)]) * np.exp(theta[b, :NG * 3].reshape(NG, 3))
        gp = np.asarray(seeds['post']) * np.exp(theta[b, NG * 3:2 * NG * 3].reshape(NG, 3))
        tau = TAU0 * np.exp(theta[b, -1])
        w = np.where(tk < s['t_fault'], 0.0,
                     np.where(tk < s['t_clear'], 1.0, np.exp(-(tk - s['t_clear']) / tau)))
        G[b] = gp[None] + (gf - gp)[None] * w[:, None, None]
    return G


def evaluate(plant, scen, theta, seeds, lam):
    G = torch.as_tensor(schedule(theta, scen, seeds, plant.n_lbl), device=plant.device, dtype=plant.dtype)
    plant.bands = (LIMIT, SETTLE_BAND)
    o = plant.simulate(plant.batch(scen), G)
    peak = o['peak'].cpu().numpy()
    tv = o['tv'].cpu().numpy()
    tf = np.array([s['t_fault'] for s in scen])
    ts = np.maximum(o['last_exit'][SETTLE_BAND].cpu().numpy() - tf, 0.0)
    feas = peak <= LIMIT
    eff = tv + GAIN_W * np.mean(theta ** 2, axis=1)
    J = np.where(feas, ts + lam * eff, T_END + 1 + peak / LIMIT + lam * eff)
    return J, dict(peak=peak, tv=tv, ts=ts, feas=feas, gmag=np.mean(theta ** 2, axis=1))


class CMA:
    """Plain (mu/mu_w, lambda)-CMA-ES (Hansen's tutorial defaults), deterministic per seed."""

    def __init__(self, dim, seed, sigma0=SIGMA0, pop=POP):
        n = dim
        self.n, self.lam, self.sigma = n, pop, sigma0
        self.rng = np.random.default_rng(seed)
        self.mu = pop // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w ** 2)
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.ds = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.m = np.zeros(n)
        self.pc, self.ps = np.zeros(n), np.zeros(n)
        self.C = np.eye(n)
        self.best_x, self.best_f = self.m.copy(), np.inf

    def ask(self):
        self.Dv, self.Bm = np.linalg.eigh(self.C)
        self.Dv = np.sqrt(np.maximum(self.Dv, 1e-20))
        self.z = self.rng.standard_normal((self.lam, self.n))
        self.y = (self.z * self.Dv) @ self.Bm.T
        return self.m + self.sigma * self.y

    def tell(self, X, f):
        idx = np.argsort(f, kind='stable')
        if f[idx[0]] < self.best_f:
            self.best_f, self.best_x = float(f[idx[0]]), X[idx[0]].copy()
        ysel = self.y[idx[:self.mu]]
        yw = self.w @ ysel
        self.m = self.m + self.sigma * yw
        Cinvsqrt = self.Bm @ np.diag(1 / self.Dv) @ self.Bm.T
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (Cinvsqrt @ yw)
        hs = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1))) < (1.4 + 2 / (self.n + 1)) * self.chiN
        self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        self.C = ((1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc) + (1 - hs) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * (ysel.T * self.w) @ ysel)
        self.sigma *= np.exp((self.cs / self.ds) * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.gen += 1

    gen = 0


def tune(plant, anchors, seeds, lams, gens=GENS, log=print):
    """anchors: list of scenario dicts with 'id'. lams: one lambda per anchor (same length)."""
    A = len(anchors)
    es = [CMA(DIM, BASE_SEED + int(a['id'])) for a in anchors]
    scen = [a for a in anchors for _ in range(POP + 1)]          # +1 slot: the current mean
    t0 = time.time()
    J0, info0 = evaluate(plant, anchors, np.zeros((A, DIM)), seeds, np.asarray(lams))
    for e, j in zip(es, J0):
        e.best_f, e.best_x = float(j), np.zeros(DIM)
    for g in range(gens):
        X = np.stack([np.vstack([e.ask(), e.m[None]]) for e in es])        # (A, POP+1, DIM)
        J, _ = evaluate(plant, scen, X.reshape(-1, DIM), seeds, np.repeat(lams, POP + 1))
        J = J.reshape(A, POP + 1)
        for i, e in enumerate(es):
            if J[i, -1] < e.best_f:
                e.best_f, e.best_x = float(J[i, -1]), X[i, -1].copy()
            e.tell(X[i, :POP], J[i, :POP])
        if g % 10 == 0 or g == gens - 1:
            log(f'  gen {g:3d}  median best J {np.median([e.best_f for e in es]):.3f}  ({time.time() - t0:.0f} s)')
    th = np.stack([e.best_x for e in es])
    J, info = evaluate(plant, anchors, th, seeds, np.asarray(lams))
    return th, J, info, info0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--anchors', required=True)
    ap.add_argument('--lam', type=float, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--gens', type=int, default=GENS)
    a = ap.parse_args()
    plant = PlantV2()
    seeds = json.load(open('seeds_v2.json'))['seeds']
    anchors = json.load(open(a.anchors))['anchors']
    th, J, info, info0 = tune(plant, anchors, seeds, np.full(len(anchors), a.lam), a.gens)
    G = schedule(th, anchors, seeds, plant.n_lbl)
    np.savez_compressed(a.out, theta=th, J=J, gains=G.astype(np.float32),
                        **{k: v for k, v in info.items()}, **{k + '_seed': v for k, v in info0.items()},
                        ids=np.array([x['id'] for x in anchors]))
    json.dump(dict(plant=PLANT_TAG, actuator_hash=actuator_hash(), lam=a.lam, settle_band=SETTLE_BAND,
                   tau0=TAU0, gain_w=GAIN_W, sigma0=SIGMA0, pop=POP, gens=a.gens, base_seed=BASE_SEED),
              open(a.out.replace('.npz', '_cfg.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
