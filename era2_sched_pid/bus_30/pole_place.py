"""
pole_place.py -- Stage 1 of Q9: per-class pole-placement seeds + the generalist baseline (Q7).

Linearised v2 plant (per machine i, around the pre-fault operating point delta0):
    d(delta)/dt   = wb * dw
    2H d(dw)/dt   = dPm - K delta - D dw,      K = dPe/d(delta) for the class's network
    dPm           = -Kp dw - Ki z - Kd d(dw)/dt,   z' = dw  (e = -dw)
Because z and delta both integrate dw, z = delta/wb exactly (same initial point), so
    (2H + Kd) d(dw)/dt = -(D + Kp) dw - (K + Ki/wb) delta
-> A(g) = [[0, wb I], [-Mi (K + diag(Ki)/wb), -Mi diag(D + Kp)]],  Mi = diag(1/(2H + Kd)).
Gen trip: the tripped machine's loop is frozen during the fault -> its gains are 0
in that linearisation (robust over which machine trips). The derivative filter,
rail, rate limit and envelope are invisible here; that is what Stage 2 polishes.

Design rule (dominant-pole region): every eigenvalue in Re <= -SIGMA and zeta >= ZETA,
reached with the smallest gains (soft constraints + small gain-norm penalty), gains
per machine (N-DoF), robust over every linearisation in the cell.

Cells: post (pre-fault network, all loops) | gen | load_a1/a2/a3 (alpha bands) | line.
Seed for an anchor = cell seed during the fault, `post` seed after clearing.
Generalist = ONE gain set robust over all cells together (pure pole placement, no polish).
Writes seeds_v2.json.
"""
import json
import os

import numpy as np
from scipy.optimize import minimize

from plant_v2 import PlantV2, NG, actuator_hash, PLANT_TAG

SIGMA = 0.5      # decay-rate target [1/s]; chosen by the time-domain sweep (see datagen notes)
ZETA = 0.3       # damping-ratio target (0.7 needs Kp~200 for no extra rescues)
MU = 1e-4        # gain-norm penalty (relative to G_SCALE)
G_SCALE = np.array([50.0, 50.0, 5.0])      # Kp, Ki, Kd scales for the penalty / init
LOG_LO, LOG_HI = np.log([0.5, 0.1, 0.01]), np.log([400.0, 2000.0, 100.0])
ALPHA_BANDS = {'load_a1': (0.10, 0.40), 'load_a2': (0.40, 0.70), 'load_a3': (0.70, 1.00)}


def k_matrix(M, N, phase, delta0):
    th = delta0 + phase
    dth = th[:, None] - th[None, :]
    Kij = M * np.sin(dth) - N * np.cos(dth)          # dPe_i/d delta_j, j != i
    np.fill_diagonal(Kij, 0.0)
    K = Kij.copy()
    K[np.diag_indices(NG)] = -Kij.sum(1)
    return K


class Lin:
    def __init__(self):
        p = PlantV2(device='cpu')
        c = p.c
        self.p = p
        self.H, self.D = c.H.numpy(), c.D.numpy()
        self.wb = c.wb
        self.phase, self.d0 = c.phase.numpy(), c.delta0.numpy()

    def K_of(self, typ, loc, alpha):
        M, N = self.p.fault_consts(typ, loc, alpha)
        return k_matrix(M.numpy(), N.numpy(), self.phase, self.d0)

    def A(self, K, g, off=None):
        g = g.copy()
        if off is not None:
            g[off] = 0.0
        Kp, Ki, Kd = g[:, 0], g[:, 1], g[:, 2]
        Mi = 1.0 / (2 * self.H + Kd)
        top = np.hstack([np.zeros((NG, NG)), self.wb * np.eye(NG)])
        bot = np.hstack([-Mi[:, None] * (K + np.diag(Ki) / self.wb), -np.diag(Mi * (self.D + Kp))])
        return np.vstack([top, bot])


def cells(lin):
    out = {'post': [(lin.K_of('gen', 1, 0.0), None)]}        # gen -> pre-fault network
    out['gen'] = [(lin.K_of('gen', 1, 0.0), j) for j in (0, 1)]          # trip G1 / G2
    for nm, (a0, a1) in ALPHA_BANDS.items():
        am = 0.5 * (a0 + a1)
        out[nm] = [(lin.K_of('load', b, am), None) for b in lin.p.c.load_bus]
    out['line'] = [(lin.K_of('line', int(i) + 1, 1.0), None) for i in lin.p.c.faultable]
    return out


def eig_cost(lin, Ks, g, sigma=None, zeta=None):
    sigma = SIGMA if sigma is None else sigma
    zeta = ZETA if zeta is None else zeta
    J, worst_s, worst_z = 0.0, np.inf, np.inf
    for K, off in Ks:
        lam = np.linalg.eigvals(lin.A(K, g, off))
        s = -lam.real
        z = s / np.maximum(np.abs(lam), 1e-12)
        J += np.sum(np.maximum(0.0, sigma - s) ** 2) + np.sum(np.maximum(0.0, zeta - z) ** 2)
        worst_s, worst_z = min(worst_s, s.min()), min(worst_z, z.min())
    return J / len(Ks), worst_s, worst_z


def design(lin, Ks, sigma=None, zeta=None, hi=None):
    def f(v):
        g = np.exp(v).reshape(NG, 3)
        J, _, _ = eig_cost(lin, Ks, g, sigma, zeta)
        return J + MU * np.sum((g / G_SCALE) ** 2)
    v0 = np.log(np.tile(G_SCALE * [0.4, 0.1, 0.2], (NG, 1))).ravel()      # neutral, fixed
    top = LOG_HI if hi is None else np.log(hi)
    v0 = np.minimum(v0, np.tile(top, NG) - 1e-6)
    bnds = list(zip(np.tile(LOG_LO, NG), np.tile(top, NG)))
    r = minimize(f, v0, method='L-BFGS-B', bounds=bnds, options=dict(maxiter=3000, maxfun=200000))
    g = np.exp(r.x).reshape(NG, 3)
    J, s, z = eig_cost(lin, Ks, g, sigma, zeta)
    return g, dict(J=float(J), min_decay=float(s), min_zeta=float(z), iters=int(r.nit))


def main():
    lin = Lin()
    C = cells(lin)
    res = {'plant': PLANT_TAG, 'actuator_hash': actuator_hash(),
           'design': dict(sigma=SIGMA, zeta=ZETA, mu=MU, g_scale=G_SCALE.tolist(),
                          linearisation='pre-fault operating point delta0, filter/rail/rate ignored'),
           'seeds': {}, 'report': {}}
    # open-loop reference (gains 0)
    for nm, Ks in C.items():
        _, s, z = eig_cost(lin, Ks, np.zeros((NG, 3)))
        print(f'open loop  {nm:8s}: min decay {s:+.3f}  min zeta {z:+.3f}')
    for nm, Ks in C.items():
        g, rep = design(lin, Ks)
        res['seeds'][nm] = g.tolist(); res['report'][nm] = rep
        print(f'seed {nm:8s}: decay {rep["min_decay"]:.3f} zeta {rep["min_zeta"]:.3f} J {rep["J"]:.2e}\n  '
              + '\n  '.join(f'G{i+1}: Kp {g[i,0]:7.2f} Ki {g[i,1]:7.2f} Kd {g[i,2]:6.2f}' for i in range(NG)))
    allK = [k for Ks in C.values() for k in Ks]
    g, rep = design(lin, allK)
    res['generalist'] = g.tolist(); res['report']['generalist'] = rep
    print(f'GENERALIST: decay {rep["min_decay"]:.3f} zeta {rep["min_zeta"]:.3f} J {rep["J"]:.2e}\n  '
          + '\n  '.join(f'G{i+1}: Kp {g[i,0]:7.2f} Ki {g[i,1]:7.2f} Kd {g[i,2]:6.2f}' for i in range(NG)))
    with open('seeds_v2.json', 'w') as fh:
        json.dump(res, fh, indent=1)


if __name__ == '__main__':
    main()
