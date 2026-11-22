"""
library.py -- spec v2.0 gain library: configuration enumeration (S23/S24) + pole-placement design
with the inter-machine / common-mode target split.

Linearisation (pole_place.py model, z = delta/wb so Ki acts as synchronising torque):
    A = [[0, wb I], [-Mi (K + diag(Ki)/wb), -Mi diag(D + Kp)]],  Mi = 1/(2H + Kd)
restricted to the machines in the network (R1d drops the disconnected machine).
Common (frequency) mode: K has zero row sums, so for uniform motion
    Msum s^2 + Bsum s + Kisum = 0,  Msum = sum(2H+Kd), Bsum = sum(D+Kp), Kisum = sum(Ki)
its slow root s_c is matched to the nearest eigenvalue, which gets the lambda_c target; every other
eigenvalue gets the inter-machine targets (Re <= -sigma_em, zeta >= zeta_em). Among gains meeting
the targets the smallest are chosen (gain-norm penalty), robust over all linearisations of the entry.

Configurations (63): R0 1 | R1 line 33, load 21 (alpha collapsed, robust over alpha 0.1..1.0),
runback G1/G2 (alpha collapsed) | R1d G1/G2 (5-machine reduction at the post-loss operating point)
| R2 per preceding family {line, load, runback-G1, runback-G2} (restored = pre-fault network).
"""
import json

import numpy as np
from scipy.optimize import minimize

from plant_v2 import PlantV2, NG
from pole_place import k_matrix, G_SCALE, MU, LOG_LO, LOG_HI

LOAD_ALPHAS = (0.1, 0.4, 0.7, 1.0)


class Lin:
    def __init__(self, p=None):
        self.p = p if p is not None else PlantV2(device='cpu', rail=1.0, rate=float('inf'), gen_mode='v13')
        c = self.p.c
        self.H, self.D = c.H.cpu().numpy(), c.D.cpu().numpy()
        self.wb = c.wb
        self.phase, self.d0 = c.phase.cpu().numpy(), c.delta0.cpu().numpy()
        self.d_post = {}

    def K(self, typ, loc, alpha, d=None):
        M, N = self.p.fault_consts(typ, loc, alpha)
        return k_matrix(M.cpu().numpy(), N.cpu().numpy(), self.phase, self.d0 if d is None else d)

    def K_disc(self, j):
        M, N = self.p.disc[j]
        return k_matrix(M.cpu().numpy(), N.cpu().numpy(), self.phase, self.d_post.get(j, self.d0))

    def A(self, K, g, live):
        idx = np.where(live)[0]; n = len(idx)
        g = g[idx]; Kr = K[np.ix_(idx, idx)]
        Kp, Ki, Kd = g[:, 0], g[:, 1], g[:, 2]
        Mi = 1.0 / (2 * self.H[idx] + Kd)
        top = np.hstack([np.zeros((n, n)), self.wb * np.eye(n)])
        bot = np.hstack([-Mi[:, None] * (Kr + np.diag(Ki) / self.wb), -np.diag(Mi * (self.D[idx] + Kp))])
        return np.vstack([top, bot])

    def common_root(self, g, live):
        idx = np.where(live)[0]
        Ms = (2 * self.H[idx] + g[idx, 2]).sum(); Bs = (self.D[idx] + g[idx, 0]).sum(); Ks = g[idx, 1].sum()
        disc = Bs ** 2 - 4 * Ms * Ks
        return (-Bs + np.sqrt(disc)) / (2 * Ms) if disc >= 0 else complex(-Bs / (2 * Ms), np.sqrt(-disc) / (2 * Ms))

    def modes(self, K, g, live):
        lam = np.linalg.eigvals(self.A(K, g, live))
        sc = self.common_root(g, live)
        ic = int(np.argmin(np.abs(lam - sc)))
        return lam, ic


def cost(lin, Ks, g, tgt):
    """tgt: dict sigma_em, zeta_em, lam_c (target Re(lambda_c) <= -lam_c)."""
    J = 0.0; rep = dict(min_decay_em=np.inf, min_zeta_em=np.inf, lam_c=-np.inf)
    for K, live in Ks:
        lam, ic = lin.modes(K, g, live)
        s = -lam.real; z = s / np.maximum(np.abs(lam), 1e-12)
        m = np.ones(len(lam), bool); m[ic] = False
        J += np.sum(np.maximum(0, tgt['sigma_em'] - s[m]) ** 2) + np.sum(np.maximum(0, tgt['zeta_em'] - z[m]) ** 2)
        J += np.maximum(0, tgt['lam_c'] - s[ic]) ** 2 + np.maximum(0, 0.0 - z[ic]) ** 2
        rep['min_decay_em'] = min(rep['min_decay_em'], s[m].min()); rep['min_zeta_em'] = min(rep['min_zeta_em'], z[m].min())
        rep['lam_c'] = max(rep['lam_c'], float(lam[ic].real))
    return J / len(Ks), rep


def design(lin, Ks, tgt, g0=None):
    def f(v):
        g = np.exp(v).reshape(NG, 3)
        return cost(lin, Ks, g, tgt)[0] + MU * np.sum((g / G_SCALE) ** 2)
    v0 = np.log(np.tile(G_SCALE * [0.4, 0.1, 0.2], (NG, 1))).ravel() if g0 is None else np.log(np.maximum(g0, 1e-3)).ravel()
    v0 = np.clip(v0, np.tile(LOG_LO, NG) + 1e-6, np.tile(LOG_HI, NG) - 1e-6)
    r = minimize(f, v0, method='L-BFGS-B', bounds=list(zip(np.tile(LOG_LO, NG), np.tile(LOG_HI, NG))),
                 options=dict(maxiter=3000, maxfun=200000))
    g = np.exp(r.x).reshape(NG, 3)
    J, rep = cost(lin, Ks, g, tgt)
    rep = {k: float(v) for k, v in rep.items()}; rep['J'] = float(J)
    return g, rep


def enumerate_configs(lin):
    """-> list of dicts: name, regime, family, lin list [(K, live)]."""
    c = lin.p.c
    allm = np.ones(NG, bool)
    Kpre = lin.K('gen', 1, 0.0)
    cfg = [dict(name='R0', regime='R0', family='none', Ks=[(Kpre, allm)])]
    for i in c.faultable:
        cfg.append(dict(name=f'R1_line_{int(i) + 1}', regime='R1', family='line', loc=int(i) + 1,
                        Ks=[(lin.K('line', int(i) + 1, 1.0), allm)]))
    for b in c.load_bus:
        cfg.append(dict(name=f'R1_load_{int(b)}', regime='R1', family='load', loc=int(b),
                        Ks=[(lin.K('load', int(b), a), allm) for a in LOAD_ALPHAS]))
    for j in (1, 2):
        cfg.append(dict(name=f'R1_runback_G{j}', regime='R1', family=f'runback_G{j}', loc=j, Ks=[(Kpre, allm)]))
    for j in (1, 2):
        live = allm.copy(); live[j - 1] = False
        cfg.append(dict(name=f'R1d_G{j}', regime='R1d', family=f'disc_G{j}', loc=j, Ks=[(lin.K_disc(j), live)]))
    for fam in ('line', 'load', 'runback_G1', 'runback_G2'):
        cfg.append(dict(name=f'R2_{fam}', regime='R2', family=fam, Ks=[(Kpre, allm)]))
    return cfg


def post_loss_angles(lin, gains, device='cuda'):
    """Operating point after a permanent disconnection: rotor angles at 8 s of a baseline run."""
    p = PlantV2(device=device, rail=1.0, rate=float('inf'), gen_mode='v13')
    from plant_v2 import const_gains
    sc = [dict(type='gen', loc=j, alpha=1.0, t_fault=1.0, t_clear=2.0) for j in (1, 2)]
    o = p.simulate(p.batch(sc), const_gains(2, p.n_lbl, gains, p.device, p.dtype), record=True, rec_every=100)
    d_end = o['rec']['d'][:, -1].cpu().numpy()
    for i, j in enumerate((1, 2)):
        lin.d_post[j] = d_end[i]
    return lin.d_post


def design_best(lin, Ks, tgt, starts=()):
    """Multi-start design (default start + given warm starts); keeps the lowest-J, then smallest-norm result."""
    best = None
    for g0 in (None,) + tuple(starts):
        g, rep = design(lin, Ks, tgt, g0)
        key = (round(rep['J'], 6), float(np.sum((g / G_SCALE) ** 2)))
        if best is None or key < best[0]:
            best = (key, g, rep)
    return best[1], best[2]
