"""
plant_v2.py -- era-2 governor-less 30-bus plant with per-machine PID (batched, torch).

Spec: PINN_Wiki specs/pid-scheduling-spec.md (307e9df + amendments 2026-11-21)
      and specs/pid-scheduling-answers.md (Q1-Q9).

Plant (v2-nogov): the era-1 swing + Kron-reduced network of bus_30/case30_physics.py,
governor (droop + Tg lag) deleted. Pm is the actuator output.

Controller: one PID per machine, all N = 6 machines (Q2), digital at the solver
step h = 1 ms:
    e      = 1 - omega_j
    xf    <- xf + h*N_D*(e - xf)               derivative filter (N_D rad/s)
    I     <- clip(I + h*Ki*e, lo, hi)          integrator carries Pm0, clamped at the rail
    P_set  = clip(Kp*e + I + Kd*N_D*(e - xf), lo, hi)
    Pm    <- Pm + clip(P_set - Pm, -RATE*h, +RATE*h)
with the rail lo/hi = Pm0 -/+ RAIL (Q1, deviation rail) and RATE = 2.5 pu/s.
Swing/angle states are then advanced one RK4 step with Pm held.

Gains are zero-order held on a GLOBAL 20 Hz grid (Q4): the gain array
G[b, k, j, 3] applies on [k*T_LBL, (k+1)*T_LBL). Labels = exactly what the sim used.

Faults (era-1 generators, unchanged electrical side):
    load : Y[b,b] += alpha during [t_fault, t_clear)
    line : branch open during [t_fault, t_clear)
    gen  : Q8 -- a plant disturbance that bypasses the actuator: machine j's
           mechanical input loses alpha*Pm0_j during the fault, and its PID/set-point
           path is frozen (integrator, filter and Pm held) for the fault duration.
           Survivors' PIDs (incl. the idle G3-G6 loops) carry the correction.
           [gen_mode='q8', legacy -- used for the wiki S7-S10 numbers]
    gen v1.3 (wiki S13, C-hybrid; gen_mode='v13'):
         alpha < 1 : connected runback -- mechanical input loses alpha*Pm0_j during
                     [t_fault, t_clear), restored at clearing; the unit's own PID
                     stays ACTIVE (no freeze). Network untouched.
         alpha = 1 : disconnection at t_fault, permanent -- machine j leaves the
                     reduced network (Kron reduction without its internal node, so
                     its Yred row/column is zero), its swing states and PID are
                     frozen, and its speed is excluded from the envelope.
    Scoring: peak (all machines in the network) and peak_excl (the faulted
    machine excluded from t_fault onward); for alpha = 1 both are the same.
"""
import hashlib
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BUS30 = os.path.normpath(os.path.join(HERE, '..', '..', 'bus_30'))
sys.path.insert(0, BUS30)
from case30_physics import CASE30, NG  # noqa: E402

PLANT_TAG = 'v2-nogov'
RAIL = 1.0          # |P_set - Pm0| <= RAIL  [pu]  (Q1)
RATE = 2.5          # |dP_set/dt| <= RATE   [pu/s] (Q1)
H_STEP = 1e-3       # solver / controller step [s]
LABEL_HZ = 20.0     # gain label rate (Q4)
T_LBL = 1.0 / LABEL_HZ
N_D = 50.0          # derivative filter bandwidth [rad/s]
T_END = 8.0         # same as era-1 datasets
LIMIT = 0.005       # omega envelope
TYPES = ['gen', 'load', 'line']


GEN_TRIP = {'q8': 'disturbance -alpha*Pm0, tripped PID frozen',
            'v13': 'S13 C-hybrid: runback alpha<1 PID active; alpha=1 permanent disconnection, speed excluded'}


def actuator_config(gen_mode='q8'):
    return dict(plant=PLANT_TAG, rail_dev=RAIL, rate=RATE, h=H_STEP, label_hz=LABEL_HZ,
                n_d=N_D, controller='digital PID per machine, I clamp at rail, ZOH gains',
                gen_trip=GEN_TRIP[gen_mode])


def actuator_hash(rail=RAIL, rate=RATE, gen_mode='q8'):
    cfg = actuator_config(gen_mode); cfg.update(rail_dev=rail, rate=rate)
    s = json.dumps(cfg, sort_keys=True).encode()
    return hashlib.sha1(s).hexdigest()[:12]


class PlantV2:
    def __init__(self, device='cuda', dtype=torch.float64, rail=RAIL, rate=RATE, gen_mode='q8'):
        """rail/rate override the module defaults (S9 matrix: rail=inf for round A; v1.2 rate=inf).
        gen_mode: 'q8' (legacy frozen-PID disturbance) | 'v13' (S13 C-hybrid)."""
        assert gen_mode in GEN_TRIP
        self.rail, self.rate, self.gen_mode = float(rail), float(rate), gen_mode
        cwd = os.getcwd()
        os.chdir(BUS30)
        try:
            self.c = CASE30(device=device, dtype=dtype)
        finally:
            os.chdir(cwd)
        c = self.c
        self.device, self.dtype = device, dtype
        self.Pm0 = c.Pm0.clone()
        self.lo = self.Pm0 - self.rail
        self.hi = self.Pm0 + self.rail
        self.M_pre, self.N_pre = c.pe_consts(c.Yred_pre)
        self.disc = {j: c.pe_consts(self.kron_drop(c.Y_pre, j)) for j in (1, 2)}
        self.n_lbl = int(round(T_END / T_LBL))
        self.steps_per_lbl = int(round(T_LBL / H_STEP))

    def kron_drop(self, Y, j):
        """Kron reduction with machine j (1-based) disconnected: its internal node and
        jXd' branch are removed, so its row/column of Yred is exactly zero and the other
        five entries equal the 5-machine reduction (loads at its bus stay in Y)."""
        c = self.c
        yg = 1.0 / (1j * torch.as_tensor(c.Xdp, dtype=c.cdtype, device=self.device))
        yg[j - 1] = 0
        idx = torch.tensor([b - 1 for b in c.gen_bus], device=self.device)
        Ybb = Y.clone()
        Ybb[idx, idx] = Ybb[idx, idx] + yg
        Ybg = torch.zeros(Y.shape[0], NG, dtype=c.cdtype, device=self.device)
        Ybg[idx, torch.arange(NG, device=self.device)] = -yg
        Vr = -torch.linalg.solve(Ybb, Ybg)
        return torch.diag(yg) + Ybg.T @ Vr

    # ---- scenarios ---------------------------------------------------------
    def fault_consts(self, typ, loc, alpha):
        """(M_f, N_f) for the faulted network; gen faults keep the pre-fault network."""
        c = self.c
        if typ == 'gen':
            return self.M_pre, self.N_pre
        Yf, _ = c.scenario(typ, int(loc), float(alpha))
        return c.pe_consts(Yf)

    def batch(self, scen):
        """scen: list of dicts with type, loc, alpha, t_fault, t_clear."""
        B = len(scen)
        dev, dt = self.device, self.dtype
        Mf = torch.empty(B, NG, NG, dtype=dt, device=dev)
        Nf = torch.empty_like(Mf)
        cache = {}
        trip = torch.zeros(B, NG, dtype=dt, device=dev)
        dead = torch.zeros(B, NG, dtype=torch.bool, device=dev)     # v13 alpha=1: disconnected
        own = torch.zeros(B, NG, dtype=torch.bool, device=dev)      # faulted machine (for peak_excl)
        tc_l = [s['t_clear'] for s in scen]
        for b, s in enumerate(scen):
            key = (s['type'], int(s['loc']), round(float(s['alpha']), 12))
            if key not in cache:
                cache[key] = self.fault_consts(*key)
            Mf[b], Nf[b] = cache[key]
            if s['type'] == 'gen':
                j = int(s['loc']) - 1
                own[b, j] = True
                if self.gen_mode == 'v13' and float(s['alpha']) >= 1.0 - 1e-12:
                    dead[b, j] = True
                    Mf[b], Nf[b] = self.disc[j + 1]
                    tc_l[b] = 1e9                                   # permanent: never cleared
                else:
                    trip[b, j] = float(s['alpha']) * self.Pm0[j]
        tf = torch.tensor([s['t_fault'] for s in scen], dtype=dt, device=dev)
        tc = torch.tensor(tc_l, dtype=dt, device=dev)
        frozen = (trip > 0) if self.gen_mode == 'q8' else dead     # PIDs held during the fault window
        return dict(Mf=Mf, Nf=Nf, trip=trip, frozen=frozen, dead=dead, own=own, tf=tf, tc=tc, B=B)

    # ---- dynamics ----------------------------------------------------------
    def _f(self, d, w, Pm, M, N):
        c = self.c
        th = d + c.phase
        dth = th.unsqueeze(2) - th.unsqueeze(1)
        Pe = (M * torch.cos(dth) + N * torch.sin(dth)).sum(-1)
        dw = w - 1.0
        return c.wb * dw, (Pm - Pe - c.D * dw) * c.i2H, Pe

    def simulate(self, bt, G, t_start=None, record=False, rec_every=1):
        """
        bt: from batch().  G: (B, n_lbl, NG, 3) gains [Kp, Ki, Kd] per 20 Hz tick.
        Starts from the exact pre-fault equilibrium at t_start (default: the earliest
        fault tick of the batch -- the plant is at rest before any fault, verified).
        Returns dict with per-scenario peak |w-1|, last-exit time t_s(band) helpers,
        total variation of Pm, and optionally trajectories.
        """
        c = self.c
        B = bt['B']
        dev, dt = self.device, self.dtype
        h = H_STEP
        if t_start is None:
            t_start = float(torch.floor(bt['tf'].min() / T_LBL) * T_LBL)
        k0 = int(round(t_start / h))
        n = int(round(T_END / h))
        d = c.delta0.expand(B, NG).clone()
        w = torch.ones(B, NG, dtype=dt, device=dev)
        I = self.Pm0.expand(B, NG).clone()
        xf = torch.zeros(B, NG, dtype=dt, device=dev)
        Pm = self.Pm0.expand(B, NG).clone()
        peak = torch.zeros(B, dtype=dt, device=dev)
        peak_x = torch.zeros(B, dtype=dt, device=dev)
        tv = torch.zeros(B, dtype=dt, device=dev)
        rec = {'t': [], 'w': [], 'Pm': [], 'Pe': [], 'd': [], 'I': []} if record else None
        last_exit = {}
        bands = getattr(self, 'bands', (LIMIT,))
        for bd in bands:
            last_exit[bd] = torch.zeros(B, dtype=dt, device=dev)
        last_exit_x = {bd: torch.zeros(B, dtype=dt, device=dev) for bd in bands}
        frozen = bt['frozen']
        dead_m, own = bt.get('dead'), bt.get('own')
        for k in range(k0, n):
            t = k * h
            tm = t + 0.5 * h
            faulted = (tm >= bt['tf']) & (tm < bt['tc'])              # (B,)
            fz = frozen & faulted.unsqueeze(1)                          # (B,NG)
            g = G[:, min(int((t + 1e-9) // T_LBL), G.shape[1] - 1)]     # (B,NG,3)
            Kp, Ki, Kd = g[..., 0], g[..., 1], g[..., 2]
            e = 1.0 - w
            dterm = N_D * (e - xf)
            xf_new = xf + h * N_D * (e - xf)
            I_new = torch.minimum(torch.maximum(I + h * Ki * e, self.lo), self.hi)
            Ps = torch.minimum(torch.maximum(Kp * e + I_new + Kd * dterm, self.lo), self.hi)
            Pm_new = Ps if self.rate == float('inf') else Pm + torch.clamp(Ps - Pm, -self.rate * h, self.rate * h)
            xf = torch.where(fz, xf, xf_new)
            I = torch.where(fz, I, I_new)
            Pm_new = torch.where(fz, Pm, Pm_new)
            tv = tv + (Pm_new - Pm).abs().sum(1)                       # actual Pm moves only
            Pm = Pm_new
            Pmech = Pm - bt['trip'] * faulted.unsqueeze(1).to(dt)
            fm = faulted.view(B, 1, 1)
            M = torch.where(fm, bt['Mf'], self.M_pre)
            N = torch.where(fm, bt['Nf'], self.N_pre)
            k1d, k1w, Pe = self._f(d, w, Pmech, M, N)
            k2d, k2w, _ = self._f(d + 0.5 * h * k1d, w + 0.5 * h * k1w, Pmech, M, N)
            k3d, k3w, _ = self._f(d + 0.5 * h * k2d, w + 0.5 * h * k2w, Pmech, M, N)
            k4d, k4w, _ = self._f(d + h * k3d, w + h * k3w, Pmech, M, N)
            live = 1.0
            if dead_m is not None:
                live = (~(dead_m & faulted.unsqueeze(1))).to(dt)        # disconnected machine: swing frozen
            d = d + live * (h / 6.0) * (k1d + 2 * k2d + 2 * k3d + k4d)
            w = w + live * (h / 6.0) * (k1w + 2 * k2w + 2 * k3w + k4w)
            adev = (w - 1.0).abs()
            dev_ = adev.amax(1)
            peak = torch.maximum(peak, dev_)
            if own is not None:
                post = own & (tm >= bt['tf']).unsqueeze(1)              # faulted machine, from t_fault on
                dev_x = torch.where(post, torch.zeros_like(adev), adev).amax(1)
            else:
                dev_x = dev_
            peak_x = torch.maximum(peak_x, dev_x)
            for bd in bands:
                last_exit[bd] = torch.where(dev_ > bd, torch.full_like(dev_, t + h), last_exit[bd])
                last_exit_x[bd] = torch.where(dev_x > bd, torch.full_like(dev_, t + h), last_exit_x[bd])
            if record and (k % rec_every == 0):
                rec['t'].append(t + h); rec['w'].append(w.clone()); rec['Pm'].append(Pm.clone())
                rec['Pe'].append(Pe.clone()); rec['d'].append(d.clone()); rec['I'].append(I.clone())
        out = dict(peak=peak, peak_excl=peak_x, tv=tv, last_exit=last_exit, last_exit_excl=last_exit_x, w_end=w, Pm_end=Pm, I_end=I)
        if record:
            out['rec'] = {kk: (torch.stack(v, 1) if kk != 't' else torch.tensor(v)) for kk, v in rec.items()}
        return out


def const_gains(B, n_lbl, g, device, dtype):
    """g: (NG,3) or (B,NG,3) constant over the episode -> (B, n_lbl, NG, 3)."""
    g = torch.as_tensor(g, dtype=dtype, device=device)
    if g.dim() == 2:
        g = g.expand(B, NG, 3)
    return g.unsqueeze(1).expand(B, n_lbl, NG, 3).contiguous()
