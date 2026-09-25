"""
case30_physics.py
=================
Differentiable torch port of the IEEE 30-bus model in `case30_lib.m`.

Ported piece by piece from the MATLAB source so the rollout inside the network
is the same system that generated the dataset:

    kron_reduce  -> CASE30.kron
    open_branch  -> CASE30._open_branch
    scenario_P   -> CASE30.scenario  (gen / load / line)
    f_rhs        -> CASE30.rhs
    rk4          -> CASE30.rk4

States, 3 per machine (NG = 6), so x is (B, 18):

    x = [delta(6), omega(6), Pm(6)]
    ddelta/dt = 2*pi*f0 * (omega - 1)
    domega/dt = (Pm - Pe - D*(omega-1)) / (2H)
    dPm/dt    = (Pset - invR*(omega-1) - Pm) / Tg

Only G1 and G2 have governors (invR = 20); G3-G6 are synchronous condensers
with invR = 0 and Pm0 = 0, so their Pm stays at 0 and they receive no dP.

Hypothesis space (N_CLASS = 57), class 0 = "no fault active":

      0        no fault
      1 -  2   gen trip on machine 1, 2        (severity alpha)
      3 - 23   load step at the 21 load buses  (Y[b,b] += alpha)
     24 - 56   line open, the 33 faultable branches (alpha = 1)

Compare with the 9-bus system: 13 classes, 9 states, 3 actuated machines.
"""

import numpy as np
import scipy.io as sio
import torch

NG, NB = 6, 30
N_STATE = 3 * NG          # 18
N_ACT = 2                 # G1, G2 -- the only machines with a governor


class CASE30:
    def __init__(self, base_mat='case30_base.mat',
                 sim_mat='case30_simulink_data.mat',
                 device='cpu', dtype=torch.float64):
        B = sio.loadmat(base_mat, squeeze_me=True)
        S = sio.loadmat(sim_mat, squeeze_me=True)
        self.device, self.dtype = device, dtype
        self.cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64

        t = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)
        c = lambda a: torch.as_tensor(np.asarray(a), dtype=self.cdtype, device=device)

        # network
        self.Y_pre = c(B['Y_pre'])                       # (30,30)
        self.gen_bus = np.asarray(B['gen_bus']).astype(int)      # 1-based
        self.load_bus = np.asarray(B['load_bus']).astype(int)    # 1-based, 21 of them
        self.br_from = np.asarray(B['br_from']).astype(int)
        self.br_to = np.asarray(B['br_to']).astype(int)
        self.br_Yff, self.br_Yft = c(B['br_Yff']), c(B['br_Yft'])
        self.br_Ytf, self.br_Ytt = c(B['br_Ytf']), c(B['br_Ytt'])
        self.faultable = np.where(np.asarray(B['br_faultable']) > 0)[0]   # 0-based, 33

        # machines
        self.H, self.D = t(S['H']), t(S['D'])
        self.Xdp = np.asarray(S['Xdp'])
        self.Pm0, self.delta0 = t(S['Pm0']), t(S['delta0'])
        self.invR = t(S['invR'])
        self.Tg, self.f0 = float(S['Tg']), float(S['f0'])
        self.wb = 2.0 * np.pi * self.f0
        self.i2H = 1.0 / (2.0 * self.H)
        # internal emf constant:  Ec = (Ed + j Eq) * exp(-j pi/2)
        self.Ec = c(np.asarray(S['Ed']) + 1j * np.asarray(S['Eq'])) * np.exp(-1j * np.pi / 2)

        self.Yred_pre, self.Vrec_pre = self.kron(self.Y_pre)
        self.x0 = torch.cat([self.delta0, torch.ones(NG, dtype=dtype, device=device),
                             self.Pm0])
        self.limit = 0.005
        self.phase = torch.angle(self.Ec).to(dtype)        # for the real path

        # hypothesis table, same (kind, loc) convention as the 9-bus
        # HYPOTHESES list: loc is 1-based machine / bus / branch row
        self.HYPOTHESES = ([('gen', int(i)) for i in [1, 2]]
                           + [('load', int(b)) for b in self.load_bus]
                           + [('line', int(i + 1)) for i in self.faultable])

        # hypothesis layout
        self.gen_machines = np.array([1, 2])              # 1-based, the producing ones
        self.N_HYP = len(self.gen_machines) + len(self.load_bus) + len(self.faultable)
        self.N_CLASS = self.N_HYP + 1                     # 57
        self.act_idx = torch.tensor([0, 1], device=device)   # machines receiving dP

    # ── network ─────────────────────────────────────────────────────────────
    def kron(self, Y):
        """Kron-reduce to machine internal nodes. Y: (...,30,30) -> Yr (...,6,6)."""
        single = (Y.dim() == 2)
        if single:
            Y = Y.unsqueeze(0)
        Bsz = Y.shape[0]
        yg = 1.0 / (1j * torch.as_tensor(self.Xdp, dtype=self.cdtype,
                                         device=self.device))
        Ybb = Y.clone()
        Ybg = torch.zeros(Bsz, NB, NG, dtype=self.cdtype, device=self.device)
        for i, b in enumerate(self.gen_bus):
            Ybb[:, b - 1, b - 1] = Ybb[:, b - 1, b - 1] + yg[i]
            Ybg[:, b - 1, i] = -yg[i]
        Vr = -torch.linalg.solve(Ybb, Ybg)                       # (B,30,6)
        Yr = torch.diag(yg).unsqueeze(0) + Ybg.transpose(1, 2) @ Vr
        if single:
            Yr, Vr = Yr[0], Vr[0]
        return Yr, Vr

    def kron_reduce(self, Y):
        """Alias with the name pinn_model.py expects; accepts (B,30,30)."""
        return self.kron(Y)

    # ── real-valued fast path (same equations, ~2x faster than complex) ─────
    def pe_consts(self, Yred):
        absE = self.Ec.abs()                               # (6,)
        outer = absE.unsqueeze(1) * absE.unsqueeze(0)      # (6,6)
        return (Yred.real * outer).contiguous(), (Yred.imag * outer).contiguous()

    def rhs_real(self, x, M, N, Pset):
        delta, omega, Pm = x[:, :NG], x[:, NG:2 * NG], x[:, 2 * NG:]
        th = delta + self.phase
        dth = th.unsqueeze(2) - th.unsqueeze(1)            # (B,6,6)
        Pe = (M * torch.cos(dth) + N * torch.sin(dth)).sum(-1)
        dw = omega - 1.0
        return torch.cat([self.wb * dw,
                          (Pm - Pe - self.D * dw) * self.i2H,
                          (Pset - self.invR * dw - Pm) / self.Tg], dim=1)

    def rk4_real(self, x, M, N, Pset, h):
        k1 = self.rhs_real(x, M, N, Pset)
        k2 = self.rhs_real(x + 0.5 * h * k1, M, N, Pset)
        k3 = self.rhs_real(x + 0.5 * h * k2, M, N, Pset)
        k4 = self.rhs_real(x + h * k3, M, N, Pset)
        return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _open_branch(self, Y, i):
        """Remove branch i (0-based) exactly: series, charging and tap."""
        f, t_ = self.br_from[i] - 1, self.br_to[i] - 1
        Y = Y.clone()
        Y[f, f] = Y[f, f] - self.br_Yff[i]
        Y[f, t_] = Y[f, t_] - self.br_Yft[i]
        Y[t_, f] = Y[t_, f] - self.br_Ytf[i]
        Y[t_, t_] = Y[t_, t_] - self.br_Ytt[i]
        return Y

    def dY_patterns(self):
        """Additive bus-matrix perturbation per hypothesis, for the soft
        network build in the model. Load: unit at (b,b), scaled by alpha.
        Line: exact removal of the branch (series, charging and tap) -- the
        9-bus shortcut -Y[r,s] would drop charging and tap, which matter here.
        Gen: zero (a trip moves the set-point, not the network)."""
        dY, is_load, is_gen, gen_idx = [], [], [], []
        for kind, loc in self.HYPOTHESES:
            M = torch.zeros_like(self.Y_pre)
            if kind == 'load':
                M[loc - 1, loc - 1] = 1.0
            elif kind == 'line':
                i = loc - 1
                f, t_ = self.br_from[i] - 1, self.br_to[i] - 1
                M[f, f] = -self.br_Yff[i]; M[f, t_] = -self.br_Yft[i]
                M[t_, f] = -self.br_Ytf[i]; M[t_, t_] = -self.br_Ytt[i]
            dY.append(M)
            is_load.append(kind == 'load')
            is_gen.append(kind == 'gen')
            gen_idx.append(loc - 1 if kind == 'gen' else 0)
        return (torch.stack(dY), torch.tensor(is_load), torch.tensor(is_gen),
                torch.tensor(gen_idx))

    # ── hypothesis <-> (type, loc) ──────────────────────────────────────────
    def decode(self, h):
        """class index -> ('none'|'gen'|'load'|'line', loc). loc is 1-based as
        in meta_loc: machine index / bus number / branch row."""
        if h == 0:
            return 'none', 0
        h -= 1
        if h < len(self.gen_machines):
            return 'gen', int(self.gen_machines[h])
        h -= len(self.gen_machines)
        if h < len(self.load_bus):
            return 'load', int(self.load_bus[h])
        h -= len(self.load_bus)
        return 'line', int(self.faultable[h] + 1)

    def encode(self, typ, loc):
        """(type, loc) from the dataset -> class index."""
        if typ == 'none':
            return 0
        if typ == 'gen':
            return 1 + int(np.where(self.gen_machines == loc)[0][0])
        if typ == 'load':
            return 1 + len(self.gen_machines) + int(np.where(self.load_bus == loc)[0][0])
        return (1 + len(self.gen_machines) + len(self.load_bus)
                + int(np.where(self.faultable == loc - 1)[0][0]))

    def scenario(self, typ, loc, alpha):
        """Faulted reduced admittance and set-point offset, as scenario_P does."""
        dPm0 = torch.zeros(NG, dtype=self.dtype, device=self.device)
        Yf = self.Y_pre
        if typ == 'gen':
            dPm0[loc - 1] = -alpha * self.Pm0[loc - 1]
        elif typ == 'load':
            Yf = self.Y_pre.clone()
            Yf[loc - 1, loc - 1] = Yf[loc - 1, loc - 1] + alpha
        elif typ == 'line':
            Yf = self._open_branch(self.Y_pre, loc - 1)
        Yred_f, _ = self.kron(Yf)
        return Yred_f, dPm0

    # ── dynamics ────────────────────────────────────────────────────────────
    def rhs(self, x, Yred, Pset):
        """x: (B,18), Yred: (B,6,6) complex, Pset: (B,6) -> dx/dt (B,18)."""
        delta, omega, Pm = x[:, :NG], x[:, NG:2 * NG], x[:, 2 * NG:]
        E = self.Ec.unsqueeze(0) * torch.exp(1j * delta.to(self.cdtype))
        Pe = torch.real(E * torch.conj(torch.einsum('bij,bj->bi', Yred, E)))
        dw = omega - 1.0
        return torch.cat([self.wb * dw,
                          (Pm - Pe - self.D * dw) * self.i2H,
                          (Pset - self.invR * dw - Pm) / self.Tg], dim=1)

    def rk4(self, x, Yred, Pset, h):
        k1 = self.rhs(x, Yred, Pset)
        k2 = self.rhs(x + 0.5 * h * k1, Yred, Pset)
        k3 = self.rhs(x + 0.5 * h * k2, Yred, Pset)
        k4 = self.rhs(x + h * k3, Yred, Pset)
        return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # ── full scenario simulation (verification / labelling, not training) ───
    def simulate(self, typ, loc, alpha, t_fault, t_clear, T=8.0, h=0.001,
                 dP=None, t_apply=None, t_remove=None):
        """Reproduce one dataset scenario. Returns t (T+1,) and x (T+1,18)."""
        Yred_f, dPm0 = self.scenario(typ, loc, alpha)
        n = int(round(T / h))
        x = self.x0.clone().unsqueeze(0)
        out = torch.empty(n + 1, N_STATE, dtype=self.dtype, device=self.device)
        out[0] = x[0]
        dP = None if dP is None else torch.as_tensor(dP, dtype=self.dtype,
                                                     device=self.device)
        for i in range(n):
            tm = (i + 0.5) * h                      # midpoint decides the segment
            faulted = (tm >= t_fault) and (tm < t_clear)
            Yr = (Yred_f if faulted else self.Yred_pre).unsqueeze(0)
            Ps = self.Pm0 + (dPm0 if faulted else 0.0)
            if dP is not None and t_apply is not None and t_apply <= tm < t_remove:
                Ps = Ps + dP
            x = self.rk4(x, Yr, Ps.unsqueeze(0), h)
            out[i + 1] = x[0]
        return torch.arange(n + 1, dtype=self.dtype, device=self.device) * h, out
