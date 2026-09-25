"""
psys_physics.py
===============
Differentiable network/fault model, generic over system size.

This replaces the 9-bus-only `ieee9_physics.py`. Sizes (NG machines, NB buses),
the governor droop (per machine, not scalar), the hypothesis list and the fault
admittance bank are all read from the system data files, so the same code runs
the 9-bus and the 30-bus system. `ieee9()` reproduces `ieee9_physics.IEEE9`
bit-for-bit; that equivalence is asserted in verify_physics.py.

Conventions copied EXACTLY from case30_lib.m / compute_delta_p.m:

  kron_reduce(Y, gb, Xdp):
      yg  = 1/(j*Xdp);  Ybb = Y + yg at each generator diagonal
      Ybg = -yg in column i at row gen_bus(i)
      Vr  = -(Ybb \\ Ybg)
      Yr  = diag(yg) + Ybg.' * Vr          (.' is NON-conjugate transpose)

  scenario_P:
      gen : dPm0(loc) = -alpha*Pm0(loc);  network unchanged
      load: Yf(loc,loc) += alpha          (loc is a BUS number)
      line: branch loc opened exactly (alpha == 1):
              Y(f,f) -= Yff;  Y(f,t) -= Yft;  Y(t,f) -= Ytf;  Y(t,t) -= Ytt
            This is open_branch() from case30_lib.m and it removes the series
            branch, its charging shunts AND its tap. The 9-bus module dropped
            the series admittance only, which is exact there but NOT here: 7 of
            the 41 30-bus branches are transformers with off-nominal taps.

  f_rhs:
      Ec = (Ed + j*Eq)*exp(-j*pi/2);  E = Ec*exp(j*delta)
      Pe = real(E .* conj(Yr @ E))
      ddelta/dt = wb*(w-1)
      dw/dt     = (Pm - Pe - D*(w-1)) / (2H)
      dPm/dt    = (Pset - invR*(w-1) - Pm) / Tg      <-- invR is PER MACHINE

The per-machine invR is the one change that is silently wrong if ported
carelessly: on the 30-bus system G3-G6 are synchronous condensers with
invR = 0, and giving them the scalar droop of G1/G2 invents a governor that
does not exist.
"""

import numpy as np
import scipy.io as sio
import torch


class PowerSystem:
    """System constants + differentiable network rebuild + RK4 dynamics."""

    def __init__(self, Y_pre, gen_bus, Xdp, H, D, Ed, Eq, Pm0, delta0,
                 invR, Tg, f0, hypotheses, dY, ctrl_machines,
                 Yred_pre_ref=None, Vrec_pre_ref=None,
                 dtype=torch.float64, device='cpu', name=''):
        self.name = name
        self.device = device
        self.rdtype = dtype
        self.cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64

        t  = lambda a: torch.tensor(np.asarray(a).ravel(), dtype=dtype, device=device)
        tc = lambda a: torch.tensor(np.asarray(a), dtype=self.cdtype, device=device)

        self.Y_pre   = tc(Y_pre)
        self.gen_bus = np.asarray(gen_bus).ravel().astype(int)
        self.NG      = len(self.gen_bus)
        self.NB      = self.Y_pre.shape[-1]
        self.Xdp     = t(Xdp)
        self.H, self.D = t(H), t(D)
        self.Ed, self.Eq = t(Ed), t(Eq)
        self.Pm0     = t(Pm0)
        self.delta0  = t(delta0)
        self.invR    = t(invR)                       # (NG,) per machine
        self.Tg      = float(np.asarray(Tg).ravel()[0])
        self.f0      = float(np.asarray(f0).ravel()[0])
        self.wb      = 2 * np.pi * self.f0

        self.Ec = (self.Ed.to(self.cdtype) + 1j * self.Eq.to(self.cdtype)) \
            * torch.exp(torch.tensor(-1j * np.pi / 2, dtype=self.cdtype, device=device))
        self.phase = torch.angle(self.Ec).to(dtype)

        # hypothesis bank -------------------------------------------------
        # Y_fault = Y_pre + scale * dY[h],  scale = alpha (load) / 1 (line) / 0 (gen)
        self.hypotheses = list(hypotheses)
        self.N_HYP      = len(self.hypotheses)
        self.dY         = tc(np.asarray(dY))                     # (N_HYP,NB,NB)
        kinds           = [k for k, _ in self.hypotheses]
        self.is_gen     = torch.tensor([k == 'gen'  for k in kinds], device=device)
        self.is_load    = torch.tensor([k == 'load' for k in kinds], device=device)
        self.is_line    = torch.tensor([k == 'line' for k in kinds], device=device)
        # machine index (0-based) each gen hypothesis trips; 0 elsewhere (masked)
        self.gen_idx = torch.tensor(
            [(loc - 1) if k == 'gen' else 0 for k, loc in self.hypotheses],
            dtype=torch.long, device=device)
        # machines that actually have a governor / turbine to move
        self.ctrl_machines = torch.tensor(np.asarray(ctrl_machines).ravel().astype(int) - 1,
                                          dtype=torch.long, device=device)
        self.n_ctrl = len(self.ctrl_machines)

        self.Yred_pre_ref = None if Yred_pre_ref is None else tc(Yred_pre_ref)
        self.Vrec_pre_ref = None if Vrec_pre_ref is None else tc(Vrec_pre_ref)
        self.Yred_pre, self.Vrec_pre = self.kron_reduce(self.Y_pre.unsqueeze(0))

    # ── network ─────────────────────────────────────────────────────────────
    def kron_reduce(self, Y):
        """Y: (B,NB,NB) complex -> Yred (B,NG,NG), Vrec (B,NB,NG)."""
        Bn = Y.shape[0]
        yg = 1.0 / (1j * self.Xdp.to(self.cdtype))
        Ybb = Y.clone()
        idx = torch.tensor(self.gen_bus - 1, device=Y.device)
        Ybb[:, idx, idx] = Ybb[:, idx, idx] + yg
        Ybg = torch.zeros(Bn, self.NB, self.NG, dtype=self.cdtype, device=Y.device)
        Ybg[:, idx, torch.arange(self.NG, device=Y.device)] = -yg
        Vr = -torch.linalg.solve(Ybb, Ybg)
        Yr = torch.diag(yg).unsqueeze(0) + Ybg.transpose(1, 2) @ Vr
        return Yr, Vr

    def faulted_Y(self, hyp_idx, alpha):
        """Bus admittance under one hypothesis. hyp_idx: int, alpha: (B,)."""
        kind, _ = self.hypotheses[hyp_idx]
        Bn = alpha.shape[0]
        scale = (alpha if kind == 'load'
                 else torch.ones_like(alpha) if kind == 'line'
                 else torch.zeros_like(alpha))
        return (self.Y_pre.unsqueeze(0)
                + scale.to(self.cdtype).view(Bn, 1, 1) * self.dY[hyp_idx].unsqueeze(0))

    def dPm0(self, hyp_idx, alpha):
        """Set-point drop for a generator trip; zeros otherwise. (B,NG)"""
        kind, loc = self.hypotheses[hyp_idx]
        d = torch.zeros(alpha.shape[0], self.NG, dtype=self.rdtype, device=alpha.device)
        if kind == 'gen':
            d[:, loc - 1] = -alpha * self.Pm0[loc - 1]
        return d

    def x0(self, Bn=1, device=None):
        """Equilibrium state [delta0, 1, Pm0], (Bn, 3*NG)."""
        dev = device or self.delta0.device
        return torch.cat([self.delta0, torch.ones_like(self.delta0), self.Pm0]) \
            .to(dev).unsqueeze(0).expand(Bn, -1).contiguous()

    # ── dynamics ────────────────────────────────────────────────────────────
    def rhs(self, x, Yred, Pset):
        n = self.NG
        delta, omega, Pm = x[:, :n], x[:, n:2*n], x[:, 2*n:3*n]
        E  = self.Ec.unsqueeze(0) * torch.exp(1j * delta.to(self.cdtype))
        I  = torch.einsum('bij,bj->bi', Yred, E)
        Pe = torch.real(E * torch.conj(I))
        dw = omega - 1.0
        return torch.cat([
            self.wb * dw,
            (Pm - Pe - self.D * dw) / (2.0 * self.H),
            (Pset - self.invR * dw - Pm) / self.Tg,
        ], dim=1)

    # real-valued fast path: |E| is constant, so
    #   Pe_i = sum_j M_ij cos(th_i-th_j) + N_ij sin(th_i-th_j),  th = delta+phase
    # M, N depend only on Yred -> built once per rollout, inner loop is real.
    def pe_consts(self, Yred):
        absE = self.Ec.abs()
        outer = absE.unsqueeze(1) * absE.unsqueeze(0)
        return (Yred.real * outer).contiguous(), (Yred.imag * outer).contiguous()

    def rhs_real(self, x, M, N, Pset):
        n = self.NG
        delta, omega, Pm = x[:, :n], x[:, n:2*n], x[:, 2*n:3*n]
        th = delta + self.phase
        dth = th.unsqueeze(2) - th.unsqueeze(1)
        Pe = (M * torch.cos(dth) + N * torch.sin(dth)).sum(-1)
        dw = omega - 1.0
        return torch.cat([
            self.wb * dw,
            (Pm - Pe - self.D * dw) / (2.0 * self.H),
            (Pset - self.invR * dw - Pm) / self.Tg,
        ], dim=1)

    def rk4(self, x, Yred, Pset, h):
        k1 = self.rhs(x, Yred, Pset)
        k2 = self.rhs(x + 0.5 * h * k1, Yred, Pset)
        k3 = self.rhs(x + 0.5 * h * k2, Yred, Pset)
        k4 = self.rhs(x + h * k3, Yred, Pset)
        return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def rk4_real(self, x, M, N, Pset, h):
        k1 = self.rhs_real(x, M, N, Pset)
        k2 = self.rhs_real(x + 0.5 * h * k1, M, N, Pset)
        k3 = self.rhs_real(x + 0.5 * h * k2, M, N, Pset)
        k4 = self.rhs_real(x + h * k3, M, N, Pset)
        return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # ── channel layout of the measurement vector ────────────────────────────
    # [delta/omega interleaved (2NG), Pm (NG), Id/Iq interleaved (2NG),
    #  Vm/theta interleaved (2NB)]  -- pack_states() in case30_lib.m
    @property
    def n_channels(self):
        return 5 * self.NG + 2 * self.NB

    @property
    def idx_delta(self):
        return np.arange(0, 2 * self.NG, 2)

    @property
    def idx_omega(self):
        return np.arange(1, 2 * self.NG, 2)

    @property
    def idx_Pm(self):
        return np.arange(2 * self.NG, 3 * self.NG)


# ── constructors ────────────────────────────────────────────────────────────
def _bank(NB, hypotheses, br=None):
    """dY[h] such that Y_fault = Y_pre + scale*dY[h]."""
    dY = np.zeros((len(hypotheses), NB, NB), dtype=np.complex128)
    for h, (kind, loc) in enumerate(hypotheses):
        if kind == 'load':
            dY[h, loc - 1, loc - 1] = 1.0                  # scaled by alpha
        elif kind == 'line':
            f, t = br['from'][loc - 1] - 1, br['to'][loc - 1] - 1
            dY[h, f, f] = -br['Yff'][loc - 1]
            dY[h, f, t] = -br['Yft'][loc - 1]
            dY[h, t, f] = -br['Ytf'][loc - 1]
            dY[h, t, t] = -br['Ytt'][loc - 1]
        # gen: network unchanged, dY stays zero
    return dY


def case30(base_mat, sys_mat, dtype=torch.float64, device='cpu'):
    B = sio.loadmat(base_mat)
    P = sio.loadmat(sys_mat)
    NB = B['Y_pre'].shape[0]

    producing = B['producing'].ravel().astype(bool)     # machines that can trip
    governor  = B['governor'].ravel().astype(bool)      # machines with a turbine
    load_bus  = B['load_bus'].ravel().astype(int)
    faultable = np.where(B['br_faultable'].ravel() > 0)[0] + 1   # 1-based rows

    hyp = ([('gen', i + 1) for i in np.where(producing)[0]]
           + [('load', int(b)) for b in load_bus]
           + [('line', int(r)) for r in faultable])
    br = {'from': B['br_from'].ravel().astype(int), 'to': B['br_to'].ravel().astype(int),
          'Yff': B['br_Yff'].ravel(), 'Yft': B['br_Yft'].ravel(),
          'Ytf': B['br_Ytf'].ravel(), 'Ytt': B['br_Ytt'].ravel()}

    return PowerSystem(
        Y_pre=B['Y_pre'], gen_bus=B['gen_bus'], Xdp=P['Xdp'], H=P['H'], D=P['D'],
        Ed=P['Ed'], Eq=P['Eq'], Pm0=P['Pm0'], delta0=P['delta0'],
        invR=P['invR'], Tg=P['Tg'], f0=P['f0'],
        hypotheses=hyp, dY=_bank(NB, hyp, br),
        ctrl_machines=np.where(governor)[0] + 1,
        Yred_pre_ref=P.get('Yred_pre'), Vrec_pre_ref=P.get('Vrec_pre'),
        dtype=dtype, device=device, name='case30')


def ieee9(base_mat, sys_mat, lines, dtype=torch.float64, device='cpu'):
    """`lines`: the (6,2) `lines` variable FROM THE DATASET, not from the doc --
    DATASET.md has rows 4 and 5 swapped."""
    B = sio.loadmat(base_mat)
    P = sio.loadmat(sys_mat)
    NB = B['Y_pre'].shape[0]
    lines = np.asarray(lines, dtype=int)
    NG = len(np.asarray(B['gen_bus']).ravel())

    hyp = ([('gen', i) for i in range(1, NG + 1)]
           + [('load', b) for b in (5, 6, 8)]
           + [('line', r) for r in range(1, len(lines) + 1)])

    # 9-bus: open_line_Y removes the SERIES admittance only (no taps there).
    dY = np.zeros((len(hyp), NB, NB), dtype=np.complex128)
    Y = np.asarray(B['Y_pre'])
    for h, (kind, loc) in enumerate(hyp):
        if kind == 'load':
            dY[h, loc - 1, loc - 1] = 1.0
        elif kind == 'line':
            r, s = lines[loc - 1] - 1
            y = -Y[r, s]
            dY[h, r, r] = -y
            dY[h, s, s] = -y
            dY[h, r, s] = -Y[r, s]
            dY[h, s, r] = -Y[s, r]

    invR = np.full(NG, 1.0 / float(np.asarray(P['R']).ravel()[0]))
    sysm = PowerSystem(
        Y_pre=B['Y_pre'], gen_bus=B['gen_bus'], Xdp=P['Xdp'], H=P['H'], D=P['D'],
        Ed=P['Ed'], Eq=P['Eq'], Pm0=P['Pm0'], delta0=P['delta0'],
        invR=invR, Tg=P['Tg'], f0=P['f0'],
        hypotheses=hyp, dY=dY, ctrl_machines=np.arange(1, NG + 1),
        Yred_pre_ref=P.get('Yred_pre'), Vrec_pre_ref=P.get('Vrec_pre'),
        dtype=dtype, device=device, name='ieee9')
    sysm.lines = lines
    return sysm
