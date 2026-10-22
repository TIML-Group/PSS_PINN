"""
ieee9_physics.py
================
Differentiable port of the network/fault model in compute_delta_p.m.

Everything here is a torch op on batched tensors, so the encoder can emit a
fault hypothesis + severity alpha and have gradients flow through the rebuilt
network into a rollout.

Conventions copied EXACTLY from compute_delta_p.m:

  kron_reduce(Y, gb, Xdp):
      yg  = 1/(j*Xdp)
      Ybb = Y with yg added at each generator's diagonal
      Ybg = -yg in column i at row gen_bus(i)
      Vr  = -(Ybb \\ Ybg)
      Yr  = diag(yg) + Ybg.' * Vr          (note: .' is NON-conjugate transpose)

  scenario_P:
      gen  : dPm0(loc) = -alpha*Pm0(loc);  network unchanged
      load : Yf(loc,loc) += alpha          (loc is a BUS number)
      line : if alpha == 1, open_line_Y(Y, from, to)
             y = -Y(r,s); Y(r,r) -= y; Y(s,s) -= y; Y(r,s) = Y(s,r) = 0
             (series admittance only -- line charging shunts stay put)

  f_rhs:
      Ec = (Ed + j*Eq) * exp(-j*pi/2)
      E  = Ec * exp(j*delta)
      Pe = real(E * conj(Yr @ E))
      ddelta/dt = wb*(w-1)
      dw/dt     = (Pm - Pe - D*(w-1)) / (2H)
      dPm/dt    = (Pset - (w-1)/R - Pm) / Tg
"""

import numpy as np
import scipy.io as sio
import torch

# 12 fault hypotheses, in a fixed order the encoder's softmax indexes into.
#   ('gen', i)  i = 1..3  generator index
#   ('load', b) b = 5,6,8 bus number
#   ('line', r) r = 1..6  row of `lines`
# NOTE: DATASET.md section 1 lists the faultable lines as
#   4-5, 4-6, 5-7, 6-9, 7-8, 8-9
# but the `lines` variable actually stored in pinn_dataset.mat has rows 4 and 5
# the other way round ([7 8] then [6 9]). meta_loc indexes THAT table, so it is
# the one that matters -- using the doc order silently mislabels every
# loc = 4 or 5 line fault. Always load it from the dataset; this default is
# only a fallback and matches the .mat, not the doc.
LINES = np.array([[4, 5], [4, 6], [5, 7], [7, 8], [6, 9], [8, 9]])
LOAD_BUSES = [5, 6, 8]

HYPOTHESES = ([('gen', i) for i in (1, 2, 3)]
              + [('load', b) for b in LOAD_BUSES]
              + [('line', r) for r in range(1, 7)])
N_HYP = len(HYPOTHESES)          # 12


class IEEE9:
    """System constants + differentiable network rebuild."""

    def __init__(self, base_mat, sys_mat, dtype=torch.float64, device='cpu',
                 lines=None):
        """lines: (6,2) int array from the dataset's `lines` variable.
        Pass it -- do not rely on the default."""
        B = sio.loadmat(base_mat)
        P = sio.loadmat(sys_mat)

        self.device = device
        self.rdtype = dtype
        self.cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64

        t  = lambda a: torch.tensor(np.asarray(a).ravel(), dtype=dtype, device=device)
        tc = lambda a: torch.tensor(np.asarray(a), dtype=self.cdtype, device=device)

        self.lines    = np.asarray(lines if lines is not None else LINES, dtype=int)
        self.Y_pre    = tc(B['Y_pre'])                     # (9,9)
        self.gen_bus  = np.asarray(B['gen_bus']).ravel().astype(int)
        self.Xdp      = t(P['Xdp'])
        self.H, self.D = t(P['H']), t(P['D'])
        self.Ed, self.Eq = t(P['Ed']), t(P['Eq'])
        self.Pm0     = t(P['Pm0'])
        self.delta0  = t(P['delta0'])
        self.R  = float(P['R'].ravel()[0])
        self.Tg = float(P['Tg'].ravel()[0])
        self.f0 = float(P['f0'].ravel()[0])
        self.wb = 2 * np.pi * self.f0

        # Ec = (Ed + j Eq) * exp(-j pi/2)
        self.Ec = (self.Ed.to(self.cdtype) + 1j * self.Eq.to(self.cdtype)) \
                  * torch.exp(torch.tensor(-1j * np.pi / 2, dtype=self.cdtype, device=device))

        self.phase = torch.angle(self.Ec).to(dtype)     # angle of Ec, real

        # stored reference matrices, for verification
        self.Yred_pre_ref = tc(P['Yred_pre'])
        self.Vrec_pre_ref = tc(P['Vrec_pre'])

        self.Yred_pre, self.Vrec_pre = self.kron_reduce(self.Y_pre.unsqueeze(0))

    # ── network ──────────────────────────────────────────────────────────────
    # Capture-safe batched solve: hipSOLVER (linalg.solve) cannot run inside a
    # CUDA-graph capture on this ROCm build (operation not permitted while
    # capturing). Pure-tensor Gaussian elimination with partial pivoting is
    # graph-recordable; accuracy and gradients match linalg.solve to ~1e-6 at
    # float32-complex on (512,9,9)x(512,9,3) (measured). Eager consumers keep
    # the exact solver; only CUDA-graph training flips the flag.
    solver_capture_safe = False

    def _solve_batched(self, A, Bm):
        if not self.solver_capture_safe:
            return torch.linalg.solve(A, Bm)
        N, Bn, M = A.shape[1], A.shape[0], Bm.shape[2]
        aug = torch.cat([A, Bm], dim=2)
        idr = torch.arange(N, device=A.device).expand(Bn, N)
        for k in range(N - 1):
            p = aug[:, k:, k].abs().argmax(dim=1) + k
            kk, pk = (torch.full_like(p, k)).unsqueeze(1), p.unsqueeze(1)
            vk, vp = idr.gather(1, kk), idr.gather(1, pk)
            swap = idr.clone()
            swap.scatter_(1, kk, vp)
            swap.scatter_(1, pk, vk)
            aug = aug.gather(1, swap.unsqueeze(2).expand(Bn, N, aug.shape[2]))
            piv = aug[:, k, k].reshape(-1, 1, 1)
            factor = aug[:, k + 1:, k].unsqueeze(2) / piv
            aug = torch.cat([aug[:, :k + 1, :],
                             aug[:, k + 1:, :] - factor * aug[:, k:k + 1, :]],
                            dim=1)
        xs = None
        for k in range(N - 1, -1, -1):
            rhs = aug[:, k, N:]
            if xs is None:
                x = (rhs / aug[:, k, k].unsqueeze(1)).unsqueeze(1)
            else:
                s = (aug[:, k, k + 1:N].unsqueeze(2) * xs).sum(1)
                x = torch.cat([((rhs - s) / aug[:, k, k].unsqueeze(1)).unsqueeze(1), xs], 1)
            xs = x
        return xs

    def kron_reduce(self, Y):
        """
        Y: (B,9,9) complex -> Yred (B,3,3), Vrec (B,9,3). Differentiable in Y.
        """
        Bn, NB, _ = Y.shape
        N = len(self.gen_bus)
        yg = 1.0 / (1j * self.Xdp.to(self.cdtype))          # (3,)

        # idx / arange are constants; caching them avoids host-side tensor
        # creation inside the forward — which is illegal while a CUDA graph
        # records (HIP: operation not permitted when stream is capturing).
        if not hasattr(self, '_kr_cache'):
            self._kr_cache = {}
        key = (Y.device, self.cdtype)
        if key not in self._kr_cache:
            self._kr_cache[key] = (
                torch.tensor(self.gen_bus - 1, device=Y.device),   # MATLAB 1-based
                torch.arange(N, device=Y.device))
        idx, arange_n = self._kr_cache[key]

        Ybb = Y.clone()
        Ybb[:, idx, idx] = Ybb[:, idx, idx] + yg

        Ybg = torch.zeros(Bn, NB, N, dtype=self.cdtype, device=Y.device)
        Ybg[:, idx, arange_n] = -yg

        Vr = -self._solve_batched(Ybb, Ybg)                  # (B,9,3)
        # Ybg.' * Vr -- NON-conjugate transpose, as in MATLAB
        Yr = torch.diag(yg).unsqueeze(0) + Ybg.transpose(1, 2) @ Vr
        return Yr, Vr

    def faulted_Y(self, hyp_idx, alpha):
        """
        Bus admittance under one hypothesis. hyp_idx: int, alpha: (B,) tensor.
        Returns (B,9,9). Differentiable in alpha (load faults only).
        """
        kind, loc = HYPOTHESES[hyp_idx]
        Bn = alpha.shape[0]
        Y = self.Y_pre.unsqueeze(0).expand(Bn, 9, 9).clone()

        if kind == 'gen':
            return Y                                  # network unchanged
        if kind == 'load':
            b = loc - 1
            Y[:, b, b] = Y[:, b, b] + alpha.to(self.cdtype)
            return Y
        # line: open the series branch (alpha is always 1 for line faults)
        r, s = self.lines[loc - 1] - 1
        y = -Y[:, r, s].clone()
        Y[:, r, r] = Y[:, r, r] - y
        Y[:, s, s] = Y[:, s, s] - y
        Y[:, r, s] = 0
        Y[:, s, r] = 0
        return Y

    def dPm0(self, hyp_idx, alpha):
        """Set-point drop for a generator trip; zeros otherwise. (B,3)"""
        kind, loc = HYPOTHESES[hyp_idx]
        Bn = alpha.shape[0]
        d = torch.zeros(Bn, 3, dtype=self.rdtype, device=alpha.device)
        if kind == 'gen':
            d[:, loc - 1] = -alpha * self.Pm0[loc - 1]
        return d

    # ── dynamics ─────────────────────────────────────────────────────────────
    def rhs(self, x, Yred, Pset):
        """
        x: (B,9) = [delta(3), omega(3), Pm(3)];  Yred: (B,3,3);  Pset: (B,3)
        """
        delta, omega, Pm = x[:, 0:3], x[:, 3:6], x[:, 6:9]
        E  = self.Ec.unsqueeze(0) * torch.exp(1j * delta.to(self.cdtype))
        I  = torch.einsum('bij,bj->bi', Yred, E)
        Pe = torch.real(E * torch.conj(I))
        dw = omega - 1.0
        return torch.cat([
            self.wb * dw,
            (Pm - Pe - self.D * dw) / (2.0 * self.H),
            (Pset - dw / self.R - Pm) / self.Tg,
        ], dim=1)

    # ── real-valued fast path ────────────────────────────────────────────────
    # The complex form (exp, einsum, conj) costs 24 tensor ops per call, so a
    # 300-step RK4 rollout issues ~28,800 kernels -- launch-bound on a GPU no
    # matter how big the batch is. |E| is CONSTANT here (Ed, Eq are fixed), so
    # the classical real form can be used instead:
    #
    #   Pe_i = sum_j  M_ij cos(th_i - th_j) + N_ij sin(th_i - th_j)
    #   M = |E_i||E_j| Re(Yred),  N = |E_i||E_j| Im(Yred),  th = delta + phase
    #
    # M and N depend only on Yred, so they are built ONCE per rollout and the
    # inner loop becomes pure real arithmetic in ~10 ops.
    def pe_consts(self, Yred):
        absE = self.Ec.abs()                              # (3,)
        outer = absE.unsqueeze(1) * absE.unsqueeze(0)     # (3,3)
        return (Yred.real * outer).contiguous(), (Yred.imag * outer).contiguous()

    def rhs_real(self, x, M, N, Pset):
        delta, omega, Pm = x[:, 0:3], x[:, 3:6], x[:, 6:9]
        th = delta + self.phase                            # (B,3)
        dth = th.unsqueeze(2) - th.unsqueeze(1)            # (B,3,3)
        Pe = (M * torch.cos(dth) + N * torch.sin(dth)).sum(-1)
        dw = omega - 1.0
        return torch.cat([
            self.wb * dw,
            (Pm - Pe - self.D * dw) / (2.0 * self.H),
            (Pset - dw / self.R - Pm) / self.Tg,
        ], dim=1)

    def rk4_real(self, x, M, N, Pset, h):
        k1 = self.rhs_real(x, M, N, Pset)
        k2 = self.rhs_real(x + 0.5 * h * k1, M, N, Pset)
        k3 = self.rhs_real(x + 0.5 * h * k2, M, N, Pset)
        k4 = self.rhs_real(x + h * k3, M, N, Pset)
        return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def rk4(self, x, Yred, Pset, h):
        k1 = self.rhs(x, Yred, Pset)
        k2 = self.rhs(x + 0.5 * h * k1, Yred, Pset)
        k3 = self.rhs(x + 0.5 * h * k2, Yred, Pset)
        k4 = self.rhs(x + h * k3, Yred, Pset)
        return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
