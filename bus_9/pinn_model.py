"""
pinn_model.py
=============
Receding-horizon fault identifier / controller for the IEEE-9 dataset.

    window of measurements
        -> soft temporal mask, length L is LEARNABLE
        -> bias-free linear projection (per timestep)
        -> 10-layer residual LSTM
        -> learnable softmax pooling
        -> 4-layer MLP trunk
        -> heads:  hypothesis (13)   alpha   x0 correction (9)   dP (3)

The heads do NOT predict OOL or delta_p as labels. They identify the ONGOING
DISTURBANCE; a differentiable rollout of the swing equations then turns that
identification into consequences:

    OOL  = does simulated |omega-1| exceed the limit over the horizon
    dP   = the set-point change that keeps the simulated omega inside it

Hypothesis 0 is "no fault active" -- essential under receding horizon, where
most decision instants are pre-fault or post-clearing.

Soft window
-----------
A hard slice is not differentiable in its own length, so the encoder always
receives W_max timesteps and each is gated by

    m_tau = sigmoid( (L - age_tau) / beta )

with age_tau the timestep's age in seconds relative to the window end. L is a
learnable scalar (sigmoid-bounded into (0, L_max)) and beta is annealed to
sharpen the edge. Gradients reach L through the masked input, so "the minimum
window that makes the disturbance identifiable" becomes a trainable quantity --
paired with a lambda*(L/L_max) penalty in the training script, since accuracy
alone would just grow L to the cap.

The mask multiplies the INPUT and the projection is BIAS-FREE, so a masked-out
timestep contributes exactly zero. This matters: with a biased MLP encoder,
zero input still maps to a nonzero embedding, so masked steps would feed a
constant vector into the LSTM and part of the gradient reaching L would be an
artifact of that bias rather than of information genuinely entering the window.
Since the window length is the headline claim, it has to be clean.

There is no MLP in front of the LSTM. An LSTM's W_ih is already a learned
linear projection of the input, the 31 features are physically meaningful and
pre-normalised, and a per-timestep MLP does no temporal mixing -- so a 4-layer
tanh stack there mostly added saturation in front of an already deep recurrent
network. The MLP after pooling is kept: it does real work mapping one pooled
state to four heterogeneous heads.

Hypothesis mixing
-----------------
Rolling out all 13 hypotheses would be 13x the cost. Instead the BUS admittance
is mixed before Kron reduction,

    Y = Y_pre + sum_h p_h * dY_h(alpha)

which is exact for a one-hot p and a smooth relaxation otherwise (each fault is
an additive perturbation of Y, so the mixture is well defined). Pe is nonlinear
in Yred, so the relaxation is only approximate mid-training; it sharpens as p
concentrates. Evaluation uses the hard argmax -- see `hard` in forward().
"""

import numpy as np
import torch
import torch.nn as nn

from ieee9_physics import IEEE9, HYPOTHESES, N_HYP

# 13 classes: 0 = no fault active, 1..12 = HYPOTHESES
N_CLASS = N_HYP + 1


# ─────────────────────────────────────────────────────────────────────────────
class MLPBlock(nn.Module):
    """n_layers Linear layers of width `hidden`, tanh between them."""

    def __init__(self, in_dim, hidden=128, n_layers=4):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.Tanh()]
            d = hidden
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class ResidualLSTMStack(nn.Module):
    """n_layers single-layer LSTMs with residual + LayerNorm between them.

    A plain nn.LSTM(num_layers=10) attenuates the gradient enough that the
    lower layers barely train; measured grad at the input-side parameters was
    ~1e-16 vs ~1e-3 for this stack.
    """

    def __init__(self, hidden=128, n_layers=10):
        super().__init__()
        self.blocks = nn.ModuleList(nn.LSTM(hidden, hidden, batch_first=True)
                                    for _ in range(n_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(n_layers))

    def forward(self, seq):
        h = seq
        for blk, nrm in zip(self.blocks, self.norms):
            h = nrm(blk(h)[0] + h)
        return h


# ─────────────────────────────────────────────────────────────────────────────
def matched_mlp_proj(n_feat, hidden, mlp_layers, lstm_layers, w_steps):
    """Per-step projection width that gives the MLP encoder the same number of
    parameters as the LSTM encoder (projection + LSTM stack + LayerNorm +
    pooling + trunk), so a difference in results cannot be put down to
    capacity."""
    lstm_enc = (n_feat * hidden
                + lstm_layers * (4 * hidden * (hidden + hidden) + 8 * hidden
                                 + 2 * hidden)
                + w_steps
                + hidden * hidden + hidden + (mlp_layers - 1) * (hidden * hidden + hidden))
    # mlp encoder: n_feat*P + (W*P*hidden + hidden) + (mlp_layers-1)*(h^2+h)
    rest = hidden + (mlp_layers - 1) * (hidden * hidden + hidden)
    P = (lstm_enc - rest) / (n_feat + w_steps * hidden)
    return max(1, int(round(P)))


class FaultEncoder(nn.Module):
    """Window of measurements -> (hypothesis, alpha, x0 correction, dP)."""

    def __init__(self, n_feat, hidden=128, mlp_layers=4, lstm_layers=10,
                 w_max_steps=64, dt_window=0.005,
                 init_window=0.15, max_window=0.30, beta0=0.04,
                 encoder='lstm', mlp_proj=None):
        super().__init__()
        self.hidden = hidden
        self.w_max_steps = w_max_steps
        self.dt_window = dt_window
        self.max_window = float(max_window)
        if encoder not in ('lstm', 'mlp'):
            raise ValueError(f"encoder must be 'lstm' or 'mlp', got {encoder!r}")
        self.encoder = encoder

        if encoder == 'lstm':
            # bias-free so masked timesteps contribute exactly zero
            self.proj  = nn.Linear(n_feat, hidden, bias=False)
            self.lstm  = ResidualLSTMStack(hidden, lstm_layers)
            self.trunk = MLPBlock(hidden, hidden, mlp_layers)
        else:
            # ABLATION: no recurrence. Each timestep is projected by the SAME
            # bias-free layer (n_feat -> mlp_proj, masked steps stay exact
            # zeros), the window is flattened, and an MLP of the same depth
            # and width as the LSTM model's trunk reads it in one pass. No
            # pooling: every timestep position has its own weights.
            # mlp_proj=None picks the width that matches the LSTM encoder's
            # parameter count (see matched_mlp_proj).
            if mlp_proj is None:
                mlp_proj = matched_mlp_proj(n_feat, hidden, mlp_layers,
                                            lstm_layers, w_max_steps)
            self.mlp_proj = mlp_proj
            self.proj  = nn.Linear(n_feat, mlp_proj, bias=False)
            self.lstm  = None
            self.trunk = MLPBlock(w_max_steps * mlp_proj, hidden, mlp_layers)
        nn.init.xavier_normal_(self.proj.weight)

        self.head_hyp   = nn.Linear(hidden, N_CLASS)
        self.head_alpha = nn.Linear(hidden, 1)
        self.head_x0    = nn.Linear(hidden, 9)     # correction to measured state
        self.head_dP    = nn.Linear(hidden, 3)
        for h in (self.head_hyp, self.head_alpha, self.head_x0, self.head_dP):
            nn.init.xavier_normal_(h.weight, gain=0.1); nn.init.zeros_(h.bias)

        # learnable window length, sigmoid-bounded into (0, max_window)
        frac = min(max(init_window / self.max_window, 1e-3), 1 - 1e-3)
        self.raw_L = nn.Parameter(torch.tensor(np.log(frac / (1 - frac)),
                                               dtype=torch.float32))
        self.register_buffer('beta', torch.tensor(float(beta0)))
        # age of each timestep relative to the window end (newest = 0)
        age = torch.arange(w_max_steps - 1, -1, -1).float() * dt_window
        self.register_buffer('age', age)
        if encoder == 'lstm':
            self.pool_logits = nn.Parameter(torch.linspace(-1.5, 1.5, w_max_steps))

    def window_len(self):
        return self.max_window * torch.sigmoid(self.raw_L)

    def mask(self):
        return torch.sigmoid((self.window_len() - self.age) / self.beta)

    def forward(self, win):
        """win: (B, W, n_feat) ending at the decision instant."""
        B, W, F = win.shape
        m = self.mask().view(1, W, 1)
        emb = self.proj(win * m)            # bias-free: masked steps -> exact 0
        if self.encoder == 'lstm':
            h = self.lstm(emb)
            w = torch.softmax(self.pool_logits + torch.log(self.mask() + 1e-8), 0)
            z = self.trunk((h * w.view(1, W, 1)).sum(1))
        else:
            z = self.trunk(emb.reshape(B, -1))
        return {
            'z': z,
            'hyp_logits': self.head_hyp(z),
            'alpha_raw':  self.head_alpha(z).squeeze(-1),
            'x0_corr':    self.head_x0(z),
            'dP_raw':     self.head_dP(z),
        }


# ─────────────────────────────────────────────────────────────────────────────
class RolloutController(nn.Module):
    """Encoder + differentiable swing-equation rollout."""

    def __init__(self, sys: IEEE9, encoder: FaultEncoder,
                 horizon=1.5, dt_roll=0.01, grace=0.2,
                 omega_limit=0.005, margin=1e-4, dp_max=1.0,
                 alpha_lo=0.10, alpha_hi=1.00,
                 dp_head='linear', dp_hidden=128, dp_fc_layers=2, n_refine=3,
                 dp_cell='lstm'):
        super().__init__()
        self.S = sys
        self.enc = encoder
        self.horizon, self.dt_roll, self.grace = horizon, dt_roll, grace
        self.n_steps = int(round(horizon / dt_roll))
        self.limit, self.margin, self.dp_max = omega_limit, margin, dp_max
        self.alpha_lo, self.alpha_hi = alpha_lo, alpha_hi

        self.dp_head = dp_head
        self.dp_hidden = dp_hidden
        self.n_refine = n_refine
        if dp_head == 'cascade':
            ctx_dim = encoder.hidden + N_CLASS + 1 + 9
            if dp_cell not in ('lstm', 'mlp'):
                raise ValueError(f"dp_cell must be 'lstm' or 'mlp', got {dp_cell!r}")
            self.dp_cell_type = dp_cell
            if dp_cell == 'lstm':
                self.dp_cell = nn.LSTMCell(ctx_dim + 3 + self.N_EXC, dp_hidden)
            else:
                # ABLATION: memoryless refinement, nothing carried between rounds
                self.dp_cell = nn.Sequential(
                    nn.Linear(ctx_dim + 3 + self.N_EXC, dp_hidden), nn.Tanh())
            fc = []
            d = dp_hidden
            for _ in range(dp_fc_layers - 1):
                fc += [nn.Linear(d, dp_hidden), nn.Tanh()]
                d = dp_hidden
            fc += [nn.Linear(d, 3)]
            self.dp_fc = nn.Sequential(*fc)
            nn.init.zeros_(self.dp_fc[-1].weight); nn.init.zeros_(self.dp_fc[-1].bias)

        # Precompute each hypothesis' additive perturbation of the bus matrix.
        # dY_unit[h] is the pattern; load faults scale by alpha, line faults are
        # binary, generator trips leave Y untouched.
        dY, is_load, is_gen, gen_idx = [], [], [], []
        Ypre = sys.Y_pre
        for kind, loc in HYPOTHESES:
            M = torch.zeros_like(Ypre)
            if kind == 'load':
                M[loc - 1, loc - 1] = 1.0                      # scaled by alpha
            elif kind == 'line':
                r, s = sys.lines[loc - 1] - 1
                y = -Ypre[r, s]
                M[r, r] = -y; M[s, s] = -y
                M[r, s] = -Ypre[r, s]; M[s, r] = -Ypre[s, r]
            dY.append(M)
            is_load.append(kind == 'load')
            is_gen.append(kind == 'gen')
            gen_idx.append(loc - 1 if kind == 'gen' else 0)
        self.register_buffer('dY', torch.stack(dY))                  # (12,9,9)
        self.register_buffer('is_load', torch.tensor(is_load))
        self.register_buffer('is_gen', torch.tensor(is_gen))
        self.register_buffer('gen_idx', torch.tensor(gen_idx))

    # ── build the faulted network from a soft hypothesis ─────────────────────
    def network(self, p, alpha, hard=False):
        """
        p: (B, N_CLASS) probabilities (class 0 = no fault), alpha: (B,)
        Returns Yred (B,3,3) and the set-point offset dPm0 (B,3).
        """
        B = p.shape[0]
        pf = p[:, 1:]                                   # (B,12) fault classes
        if hard:
            k = pf.argmax(1)
            pf = torch.zeros_like(pf).scatter_(1, k[:, None], 1.0)
            pf = pf * (p[:, 1:].sum(1, keepdim=True) > p[:, 0:1]).float()

        # scale: alpha for load faults, 1 for line faults, 0 for gen (no dY)
        scale = torch.where(self.is_load.view(1, -1), alpha.view(-1, 1),
                            torch.ones_like(pf))
        w = (pf * scale).to(self.S.cdtype)                          # (B,12)
        Y = self.S.Y_pre.unsqueeze(0) + torch.einsum('bh,hij->bij', w, self.dY)
        Yred, _ = self.S.kron_reduce(Y)

        # generator trips shift the set-point instead of the network
        gw = pf * self.is_gen.view(1, -1).float() * alpha.view(-1, 1)
        dPm0 = torch.zeros(B, 3, dtype=alpha.dtype, device=alpha.device)
        dPm0 = dPm0.index_add(1, self.gen_idx, -gw * self.S.Pm0[self.gen_idx])
        return Yred, dPm0

    # ── rollout ──────────────────────────────────────────────────────────────
    def rollout(self, x0, Yred, Pset_fault, dP=None, record=True):
        """
        Integrate `n_steps` of RK4 holding the fault. dP, if given, is added to
        the set-point for the whole horizon (receding horizon assumes the fault
        persists; the next decision corrects if it clears).
        Returns omega trace (B, n_steps+1, 3) if record else final state.
        """
        Ps = Pset_fault if dP is None else Pset_fault + dP
        # Build the real-valued Pe constants ONCE, then run the loop in pure
        # real arithmetic -- complex exp/einsum per step is ~2x the wall time.
        M, Nc = self.S.pe_consts(Yred)
        x = x0
        w_trace = [x[:, 3:6]] if record else None
        for _ in range(self.n_steps):
            x = self.S.rk4_real(x, M, Nc, Ps, self.dt_roll)
            if record:
                w_trace.append(x[:, 3:6])
        return torch.stack(w_trace, 1) if record else x

    def judged(self, w_trace, vmask=None):
        """
        Peak |omega-1| after the governor grace period, restricted to the
        VALID part of the horizon when vmask is given.

        The trajectory loss is masked at the point the real network changes
        (clearing, or a new fault), so beyond that the rollout is unsupervised.
        Reading OOL from there judges the model on a region it was never
        trained for -- and it failed in the unsafe direction (missing genuine
        violations). Both the prediction and the measured target use the same
        mask, so the comparison is like for like.

        Returns (peak, has_valid). has_valid is False where the valid horizon
        ends before the grace period, so there is nothing to judge.
        """
        i0 = int(round(self.grace / self.dt_roll))
        d = (w_trace[:, i0:, :] - 1.0).abs()               # (B, T', 3)
        if vmask is None:
            return d.amax(dim=2).amax(dim=1), torch.ones(d.shape[0],
                                                         dtype=torch.bool,
                                                         device=d.device)
        m = vmask[:, i0:].unsqueeze(-1).expand_as(d)
        has = m.flatten(1).sum(1) > 0
        peak = d.masked_fill(m == 0, -1.0).amax(dim=2).amax(dim=1)
        return peak, has

    def violation(self, w_trace, vmask=None):
        """
        Mean excess over the envelope across ALL valid timesteps.

        A hinge on the peak alone puts gradient on one timestep; this spreads
        it over every step that is outside the envelope, which is what actually
        trains dP.
        """
        i0 = int(round(self.grace / self.dt_roll))
        d = (w_trace[:, i0:, :] - 1.0).abs()
        ex = torch.relu(d - (self.limit - self.margin))
        if vmask is None:
            return ex.mean(dim=(1, 2))
        m = vmask[:, i0:].unsqueeze(-1).expand_as(d)
        return (ex * m).flatten(1).sum(1) / m.flatten(1).sum(1).clamp(min=1)

    # ── excursion description ────────────────────────────────────────────────
    N_EXC = 10

    def excursion_features(self, w_trace, vmask=None):
        """
        Compact description of a predicted excursion, for conditioning dP.

        The one-shot head never saw this: the correction needed depends on how
        far past the limit omega goes, when, and in which direction -- all of
        which the rollout computes AFTER the old head had already emitted dP.
        Signed values matter: a generator trip drives omega down, a load fault
        up, and dP must follow the sign.
        """
        i0 = int(round(self.grace / self.dt_roll))
        d = w_trace[:, i0:, :] - 1.0                       # (B,T',3) SIGNED
        B, T, G = d.shape
        if vmask is None:
            m = torch.ones_like(d)
        else:
            m = vmask[:, i0:].unsqueeze(-1).expand_as(d)
        a = (d * m).abs()
        peak_g, idx_g = a.max(dim=1)                       # (B,3) per generator
        sign_g = torch.gather(d, 1, idx_g.unsqueeze(1)).squeeze(1).sign()
        peak = peak_g.max(dim=1).values                    # (B,)
        t_peak = idx_g.float().mean(dim=1) / max(T - 1, 1)
        excess = (torch.relu(a - (self.limit - self.margin)) * m).flatten(1).sum(1) \
            / m.flatten(1).sum(1).clamp(min=1)
        valid = m.flatten(1).sum(1) / (T * G)
        mean_d = (d * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)   # (B,3)
        return torch.cat([
            (peak / self.limit).unsqueeze(1),
            peak_g * sign_g / self.limit,
            t_peak.unsqueeze(1),
            (excess / self.limit).unsqueeze(1),
            valid.unsqueeze(1),
            mean_d / self.limit,
        ], dim=1)                                          # (B, N_EXC)

    # ── full forward ─────────────────────────────────────────────────────────
    def forward(self, win, x_meas, hard=False, ctrl=True, vmask=None):
        """
        win:    (B, W, n_feat) measurement window ending at the decision instant
        x_meas: (B, 9) measured [delta_rel, omega, Pm] at that instant
        ctrl:   run the control rollout. Set False while the control loss is
                still switched off (--ctrl_start) -- it is a second full
                rollout, so skipping it nearly halves the step cost.
        """
        out = self.enc(win)
        p = torch.softmax(out['hyp_logits'], dim=1)
        alpha = self.alpha_lo + (self.alpha_hi - self.alpha_lo) * \
                torch.sigmoid(out['alpha_raw'])

        Yred, dPm0 = self.network(p, alpha, hard=hard)
        x0 = x_meas + 0.1 * out['x0_corr']          # small learned denoising
        Pset = self.S.Pm0.unsqueeze(0) + dPm0

        # identification branch: no correction, compare against measured future
        w_free = self.rollout(x0, Yred, Pset, dP=None)

        # control branch: parameters DETACHED so the controller cannot reduce
        # its loss by mis-identifying the system into an easy one.
        res = {**out, 'p': p, 'alpha': alpha, 'x0': x0,
               'Yred': Yred, 'Pset': Pset, 'w_free': w_free}
        if not ctrl:
            res['dP'] = self.dp_max * torch.tanh(out['dP_raw'])
            return res

        x0d, Yd, Pd = x0.detach(), Yred.detach(), Pset.detach()

        if self.dp_head == 'linear':
            dP = self.dp_max * torch.tanh(out['dP_raw'])
            res['dP'] = dP
            res['w_ctrl'] = self.rollout(x0d, Yd, Pd, dP=dP)
            return res

        # ── cascaded head: identification -> prediction -> control ──────────
        # Strictly staged: the context is DETACHED, so the control objective
        # trains only this head and cannot reshape the identification features
        # (the same reason Yred/x0/Pset are detached).
        #
        # Each refinement step rolls the CURRENT dP forward, describes the
        # resulting excursion, and feeds that plus the current dP into an
        # LSTMCell whose state carries across steps -- an unrolled learned
        # optimiser with the simulator in the loop, rather than a single
        # feedforward guess. Unrolled during TRAINING too, so the head learns
        # to emit corrections that refine well, not just ones that score well
        # cold.
        ctx = torch.cat([out['z'], p, alpha.unsqueeze(1), x0], dim=1).detach()
        B = ctx.shape[0]
        h = ctx.new_zeros(B, self.dp_hidden)
        c = ctx.new_zeros(B, self.dp_hidden)
        raw = ctx.new_zeros(B, 3)
        w_cur = w_free.detach()
        traces = []
        for _ in range(self.n_refine):
            exc = self.excursion_features(w_cur, vmask)
            inp = torch.cat([ctx, self.dp_max * torch.tanh(raw), exc], dim=1)
            if self.dp_cell_type == 'lstm':
                h, c = self.dp_cell(inp, (h, c))
            else:
                h = self.dp_cell(inp)
            raw = raw + self.dp_fc(h)
            dP = self.dp_max * torch.tanh(raw)
            w_cur = self.rollout(x0d, Yd, Pd, dP=dP)
            traces.append(w_cur)
        res['dP'] = dP
        res['w_ctrl'] = w_cur
        res['w_ctrl_steps'] = traces
        return res
