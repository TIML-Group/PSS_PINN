"""ident_model.py -- the identification network and era-1 featurizer (single definition, imported by
train_ident_v2.py, eval_ident_v2.py and profile_latency_e2e.py; checkpoints load with strict=True).

Trunks (training spec v1.1 amendment):
  base -- design of record: flatten (128 x PROJ) -> era-1 MLPBlock 384 x 6 (2.34 M params)
  A    -- temporal pooling: shared per-step MLPBlock on [projection (32) | elapsed/L_max] (192 x 4), then
          mask-weighted mean + soft-masked max over the observed steps -> heads (~0.14 M params)
  B    -- narrow flatten: flatten -> MLPBlock 128 x 3 (spec shape; ~0.56 M params, the first layer 4096 x 128)
  B64  -- narrow flatten at width 64 x 3 (~0.27 M params; the spec's ~0.25 M parameter target)"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'bus_30'))
from pinn_model30 import MLPBlock, HIDDEN, MLP_LAYERS  # noqa: E402

DT_W, W_STEPS = 2e-3, 128
L_MAX = DT_W * W_STEPS
PROJ = 32
BETA0 = 0.010
L_INIT = 0.128
NG, NB, C = 6, 30, 90
NF = (NG - 1) + 4 * NG + NB + (NB - 1)          # 88

D_IDX = list(range(0, 2 * NG, 2)); W_IDX = list(range(1, 2 * NG, 2)); PM_IDX = list(range(2 * NG, 3 * NG))
ID_IDX = list(range(3 * NG, 5 * NG, 2)); IQ_IDX = list(range(3 * NG + 1, 5 * NG, 2))
VM_IDX = list(range(5 * NG, C, 2)); TH_IDX = list(range(5 * NG + 1, C, 2))


def featurize(raw):
    d = raw[..., D_IDX]; w = raw[..., W_IDX]; th = raw[..., TH_IDX]
    return np.concatenate([d[..., 1:] - d[..., :1], (w - 1.0) / 0.01, raw[..., PM_IDX], raw[..., ID_IDX],
                           raw[..., IQ_IDX], raw[..., VM_IDX], th[..., 1:] - th[..., :1]], -1).astype(np.float32)


TRUNKS = {'base': ('flat', HIDDEN, MLP_LAYERS), 'A': ('pool', 192, 4), 'B': ('flat', 128, 3), 'B64': ('flat', 64, 3)}


class IdentNet(nn.Module):
    def __init__(self, variant, profile, trunk='base'):
        super().__init__()
        self.variant, self.profile, self.trunk_kind = variant, profile, trunk
        kind, width, layers = TRUNKS[trunk]
        self.pool = kind == 'pool'
        self.proj = nn.Linear(NF, PROJ, bias=False); nn.init.xavier_normal_(self.proj.weight)
        self.trunk = MLPBlock(PROJ + 1 if self.pool else W_STEPS * PROJ, width, layers)
        zdim = 2 * width if self.pool else width
        if variant == 'cls':
            self.head_c = nn.Linear(zdim, 59); self.head_a = nn.Linear(zdim, 1)
        else:
            self.head_g = nn.Linear(zdim, 36)
        for m in (self.head_c, self.head_a) if variant == 'cls' else (self.head_g,):
            nn.init.xavier_normal_(m.weight, gain=0.1); nn.init.zeros_(m.bias)
        self.register_buffer('elapsed', (torch.arange(W_STEPS) + 1).float() * DT_W)
        self.register_buffer('beta', torch.tensor(BETA0))
        if profile == 'learned':
            f = L_INIT / L_MAX
            self.raw_L = nn.Parameter(torch.tensor(float(np.log(f / (1 - f)))))
        else:
            self.register_buffer('L_fixed', torch.tensor(L_MAX if profile == 'none' else float(profile) / 1000))

    def L(self):
        return L_MAX * torch.sigmoid(self.raw_L) if self.profile == 'learned' else self.L_fixed

    def mask(self):
        if self.profile == 'learned':
            return torch.sigmoid((self.L() - self.elapsed + 0.5 * DT_W) / self.beta)
        return (self.elapsed <= self.L() + 1e-9).float()

    def forward(self, x):
        m = self.mask()
        e = self.proj(x * m.view(1, -1, 1))
        if self.pool:
            pos = (self.elapsed / L_MAX).view(1, -1, 1).expand(e.shape[0], -1, 1)
            h = self.trunk(torch.cat([e, pos], -1))                                  # (B, W, width)
            mw = m.view(1, -1, 1)
            mean = (h * mw).sum(1) / mw.sum(1).clamp(min=1e-6)
            mx = (h - 4.0 * (1.0 - mw)).amax(1)                                      # tanh in [-1,1]: masked steps never win
            z = torch.cat([mean, mx], -1)
        else:
            z = self.trunk(e.flatten(1))
        if self.variant == 'cls':
            return dict(logits=self.head_c(z), alpha=torch.sigmoid(self.head_a(z).squeeze(-1)))
        return dict(logg=self.head_g(z))
