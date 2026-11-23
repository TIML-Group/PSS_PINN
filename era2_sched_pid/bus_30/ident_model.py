"""ident_model.py -- the identification MLP and era-1 featurizer as defined in train_ident_v2.py (kept identical;
checkpoints load with strict=True, so any drift fails loudly)."""
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


class IdentNet(nn.Module):
    def __init__(self, variant, profile):
        super().__init__()
        self.variant, self.profile = variant, profile
        self.proj = nn.Linear(NF, PROJ, bias=False)
        self.trunk = MLPBlock(W_STEPS * PROJ, HIDDEN, MLP_LAYERS)
        if variant == 'cls':
            self.head_c = nn.Linear(HIDDEN, 59); self.head_a = nn.Linear(HIDDEN, 1)
        else:
            self.head_g = nn.Linear(HIDDEN, 36)
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
        z = self.trunk(self.proj(x * self.mask().view(1, -1, 1)).flatten(1))
        if self.variant == 'cls':
            return dict(logits=self.head_c(z), alpha=torch.sigmoid(self.head_a(z).squeeze(-1)))
        return dict(logg=self.head_g(z))
