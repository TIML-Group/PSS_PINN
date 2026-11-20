"""
oneshot_common.py
=================
Shared pieces for the 9-bus one-shot dP experiment (no RK4 at decision time).

  measurement window -> frozen MLP encoder (paper's MLP+MLP checkpoint)
                     -> OneShotHead -> bounded dP (one per generator)

Window featurisation is copied verbatim from verify_true.make_batch so the
encoder sees exactly what it saw in training / verification.
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn

BUS9 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BUS9)
from ieee9_physics import IEEE9                               # noqa: E402
from pinn_model import FaultEncoder, RolloutController, N_CLASS  # noqa: E402

TEACHER = os.path.join(BUS9, 'receding_fullmlp', 'refined', 'best.pt')
DATA = os.path.join(BUS9, 'data_mm')

D_IDX, W_IDX, PM_IDX = [0, 2, 4], [1, 3, 5], [6, 7, 8]
ID_IDX, IQ_IDX = [9, 11, 13], [10, 12, 14]
VM_IDX = list(range(15, 33, 2)); TH_IDX = list(range(16, 33, 2))


def load_teacher(device, ckpt=TEACHER, dtype=torch.float32):
    """Teacher MLP+MLP controller (encoder + cascade head), eval mode."""
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    A = ck['args']
    md = np.load(os.path.join(DATA, 'meta.npz'))
    S = IEEE9(os.path.join(BUS9, 'ieee9_base.mat'),
              os.path.join(BUS9, 'ieee9_gov_simulink_data.mat'),
              dtype=dtype, device=device, lines=md['lines'].astype(int))
    enc = FaultEncoder(ck['n_feat'], w_max_steps=ck['w_steps'],
                       dt_window=A['dt_window'],
                       lstm_layers=A.get('lstm_layers', 10),
                       init_window=A['init_window'] / A['w_max'],
                       max_window=A['w_max'], beta0=A['beta0'],
                       encoder=A.get('encoder', 'lstm'),
                       mlp_proj=A.get('mlp_proj'))
    model = RolloutController(S, enc, horizon=A['horizon'], dt_roll=A['dt_roll'],
                              omega_limit=float(md['omega_limit']),
                              margin=A.get('margin', 1e-4),
                              dp_head=A.get('dp_head', 'linear'),
                              dp_hidden=A.get('dp_hidden', 128),
                              dp_fc_layers=A.get('dp_fc_layers', 2),
                              n_refine=A.get('n_refine', 3),
                              dp_cell=A.get('dp_cell', 'lstm')).to(device)
    model.load_state_dict(ck['model']); model.eval()
    return model, ck, A, S


def featurize(win, bs):
    """win: (bs, W, 33) raw channels -> (feat, x_meas), as verify_true."""
    d = win[..., D_IDX]; w = win[..., W_IDX]; pm = win[..., PM_IDX]
    th = win[..., TH_IDX]
    feat = np.concatenate([d[..., 1:] - d[..., :1], (w - 1.0) / 0.01, pm,
                           win[..., ID_IDX], win[..., IQ_IDX],
                           win[..., VM_IDX], th[..., 1:] - th[..., :1]], -1)
    re = win[:, -1, :]; dd = re[:, D_IDX]
    x_meas = np.concatenate([np.stack([np.zeros(bs), dd[:, 1] - dd[:, 0],
                                       dd[:, 2] - dd[:, 0]], 1),
                             re[:, W_IDX], re[:, PM_IDX]], 1)
    return feat.astype(np.float32), x_meas.astype(np.float32)


def raw_windows(X, ks, tk, W_STEPS, W_STRIDE, DT):
    i_end = np.round(np.asarray(tk) / DT).astype(int)
    win = np.empty((len(ks), W_STEPS, 33), np.float32)
    for j, (k, ie) in enumerate(zip(ks, i_end)):
        # windows reaching before t = 0 (t_apply < w_max) repeat the first,
        # pre-fault sample; the encoder's learned mask (~0.28 s) zeroes them.
        idx = np.clip(ie - np.arange(W_STEPS - 1, -1, -1) * W_STRIDE, 0, None)
        win[j] = np.asarray(X[k])[idx]
    return win


class OneShotHead(nn.Module):
    """Encoder output -> bounded dP (3). Input = [z, p, alpha, x0], the same
    context the cascade head reads, minus the rollout excursion features."""

    def __init__(self, hidden_in=128, hidden=128, layers=2, dp_max=1.0):
        super().__init__()
        d_in = hidden_in + N_CLASS + 1 + 9
        net, d = [], d_in
        for _ in range(layers):
            net += [nn.Linear(d, hidden), nn.Tanh()]
            d = hidden
        net += [nn.Linear(d, 3)]
        self.net = nn.Sequential(*net)
        self.dp_max = dp_max
        self.register_buffer('mu', torch.zeros(d_in))
        self.register_buffer('sd', torch.ones(d_in))

    @staticmethod
    def context(model, out, x_meas):
        """Encoder outputs -> head input (no rollout, no network build)."""
        p = torch.softmax(out['hyp_logits'], dim=1)
        alpha = model.alpha_lo + (model.alpha_hi - model.alpha_lo) * \
            torch.sigmoid(out['alpha_raw'])
        x0 = x_meas + 0.1 * out['x0_corr']
        return torch.cat([out['z'], p, alpha.unsqueeze(1), x0], dim=1), p

    def forward(self, ctx):
        return self.dp_max * torch.tanh(self.net((ctx - self.mu) / self.sd))
