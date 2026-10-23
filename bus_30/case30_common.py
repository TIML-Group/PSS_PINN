"""
case30_common.py
================
Shared loader and batch builder for the 30-bus post-training scripts
(refine_dp30.py, diagnose_dp30.py, verify_true30.py).

On the 9-bus side the batch builder and the actionability gate were copied into
four scripts, and they drifted apart twice (refine_dp.py once set its rescue
floor on a different definition than training used). Here the three
post-training scripts import ONE copy. It mirrors train_case30.py line for
line; `python case30_common.py --selftest` checks that against the training
script's own make_batch on identical random draws.

Everything that defines the model's inputs comes from the checkpoint, never
recomputed:
  * the split                    (ck['split'])
  * feature normalisation        (ck['feat_mu'], ck['feat_sd'])
  * window / rollout geometry    (ck['w_steps'], ck['args'])
  * mask sharpness beta          (stored in the state dict; NOT overwritten)

The last point differs from the 9-bus scripts, which forced beta to
--beta_min after loading. beta anneals over the first 60% of the planned
epochs, so a checkpoint selected early (epoch 45 of 200 -> beta ~0.027 s)
was trained and validated with a much softer mask edge than beta_min
(0.005 s). Forcing it evaluates a different window than the one reported.
"""

import os

import numpy as np
import torch

from case30_physics import CASE30, NG, NB, N_STATE, N_ACT
from pinn_model30 import FaultEncoder, RolloutController, N_CLASS

TYPE_NAMES = ['gen', 'load', 'line']
AGE_BINS = [(0.0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1e9)]

# channel layout, identical to train_case30.py
D_IDX = list(range(0, 2 * NG, 2))
W_IDX = list(range(1, 2 * NG, 2))
PM_IDX = list(range(2 * NG, 3 * NG))
FUT_IDX = W_IDX + PM_IDX
ID_IDX = list(range(3 * NG, 5 * NG, 2))
IQ_IDX = list(range(3 * NG + 1, 5 * NG, 2))


def featurize(raw):
    """raw: (..., 90) -> (..., 88). Same as train_case30.featurize."""
    n_ch = raw.shape[-1]
    vm_idx = list(range(5 * NG, n_ch, 2))
    th_idx = list(range(5 * NG + 1, n_ch, 2))
    d = raw[..., D_IDX]; w = raw[..., W_IDX]; pm = raw[..., PM_IDX]
    th = raw[..., th_idx]
    return np.concatenate([d[..., 1:] - d[..., :1], (w - 1.0) / 0.01, pm,
                           raw[..., ID_IDX], raw[..., IQ_IDX],
                           raw[..., vm_idx], th[..., 1:] - th[..., :1]], axis=-1)


def actionable(has, x_meas, active, limit):
    """
    THE gate. A decision instant counts toward rescue only if
      * the judged window is non-empty           (has)
      * omega is still inside the envelope now   (reachable)
      * a fault is active now                    (something to correct)
    Identical to train_case30.py (lines `reach = has_c & (dev_now <= ...) &
    active`). Change it here and there together, or not at all.
    """
    dev_now = (x_meas[:, NG:2 * NG] - 1.0).abs().amax(dim=1)
    return has & (dev_now <= limit) & active


class Setup:
    """Checkpoint + data + model, built exactly the way training built them."""

    def __init__(self, ckpt, data, base='case30_base.mat',
                 sys_mat='case30_simulink_data.mat', device=None, verbose=True,
                 dt_roll=None, n_roll=None, n_refine=None):
        """dt_roll/n_roll/n_refine override the checkpoint's rollout geometry
        (the measured window is untouched). Mirror of the 9-bus verify_true.py
        sweep support: dt_roll changes the SAMPLE SPACING (horizon kept:
        n_roll = horizon / dt_roll); an explicit --n_roll sets the count
        directly. Shapes are geometry-free (LSTM is hidden-only), so the
        strict state-dict load is unaffected."""
        for f_ in (ckpt, base, sys_mat, os.path.join(data, 'X.npy')):
            if not os.path.exists(f_):
                raise SystemExit(f"not found: {f_}")
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        dev = self.device
        ck = torch.load(ckpt, map_location=dev, weights_only=False)
        self.ck, self.A = ck, ck['args']
        A = self.A

        # data
        self.X = np.load(os.path.join(data, 'X.npy'), mmap_mode='r')
        # Whole file into RAM when it fits: every epoch re-touches all scenarios in random
        # order, so an mmap over a spinning disk thrashes (GPU idles waiting on batch build).
        if self.X.nbytes < 4e9:
            self.X = np.asarray(self.X)
        md = np.load(os.path.join(data, 'meta.npz'), allow_pickle=True)
        g = lambda k: md[k].ravel()
        self.t_grid = g('t'); self.DT = float(md['dt_solver'])
        self.T = self.X.shape[1]; self.N_CH = self.X.shape[2]
        self.y_type = g('y_type').astype(int)
        self.meta_loc = g('meta_loc').astype(int)
        self.meta_alpha = g('meta_alpha')
        self.t_fault = g('y_t_fault'); self.t_clear = g('meta_t_clear')
        self.dp_status = g('dp_status').astype(int)
        self.y_dp = (md['y_delta_p'][:, :N_ACT] if 'y_delta_p' in md
                     else np.zeros((len(self.y_type), N_ACT)))
        self.OMEGA_LIMIT = float(md['omega_limit'])
        sp = ck['split']
        self.idx_tr = np.asarray(sp['train']); self.idx_va = np.asarray(sp['val'])
        self.idx_te = np.asarray(sp['test'])

        # geometry from the checkpoint, with rollout overrides applied
        self.W_STEPS = int(ck['w_steps'])
        self.W_STRIDE = int(round(A['dt_window'] / self.DT))
        self.DT_ROLL_EFF = A['dt_roll'] if dt_roll is None else float(dt_roll)
        self.ROLL_STRIDE = int(round(self.DT_ROLL_EFF / self.DT))
        self.N_ROLL_CFG = int(round(A['horizon'] / A['dt_roll']))
        if n_roll is not None:
            self.N_ROLL = int(n_roll)
        elif dt_roll is not None:
            # keep the horizon, change the spacing
            self.N_ROLL = int(round(A['horizon'] / self.DT_ROLL_EFF))
        else:
            self.N_ROLL = self.N_ROLL_CFG
        self.N_REFINE = A.get('n_refine', 3) if n_refine is None else int(n_refine)
        self.HORIZON_EFF = self.N_ROLL * self.DT_ROLL_EFF
        self.dt_roll_override = (dt_roll is not None or n_roll is not None
                                 or n_refine is not None)
        if verbose and self.dt_roll_override:
            print(f"Rollout override: dt_roll {self.DT_ROLL_EFF*1000:.0f} ms, "
                  f"n_roll {self.N_ROLL} (horizon eff "
                  f"{self.HORIZON_EFF:.2f} s), n_refine {self.N_REFINE}")
        if 'feat_mu' in ck:
            self.FEAT_MU = np.asarray(ck['feat_mu'], np.float32)
            self.FEAT_SD = np.asarray(ck['feat_sd'], np.float32)
            norm_msg = 'from checkpoint'
        else:
            n = int(ck['n_feat'])
            self.FEAT_MU = np.zeros(n, np.float32); self.FEAT_SD = np.ones(n, np.float32)
            norm_msg = 'NONE (checkpoint predates normalisation)'

        # system + model
        self.S = CASE30(base, sys_mat, dtype=torch.float32, device=dev)
        self.HYPOTHESES = self.S.HYPOTHESES
        enc = FaultEncoder(int(ck['n_feat']), w_max_steps=self.W_STEPS,
                           dt_window=A['dt_window'],
                           lstm_layers=A['lstm_layers'],
                           init_window=A['init_window'], max_window=A['w_max'],
                           beta0=A['beta0'], encoder=A.get('encoder', 'lstm'),
                           mlp_proj=A.get('mlp_proj', 160))
        self.model = RolloutController(
            self.S, enc, horizon=self.HORIZON_EFF, dt_roll=self.DT_ROLL_EFF,
            omega_limit=self.OMEGA_LIMIT, margin=A['margin'],
            dp_head=A['dp_head'], dp_hidden=A['dp_hidden'],
            dp_fc_layers=A['dp_fc_layers'], n_refine=self.N_REFINE,
            dp_cell=A.get('dp_cell', 'lstm')).to(dev)
        self.model.load_state_dict(ck['model'])      # strict: shapes must match
        self.enc = enc

        if verbose:
            print(f"Device: {dev}")
            print(f"Loaded {ckpt}: epoch {ck['epoch']}, window "
                  f"{ck['window_s']*1000:.1f} ms, mask beta "
                  f"{float(enc.beta)*1000:.1f} ms (from checkpoint), "
                  f"dP head '{A['dp_head']}', encoder "
                  f"'{A.get('encoder', 'lstm')}', dP cell '{A.get('dp_cell', 'lstm')}'")
            print(f"Split from checkpoint: {len(self.idx_tr)} train / "
                  f"{len(self.idx_va)} val / {len(self.idx_te)} test   "
                  f"| feature normalisation: {norm_msg}")

    # ── batch builder: mirrors train_case30.make_batch ──────────────────────
    def make_batch(self, ks, rng, tk_fixed=None, onset_only=False):
        """
        Decision instants drawn as in training: uniform over the run, with a
        share p_active moved into [t_fault, t_fault + onset_window).
        onset_only=True moves EVERY instant there (diagnose_dp30.py).
        tk_fixed overrides the draw entirely (plot case selection).
        Returns a dict of CPU tensors plus the numpy ks / tk / valid_t.
        """
        A = self.A
        ks = np.asarray(ks); bs = len(ks)
        lo, hi = A['w_max'], self.t_grid[-1] - A['horizon']
        tk = rng.uniform(lo, hi, bs)
        if tk_fixed is not None:
            tk = np.asarray(tk_fixed, dtype=float).copy()
        elif onset_only or A.get('p_active', 0) > 0:
            pick = (np.ones(bs, bool) if onset_only
                    else rng.random(bs) < A['p_active'])
            a = np.clip(self.t_fault[ks], lo, hi)
            b = np.clip(self.t_clear[ks], lo, hi)
            if A.get('onset_window', 0) > 0:
                b = np.minimum(b, a + A['onset_window'])
            ok = pick & (b > a)
            tk[ok] = rng.uniform(a[ok], b[ok])
        i_end = np.round(tk / self.DT).astype(int)

        win = np.empty((bs, self.W_STEPS, self.N_CH), np.float32)
        fut = np.empty((bs, self.N_ROLL + 1, 2 * NG), np.float32)
        for j, (k, ie) in enumerate(zip(ks, i_end)):
            i0 = ie - (self.W_STEPS - 1) * self.W_STRIDE
            blk = np.asarray(self.X[k, i0:ie + self.N_ROLL * self.ROLL_STRIDE + 1])
            o = ie - i0
            win[j] = blk[0:o + 1:self.W_STRIDE]
            f = blk[o::self.ROLL_STRIDE][:self.N_ROLL + 1][:, FUT_IDX]
            if f.shape[0] < self.N_ROLL + 1:
                # the scenario ends before the horizon (rare, huge dt_roll)
                f = np.concatenate([f, np.repeat(f[-1:], self.N_ROLL + 1 - f.shape[0], 0)])
            fut[j] = f

        feat = (featurize(win) - self.FEAT_MU) / self.FEAT_SD
        re = win[:, -1, :]; d = re[:, D_IDX]
        x_meas = np.concatenate([d - d[:, :1], re[:, W_IDX], re[:, PM_IDX]], 1)

        active = (tk >= self.t_fault[ks]) & (tk < self.t_clear[ks])
        t_change = np.where(active, self.t_clear[ks],
                            np.where(tk < self.t_fault[ks], self.t_fault[ks], np.inf))
        valid_t = np.clip(t_change - tk, 0.0, A['horizon'])
        step_t = np.arange(self.N_ROLL + 1) * self.DT_ROLL_EFF
        vmask = (step_t[None, :] <= valid_t[:, None]).astype(np.float32)
        age = (tk - self.t_fault[ks]).astype(np.float32)
        return dict(feat=torch.tensor(feat),
                    x_meas=torch.tensor(x_meas, dtype=torch.float32),
                    fut=torch.tensor(fut), vmask=torch.tensor(vmask),
                    active=torch.tensor(active), age=torch.tensor(age),
                    ks=ks, tk=tk, valid_t=valid_t)

    def to_dev(self, b):
        return {k: (v.to(self.device) if torch.is_tensor(v) else v)
                for k, v in b.items()}

    # ── ground truth for a scenario ─────────────────────────────────────────
    def true_network(self, ks):
        """Exact faulted reduced network and set-point offset from METADATA
        (not the model's hypothesis). Same construction as case30_lib.m."""
        n = len(ks)
        Yf = torch.zeros(n, NG, NG, dtype=self.S.cdtype, device=self.device)
        dPm = torch.zeros(n, NG, dtype=torch.float32, device=self.device)
        for j, k in enumerate(ks):
            typ = TYPE_NAMES[self.y_type[k] - 1]
            Yf[j], dPm[j] = self.S.scenario(typ, int(self.meta_loc[k]),
                                            float(self.meta_alpha[k]))
        return Yf, dPm


def fmt_age(bins, hits, ns):
    return "  ".join(
        f"{lo*1000:.0f}-{(hi*1000 if hi < 1e8 else 9999):.0f}ms:"
        f"{(h / n if n else float('nan')):.2f}(n={n})"
        for (lo, hi), h, n in zip(bins, hits, ns))


# ── self-test against the training script ───────────────────────────────────
if __name__ == '__main__':
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--train_script', default='train_case30.py')
    a = ap.parse_args()
    su = Setup(a.ckpt, a.data)
    # run train_case30.py up to (not including) its model section, with the
    # checkpoint's own arguments, and compare the two builders draw for draw
    src = open(a.train_script, encoding='utf-8').read()
    src = src[:src.index('# ── model ──')]
    argv = ['train_case30.py', '--data', a.data, '--init', a.ckpt]
    for k in ('w_max', 'dt_window', 'horizon', 'dt_roll', 'p_active',
              'onset_window', 'seed'):
        argv += [f'--{k}', str(su.A[k])]
    sys.argv = argv
    g = {'__name__': 'train_probe'}
    exec(compile(src, a.train_script, 'exec'), g)
    ks = su.idx_va[:32]
    tb = g['make_batch'](ks, np.random.default_rng(5))
    cb = su.make_batch(ks, np.random.default_rng(5))
    pairs = [('feat', 0), ('x_meas', 1), ('fut', 2), ('active', 5),
             ('vmask', 6), ('age', 7)]
    ok = True
    for name, i in pairs:
        d = float((tb[i].float() - cb[name].float()).abs().max())
        ok &= d == 0.0
        print(f"  {name:7s} max |train - common| = {d:.3e}")
    print("IDENTICAL to train_case30.make_batch" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)
