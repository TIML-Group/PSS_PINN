"""
alpha_residual.py -- signed severity error, split by fault type (9-bus).

The reported alpha metric is an RMSE, which discards sign. A model that
systematically UNDER-estimates severity predicts a milder excursion, proposes
an undersized dP, and under-rescues; one that over-estimates wastes control
effort. Those are different failures with the same RMSE.

One structural caveat the table makes explicit. alpha is produced as

    alpha = alpha_lo + (alpha_hi - alpha_lo) * sigmoid(raw)      (0.10 .. 1.00)

so alpha = 1.00 is reachable only in the limit. LINE faults are labelled
alpha = 1.0 exactly (a line is open or it is not), which means their residual
is bounded at or below zero BY CONSTRUCTION -- a negative mean there is the
parameterisation, not the model. Only gen and load faults carry an alpha the
model can miss in either direction, so the honest headline is the gen+load
row, reported separately below.

Only the ENCODER runs here: alpha is a head on z and needs no rollout, so this
is fast. The mask sharpness beta is taken from the checkpoint as trained --
forcing beta_min would evaluate a different window than the one reported
(see refine_dp.py).

    python alpha_residual.py --data data_mm \
        --ckpt "LSTM+LSTM=receding_cascade/refined/best.pt" \
        --ckpt "MLP+MLP=receding_fullmlp/refined/best.pt"
"""
import argparse
import json
import os

import numpy as np
import torch

from pinn_model import FaultEncoder

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', action='append', required=True,
                help='"Label=path/to/best.pt", repeatable.')
ap.add_argument('--data', required=True, help='directory from convert_dataset.py')
ap.add_argument('--split', default='test', choices=['val', 'test', 'train'])
ap.add_argument('--n', type=int, default=4096)
ap.add_argument('--batch', type=int, default=512)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', default=None, help='optional JSON path')
args = ap.parse_args()

specs = []
for c in args.ckpt:
    if '=' not in c:
        raise SystemExit(f"--ckpt needs 'Label=path', got {c!r}")
    lab, path = c.split('=', 1)
    specs.append((lab.strip(), path.strip()))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── data (layout mirrors train_receding.py exactly) ─────────────────────────
X = np.load(os.path.join(args.data, 'X.npy'), mmap_mode='r')        # (N, T, 33)
md = np.load(os.path.join(args.data, 'meta.npz'))
g = lambda k: md[k].ravel()
t_grid = md['t'].ravel(); DT = float(md['dt_solver'])
y_type = g('y_type').astype(int)
meta_alpha = g('meta_alpha')
t_fault = g('y_t_fault'); t_clear = g('meta_t_clear')

D_IDX, W_IDX, PM_IDX = [0, 2, 4], [1, 3, 5], [6, 7, 8]
ID_IDX, IQ_IDX = [9, 11, 13], [10, 12, 14]
VM_IDX = list(range(15, 33, 2))
TH_IDX = list(range(16, 33, 2))
TYPE_NAMES = ['gen', 'load', 'line']
ALPHA_BINS = [(0.10, 0.30), (0.30, 0.50), (0.50, 0.70), (0.70, 0.90), (0.90, 1.01)]


def featurize(raw):
    d = raw[..., D_IDX]; w = raw[..., W_IDX]; pm = raw[..., PM_IDX]
    th = raw[..., TH_IDX]
    return np.concatenate([d[..., 1:] - d[..., :1], (w - 1.0) / 0.01, pm,
                           raw[..., ID_IDX], raw[..., IQ_IDX],
                           raw[..., VM_IDX], th[..., 1:] - th[..., :1]], axis=-1)


def collect(label, path, first=False):
    ck = torch.load(path, map_location=device, weights_only=False)
    A = ck['args']
    W_STEPS = int(round(A['w_max'] / A['dt_window']))
    W_STRIDE = int(round(A['dt_window'] / DT))
    sp = ck['split']
    idx = np.asarray({'val': sp['val'], 'test': sp['test'],
                      'train': sp['train']}[args.split])

    enc = FaultEncoder(int(ck['n_feat']), w_max_steps=W_STEPS,
                       dt_window=A['dt_window'],
                       lstm_layers=A.get('lstm_layers', 10),
                       init_window=A['init_window'] / A['w_max'],
                       max_window=A['w_max'], beta0=A['beta0'],
                       encoder=A.get('encoder', 'lstm'),
                       mlp_proj=A.get('mlp_proj')).to(device)
    # load only the encoder's tensors out of the full controller state dict
    sd = {k[len('enc.'):]: v for k, v in ck['model'].items() if k.startswith('enc.')}
    enc.load_state_dict(sd)                       # strict: shapes must match
    enc.eval()
    ALPHA_LO, ALPHA_HI = 0.10, 1.00               # RolloutController defaults

    if first:
        print(f"Device: {device}")
    print(f"  {label}: epoch {ck['epoch']}, window {ck['window_s']*1000:.1f} ms, "
          f"beta {float(enc.beta)*1000:.1f} ms (from checkpoint), "
          f"encoder '{A.get('encoder', 'lstm')}'")

    lo, hi = A['w_max'], t_grid[-1] - A['horizon']
    rng = np.random.default_rng(args.seed)        # same draw for every model
    a_pred, a_true, typ = [], [], []
    done = 0
    while done < args.n:
        bs = min(args.batch, args.n - done)
        ks = rng.choice(idx, bs)
        tk = rng.uniform(lo, hi, bs)
        if A.get('p_active', 0) > 0:
            pick = rng.random(bs) < A['p_active']
            a = np.clip(t_fault[ks], lo, hi); b_ = np.clip(t_clear[ks], lo, hi)
            if A.get('onset_window', 0) > 0:
                b_ = np.minimum(b_, a + A['onset_window'])
            ok = pick & (b_ > a)
            tk[ok] = rng.uniform(a[ok], b_[ok])
        i_end = np.round(tk / DT).astype(int)

        win = np.empty((bs, W_STEPS, X.shape[2]), np.float32)
        for j, (k, ie) in enumerate(zip(ks, i_end)):
            i0 = ie - (W_STEPS - 1) * W_STRIDE
            win[j] = np.asarray(X[k, i0:ie + 1:W_STRIDE])
        feat = torch.tensor(featurize(win)).to(device)

        with torch.no_grad():
            out = enc(feat)
            al = ALPHA_LO + (ALPHA_HI - ALPHA_LO) * torch.sigmoid(out['alpha_raw'])

        act = (tk >= t_fault[ks]) & (tk < t_clear[ks])
        a_pred.append(al.cpu().numpy()[act])
        a_true.append(meta_alpha[ks][act])
        typ.append(y_type[ks][act])
        done += bs
        print(f"\r    {done}/{args.n}", end='', flush=True)
    print()
    return (np.concatenate(a_pred), np.concatenate(a_true),
            np.concatenate(typ).astype(int))


def stats(res):
    if not len(res):
        return dict(n=0)
    return dict(n=int(len(res)), bias=float(res.mean()),
                rmse=float(np.sqrt((res ** 2).mean())),
                sd=float(res.std()), under=float((res < 0).mean()),
                p05=float(np.percentile(res, 5)),
                p95=float(np.percentile(res, 95)))


print(f"\nSeverity (alpha) signed residual -- {args.split} split, "
      f"{args.n} instants, active-fault only")
print("positive bias = model OVER-estimates severity; negative = UNDER-estimates\n")

results = {}
for i, (lab, path) in enumerate(specs):
    ap_, at_, ty_ = collect(lab, path, first=(i == 0))
    res = ap_ - at_
    gl = ty_ != 3
    per = {'ALL': stats(res), 'gen+load': stats(res[gl])}
    for ti_, tname in enumerate(TYPE_NAMES):
        per[tname] = stats(res[ty_ == ti_ + 1])
    results[lab] = dict(per_type=per, by_alpha={},
                        mean_pred_line=float(ap_[ty_ == 3].mean())
                        if (ty_ == 3).any() else float('nan'))
    for lo_, hi_ in ALPHA_BINS:
        m = gl & (at_ >= lo_) & (at_ < hi_)
        if m.sum():
            results[lab]['by_alpha'][f"{lo_:.2f}-{hi_:.2f}"] = stats(res[m])

    print(f"\n{'='*74}")
    print(f"{lab}")
    print(f"{'='*74}")
    print(f"  {'group':10s} {'n':>6s} {'bias':>9s} {'RMSE':>8s} {'sd':>8s} "
          f"{'under%':>8s} {'5th':>8s} {'95th':>8s}")
    for key in ('gen', 'load', 'line', 'gen+load', 'ALL'):
        s = per[key]
        if not s['n']:
            continue
        mark = '  <- headline' if key == 'gen+load' else (
            '  <- bounded by construction' if key == 'line' else '')
        print(f"  {key:10s} {s['n']:6d} {s['bias']:+9.4f} {s['rmse']:8.4f} "
              f"{s['sd']:8.4f} {100*s['under']:7.1f}% {s['p05']:+8.3f} "
              f"{s['p95']:+8.3f}{mark}")

    print(f"\n  gen+load by true severity:")
    print(f"  {'alpha':12s} {'n':>6s} {'bias':>9s} {'RMSE':>8s} {'under%':>8s}")
    for k, s in results[lab]['by_alpha'].items():
        print(f"  {k:12s} {s['n']:6d} {s['bias']:+9.4f} {s['rmse']:8.4f} "
              f"{100*s['under']:7.1f}%")

    gls = per['gen+load']
    if gls['n']:
        se = gls['sd'] / np.sqrt(gls['n'])
        z = gls['bias'] / se if se > 0 else 0.0
        print(f"\n  gen+load bias {gls['bias']:+.4f} +/- {1.96*se:.4f} (95% CI), "
              f"z = {z:+.1f}")
        if abs(z) < 2:
            print("  -> NOT significantly biased: the alpha error is symmetric, so a\n"
                  "     severity bias does not explain any rescue shortfall.")
        elif z < 0:
            print("  -> UNDER-estimates severity. Predicts a milder excursion than the\n"
                  "     truth, so dP is sized small -- consistent with under-rescue.")
        else:
            print("  -> OVER-estimates severity. Predicts a worse excursion than the\n"
                  "     truth, so dP is sized large -- wasted control effort.")
    print(f"\n  mean predicted alpha on line faults (true = 1.000): "
          f"{results[lab]['mean_pred_line']:.4f}")

if len(specs) > 1:
    print(f"\n{'='*74}")
    print("COMPARISON (gen+load, the two-sided group)")
    print(f"{'='*74}")
    print(f"  {'model':22s} {'bias':>9s} {'RMSE':>8s} {'under%':>8s}")
    for lab in results:
        s = results[lab]['per_type']['gen+load']
        print(f"  {lab:22s} {s['bias']:+9.4f} {s['rmse']:8.4f} "
              f"{100*s['under']:7.1f}%")
    print("\n  A difference in RMSE with similar bias means one model is noisier.")
    print("  A difference in BIAS means they fail in different directions, which")
    print("  is what would link an alpha error to a rescue-rate difference.")

if args.out:
    json.dump(results, open(args.out, 'w'), indent=1)
    print(f"\n  -> {args.out}")
