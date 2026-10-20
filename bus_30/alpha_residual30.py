"""
alpha_residual30.py -- signed severity error, split by fault type (30-bus).

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

Compares any number of checkpoints on identical decision instants.

    python alpha_residual30.py --data data30_mm \
        --ckpt "LSTM+LSTM=case30_ctrl/refined/best.pt" \
        --ckpt "MLP+MLP=case30_fullmlp_ctrl/refined/best.pt"
"""
import argparse
import json
import os

import numpy as np
import torch

from case30_common import Setup, TYPE_NAMES

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', action='append', required=True,
                help='"Label=path/to/best.pt", repeatable.')
ap.add_argument('--data', required=True)
ap.add_argument('--base', default='case30_base.mat')
ap.add_argument('--sys', default='case30_simulink_data.mat')
ap.add_argument('--split', default='test', choices=['val', 'test', 'train'])
ap.add_argument('--n', type=int, default=4096,
                help='decision instants (drawn once, shared by every model)')
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

ALPHA_BINS = [(0.10, 0.30), (0.30, 0.50), (0.50, 0.70), (0.70, 0.90), (0.90, 1.01)]


def collect(label, path, first=False):
    """Signed residuals on ACTIVE-fault instants, in the shared draw order."""
    su = Setup(path, args.data, args.base, args.sys, verbose=first)
    model = su.model
    model.eval()
    idx = {'val': su.idx_va, 'test': su.idx_te, 'train': su.idx_tr}[args.split]

    # Same seed for every model -> identical scenarios AND identical instants.
    rng = np.random.default_rng(args.seed)
    a_pred, a_true, typ = [], [], []
    done = 0
    while done < args.n:
        bs = min(args.batch, args.n - done)
        b = su.to_dev(su.make_batch(rng.choice(idx, bs), rng))
        with torch.no_grad():
            # the encoder alone -- alpha needs no rollout
            out = model.enc(b['feat'])
            al = model.alpha_lo + (model.alpha_hi - model.alpha_lo) * \
                torch.sigmoid(out['alpha_raw'])
        act = b['active'].cpu().numpy().astype(bool)
        ks = b['ks']
        a_pred.append(al.cpu().numpy()[act])
        a_true.append(su.meta_alpha[ks][act])
        typ.append(su.y_type[ks][act])
        done += bs
        print(f"\r  {label}: {done}/{args.n}", end='', flush=True)
    print()
    return (np.concatenate(a_pred), np.concatenate(a_true),
            np.concatenate(typ).astype(int))


def stats(res):
    """res: signed (pred - true)."""
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
    gl = ty_ != 3                                   # gen + load: two-sided
    per = {'ALL': stats(res), 'gen+load': stats(res[gl])}
    for ti_, tname in enumerate(TYPE_NAMES):
        per[tname] = stats(res[ty_ == ti_ + 1])
    results[lab] = dict(per_type=per,
                        by_alpha={}, mean_pred_line=float(ap_[ty_ == 3].mean())
                        if (ty_ == 3).any() else float('nan'))

    # severity-binned, gen+load only (line alpha is a single point at 1.0)
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
        # is the bias real, or just noise? standard error of the mean
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
