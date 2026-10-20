"""
predict9.py
===========
Run a trained 9-bus model the way it would run in a real system: on a ROLLING
window of measurements that ends "now", with nothing from the future.

Drop-in for the 9-bus folder: needs only ieee9_physics.py, pinn_model.py, the
two ieee9 .mat files, a checkpoint and (for dataset scenarios) the converted
data folder. Same modes and output as predict30.py.

1. One decision on a dataset scenario, at a chosen time:

    python predict9.py --ckpt receding_cascade/best.pt --data data_mm --scenario 42 --t 2.10

   --t defaults to fault time + 100 ms.

2. One decision on your own signal (any length; its LAST sample is "now"):

    python predict9.py --ckpt receding_cascade/best.pt --data data_mm --signal my_signal.npy --dt_in 0.001

   The signal is (T, 33) in the dataset's channel order (delta/omega per
   generator, Pm, Id/Iq, Vm/theta per bus). --dt_in is its sample spacing;
   it is decimated to the model's 10 ms, keeping the last sample. Give --data
   too, so the line table is read from the dataset (see below).

3. Streaming: slide along a scenario, re-deciding every --every seconds:

    python predict9.py --ckpt receding_cascade/best.pt --data data_mm --scenario 42 --stream

   Covers the whole 8 s run by default and writes a full-run figure, a zoom on
   the fault and a CSV. Several checkpoints can be overlaid.

   To find scenarios worth streaming:

    python predict9.py --ckpt receding_cascade/best.pt --data data_mm --list test

Notes specific to the 9-bus
---------------------------
* The line table is read from the dataset (`lines` in meta.npz). DATASET.md
  lists rows 4 and 5 in the wrong order; the stored table is authoritative,
  and line hypotheses are built from it.
* The converted 9-bus data is at 5 ms; the model reads every 2nd sample
  (its --dt_window, 10 ms). That is handled automatically.
* The mask sharpness is taken from the checkpoint as trained. (refine_dp.py,
  diagnose_dp.py and verify_true.py overwrite it with --beta_min on load, so
  their window can differ slightly from what this script and training use.)
"""

import argparse, csv, inspect, os, sys

import numpy as np
import torch

from ieee9_physics import IEEE9, HYPOTHESES
from pinn_model import FaultEncoder, RolloutController

NG, NB = 3, 9

TYPE_NAMES = ['gen', 'load', 'line']

ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
ap.add_argument('--ckpt', nargs='+', required=True,
                help='checkpoint(s); several are overlaid in --stream plots')
ap.add_argument('--base', default='ieee9_base.mat')
ap.add_argument('--sys', default='ieee9_gov_simulink_data.mat')
ap.add_argument('--data', default=None, help='converted dataset folder (data_mm)')
ap.add_argument('--scenario', type=int, default=None)
ap.add_argument('--t', type=float, default=None,
                help='decision time [s] (default: fault time + 0.1 s)')
ap.add_argument('--signal', default=None, help='.npy or .csv, shape (T, 33)')
ap.add_argument('--dt_in', type=float, default=0.01,
                help='sample spacing of --signal [s] (default 0.01)')
ap.add_argument('--stream', action='store_true')
ap.add_argument('--every', type=float, default=0.02, help='stream step [s]')
ap.add_argument('--t0', type=float, default=None,
                help='stream start (default: 0, the start of the run)')
ap.add_argument('--t1', type=float, default=None,
                help='stream end (default: the end of the run, 8 s)')
ap.add_argument('--verbose', action='store_true',
                help='print every streamed decision (default: only the ones '
                     'where the decision changes)')
ap.add_argument('--out', default='predictions', help='folder for figures/CSV')
ap.add_argument('--list', choices=['test', 'val', 'train'], default=None,
                help='list the scenarios of this split (from the first '
                     'checkpoint) that go out of limit, then exit')
args = ap.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for f_ in (args.base, args.sys) + tuple(args.ckpt):
    if not os.path.exists(f_):
        sys.exit(f"not found: {f_}")
LINES, LIMIT = None, 0.005
if args.data:
    _md = np.load(os.path.join(args.data, 'meta.npz'), allow_pickle=True)
    for key in ('lines', 'branches'):
        if key in _md:
            LINES = np.asarray(_md[key]).astype(int)
            LINES = LINES.T if LINES.shape[0] == 2 else LINES
            break
    if 'omega_limit' in _md:
        LIMIT = float(np.asarray(_md['omega_limit']).ravel()[0])
if LINES is None:
    print("WARNING: no --data given, so the line table is the module default. "
          "Pass --data to use the dataset's own table.")
    S = IEEE9(args.base, args.sys, dtype=torch.float32, device=device)
else:
    S = IEEE9(args.base, args.sys, dtype=torch.float32, device=device, lines=LINES)
LINE_TABLE = np.asarray(S.lines)


def decode9(h):
    """class index -> (kind, loc, readable name)."""
    if h == 0:
        return 'none', 0, 'no fault'
    kind, loc = HYPOTHESES[h - 1]
    if kind == 'line':
        a, b = LINE_TABLE[loc - 1]
        return kind, loc, f"line {loc} ({a}-{b})"
    if kind == 'gen':
        return kind, loc, f"gen G{loc}"
    return kind, loc, f"load bus {loc}"

# ── features: identical to train_receding.featurize ─────────────────────────
D_IDX = list(range(0, 2 * NG, 2))
W_IDX = list(range(1, 2 * NG, 2))
PM_IDX = list(range(2 * NG, 3 * NG))
ID_IDX = list(range(3 * NG, 5 * NG, 2))
IQ_IDX = list(range(3 * NG + 1, 5 * NG, 2))
N_CH = 5 * NG + 2 * NB                                       # 33
VM_IDX = list(range(5 * NG, N_CH, 2))
TH_IDX = list(range(5 * NG + 1, N_CH, 2))


def featurize(raw):
    d = raw[..., D_IDX]; w = raw[..., W_IDX]; pm = raw[..., PM_IDX]
    th = raw[..., TH_IDX]
    return np.concatenate([d[..., 1:] - d[..., :1], (w - 1.0) / 0.01, pm,
                           raw[..., ID_IDX], raw[..., IQ_IDX],
                           raw[..., VM_IDX], th[..., 1:] - th[..., :1]], axis=-1)


# ── models ───────────────────────────────────────────────────────────────────
def load_model(path):
    ck = torch.load(path, map_location=device, weights_only=False)
    A = ck['args']
    enc_kw = dict(w_max_steps=int(ck['w_steps']), dt_window=A['dt_window'],
                  lstm_layers=A.get('lstm_layers', 10), init_window=A['init_window'],
                  max_window=A['w_max'], beta0=A['beta0'])
    ctl_kw = dict(horizon=A['horizon'], dt_roll=A['dt_roll'],
                  omega_limit=LIMIT, margin=A.get('margin', 1e-4),
                  dp_head=A.get('dp_head', 'linear'), dp_hidden=A.get('dp_hidden', 128),
                  dp_fc_layers=A.get('dp_fc_layers', 2), n_refine=A.get('n_refine', 3))
    encoder = A.get('encoder', 'lstm'); dp_cell = A.get('dp_cell', 'lstm')
    # Passed only when they differ from the default, so this file also runs
    # next to the ORIGINAL pinn_model.py, which does not know them.
    if encoder != 'lstm':
        if 'encoder' not in inspect.signature(FaultEncoder.__init__).parameters:
            sys.exit(f"{path} is an '{encoder}'-encoder model, but the "
                     f"pinn_model.py in this folder has no --encoder option. "
                     f"Use the updated pinn_model.py.")
        enc_kw.update(encoder=encoder, mlp_proj=A.get('mlp_proj'))
    if dp_cell != 'lstm':
        if 'dp_cell' not in inspect.signature(RolloutController.__init__).parameters:
            sys.exit(f"{path} uses dp_cell='{dp_cell}', which the pinn_model.py "
                     f"in this folder does not support.")
        ctl_kw.update(dp_cell=dp_cell)
    enc = FaultEncoder(int(ck['n_feat']), **enc_kw)
    model = RolloutController(S, enc, **ctl_kw).to(device)
    model.load_state_dict(ck['model'])          # also restores the mask beta
    model.eval()
    n = int(ck['n_feat'])
    mu = np.asarray(ck.get('feat_mu', np.zeros(n)), np.float32)
    sd = np.asarray(ck.get('feat_sd', np.ones(n)), np.float32)
    name = os.path.basename(os.path.dirname(os.path.abspath(path))) or path
    label = f"{name} ({encoder})"
    print(f"loaded {path}: {encoder} encoder, window {float(enc.window_len().detach())*1000:.0f} ms, "
          f"epoch {ck.get('epoch', '?')}")
    return dict(model=model, mu=mu, sd=sd, W=int(ck['w_steps']), label=label,
                split=ck.get('split'), dtw=float(A['dt_window']),
                window_s=float(enc.window_len().detach()))


models = [load_model(p) for p in args.ckpt]

if args.list:
    if args.data is None or models[0]['split'] is None:
        sys.exit("--list needs --data and a checkpoint that stores its split")
    md = np.load(os.path.join(args.data, 'meta.npz'), allow_pickle=True)
    g = lambda key: md[key].ravel()
    idx = np.asarray(models[0]['split'][args.list])
    ool = g('y_ool').astype(bool)[idx]
    st = g('dp_status').astype(int)[idx] if 'dp_status' in md else np.zeros(len(idx), int)
    print(f"\n{args.list} split: {len(idx)} scenarios, {int(ool.sum())} go out of "
          f"limit.  dp_status 1 = correctable, 3 = not correctable within +/-1 pu\n")
    print(f"  {'scenario':>8s}  {'type':5s} {'loc':>4s} {'alpha':>6s} "
          f"{'fault [s]':>12s} {'peak':>8s} {'status':>6s}")
    for k in idx[ool]:
        print(f"  {k:8d}  {TYPE_NAMES[int(g('y_type')[k]) - 1]:5s} "
              f"{int(g('meta_loc')[k]):4d} {g('meta_alpha')[k]:6.2f} "
              f"{g('y_t_fault')[k]:5.2f}-{g('meta_t_clear')[k]:5.2f} "
              f"{g('meta_maxdev')[k]:8.5f} {int(g('dp_status')[k]) if 'dp_status' in md else -1:6d}")
    sys.exit(0)


# ── one decision ─────────────────────────────────────────────────────────────
def window_ending(sig, i_end, W, stride=1):
    """(W, 33) window ending at row i_end (inclusive), taking every `stride`th
    row. Rows before the start of the signal are filled with its first row."""
    idx = i_end - stride * np.arange(W - 1, -1, -1)
    return sig[np.clip(idx, 0, None)]


def stride_for(M, dt_data):
    st = int(round(M['dtw'] / dt_data))
    if st < 1 or abs(st * dt_data - M['dtw']) > 1e-9:
        sys.exit(f"{M['label']} reads every {M['dtw']*1000:.0f} ms; data at "
                 f"{dt_data*1000:g} ms does not divide that")
    return st


@torch.no_grad()
def decide_batch(M, wins, bs=128):
    """wins: (N, W, 33) raw windows, each ending at its decision instant.
    Decisions are independent, so they are run in batches (a full 8 s stream
    is ~400 of them per model)."""
    out = []
    for s0 in range(0, len(wins), bs):
        w_ = wins[s0:s0 + bs]
        feat = (featurize(w_) - M['mu']) / M['sd']
        re = w_[:, -1]; d = re[:, D_IDX]
        x_meas = np.concatenate([d - d[:, :1], re[:, W_IDX], re[:, PM_IDX]], 1)
        f = torch.tensor(feat, dtype=torch.float32, device=device)
        x = torch.tensor(x_meas, dtype=torch.float32, device=device)
        o = M['model'](f, x, hard=True, ctrl=True)
        P = o['p'].cpu().numpy()
        peak = M['model'].judged(o['w_free'])[0].cpu().numpy()
        peak_c = M['model'].judged(o['w_ctrl'])[0].cpu().numpy()
        alpha = o['alpha'].cpu().numpy(); dP = o['dP'].cpu().numpy()
        for j in range(len(w_)):
            p = P[j]
            # The SAME rule the model uses to build its network (hard=True):
            # a fault is declared if the fault hypotheses together outweigh
            # "no fault", and then the most likely fault is taken. A plain
            # argmax over all 13 classes can say "no fault" while the physics
            # is actually running a fault.
            p_fault = float(p[1:].sum())
            h = 1 + int(p[1:].argmax()) if p_fault > p[0] else 0
            kind, loc, name = decode9(h)
            out.append(dict(h=h, p=float(p[h]), p_fault=p_fault, p_all=p,
                            kind=kind, loc=loc, name=name, alpha=float(alpha[j]),
                            peak=float(peak[j]), peak_ctrl=float(peak_c[j]),
                            ool=float(peak[j]) > LIMIT, dP=dP[j]))
    return out


def decide(M, win_raw):
    return decide_batch(M, win_raw[None])[0]


def fmt(r):
    return (f"{r['name']:>16s} p={r['p']:.2f} (any fault {r['p_fault']:.2f})  "
            f"alpha={r['alpha']:.2f}  "
            f"peak={r['peak']:.5f} {'OOL' if r['ool'] else 'ok ':3s}  "
            f"dP=[{r['dP'][0]:+.3f}, {r['dP'][1]:+.3f}, {r['dP'][2]:+.3f}]  "
            f"(peak with dP {r['peak_ctrl']:.5f})")


# ── dataset access ───────────────────────────────────────────────────────────
def load_scenario(k):
    if args.data is None:
        sys.exit("--scenario needs --data")
    X = np.load(os.path.join(args.data, 'X.npy'), mmap_mode='r')
    md = np.load(os.path.join(args.data, 'meta.npz'), allow_pickle=True)
    g = lambda key: md[key].ravel()
    info = dict(sig=np.asarray(X[k], np.float32), t=g('t'),
                dt=float(np.asarray(md['dt_solver']).ravel()[0]),
                typ=TYPE_NAMES[int(g('y_type')[k]) - 1], loc=int(g('meta_loc')[k]),
                alpha=float(g('meta_alpha')[k]), tf=float(g('y_t_fault')[k]),
                tc=float(g('meta_t_clear')[k]), ool=bool(g('y_ool')[k]),
                dp=np.asarray(md['y_delta_p'][k][:3]) if 'y_delta_p' in md else None,
                status=int(g('dp_status')[k]) if 'dp_status' in md else None)
    nm = decode9(HYPOTHESES.index((info['typ'], info['loc'])) + 1)[2]
    print(f"\nscenario {k}: {nm}, alpha "
          f"{info['alpha']:.2f}, {info['tf']:.2f}-{info['tc']:.2f} s | "
          f"out of limit in data: {info['ool']}"
          + (f" | dp_status {info['status']}, y_delta_p = "
             f"[{info['dp'][0]:+.3f}, {info['dp'][1]:+.3f}, {info['dp'][2]:+.3f}]"
             if info['dp'] is not None else ""))
    return info


def truth_class(info):
    return HYPOTHESES.index((info['typ'], info['loc'])) + 1


os.makedirs(args.out, exist_ok=True)

# ── mode 2: own signal ───────────────────────────────────────────────────────
if args.signal:
    sig = (np.load(args.signal) if args.signal.endswith('.npy')
           else np.loadtxt(args.signal, delimiter=','))
    sig = np.asarray(sig, np.float32)
    if sig.ndim != 2 or sig.shape[1] != N_CH:
        sys.exit(f"signal must be (T, {N_CH}); got {sig.shape}")
    dur = len(sig) * args.dt_in
    print(f"\nsignal: {len(sig)} samples at {args.dt_in*1000:g} ms = "
          f"{dur*1000:.0f} ms of history")
    for M in models:
        st = stride_for(M, args.dt_in)
        if dur < M['window_s'] + 0.05:
            print(f"  WARNING: {M['label']} uses ~{M['window_s']*1000:.0f} ms; the "
                  f"signal has {dur*1000:.0f} ms. The oldest part is padded.")
        r = decide(M, window_ending(sig, len(sig) - 1, M['W'], st))
        print(f"  {M['label']:>28s}: {fmt(r)}")
    sys.exit(0)

if args.scenario is None:
    sys.exit("give --scenario (with --data) or --signal")
info = load_scenario(args.scenario)
sig, t_grid, DTD = info['sig'], info['t'], info['dt']
k_true = truth_class(info)

# ── mode 1: single decision ──────────────────────────────────────────────────
if not args.stream:
    tk = args.t if args.t is not None else info['tf'] + 0.1
    i = int(round(tk / DTD))
    active = info['tf'] <= tk < info['tc']
    print(f"decision at t = {tk:.3f} s ({(tk - info['tf'])*1000:+.0f} ms from onset; "
          f"fault {'ACTIVE' if active else 'not active'}), using data up to t only")
    for M in models:
        r = decide(M, window_ending(sig, i, M['W'], stride_for(M, DTD)))
        ok = (r['h'] == (k_true if active else 0))
        print(f"  {M['label']:>28s}: {fmt(r)}  "
              f"[identification {'correct' if ok else 'WRONG'}]")
    fut = np.abs(sig[i:i + int(round(1.5 / DTD)) + 1][:, W_IDX] - 1).max()
    print(f"  actual peak |w-1| over the next 1.5 s (measured, no dP): {fut:.5f}")
    sys.exit(0)

# ── mode 3: streaming ────────────────────────────────────────────────────────
t0 = args.t0 if args.t0 is not None else float(t_grid[0])
t1 = args.t1 if args.t1 is not None else float(t_grid[-1])
times = np.arange(t0, t1 + 1e-9, args.every)
dev = np.abs(sig[:, W_IDX] - 1).max(1)
cross = np.where((dev > LIMIT) & (t_grid >= info['tf']))[0]
t_cross = float(t_grid[cross[0]]) if len(cross) else None

print(f"streaming {len(times)} decisions, {t0:.2f}-{t1:.2f} s every "
      f"{args.every*1000:.0f} ms"
      + ("" if args.verbose else "  (showing only decisions that change)") + "\n")
results = {}
for M in models:
    idx_end = np.clip(np.round(times / DTD).astype(int), 0, len(sig) - 1)
    st = stride_for(M, DTD)
    wins = np.stack([window_ending(sig, i, M['W'], st) for i in idx_end])
    rows = decide_batch(M, wins)
    print(f"── {M['label']}")
    print(f"  {'t [s]':>7s} {'age':>7s}  decision")
    prev = None
    for tk, r in zip(times, rows):
        r['t'] = float(tk); r['age'] = float(tk - info['tf'])
        active = info['tf'] <= tk < info['tc']
        r['phase'] = ('pre-fault' if tk < info['tf'] else
                      'fault' if active else 'cleared')
        r['correct'] = r['h'] == (k_true if active else 0)
        r['p_true'] = float(r['p_all'][k_true if active else 0])
        key = (r['h'], r['ool'], r['phase'])
        if args.verbose or key != prev or tk == times[-1]:
            print(f"  {tk:7.3f} {r['age']*1000:+6.0f}ms  {fmt(r)}  [{r['phase']}]"
                  f"{'' if r['correct'] else '  <- wrong fault'}")
        prev = key
    results[M['label']] = rows
    print()

print("=" * 72)
print(f"SUMMARY  scenario {args.scenario}: {decode9(k_true)[2]}, "
      f"alpha {info['alpha']:.2f}, onset {info['tf']:.3f} s")
print(f"  omega leaves the band in the MEASURED data at: "
      + (f"{t_cross:.3f} s ({(t_cross - info['tf'])*1000:.0f} ms after onset)"
         if t_cross is not None else "never (within the run)"))
for lab, rows in results.items():
    flags = [r for r in rows if r['ool'] and r['age'] >= 0]
    print(f"  {lab}:")
    pre = [r for r in rows if r['age'] < 0]
    if pre:
        n_fa = sum(r['ool'] for r in pre)
        print(f"    before onset: {n_fa} of {len(pre)} decisions predicted "
              f"out-of-limit" + ("  <- false alarms" if n_fa else " (correct)"))
    if not flags:
        print(f"    never predicted out-of-limit in this range")
        continue
    f0 = flags[0]
    lead = (t_cross - f0['t']) if t_cross is not None else None
    print(f"    first OOL prediction at {f0['t']:.3f} s "
          f"({f0['age']*1000:.0f} ms after onset)"
          + (f", {lead*1000:.0f} ms BEFORE omega actually leaves the band"
             if lead is not None and lead > 0 else
             f", {-lead*1000:.0f} ms after omega already left the band"
             if lead is not None else ", but omega never leaves the band (false alarm)"))
    print(f"    identified as {f0['name']} (p={f0['p']:.2f}, "
          f"{'correct' if f0['correct'] else 'WRONG'}), alpha {f0['alpha']:.2f} "
          f"(true {info['alpha']:.2f})")
    print(f"    dP at that moment: G1 {f0['dP'][0]:+.3f}, G2 {f0['dP'][1]:+.3f}, "
          f"G3 {f0['dP'][2]:+.3f} pu"
          f" -> predicted peak {f0['peak']:.5f} without, {f0['peak_ctrl']:.5f} with")
    post = [r for r in rows if r['phase'] == 'cleared']
    if post:
        n_nf = sum(r['h'] == 0 for r in post)
        print(f"    after clearing: 'no fault' in {n_nf} of {len(post)} decisions "
              f"({100*n_nf/len(post):.0f}%)")
    first_ok = next((r for r in rows if r['age'] >= 0 and r['correct']), None)
    if first_ok:
        print(f"    fault first identified correctly at {first_ok['age']*1000:.0f} ms "
              f"after onset")

# CSV
tag = f"sc{args.scenario}"
with open(os.path.join(args.out, f"stream_{tag}.csv"), 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['model', 't', 'age_ms', 'hypothesis', 'p', 'correct', 'alpha',
                'peak_pred', 'ool_pred', 'dP_G1', 'dP_G2', 'dP_G3', 'peak_with_dP'])
    for lab, rows in results.items():
        for r in rows:
            w.writerow([lab, f"{r['t']:.3f}", f"{r['age']*1000:.0f}",
                        r['name'], f"{r['p']:.3f}", int(r['correct']),
                        f"{r['alpha']:.3f}", f"{r['peak']:.5f}", int(r['ool']),
                        f"{r['dP'][0]:.4f}", f"{r['dP'][1]:.4f}", f"{r['dP'][2]:.4f}",
                        f"{r['peak_ctrl']:.5f}"])

# figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def make_fig(lo_t, hi_t, suffix='', title_extra=''):
    fig, ax = plt.subplots(4, 1, figsize=(11, 10), sharex=True,
                           gridspec_kw={'height_ratios': [2.2, 1.4, 1.4, 1.2]})
    m = (t_grid >= lo_t) & (t_grid <= hi_t)
    cols = ['#1f77b4', '#d62728', '#2ca02c']
    for g in range(NG):
        ax[0].plot(t_grid[m], sig[m][:, W_IDX[g]], color=cols[g], lw=1.4,
                   label=f"G{g+1}")
    for a_ in ax:
        a_.axvspan(info['tf'], info['tc'], color='#ffe9b0', alpha=0.5, zorder=0)
        if t_cross is not None:
            a_.axvline(t_cross, color='k', ls=':', lw=1)
    for s_ in (1, -1):
        ax[0].axhline(1 + s_ * LIMIT, color='k', ls='--', lw=0.8)
    ax[0].set_ylabel('measured ω [pu]')
    ax[0].legend(ncol=3, fontsize=8, loc='lower left')
    ax[0].set_title(f"scenario {args.scenario}: {decode9(k_true)[2]}, "
                    f"α = {info['alpha']:.2f}   |   decisions every "
                    f"{args.every*1000:.0f} ms, each using only data up to that "
                    f"moment" + title_extra, fontsize=10)

    mstyle = ['-', '--', ':', '-.']
    mcol = ['#534AB7', '#D85A30', '#0F6E56', '#854F0B']
    for j, (lab, rows_all) in enumerate(results.items()):
        rows = [r for r in rows_all if lo_t <= r['t'] <= hi_t]
        tt = [r['t'] for r in rows]
        c, ls = mcol[j % 4], mstyle[j % 4]
        ax[1].plot(tt, [r['peak'] for r in rows], color=c, ls=ls, marker='.', ms=3,
                   label=f"{lab}: predicted peak, no dP")
        ax[1].plot(tt, [r['peak_ctrl'] for r in rows], color=c, ls=ls, lw=0.8,
                   alpha=0.6, label=f"{lab}: predicted peak, with dP")
        # one colour per model; generators told apart by line weight
        for g, (lw, al) in enumerate([(1.8, 1.0), (1.1, 0.7), (0.7, 0.5)]):
            ax[2].plot(tt, [r['dP'][g] for r in rows], color=c, ls=ls, lw=lw,
                       alpha=al, label=f"{lab}: G{g+1}")
        ax[3].plot(tt, [r['p_true'] for r in rows], color=c, ls=ls, marker='.', ms=3,
                   label=lab)
        flags = [r for r in rows_all if r['ool'] and r['age'] >= 0]
        if flags and lo_t <= flags[0]['t'] <= hi_t:
            for a_ in ax:
                a_.axvline(flags[0]['t'], color=c, ls=ls, lw=1.2, alpha=0.8)
    ax[1].axhline(LIMIT, color='k', ls='--', lw=0.8)
    ax[1].set_ylabel('predicted peak\n|ω−1| next 1.5 s')
    ax[1].legend(fontsize=7, loc='upper left')
    ax[2].axhline(0, color='0.5', lw=0.6)
    ax[2].set_ylabel('ΔP [pu]')
    ax[2].legend(fontsize=7, loc='upper left', ncol=3)
    ax[3].set_ylim(-0.05, 1.05)
    ax[3].set_ylabel('p(correct\nhypothesis)')
    ax[3].set_xlabel('decision time [s]   (shaded: fault active; dotted: ω leaves '
                     'the band in the data; coloured lines: first OOL prediction)')
    ax[3].legend(fontsize=7, loc='lower right')
    ax[0].set_xlim(lo_t, hi_t)
    fig.tight_layout()
    p_ = os.path.join(args.out, f"stream_{tag}{suffix}.png")
    try:
        fig.savefig(p_, dpi=150)
    except OSError:
        # On Windows a PNG that is open in an image viewer (or being synced
        # by OneDrive) cannot be overwritten; save under a new name instead
        import time
        alt = p_.replace('.png', time.strftime('_%H%M%S.png'))
        print(f"  could not overwrite {p_} (is it open in an image viewer?) "
              f"-> saving as {alt}")
        fig.savefig(alt, dpi=150); p_ = alt
    plt.close(fig)
    return p_


# full run (0-8 s) and a zoom on the fault
p_full = make_fig(float(t_grid[0]), float(t_grid[-1]), '', '  (full run)')
z0 = max(float(t_grid[0]), info['tf'] - 0.3)
z1 = min(float(t_grid[-1]), min(info['tc'], info['tf'] + 1.0) + 0.3)
p_zoom = make_fig(z0, z1, '_zoom', '  (zoom on the fault)')
print(f"\n-> {p_full}\n-> {p_zoom}\n-> {os.path.join(args.out, f'stream_{tag}.csv')}")

