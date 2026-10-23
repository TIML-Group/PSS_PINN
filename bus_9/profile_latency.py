"""
profile_latency.py -- 9-bus forward-pass latency attribution.

Times every section of the decision forward pass with CUDA events, counts
kernel launches with the torch profiler, and reports static FLOP counts --
three views of the same forward pass in one run:

    python profile_latency.py --ckpt receding_cascade/refined_common/best.pt
    python profile_latency.py --ckpt receding_cascade/refined/best.pt --batch 1 256

Sections mirror pinn_model.RolloutController.forward (cascade head, ctrl=True):

  enc      enc(win): window featurize + LSTM blocks + trunk + heads
  net      network(p, alpha, hard): candidate Y matrix + kron_reduce + pe_consts
  wfree    free rollout (N_ROLL x rk4_real), no correction
  refine k excursion_features + dp_cell + dp_fc + controlled rollout, per step
  judged   judged() + violation() envelope summaries

For each section and batch size we report:
  ms           CUDA-event median wall time
  launches     cudaLaunchKernel count per call (torch.profiler)
  MFLOPs       static float-op count (batch-scaled)
  ms/launch    wall per kernel launch -- the launch-bound indicator
"""

import argparse, json, math, os, sys, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
# (Windows console default cp1252 chokes on box-drawing chars in the report)
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='receding_cascade/refined_common/best.pt')
ap.add_argument('--data', default='data_mm')
ap.add_argument('--base', default='ieee9_base.mat')
ap.add_argument('--sysfile', default='ieee9_gov_simulink_data.mat')
ap.add_argument('--batch', type=int, nargs='+', default=[1, 256])
ap.add_argument('--reps', type=int, default=100, help='timed reps per section')
ap.add_argument('--prof_reps', type=int, default=10, help='profiled reps per section')
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--out', default='profile_latency_out')
args = ap.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sync = lambda: (torch.cuda.synchronize() if device.type == 'cuda' else None)

# ------------------------------------------------------------------ loading --
# (mirrors compare_models.py 9-bus branch)
_md = np.load(os.path.join(args.data, 'meta.npz'))
_X = np.load(os.path.join(args.data, 'X.npy'), mmap_mode='r')
_lines = (_md['lines'] if 'lines' in _md.files else _md['branches']).astype(int)
from ieee9_physics import IEEE9
from pinn_model import FaultEncoder, RolloutController
S = IEEE9(args.base, args.sysfile, dtype=torch.float32, device=device, lines=_lines)

DT = float(_md['dt_solver'])
D_IDX, W_IDX, PM_IDX = [0, 2, 4], [1, 3, 5], [6, 7, 8]
ID_IDX, IQ_IDX = [9, 11, 13], [10, 12, 14]
VM_IDX = list(range(15, 33, 2)); TH_IDX = list(range(16, 33, 2))
FUT_IDX = [1, 3, 5, 6, 7, 8]

ck = torch.load(args.ckpt, map_location=device, weights_only=False)
A = ck['args']
W_STEPS = int(ck['w_steps'])
W_STRIDE = int(round(A['dt_window'] / DT))
ROLL_STRIDE = int(round(A['dt_roll'] / DT))
N_ROLL = int(round(A['horizon'] / A['dt_roll']))
GRACE_STEPS = int(round(0.2 / A['dt_roll']))

enc = FaultEncoder(int(ck['n_feat']), w_max_steps=W_STEPS,
                   dt_window=A['dt_window'],
                   lstm_layers=A.get('lstm_layers', 10),
                   init_window=A['init_window'], max_window=A['w_max'],
                   beta0=A['beta0'], encoder=A.get('encoder', 'lstm'),
                   mlp_proj=A.get('mlp_proj'))
model = RolloutController(
    S, enc, horizon=A['horizon'], dt_roll=A['dt_roll'],
    omega_limit=float(_md['omega_limit']), margin=A.get('margin', 1e-4),
    dp_head=A.get('dp_head', 'linear'), dp_hidden=A.get('dp_hidden', 128),
    dp_fc_layers=A.get('dp_fc_layers', 2), dp_cell=A.get('dp_cell', 'lstm'),
    alpha_lo=A.get('alpha_lo', 0.10), alpha_hi=A.get('alpha_hi', 1.00)).to(device)
model.load_state_dict(ck['model'])
model.eval()

# ------------------------------------------------------------------- batch --
KS_TEST = np.asarray(ck['split']['test'])
rng0 = np.random.default_rng(args.seed)

def make_batch(bs, rng):
    ks = rng0.permutation(KS_TEST)[:bs]
    # window + horizon must exist in the trace:
    lo = (W_STEPS - 1) * W_STRIDE * DT + 0.1
    hi = 8.0 - N_ROLL * ROLL_STRIDE * DT - 0.1
    tk = rng.uniform(lo, hi, size=bs)
    i_end = np.round(tk / DT).astype(int)
    win = np.empty((bs, W_STEPS, _X.shape[2]), np.float32)
    for j, (k, ie) in enumerate(zip(ks, i_end)):
        i0 = ie - (W_STEPS - 1) * W_STRIDE
        blk = np.asarray(_X[k, i0:ie + N_ROLL * ROLL_STRIDE + 1])
        o = ie - i0
        win[j] = blk[0:o + 1:W_STRIDE]
    d = win[..., D_IDX]; w = win[..., W_IDX]; pm = win[..., PM_IDX]
    th = win[..., TH_IDX]
    feat = np.concatenate([d[..., 1:] - d[..., :1], (w - 1.0) / 0.01, pm,
                           win[..., ID_IDX], win[..., IQ_IDX],
                           win[..., VM_IDX], th[..., 1:] - th[..., :1]], -1)
    re = win[:, -1, :]; dd = re[:, D_IDX]
    x_meas = np.concatenate([np.stack([np.zeros(bs), dd[:, 1] - dd[:, 0],
                                       dd[:, 2] - dd[:, 0]], 1),
                             re[:, W_IDX], re[:, PM_IDX]], 1)
    tf, tc = _md['y_t_fault'].ravel()[ks], _md['meta_t_clear'].ravel()[ks]
    active = (tk >= tf) & (tk < tc)
    t_change = np.where(active, tc, np.where(tk < tf, tf, np.inf))
    valid_t = np.clip(t_change - tk, 0.0, A['horizon'])
    step_t = np.arange(N_ROLL + 1) * A['dt_roll']
    vmask = (step_t[None, :] <= valid_t[:, None]).astype(np.float32)
    T = lambda a, dt=None: torch.tensor(a, dtype=dt).to(device)
    return dict(feat=T(feat), x_meas=T(x_meas, torch.float32), vmask=T(vmask),
                active=T(active))

# --------------------------------------------------------- FLOP accounting --
def flop_table(bs):
    fc = lambda i, o: 2 * i * o
    g = 4 * 128                                    # gates x hidden
    lstm_blk = 2 * g * 128 * 2                     # ih + hh, x2 for MAC
    enc = (fc(31, 128) + 3 * lstm_blk + 4 * fc(128, 128)
           + fc(128, 13) + fc(128, 1) + fc(128, 9) + fc(128, 3))
    rms = 2 * 128 * 2 * 3                          # 2 norms
    rk4_step = 4 * 140                             # 4 rhs_real evals, ~140 flops each (B=1)
    wfree = N_ROLL * rk4_step
    dp_step = (2 * 4 * 128 * 164 + 2 * 4 * 128 * 128   # cell ih + hh
               + fc(128, 128) + fc(128, 3) + 2 * 128 * 14)  # fc + excursion
    n_ref = getattr(model, 'n_refine', 3)
    refine = n_ref * (dp_step + wfree)
    netk = fc(128, 128) + 2 * 3 * 3 * 3 * 2      # kron proj + M/N outer products
    return {'enc': enc + rms, 'net': netk, 'w_free': wfree,
            f'refine x{n_ref}': refine, 'judged': 0}

# ------------------------------------------------------------- timing core --
def time_section(fn):
    for _ in range(5):
        fn()
    sync()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
          for _ in range(args.reps)] if device.type == 'cuda' else None
    times = []
    if ev:
        for a, b in ev:
            a.record(); fn(); b.record()
        sync()
        times = [a.elapsed_time(b) for a, b in ev]
    else:
        for _ in range(args.reps):
            t = time.perf_counter(); fn(); times.append((time.perf_counter() - t) * 1e3)
    return float(np.median(times)), float(np.percentile(times, 90))

def profile_launches(fn):
    from torch.profiler import profile, ProfilerActivity
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == 'cuda' else [])
    fn()
    with profile(activities=acts) as prof:
        for _ in range(args.prof_reps):
            fn()
        sync()
    evs = prof.key_averages()
    n_l = sum(getattr(e, 'count', 0) for e in evs if 'cudaLaunchKernel' in e.key)
    n_e = sum(getattr(e, 'count', 0) for e in evs if e.key.startswith('aten::') or
              ('kernel' in e.key.lower() and 'Launch' not in e.key))
    return n_l / args.prof_reps, sum(e.self_device_time_total for e in evs) / args.prof_reps / 1e3

# ------------------------------------------------------------ build sections --
model.n_refine = getattr(model, 'n_refine', 3)
print(f"ckpt: {args.ckpt}   encoder: {A.get('encoder','lstm')}   dp_head: {A.get('dp_head','linear')}"
      f"   n_refine: {model.n_refine}   N_ROLL: {N_ROLL}   device: {device.type}\n")

results = {}
for bs in args.batch:
    b = make_batch(bs, np.random.default_rng(args.seed))
    with torch.no_grad():
        o_enc = model.enc(b['feat'])
        p = torch.softmax(o_enc['hyp_logits'], dim=1)
        alpha = model.alpha_lo + (model.alpha_hi - model.alpha_lo) * \
                torch.sigmoid(o_enc['alpha_raw'])
        Yred, dPm0 = model.network(p, alpha, hard=True)
        x0 = b['x_meas'] + 0.1 * o_enc['x0_corr']
        Pset = model.S.Pm0.unsqueeze(0) + dPm0

        secs = {}
        secs['enc'] = lambda: model.enc(b['feat'])
        secs['net'] = lambda: model.network(p, alpha, hard=True)
        secs['w_free'] = lambda: model.rollout(x0, Yred, Pset, dP=None)

        def make_refine():
            def fn():
                ctx = torch.cat([o_enc['z'], p, alpha.unsqueeze(1), x0], dim=1)
                B = ctx.shape[0]
                h = ctx.new_zeros(B, model.dp_hidden)
                c = ctx.new_zeros(B, model.dp_hidden)
                raw = ctx.new_zeros(B, 3)
                w_cur = model.rollout(x0, Yred, Pset, dP=model.dp_max * torch.tanh(raw))
                x0d, Yd, Pd = x0, Yred, Pset
                for _ in range(model.n_refine):
                    exc = model.excursion_features(w_cur, b['vmask'])
                    inp = torch.cat([ctx, model.dp_max * torch.tanh(raw), exc], dim=1)
                    h, c = model.dp_cell(inp, (h, c))
                    raw = raw + model.dp_fc(h)
                    dP = model.dp_max * torch.tanh(raw)
                    w_cur = model.rollout(x0d, Yd, Pd, dP=dP)
                return model.judged(w_cur, b['vmask'])[0].mean()
            return fn
        secs['cascade_full'] = make_refine()

        tab = {}
        for name, fn in secs.items():
            ms, p90 = time_section(fn)
            launches, dev_us = profile_launches(fn)
            tab[name] = dict(ms=ms, p90=p90, launches=launches, gpu_us=dev_us)
        results[bs] = dict(sections=tab, flops=flop_table(bs))

# ------------------------------------------------------------------ report --
fl = lambda i, o: 2 * i * o
print(f"{'section':14s} {'batch':>6s} {'ms median':>10s} {'p90 ms':>8s} "
      f"{'launches':>8s} {'µs GPU busy':>11s} {'MFLOPs':>8s} {'µs/launch':>9s}")
for bs, R in results.items():
    for name, t in R['sections'].items():
        mf = R['flops'].get(name, 0) / 1e6
        per = (t['ms'] * 1000 / t['launches']) if t['launches'] else float('nan')
        print(f"{name:14s} {bs:>6d} {t['ms']:>10.3f} {t['p90']:>8.3f} "
              f"{t['launches']:>8.1f} {t['gpu_us']:>11.1f} {mf:>8.2f} {per:>9.1f}")
    # full decision = sum of sections (they are the disjoint phases of forward)
    tot = sum(t['ms'] for t in R['sections'].values())
    print(f"{'SUM(sections)':14s} {bs:>6d} {tot:>10.3f}")
    def _whole():
        with torch.no_grad():
            model(b['feat'], b['x_meas'], hard=True, ctrl=True, vmask=b['vmask'])
    whole_ms, _ = time_section(_whole)
    print(f"{'full forward':14s} {bs:>6d} {whole_ms:>10.3f}   (end-to-end measured)\n")


# -- per-call sync attribution (launch overhead is invisible to pure event timing) --
tot = {}
_sync_t = lambda fn: (sync(), time.perf_counter(), fn(), sync(),
                      (time.perf_counter() - _t0) * 1e3)[4] if False else None
def timed_attribution(bs):
    b = make_batch(bs, np.random.default_rng(args.seed))
    with torch.no_grad():
        o_enc = model.enc(b['feat'])
        p = torch.softmax(o_enc['hyp_logits'], dim=1)
        alpha = model.alpha_lo + (model.alpha_hi - model.alpha_lo) * \
                torch.sigmoid(o_enc['alpha_raw'])
        Yred, dPm0 = model.network(p, alpha, hard=True)
        x0 = b['x_meas'] + 0.1 * o_enc['x0_corr']
        Pset = model.S.Pm0.unsqueeze(0) + dPm0
        def wrap(obj, name, attr=None):  # unused
            if attr is None:
                orig = obj
                def w(*a, **k):
                    sync(); t = time.perf_counter()
                    r = orig(*a, **k)
                    sync(); tot[name] = tot.get(name, 0) + (time.perf_counter()-t)*1e3
                    return r
                return w
            fwd = type(obj).forward
            def w(self, *a, **k):
                sync(); t = time.perf_counter()
                r = fwd(self, *a, **k)
                sync(); tot[name] = tot.get(name, 0) + (time.perf_counter()-t)*1e3
                return r
            type(obj).forward = w
            return w
        ro = model.rollout
        calls = {'n': 0}
        def roll_cnt(*a, **k):
            calls['n'] += 1
            sync(); t = time.perf_counter()
            r = ro(*a, **k)
            sync(); tot['rollout'] = tot.get('rollout', 0) + (time.perf_counter()-t)*1e3
            return r
        model.rollout = roll_cnt
        enc_f = type(model.enc).forward
        def enc_w(self, *a, **k):
            sync(); t = time.perf_counter()
            r = enc_f(self, *a, **k)
            sync(); tot['enc'] = tot.get('enc', 0) + (time.perf_counter()-t)*1e3
            return r
        type(model.enc).forward = enc_w
        type(model).network = wrap(model, 'net', None) if False else type(model).network
        net_f = type(model).network
        def net_w(self, *a, **k):
            sync(); t = time.perf_counter()
            r = net_f(self, *a, **k)
            sync(); tot['net'] = tot.get('net', 0) + (time.perf_counter()-t)*1e3
            return r
        type(model).network = net_w
        for _ in range(3):
            model(b['feat'], b['x_meas'], hard=True, ctrl=True, vmask=b['vmask'])
        sync(); tot.clear()
        t0 = time.perf_counter()
        for _ in range(args.reps):
            model(b['feat'], b['x_meas'], hard=True, ctrl=True, vmask=b['vmask'])
        sync(); wall = (time.perf_counter()-t0)/args.reps*1000
        model.rollout = ro
        return wall, calls['n'] // (args.reps + 3)

for bs in args.batch:
    wall, n_roll = timed_attribution(bs)
    print(f"\n-- synced per-phase attribution  @batch {bs}  ({n_roll} rollout calls/decision) --")
    for k, v in sorted(tot.items(), key=lambda kv: -kv[1]):
        print(f"  {k:10s} {v/args.reps:9.2f} ms   ({v/args.reps/wall*100:5.1f} %)")
    print(f"  {'TOTAL':10s} {wall:9.2f} ms   unaccounted "
          f"{wall - sum(tot.values())/args.reps:.1f} ms")
    results.setdefault(bs, {})['sync_attribution'] = dict(
        wall_ms=wall, rollout_calls_per_decision=n_roll,
        **{k: v/args.reps for k, v in tot.items()})

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, 'profile_latency.json'), 'w') as f:
    json.dump(results, f, indent=1, default=str)
print(f"-> {args.out}/profile_latency.json")