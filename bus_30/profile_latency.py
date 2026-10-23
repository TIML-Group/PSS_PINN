"""
profile_latency.py -- 30-bus forward-pass latency attribution.

Same protocol as the 9-bus profiler (bus_9/profile_latency.py): CUDA-event
medians + p90 per section, kernel-launch counts, static FLOPs, synced
end-to-end attribution, and (with --graph) whole-decision CUDA-graph capture
and replay timing vs eager.

Sections mirror pinn_model30.RolloutController.forward (cascade head, ctrl):

  enc      enc(win): window featurize + LSTM blocks + trunk + heads
  net      network(p, alpha, hard): candidate Y matrices + kron_reduce + pe_consts
  wfree    free rollout (N_ROLL x rk4_real), no correction
  cascade  excursion_features + dp_cell + dp_fc + controlled rollout, per step
  judged   judged() + violation() envelope summaries

Run (from bus_30/, needs data30_mm):
  python profile_latency.py --ckpt case30_ctrl/refined_common/best.pt \
      --data data30_mm --batch 1 256
  python profile_latency.py ... --graph   (adds the graph replay column)
"""

import argparse, json, math, os, sys, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='case30_ctrl/refined_common/best.pt')
ap.add_argument('--data', default='data30_mm')
ap.add_argument('--base', default='case30_base.mat')
ap.add_argument('--sysfile', default='case30_simulink_data.mat')
ap.add_argument('--batch', type=int, nargs='+', default=[1, 256])
ap.add_argument('--reps', type=int, default=100, help='timed reps per section')
ap.add_argument('--prof_reps', type=int, default=10, help='profiled reps per section')
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--graph', action='store_true', default=False,
                help='Capture the whole decision forward into ONE CUDA graph per'
                     ' batch size and re-time it (eager numbers stay measured so'
                     ' the verdict table is eager vs graph side by side).')
ap.add_argument('--n_roll', type=int, default=None,
                help='override the rollout step count (horizon = n_roll x dt_roll)')
ap.add_argument('--dt_roll', type=float, default=None,
                help='override the rollout physics step (horizon kept)')
ap.add_argument('--n_refine', type=int, default=None,
                help='override the cascade refinement count')
ap.add_argument('--out', default='profile_latency_out')
args = ap.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sync = lambda: (torch.cuda.synchronize() if device.type == 'cuda' else None)

# ------------------------------------------------------------------ loading --
# (mirrors compare_models.py 30-bus branch; Setup applies the sweep overrides)
import case30_common as C30
from case30_physics import NG, N_STATE
su = C30.Setup(args.ckpt, args.data, args.base, args.sysfile, device=device,
               verbose=False, dt_roll=args.dt_roll, n_roll=args.n_roll,
               n_refine=args.n_refine)
model, A, ck = su.model, su.A, su.ck
_S = su.S
DT = su.DT
W_STEPS, W_STRIDE = su.W_STEPS, su.W_STRIDE
N_ROLL = su.N_ROLL
DT_ROLL_EFF = su.DT_ROLL_EFF
N_REFINE = su.N_REFINE
model.eval()

# ------------------------------------------------------------------- batch --
def make_batch(bs, rng):
    ks = rng.permutation(np.asarray(ck['split']['test']))[:bs]
    # fixed instants so a re-run times the same tensors; tk uniform in the
    # trace, kept clear of the window/horizon edges
    lo = A['w_max'] + 0.1
    hi = float(su.t_grid[-1]) - A['horizon'] - 0.1
    tk = rng.uniform(lo, hi, size=bs)
    b = su.make_batch(ks, np.random.default_rng(0), tk_fixed=tk)
    return su.to_dev(b)

# --------------------------------------------------------- FLOP accounting --
def flop_table(bs):
    fc = lambda i, o: 2 * i * o
    w = int(A['dp_hidden'])
    f_in = int(ck['n_feat'])
    H = 128 if A.get('dp_head') != 'linear' else w
    # the trunk hidden is A['dp_hidden'] for the 30-bus ctrl build
    enc = (fc(f_in, w) + 3 * w * w * 4 + 4 * fc(w, w)
           + fc(w, len(su.HYPOTHESES)) + fc(w, 1) + fc(w, 2 * NG) + fc(w, 2))
    trunk = 4 * fc(w, w)
    rms = 2 * w * 2 * 3
    rk4_step = 4 * (NG * (NG * 2 + 18))
    wfree = N_ROLL * rk4_step * bs and N_ROLL * rk4_step
    dp_step = (2 * 4 * w * (w + 14) + 2 * 4 * w * w
               + fc(w, w) + fc(w, 2) + 2 * w * 14)
    refine = N_REFINE * (dp_step + wfree)
    netk = fc(w, w) + 2 * NG * NG * NG * 2
    return {'enc': enc + trunk + rms, 'net': netk, 'w_free': wfree,
            f'refine x{N_REFINE}': refine, 'judged': 0}

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
print(f"ckpt: {args.ckpt}   encoder: {A.get('encoder','lstm')}   dp_head: "
      f"{A.get('dp_head','linear')}   n_refine: {N_REFINE}   N_ROLL: {N_ROLL}   "
      f"dt_roll_eff: {DT_ROLL_EFF*1000:.0f} ms   device: {device.type}\n")

results = {}
if args.graph and os.path.exists(os.path.join(args.out, 'profile_latency.json')):
    results = {int(k): v for k, v in
               json.load(open(os.path.join(args.out, 'profile_latency.json'))).items()}
    args.batch = [int(x) for x in args.batch]
    print(f"(loaded eager results from {args.out}/profile_latency.json;"
          f" graph pass only)")
for bs in args.batch:
    if bs in results and 'sections' in results.get(bs, {}):
        continue
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
                raw = ctx.new_zeros(B, 2)
                w_cur = model.rollout(x0, Yred, Pset, dP=model.dp_max * torch.tanh(raw))
                for _ in range(model.n_refine):
                    exc = model.excursion_features(w_cur, b['vmask'])
                    inp = torch.cat([ctx, model.dp_max * torch.tanh(raw), exc], dim=1)
                    h, c = model.dp_cell(inp, (h, c))
                    raw = raw + model.dp_fc(h)
                    dP = model.dp_max * torch.tanh(raw)
                    w_cur = model.rollout(x0, Yred, Pset, dP=dP)
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
print(f"{'section':14s} {'batch':>6s} {'ms median':>10s} {'p90 ms':>8s} "
      f"{'launches':>8s} {'us GPU busy':>11s} {'MFLOPs':>8s} {'us/launch':>9s}")
for bs, R in results.items():
    for name, t in R['sections'].items():
        mf = R['flops'].get(name, 0) / 1e6
        per = (t['ms'] * 1000 / t['launches']) if t['launches'] else float('nan')
        print(f"{name:14s} {bs:>6d} {t['ms']:>10.3f} {t['p90']:>8.3f} "
              f"{t['launches']:>8.1f} {t['gpu_us']:>11.1f} {mf:>8.2f} {per:>9.1f}")
    tot = sum(t['ms'] for t in R['sections'].values())
    print(f"{'SUM(sections)':14s} {bs:>6d} {tot:>10.3f}")
    if 'sync_attribution' in R:
        whole_ms = R['sync_attribution']['wall_ms']
        print(f"{'full forward':14s} {bs:>6d} {whole_ms:>10.3f}   (end-to-end, synced)\n")
    else:
        def _whole(bs=bs):
            bb = make_batch(bs, np.random.default_rng(args.seed))
            with torch.no_grad():
                model(bb['feat'], bb['x_meas'], hard=True, ctrl=True,
                      vmask=bb['vmask'])
        whole_ms, _ = time_section(_whole)
        print(f"{'full forward':14s} {bs:>6d} {whole_ms:>10.3f}   (end-to-end measured)\n")

# -- per-call sync attribution (launch overhead is invisible to event timing) --
for bs in args.batch:
    if 'sync_attribution' in results.get(bs, {}):
        continue
    b = make_batch(bs, np.random.default_rng(args.seed))
    with torch.no_grad():
        tot = {}
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
        type(model.enc).forward = enc_f
        type(model).network = net_f
    print(f"\n-- synced per-phase attribution  @batch {bs}  "
          f"({calls['n'] // (args.reps + 3)} rollout calls/decision) --")
    for k, v in sorted(tot.items(), key=lambda kv: -kv[1]):
        print(f"  {k:10s} {v/args.reps:9.2f} ms   ({v/args.reps/wall*100:5.1f} %)")
    print(f"  {'TOTAL':10s} {wall:9.2f} ms   unaccounted "
          f"{wall - sum(tot.values())/args.reps:.1f} ms")
    results.setdefault(bs, {})['sync_attribution'] = dict(
        wall_ms=wall, rollout_calls_per_decision=calls['n'] // (args.reps + 3),
        **{k: v/args.reps for k, v in tot.items()})

# -- whole-decision CUDA-graph capture (optional) --
if args.graph:
    print('\n-- whole-decision CUDA-graph capture --')
    # refine_dp30's proven recipe: Windows stack overflow under capture on the
    # main thread + per-thread hipBLAS handles -> warmup AND capture on one
    # large-stack worker thread; hipSOLVER cannot run inside capture -> the
    # capture-safe batched solver (same switch the refine step uses).
    import threading as _th

    _S.solver_capture_safe = True
    graph_tag = ((f'_roll{args.n_roll}' if args.n_roll else '') +
                 (f'_dt{args.dt_roll}' if args.dt_roll else '') +
                 (f'_ref{args.n_refine}' if args.n_refine is not None else ''))
    try:
        for bs in args.batch:
            b = make_batch(bs, np.random.default_rng(args.seed))
            st = {k: torch.empty_like(v) for k, v in
                  (('feat', b['feat']), ('x_meas', b['x_meas']),
                   ('vmask', b['vmask']))}
            for k in st:
                st[k].copy_(b[k])

            def _dec():
                with torch.no_grad():
                    model(st['feat'], st['x_meas'], hard=True, ctrl=True,
                          vmask=st['vmask'])
            _dec(); _dec()   # main thread: materialize lazy state

            holder = {}
            def _capture():
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        _dec()
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    with torch.no_grad():
                        holder['out'] = model(st['feat'], st['x_meas'],
                                              hard=True, ctrl=True,
                                              vmask=st['vmask'])
                holder['graph'] = g
            old_stack = _th.stack_size()
            for _sz in (256, 128, 64, 32):
                try:
                    _th.stack_size(_sz * 1024 * 1024)
                    break
                except (ValueError, OverflowError):
                    continue
            th_ = _th.Thread(target=_capture)
            th_.start(); th_.join()
            _th.stack_size(old_stack)
            if 'graph' not in holder:
                print(f'  batch {bs}: capture failed on the worker thread')
                continue
            g = holder['graph']
            # correctness: replay vs eager under the SAME capture-safe solver
            with torch.no_grad():
                out_e = model(b['feat'], b['x_meas'], hard=True, ctrl=True,
                              vmask=b['vmask'])
            g.replay(); sync()
            dmax = max(((out_e[k] - holder['out'][k]).abs().max().item()
                        for k in out_e
                        if torch.is_tensor(out_e.get(k)) and k in holder['out']
                        and torch.is_tensor(holder['out'][k])), default=float('nan'))
            EAGER_MS = (results.get(bs, {}).get('sync_attribution') or {}).get('wall_ms')
            for _ in range(5):
                g.replay()
            sync(); t = time.perf_counter()
            for _ in range(args.reps):
                g.replay()
            sync()
            ms = (time.perf_counter() - t) / args.reps * 1000
            results.setdefault(bs, {}).update(
                **{f'graph_decision_ms{graph_tag}': ms,
                   f'graph_maxdiff{graph_tag}': dmax,
                   f'eager_ms{graph_tag}': EAGER_MS},
                eager_ms=EAGER_MS, rollout_calls=2 + N_REFINE)
            print(f'  batch {bs}: graph replay = {ms:.3f} ms | '
                  f'eager = {EAGER_MS:.1f} ms | speedup {EAGER_MS/ms:.0f}x | '
                  f'max|diff| {dmax:.2e}\n')
    finally:
        _S.solver_capture_safe = False

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, 'profile_latency.json'), 'w') as f:
    json.dump(results, f, indent=1, default=str)
print(f"-> {args.out}/profile_latency.json")
