"""
refine_dp30.py
==============
30-bus port of refine_dp.py. Second-stage refinement: take a trained
checkpoint and push the control effort ||dP|| down as far as it will go
WITHOUT giving up rescue rate.

    python refine_dp30.py --ckpt case30_ctrl/best.pt --data data30_mm --floor_tol 0

The method is unchanged from the 9-bus script:

    minimise   ||dP||^2
    subject to rescue_rate >= floor

with an adaptive multiplier (dual ascent),

    loss   = ||dP||^2 + lambda * (failure surrogate + violation)
    lambda <- lambda * exp(eta * (floor - rescue))     each epoch

The floor defaults to the loaded model's own measured validation rescue rate
(minus --floor_tol), so refinement can never end up worse than its start.
Selection is lexicographic: a checkpoint is better only if it MEETS the floor
and has a lower ||dP||. Only the dP head trains (identification is frozen).

What changed for the 30-bus system
----------------------------------
* dP is 2-wide (G1, G2); the rest comes from pinn_model30 / case30_physics.
* Batches, features, normalisation and the actionability gate come from
  case30_common.py, which is checked draw-for-draw against train_case30.py.
  The 9-bus version kept its own copy, and that copy drifted once -- it set
  the floor on a different rescue definition than training reported.
* The mask sharpness beta is kept as the checkpoint stored it.
* ||dP|| is also reported split into cases that NEED rescue and those that
  don't, so a reduction can be told apart from "stays quiet when nothing is
  wrong".

Before trusting any refined number, run verify_true30.py on the result:
refinement shrinks dP until the SIMULATED peak only just clears
limit - margin, and the model's own trajectory error (~9.5e-4 omega RMSE on
the 30-bus test set) is about the size of the default margin.
"""

import argparse, json, os, sys, time

import numpy as np
import torch
import torch.nn as nn

from case30_common import Setup, actionable, AGE_BINS, fmt_age, NG

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--data', required=True)
ap.add_argument('--base', default='case30_base.mat')
ap.add_argument('--sys', default='case30_simulink_data.mat')
ap.add_argument('--out', default=None, help='default: <ckpt dir>/refined')
ap.add_argument('--epochs', type=int, default=200)
ap.add_argument('--batch_size', type=int, default=512)
ap.add_argument('--instants_per_scenario', type=int, default=2)
ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--rescue_floor', type=float, default=None,
                help='Rescue rate to hold. Default: the loaded model measured '
                     'on validation, minus --floor_tol.')
ap.add_argument('--floor_tol', type=float, default=0.005)
ap.add_argument('--lam0', type=float, default=10.0)
ap.add_argument('--lam_eta', type=float, default=5.0)
ap.add_argument('--lam_min', type=float, default=0.1)
ap.add_argument('--lam_max', type=float, default=1e4)
ap.add_argument('--patience', type=int, default=30)
ap.add_argument('--warmup', type=int, default=40,
                help='Epochs before patience starts counting. lambda starts at '
                     '--lam0 (deliberately high, so the constraint holds from '
                     'the first step) and needs time to decay to the exchange '
                     'rate where ||dP|| can shrink; on the 9-bus run ||dP|| sat '
                     'ABOVE the baseline for the first 23 epochs and only then '
                     'fell to -26%. Stopping inside that phase ends the search '
                     'before it starts.')
ap.add_argument('--min_delta', type=float, default=1e-4)
ap.add_argument('--val_instants', type=int, default=4096)
ap.add_argument('--unfreeze_all', action='store_true')
ap.add_argument('--print_every', type=int, default=1)
ap.add_argument('--no_progress', action='store_true')
ap.add_argument('--cuda_graph', action='store_true',
                help='capture the training step as a CUDA graph (launch '
                     'overhead of the unrolled rollout dominates; needs '
                     'full-size batches, ROCm/HIP capture, --batch_size '
                     'dividing the epoch)')
args = ap.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)
su = Setup(args.ckpt, args.data, args.base, args.sys)
device, model, A, ck = su.device, su.model, su.A, su.ck
if device.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(args.ckpt)),
                                   'refined')
os.makedirs(out_dir, exist_ok=True)


# ── progress bar ─────────────────────────────────────────────────────────────
def _hms(sec):
    sec = int(max(0, sec))
    d, sec = divmod(sec, 86400); h, sec = divmod(sec, 3600); m, sec = divmod(sec, 60)
    return f"{d}d {h:02d}:{m:02d}:{sec:02d}" if d else f"{h:02d}:{m:02d}:{sec:02d}"


class Progress:
    def __init__(self, n_epochs, n_batches, width=22, enabled=True):
        self.ne, self.nb, self.w = n_epochs, n_batches, width
        self.enabled = enabled and sys.stderr.isatty()
        self.t0 = time.time(); self.ep_times = []; self.ep_start = self.t0

    def batch(self, epoch, bi, stats, extra=""):
        if not self.enabled:
            return
        frac = (bi + 1) / self.nb
        bar = "#" * int(self.w * frac) + "." * (self.w - int(self.w * frac))
        el = time.time() - self.ep_start
        rate = (bi + 1) / max(el, 1e-9)
        eta_ep = (self.nb - bi - 1) / max(rate, 1e-9)
        per_ep = np.mean(self.ep_times) if self.ep_times else el / max(frac, 1e-9)
        eta_run = eta_ep + per_ep * (self.ne - epoch)
        txt = "  ".join(f"{k}={v:.4f}" for k, v in stats.items())
        if extra:
            txt += "  " + extra
        sys.stderr.write(f"\rep {epoch}/{self.ne} [{bar}] {bi+1}/{self.nb}  {txt}  "
                         f"{rate:.1f} b/s  ep {_hms(eta_ep)}  run {_hms(eta_run)}   ")
        sys.stderr.flush()

    def epoch_start(self):
        self.ep_start = time.time()

    def epoch_end(self):
        self.ep_times = (self.ep_times + [time.time() - self.ep_start])[-20:]
        if self.enabled:
            sys.stderr.write("\r" + " " * 140 + "\r"); sys.stderr.flush()

    def note(self, msg):
        if self.enabled:
            sys.stderr.write("\r" + " " * 140 + "\r"); sys.stderr.flush()
        print(msg)

    def elapsed(self):
        return time.time() - self.t0


# ── freeze all but the dP head ───────────────────────────────────────────────
if not args.unfreeze_all:
    for q in model.parameters():
        q.requires_grad_(False)
    train_mods = ([model.dp_cell, model.dp_fc] if A['dp_head'] == 'cascade'
                  else [model.enc.head_dP])
    for m_ in train_mods:
        for q in m_.parameters():
            q.requires_grad_(True)
trainable = [q for q in model.parameters() if q.requires_grad]
print(f"Trainable parameters: {sum(q.numel() for q in trainable):,} "
      f"of {sum(q.numel() for q in model.parameters()):,}"
      f"{'  (all unfrozen)' if args.unfreeze_all else ''}")

# capturable=True is a CUDA-graph requirement: the optimizer step keeps its
# state tensors on-device so a captured step can be replayed. The eager path
# is math-identical; capturable Adam only moves the scalars, at a small
# per-step cost that graph replay repays many times over.
opt = torch.optim.Adam(trainable, lr=args.lr, capturable=args.cuda_graph)
val_bs = min(args.batch_size, args.val_instants)
n_val_batches = max(1, args.val_instants // val_bs)


# ── speedups (2026-10-22), twin of refine_dp.py: val-cache + prefetch ────────
import queue as _queue
from concurrent.futures import ThreadPoolExecutor as _TPE

VALCACHE = {}   # id(pool) -> list of batch dicts (CPU-side)

def _val_batches(pool):
    b = VALCACHE.get(id(pool))
    if b is None:
        rng_v = np.random.default_rng(12345)
        b = [su.make_batch(rng_v.choice(pool, val_bs, replace=True), rng_v)
             for _ in range(n_val_batches)]
        VALCACHE[id(pool)] = b
    return b


class _Prefetcher:
    def __init__(self, orders, seeds):
        # orders: per-epoch permutations (pre-drawn, deterministic schedule);
        # seeds[ep][idx] is the rng seed for that epoch's batch idx.
        self.ex = _TPE(max_workers=1)
        self.q = _queue.deque()
        self.orders, self.seeds, self.n_ep = orders, seeds, len(orders)
        self.epoch = 0
        self.i = 0              # ABSOLUTE cursor; epoch = i // n_batches

    def _ep_idx(self, bi):
        ep, idx = divmod(bi, n_batches)
        return ep, idx

    def _submit(self, bi):
        ep, idx = self._ep_idx(bi)
        if ep >= self.n_ep:          # beyond the final epoch: permanent marker
            self.q.append((self.n_ep - 1, None))
            return
        if ep < self.epoch:          # stale slot (only if pipeline backed up)
            self.q.append((ep, None))
            return
        ks_b = self.orders[ep][idx * args.batch_size:(idx + 1) * args.batch_size]
        if not len(ks_b):
            self.q.append((ep, None))
            return
        seed = self.seeds[ep][idx]
        self.q.append((ep, self.ex.submit(
            lambda s=seed, k=ks_b, e=ep: (e, su.make_batch(k, np.random.default_rng(s))))))

    def start(self):
        for _ in range(2):
            self._submit(self.i)
            self.i += 1

    def next(self):
        """Next batch of the CURRENT epoch; entries from earlier epochs are
        drained and discarded; returns None at end of the current epoch."""
        while True:
            ep, item = self.q.popleft()
            val = item.result() if item is not None else None
            self._submit(self.i)
            self.i += 1
            if ep != self.epoch:
                continue             # stale entry: discard
            if val is None:
                return None          # end-of-epoch marker for THIS epoch
            _, batch = val
            if _ != self.epoch:
                continue
            return batch

    def close(self):
        self.ex.shutdown(wait=False)


@torch.no_grad()
def evaluate(pool):
    model.eval()
    hit = n_need = ok_hit = ok_n = 0
    norm_all = []; norm_need = []; norm_quiet = []
    age_hit = [0] * len(AGE_BINS); age_n = [0] * len(AGE_BINS)
    for b_cpu in _val_batches(pool):
        b = su.to_dev(b_cpu)
        o = model(b['feat'], b['x_meas'], hard=True, ctrl=True, vmask=b['vmask'])
        pk_t, _ = model.judged(b['fut'][:, :, 0:NG], b['vmask'])
        pk_c, has_c = model.judged(o['w_ctrl'], b['vmask'])
        reach = actionable(has_c, b['x_meas'], b['active'], model.limit)
        need = (pk_t > model.limit) & reach
        won = pk_c <= model.limit
        hit += int(won[need].sum()); n_need += int(need.sum())
        ok_hit += int(won[reach].sum()); ok_n += int(reach.sum())
        nrm = o['dP'].pow(2).sum(1).sqrt()
        norm_all.append(nrm.cpu()); norm_need.append(nrm[need].cpu())
        norm_quiet.append(nrm[~need].cpu())
        ag = b['age'].cpu().numpy(); nd = need.cpu().numpy(); wn = won.cpu().numpy()
        for i, (b0, b1) in enumerate(AGE_BINS):
            m = nd & (ag >= b0) & (ag < b1)
            if m.sum():
                age_hit[i] += int(wn[m].sum()); age_n[i] += int(m.sum())
    model.train()
    mean = lambda L: float(torch.cat(L).mean()) if sum(len(x) for x in L) else float('nan')
    return {'rescue': hit / max(n_need, 1), 'n_need': n_need,
            'ok': ok_hit / max(ok_n, 1), 'norm': mean(norm_all),
            'norm_need': mean(norm_need), 'norm_quiet': mean(norm_quiet),
            'age_hit': age_hit, 'age_n': age_n}


base = evaluate(su.idx_va)
# the unrefined model on TEST too, so the final comparison is test vs test.
# (The 9-bus script compared a validation baseline against a test result.)
base_te = evaluate(su.idx_te)
floor = (args.rescue_floor if args.rescue_floor is not None
         else max(0.0, base['rescue'] - args.floor_tol))
print(f"\nBaseline (validation): rescue {base['rescue']:.4f} on {base['n_need']} "
      f"cases, ok {base['ok']:.4f}, ||dP|| {base['norm']:.4f} pu "
      f"(needing rescue {base['norm_need']:.4f}, no rescue needed "
      f"{base['norm_quiet']:.4f})")
print(f"Constraint: hold rescue >= {floor:.4f}; minimise ||dP|| under it.")
print(f"Margin {model.margin:g} (target peak {model.limit - model.margin:.4f})\n")

lam = args.lam0
# Start the search AT the baseline: a checkpoint is kept only if it is
# feasible AND cheaper than the model we started from. (Starting from inf, as
# the 9-bus script did, saved the first feasible epoch even when its ||dP|| was
# higher than the baseline's, so a run that never improved could still hand
# back a worse model.)
best_norm = base['norm']; best_epoch = 0; since = 0
n_feasible = 0; min_norm_feasible = float('inf')
epoch_units = np.repeat(su.idx_tr, args.instants_per_scenario)
n_batches = max(1, int(np.ceil(len(epoch_units) / args.batch_size)))
rng_b = np.random.default_rng(args.seed + 1)
trace = []; t0 = time.time()
prog = Progress(args.epochs, n_batches, enabled=not args.no_progress)
print(f"Epoch = {len(epoch_units)} decision instants "
      f"({len(su.idx_tr)} scenarios x {args.instants_per_scenario}) = {n_batches} "
      f"batches   |   validation: {n_val_batches * val_bs} fixed instants each epoch")

# Pre-draw every epoch's permutation and batch seeds up front: deterministic
# schedule per --seed, independent of pipeline timing.
_all_orders = [rng_b.permutation(epoch_units) for _ in range(args.epochs)]
_all_seeds = [[int(rng_b.integers(0, 2**63 - 1)) for _ in range(n_batches)]
              for _ in range(args.epochs)]
pf = _Prefetcher(_all_orders, _all_seeds)
pf.start()

# ── CUDA graph capture of the training step ─────────────────────────────────
# Twin of refine_dp.py (commit 3246cfd). The step is launch-overhead bound:
# the rollout runs n_steps tiny RK4 sub-steps and the cascade head unrolls
# n_refine simulator-in-the-loop cells (each with its own rollout), all of
# which submit tiny kernels per call. Shapes are static (full batches only;
# a partial final batch falls back to eager), the control flow is python-flag
# only, and the dual-ascent lam is read from a device tensor (lam_t) because a
# python float would be baked into the graph.
CG = args.cuda_graph and device.type == 'cuda'
if args.cuda_graph and device.type != 'cuda':
    print("--cuda_graph requested but no CUDA device; running eager.")
cg_graph = None
cg_out = lam_t = static = None
captured_this_batch = False
CG_KEYS = ('feat', 'x_meas', 'fut', 'vmask', 'active', 'age')
for epoch in range(1, args.epochs + 1):
    prog.epoch_start()
    pf.epoch = epoch - 1   # next() drops entries from earlier epochs
    agg = dict(norm=0., fail=0., n=0)
    for bi in range(n_batches):
        got = pf.next()
        if got is None:
            continue
        b = su.to_dev(got)
        full_batch = b['feat'].shape[0] == args.batch_size
        captured_this_batch = False

        if CG and cg_graph is None and full_batch:
            # First full batch: create the lam device tensor, warm up on a side
            # stream, then capture one step (forward + backward + clip + step).
            lam_t = torch.full((), lam, device=b['feat'].device)
            su.S.solver_capture_safe = True   # hipSOLVER can't run inside capture
            static = {k: b[k] for k in CG_KEYS}

            def _step():
                o = model(static['feat'], static['x_meas'], ctrl=True,
                          vmask=static['vmask'])
                pk_c, has_c = model.judged(o['w_ctrl'], static['vmask'])
                viol = model.violation(o['w_ctrl'], static['vmask'])
                reach = actionable(has_c, static['x_meas'], static['active'],
                                   model.limit)
                rm = reach.float()
                tgt = model.limit - model.margin
                p_fail = torch.sigmoid((pk_c - tgt) / A['dp_tau'])
                env = (((p_fail + A['w_viol'] * viol / model.limit) * rm).sum()
                       / rm.sum().clamp(min=1))
                nrm = o['dP'].pow(2).sum(1).mean()
                (nrm + lam_t * env).backward()
                nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                return nrm, env

            # Warmup AND capture run on the same large-stack worker thread:
            # (a) the step records tens of thousands of tiny kernels (unrolled
            #     RK4 rollouts + their unrolled backward), which overflowed the
            #     default 8 MB main-thread stack on Windows Python (0xC00000FD);
            #     256/128 MB are REJECTED by Windows Python _threading — try
            #     64 -> 32 -> 16;
            # (b) hipBLAS handles are per-thread and CANNOT be created during
            #     capture, so the warmup must materialize them on this thread
            #     BEFORE the capture attempt.
            import threading as _th
            holder = {}

            def _capture():
                s_ = torch.cuda.Stream()
                s_.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s_):
                    for _ in range(3):
                        _step()
                torch.cuda.current_stream().wait_stream(s_)
                opt.zero_grad(set_to_none=True)   # warmup left grads behind
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    holder['out'] = _step()
                holder['graph'] = g

            old_stack = _th.stack_size()
            for _sz in (64, 32, 16):
                try:
                    _th.stack_size(_sz * 1024 * 1024)
                    break
                except (ValueError, OverflowError):
                    continue
            th_ = _th.Thread(target=_capture)
            th_.start(); th_.join()
            _th.stack_size(old_stack)
            if 'graph' not in holder:
                raise RuntimeError('CUDA graph capture failed on the worker thread')
            cg_graph = holder['graph']
            cg_out = holder['out']
            # the captured forward/backward/step already ran on THIS batch;
            # skip its replay and fall through to the agg update.
            captured_this_batch = True

        if CG and cg_graph is not None and full_batch:
            if not captured_this_batch:   # capture batch: its step already ran
                for q in trainable:
                    if q.grad is not None:
                        q.grad.zero_()   # zero_grad(set_to_none) is illegal around graphs
                for k in CG_KEYS:
                    static[k].copy_(b[k], non_blocking=True)
                lam_t.fill_(lam)
                cg_graph.replay()
            norm, envelope = cg_out
        else:
            opt.zero_grad(set_to_none=True)
            o = model(b['feat'], b['x_meas'], ctrl=True, vmask=b['vmask'])
            pk_c, has_c = model.judged(o['w_ctrl'], b['vmask'])
            viol = model.violation(o['w_ctrl'], b['vmask'])
            # same gate in the loss as in the metric: instants with nothing to
            # correct would otherwise push dP UP, against the whole point here
            rm = actionable(has_c, b['x_meas'], b['active'], model.limit).float()
            tgt = model.limit - model.margin
            p_fail = torch.sigmoid((pk_c - tgt) / A['dp_tau'])
            envelope = (((p_fail + A['w_viol'] * viol / model.limit) * rm).sum()
                        / rm.sum().clamp(min=1))
            norm = o['dP'].pow(2).sum(1).mean()
            (norm + lam * envelope).backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
        agg['norm'] += float(norm.detach()); agg['fail'] += float(envelope.detach())
        agg['n'] += 1
        prog.batch(epoch, bi, {'|dP|^2': agg['norm'] / agg['n'],
                               'env': agg['fail'] / agg['n']},
                   extra=f"lam={lam:8.2f}  best|dP|="
                         f"{(best_norm if best_norm < 1e9 else float('nan')):.4f}pu")

    prog.epoch_end()
    v = evaluate(su.idx_va)
    trace.append({'epoch': epoch, 'rescue': v['rescue'], 'norm': v['norm'],
                  'norm_need': v['norm_need'], 'norm_quiet': v['norm_quiet'],
                  'ok': v['ok'], 'lam': lam})
    lam = float(np.clip(lam * np.exp(args.lam_eta * (floor - v['rescue'])),
                        args.lam_min, args.lam_max))

    feasible = v['rescue'] >= floor
    improved = feasible and v['norm'] < best_norm - args.min_delta
    if feasible:
        n_feasible += 1
        min_norm_feasible = min(min_norm_feasible, v['norm'])
    if improved:
        best_norm, best_epoch, since = v['norm'], epoch, 0
        torch.save({**ck, 'model': model.state_dict(), 'epoch': epoch,
                    'refined_from': args.ckpt, 'refine_args': vars(args),
                    'val': {k: v[k] for k in ('rescue', 'norm', 'ok', 'n_need')},
                    'rescue_floor': floor, 'lam': lam},
                   os.path.join(out_dir, 'best.pt'))
    elif epoch > args.warmup:
        since += 1

    if epoch % args.print_every == 0:
        flag = '  <-- best' if improved else (
            '  INFEASIBLE' if not feasible else (
                f'  (warm-up {epoch}/{args.warmup})' if epoch <= args.warmup
                else f'  ({since}/{args.patience})'))
        prog.note(f"  ep {epoch:4d}  rescue={v['rescue']:.4f}  "
                  f"||dP||={v['norm']:.4f}pu ({100*v['norm']/base['norm']:.1f}% of "
                  f"baseline; need {v['norm_need']:.3f} / quiet {v['norm_quiet']:.3f})  "
                  f"ok={v['ok']:.4f}  lam={lam:8.2f}  [{_hms(prog.elapsed())}]{flag}")
    if args.patience and since >= args.patience:
        print(f"\nStopped: no feasible ||dP|| reduction for {args.patience} epochs.")
        break

json.dump(trace, open(os.path.join(out_dir, 'refine_pareto.json'), 'w'), indent=1)

print(f"\n{'='*68}")
print(f"REFINEMENT  ({time.time()-t0:.0f}s, best epoch {best_epoch})")
print(f"{'='*68}")
if best_epoch:
    bk = torch.load(os.path.join(out_dir, 'best.pt'), map_location=device,
                    weights_only=False)
    model.load_state_dict(bk['model'])
    te = evaluate(su.idx_te)
    red = 100 * (1 - te['norm'] / base_te['norm'])
    print(f"  (all figures: TEST split, unrefined -> refined)")
    print(f"  ||dP||       {base_te['norm']:.4f} -> {te['norm']:.4f} pu   "
          f"({red:.1f}% {'reduction' if red >= 0 else 'INCREASE'})")
    print(f"     needing rescue   {base_te['norm_need']:.4f} -> {te['norm_need']:.4f} pu")
    print(f"     no rescue needed {base_te['norm_quiet']:.4f} -> {te['norm_quiet']:.4f} pu")
    print(f"  rescue       {base_te['rescue']:.4f} -> {te['rescue']:.4f}   "
          f"(floor {floor:.4f} set on validation; n={te['n_need']})")
    print(f"  overall ok   {base_te['ok']:.4f} -> {te['ok']:.4f}")
    print(f"  by fault age {fmt_age(AGE_BINS, te['age_hit'], te['age_n'])}")
    json.dump({'baseline_val': base, 'baseline_test': base_te, 'test': te,
               'floor': floor,
               'best_epoch': best_epoch},
              open(os.path.join(out_dir, 'refine_results.json'), 'w'), indent=1)
    print(f"\n  -> {out_dir}/best.pt, refine_results.json, refine_pareto.json")
    print(f"  NEXT: python verify_true30.py --ckpt {out_dir}/best.pt "
          f"--data {args.data}")
else:
    # say WHICH way it failed -- they call for opposite fixes
    if n_feasible == 0:
        print("  Rescue never reached the floor. --lam0 is too low to hold the "
              "constraint, or the floor is above what the head can do.")
    else:
        print(f"  {n_feasible} epoch(s) met the rescue floor, but none was cheaper "
              f"than the starting model (best feasible ||dP|| "
              f"{min_norm_feasible:.4f} vs baseline {base['norm']:.4f} pu).")
        print(f"  lambda ended at {lam:.2f}. If it was still falling, the search "
              f"stopped before the norm term took over: raise --warmup or lower "
              f"--lam0.")

# Shutdown hygiene (CG runs only): pf's worker is a non-daemon thread that can
# hang interpreter exit holding a CUDA context; and HIP graphs hold a private
# allocator pool whose release deadlocks this torch build's interpreter
# shutdown on Windows (graph.reset()/empty_cache() both hang) — bypass
# teardown entirely and let the OS reclaim at process exit. Everything before
# this point is already flushed and saved.
pf.close()
if CG and cg_graph is not None:
    import sys as _sys
    _sys.stdout.flush(); _sys.stderr.flush()
    os._exit(0)
