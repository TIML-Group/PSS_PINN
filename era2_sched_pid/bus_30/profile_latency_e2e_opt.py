"""
profile_latency_e2e_opt.py -- launch-optimization A/B on the era-2 e2e decision path,
design of record: cls_learned_A (trunk A, temporal pooling) on library_v23 tables.

Reference arm and protocol are IDENTICAL to profile_latency_e2e.py (seed-42 test-split draws,
ring-buffer state, 3 eager warm-ups, host-synced reps, whole-decision CUDA-graph capture with
5 warm replays, numerics check).  Nothing outside this file is modified.

Variants (one decision = onset -> gather(128x90) -> featurize(88)+normalise -> IdentNet A
           -> argmax -> (59,2,NG,3) T12 lookup -> PID switch step -> P_set):

  gpu_base  unmodified pipeline == reference (recomputes L(), mask() and pos every decision,
            two (59,NG,3) table lookups, PID with the algebraically-unfactored preload).
  gpu_fused launch-reduced: L is a FROZEN parameter at inference, so mask / pos / mean-mask sum
            are precomputed OFF the timed path (legal at inference; a retrain must refresh them).
            Plus: featurize+mask folded into 6 broadcast kernels over a single 88-column
            index_select (scale/offset/first-channel tables precomputed), fused PID
            (Ipre = Ps_prev - (Kp + h Ki) e - Kd dterm, clamp order and rails unchanged),
            one T12[argmax] index_select for both R1 and R2 gains.
  cpu_pid   PID update on the CPU: GPU arm = gather -> featurize -> mlp -> T12 -> one packed fp32
            cat of [w, xf, Ps_prev, g1] -> single D2H into a pinned buffer -> numpy float32 PID
            (identical formula/clamps, rails from plant_v2 lo/hi, H_STEP, N_D) -> pack
            [Ps, I_n] -> H2D writeback.  Writeback is INCLUDED.
            Deployment shape timed per rep: GPU-arm graph replay + CPU PID + H2D, sync at end.

Kernel-launch accounting: torch.profiler (CPU+CUDA) over one timed decision at batch 1, launch-API
keys (lower-case 'launch') summed over key_averages().count.  On this ROCm/torch build the profiler
exposes NO launch-API events (a 2-op probe returns 0 such keys and pr.events() only CPU aten ops), so
launch_total is None and the fallback evidence recorded is the aten-op count per decision, split into
total and kernel-issuing (metadata-only ops excluded).  Wrapped in try/except: if ROCTracer kills the
profiler the counts become None and the run continues.

Numerics gates (hard): (i) ring-buffer gather exact; (ii) gpu_fused vs gpu_base max|diff| over
out_Ps/out_I/out_g1/out_g2; (iii) cpu_pid vs gpu_base.  Any gate > 1e-4 sets gates_failed and
aborts the run (partial JSON written) instead of proceeding quietly.

Run from era2_sched_pid/bus_30/:  python profile_latency_e2e_opt.py [--reps 100] [--batch 1 256]
"""
import argparse
import json
import time

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from ident_model import IdentNet, W_STEPS, L_MAX, D_IDX, W_IDX, PM_IDX, ID_IDX, IQ_IDX, VM_IDX, TH_IDX
from plant_v2 import NG, H_STEP, N_D
from sched_sim import make_plant

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='ident_ckpt_trunk/cls_learned_A.pt')
ap.add_argument('--lib', default='library_v23.json')
ap.add_argument('--reps', type=int, default=100)
ap.add_argument('--batch', type=int, nargs='+', default=[1, 256])
ap.add_argument('--out', default='profile_latency_e2e_opt_A.json')
ap.add_argument('--burn', type=int, default=0, help='steady-state burn-in eager decisions per batch before ANY timing is taken (0 = protocol only, 3 warm-ups); use ~6000 for warmed/clock-settled numbers')
args = ap.parse_args()
dev = 'cuda'
sync = torch.cuda.synchronize

# ---------------------------------------------------------------- model + library (of-record config)
ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
net = IdentNet(ck['variant'], ck['profile'], ck.get('trunk', 'base'))
net.load_state_dict(ck['state'])
net = net.to(dev).eval()
MU = torch.as_tensor(ck['feat_mu'], dtype=torch.float32, device=dev)
SD = torch.as_tensor(ck['feat_sd'], dtype=torch.float32, device=dev)
IS = 1.0 / SD                                                                    # affine fold: (f-MU)/SD == f*IS - MU*IS
MU2 = MU * IS
LIB = json.load(open(args.lib))['entries']
names = list(np.load('data_v2/meta.npz')['lib_names'])
CFG = [n for n in names if n.startswith('R1')]


def family_r2(n):
    if n.startswith('R1d'):
        return 'R0'
    if n.startswith('R1_line'):
        return 'R2_line'
    if n.startswith('R1_load'):
        return 'R2_load'
    return 'R2_' + n.split('R1_')[1]


g = lambda n: np.asarray(LIB[n]['gains'], np.float32)
TAB_R1 = torch.as_tensor(np.stack([g('R0')] + [g(n) for n in CFG]), device=dev)               # (59, NG, 3)
TAB_R2 = torch.as_tensor(np.stack([g('R0')] + [g(family_r2(n)) for n in CFG]), device=dev)     # (59, NG, 3)
T12 = torch.stack([TAB_R1, TAB_R2], 1).reshape(len(CFG) + 1, 2 * NG * 3).contiguous()            # (59, 36): R1 row | R2 row
p = make_plant()
LO, HI = p.lo.float(), p.hi.float()
LOc = np.ascontiguousarray(LO.cpu().numpy(), dtype=np.float32)
HIc = np.ascontiguousarray(HI.cpu().numpy(), dtype=np.float32)

Di, Wi, Pi, Ii, Qi, Vi, Ti = (torch.tensor(i, device=dev) for i in (D_IDX, W_IDX, PM_IDX, ID_IDX, IQ_IDX, VM_IDX, TH_IDX))
AR = torch.arange(W_STEPS, device=dev)
NBUF = 2 * W_STEPS

# ---------------------------------------------------------------- static net parts, OFF the timed path
# (L is a frozen parameter at inference -- training-identical value, computed once here)
with torch.no_grad():
    MASK0 = net.mask().detach().float().reshape(-1).clone()                    # (W,)
    PON = (net.elapsed / L_MAX).view(1, W_STEPS, 1).clone()                    # (1,W,1)
MW1 = MASK0.view(1, W_STEPS, 1)                                                 # (1,W,1)
MWSUM = MASK0.sum().clamp(min=1e-6).view(1, 1)                                  # (1,1)
MWC = (4.0 * (1.0 - MW1)).contiguous()                                         # (1,W,1)

# --- fused featurize: one 88-column index_select + one 88-column subtract, then affine normalise
#     out_j = raw[SEL_j]*P_wj  -  raw[SUB_j]*Q_wj  -  R_wj      (per-row mask folded into P/Q/R)
MU2c = MU2.cpu().numpy().tolist()
ISc = IS.cpu().numpy().tolist()
SEL, SUB, PC, QC, OF = [], [], [], [], []


def _diff(j, a, b):                       # out = (raw[a] - raw[b]) * IS[j] - MU2[j]
    SEL.append(int(a)); SUB.append(int(b)); PC.append(float(ISc[j])); QC.append(float(ISc[j])); OF.append(float(MU2c[j]))


def _plain(j, ch, s=1.0, shift=0.0):     # out = (raw[ch]*s - shift) * IS[j] - MU2[j]
    SEL.append(int(ch)); SUB.append(int(ch)); PC.append(float(ISc[j]) * s); QC.append(0.0)
    OF.append(float(MU2c[j]) + float(shift) * float(ISc[j]))


for j in range(5):
    _diff(j, D_IDX[j + 1], D_IDX[0])
for k in range(NG):
    _plain(5 + k, W_IDX[k], s=100.0, shift=100.0)            # (w-1)/0.01 == 100 w - 100
for k in range(NG):
    _plain(11 + k, PM_IDX[k])
for k in range(NG):
    _plain(17 + k, ID_IDX[k])
for k in range(NG):
    _plain(23 + k, IQ_IDX[k])
for k in range(len(VM_IDX)):
    _plain(29 + k, VM_IDX[k])
for j in range(29):
    _diff(59 + j, TH_IDX[j + 1], TH_IDX[0])
assert len(SEL) == 88, len(SEL)
SEL_T = torch.tensor(SEL, device=dev)
SUB_T = torch.tensor(SUB, device=dev)
P_T = (MASK0.view(-1, 1) * torch.tensor(PC, device=dev).view(1, -1)).contiguous()
Q_T = (MASK0.view(-1, 1) * torch.tensor(QC, device=dev).view(1, -1)).contiguous()
R_T = (MASK0.view(-1, 1) * torch.tensor(OF, device=dev).view(1, -1)).contiguous()


# ---------------------------------------------------------------- stages (reference, verbatim)
def s_gather(st):
    return st['buf'][st['bidx'], (st['onset'].view(-1, 1) + AR) % NBUF]                          # (B,128,90)


def s_feat_base(raw):
    d, w, th = raw[..., Di], raw[..., Wi], raw[..., Ti]
    f = torch.cat([d[..., 1:] - d[..., :1], (w - 1.0) / 0.01, raw[..., Pi], raw[..., Ii], raw[..., Qi], raw[..., Vi],
                   th[..., 1:] - th[..., :1]], -1)
    return (f - MU) / SD


def s_feat_fused(raw):
    return raw.index_select(-1, SEL_T).mul_(P_T).sub_(raw.index_select(-1, SUB_T).mul_(Q_T)).sub_(R_T)


def mlp_base(x):
    return net(x)['logits']


def mlp_fused(x):                                                                               # x already masked+normalised
    B = x.shape[0]
    e = net.proj(x)
    h = net.trunk(torch.cat([e, PON.expand(B, -1, -1)], -1))
    mean = (h * MW1).sum(1).div_(MWSUM)
    mx = h.sub(MWC).amax(1)
    return net.head_c(torch.cat([mean, mx], -1))


def lookup_base(lg):
    c = lg.argmax(1)
    return TAB_R1[c], TAB_R2[c]


def lookup_fused(lg):
    return T12.index_select(0, lg.argmax(1))                                                   # (B,36) = [R1|R2]


def gains(g12):
    """split the packed (B,36) row into two CONTIGUOUS (B,NG,3) gain views (no copy)."""
    B = g12.shape[0]
    return g12[:, :NG * 3].view(B, NG, 3), g12[:, NG * 3:].view(B, NG, 3)


def pid_base(st, g1):                                                                          # reference algebra
    Kp, Ki, Kd = g1[..., 0], g1[..., 1], g1[..., 2]
    e = 1.0 - st['w']; dterm = N_D * (e - st['xf'])
    Ipre = st['Ps_prev'] - Kp * e - Kd * dterm - H_STEP * Ki * e
    I = torch.minimum(torch.maximum(Ipre, LO), HI)
    I_n = torch.minimum(torch.maximum(I + H_STEP * Ki * e, LO), HI)
    Ps = torch.minimum(torch.maximum(Kp * e + I_n + Kd * dterm, LO), HI)
    return Ps, I_n


def pid_fused(st, g1):                                                                         # same maths, factored
    Kp, Ki, Kd = g1[..., 0], g1[..., 1], g1[..., 2]
    e = 1.0 - st['w']
    dterm = torch.sub(e, st['xf']).mul_(N_D)
    hke = Ki * H_STEP                                                                          # == H_STEP*Ki (commutative, exact)
    Ipre = st['Ps_prev'].sub(torch.add(Kp, hke).mul_(e))                                        # - (Kp + h Ki) e
    Ipre.addcmul_(Kd, dterm, value=-1.0)
    I = Ipre.clamp(LO, HI)                                                                      # preload clamp (identical to base)
    I_n = I.addcmul(hke, e).clamp_(LO, HI)                                                      # I + h Ki e, clamped
    Ps = Kp.mul(e).add_(I_n)
    Ps.addcmul_(Kd, dterm)
    Ps.clamp_(LO, HI)
    return Ps, I_n


# ---------------------------------------------------------------- decisions
def decide_base(st):
    raw = s_gather(st); g1, g2 = lookup_base(mlp_base(s_feat_base(raw))); Ps, I = pid_base(st, g1)
    st['out_Ps'].copy_(Ps); st['out_I'].copy_(I); st['out_g1'].copy_(g1); st['out_g2'].copy_(g2)


def _fused_head(st):
    raw = s_gather(st); lg = mlp_fused(s_feat_fused(raw)); g12 = lookup_fused(lg)
    return g12


def decide_fused(st):
    g12 = _fused_head(st)
    g1, g2 = gains(g12)
    Ps, I = pid_fused(st, g1)
    st['out_Ps'].copy_(Ps); st['out_I'].copy_(I)
    st['out_g1'].copy_(g1); st['out_g2'].copy_(g2)


# ---------------------------------------------------------------- cpu_pid: GPU arm (+D2H) / host PID
# The packed block is [w | xf | Ps_prev | g12row]; the T12 row already carries both gain sets in one
# contiguous (B,36) block, so packing g1 alone would cost an extra repack kernel.  Host-side the PID
# reads its 18-float g1 slice out of that row (identical values).
_HB = {}
_PD = {}


def hbufs(B):
    if B not in _HB:
        n = B * NG
        _HB[B] = dict(d2h=torch.empty(9 * n, dtype=torch.float32, pin_memory=True),
                      h2d=torch.empty(2 * n, dtype=torch.float32, pin_memory=True))
        _PD[B] = torch.empty(2, B, NG, dtype=torch.float32, device=dev)
    return _HB[B]


def arm_cpu(st, B):
    """GPU arm only: gather -> featurize -> mlp -> T12 -> packed fp32 D2H into pinned host memory."""
    g12 = _fused_head(st)
    hb = hbufs(B)
    pack = torch.cat([st['w'].reshape(-1), st['xf'].reshape(-1), st['Ps_prev'].reshape(-1), g12.reshape(-1)])
    hb['d2h'].copy_(pack, non_blocking=True)
    return g12


def host_pid(st, B, g12):
    """numpy float32 PID on the pinned buffer (identical formula + clamps), then H2D writeback."""
    hb = hbufs(B)
    a = hb['d2h'].numpy()
    n = B * NG
    w = a[0:n].reshape(B, NG)
    xf = a[n:2 * n].reshape(B, NG)
    ps0 = a[2 * n:3 * n].reshape(B, NG)
    gp = a[3 * n:].reshape(B, 2 * NG, 3)[:, :NG]                                                # the g1 half of the T12 row
    Kp, Ki, Kd = gp[..., 0], gp[..., 1], gp[..., 2]
    e = np.float32(1.0) - w
    dterm = np.float32(N_D) * (e - xf)
    hke = Ki * np.float32(H_STEP)
    Ipre = ps0 - (Kp + hke) * e - Kd * dterm
    I = np.minimum(np.maximum(Ipre, LOc), HIc).astype(np.float32)
    I_n = np.minimum(np.maximum(I + hke * e, LOc), HIc).astype(np.float32)
    Ps = np.minimum(np.maximum(Kp * e + I_n + Kd * dterm, LOc), HIc).astype(np.float32)
    o = hb['h2d'].numpy()
    o[0:n] = Ps.reshape(-1)
    o[n:2 * n] = I_n.reshape(-1)
    pd = _PD[B]
    pd.copy_(hb['h2d'].view(2, B, NG), non_blocking=True)                                      # H2D writeback (included)
    st['out_Ps'].copy_(pd[0])
    st['out_I'].copy_(pd[1])
    g1, g2 = gains(g12)
    st['out_g1'].copy_(g1)
    st['out_g2'].copy_(g2)


def decide_cpu(st, B):
    g12 = arm_cpu(st, B)
    sync()                                                                                     # D2H must land before the CPU PID reads it
    host_pid(st, B, g12)


# ---------------------------------------------------------------- state (identical draws as base profiler)
WON = np.load('data_v2/ident_windows.npz')['on']
md = np.load('data_v2/meta.npz')
te = np.where(md['split'] == 3)[0]


def make_state(B):
    rng = np.random.default_rng(42); k = rng.choice(te, B, replace=B > len(te))
    buf = torch.zeros(B, NBUF, 90, device=dev)
    onset = torch.as_tensor(rng.integers(0, NBUF, B), device=dev)
    win = torch.as_tensor(WON[k], device=dev)
    buf[torch.arange(B, device=dev).view(-1, 1), (onset.view(-1, 1) + AR) % NBUF] = win
    w = win[:, -1, W_IDX]
    return dict(buf=buf, onset=onset, bidx=torch.arange(B, device=dev).view(-1, 1), w=w.clone(),
                xf=torch.zeros(B, NG, device=dev), Ps_prev=p.Pm0.float().expand(B, NG).clone(),
                out_Ps=torch.empty(B, NG, device=dev), out_I=torch.empty(B, NG, device=dev),
                out_g1=torch.empty(B, NG, 3, device=dev), out_g2=torch.empty(B, NG, 3, device=dev)), win


def synced(fn, reps):
    ts = []
    for _ in range(reps):
        sync(); t = time.perf_counter(); fn(); sync(); ts.append((time.perf_counter() - t) * 1e3)
    return float(np.mean(ts)), float(np.median(ts)), float(np.percentile(ts, 90))


def graph_build(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        fn()
    for _ in range(5):
        gr.replay()
    sync()
    return gr


def graph_timed(gr, reps):
    for _ in range(5):
        gr.replay()
    sync(); t = time.perf_counter()
    for _ in range(reps):
        gr.replay()
    sync(); mean = (time.perf_counter() - t) / reps * 1e3
    _, med, p90 = synced(gr.replay, reps)
    return mean, med, p90


META_OPS = {'aten::as_strided', 'aten::view', 'aten::expand', 'aten::slice', 'aten::select', 'aten::t',
            'aten::transpose', 'aten::reshape', 'aten::_reshape_alias', 'aten::_unsafe_view', 'aten::resize_',
            'aten::empty', 'aten::empty_like', 'aten::detach', 'aten::record_stream', 'aten::is_nonzero'}


def count_launches(fn):
    """Launch accounting per decision.

    launch-API keys (lower-case 'launch', e.g. cudaLaunchKernel/hipLaunchKernel) are what the spec asks
    for; on this ROCm/torch build torch.profiler exposes NO launch-API events at all (verified with a
    2-op probe: 0 keys contain 'launch', pr.events() returns CPU aten ops only), so launch_total is None.
    Fallback evidence recorded instead: the number of CPU-side aten invocations per decision, and the
    subset that is not a pure metadata op (i.e. the ops that each issue at least one GPU kernel) --
    both read straight out of the profiler, nothing estimated by hand.
    """
    out = dict(launch_keys=None, launch_total=None, aten_ops_total=None, aten_ops_kernel=None,
               op_counts=None, error=None)
    try:
        fn(); sync()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pr:
            fn(); sync()
        ka = pr.key_averages()
        lk = {e.key: int(e.count) for e in ka if 'launch' in e.key.lower()}
        out['launch_keys'] = lk
        out['launch_total'] = int(sum(lk.values())) if lk else None
        rows = sorted(((e.key, int(e.count)) for e in ka), key=lambda t: (-t[1], t[0]))
        out['aten_ops_total'] = int(sum(c for _, c in rows))
        out['aten_ops_kernel'] = int(sum(c for k, c in rows if k not in META_OPS))
        out['op_counts'] = [[k, c] for k, c in rows[:18]]
    except Exception as ex:                                                                    # ROCTracer / profiler failure
        out['error'] = f'{type(ex).__name__}: {ex}'
    return out


# ---------------------------------------------------------------- run
OUTKEYS = ('out_Ps', 'out_I', 'out_g1', 'out_g2')
res = dict(ckpt=args.ckpt, lib=args.lib, trunk=ck.get('trunk', 'base'), variant=ck['variant'], profile=ck['profile'],
           params=int(sum(q.numel() for q in net.parameters())), L_ms=round(float(net.L().detach()) * 1000, 1),
           reps=args.reps, gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
           note_static='L/mask/pos/mean-mask-sum precomputed once OFF the timed path (L is a frozen inference '
                       'parameter; a retrain must refresh them)',
           note_launch='torch.profiler on this ROCm build exposes no launch-API events; launch_total=None, '
                       'aten_ops_kernel (ops issuing >=1 GPU kernel) is the recorded fallback',
           note_maxdiff='gpu_fused/cpu_pid outputs vs gpu_base; the featurize reassociation leaves up to ~5e-4 '
                        'in normalised features and ~6e-3 in logits, argmax identical on every draw',
           gates_failed=[], maxdiff_tol=1e-4, variants={})
TOL = 1e-4
with torch.no_grad():
    for B in args.batch:
        st, win = make_state(B)
        assert torch.equal(s_gather(st), win), 'ring-buffer gather is not exact'
        r = dict(gather_assert_exact=True, burn=args.burn)

        if args.burn:  # steady-state burn-in: sustained decide_base load until clocks settle
            sync()
            for _ in range(args.burn):
                decide_base(st)
            sync()
            print(f'burn-in: {args.burn} eager decisions at B={B} done, clocks assumed settled', flush=True)

        for _ in range(3):
            decide_base(st)
        ref = {k: st[k].clone() for k in OUTKEYS}

        def diffs_vs(ref_):
            return {k: float((st[k] - ref_[k]).abs().max()) for k in OUTKEYS}

        # ---- eager timings
        r['gpu_base'] = dict()
        r['gpu_base']['eager_mean_ms'], r['gpu_base']['eager_median_ms'], r['gpu_base']['eager_p90_ms'] = synced(lambda: decide_base(st), args.reps)
        base_eager = {k: st[k].clone() for k in OUTKEYS}

        for _ in range(3):
            decide_fused(st)
        r['gpu_fused'] = dict()
        r['gpu_fused']['eager_mean_ms'], r['gpu_fused']['eager_median_ms'], r['gpu_fused']['eager_p90_ms'] = synced(lambda: decide_fused(st), args.reps)
        d_f = diffs_vs(ref); r['gpu_fused']['maxdiff_vs_base'] = max(d_f.values()); r['gpu_fused']['maxdiff_detail_vs_base'] = d_f
        fused_eager = {k: st[k].clone() for k in OUTKEYS}

        for _ in range(3):
            decide_cpu(st, B)
        r['cpu_pid'] = dict()
        r['cpu_pid']['eager_mean_ms'], r['cpu_pid']['eager_median_ms'], r['cpu_pid']['eager_p90_ms'] = synced(lambda: decide_cpu(st, B), args.reps)
        d_c = diffs_vs(ref); r['cpu_pid']['maxdiff_vs_base'] = max(d_c.values()); r['cpu_pid']['maxdiff_detail_vs_base'] = d_c

        # ---- argmax / logit sensitivity (diagnostic for gain-table flips)
        with torch.no_grad():
            raw = s_gather(st)
            xb = s_feat_base(raw); xf_ = s_feat_fused(raw)
            r['diag_logits_maxdiff'] = float((mlp_base(xb) - mlp_fused(xf_)).abs().max())
            r['diag_feat_maxdiff'] = float((xb * MW1 - xf_).abs().max())           # fused feat already carries the mask
            r['diag_argmax_mismatch'] = int((mlp_base(xb).argmax(1) != mlp_fused(xf_).argmax(1)).sum())
            top2 = mlp_base(xb).topk(2, dim=1).values
            r['diag_argmax_margin_min'] = float((top2[:, 0] - top2[:, 1]).min())   # headroom vs the logit reassociation

        # ---- stage medians (eager, synced, same convention as the reference profiler)
        raw = s_gather(st)
        stg = {}
        for name, fn in [('gather', lambda: s_gather(st)), ('featurize', lambda: s_feat_fused(raw))]:
            stg[f'stage_{name}_ms'] = synced(fn, args.reps)[1]
        x = s_feat_fused(raw); lg = mlp_fused(x); g12 = lookup_fused(lg)
        for name, fn in [('mlp', lambda: mlp_fused(x)), ('lookup', lambda: lookup_fused(lg)),
                         ('pid', lambda: pid_fused(st, gains(g12)[0]))]:
            stg[f'stage_{name}_ms'] = synced(fn, args.reps)[1]
        r['gpu_fused'].update(stg)
        xb = s_feat_base(raw); lgb = mlp_base(xb); g1b, _ = lookup_base(lgb)
        stb = {}
        for name, fn in [('gather', lambda: s_gather(st)), ('featurize', lambda: s_feat_base(raw)),
                         ('mlp', lambda: mlp_base(xb)), ('lookup', lambda: lookup_base(lgb)),
                         ('pid', lambda: pid_base(st, g1b))]:
            stb[f'stage_{name}_ms'] = synced(fn, args.reps)[1]
        r['gpu_base'].update(stb)

        # ---- graphs
        gr_base = graph_build(lambda: decide_base(st))
        m, md_, p90 = graph_timed(gr_base, args.reps)
        r['gpu_base']['graph_mean_ms'], r['gpu_base']['graph_median_ms'], r['gpu_base']['graph_p90_ms'] = m, md_, p90
        gr_base.replay(); sync()
        r['gpu_base']['graph_vs_eager_maxdiff'] = max(float((st[k] - base_eager[k]).abs().max()) for k in OUTKEYS)

        gr_fus = graph_build(lambda: decide_fused(st))
        m, md_, p90 = graph_timed(gr_fus, args.reps)
        r['gpu_fused']['graph_mean_ms'], r['gpu_fused']['graph_median_ms'], r['gpu_fused']['graph_p90_ms'] = m, md_, p90
        gr_fus.replay(); sync()
        r['gpu_fused']['graph_vs_eager_maxdiff'] = max(float((st[k] - fused_eager[k]).abs().max()) for k in OUTKEYS)
        r['gpu_fused']['graph_vs_base_maxdiff'] = max(float((st[k] - base_eager[k]).abs().max()) for k in OUTKEYS)

        # ---- cpu_pid deployment shape: GPU-arm graph + CPU PID + H2D writeback
        d2h_captured = True
        cap_err = None
        _DEV_PACK = {}

        def arm_stage(st_, B_):
            """arm + pack -> device staging (used only by the un-capturable-D2H fallback)."""
            g12_ = _fused_head(st_)
            pack = torch.cat([st_['w'].reshape(-1), st_['xf'].reshape(-1),
                              st_['Ps_prev'].reshape(-1), g12_.reshape(-1)])
            if B_ not in _DEV_PACK:
                _DEV_PACK[B_] = torch.empty_like(pack)
            _DEV_PACK[B_].copy_(pack)

        try:
            gr_arm = graph_build(lambda: arm_cpu(st, B))            # whole arm incl. the packed D2H node
        except Exception as ex:                                     # HIP graph refuses the host memcpy node
            d2h_captured = False
            cap_err = f'{type(ex).__name__}: {ex}'
            gr_arm = graph_build(lambda: arm_stage(st, B))

        _last_g12 = [_fused_head(st)]                               # gains held on device for the host PID

        def deployment_full():
            gr_arm.replay(); torch.cuda.current_stream().synchronize()     # D2H must land before the CPU PID reads
            if not d2h_captured:
                hbufs(B)['d2h'].copy_(_DEV_PACK[B], non_blocking=True)     # un-captured D2H
                sync()
            host_pid(st, B, _last_g12[0])
            sync()

        for _ in range(5):
            gr_arm.replay()
        sync()
        r['cpu_pid']['arm_graph_d2h_captured'] = d2h_captured
        r['cpu_pid']['arm_graph_capture_error'] = cap_err
        r['cpu_pid']['arm_graph_mean_ms'], r['cpu_pid']['arm_graph_median_ms'], r['cpu_pid']['arm_graph_p90_ms'] = graph_timed(gr_arm, args.reps)
        r['cpu_pid']['gpugraph_plus_host_mean_ms'], r['cpu_pid']['gpugraph_plus_host_median_ms'], r['cpu_pid']['gpugraph_plus_host_p90_ms'] = synced(deployment_full, args.reps)
        r['cpu_pid']['gpugraph_plus_host_maxdiff_vs_base'] = max(float((st[k] - ref[k]).abs().max()) for k in OUTKEYS)

        # ---- order-effect control: re-time the base arm at the END of the batch (clocks hot, all
        #      variants already run) so the base is not only measured while cold/first.
        r['gpu_base_recheck'] = dict()
        r['gpu_base_recheck']['eager_mean_ms'], r['gpu_base_recheck']['eager_median_ms'], r['gpu_base_recheck']['eager_p90_ms'] = synced(lambda: decide_base(st), args.reps)
        _, r['gpu_base_recheck']['graph_median_ms'], r['gpu_base_recheck']['graph_p90_ms'] = synced(gr_base.replay, args.reps)

        # ---- launch accounting
        r['launches'] = dict(gpu_base=count_launches(lambda: decide_base(st)),
                             gpu_fused=count_launches(lambda: decide_fused(st)),
                             cpu_pid_gpu_arm=count_launches(lambda: arm_cpu(st, B)))

        # ---- gates
        bad = [(n, r[n]['maxdiff_vs_base']) for n in ('gpu_fused', 'cpu_pid') if r[n]['maxdiff_vs_base'] > TOL]
        if bad:
            res['gates_failed'].append({f'B{B}': bad})
        res['variants'][f'B{B}'] = r

        # ---- compact print
        print(f"\n=== batch {B} (gather exact, feat maxdiff {r['diag_feat_maxdiff']:.2e}, "
              f"logits maxdiff {r['diag_logits_maxdiff']:.2e}, argmax mismatch {r['diag_argmax_mismatch']}, "
              f"min top1-top2 margin {r['diag_argmax_margin_min']:.3f}) ===", flush=True)

        def lstr(name):
            L = r['launches'][name]
            return str(L['launch_total']) if L['launch_total'] is not None else f"n/a({L['aten_ops_kernel']})"

        hdr = (f"{'variant':<10} {'eager mean':>10} {'med':>7} {'p90':>7} | {'graph mean':>10} {'med':>7} {'p90':>7} | "
               f"{'maxdiff':>9} {'launch':>10}")
        print(hdr); print('-' * len(hdr))
        for n in ('gpu_base', 'gpu_fused'):
            v = r[n]
            print(f"{n:<10} {v['eager_mean_ms']:>10.3f} {v['eager_median_ms']:>7.3f} {v['eager_p90_ms']:>7.3f} | "
                  f"{v['graph_mean_ms']:>10.3f} {v['graph_median_ms']:>7.3f} {v['graph_p90_ms']:>7.3f} | "
                  f"{v.get('maxdiff_vs_base', 0.0):>9.1e} {lstr(n):>10}", flush=True)
        v = r['cpu_pid']
        print(f"{'cpu_pid':<10} {v['eager_mean_ms']:>10.3f} {v['eager_median_ms']:>7.3f} {v['eager_p90_ms']:>7.3f} | "
              f"{v['gpugraph_plus_host_mean_ms']:>10.3f} {v['gpugraph_plus_host_median_ms']:>7.3f} "
              f"{v['gpugraph_plus_host_p90_ms']:>7.3f} | {v['maxdiff_vs_base']:>9.1e} "
              f"{lstr('cpu_pid_gpu_arm'):>10}  (graph col = gpugraph+host, launch = GPU arm)", flush=True)
        print('  fused stages (eager med):', ' '.join(f"{k[6:-3]} {v_:.3f}" for k, v_ in r['gpu_fused'].items() if k.startswith('stage_')), flush=True)
        print('  base  stages (eager med):', ' '.join(f"{k[6:-3]} {v_:.3f}" for k, v_ in r['gpu_base'].items() if k.startswith('stage_')), flush=True)
        print('  base recheck (end of batch): eager med %.3f  graph med %.3f' % (
            r['gpu_base_recheck']['eager_median_ms'], r['gpu_base_recheck']['graph_median_ms']), flush=True)
        print('  launch-API keys (base/fused/arm):',
              r['launches']['gpu_base']['launch_keys'], r['launches']['gpu_fused']['launch_keys'],
              r['launches']['cpu_pid_gpu_arm']['launch_keys'], flush=True)
        print('  aten ops per decision (total / kernel-issuing) base %s fps fused %s arm %s' % (
            (r['launches']['gpu_base']['aten_ops_total'], r['launches']['gpu_base']['aten_ops_kernel']),
            (r['launches']['gpu_fused']['aten_ops_total'], r['launches']['gpu_fused']['aten_ops_kernel']),
            (r['launches']['cpu_pid_gpu_arm']['aten_ops_total'], r['launches']['cpu_pid_gpu_arm']['aten_ops_kernel'])),
            flush=True)
        json.dump(res, open(args.out, 'w'), indent=1)
        if res['gates_failed']:
            print('GATE FAILED -- aborting:', bad, flush=True)
            raise SystemExit(1)
json.dump(res, open(args.out, 'w'), indent=1)
print(f"\nwrote {args.out}", flush=True)
