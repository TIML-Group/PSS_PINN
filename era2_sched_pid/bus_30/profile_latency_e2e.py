"""
profile_latency_e2e.py -- S31 era-2 end-to-end decision latency for the design of record (cls_learned).

Measured path (one decision, controller architecture only -- the 57 ms observation wait is NOT included, the
era-1 profile convention; observation cost lives in the S29 switching-delay table):
    onset trigger (onset index into the on-GPU 2 ms measurement ring buffer)
      -> window gather (128 x 90 raw channels, onset-aligned)
      -> era-1 featurize (88) + train-split normalisation
      -> IdentNet forward (bias-free projection 88->32, learned-window mask, 384x6 tanh MLP, heads)
      -> argmax + library lookup (R1 entry of the predicted configuration, R2 entry of its family;
         abstain -> R0 generalist)
      -> PID command update at the switch step: bumpless integrator preload (position form, clamped)
         + one PID step for all 6 machines -> P_set.
Protocol mirrors bus_9/bus_30 profile_latency.py: warm state (3 eager warm-ups; 5 graph warm replays), reps timed
decisions; eager = host-synced per decision (perf_counter, sync after each), graph = whole decision captured as ONE
CUDA graph, replayed reps times (mean, sync at end) plus per-replay synced median/p90. Per-stage synced
attribution in eager. Batch 1 is the operating mode; batch 256 printed for reference.

Run (from era2_sched_pid/bus_30/):  python profile_latency_e2e.py [--reps 100] [--batch 1 256]
"""
import argparse
import json
import time

import numpy as np
import torch

from ident_model import IdentNet, W_STEPS, D_IDX, W_IDX, PM_IDX, ID_IDX, IQ_IDX, VM_IDX, TH_IDX
from plant_v2 import NG, H_STEP, N_D
from sched_sim import make_plant

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='ident_ckpt/cls_learned.pt')
ap.add_argument('--reps', type=int, default=100)
ap.add_argument('--batch', type=int, nargs='+', default=[1, 256])
ap.add_argument('--out', default='profile_latency_e2e.json')
args = ap.parse_args()
dev = 'cuda'
sync = torch.cuda.synchronize

# ---------------------------------------------------------------- model + library --
ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
net = IdentNet(ck['variant'], ck['profile']); net.load_state_dict(ck['state']); net = net.to(dev).eval()
MU = torch.as_tensor(ck['feat_mu'], dtype=torch.float32, device=dev)
SD = torch.as_tensor(ck['feat_sd'], dtype=torch.float32, device=dev)
LIB = json.load(open('library_v21.json'))['entries']
names = list(np.load('data_v2/meta.npz')['lib_names']); CFG = [n for n in names if n.startswith('R1')]


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
TAB_R2 = torch.as_tensor(np.stack([g('R0')] + [g(family_r2(n)) for n in CFG]), device=dev)    # (59, NG, 3)
p = make_plant(); LO, HI = p.lo.float(), p.hi.float()
G0 = TAB_R1[0]                                                                                   # R0 gains in force

Di, Wi, Pi, Ii, Qi, Vi, Ti = (torch.tensor(i, device=dev) for i in (D_IDX, W_IDX, PM_IDX, ID_IDX, IQ_IDX, VM_IDX, TH_IDX))
AR = torch.arange(W_STEPS, device=dev)
NBUF = 2 * W_STEPS                                                                               # 512 ms ring buffer


# ---------------------------------------------------------------- stages --
def s_gather(st):
    return st['buf'][st['bidx'], (st['onset'].view(-1, 1) + AR) % NBUF]                          # (B,128,90)


def s_feat(raw):
    d, w, th = raw[..., Di], raw[..., Wi], raw[..., Ti]
    f = torch.cat([d[..., 1:] - d[..., :1], (w - 1.0) / 0.01, raw[..., Pi], raw[..., Ii], raw[..., Qi], raw[..., Vi],
                   th[..., 1:] - th[..., :1]], -1)
    return (f - MU) / SD


def s_mlp(x):
    return net(x)['logits']


def s_lookup(logits):
    c = logits.argmax(1)
    return TAB_R1[c], TAB_R2[c]


def s_pid(st, g1):
    """switch step R0 -> R1: bumpless preload of the integrator, then one PID step (all machines)."""
    Kp, Ki, Kd = g1[..., 0], g1[..., 1], g1[..., 2]
    e = 1.0 - st['w']; dterm = N_D * (e - st['xf'])
    Ipre = st['Ps_prev'] - Kp * e - Kd * dterm - H_STEP * Ki * e
    I = torch.minimum(torch.maximum(Ipre, LO), HI)
    I_n = torch.minimum(torch.maximum(I + H_STEP * Ki * e, LO), HI)
    Ps = torch.minimum(torch.maximum(Kp * e + I_n + Kd * dterm, LO), HI)
    return Ps, I_n


def decide(st):
    raw = s_gather(st); x = s_feat(raw); lg = s_mlp(x); g1, g2 = s_lookup(lg); Ps, I = s_pid(st, g1)
    st['out_Ps'].copy_(Ps); st['out_I'].copy_(I); st['out_g1'].copy_(g1); st['out_g2'].copy_(g2)


# ---------------------------------------------------------------- state (real test windows) --
WON = np.load('data_v2/ident_windows.npz')['on']
md = np.load('data_v2/meta.npz'); te = np.where(md['split'] == 3)[0]


def make_state(B):
    rng = np.random.default_rng(42); k = rng.choice(te, B, replace=B > len(te))
    buf = torch.zeros(B, NBUF, 90, device=dev)
    onset = torch.as_tensor(rng.integers(0, NBUF, B), device=dev)
    win = torch.as_tensor(WON[k], device=dev)
    buf[torch.arange(B, device=dev).view(-1, 1), (onset.view(-1, 1) + AR) % NBUF] = win
    w = win[:, -1, W_IDX]
    return dict(buf=buf, onset=onset, bidx=torch.arange(B, device=dev).view(-1, 1), w=w.clone(), xf=torch.zeros(B, NG, device=dev),
                Ps_prev=p.Pm0.float().expand(B, NG).clone(),
                out_Ps=torch.empty(B, NG, device=dev), out_I=torch.empty(B, NG, device=dev),
                out_g1=torch.empty(B, NG, 3, device=dev), out_g2=torch.empty(B, NG, 3, device=dev)), win


def synced(fn, reps):
    ts = []
    for _ in range(reps):
        sync(); t = time.perf_counter(); fn(); sync(); ts.append((time.perf_counter() - t) * 1e3)
    return float(np.mean(ts)), float(np.median(ts)), float(np.percentile(ts, 90))


res = dict(ckpt=args.ckpt, L_ms=round(float(net.L().detach()) * 1000, 1), reps=args.reps,
           gpu=torch.cuda.get_device_name(0), torch=torch.__version__)
with torch.no_grad():
    for B in args.batch:
        st, win = make_state(B)
        assert torch.equal(s_gather(st), win)                                  # ring-buffer gather is exact
        for _ in range(3):
            decide(st)
        r = {}
        # per-stage synced attribution (eager)
        raw = s_gather(st); x = s_feat(raw); lg = s_mlp(x); g1, _ = s_lookup(lg)
        for name, fn in [('gather', lambda: s_gather(st)), ('featurize', lambda: s_feat(raw)), ('mlp', lambda: s_mlp(x)),
                         ('lookup', lambda: s_lookup(lg)), ('pid_update', lambda: s_pid(st, g1))]:
            r[f'stage_{name}_ms'] = synced(fn, args.reps)[1]
        # whole decision eager
        r['eager_mean_ms'], r['eager_median_ms'], r['eager_p90_ms'] = synced(lambda: decide(st), args.reps)
        ref = {k: st[k].clone() for k in ('out_Ps', 'out_g1', 'out_g2')}
        # whole decision CUDA graph
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                decide(st)
        torch.cuda.current_stream().wait_stream(s)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            decide(st)
        for _ in range(5):
            gr.replay()
        sync(); t = time.perf_counter()
        for _ in range(args.reps):
            gr.replay()
        sync(); r['graph_mean_ms'] = (time.perf_counter() - t) / args.reps * 1e3
        _, r['graph_median_ms'], r['graph_p90_ms'] = synced(gr.replay, args.reps)
        r['graph_maxdiff'] = max(float((st[k] - ref[k]).abs().max()) for k in ref)
        res[f'B{B}'] = r
        print(f"batch {B}: eager {r['eager_mean_ms']:.3f} ms (median {r['eager_median_ms']:.3f}, p90 {r['eager_p90_ms']:.3f}) | "
              f"graph {r['graph_mean_ms']:.3f} ms (median {r['graph_median_ms']:.3f}, p90 {r['graph_p90_ms']:.3f}) | "
              f"max|diff| {r['graph_maxdiff']:.1e}", flush=True)
        print('   stages (eager, synced median):', ' '.join(f"{k[6:-3]} {v:.3f}" for k, v in r.items() if k.startswith('stage_')),
              flush=True)
json.dump(res, open(args.out, 'w'), indent=1)
