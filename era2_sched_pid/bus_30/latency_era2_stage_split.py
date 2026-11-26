"""
era2_stage_split.py -- era-2 final controller (fused cls_learned_A, library v2.3) per-stage latency, batch 1,
on GPU (eager + per-segment CUDA graphs) and on CPU (eager).  The fused stage functions are taken VERBATIM from the
committed profile_latency_e2e_opt.py (source exec'd up to its state section, with only the device switched);
the repo file is not modified.
Segments: encoder = gather + featurize + IdentNet forward; lookup = argmax + T12 gain lookup; pid = bumpless PID step.
Usage (from era2_sched_pid/bus_30):  python latency_era2_stage_split.py --device cuda|cpu --out latency_era2_stage_split_<dev>.json [--reps 300] [--burn 3000]
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.getcwd())
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--device', choices=['cuda', 'cpu'], required=True)
ap.add_argument('--reps', type=int, default=300)
ap.add_argument('--burn', type=int, default=3000)
ap.add_argument('--threads', type=int, default=0)
ap.add_argument('--out', required=True)
a = ap.parse_args()
if a.threads:
    torch.set_num_threads(a.threads)
torch.set_grad_enabled(False)

src = open('profile_latency_e2e_opt.py', encoding='utf-8').read()
src = src[:src.index('# ---------------------------------------------------------------- state (identical draws')]
for old, new in [("dev = 'cuda'", f"dev = '{a.device}'"), ("p = make_plant()", "p = make_plant(dev)")]:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
if a.device == 'cpu':
    src = src.replace("sync = torch.cuda.synchronize", "sync = lambda: None")
sys.argv = ['x', '--batch', '1']
NS = {'__name__': 'opt_src'}
exec(compile(src, 'profile_latency_e2e_opt.py', 'exec'), NS)
dev, NG, NBUF, AR, W_IDX, p = NS['dev'], NS['NG'], NS['NBUF'], NS['AR'], NS['W_IDX'], NS['p']
sync = NS['sync']

# state: same seed-42 test draw as the committed profilers, batch 1
WON = np.load('data_v2/ident_windows.npz')['on']; md = np.load('data_v2/meta.npz'); te = np.where(md['split'] == 3)[0]
rng = np.random.default_rng(42); k = rng.choice(te, 1)
buf = torch.zeros(1, NBUF, 90, device=dev); onset = torch.as_tensor(rng.integers(0, NBUF, 1), device=dev)
win = torch.as_tensor(WON[k], device=dev)
buf[torch.arange(1, device=dev).view(-1, 1), (onset.view(-1, 1) + AR) % NBUF] = win
st = dict(buf=buf, onset=onset, bidx=torch.arange(1, device=dev).view(-1, 1), w=win[:, -1, W_IDX].clone(),
          xf=torch.zeros(1, NG, device=dev), Ps_prev=p.Pm0.float().to(dev).expand(1, NG).clone(),
          out_Ps=torch.empty(1, NG, device=dev), out_I=torch.empty(1, NG, device=dev),
          out_g1=torch.empty(1, NG, 3, device=dev), out_g2=torch.empty(1, NG, 3, device=dev))

s_gather, s_feat, mlp, lookup, gains, pid = (NS[n] for n in ('s_gather', 's_feat_fused', 'mlp_fused', 'lookup_fused', 'gains', 'pid_fused'))
decide = NS['decide_fused']

# static intermediates so each segment is a self-contained callable
S = {}
S['lg'] = mlp(s_feat(s_gather(st))).clone(); S['g12'] = lookup(S['lg']).clone()


def seg_encoder():
    S['lg'].copy_(mlp(s_feat(s_gather(st))))


def seg_lookup():
    S['g12'].copy_(lookup(S['lg']))


def seg_pid():
    g1, g2 = gains(S['g12']); Ps, I = pid(st, g1)
    st['out_Ps'].copy_(Ps); st['out_I'].copy_(I); st['out_g1'].copy_(g1); st['out_g2'].copy_(g2)


def whole():
    decide(st)


def synced(fn, reps):
    ts = []
    for _ in range(reps):
        sync(); t = time.perf_counter(); fn(); sync(); ts.append((time.perf_counter() - t) * 1e3)
    return dict(mean=float(np.mean(ts)), median=float(np.median(ts)), p90=float(np.percentile(ts, 90)))


# numerics: segmented path == whole decision
whole(); ref = st['out_Ps'].clone(); seg_encoder(); seg_lookup(); seg_pid()
d_seg = float((st['out_Ps'] - ref).abs().max())

for _ in range(a.burn):
    whole()
SEGS = [('encoder', seg_encoder), ('lookup', seg_lookup), ('pid', seg_pid), ('whole', whole)]
# fine split inside the encoder (eager)
raw = s_gather(st); x = s_feat(raw)
FINE = [('gather', lambda: s_gather(st)), ('featurize', lambda: s_feat(raw)), ('identnet', lambda: mlp(x))]
res = dict(device=a.device, name=(torch.cuda.get_device_name(0) if a.device == 'cuda' else 'cpu'),
           threads=torch.get_num_threads(), torch=torch.__version__, batch=1, reps=a.reps, burn=a.burn,
           ckpt='ident_ckpt_trunk/cls_learned_A.pt', lib='library_v23.json', maxdiff_segmented_vs_whole=d_seg,
           eager={n: synced(f, a.reps) for n, f in SEGS + FINE})
if a.device == 'cuda':
    def gb(fn):
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
        return gr
    G = {n: gb(f) for n, f in SEGS}
    res['graph'] = {n: synced(G[n].replay, a.reps) for n, _ in SEGS}
    whole(); e_out = st['out_Ps'].clone(); G['whole'].replay(); sync()
    res['maxdiff_graph_vs_eager'] = float((st['out_Ps'] - e_out).abs().max())
print(json.dumps(res, indent=1))
json.dump(res, open(a.out, 'w'), indent=1)
