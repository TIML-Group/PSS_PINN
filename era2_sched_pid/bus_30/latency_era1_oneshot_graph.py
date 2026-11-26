"""
time_oneshot_graph.py -- era-1 MLP encoder + one-shot dP head, single decision on GPU, eager vs whole-decision
CUDA graph (era-2 §32 protocol: window already on the GPU, featurize on GPU, batch 1, warm, host-synced per decision).

Decision = raw window (W x C, on GPU) -> featurize (+ normalise on 30-bus) + x_meas -> frozen MLP encoder ->
context [z, p, alpha, x0] -> one-shot head -> dP.  No RK4.  Repo files are imported read-only, nothing written there.
Numerics: GPU-featurize path vs the repo's numpy path (time30.py / time_neural.py), and graph vs eager.
Usage (from era2_sched_pid/bus_30):  python latency_era1_oneshot_graph.py --sys 30|9 --out latency_era1_oneshot_graph_<sys>.json  [--n 300]
"""
import argparse, json, os, sys, time
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--sys', choices=['30', '9'], required=True)
ap.add_argument('--n', type=int, default=300)
ap.add_argument('--warm', type=int, default=30)
ap.add_argument('--out', required=True)
args = ap.parse_args()
torch.set_grad_enabled(False)
dev = torch.device('cuda'); sync = torch.cuda.synchronize
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # PSS_PINN

if args.sys == '30':
    sys.path.insert(0, os.path.join(ROOT, 'bus_30', 'oneshot_dp30'))
    from common30 import TEACHER, DATA, BUS30, INSTANTS, OneShotHead30, load_head
    from case30_common import Setup, featurize as feat_np, D_IDX, W_IDX, PM_IDX, ID_IDX, IQ_IDX
    su = Setup(TEACHER, DATA, os.path.join(BUS30, 'case30_base.mat'), os.path.join(BUS30, 'case30_simulink_data.mat'), device=dev)
    model = su.model.eval()
    HEAD = 'run_novmask'
    head = load_head(os.path.join(ROOT, 'bus_30', 'oneshot_dp30', HEAD, 'head.pt'), dev)
    inst = json.load(open(INSTANTS)); ks = np.asarray(inst['ks'][:2048]); tk = np.asarray(inst['tk'][:2048])
    sel = np.random.default_rng(0).choice(2048, args.n + args.warm, replace=False)
    wins = []
    for i in sel:
        ie = int(round(tk[i] / su.DT)); i0 = ie - (su.W_STEPS - 1) * su.W_STRIDE
        wins.append(np.asarray(su.X[ks[i], i0:ie + 1:su.W_STRIDE], np.float32)[None])
    C = wins[0].shape[-1]; NG = len(D_IDX)
    VM = list(range(5 * NG, C, 2)); TH = list(range(5 * NG + 1, C, 2))
    MU = torch.as_tensor(np.asarray(su.FEAT_MU, np.float32), device=dev); SD = torch.as_tensor(np.asarray(su.FEAT_SD, np.float32), device=dev)
    ctx_fn = lambda o, xm: OneShotHead30.context(model, o, xm)

    def ref_np(win):   # time30.py path
        feat = (feat_np(win) - su.FEAT_MU) / su.FEAT_SD
        re = win[:, -1, :]; d = re[:, D_IDX]
        xm = np.concatenate([d - d[:, :1], re[:, W_IDX], re[:, PM_IDX]], 1)
        f = torch.from_numpy(feat.astype(np.float32)).to(dev); x = torch.from_numpy(xm.astype(np.float32)).to(dev)
        return head(ctx_fn(model.enc(f), x))

    def xmeas(win):
        re = win[:, -1, :]; d = re[:, Di]
        return torch.cat([d - d[:, :1], re[:, Wi], re[:, Pi]], 1)
    NORM = True
else:
    sys.path.insert(0, os.path.join(ROOT, 'bus_9', 'oneshot_dp'))
    from oneshot_common import load_teacher, featurize as feat9, raw_windows, OneShotHead, DATA, D_IDX, W_IDX, PM_IDX, ID_IDX, IQ_IDX, VM_IDX as VM, TH_IDX as TH
    model, ck, A, S = load_teacher(dev)
    HEAD = 'run_teacher_v3_novmask'
    hk = torch.load(os.path.join(ROOT, 'bus_9', 'oneshot_dp', HEAD, 'head.pt'), map_location=dev, weights_only=False)
    head = OneShotHead(hidden=hk['args']['hidden'], layers=hk['args']['layers']).to(dev); head.load_state_dict(hk['head']); head.eval()
    X = np.load(os.path.join(DATA, 'X.npy'), mmap_mode='r'); md = np.load(os.path.join(DATA, 'meta.npz'))
    DT = float(md['dt_solver']); W_STEPS = ck['w_steps']; W_STRIDE = int(round(A['dt_window'] / DT))
    inst = json.load(open(os.path.join(os.path.dirname(DATA), 'compare_out_s42', 'instants.json')))
    ks = np.asarray(inst['ks'][:2048]); tk = np.asarray(inst['tk'][:2048])
    sel = np.random.default_rng(0).choice(len(ks), args.n + args.warm, replace=False)
    wins = [raw_windows(X, ks[i:i + 1], tk[i:i + 1], W_STEPS, W_STRIDE, DT) for i in sel]
    ctx_fn = lambda o, xm: OneShotHead.context(model, o, xm)[0]

    def ref_np(win):   # time_neural.py path
        feat, xm = feat9(win, 1)
        return head(ctx_fn(model.enc(torch.from_numpy(feat).to(dev)), torch.from_numpy(xm).to(dev)))

    def xmeas(win):
        re = win[:, -1, :]; d = re[:, Di]
        return torch.cat([torch.zeros_like(d[:, :1]), d[:, 1:] - d[:, :1], re[:, Wi], re[:, Pi]], 1)
    NORM = False

Di, Wi, Pi, Ii, Qi, Vi, Ti = (torch.tensor(i, device=dev) for i in (D_IDX, W_IDX, PM_IDX, ID_IDX, IQ_IDX, VM, TH))


def feat_gpu(win):
    d, w, th = win[..., Di], win[..., Wi], win[..., Ti]
    f = torch.cat([d[..., 1:] - d[..., :1], (w - 1.0) / 0.01, win[..., Pi], win[..., Ii], win[..., Qi], win[..., Vi],
                   th[..., 1:] - th[..., :1]], -1)
    return (f - MU) / SD if NORM else f


def decide(win):
    return head(ctx_fn(model.enc(feat_gpu(win)), xmeas(win)))


W = [torch.as_tensor(w, device=dev) for w in wins]
static_in = W[0].clone()

# numerics: GPU featurize path vs repo numpy path
d_ref = max(float((decide(W[j]) - ref_np(wins[j])).abs().max()) for j in range(len(W)))

# eager, host-synced per decision
for j in range(args.warm):
    decide(W[j])
te = []
for j in range(args.warm, len(W)):
    static_in.copy_(W[j]); sync(); t = time.perf_counter(); decide(static_in); sync(); te.append((time.perf_counter() - t) * 1e3)

# whole decision as ONE CUDA graph
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        decide(static_in)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_out = decide(static_in)
for _ in range(5):
    g.replay()
tg, d_graph = [], 0.0
for j in range(args.warm, len(W)):
    static_in.copy_(W[j]); sync(); t = time.perf_counter(); g.replay(); sync(); tg.append((time.perf_counter() - t) * 1e3)
    d_graph = max(d_graph, float((static_out - decide(static_in)).abs().max()))
# back-to-back replay mean (era-1 profile_latency convention)
sync(); t = time.perf_counter()
for _ in range(100):
    g.replay()
sync(); bb = (time.perf_counter() - t) / 100 * 1e3

st = lambda v: dict(mean=float(np.mean(v)), median=float(np.median(v)), p90=float(np.percentile(v, 90)), p95=float(np.percentile(v, 95)))
res = dict(system=f'{args.sys}-bus', head=HEAD, encoder='era-1 MLP (frozen, MLP+MLP teacher)', batch=1, n=args.n, warm=args.warm,
           gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
           params=dict(encoder=sum(q.numel() for q in model.enc.parameters()), head=sum(q.numel() for q in head.parameters())),
           eager_ms=st(te), graph_ms=st(tg), graph_backtoback_mean_ms=bb,
           maxdiff_gpu_feat_vs_numpy_path=d_ref, maxdiff_graph_vs_eager=d_graph)
print(json.dumps(res, indent=1))
json.dump(res, open(args.out, 'w'), indent=1)
