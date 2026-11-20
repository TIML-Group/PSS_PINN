"""
time_neural.py -- end-to-end single-decision latency, batch 1, warm.

Timed span: raw 33-channel window (numpy, already buffered) -> featurise ->
host-to-device -> encoder -> dP (one-shot head, or the teacher's full RK4
cascade) -> dP back on the host. torch.cuda.synchronize() before the start
stamp; the .cpu() at the end blocks until the action exists.
"""
import argparse, json, os, time
import numpy as np
import torch

from oneshot_common import (load_teacher, featurize, raw_windows, OneShotHead, DATA)

ap = argparse.ArgumentParser()
ap.add_argument('--head', default=os.path.join(os.path.dirname(__file__), 'run_label_tapply', 'head.pt'))
ap.add_argument('--n', type=int, default=300)
ap.add_argument('--warm', type=int, default=30)
ap.add_argument('--device', default='cuda')
ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'latency_neural.json'))
args = ap.parse_args()
dev = torch.device(args.device)

model, ck, A, S = load_teacher(dev)
hk = torch.load(args.head, map_location=dev, weights_only=False)
head = OneShotHead(hidden=hk['args']['hidden'], layers=hk['args']['layers']).to(dev)
head.load_state_dict(hk['head']); head.eval()

X = np.load(os.path.join(DATA, 'X.npy'), mmap_mode='r')
md = np.load(os.path.join(DATA, 'meta.npz'))
DT = float(md['dt_solver'])
W_STEPS = ck['w_steps']; W_STRIDE = int(round(A['dt_window'] / DT))
inst = json.load(open(os.path.join(os.path.dirname(DATA), 'compare_out_s42', 'instants.json')))
ks = np.asarray(inst['ks'][:2048]); tk = np.asarray(inst['tk'][:2048])
sel = np.random.default_rng(0).choice(len(ks), args.n + args.warm, replace=False)
wins = [raw_windows(X, ks[i:i + 1], tk[i:i + 1], W_STEPS, W_STRIDE, DT) for i in sel]
sync = (lambda: torch.cuda.synchronize()) if dev.type == 'cuda' else (lambda: None)


@torch.no_grad()
def oneshot(win):
    feat, xm = featurize(win, 1)
    f = torch.from_numpy(feat).to(dev); x = torch.from_numpy(xm).to(dev)
    o = model.enc(f)
    c, _ = OneShotHead.context(model, o, x)
    return head(c).cpu()


@torch.no_grad()
def encoder_only(win):
    feat, xm = featurize(win, 1)
    f = torch.from_numpy(feat).to(dev)
    return model.enc(f)['dP_raw'].cpu()


@torch.no_grad()
def teacher(win):
    feat, xm = featurize(win, 1)
    f = torch.from_numpy(feat).to(dev); x = torch.from_numpy(xm).to(dev)
    return model(f, x, hard=True, ctrl=True)['dP'].cpu()


res = {'device': str(dev), 'n': args.n, 'warm': args.warm,
       'gpu': torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'}
for name, fn in [('encoder_only', encoder_only), ('oneshot', oneshot), ('teacher_mlp_rk4', teacher)]:
    ts = []
    for j, w in enumerate(wins):
        sync(); t0 = time.perf_counter()
        fn(w)
        t1 = time.perf_counter()
        if j >= args.warm:
            ts.append((t1 - t0) * 1e3)
    ts = np.array(ts)
    res[name] = dict(median_ms=float(np.median(ts)), p95_ms=float(np.percentile(ts, 95)),
                     mean_ms=float(ts.mean()), max_ms=float(ts.max()))
    print(f'{name:16s} median {np.median(ts):8.3f} ms  p95 {np.percentile(ts,95):8.3f} ms', flush=True)
json.dump(res, open(args.out, 'w'), indent=1)
