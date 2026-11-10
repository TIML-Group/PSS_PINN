# Recompute the per-channel normalisation stats of bus_30/train_case30.py (feat_stats).
# Replicates: rng(0) for scenario/time draws, W_STEPS*W_STRIDE offset, featurize(),
# over the 90/5/5 train split regenerated with each seed (default 0; protocol 42).
import json, sys, os
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
DD = os.path.join(BASE, "data30_mm")
X = np.load(os.path.join(DD, "X.npy"), mmap_mode="r")
N, T, N_CH = X.shape
info = json.load(open(os.path.join(DD, "info.json")))
for k in info:
    print("info:", k, "=", info[k] if not isinstance(info[k], (list, dict)) else str(info[k])[:80])
DT = float(info.get("dt", info.get("DT", info.get("dt_meas", 0.01))))
W_MAX, DT_WIN = 1.00, 0.01
W_STEPS = int(round(W_MAX / DT_WIN))
W_STRIDE = int(round(DT_WIN / DT))
LO = W_STEPS * W_STRIDE
print(f"N={N} T={T} N_CH={N_CH} DT={DT} lo={LO}")

md = np.load(os.path.join(DD, "meta.npz"))
key_failed = [k for k in md.files if "fail" in k]
failed = md[key_failed[0]].astype(bool) if key_failed else np.zeros(N, bool)
print("failed channel:", key_failed, "| failed count:", int(failed.sum()))

NG = 6
NB = 30
idx_d = list(range(0, 2*NG, 2)); idx_w = list(range(1, 2*NG, 2)); idx_pm = list(range(2*NG, 3*NG))
idx_id = list(range(3*NG, 5*NG, 2)); idx_iq = list(range(3*NG+1, 5*NG, 2))
idx_vm = list(range(5*NG, N_CH, 2)); idx_th = list(range(5*NG+1, N_CH, 2))

def featurize(raw):
    d = raw[..., idx_d]; w = raw[..., idx_w]; pm = raw[..., idx_pm]; th = raw[..., idx_th]
    d_rel = d[..., 1:] - d[..., :1]; th_rel = th[..., 1:] - th[..., :1]
    return np.concatenate([d_rel, (w - 1.0) / 0.01, pm,
                           raw[..., idx_id], raw[..., idx_iq],
                           raw[..., idx_vm], th_rel], axis=-1)

GROUPS = [("ang_diff", len(idx_d)-1), ("omega/0.01", len(idx_w)), ("Pm", len(idx_pm)),
          ("Id", len(idx_id)), ("Iq", len(idx_iq)), ("Vm", len(idx_vm)), ("th_diff", len(idx_th)-1)]

def run(seed_split, tag):
    rng = np.random.default_rng(seed_split)
    ok = np.where(~failed)[0]
    perm = rng.permutation(ok)
    n_tr = int(0.90 * len(perm))
    idx_tr = perm[:n_tr]
    fs = np.random.default_rng(0)          # feat_stats default rng
    ks = fs.choice(idx_tr, size=min(256, len(idx_tr)), replace=False)
    acc = []
    for k in ks:
        ts = fs.integers(LO, T, size=32)
        acc.append(featurize(np.asarray(X[k][ts])))
    a = np.concatenate(acc, 0)
    tot_rms = np.sqrt(np.mean(a * a))
    tv_rms = np.sqrt(np.mean(a.var(0)))    # time-varying part (std, channel RMS-avg)
    static = np.sqrt(np.mean(a.mean(0) ** 2))
    print(f"\n[{tag}] rows={len(a)}  global RMS={tot_rms:.4g}  time-varying RMS={tv_rms:.4g}"
          f"  static RMS={static:.4g}")
    print(f"  tv/total = {tv_rms/tot_rms:.4f}  -> static drowns signal by {tot_rms/tv_rms:.1f}:1")
    o = 0
    for name, n in GROUPS:
        g = a[:, o:o+n]
        g_rms = np.sqrt(np.mean(g*g)); g_tv = np.sqrt(np.mean(g.var(0))); g_st = np.sqrt(np.mean(g.mean(0)**2))
        rel = g_tv / max(g_rms, 1e-12)
        print(f"    {name:>10}: rms={g_rms:8.4f} tv_rms={g_tv:8.4f} tv/rms={rel:.4f} static/sig={ (g_st/max(g_tv,1e-12)):.1f}")
        o += n
    # overall: per-channel 6%-style figure the comment quoted
    per = np.sqrt(a.var(0) / np.clip(np.mean(a*a, 0), 1e-12, None))
    print(f"  per-channel tv/rms: median={np.median(per):.4f} mean={per.mean():.4f} "
          f"p10={np.percentile(per,10):.4f} p90={np.percentile(per,90):.4f}")

run(0, "seed 0 (arg default)")
run(42, "seed 42 (protocol)")
