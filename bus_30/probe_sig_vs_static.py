# Direct test of the paper's claim: "static operating-point offsets dwarf the
# fault signature by <ratio>". Signature = value during an ACTIVE fault minus
# the scenario's pre-fault baseline; static = the baseline itself (the
# operating-point offset the encoder could latch onto). Featurized (88) channels.
import json, math, os
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
DD = os.path.join(BASE, "data30_mm")
X = np.load(os.path.join(DD, "X.npy"), mmap_mode="r")
N, T, N_CH = X.shape
md = np.load(os.path.join(DD, "meta.npz"))
failed = md["failed"].astype(bool)
t_clear = np.asarray(md["meta_t_clear"], float) if "meta_t_clear" in md.files else None

DT = 0.01
W_MAX_S = 1.00
W_STEPS = int(round(W_MAX_S / 0.01)); W_STRIDE = int(round(0.01 / DT))
LO = W_STEPS * W_STRIDE

NG = 6
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

rng = np.random.default_rng(42)
ok = np.where(~failed)[0]
perm = rng.permutation(ok)
idx_tr = perm[:int(0.90 * len(perm))]
fs = np.random.default_rng(0)
ks = fs.choice(idx_tr, size=min(256, len(idx_tr)), replace=False)

sig_acc = {n: [] for n, _ in GROUPS}
sta_acc = {n: [] for n, _ in GROUPS}
n_used = 0
for k in ks:
    tc = int(round(t_clear[k] / DT)) if t_clear is not None else None
    # pre-fault baseline: 32 steps in the window just before onset of sampling region
    pre_t = max(LO, 0)
    pre_t = np.arange(LO, LO + 32)
    feat = np.asarray(X[k][pre_t])
    # need the whole feature row over full run to find active-fault samples cheaply:
    row = np.asarray(X[k][LO:T])
    n_samples = 24
    ts = fs.integers(LO, T, size=n_samples)
    ts = ts[(ts - LO >= 0)]
    feats = np.asarray(X[k][ts])
    F = feats.reshape(-1, feats.shape[-1])
    b = F.mean(0, keepdims=True)
    Fc = F - b
    # fault-active mask: between onset and clearing
    on, cl = int(round(md["onset_s"][k] / DT)) if "onset_s" in md.files else None, None
    # fall back: y_t_fault / t_fault naming
    if on is None:
        tf = np.asarray(md["y_t_fault"], float)
        on = int(round(tf[k] / DT))
    cl = int(round(t_clear[k] / DT))
    act = (ts >= on) & (ts < cl)
    if act.sum() < 4:
        continue
    n_used += act.sum()
    Fv = featurize(np.asarray(X[k][ts]))          # (n_samples, 88)
    Fv_pre = featurize(np.asarray(X[k][pre_t]))   # baseline features
    base = Fv_pre.mean(0, keepdims=True)
    S = (Fv - base)[act]                          # signature rows (during fault)
    B = base[0]                                   # static offsets themselves
    o = 0
    for name, n in GROUPS:
        sig_acc[name].append(np.sqrt(np.mean(S[:, o:o+n] ** 2)))
        sta_acc[name].append(np.sqrt(np.mean(B[o:o+n] ** 2)))
        o += n

print(f"scenarios used={n_used} (active-fault instants)")
o = 0
for name, n in GROUPS:
    s = float(np.mean(sig_acc[name])); b = float(np.mean(sta_acc[name]))
    print(f"  {name:>10}: static(baseline) RMS={b:8.4f}  fault-signature RMS={s:8.4f}  static/sig={b/max(s,1e-12):6.1f}:1")
# aggregate across the static-heavy non-angle channels only
names = ["Pm", "Id", "Iq", "Vm"]
s_all = float(np.mean([np.mean(sig_acc[n]) for n in names]))
b_all = float(np.mean([np.mean(sta_acc[n]) for n in names]))
print(f"  static-heavy channels (Pm/Id/Iq/Vm) mean: static={b_all:.4f} sig={s_all:.4f} ratio={b_all/s_all:.1f}:1")
