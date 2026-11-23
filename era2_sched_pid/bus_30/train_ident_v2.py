"""
train_ident_v2.py -- Era-2 identification MLP (wiki training spec v1.0, 2026-11-22).

    python train_ident_v2.py --variant cls --profile 64        (fixed 64 ms window)
    python train_ident_v2.py --variant reg --profile learned   (learnable window, era-1 w_max mechanism)
    python train_ident_v2.py --variant cls --profile none      (no window limit: the full 256 ms span)

Data: data_v2 (11,000 scenarios, X recorded under R0 = generalist gains, no switching).
Decision protocol (training spec): latched, ONE switch per fault onset at t_d = t_fault + L, where L is the
profile's window; the identifier sees only measurements in [t_fault, t_d]. Onset is taken as known (trigger
instant), like the relay-supplied t_clear -- assumption recorded in the wiki.
Input: onset-aligned window of 256 ms at 2 ms (128 steps, every 2nd 1 ms sample) -> era-1 featurize (88
features: relative angles, (omega-1)/0.01, Pm, Id, Iq, Vm, relative theta), per-channel normalisation from
TRAIN windows only. A mask over elapsed time since onset hides samples after the decision instant:
hard for fixed profiles {32, 64, 128 ms, none = 256 ms}; learnable for 'learned' (sigmoid edge, length
L in (0, 256 ms], penalty LAM_WIN * L / L_max, beta annealed) -- the era-1 w_max mechanism moved to onset.
Pre-fault windows (2 per scenario, entirely before t_fault) are the abstain class (stay on R0).
Model: bias-free per-step projection (88 -> PROJ) so masked steps are exact zeros, flatten, era-1 MLPBlock
(HIDDEN x MLP_LAYERS, tanh), heads.
  V-CLS: classes = 0 (abstain) + the 58 R1/R1d library configurations; loss = class-weighted CE
         (weights n_c^-1/2, mean 1, thin line classes up-weighted) + LAM_A * L1(alpha) on gen-runback/load.
  V-REG: regresses log-gains of the R1/R1d entry AND of the R2 entry (2 x 18); loss = L1 on log-gains
         (R2 part masked for alpha = 1 and pre-fault windows; pre-fault R1 target = R0 gains).
Splits: data_v2 'split' (main 8000/1000/1000, alpha=1 300/100/100 per machine), frozen.
Adam, cosine LR 1e-3 -> 1e-5, batch 512, up to EPOCHS, early stop on val loss (patience). Checkpoint ->
ident_ckpt/<variant>_<profile>.pt
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'bus_30'))
from pinn_model30 import MLPBlock, HIDDEN, MLP_LAYERS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--variant', choices=['cls', 'reg'], required=True)
ap.add_argument('--profile', required=True, help="32 | 64 | 128 | none | learned")
ap.add_argument('--data', default='data_v2')
ap.add_argument('--epochs', type=int, default=300)
ap.add_argument('--patience', type=int, default=40)
ap.add_argument('--batch', type=int, default=512)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', default='ident_ckpt')
args = ap.parse_args()

# ---- pinned constants (changes = dated amendments, training spec) ----
DT_W, W_STEPS = 2e-3, 128             # 2 ms x 128 = 256 ms max window
L_MAX = DT_W * W_STEPS
PROJ = 32
LAM_A = 1.0                           # V-CLS alpha L1 weight
LAM_WIN = 0.05                        # learned-window penalty (era-1 lam_window default)
BETA0, BETA_MIN, L_INIT = 0.010, 0.002, 0.128
LR, LR_MIN = 1e-3, 1e-5
NG, NB = 6, 30

torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = 'cuda'
os.makedirs(args.out, exist_ok=True)
tag = f'{args.variant}_{args.profile}'
t0 = time.time()

# ---- data ----
md = np.load(os.path.join(args.data, 'meta.npz'))
X = np.load(os.path.join(args.data, 'X.npy'), mmap_mode='r')
N, T, C = X.shape
LIB = json.load(open('library_v21.json'))['entries']
names = list(md['lib_names'])
CFG = [n for n in names if n.startswith('R1')]                     # 58 R1 / R1d configurations
assert len(CFG) == 58, len(CFG)
cls = np.array([CFG.index(names[i]) + 1 for i in md['lib_R1']])      # 1..58
logg = {n: np.log(np.maximum(np.array(LIB[n]['gains']), 1e-6)).ravel() for n in names}
y_r1 = np.stack([logg[names[i]] for i in md['lib_R1']])               # (N, 18)
y_r2 = np.stack([logg[names[i]] if i >= 0 else logg['R0'] for i in md['lib_R2']])
m_r2 = (md['lib_R2'] >= 0)
alpha = md['meta_alpha'].astype(np.float32)
a_mask = ((md['y_type'] == 2) | ((md['y_type'] == 1) & (alpha < 1))).astype(np.float32)
split = md['split']
tf = md['y_t_fault']; i_f = np.round(tf / 1e-3).astype(int)
ST = int(round(DT_W / 1e-3))

cache = os.path.join(args.data, 'ident_windows.npz')
if os.path.exists(cache):
    z = np.load(cache); WON, WPRE = z['on'], z['pre']
else:
    rng = np.random.default_rng(1234)
    WON = np.empty((N, W_STEPS, C), np.float32); WPRE = np.empty((N, 2, W_STEPS, C), np.float32)
    span = W_STEPS * ST
    for k in range(N):
        WON[k] = X[k, i_f[k]:i_f[k] + span:ST]
        for r in range(2):
            s0 = int(rng.integers(100, i_f[k] - span))
            WPRE[k, r] = X[k, s0:s0 + span:ST]
    np.savez(cache, on=WON, pre=WPRE)
print(f'windows ready {WON.shape} ({time.time() - t0:.0f}s)', flush=True)

D_IDX = list(range(0, 2 * NG, 2)); W_IDX = list(range(1, 2 * NG, 2)); PM_IDX = list(range(2 * NG, 3 * NG))
ID_IDX = list(range(3 * NG, 5 * NG, 2)); IQ_IDX = list(range(3 * NG + 1, 5 * NG, 2))
VM_IDX = list(range(5 * NG, C, 2)); TH_IDX = list(range(5 * NG + 1, C, 2))


def featurize(raw):
    d = raw[..., D_IDX]; w = raw[..., W_IDX]; th = raw[..., TH_IDX]
    return np.concatenate([d[..., 1:] - d[..., :1], (w - 1.0) / 0.01, raw[..., PM_IDX], raw[..., ID_IDX],
                           raw[..., IQ_IDX], raw[..., VM_IDX], th[..., 1:] - th[..., :1]], -1).astype(np.float32)


F_ON = featurize(WON); F_PRE = featurize(WPRE)
tr = np.where(split == 1)[0]; va = np.where(split == 2)[0]; te = np.where(split == 3)[0]
allf = np.concatenate([F_ON[tr].reshape(-1, F_ON.shape[-1]), F_PRE[tr].reshape(-1, F_ON.shape[-1])])
MU = allf.mean(0); SD = allf.std(0); SD[SD < 1e-6] = 1.0
NF = F_ON.shape[-1]


def build(idx):
    """samples for scenario ids: onset windows (label = config) + 2 pre-fault windows (label = 0)."""
    on = (F_ON[idx] - MU) / SD; pre = (F_PRE[idx].reshape(-1, W_STEPS, NF) - MU) / SD
    xin = np.concatenate([on, pre])
    n, n2 = len(idx), 2 * len(idx)
    c = np.concatenate([cls[idx], np.zeros(n2, int)])
    a = np.concatenate([alpha[idx], np.zeros(n2, np.float32)]); am = np.concatenate([a_mask[idx], np.zeros(n2, np.float32)])
    r0 = np.tile(logg['R0'], (n2, 1))
    y1 = np.concatenate([y_r1[idx], r0]); y2 = np.concatenate([y_r2[idx], r0])
    m2 = np.concatenate([m_r2[idx], np.zeros(n2, bool)]).astype(np.float32)
    T_ = lambda v, dt=torch.float32: torch.as_tensor(v, dtype=dt, device=dev)
    return dict(x=T_(xin), c=T_(c, torch.long), a=T_(a), am=T_(am), y1=T_(y1), y2=T_(y2), m2=T_(m2))


D_tr, D_va = build(tr), build(va)
counts = np.bincount(cls[tr], minlength=59).astype(float); counts[0] = 2 * len(tr)
wc = np.where(counts > 0, counts ** -0.5, 0.0); wc = wc / wc[counts > 0].mean()
W_CLS = torch.as_tensor(wc, dtype=torch.float32, device=dev)


class IdentNet(nn.Module):
    def __init__(self, variant, profile):
        super().__init__()
        self.variant, self.profile = variant, profile
        self.proj = nn.Linear(NF, PROJ, bias=False); nn.init.xavier_normal_(self.proj.weight)
        self.trunk = MLPBlock(W_STEPS * PROJ, HIDDEN, MLP_LAYERS)
        if variant == 'cls':
            self.head_c = nn.Linear(HIDDEN, 59); self.head_a = nn.Linear(HIDDEN, 1)
        else:
            self.head_g = nn.Linear(HIDDEN, 36)
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.proj and m.bias is not None and m.out_features in (59, 1, 36):
                nn.init.xavier_normal_(m.weight, gain=0.1); nn.init.zeros_(m.bias)
        self.register_buffer('elapsed', (torch.arange(W_STEPS) + 1).float() * DT_W)   # end of each sample bin
        self.register_buffer('beta', torch.tensor(BETA0))
        if profile == 'learned':
            f = L_INIT / L_MAX
            self.raw_L = nn.Parameter(torch.tensor(float(np.log(f / (1 - f)))))
        else:
            self.register_buffer('L_fixed', torch.tensor(L_MAX if profile == 'none' else float(profile) / 1000))

    def L(self):
        return L_MAX * torch.sigmoid(self.raw_L) if self.profile == 'learned' else self.L_fixed

    def mask(self):
        if self.profile == 'learned':
            return torch.sigmoid((self.L() - self.elapsed + 0.5 * DT_W) / self.beta)
        return (self.elapsed <= self.L() + 1e-9).float()

    def forward(self, x):
        z = self.trunk(self.proj(x * self.mask().view(1, -1, 1)).flatten(1))
        if self.variant == 'cls':
            return dict(logits=self.head_c(z), alpha=torch.sigmoid(self.head_a(z).squeeze(-1)))
        return dict(logg=self.head_g(z))


def loss_fn(net, o, b):
    if net.variant == 'cls':
        ce = nn.functional.cross_entropy(o['logits'], b['c'], weight=W_CLS)
        la = (b['am'] * (o['alpha'] - b['a']).abs()).sum() / b['am'].sum().clamp(min=1)
        main = ce + LAM_A * la; parts = dict(ce=ce.item(), alpha_l1=la.item())
    else:
        l1 = (o['logg'][:, :18] - b['y1']).abs().mean(1)
        l2 = ((o['logg'][:, 18:] - b['y2']).abs().mean(1) * b['m2'])
        main = l1.mean() + l2.sum() / b['m2'].sum().clamp(min=1); parts = dict(r1=l1.mean().item())
    if net.profile == 'learned':
        main = main + LAM_WIN * net.L() / L_MAX
    return main, parts


@torch.no_grad()
def evaluate(net, D):
    net.eval(); tot, n = 0.0, 0; corr = 0
    for i in range(0, len(D['x']), 2048):
        b = {k: v[i:i + 2048] for k, v in D.items()}
        o = net(b['x']); l, _ = loss_fn(net, o, b); tot += l.item() * len(b['x']); n += len(b['x'])
        if net.variant == 'cls':
            corr += (o['logits'].argmax(1) == b['c']).sum().item()
    net.train()
    return tot / n, (corr / n if net.variant == 'cls' else None)


net = IdentNet(args.variant, args.profile).to(dev)
opt = torch.optim.Adam(net.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=LR_MIN)
best, best_ep, wait, hist = np.inf, -1, 0, []
ntr = len(D_tr['x'])
for ep in range(args.epochs):
    if args.profile == 'learned':
        net.beta.fill_(max(BETA_MIN, BETA0 * (BETA_MIN / BETA0) ** (ep / max(1, args.epochs * 0.6))))
    perm = torch.randperm(ntr, device=dev)
    for i in range(0, ntr, args.batch):
        j = perm[i:i + args.batch]; b = {k: v[j] for k, v in D_tr.items()}
        l, _ = loss_fn(net, net(b['x']), b)
        opt.zero_grad(); l.backward(); opt.step()
    sched.step()
    vl, vacc = evaluate(net, D_va)
    hist.append(dict(ep=ep, val=vl, val_acc=vacc, L=float(net.L())))
    if vl < best - 1e-5:
        best, best_ep, wait = vl, ep, 0
        BEST = dict(state={k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, variant=args.variant,
                    profile=args.profile, L=float(net.L()), feat_mu=MU, feat_sd=SD, cfg=CFG, lib_names=names, epoch=ep,
                    val=vl, val_acc=vacc, consts=dict(DT_W=DT_W, W_STEPS=W_STEPS, PROJ=PROJ, LAM_A=LAM_A,
                                                       LAM_WIN=LAM_WIN, HIDDEN=HIDDEN, MLP_LAYERS=MLP_LAYERS,
                                                       BETA_MIN=BETA_MIN), split=dict(train=tr, val=va, test=te))
    else:
        wait += 1
    if ep % 10 == 0 or wait == 0:
        print(f'{tag} ep {ep:3d} val {vl:.4f}' + (f' acc {vacc:.4f}' if vacc is not None else '') +
              f'  L {float(net.L()) * 1000:.1f} ms  best@{best_ep}  ({time.time() - t0:.0f}s)', flush=True)
    if args.patience and wait >= args.patience:
        break
# best checkpoint kept in memory during training and written ONCE at the end (repeated overwrites of the same
# .pt hit Windows error 1224 -- file briefly mapped by another process); temp file + replace, with retries
dst = os.path.join(args.out, f'{tag}.pt'); tmp = dst + '.tmp'
for attempt in range(10):
    try:
        torch.save(BEST, tmp); os.replace(tmp, dst); break
    except (RuntimeError, OSError) as e:
        print(f'save retry {attempt}: {e}', flush=True); time.sleep(2)
else:
    raise SystemExit(f'could not write {dst}')
json.dump(dict(tag=tag, best_epoch=best_ep, best_val=best, hist=hist, runtime_s=time.time() - t0),
          open(os.path.join(args.out, f'{tag}_hist.json'), 'w'), indent=1)
print(f'DONE {tag} best val {best:.4f} @ {best_ep}  ({time.time() - t0:.0f}s)', flush=True)
