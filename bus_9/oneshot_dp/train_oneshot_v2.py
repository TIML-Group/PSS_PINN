"""
train_oneshot_v2.py
===================
v2 of the one-shot head: same frozen MLP encoder and head as train_oneshot.py,
but trained on the PER-TIME-STEP labels from step_labels/ (MATLAB optimizer,
"first action at this step"). Each sample is a window ending at a recorded
step t_i -- only measurements up to t_i -- with that step's label. Steps
outside [t_fault, t_clear) are dP = 0. Train/val = the checkpoint's train/val
split; the frozen seed-42 test draw (test split) is never touched.

Status-3 steps (no feasible dP within the rails) keep their best-effort dP as
the target by default (--drop_infeasible to exclude them).

    python oneshot_dp/train_oneshot_v2.py
"""
import argparse, json, os, time
import numpy as np
import torch

from oneshot_common import (load_teacher, featurize, raw_windows, OneShotHead,
                            DATA)

ap = argparse.ArgumentParser()
ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'run_step_labels'))
ap.add_argument('--labels', default=os.path.join(os.path.dirname(__file__), 'step_labels', 'step_labels.npz'))
ap.add_argument('--off_fault_per_sc', type=int, default=20, help='zero-label windows per scenario outside the fault')
ap.add_argument('--fault_stride', type=int, default=1, help='use every n-th fault step')
ap.add_argument('--drop_infeasible', action='store_true')
ap.add_argument('--epochs', type=int, default=4000)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--wd', type=float, default=1e-5)
ap.add_argument('--batch', type=int, default=1024)
ap.add_argument('--hidden', type=int, default=128)
ap.add_argument('--layers', type=int, default=2)
ap.add_argument('--patience', type=int, default=400)
ap.add_argument('--seed', type=int, default=42)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model, ck, A, S = load_teacher(dev)
for p_ in model.parameters():
    p_.requires_grad_(False)

X = np.load(os.path.join(DATA, 'X.npy'), mmap_mode='r')
md = np.load(os.path.join(DATA, 'meta.npz'))
DT = float(md['dt_solver'])
W_STEPS = ck['w_steps']; W_STRIDE = int(round(A['dt_window'] / DT))
t_apply = md['t_apply'].ravel(); y = md['y_delta_p'].astype(np.float32)
status = md['dp_status'].ravel()


L = np.load(args.labels)
Lk, Ls, Ldp, Lst = L['k'].astype(int) - 1, L['step'].astype(int) - 1, L['dp'], L['status']
t_grid = md['t']; t_fault = md['y_t_fault'].ravel(); t_clear = md['meta_t_clear'].ravel()


def samples(ks, rng):
    """(k, step index, target dP) for scenarios ks: every fault step (stride),
    plus off_fault_per_sc zero-label steps before the fault / after clearing."""
    sel = np.isin(Lk, ks)
    if args.fault_stride > 1:
        sel &= (Ls % args.fault_stride) == 0
    if args.drop_infeasible:
        sel &= Lst != 3
    K, I, Y = [Lk[sel]], [Ls[sel]], [Ldp[sel].astype(np.float32)]
    lo = int(np.ceil(0.3 / DT))                      # >= encoder window length
    for k in ks:
        off = np.where(((t_grid < t_fault[k] - 1e-9) | (t_grid >= t_clear[k] - 1e-9))
                       & (np.arange(len(t_grid)) >= lo))[0]
        pick = rng.choice(off, min(args.off_fault_per_sc, len(off)), replace=False)
        K.append(np.full(len(pick), k)); I.append(pick); Y.append(np.zeros((len(pick), 3), np.float32))
    return np.concatenate(K), np.concatenate(I), np.concatenate(Y)


@torch.no_grad()
def contexts(K, I):
    out_c = []
    for i in range(0, len(K), 2048):
        kb, ib = K[i:i + 2048], I[i:i + 2048]
        win = raw_windows(X, kb, t_grid[ib], W_STEPS, W_STRIDE, DT)
        feat, xm = featurize(win, len(kb))
        o = model.enc(torch.tensor(feat, device=dev))
        c, _ = OneShotHead.context(model, o, torch.tensor(xm, device=dev))
        out_c.append(c)
    return torch.cat(out_c)


tr = np.asarray(ck['split']['train']); va = np.asarray(ck['split']['val'])
rng = np.random.default_rng(args.seed)
Ktr, Itr, ytr = samples(tr, rng); Kva, Iva, yva = samples(va, rng)
t0 = time.time()
Ctr, Cva = contexts(Ktr, Itr), contexts(Kva, Iva)
Ytr = torch.tensor(ytr, device=dev); Yva = torch.tensor(yva, device=dev)
print(f'contexts: train {tuple(Ctr.shape)} val {tuple(Cva.shape)} ({time.time()-t0:.1f}s); '
      f'nonzero labels train {(ytr != 0).any(1).mean():.3f}', flush=True)
y = None

head = OneShotHead(hidden=args.hidden, layers=args.layers).to(dev)
head.mu.copy_(Ctr.mean(0)); head.sd.copy_(Ctr.std(0).clamp_min(1e-4))
opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
best, best_ep, hist = 1e9, -1, []
n = len(Ktr)
for ep in range(args.epochs):
    head.train()
    perm = torch.randperm(n, device=dev)
    for i in range(0, n, args.batch):
        b = perm[i:i + args.batch]
        loss = (head(Ctr[b]) - Ytr[b]).pow(2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    sched.step()
    head.eval()
    with torch.no_grad():
        ltr = float((head(Ctr) - Ytr).pow(2).sum(1).mean())
        lva = float((head(Cva) - Yva).pow(2).sum(1).mean())
    hist.append((ep, ltr, lva))
    if lva < best:
        best, best_ep = lva, ep
        torch.save({'head': head.state_dict(), 'epoch': ep, 'val_mse': lva,
                    'args': vars(args), 'teacher': os.path.relpath(
                        os.path.join(os.path.dirname(DATA), 'receding_fullmlp',
                                     'refined', 'best.pt'), os.path.dirname(DATA)),
                    'target': 'per-step MATLAB labels', 'labels': args.labels},
                   os.path.join(args.out, 'head.pt'))
    if ep % 200 == 0:
        print(f'ep {ep:5d}  train {ltr:.5f}  val {lva:.5f}  best {best:.5f}@{best_ep}', flush=True)
    if ep - best_ep > args.patience:
        break

ck_h = torch.load(os.path.join(args.out, 'head.pt'), map_location=dev, weights_only=False)
head.load_state_dict(ck_h['head']); head.eval()
with torch.no_grad():
    pva = head(Cva).cpu().numpy()
yv = yva
rmse = float(np.sqrt(((pva - yv) ** 2).sum(1).mean()))
base = float(np.sqrt((yv ** 2).sum(1).mean()))
corr = [float(np.corrcoef(pva[:, g], yv[:, g])[0, 1]) for g in range(3)]
res = dict(best_epoch=best_ep, val_mse=best, val_rmse_l2=rmse,
           val_rmse_zero_pred=base, val_corr_per_gen=corr,
           val_mean_l2_pred=float(np.linalg.norm(pva, axis=1).mean()),
           val_mean_l2_label=float(np.linalg.norm(yv, axis=1).mean()),
           n_train=int(n), n_val=int(len(Kva)))
json.dump(res, open(os.path.join(args.out, 'train_results.json'), 'w'), indent=1)
json.dump(hist, open(os.path.join(args.out, 'history.json'), 'w'))
print(json.dumps(res, indent=1))
