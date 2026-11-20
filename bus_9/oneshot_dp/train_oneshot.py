"""
train_oneshot.py
================
Train the one-shot dP head on the FROZEN MLP encoder of the paper's MLP+MLP
checkpoint. Targets: the dataset's y_delta_p labels (no per-window RK4-teacher
dP was ever saved). Each training window ends at the label's own application
time t_apply = t_fault + 0.05 s. Scenarios with dp_status 0 carry dP = 0.

    python oneshot_dp/train_oneshot.py          (from bus_9)
"""
import argparse, json, os, time
import numpy as np
import torch

from oneshot_common import (load_teacher, featurize, raw_windows, OneShotHead,
                            DATA)

ap = argparse.ArgumentParser()
ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'run_label_tapply'))
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


@torch.no_grad()
def contexts(ks):
    out_c = []
    for i in range(0, len(ks), 512):
        kb = ks[i:i + 512]
        win = raw_windows(X, kb, t_apply[kb], W_STEPS, W_STRIDE, DT)
        feat, xm = featurize(win, len(kb))
        o = model.enc(torch.tensor(feat, device=dev))
        c, _ = OneShotHead.context(model, o, torch.tensor(xm, device=dev))
        out_c.append(c)
    return torch.cat(out_c)


tr = np.asarray(ck['split']['train']); va = np.asarray(ck['split']['val'])
t0 = time.time()
Ctr, Cva = contexts(tr), contexts(va)
Ytr = torch.tensor(y[tr], device=dev); Yva = torch.tensor(y[va], device=dev)
print(f'contexts: train {tuple(Ctr.shape)} val {tuple(Cva.shape)} '
      f'({time.time()-t0:.1f}s); nonzero labels train {(y[tr]!=0).any(1).mean():.3f}')

head = OneShotHead(hidden=args.hidden, layers=args.layers).to(dev)
head.mu.copy_(Ctr.mean(0)); head.sd.copy_(Ctr.std(0).clamp_min(1e-4))
opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
best, best_ep, hist = 1e9, -1, []
n = len(tr)
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
                    'target': 'y_delta_p @ t_apply'},
                   os.path.join(args.out, 'head.pt'))
    if ep % 200 == 0:
        print(f'ep {ep:5d}  train {ltr:.5f}  val {lva:.5f}  best {best:.5f}@{best_ep}', flush=True)
    if ep - best_ep > args.patience:
        break

ck_h = torch.load(os.path.join(args.out, 'head.pt'), map_location=dev, weights_only=False)
head.load_state_dict(ck_h['head']); head.eval()
with torch.no_grad():
    pva = head(Cva).cpu().numpy()
yv = y[va]
rmse = float(np.sqrt(((pva - yv) ** 2).sum(1).mean()))
base = float(np.sqrt((yv ** 2).sum(1).mean()))
corr = [float(np.corrcoef(pva[:, g], yv[:, g])[0, 1]) for g in range(3)]
res = dict(best_epoch=best_ep, val_mse=best, val_rmse_l2=rmse,
           val_rmse_zero_pred=base, val_corr_per_gen=corr,
           val_mean_l2_pred=float(np.linalg.norm(pva, axis=1).mean()),
           val_mean_l2_label=float(np.linalg.norm(yv, axis=1).mean()),
           n_train=int(n), n_val=int(len(va)))
json.dump(res, open(os.path.join(args.out, 'train_results.json'), 'w'), indent=1)
json.dump(hist, open(os.path.join(args.out, 'history.json'), 'w'))
print(json.dumps(res, indent=1))
