"""
diagnose_dp.py
==============
Why is the dP rescue rate only ~54% when ~86% of out-of-limit cases are
correctable?

Three possible causes, and they are distinguishable. For the SAME frozen
identified dynamics, compare:

  (a) HEAD      dP straight from the feedforward head (one shot, no iteration)
  (b) ORACLE    dP found by direct gradient descent on the SAME objective
                (failure surrogate + violation + w_dp_norm * ||dP||^2)
  (c) FEASIBLE  dP found by gradient descent on peak deviation ALONE,
                with no norm penalty -- i.e. can ANY dP within +/-1 pu rescue
                this case at all

Read the result:

  (c) >> (b)              -> the norm penalty is suppressing rescue.
                             Objective balance problem; lower --w_dp_norm.
  (b) ~ (c), both >> (a)  -> a solution exists and the objective finds it, but
                             the head cannot produce it. Amortisation gap;
                             the head architecture or its inputs are the limit.
  (a) ~ (b) ~ (c)         -> genuinely infeasible; you are at the ceiling and
                             the remaining failures are the status-3 cases a
                             constant set-point shift cannot fix.

Also reports head output statistics (saturation, spread, correlation with the
oracle) which speak directly to whether the head has collapsed.

    python diagnose_dp.py --ckpt receding_output/best.pt --data data_mm
"""

import argparse, json, os

import numpy as np
import torch

from ieee9_physics import IEEE9, HYPOTHESES
from pinn_model import FaultEncoder, RolloutController

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--data', required=True, help='memmap dir from convert_dataset.py')
ap.add_argument('--base', default='ieee9_base.mat')
ap.add_argument('--sys', default='ieee9_gov_simulink_data.mat')
ap.add_argument('--split', default='test', choices=['val', 'test', 'train'])
ap.add_argument('--n', type=int, default=512, help='decision instants to test')
ap.add_argument('--steps', type=int, default=400, help='oracle optimiser steps')
ap.add_argument('--oracle_lr', type=float, default=0.05)
ap.add_argument('--seed', type=int, default=7)
ap.add_argument('--out', default=None)
args = ap.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ck = torch.load(args.ckpt, map_location=device, weights_only=False)
A = ck['args']
print(f"checkpoint: {args.ckpt}  (epoch {ck['epoch']}, window {ck['window_s']*1000:.1f} ms, "
      f"dP head '{ck['args'].get('dp_head','linear')}')")

# ── data ─────────────────────────────────────────────────────────────────────
X = np.load(os.path.join(args.data, 'X.npy'), mmap_mode='r')
md = np.load(os.path.join(args.data, 'meta.npz'))
DT = float(md['dt_solver']); t_grid = md['t']
lines = md['lines'].astype(int)
y_type = md['y_type'].ravel().astype(int)
meta_loc = md['meta_loc'].ravel().astype(int)
meta_alpha = md['meta_alpha'].ravel()
t_fault = md['y_t_fault'].ravel(); t_clear = md['meta_t_clear'].ravel()
dp_status = md['dp_status'].ravel().astype(int)
y_dp = md['y_delta_p'] if 'y_delta_p' in md else np.zeros((len(y_type), 3))
OMEGA_LIMIT = float(md['omega_limit'])
pool = np.asarray(ck['split'][args.split])
print(f"split '{args.split}': {len(pool)} scenarios")

# ── feature construction (must match train_receding.py) ─────────────────────
D_IDX, W_IDX, PM_IDX = [0, 2, 4], [1, 3, 5], [6, 7, 8]
ID_IDX, IQ_IDX = [9, 11, 13], [10, 12, 14]
VM_IDX = list(range(15, 33, 2)); TH_IDX = list(range(16, 33, 2))
FUT_IDX = [1, 3, 5, 6, 7, 8]

W_STEPS = ck['w_steps']
W_STRIDE = int(round(A['dt_window'] / DT))
ROLL_STRIDE = int(round(A['dt_roll'] / DT))
N_ROLL = int(round(A['horizon'] / A['dt_roll']))

TYPE_NAMES = ['gen', 'load', 'line']
sc_class = np.array([HYPOTHESES.index((TYPE_NAMES[y_type[k]-1], meta_loc[k])) + 1
                     for k in range(len(y_type))])


def build(ks, rng):
    ks = np.asarray(ks); bs = len(ks)
    lo, hi = A['w_max'], t_grid[-1] - A['horizon']
    tk = rng.uniform(lo, hi, bs)
    # only instants with an active fault are interesting for control, and
    # concentrated near onset -- matching the training distribution
    a = np.clip(t_fault[ks], lo, hi); b = np.clip(t_clear[ks], lo, hi)
    ow = A.get('onset_window', 0.0)
    if ow and ow > 0:
        b = np.minimum(b, a + ow)
    ok = b > a
    tk[ok] = rng.uniform(a[ok], b[ok])
    i_end = np.round(tk / DT).astype(int)

    win = np.empty((bs, W_STEPS, 33), np.float32)
    fut = np.empty((bs, N_ROLL + 1, 6), np.float32)
    for j, (k, ie) in enumerate(zip(ks, i_end)):
        i0 = ie - (W_STEPS - 1) * W_STRIDE
        blk = np.asarray(X[k, i0:ie + N_ROLL * ROLL_STRIDE + 1])
        o = ie - i0
        win[j] = blk[0:o + 1:W_STRIDE]
        fut[j] = blk[o::ROLL_STRIDE][:N_ROLL + 1][:, FUT_IDX]

    d = win[..., D_IDX]; w = win[..., W_IDX]; pm = win[..., PM_IDX]
    th = win[..., TH_IDX]
    feat = np.concatenate([d[..., 1:] - d[..., :1], (w - 1.0) / 0.01, pm,
                           win[..., ID_IDX], win[..., IQ_IDX],
                           win[..., VM_IDX], th[..., 1:] - th[..., :1]], -1)
    re = win[:, -1, :]; dd = re[:, D_IDX]
    x_meas = np.concatenate([np.stack([np.zeros(bs), dd[:, 1]-dd[:, 0],
                                       dd[:, 2]-dd[:, 0]], 1),
                             re[:, W_IDX], re[:, PM_IDX]], 1)

    active = (tk >= t_fault[ks]) & (tk < t_clear[ks])
    t_change = np.where(active, t_clear[ks],
                        np.where(tk < t_fault[ks], t_fault[ks], np.inf))
    valid_t = np.clip(t_change - tk, 0.0, A['horizon'])
    step_t = np.arange(N_ROLL + 1) * A['dt_roll']
    vmask = (step_t[None, :] <= valid_t[:, None]).astype(np.float32)
    return (torch.tensor(feat), torch.tensor(x_meas, dtype=torch.float32),
            torch.tensor(fut), torch.tensor(vmask), ks, tk, active)


# ── model ────────────────────────────────────────────────────────────────────
S = IEEE9(args.base, args.sys, dtype=torch.float32, device=device, lines=lines)
enc = FaultEncoder(ck['n_feat'], w_max_steps=W_STEPS, dt_window=A['dt_window'],
                   lstm_layers=A.get('lstm_layers', 10),
                   init_window=A['init_window'] / A['w_max'],
                   max_window=A['w_max'], beta0=A['beta0'])
model = RolloutController(S, enc, horizon=A['horizon'], dt_roll=A['dt_roll'],
                          omega_limit=OMEGA_LIMIT, margin=A.get('margin', 1e-4),
                          dp_head=A.get('dp_head', 'linear'),
                          dp_hidden=A.get('dp_hidden', 128),
                          dp_fc_layers=A.get('dp_fc_layers', 2),
                          n_refine=A.get('n_refine', 3)).to(device)
model.load_state_dict(ck['model']); model.eval()
enc.beta.fill_(A['beta_min'])

rng = np.random.default_rng(args.seed)
ks = rng.choice(pool, args.n, replace=True)
feat, x_meas, fut, vmask, ks, tk, active = build(ks, rng)
feat, x_meas, vmask = feat.to(device), x_meas.to(device), vmask.to(device)
fut = fut.to(device)

# ── frozen identified dynamics ──────────────────────────────────────────────
with torch.no_grad():
    o = model(feat, x_meas, hard=True, ctrl=True, vmask=vmask)
x0, Yred, Pset = o['x0'].detach(), o['Yred'].detach(), o['Pset'].detach()
dP_head = o['dP'].detach()
tgt = model.limit - model.margin

# which cases actually need rescue: measured future leaves the envelope.
# REACHABLE = omega still inside the envelope at the decision instant, so a
# set-point change can still act (the governor needs Tg = 0.2 s).
pk_true, has = model.judged(fut[:, :, 0:3], vmask)
dev_now = (x_meas[:, 3:6] - 1.0).abs().amax(dim=1)
act_t = torch.as_tensor(np.asarray(active), device=device)
reach = has & (dev_now <= model.limit) & act_t   # matches train/refine/verify
need_all = (pk_true > model.limit) & has
need = (pk_true > model.limit) & reach
print(f"instants: {args.n}   fault active: {int(active.sum())}   "
      f"needing rescue: {int(need_all.sum())}   of which reachable: "
      f"{int(need.sum())} ({100*int(need.sum())/max(int(need_all.sum()),1):.0f}%)")


def rescue_rate(dP, mask):
    with torch.no_grad():
        w = model.rollout(x0, Yred, Pset, dP=dP)
        pk, hv = model.judged(w, vmask)
        sel = mask & hv
        return (float((pk[sel] <= model.limit).float().mean()) if int(sel.sum())
                else float('nan')), pk


def optimise(w_norm, label):
    """Direct gradient descent on dP for each sample, model frozen."""
    raw = torch.zeros_like(dP_head, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=args.oracle_lr)
    for i in range(args.steps):
        opt.zero_grad()
        dP = model.dp_max * torch.tanh(raw)
        w = model.rollout(x0, Yred, Pset, dP=dP)
        pk, hv = model.judged(w, vmask)
        viol = model.violation(w, vmask)
        p_fail = torch.sigmoid((pk - tgt) / A['dp_tau'])
        rm = reach.float()
        loss = (((p_fail + A['w_viol'] * viol / model.limit) * rm).sum()
                / rm.sum().clamp(min=1)
                + w_norm * (dP ** 2).sum(1).mean())
        loss.backward(); opt.step()
        if (i + 1) % max(1, args.steps // 4) == 0:
            print(f"    {label}: step {i+1}/{args.steps}  loss {float(loss.detach()):.4f}")
    return (model.dp_max * torch.tanh(raw)).detach()


print("\noptimising oracles ...")
dP_obj = optimise(A['w_dp_norm'], 'objective')
dP_feas = optimise(0.0, 'feasibility')

r_head, _ = rescue_rate(dP_head, need)
r_obj, _ = rescue_rate(dP_obj, need)
r_feas, _ = rescue_rate(dP_feas, need)
nrm = lambda d, m: float(d[m].pow(2).sum(1).sqrt().mean()) if int(m.sum()) else float('nan')

print(f"\n{'='*70}")
print(f"RESCUE RATE on the {int(need.sum())} REACHABLE cases that need it")
print(f"{'='*70}")
print(f"  (a) HEAD      (feedforward, one shot) : {r_head:.4f}   "
      f"mean ||dP|| {nrm(dP_head, need):.4f} pu")
print(f"  (b) ORACLE    (same objective)        : {r_obj:.4f}   "
      f"mean ||dP|| {nrm(dP_obj, need):.4f} pu")
print(f"  (c) FEASIBLE  (no norm penalty)       : {r_feas:.4f}   "
      f"mean ||dP|| {nrm(dP_feas, need):.4f} pu")
lab = np.linalg.norm(y_dp[ks.astype(int)], axis=1)
print(f"      reference: mean ||y_delta_p||      = {lab[need.cpu().numpy()].mean():.4f} pu")

print(f"\n{'-'*70}\nDIAGNOSIS")
gap_a = (r_obj - r_head); gap_n = (r_feas - r_obj)
if gap_n > 0.10:
    print(f"  Norm penalty is costing {gap_n*100:.1f} points of rescue rate.")
    print(f"  -> OBJECTIVE BALANCE. Lower --w_dp_norm (currently {A['w_dp_norm']}).")
if gap_a > 0.10:
    print(f"  Oracle beats the head by {gap_a*100:.1f} points on the SAME objective.")
    print(f"  -> AMORTISATION GAP. The solution exists and the loss finds it, but")
    print(f"     the head cannot produce it. Head capacity / inputs are the limit.")
if gap_a <= 0.10 and gap_n <= 0.10:
    print(f"  Head is within {gap_a*100:.1f} points of the oracle, and removing the")
    print(f"  norm penalty adds only {gap_n*100:.1f}. -> NEAR THE FEASIBLE CEILING;")
    print(f"     remaining failures are cases a constant set-point shift cannot fix.")

# ── head output statistics ──────────────────────────────────────────────────
print(f"\n{'-'*70}\nHEAD OUTPUT (is it collapsed or saturated?)")
h = dP_head.cpu().numpy(); f_ = dP_feas.cpu().numpy()
print(f"  per-generator std across samples : {h.std(0).round(4)}")
print(f"  oracle per-gen std               : {f_.std(0).round(4)}")
print(f"  |dP| at the tanh rail (>0.95)    : {100*(np.abs(h)>0.95*model.dp_max).mean():.1f}%")
nd = need.cpu().numpy()
if nd.sum() > 2:
    cc = [np.corrcoef(h[nd, i], f_[nd, i])[0, 1] for i in range(3)]
    print(f"  corr(head, feasible oracle)      : {np.round(cc,3)}")
    print(f"  (low correlation = the head is not tracking what the case needs)")

if args.out or True:
    path = args.out or os.path.join(os.path.dirname(os.path.abspath(args.ckpt)),
                                    'dp_diagnosis.json')
    json.dump({'head': r_head, 'oracle_objective': r_obj, 'feasible': r_feas,
               'n_need': int(need.sum()), 'n_instants': int(args.n),
               'norm_head': nrm(dP_head, need), 'norm_oracle': nrm(dP_obj, need),
               'norm_feasible': nrm(dP_feas, need),
               'w_dp_norm': A['w_dp_norm'], 'epoch': int(ck['epoch'])},
              open(path, 'w'), indent=1)
    print(f"\n-> {path}")
