"""
diagnose_dp30.py
================
30-bus port of diagnose_dp.py. Is the dP head the bottleneck, or the physics?

For the SAME frozen identified dynamics, three dP sources are compared on the
decision instants that need rescue:

  (a) HEAD      dP emitted by the trained head
  (b) ORACLE    dP found by direct gradient descent on the SAME objective
                (failure surrogate + violation + w_dp_norm * ||dP||^2)
  (c) FEASIBLE  dP found by gradient descent on the envelope ALONE, no norm
                penalty -- can ANY dP within +/-1 pu rescue this case at all

  (c) >> (b)              -> the norm penalty is suppressing rescue
  (b) ~ (c), both >> (a)  -> amortisation gap: a solution exists, the head
                             cannot produce it
  (a) ~ (b) ~ (c)         -> at the feasibility ceiling

Added for the 30-bus system: the same three rates split by the dataset's
dp_status and by fault type. The 30-bus grid is actuator-limited -- only G1
and G2 have governors, and 16% of out-of-limit scenarios are dp_status 3 (no
dP within +/-1 pu fixes them, mostly large G1 trips). If the head's misses sit
in status 3 and FEASIBLE misses them too, the controller is at the physical
ceiling, not short of it.

All three are computed in the model's IDENTIFIED dynamics, so "feasible"
means feasible for the system the model believes it is looking at. With
identification at ~99% that is close to the truth, but a misidentified case
can look rescuable here and not be; verify_true30.py is the check against the
real system.

    python diagnose_dp30.py --ckpt case30_ctrl/best.pt --data data30_mm
"""

import argparse, json, os

import numpy as np
import torch

from case30_common import Setup, actionable, TYPE_NAMES, NG, N_ACT

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--data', required=True)
ap.add_argument('--base', default='case30_base.mat')
ap.add_argument('--sys', default='case30_simulink_data.mat')
ap.add_argument('--split', default='test', choices=['val', 'test', 'train'])
ap.add_argument('--n', type=int, default=512, help='decision instants to test')
ap.add_argument('--steps', type=int, default=400, help='oracle optimiser steps')
ap.add_argument('--oracle_lr', type=float, default=0.05)
ap.add_argument('--batch', type=int, default=512,
                help='instants per oracle batch (VRAM: the oracle holds a full '
                     'rollout graph per instant)')
ap.add_argument('--seed', type=int, default=7)
ap.add_argument('--out', default=None)
args = ap.parse_args()

su = Setup(args.ckpt, args.data, args.base, args.sys)
device, model, A = su.device, su.model, su.A
model.eval()
for q in model.parameters():
    q.requires_grad_(False)
pool = {'train': su.idx_tr, 'val': su.idx_va, 'test': su.idx_te}[args.split]
print(f"split '{args.split}': {len(pool)} scenarios")

rng = np.random.default_rng(args.seed)
ks_all = rng.choice(pool, args.n, replace=True)
tgt = model.limit - model.margin


def optimise(x0, Yred, Pset, vmask, reach, w_norm):
    """Direct gradient descent on dP per instant, model frozen."""
    raw = torch.zeros(x0.shape[0], N_ACT, device=device, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=args.oracle_lr)
    rm = reach.float()
    for _ in range(args.steps):
        opt.zero_grad()
        dP = model.dp_max * torch.tanh(raw)
        w = model.rollout(x0, Yred, Pset, dP=dP)
        pk, _ = model.judged(w, vmask)
        viol = model.violation(w, vmask)
        p_fail = torch.sigmoid((pk - tgt) / A['dp_tau'])
        loss = (((p_fail + A['w_viol'] * viol / model.limit) * rm).sum()
                / rm.sum().clamp(min=1) + w_norm * (dP ** 2).sum(1).mean())
        loss.backward(); opt.step()
    return (model.dp_max * torch.tanh(raw)).detach()


@torch.no_grad()
def peak_with(x0, Yred, Pset, vmask, dP):
    w = model.rollout(x0, Yred, Pset, dP=dP)
    return model.judged(w, vmask)[0]


rows = {k: [] for k in ('need', 'need_all', 'pk_head', 'pk_obj', 'pk_feas',
                        'dP_head', 'dP_obj', 'dP_feas', 'ks')}
print("\noptimising oracles ...")
for s0 in range(0, args.n, args.batch):
    ks = ks_all[s0:s0 + args.batch]
    b = su.to_dev(su.make_batch(ks, rng, onset_only=True))
    with torch.no_grad():
        o = model(b['feat'], b['x_meas'], hard=True, ctrl=True, vmask=b['vmask'])
    x0, Yred, Pset = o['x0'].detach(), o['Yred'].detach(), o['Pset'].detach()
    pk_true, has = model.judged(b['fut'][:, :, 0:NG], b['vmask'])
    reach = actionable(has, b['x_meas'], b['active'], model.limit)
    need = (pk_true > model.limit) & reach
    need_all = (pk_true > model.limit) & has
    dP_head = o['dP'].detach()
    with torch.enable_grad():
        dP_obj = optimise(x0, Yred, Pset, b['vmask'], reach, A['w_dp_norm'])
        dP_feas = optimise(x0, Yred, Pset, b['vmask'], reach, 0.0)
    for name, dP in (('head', dP_head), ('obj', dP_obj), ('feas', dP_feas)):
        rows[f'pk_{name}'].append(peak_with(x0, Yred, Pset, b['vmask'], dP).cpu())
        rows[f'dP_{name}'].append(dP.cpu())
    rows['need'].append(need.cpu()); rows['need_all'].append(need_all.cpu())
    rows['ks'].append(torch.as_tensor(b['ks']))
    print(f"  {min(s0 + args.batch, args.n)}/{args.n}")

R = {k: torch.cat(v) for k, v in rows.items()}
need = R['need'].numpy().astype(bool)
ks = R['ks'].numpy()
print(f"\ninstants: {args.n}   needing rescue: {int(R['need_all'].sum())}   "
      f"of which actionable: {int(need.sum())} "
      f"({100*need.sum()/max(int(R['need_all'].sum()),1):.0f}%)")


def rate(name, m):
    m = m & need
    return (float((R[f'pk_{name}'].numpy()[m] <= model.limit).mean())
            if m.sum() else float('nan')), int(m.sum())


def nrm(name, m):
    m = m & need
    d = R[f'dP_{name}'].numpy()[m]
    return float(np.linalg.norm(d, axis=1).mean()) if m.sum() else float('nan')


allm = np.ones_like(need)
r_head, n_need = rate('head', allm); r_obj, _ = rate('obj', allm)
r_feas, _ = rate('feas', allm)
print(f"\n{'='*70}")
print(f"RESCUE RATE on the {n_need} ACTIONABLE cases that need it")
print(f"{'='*70}")
print(f"  (a) HEAD      (trained head)          : {r_head:.4f}   "
      f"mean ||dP|| {nrm('head', allm):.4f} pu")
print(f"  (b) ORACLE    (same objective)        : {r_obj:.4f}   "
      f"mean ||dP|| {nrm('obj', allm):.4f} pu")
print(f"  (c) FEASIBLE  (no norm penalty)       : {r_feas:.4f}   "
      f"mean ||dP|| {nrm('feas', allm):.4f} pu")
lab = np.linalg.norm(su.y_dp[ks], axis=1)
print(f"      reference: mean ||y_delta_p||      = "
      f"{lab[need].mean() if need.sum() else float('nan'):.4f} pu")

# ── split by dp_status and fault type ────────────────────────────────────────
st = su.dp_status[ks]; ty = su.y_type[ks]
print(f"\n{'-'*70}\nBY DATASET LABEL (where do the misses live?)")
print(f"  {'group':34s} {'n':>5s} {'head':>7s} {'oracle':>7s} {'feasible':>9s}")
groups = [('dp_status 1  (correctable)', st == 1),
          ('dp_status 3  (not correctable)', st == 3),
          ('dp_status 0/2 (label says fine)', (st == 0) | (st == 2))]
groups += [(f'type {t}', ty == i + 1) for i, t in enumerate(TYPE_NAMES)]
by_group = {}
for label, m in groups:
    h, n = rate('head', m); ob, _ = rate('obj', m); fe, _ = rate('feas', m)
    by_group[label] = {'n': n, 'head': h, 'oracle': ob, 'feasible': fe}
    if n:
        print(f"  {label:34s} {n:5d} {h:7.3f} {ob:7.3f} {fe:9.3f}")
miss_head = need & (R['pk_head'].numpy() > model.limit)
miss_feas = need & (R['pk_feas'].numpy() > model.limit)
if miss_head.sum():
    print(f"\n  head misses: {int(miss_head.sum())}, of which "
          f"{int((miss_head & miss_feas).sum())} are ALSO missed by the "
          f"feasibility oracle ({100*(miss_head & miss_feas).sum()/miss_head.sum():.0f}%)"
          f" -> beyond any dP in +/-{model.dp_max:g} pu")
    print(f"  head misses in dp_status 3: {int((miss_head & (st == 3)).sum())}")

print(f"\n{'-'*70}\nDIAGNOSIS")
gap_a = r_obj - r_head; gap_n = r_feas - r_obj
if gap_n > 0.10:
    print(f"  Norm penalty is costing {gap_n*100:.1f} points of rescue rate.")
    print(f"  -> OBJECTIVE BALANCE. Lower --w_dp_norm (currently {A['w_dp_norm']}).")
if gap_a > 0.10:
    print(f"  Oracle beats the head by {gap_a*100:.1f} points on the SAME objective.")
    print(f"  -> AMORTISATION GAP: the head cannot produce a solution that exists.")
if gap_a <= 0.10 and gap_n <= 0.10:
    print(f"  Head is within {gap_a*100:.1f} points of the oracle; removing the norm")
    print(f"  penalty adds {gap_n*100:.1f}. -> NEAR THE FEASIBLE CEILING for this")
    print(f"  actuator (G1, G2 within +/-{model.dp_max:g} pu).")

print(f"\n{'-'*70}\nHEAD OUTPUT (collapsed or saturated?)")
h = R['dP_head'].numpy(); f_ = R['dP_feas'].numpy()
print(f"  per-machine std (G1, G2)         : {h.std(0).round(4)}")
print(f"  oracle per-machine std           : {f_.std(0).round(4)}")
print(f"  |dP| at the tanh rail (>0.95)    : "
      f"{100*(np.abs(h) > 0.95*model.dp_max).mean():.1f}%")
print(f"  feasible oracle at the rail      : "
      f"{100*(np.abs(f_[need]) > 0.95*model.dp_max).mean() if need.sum() else float('nan'):.1f}%"
      f"  (high = actuator-limited)")
if need.sum() > 2:
    cc = [np.corrcoef(h[need, i], f_[need, i])[0, 1] for i in range(N_ACT)]
    print(f"  corr(head, feasible oracle)      : {np.round(cc, 3)}")

path = args.out or os.path.join(os.path.dirname(os.path.abspath(args.ckpt)),
                                'dp_diagnosis.json')
json.dump({'head': r_head, 'oracle_objective': r_obj, 'feasible': r_feas,
           'n_need': n_need, 'n_instants': int(args.n),
           'norm_head': nrm('head', allm), 'norm_oracle': nrm('obj', allm),
           'norm_feasible': nrm('feas', allm), 'by_group': by_group,
           'w_dp_norm': A['w_dp_norm'], 'epoch': int(su.ck['epoch'])},
          open(path, 'w'), indent=1)
print(f"\n-> {path}")
