"""
needy_recount.py -- wiki S24/S38 chain item: needy-line re-count on the test split (quota retired in S18; the count is
the recorded number).

Needy = the S18 generalist (R0 = g18, no switching) leaves the envelope: excluded peak > LIMIT (S15 excluded scoring).
Counted from dataset v2 meta (gen_peak_excl) and confirmed by re-simulating the generalist on the counted cases.
For every needy line case (any split) and the near-miss line cases (generalist within 10 % of LIMIT) it reports the
line, the fault duration, and whether the oracle (library v2.3, switched at t_fault) and cls_learned_A (latched at
t_fault + 28 ms, library v2.3) hold the envelope. Non-test cases are in-sample for cls_learned_A (train/val).
Context rows: needy counts of the other test classes. Writes needy_recount.json; stdout -> needy_recount.log.
Usage:  python needy_recount.py [--lib library_v23.json] [--ckpt ident_ckpt_trunk/cls_learned_A.pt]
"""
import argparse
import json

import numpy as np
import torch

from ident_model import IdentNet, featurize
from plant_v2 import LIMIT
from sched_sim import make_plant, simulate_sched

ap = argparse.ArgumentParser()
ap.add_argument('--lib', default='library_v23.json')
ap.add_argument('--ckpt', default='ident_ckpt_trunk/cls_learned_A.pt')
args = ap.parse_args()

md = np.load('data_v2/meta.npz')
LIB = json.load(open(args.lib))['entries']
names = list(md['lib_names']); CFG = [n for n in names if n.startswith('R1')]
G = {n: np.array(LIB[n]['gains']) for n in names}; g18 = G['R0']
ty, loc, al, blk, sp = md['y_type'], md['meta_loc'], md['meta_alpha'], md['block'], md['split']
tf, tc, gpk = md['y_t_fault'], md['meta_t_clear'], md['gen_peak_excl']
SPLIT = {1: 'train', 2: 'val', 3: 'test'}
p = make_plant()


def family_r2(n):
    if n.startswith('R1d'):
        return None
    if n.startswith('R1_line'):
        return 'R2_line'
    if n.startswith('R1_load'):
        return 'R2_load'
    return 'R2_' + n.split('R1_')[1]


def scen(i):
    return dict(type={1: 'gen', 2: 'load', 3: 'line'}[int(ty[i])], loc=int(loc[i]), alpha=float(al[i]),
                t_fault=float(tf[i]), t_clear=(float(tc[i]) if blk[i] == 0 else float(tf[i]) + 1.0))


def sim(idx, g1, g2, L):
    S = [scen(i) for i in idx]
    t1 = tf[idx] + L; t2 = np.where(blk[idx] == 1, np.inf, np.maximum(np.where(blk[idx] == 1, np.inf, tc[idx]), t1))
    G3 = np.stack([np.broadcast_to(g18, g1.shape), g1, g2], 1)
    return np.asarray(simulate_sched(p, S, G3, t_sw=(t1, t2))['peak_excl'])


line = (blk == 0) & (ty == 3)
needy_line = np.where(line & (gpk > LIMIT))[0]
near_line = np.where(line & (gpk <= LIMIT) & (gpk > 0.9 * LIMIT))[0]
idx = np.concatenate([needy_line, near_line, np.where(line & (sp == 3))[0]])
idx = np.unique(idx)

# cls_learned_A predictions on these windows
ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
net = IdentNet(ck['variant'], ck['profile'], ck.get('trunk', 'base')).cuda(); net.load_state_dict(ck['state']); net.eval()
WON = np.load('data_v2/ident_windows.npz')['on']
x = torch.as_tensor((featurize(WON[idx]) - ck['feat_mu']) / ck['feat_sd'], dtype=torch.float32, device='cuda')
with torch.no_grad():
    c = net(x)['logits'].argmax(1).cpu().numpy()
L = round(float(net.L().detach()) * 1000) / 1000
pg1 = np.stack([G[CFG[k - 1]] if k > 0 else g18 for k in c])
pg2 = np.stack([G[family_r2(CFG[k - 1])] if (k > 0 and family_r2(CFG[k - 1])) else g18 for k in c])
tg1 = np.stack([G[names[i]] for i in md['lib_R1'][idx]])
tg2 = np.stack([G[names[i]] if i >= 0 else g18 for i in md['lib_R2'][idx]])
gen = sim(idx, np.broadcast_to(g18, tg1.shape).copy(), np.broadcast_to(g18, tg1.shape).copy(), 0.0)
orc = sim(idx, tg1, tg2, 0.0)
mlp = sim(idx, pg1, pg2, L)
true_c = np.array([CFG.index(names[i]) + 1 for i in md['lib_R1'][idx]])
pos = {i: k for k, i in enumerate(idx)}

print(f'library {args.lib}; cls_learned_A L = {L * 1000:.0f} ms; LIMIT {LIMIT}')
print(f'generalist re-simulated vs meta: max |diff| {np.abs(gen - gpk[idx]).max():.2e}')
res = dict(lib=args.lib, ckpt=args.ckpt, L_ms=L * 1000, limit=LIMIT, counts={}, cases=[])
for s in (3, 2, 1):
    m = line & (sp == s)
    res['counts'][f'line_{SPLIT[s]}'] = dict(n=int(m.sum()), needy=int((m & (gpk > LIMIT)).sum()),
                                             near_miss_10pct=int((m & (gpk <= LIMIT) & (gpk > 0.9 * LIMIT)).sum()))
res['counts']['line_all'] = dict(n=int(line.sum()), needy=int(len(needy_line)), near_miss_10pct=int(len(near_line)))
te = sp == 3
for nm, m in [('gen runback', (blk == 0) & (ty == 1)), ('load', (blk == 0) & (ty == 2)),
              ('alpha1 G1', (blk == 1) & (loc == 1)), ('alpha1 G2', (blk == 1) & (loc == 2))]:
    res['counts'][f'{nm} (test)'] = dict(n=int((m & te).sum()), needy=int((m & te & (gpk > LIMIT)).sum()))
for k, v in res['counts'].items():
    print(f'  {k:22s} {v}')
print('needy + near-miss line cases (peak_excl, pu):')
for kind, arr in (('needy', needy_line), ('near', near_line)):
    for i in arr:
        k = pos[i]
        row = dict(kind=kind, idx=int(i), split=SPLIT[int(sp[i])], line=int(loc[i]), dur_s=round(float(tc[i] - tf[i]), 4),
                   gen=float(gen[k]), oracle=float(orc[k]), cls_learned_A=float(mlp[k]), id_ok=bool(c[k] == true_c[k]),
                   oracle_in=bool(orc[k] <= LIMIT), mlp_in=bool(mlp[k] <= LIMIT))
        res['cases'].append(row)
        print(f"  {kind:5s} {row['split']:5s} line {row['line']:2d} dur {row['dur_s']:.3f}s  gen {row['gen']:.5f}  "
              f"oracle {row['oracle']:.5f} ({'in' if row['oracle_in'] else 'OUT'})  cls_learned_A {row['cls_learned_A']:.5f} "
              f"({'in' if row['mlp_in'] else 'OUT'}; ID {'ok' if row['id_ok'] else 'WRONG'})")
tl = np.array([pos[i] for i in np.where(line & te)[0]])
res['test_line_in_env'] = dict(generalist=float((gen[tl] <= LIMIT).mean()), oracle=float((orc[tl] <= LIMIT).mean()),
                               cls_learned_A=float((mlp[tl] <= LIMIT).mean()), max_generalist=float(gen[tl].max()))
print('test line in-envelope:', res['test_line_in_env'])
json.dump(res, open('needy_recount.json', 'w'), indent=1)
