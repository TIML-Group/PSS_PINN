"""
eval_ident_v2.py -- Era-2 training spec v1.0 evaluation: every checkpoint in ident_ckpt/ on the TEST split
(1,000 main + 200 alpha = 1), in true dynamics (sched_sim, CUDA).

Scheduled run per test scenario: R0 = S18 generalist until the latched decision t_d = t_fault + L; then
  V-CLS: the library entry of the predicted configuration (abstain -> stay on R0); R2 = the library R2 entry
         of the predicted family (R0 if abstained or a disconnection was predicted);
  V-REG: the predicted gains (exp of the log-gain output, clipped to the library's per-gain range).
R2 switch at the relay-supplied t_clear (no clearing for alpha = 1); if the decision comes after t_clear the
scenario goes straight to R2 at t_d. References on the same scenarios: generalist (meta gen_*), oracle
(library, switched at t_fault: meta orc_*), and the DELAYED oracle (library gains, switched at t_d) whose
gap to the oracle is the switching-delay cost of the profile.
Acceptance bars (pre-registered, per class): (1) never worse than the generalist (median peak and
in-envelope rate), (2) alpha = 1 G1 rescue = oracle's (100 %), (3) median peak within 10 % of the oracle's,
(4) headline = generalist -> scheduled median-peak distance. Diagnostics: configuration accuracy, line
confusion, alpha-MAE, switching-delay cost, wrong-schedule cost (misidentified cases: scheduled vs delayed
oracle), effort (TV, survivors' rail-binding, bump). Writes ident_eval.json and ident_eval.md.
"""
import glob
import json
import os

import numpy as np
import torch

from ident_model import IdentNet, featurize, W_STEPS
from sched_sim import make_plant, simulate_sched
from plant_v2 import LIMIT, NG

md = np.load('data_v2/meta.npz')
WON = np.load('data_v2/ident_windows.npz')['on']
LIB = json.load(open('library_v21.json'))['entries']
names = list(md['lib_names']); CFG = [n for n in names if n.startswith('R1')]
G = {n: np.array(LIB[n]['gains']) for n in names}; g18 = G['R0']
allg = np.stack([G[n] for n in names]); GLO, GHI = allg.min(0), allg.max(0)
te = np.where(md['split'] == 3)[0]
ty, loc, al, blk = md['y_type'][te], md['meta_loc'][te], md['meta_alpha'][te], md['block'][te]
tf, tc = md['y_t_fault'][te], md['meta_t_clear'][te]
true_cls = np.array([CFG.index(names[i]) + 1 for i in md['lib_R1'][te]])
true_r2 = md['lib_R2'][te]
GROUPS = {'gen runback': (blk == 0) & (ty == 1), 'load': (blk == 0) & (ty == 2), 'line': (blk == 0) & (ty == 3),
          'alpha1 G1': (blk == 1) & (loc == 1), 'alpha1 G2': (blk == 1) & (loc == 2)}
S = [dict(type={1: 'gen', 2: 'load', 3: 'line'}[int(ty[i])], loc=int(loc[i]), alpha=float(al[i]), t_fault=float(tf[i]),
          t_clear=(float(tc[i]) if blk[i] == 0 else float(tf[i]) + 1.0)) for i in range(len(te))]
gen_pk, orc_pk = md['gen_peak_excl'][te], md['orc_peak_excl'][te]
p = make_plant()


def family_r2(c):
    n = CFG[c - 1]
    if n.startswith('R1d'):
        return None
    if n.startswith('R1_line'):
        return 'R2_line'
    if n.startswith('R1_load'):
        return 'R2_load'
    return 'R2_' + n.split('R1_')[1]


def predict(ck):
    net = IdentNet(ck['variant'], ck['profile']).cuda(); net.load_state_dict(ck['state']); net.eval()
    x = torch.as_tensor((featurize(WON[te]) - ck['feat_mu']) / ck['feat_sd'], dtype=torch.float32, device='cuda')
    with torch.no_grad():
        o = net(x)
    L = round(float(net.L().detach()) * 1000) / 1000
    if ck['variant'] == 'cls':
        c = o['logits'].argmax(1).cpu().numpy(); a = o['alpha'].cpu().numpy()
        g1 = np.stack([G[CFG[k - 1]] if k > 0 else g18 for k in c])
        g2 = np.stack([G[family_r2(k)] if (k > 0 and family_r2(k)) else g18 for k in c])
        return dict(L=L, cls=c, alpha=a, g1=g1, g2=g2)
    lg = o['logg'].cpu().numpy()
    g1 = np.clip(np.exp(lg[:, :18]).reshape(-1, NG, 3), GLO, GHI); g2 = np.clip(np.exp(lg[:, 18:]).reshape(-1, NG, 3), GLO, GHI)
    return dict(L=L, g1=g1, g2=g2)


def run(g1, g2, L):
    t1 = tf + L; t2 = np.where(blk == 1, np.inf, np.maximum(tc_safe, t1))
    G3 = np.stack([np.broadcast_to(g18, g1.shape), g1, g2], 1)
    return simulate_sched(p, S, G3, t_sw=(t1, t2))


tc_safe = np.where(blk == 1, np.inf, tc)
true_g1 = np.stack([G[names[i]] for i in md['lib_R1'][te]])
true_g2 = np.stack([G[names[i]] if i >= 0 else g18 for i in true_r2])
res, delayed_cache = {}, {}
for f in sorted(glob.glob('ident_ckpt/*.pt')):
    ck = torch.load(f, map_location='cpu', weights_only=False)
    tag = f"{ck['variant']}_{ck['profile']}"
    pr = predict(ck)
    o = run(pr['g1'], pr['g2'], pr['L'])
    if pr['L'] not in delayed_cache:
        delayed_cache[pr['L']] = run(true_g1, true_g2, pr['L'])
    od = delayed_cache[pr['L']]
    pk, pkd = o['peak_excl'], od['peak_excl']
    row = dict(L_ms=pr['L'] * 1000, val=float(ck['val']), epoch=int(ck['epoch']), groups={})
    if ck['variant'] == 'cls':
        ok = pr['cls'] == true_cls
        row['config_acc'] = float(ok.mean())
        am = ((ty == 2) | ((ty == 1) & (al < 1)))
        row['alpha_mae'] = float(np.abs(pr['alpha'][am] - al[am]).mean())
        lm = GROUPS['line']; wrong = lm & ~ok
        row['line_errors'] = [dict(true=CFG[true_cls[i] - 1], pred=(CFG[pr['cls'][i] - 1] if pr['cls'][i] else 'abstain'))
                              for i in np.where(wrong)[0]]
        row['abstained'] = int((pr['cls'] == 0).sum())
    else:
        lg_err = np.abs(np.log(pr['g1']) - np.log(true_g1)).mean((1, 2))
        ok = lg_err < 0.05                                   # 'identified' = within ~5 % of the true entry's gains
        row['loggain_mae_R1'] = float(lg_err.mean()); row['within5pct'] = float(ok.mean())
    for gname, m in GROUPS.items():
        mis = m & ~ok
        row['groups'][gname] = dict(
            n=int(m.sum()),
            in_env=dict(generalist=float((gen_pk[m] <= LIMIT).mean()), scheduled=float((pk[m] <= LIMIT).mean()),
                        oracle=float((orc_pk[m] <= LIMIT).mean()), delayed_oracle=float((pkd[m] <= LIMIT).mean())),
            median_peak=dict(generalist=float(np.median(gen_pk[m])), scheduled=float(np.median(pk[m])),
                             oracle=float(np.median(orc_pk[m])), delayed_oracle=float(np.median(pkd[m]))),
            headline_distance=float(np.median(gen_pk[m]) - np.median(pk[m])),
            headline_distance_rel=float(1 - np.median(pk[m]) / np.median(gen_pk[m])),
            n_worse_than_generalist=int((pk[m] > gen_pk[m] * (1 + 1e-3) + 1e-7).sum()),
            switching_delay_cost=float(np.median(pkd[m]) - np.median(orc_pk[m])),
            misidentified=int(mis.sum()),
            wrong_schedule_cost_median=(float(np.median(pk[mis] - pkd[mis])) if mis.any() else 0.0),
            wrong_schedule_cost_max=(float((pk[mis] - pkd[mis]).max()) if mis.any() else 0.0),
            tv_median=float(np.median(o['tv'][m])), rail_surv_max=float(o['rail_frac_surv'][m].max()),
            bump_max=float(o['bump'][m].max()))
        gr = row['groups'][gname]
        gr['bar1_not_worse'] = bool(gr['median_peak']['scheduled'] <= gr['median_peak']['generalist'] * (1 + 1e-6)
                                    and gr['in_env']['scheduled'] >= gr['in_env']['generalist'])
        gr['bar3_within10pct_oracle'] = bool(gr['median_peak']['scheduled'] <= 1.10 * gr['median_peak']['oracle'] + 1e-7)
    row['bar2_alpha1_G1_rescue'] = row['groups']['alpha1 G1']['in_env']['scheduled']
    row['all_bars'] = bool(all(g['bar1_not_worse'] and g['bar3_within10pct_oracle'] for g in row['groups'].values())
                           and row['bar2_alpha1_G1_rescue'] >= 1.0)
    res[tag] = row
    print(tag, f"L {row['L_ms']:.0f} ms", f"acc {row.get('config_acc', row.get('within5pct')):.4f}",
          'bars', row['all_bars'], ' | ', ' '.join(f"{g}: {v['median_peak']['scheduled']:.5f}({v['median_peak']['oracle']:.5f}) env {v['in_env']['scheduled']:.2f}"
                                                   for g, v in row['groups'].items()), flush=True)
json.dump(res, open('ident_eval.json', 'w'), indent=1)

# markdown summary
L = ['| Model | L [ms] | ID acc | all bars | ' + ' | '.join(f'{g}: sched / oracle / gen median peak, in-env' for g in GROUPS) + ' |',
     '|---' * (4 + len(GROUPS)) + '|']
for tag, r in res.items():
    acc = r.get('config_acc', r.get('within5pct'))
    L.append(f"| {tag} | {r['L_ms']:.0f} | {acc:.4f} | {'yes' if r['all_bars'] else 'no'} | " + ' | '.join(
        f"{v['median_peak']['scheduled']:.5f} / {v['median_peak']['oracle']:.5f} / {v['median_peak']['generalist']:.5f}, {v['in_env']['scheduled']:.2f}"
        for v in r['groups'].values()) + ' |')
open('ident_eval.md', 'w', encoding='utf-8').write('\n'.join(L) + '\n')
