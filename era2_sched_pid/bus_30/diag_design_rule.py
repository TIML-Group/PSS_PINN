"""diag_design_rule.py -- DIAGNOSTIC (not adopted): same S24 targets, but the tie-break among gains meeting
the targets is 'closest to the S18 baseline' (MU * sum log(g/g18)^2) instead of 'smallest gains'
(MU * sum (g/G_SCALE)^2). Re-runs design + verification pass 1 only; writes diag_design_rule.json."""
import json, time, numpy as np
import library as L
import build_library as BL
from scipy.optimize import minimize
from pole_place import LOG_LO, LOG_HI
g18 = BL.g18; MU = 1e-3


def design_near(lin, Ks, tgt):
    def f(v):
        g = np.exp(v).reshape(6, 3)
        return L.cost(lin, Ks, g, tgt)[0] + MU * np.sum(np.log(g / g18) ** 2)
    v0 = np.log(g18).ravel()
    r = minimize(f, v0, method='L-BFGS-B', bounds=list(zip(np.tile(LOG_LO, 6), np.tile(LOG_HI, 6))), options=dict(maxiter=3000, maxfun=200000))
    g = np.exp(r.x).reshape(6, 3); J, rep = L.cost(lin, Ks, g, tgt); rep = {k: float(v) for k, v in rep.items()}; rep['J'] = float(J)
    return g, rep


t0 = time.time()
lin = L.Lin(); L.post_loss_angles(lin, g18); C = {c['name']: c for c in L.enumerate_configs(lin)}
p = BL.make_plant(); lib = {'R0': g18.copy()}; rep = {}
for nm, cf in C.items():
    if nm == 'R0': continue
    tg = BL.TG[cf['regime']]
    lib[nm], rep[nm] = design_near(lin, cf['Ks'], dict(sigma_em=BL.TG['sigma_em'], zeta_em=tg['zeta'], lam_c=tg['lam_c']))
names = [n for n in C if n != 'R0']
v = BL.verify(p, C, lib, names)
fail = [n for n in names if v[n]['fails']]; ni = [n for n in names if v[n]['no_improvement']]
act = [v[n] for n in names if not v[n]['inactive']]
print(f'max design J {max(r["J"] for r in rep.values()):.2e}; fail {len(fail)}: {fail}\nno-improvement {len(ni)}: {ni}\n'
      f'active {len(act)}, median gap {np.median([a["gap_rel"] for a in act])*100:+.1f}%  ({time.time()-t0:.0f}s)')
for n in fail + ni + ['R1d_G1', 'R2_line', 'R1_line_1', 'R1_runback_G1']:
    x = v[n]; print(f"  {n:16s} {','.join(x['fails']) or '-':30s} med {x['entry_median']:.4f} ref {x['ref_median']:.4f} gap {x['gap_rel']*100:+.1f}% env {x['entry_in_env']:.2f}/{x['ref_in_env']:.2f} rail {x['rail_max']:.3f}")
json.dump(dict(rule='closest-to-baseline tie-break', mu=MU, design=rep, verify=v, gains={k: g.tolist() for k, g in lib.items()}), open('diag_design_rule.json', 'w'), indent=1)
