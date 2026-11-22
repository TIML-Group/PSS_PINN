"""
lamc_sweep.py -- S24: lambda_c / zeta sweep on true dynamics (CUDA), before freezing the per-regime
pole targets. sigma_em = 0.5 fixed. Candidate entries are pole-placed per configuration; the other
regimes run the S18 baseline (R0 = baseline per S24). Metrics per (regime, lam_c, zeta):
median / max excluded peak, median t_s (1e-3 band, from t_fault), in-envelope, rail-binding,
TV, and the same for the baseline on identical cases. Writes lamc_sweep.json.
"""
import json, time
import numpy as np
from library import Lin, enumerate_configs, design_best, post_loss_angles
from sched_sim import make_plant, simulate_sched
from plant_v2 import LIMIT

LAMC = (0.3, 0.5, 1.0, 2.0); ZETA = (0.3, 0.5); SIG = 0.5
DUR = (0.1, 0.5, 1.5, 3.0)
g18 = np.array(json.load(open('baseline_ceiling_v13x.json'))['gains'])
lin = Lin(); post_loss_angles(lin, g18); C = {c['name']: c for c in enumerate_configs(lin)}
p = make_plant()


def cases(name):
    cf = C[name]
    if cf['regime'] == 'R1d':
        return [dict(type='gen', loc=cf['loc'], alpha=1.0, t_fault=1.0, t_clear=2.0)]
    fam = cf['family'] if cf['regime'] == 'R1' else name.replace('R2_', '')
    if fam == 'line':
        locs = [cf['loc']] if cf['regime'] == 'R1' else [1, 5, 20]
        return [dict(type='line', loc=l, alpha=1.0, t_fault=1.0, t_clear=1.0 + d) for l in locs for d in DUR]
    if fam == 'load':
        locs = [cf['loc']] if cf['regime'] == 'R1' else [int(lin.p.c.load_bus[0]), int(lin.p.c.load_bus[-1])]
        return [dict(type='load', loc=l, alpha=a, t_fault=1.0, t_clear=1.0 + d) for l in locs for a in (0.4, 0.7, 1.0) for d in DUR]
    j = int(fam[-1])
    return [dict(type='gen', loc=j, alpha=a, t_fault=1.0, t_clear=1.0 + d) for a in (0.4, 0.7, 0.95) for d in DUR]


SAMPLE = {'R1': ['R1_line_1', 'R1_line_5', 'R1_line_20', f'R1_load_{int(lin.p.c.load_bus[0])}',
                 f'R1_load_{int(lin.p.c.load_bus[-1])}', 'R1_runback_G1', 'R1_runback_G2'],
          'R1d': ['R1d_G1', 'R1d_G2'],
          'R2': ['R2_line', 'R2_load', 'R2_runback_G1', 'R2_runback_G2']}
slot = {'R1': 1, 'R1d': 1, 'R2': 2}


def metrics(o):
    return dict(peak_med=float(np.median(o['peak_excl'])), peak_max=float(o['peak_excl'].max()),
                ts_med=float(np.median(o['ts'])), ts_max=float(o['ts'].max()),
                in_env=float((o['peak_excl'] <= LIMIT).mean()), rail_med=float(np.median(o['rail_frac'])),
                tv_med=float(np.median(o['tv'])), dev8_med=float(np.median(o['w_end'])))


t0 = time.time(); rows = []; base = {}
for reg, names in SAMPLE.items():
    for nm in names:
        S = cases(nm); G = np.broadcast_to(g18, (len(S), 3, 6, 3)).copy()
        base[nm] = metrics(simulate_sched(p, S, G))
        prev = None
        for z in ZETA:
            for lc in LAMC:
                g, rep = design_best(lin, C[nm]['Ks'], dict(sigma_em=SIG, zeta_em=z, lam_c=lc),
                                     starts=(g18,) + ((prev,) if prev is not None else ()))
                prev = g
                Gc = G.copy(); Gc[:, slot[reg]] = g
                m = metrics(simulate_sched(p, S, Gc))
                rows.append(dict(regime=reg, config=nm, zeta=z, lam_c=lc, design=rep, gains=g.round(4).tolist(), **m))
                print(f"{nm:16s} z{z} lc{lc}: J {rep['J']:.1e} lamc {rep['lam_c']:+.2f} | peak med {m['peak_med']:.4f} max {m['peak_max']:.4f} "
                      f"(base {base[nm]['peak_med']:.4f}/{base[nm]['peak_max']:.4f}) ts med {m['ts_med']:.2f} (base {base[nm]['ts_med']:.2f}) "
                      f"env {m['in_env']:.2f} rail {m['rail_med']:.3f}  [{time.time()-t0:.0f}s]", flush=True)
json.dump(dict(sigma_em=SIG, lamc=LAMC, zeta=ZETA, durations=DUR, baseline=base, rows=rows), open('lamc_sweep.json', 'w'), indent=1)
