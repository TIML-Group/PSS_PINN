"""
disc_v14.py -- wiki S20 (v1.4): alpha = 1 gen faults are PERMANENT within the 8 s run (no clearing).
Re-measures the alpha = 1 group's needy status against the S18 baseline (ceiling re-tune,
baseline_ceiling_v13x.json), with pole placement as reference. Survivors' references stay at Pm0,
so the S20 floating band |P_set - P_ref| <= 1 equals the v1.2 rail (rail 1.0, no rate limit).
Cases: every gen case of the 1,200 seed-7 eval set at alpha = 1 (417; same machine and t_fault;
t_clear ignored -- plant_v2 v13 already never clears alpha = 1). Excluded scoring (the tripped
unit is out of the network). Also a gain-multiplier probe on the S18 gains for G1.
Writes disc_v14.json.
"""
import json

import numpy as np
import scipy.io as sio
import torch

from plant_v2 import PlantV2, const_gains, LIMIT

BAND = 1e-3
M = sio.loadmat('matrix_scen.mat', squeeze_me=True); ne = int(M['n_eval'])
S = [dict(type='gen', loc=int(M['loc'][k]), alpha=1.0, t_fault=float(M['tf'][k]), t_clear=float(M['tc'][k]))
     for k in range(ne) if int(M['typ'][k]) == 1]
loc = np.array([s['loc'] for s in S]); tf = np.array([s['t_fault'] for s in S])
gains = {'s18_retune': np.array(json.load(open('baseline_ceiling_v13x.json'))['gains']), 'pole': np.array(M['Gpole'])}
p = PlantV2(device='cuda', rail=1.0, rate=float('inf'), gen_mode='v13'); p.bands = (BAND,)


def run(g, sc):
    o = p.simulate(p.batch(sc), const_gains(len(sc), p.n_lbl, g, p.device, p.dtype), record=True, rec_every=10)
    Pm = o['rec']['Pm'].cpu().numpy(); Pm0 = p.Pm0.cpu().numpy()
    rail = (np.abs(Pm - Pm0) >= 1.0 - 1e-6)                      # (B,T,NG)
    return dict(px=o['peak_excl'].cpu().numpy(), tv=o['tv'].cpu().numpy(),
                ts=np.maximum(o['last_exit_excl'][BAND].cpu().numpy() - np.array([s['t_fault'] for s in sc]), 0),
                wend=(o['w_end'] - 1).abs().amax(1).cpu().numpy(), Pm_end=o['Pm_end'].cpu().numpy(),
                rail_frac=rail.any(2).mean(1), rail_machines=rail.any(1).sum(1))


out = dict(note='S20 v1.4 alpha=1 permanent disconnection, 8 s, excluded scoring', limit=LIMIT, n=len(S))
for k, g in gains.items():
    r = run(g, S)
    row = {}
    for j in (1, 2):
        m = loc == j
        row[f'G{j}'] = dict(n=int(m.sum()), in_envelope=float((r['px'][m] <= LIMIT).mean()),
                            peak_min=float(r['px'][m].min()), peak_max=float(r['px'][m].max()),
                            ts_median=float(np.median(r['ts'][m])), dev_at_8s_max=float(r['wend'][m].max()),
                            tv_median=float(np.median(r['tv'][m])), rail_bind_frac_median=float(np.median(r['rail_frac'][m])),
                            machines_hitting_rail_max=int(r['rail_machines'][m].max()),
                            Pm_end_example=np.round(r['Pm_end'][np.where(m)[0][0]], 3).tolist())
    row['overall_in_envelope'] = float((r['px'] <= LIMIT).mean())
    out[k] = row
    np.savez(f'disc_v14_{k}_py.npz', px=r['px'], ts=r['ts'], tv=r['tv'])
    print(k, json.dumps(row))
# gain-multiplier probe on the S18 gains (G1 and G2, t_fault 1.0)
probe = {}
g18 = gains['s18_retune']
for mult in (1, 1.5, 2, 3, 5, 10):
    sc = [dict(type='gen', loc=j, alpha=1.0, t_fault=1.0, t_clear=2.0) for j in (1, 2)]
    r = run(g18 * mult, sc)
    probe[f'x{mult}'] = dict(G1=float(r['px'][0]), G2=float(r['px'][1]), G1_rail_frac=float(r['rail_frac'][0]),
                             G1_ts=float(r['ts'][0]))
    print('x', mult, probe[f'x{mult}'])
out['gain_probe_s18'] = probe
json.dump(out, open('disc_v14.json', 'w'), indent=1)
