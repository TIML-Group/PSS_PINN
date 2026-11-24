import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
import numpy as np
import torch
from case30_common import Setup, NG, N_STATE

DT = 0.005
device = torch.device('cpu')
su = Setup('case30_ctrl/refined_full/best.pt', 'data30_mm', device=device)
model = su.model
model.eval()
S = su.S
t_fault, t_clear, t_grid = su.t_fault, su.t_clear, su.t_grid
n_steps = int(round(float(t_grid[-1]) / DT))
ti = np.arange(n_steps + 1, dtype=np.float32) * DT

recs = json.load(open('verify_end_mech/end_mech_records.json'))['records']
lost = [r for r in recs if r['resc_win'] and not r['resc_run']]
rec = [r for r in lost if r['k'] == 8777 and abs(r['dp_norm'] - 1.22) < 0.01][0]
print("rec:", {k: rec[k] for k in ('k', 'tk', 'tf', 'tc', 'pk_win', 'pk_run',
                                  'exit_after_clear', 'dp_norm')})

with torch.no_grad():
    rng = np.random.default_rng(0)
    b = su.to_dev(su.make_batch(np.array([rec['k']]), rng,
                                tk_fixed=np.array([rec['tk']])))
    o = model(b['feat'], b['x_meas'], hard=True, ctrl=True, vmask=b['vmask'])
    dP = o['dP']
    n = 1
    Yf, dPm = su.true_network(np.array([rec['k']]))
    tf = torch.tensor([rec['tf']], dtype=torch.float32)
    tc = torch.tensor([rec['tc']], dtype=torch.float32)
    tk = torch.tensor([rec['tk']], dtype=torch.float32)
    dP6 = model.scatter_dp(dP)
    print("dP:", dP.tolist(), "dP6:", dP6.tolist(), "dPm:", dPm.tolist())
    x = S.x0.to(torch.float32).unsqueeze(0).expand(n, N_STATE).clone()
    traj = torch.zeros(n, n_steps + 1, NG)
    Mp, Np = S.pe_consts(S.Yred_pre.expand(n, NG, NG))
    Mf, Nf = S.pe_consts(Yf)
    traj[:, 0] = x[:, NG:2 * NG]
    for i in range(n_steps):
        tm = i * DT + DT / 2
        act = (tm >= rec['tf']) & (tm < rec['tc'])
        av = torch.tensor([[act]], dtype=torch.float32)
        M = torch.where(av.view(n, 1, 1) > 0.5, Mf, Mp)
        N = torch.where(av.view(n, 1, 1) > 0.5, Nf, Np)
        Ps = S.Pm0.unsqueeze(0) + dPm * av.view(n, 1)
        on = (tm >= rec['tk']) & (tm < rec['tc'])
        ov = torch.tensor([[on]], dtype=torch.float32)
        Ps = Ps + dP6 * ov
        x = S.rk4_real(x, M, N, Ps, DT)
        traj[:, i + 1] = x[:, NG:2 * NG]
    d = (traj[0] - 1.0).abs().amax(1).numpy()
    big = np.nonzero(d > 0.1)[0]
    print("n spikes >0.1:", len(big), "first idx:", big[:5].tolist(),
          "times:", ti[big[:5]].tolist(), "d there:", d[big[:5]].tolist())
    print("d[0:5]:", d[:5].tolist())
    print("max d (t>0.2s):", d[40:].max(), "argmax t =", ti[40 + d[40:].argmax()])
