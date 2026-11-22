"""
gen_dataset_v2.py -- Era-2 dataset v2 (spec v2.1 S5, wiki S23-S26): 11,000 scenarios on the v1.4 plant.

Scenarios
  main  : 10,000, era-1 mix and ranges (uniform type; gen/load alpha ~ U[0.10, 1.00], line alpha = 1;
          t_fault ~ U[0.50, 4.00], duration ~ U[0.05, 3.00], both on the 1 ms grid), SEED 42 (numpy).
          Gen alphas from the continuous draw are all < 1 -> connected runbacks (S13).
  alpha1: 1,000 permanent disconnections (S20/S23): 500 G1 + 500 G2, t_fault ~ U[0.50, 4.00], t_clear = N/A
          (NaN), SEED 43. Reported as a separate block (block = 1).
Input X (identification input, S25 note): the noisy measurement trajectory under R0 gains (= S18 generalist,
  no switching) -- what the identifier observes before it schedules anything. Every 1 ms, t = 0..8 s
  (8001 samples), era-1 channel layout (90 = delta/omega interleaved | Pm | Id/Iq interleaved |
  Vm/theta interleaved, theta unwrapped) and era-1 noise levels, per-scenario RNG seed 42 + k.
  Pm channel = mechanical power actually delivered (runback: P_set - alpha*Pm0 during the fault;
  disconnected unit: 0 after t_fault). The disconnected unit's delta/omega are held (not simulated, S20);
  its Id/Iq are 0 (its row of the reduced network is zero).
Labels / metadata (meta.npz): y_type (1 gen, 2 load, 3 line), y_t_fault, meta_alpha, meta_loc, meta_t_clear
  (NaN for alpha = 1), block, permanent; library pointers lib_R1 (R1 / R1d entry index) and lib_R2 (R2 entry
  index, -1 for alpha = 1), entry names in lib_names; switch events t_sw1 = t_fault, t_sw2 = t_clear or NaN;
  outcomes under the generalist (R0 everywhere) and under the oracle schedule (library, switch at t_fault):
  peak_excl, ts, tv; y_ool = generalist excluded peak > 0.005; split (1 train, 2 val, 3 test): main 80/10/10
  (rng SEED+1, as era-1); alpha1 block per machine 300/100/100 so the test split has >= 100 per machine.
Output (scenario-major memmap, the layout train_case30.py reads after convert_case30.py; stride 1):
  data_v2/X.npy (11000, 8001, 90) float32, data_v2/meta.npz, data_v2/info.json
"""
import json
import os
import time

import numpy as np
import torch

from sched_sim import make_plant, simulate_sched
from plant_v2 import NG, LIMIT

OUT = 'data_v2'
N_MAIN, N_A1, SEED = 10000, 1000, 42
DT, T_END = 1e-3, 8.0
NT = int(round(T_END / DT)) + 1
NB = 30
NC = 5 * NG + 2 * NB
BATCH = 250
RANGE = dict(t_fault=(0.5, 4.0), duration=(0.05, 3.0), alpha=(0.1, 1.0))
NS = dict(delta=0.01, omega=5e-5, Pm=0.005, Id=0.005, Iq=0.005, Vm=0.001, Th=0.01)


def draw(p):
    c = p.c
    cand = {1: [1, 2], 2: [int(b) for b in c.load_bus], 3: [int(i) + 1 for i in c.faultable]}
    rng = np.random.default_rng(SEED)
    typ = rng.integers(1, 4, N_MAIN)
    S = []
    for k in range(N_MAIN):
        t = int(typ[k])
        a = 1.0 if t == 3 else float(rng.uniform(*RANGE['alpha']))
        loc = int(rng.choice(cand[t]))
        tf = round(float(rng.uniform(*RANGE['t_fault'])) / DT) * DT
        tc = tf + max(DT, round(float(rng.uniform(*RANGE['duration'])) / DT) * DT)
        S.append(dict(type={1: 'gen', 2: 'load', 3: 'line'}[t], loc=loc, alpha=a, t_fault=tf, t_clear=tc, block=0))
    assert all(s['alpha'] < 1 for s in S if s['type'] == 'gen')
    rng = np.random.default_rng(SEED + 1)
    for k in range(N_A1):
        tf = round(float(rng.uniform(*RANGE['t_fault'])) / DT) * DT
        S.append(dict(type='gen', loc=1 if k < N_A1 // 2 else 2, alpha=1.0, t_fault=tf, t_clear=float('nan'), block=1))
    return S


def splits(S):
    sp = np.zeros(len(S), np.int8)
    rng = np.random.default_rng(SEED + 1); perm = rng.permutation(N_MAIN)
    sp[perm[:int(0.8 * N_MAIN)]] = 1; sp[perm[int(0.8 * N_MAIN):int(0.9 * N_MAIN)]] = 2; sp[perm[int(0.9 * N_MAIN):]] = 3
    rng = np.random.default_rng(SEED + 2)
    for j in (1, 2):
        idx = np.array([i for i, s in enumerate(S) if s['block'] == 1 and s['loc'] == j]); idx = idx[rng.permutation(len(idx))]
        sp[idx[:300]] = 1; sp[idx[300:400]] = 2; sp[idx[400:]] = 3
    return sp


class Networks:
    """Per-scenario reduced admittance and voltage-reconstruction matrices (pre / fault-or-disconnected)."""

    def __init__(self, p):
        self.p, self.c = p, p.c
        self.pre = self.c.kron(self.c.Y_pre)
        self.cache = {}

    def kron_drop(self, j):
        c = self.c
        yg = 1.0 / (1j * torch.as_tensor(c.Xdp, dtype=c.cdtype, device=c.Y_pre.device)); yg[j - 1] = 0
        idx = torch.tensor([b - 1 for b in c.gen_bus], device=c.Y_pre.device)
        Ybb = c.Y_pre.clone(); Ybb[idx, idx] = Ybb[idx, idx] + yg
        Ybg = torch.zeros(NB, NG, dtype=c.cdtype, device=c.Y_pre.device); Ybg[idx, torch.arange(NG, device=c.Y_pre.device)] = -yg
        Vr = -torch.linalg.solve(Ybb, Ybg)
        return torch.diag(yg) + Ybg.T @ Vr, Vr

    def fault(self, s):
        key = (s['type'], s['loc'], round(s['alpha'], 12))
        if key not in self.cache:
            c = self.c
            if s['type'] == 'gen':
                self.cache[key] = self.kron_drop(s['loc']) if s['alpha'] >= 1 - 1e-12 else self.pre
            elif s['type'] == 'load':
                Y = c.Y_pre.clone(); Y[s['loc'] - 1, s['loc'] - 1] += s['alpha']; self.cache[key] = c.kron(Y)
            else:
                self.cache[key] = c.kron(c._open_branch(c.Y_pre, s['loc'] - 1))
        return self.cache[key]


def channels(p, nets, sc, rec):
    """rec from simulate_sched(record=True, rec_every=1, t_start=0) -> (B, NT, 90) float32 clean."""
    c = p.c; dev = p.device
    t = torch.as_tensor(rec['t'], device=dev)                           # (NT,)
    d = torch.as_tensor(rec['d'], device=dev); w = torch.as_tensor(rec['w'], device=dev)
    pm = torch.as_tensor(rec['Pmech'], device=dev)
    # rec['Pmech'][:, i] is the power applied during the step ENDING at t_i; the sample at t_i must carry the
    # value applied during [t_i, t_i + h) (Simulink's held output) -> shift one sample (checked vs Simulink)
    pm = torch.cat([pm[:, 1:], pm[:, -1:]], 1)
    B = d.shape[0]
    tf = torch.tensor([s['t_fault'] for s in sc], device=dev, dtype=d.dtype)
    tc = torch.tensor([s['t_clear'] if np.isfinite(s['t_clear']) else 1e9 for s in sc], device=dev, dtype=d.dtype)
    dead = torch.zeros(B, NG, dtype=torch.bool, device=dev)
    for b, s in enumerate(sc):
        if s['type'] == 'gen' and s['alpha'] >= 1 - 1e-12:
            dead[b, s['loc'] - 1] = True
    fa = (t.unsqueeze(0) >= tf.unsqueeze(1) - 1e-9) & (t.unsqueeze(0) < tc.unsqueeze(1) - 1e-9)   # (B, NT)
    pm = torch.where(dead.unsqueeze(1) & fa.unsqueeze(2), torch.zeros_like(pm), pm)
    Yp, Vp = nets.pre
    Yf = torch.stack([nets.fault(s)[0] for s in sc]); Vf = torch.stack([nets.fault(s)[1] for s in sc])
    cd = torch.complex64
    rot = torch.exp(1j * (d - np.pi / 2).to(torch.float32)).to(cd)                    # (B,NT,NG)
    Edq = (c.Ec * np.exp(1j * np.pi / 2)).to(cd)                                       # Ed + j Eq
    E = Edq * rot
    I_pre = E @ Yp.to(cd).T; V_pre = E @ Vp.to(cd).T
    I_f = torch.einsum('btj,bij->bti', E, Yf.to(cd)); V_f = torch.einsum('btj,bij->bti', E, Vf.to(cd))
    m = fa.unsqueeze(2)
    I = torch.where(m, I_f, I_pre); V = torch.where(m, V_f, V_pre)
    Idq = I / rot
    X = torch.zeros(B, len(t), NC, dtype=torch.float32, device=dev)
    X[:, :, 0:2 * NG:2] = d.float(); X[:, :, 1:2 * NG:2] = w.float()
    X[:, :, 2 * NG:3 * NG] = pm.float()
    X[:, :, 3 * NG:5 * NG:2] = Idq.real; X[:, :, 3 * NG + 1:5 * NG:2] = Idq.imag
    X[:, :, 5 * NG::2] = V.abs()
    th = torch.angle(V).cpu().numpy()
    X[:, :, 5 * NG + 1::2] = torch.as_tensor(np.unwrap(th, axis=1), device=dev, dtype=torch.float32)
    return X.cpu().numpy()


def noise_vec():
    sig = np.zeros(NC, np.float32)
    sig[0:2 * NG:2] = NS['delta']; sig[1:2 * NG:2] = NS['omega']; sig[2 * NG:3 * NG] = NS['Pm']
    sig[3 * NG:5 * NG:2] = NS['Id']; sig[3 * NG + 1:5 * NG:2] = NS['Iq']
    sig[5 * NG::2] = NS['Vm']; sig[5 * NG + 1::2] = NS['Th']
    return sig


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    p = make_plant(); nets = Networks(p)
    L = json.load(open('library_v21.json'))
    names = list(L['entries'].keys()); idx_of = {n: i for i, n in enumerate(names)}
    G = {n: np.array(v['gains']) for n, v in L['entries'].items()}
    g18 = G['R0']
    S = draw(p); N = len(S)
    import build_library as BL
    r1 = np.array([idx_of[BL.r1_name(s)] for s in S])
    r2 = np.array([-1 if (s['type'] == 'gen' and s['alpha'] >= 1 - 1e-12) else idx_of[BL.r2_name(s)] for s in S])
    # ---- outcomes: generalist and oracle schedule (metrics only, one batch each) ----
    out = {}
    for tag, G3 in (('gen', np.broadcast_to(g18, (N, 3, NG, 3)).copy()),
                    ('orc', np.array([[g18, G[names[r1[i]]], (G[names[r2[i]]] if r2[i] >= 0 else g18)] for i in range(N)]))):
        o = simulate_sched(p, S, G3)
        out[tag] = {k: o[k] for k in ('peak_excl', 'ts', 'tv', 'bump', 'rail_frac')}
        print(f'{tag}: in-envelope {np.mean(o["peak_excl"] <= LIMIT):.4f}  median peak {np.median(o["peak_excl"]):.5f}  '
              f'median ts {np.median(o["ts"]):.3f}  bump max {o["bump"].max():.1e}  ({time.time() - t0:.0f}s)', flush=True)
    # ---- X under R0 gains, every 1 ms, noisy ----
    X = np.lib.format.open_memmap(os.path.join(OUT, 'X.npy'), mode='w+', dtype=np.float32, shape=(N, NT, NC))
    sig = noise_vec(); maxdev = np.zeros(N)
    for b0 in range(0, N, BATCH):
        sc = S[b0:b0 + BATCH]
        o = simulate_sched(p, sc, np.broadcast_to(g18, (len(sc), 3, NG, 3)).copy(), record=True, rec_every=1, t_start=0.0)
        assert o['rec']['t'].shape[0] == NT, o['rec']['t'].shape
        Xc = channels(p, nets, sc, o['rec'])
        for j in range(len(sc)):
            k = b0 + j
            rs = np.random.default_rng(SEED + k + 1)
            X[k] = Xc[j] + rs.standard_normal((NT, NC), dtype=np.float32) * sig
        maxdev[b0:b0 + len(sc)] = o['peak_excl']
        print(f'  X rows {b0}-{b0 + len(sc) - 1} written  ({time.time() - t0:.0f}s)', flush=True)
    X.flush(); del X
    sp = splits(S)
    meta = dict(t=np.arange(NT) * DT, y_type=np.array([{'gen': 1, 'load': 2, 'line': 3}[s['type']] for s in S], np.int8),
                y_t_fault=np.array([s['t_fault'] for s in S]), meta_alpha=np.array([s['alpha'] for s in S]),
                meta_loc=np.array([s['loc'] for s in S]), meta_t_clear=np.array([s['t_clear'] for s in S]),
                block=np.array([s['block'] for s in S], np.int8), permanent=np.array([s['block'] == 1 for s in S]),
                lib_R1=r1, lib_R2=r2, lib_names=np.array(names), t_sw1=np.array([s['t_fault'] for s in S]),
                t_sw2=np.array([s['t_clear'] for s in S]), y_ool=out['gen']['peak_excl'] > LIMIT,
                meta_maxdev=maxdev, split=sp, noise_std=sig, omega_limit=LIMIT,
                gen_peak_excl=out['gen']['peak_excl'], gen_ts=out['gen']['ts'], gen_tv=out['gen']['tv'],
                orc_peak_excl=out['orc']['peak_excl'], orc_ts=out['orc']['ts'], orc_tv=out['orc']['tv'],
                orc_bump=out['orc']['bump'], orc_rail_frac=out['orc']['rail_frac'],
                state_names=np.array(','.join([f'{a}{i + 1}' for i in range(NG) for a in ('delta', 'omega')]
                                              + [f'Pm{i + 1}' for i in range(NG)]
                                              + [f'{a}{i + 1}' for i in range(NG) for a in ('Id', 'Iq')]
                                              + [f'{a}{b + 1}' for b in range(NB) for a in ('Vm', 'Th')])))
    np.savez(os.path.join(OUT, 'meta.npz'), **meta)
    info = dict(N=N, n_main=N_MAIN, n_alpha1=N_A1, T=NT, dt=DT, stride=1, channels=NC, dtype='float32',
                plant='v2-nogov v1.4 (S20 permanent alpha=1, S22 static band, no rate limit)',
                input_controller='R0 = S18 generalist (no switching)', library='library_v21.json (S26)',
                seeds=dict(main=SEED, alpha1=SEED + 1, noise='SEED + k + 1', split_main=SEED + 1, split_alpha1=SEED + 2),
                runtime_s=time.time() - t0,
                summary={tag: dict(in_envelope_all=float(np.mean(out[tag]['peak_excl'] <= LIMIT)),
                                   in_envelope_main=float(np.mean(out[tag]['peak_excl'][:N_MAIN] <= LIMIT)),
                                   in_envelope_alpha1=float(np.mean(out[tag]['peak_excl'][N_MAIN:] <= LIMIT)),
                                   median_peak=float(np.median(out[tag]['peak_excl'])), median_ts=float(np.median(out[tag]['ts'])))
                         for tag in out})
    json.dump(info, open(os.path.join(OUT, 'info.json'), 'w'), indent=1)
    print('DONE', json.dumps(info['summary']), f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
