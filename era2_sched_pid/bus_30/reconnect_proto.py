"""
reconnect_proto.py -- EXPLORATORY (not spec): alpha = 1 full trip, post-reconnection transition.

Plant v2-nogov, v1.2 actuator, wiki S18 baseline gains (ceiling re-tune, baseline_ceiling_v13x.json).
Sequence for a full trip of machine j:
  t < t_fault          normal (all machines, reference P_ref = Pm0).
  [t_fault, t_clear)   machine j disconnected: Kron reduction without its node, its states and PID
                       frozen (not simulated, S18), its speed outside the envelope.
  t_clear              resync like a sync-check relay (S18): omega_j = 1, angle aligned with its
                       terminal-bus voltage of the 5-machine network (=> current through Xd' purely
                       reactive => Pe_j = 0 at closing, checked numerically), Pm_j initialised to the
                       Pe_j it takes on at closing; PID re-entered bumpless (correction integrator 0,
                       derivative filter at e).  Speed rejoins the envelope.
Variants after reconnection:
  'none'      no planned return: P_ref_j stays at Pe_j(closing); survivors keep carrying the loss
              through their integrators.
  'ramp<R>'   gradual handoff: at closing each survivor's integrator-held extra is absorbed into its
              reference (bumpless: P_set unchanged), then P_ref_j ramps Pe_j -> Pm0_j at R pu/s and
              each survivor's reference returns to its Pm0 in proportion (same fraction s(t) of its
              own extra), so the reference total is (nearly) conserved; all PIDs stay active.

Limits (proposal, flagged for Ashhar):
  * +-1 pu correction limit = the PID's authority around the machine's CURRENT reference:
    |P_set - P_ref(t)| <= 1.  With P_ref = Pm0 this is exactly the v1.2 rail.  Survivors also keep
    their absolute v1.2 rail Pm0 +- 1 (intersection), so they are never pushed past their physical band.
    G1 cannot satisfy |P_set - Pm0| <= 1 while it is below 1.58 pu, so its band follows the reference,
    capped above at Pm0 + 1.
  * 2.5 pu/s applies to the planned reference ramp (the dispatch target), not to the PID's fast
    corrections (v1.2 removed the actuator rate limiter).  Survivors' references move at
    extra_i/total_extra * R <= R.
Writes reconnect_proto.json + reconnect_proto.npz (trajectories, every 10 ms).
"""
import json

import numpy as np
import torch

from plant_v2 import PlantV2, NG, LIMIT, H_STEP, N_D

T_END = 12.0          # longer than the 8 s datasets so the handoff can settle (exploratory)
BAND = 1e-3
SHARE_TOL = 0.02      # "regained its share": |Pm_j - Pm0_j| <= 2 % of Pm0_j at T_END


def kron_drop_v(p, j):
    c = p.c
    yg = 1.0 / (1j * torch.as_tensor(c.Xdp, dtype=c.cdtype, device=p.device)); yg[j - 1] = 0
    idx = torch.tensor([b - 1 for b in c.gen_bus], device=p.device)
    Ybb = c.Y_pre.clone(); Ybb[idx, idx] = Ybb[idx, idx] + yg
    Ybg = torch.zeros(c.Y_pre.shape[0], NG, dtype=c.cdtype, device=p.device)
    Ybg[idx, torch.arange(NG, device=p.device)] = -yg
    Vr = -torch.linalg.solve(Ybb, Ybg)
    return torch.diag(yg) + Ybg.T @ Vr, Vr


def run(cases, gains, device='cpu', rec_every=10):
    """cases: dicts loc (1|2), t_fault, t_clear, rate (0 = no handoff)."""
    p = PlantV2(device=device, rail=1.0, rate=float('inf'), gen_mode='v13')
    c, dt, h = p.c, p.dtype, H_STEP
    B = len(cases)
    Pm0 = p.Pm0.expand(B, NG)
    Mpre, Npre = p.M_pre, p.N_pre
    disc = {}
    for j in (1, 2):
        Yd, Vd = kron_drop_v(p, j); disc[j] = (*c.pe_consts(Yd), Vd)
    Md = torch.stack([disc[s['loc']][0] for s in cases]); Nd = torch.stack([disc[s['loc']][1] for s in cases])
    Vd = torch.stack([disc[s['loc']][2] for s in cases])
    jm = torch.zeros(B, NG, dtype=torch.bool); jm[torch.arange(B), [s['loc'] - 1 for s in cases]] = True
    gb = torch.tensor([c.gen_bus[s['loc'] - 1] - 1 for s in cases])
    tf = torch.tensor([s['t_fault'] for s in cases], dtype=dt); tc = torch.tensor([s['t_clear'] for s in cases], dtype=dt)
    R = torch.tensor([s['rate'] for s in cases], dtype=dt)
    g = torch.as_tensor(gains, dtype=dt).expand(B, NG, 3)
    Kp, Ki, Kd = g[..., 0], g[..., 1], g[..., 2]
    d = c.delta0.expand(B, NG).clone(); w = torch.ones(B, NG, dtype=dt)
    J = torch.zeros(B, NG, dtype=dt)                 # correction integrator (P_set = P_ref + PID)
    xf = torch.zeros(B, NG, dtype=dt); Pm = Pm0.clone(); Pref = Pm0.clone()
    Pref0 = Pm0.clone(); extra = torch.zeros(B, NG, dtype=dt); dP1 = torch.ones(B, dtype=dt)
    reconnected = torch.zeros(B, dtype=torch.bool)
    out = dict(pe_at_close=torch.zeros(B, dtype=dt), ref_mismatch=torch.zeros(B, dtype=dt),
               peak_disc=torch.zeros(B, dtype=dt), peak_post=torch.zeros(B, dtype=dt),
               last_exit_post=torch.zeros(B, dtype=dt), jump_at_close=torch.zeros(B, dtype=dt),
               rail_bind_steps=torch.zeros(B, dtype=dt))
    rec = {k: [] for k in ('t', 'w', 'Pm', 'Pref')}
    n = int(round(T_END / h))
    for k in range(n):
        t = k * h; tm = t + 0.5 * h
        dead = jm & ((tm >= tf) & (tm < tc)).unsqueeze(1)
        # ---- reconnection event (first step at/after t_clear) ----
        ev = (tm >= tc) & ~reconnected
        if ev.any():
            b = ev.nonzero().flatten()
            E = c.Ec.abs() * torch.exp(1j * (d[b] + c.phase)).to(c.cdtype)
            Vt = (Vd[b] @ E.unsqueeze(-1)).squeeze(-1)[torch.arange(len(b)), gb[b]]
            jj = jm[b].float().argmax(1)
            d[b, jj] = torch.angle(Vt).to(dt) - c.phase[jj]
            w[b, jj] = 1.0
            th = d[b] + c.phase; dth = th.unsqueeze(2) - th.unsqueeze(1)
            Pe = (Mpre * torch.cos(dth) + Npre * torch.sin(dth)).sum(-1)
            pe_j = Pe[torch.arange(len(b)), jj]
            out['pe_at_close'][b] = pe_j
            out['jump_at_close'][b] = (pe_j - Pm[b, jj]).abs()     # Pm_j was Pm0 frozen; now set to Pe
            Pm[b, jj] = pe_j; Pref[b, jj] = pe_j; J[b, jj] = 0.0; xf[b, jj] = 0.0
            Pref0[b, jj] = pe_j
            hand = R[b] > 0
            if hand.any():
                bh = b[hand]; sm = ~jm[bh]
                Pref[bh] = torch.where(sm, Pref[bh] + J[bh], Pref[bh])  # absorb survivors' extra (bumpless)
                J[bh] = torch.where(sm, torch.zeros_like(J[bh]), J[bh])
                Pref0[bh] = Pref[bh].clone()
                extra[bh] = torch.where(sm, Pref[bh] - Pm0[bh], torch.zeros_like(J[bh]))
                dP1[bh] = Pm0[bh][jm[bh]] - pe_j[hand]
                out['ref_mismatch'][bh] = extra[bh].sum(1) - dP1[bh]
            reconnected |= ev
        # ---- planned reference ramp ----
        post = reconnected & (R > 0)
        if post.any():
            s = torch.clamp(R * (t - tc) / dP1, 0.0, 1.0) * post.to(dt)
            Pref = torch.where(post.unsqueeze(1),
                               torch.where(jm, Pref0 + s.unsqueeze(1) * dP1.unsqueeze(1), Pref0 - s.unsqueeze(1) * extra),
                               Pref)
        # ---- limits ----
        lo = torch.where(jm & reconnected.unsqueeze(1), Pref - 1.0, torch.maximum(Pref - 1.0, Pm0 - 1.0))
        hi = torch.minimum(Pref + 1.0, Pm0 + 1.0)
        # ---- PID ----
        e = 1.0 - w
        dterm = N_D * (e - xf)
        xf_n = xf + h * N_D * (e - xf)
        J_n = torch.minimum(torch.maximum(J + h * Ki * e, lo - Pref), hi - Pref)
        Ps_raw = Pref + Kp * e + J_n + Kd * dterm
        Ps = torch.minimum(torch.maximum(Ps_raw, lo), hi)
        out['rail_bind_steps'] += ((Ps != Ps_raw) & ~dead).any(1).to(dt) * reconnected.to(dt)
        xf = torch.where(dead, xf, xf_n); J = torch.where(dead, J, J_n); Pm = torch.where(dead, Pm, Ps)
        # ---- network + swing ----
        fm = (tm >= tf) & (tm < tc)
        M = torch.where(fm.view(B, 1, 1), Md, Mpre); N = torch.where(fm.view(B, 1, 1), Nd, Npre)
        live = (~dead).to(dt)
        k1d, k1w, _ = p._f(d, w, Pm, M, N)
        k2d, k2w, _ = p._f(d + 0.5 * h * k1d, w + 0.5 * h * k1w, Pm, M, N)
        k3d, k3w, _ = p._f(d + 0.5 * h * k2d, w + 0.5 * h * k2w, Pm, M, N)
        k4d, k4w, _ = p._f(d + h * k3d, w + h * k3w, Pm, M, N)
        d = d + live * (h / 6.0) * (k1d + 2 * k2d + 2 * k3d + k4d)
        w = w + live * (h / 6.0) * (k1w + 2 * k2w + 2 * k3w + k4w)
        adev = torch.where(dead, torch.zeros_like(w), (w - 1.0).abs())
        dev = adev.amax(1)
        in_disc = fm; in_post = reconnected
        out['peak_disc'] = torch.where(in_disc, torch.maximum(out['peak_disc'], dev), out['peak_disc'])
        out['peak_post'] = torch.where(in_post, torch.maximum(out['peak_post'], dev), out['peak_post'])
        out['last_exit_post'] = torch.where(in_post & (dev > BAND), torch.full_like(dev, t + h), out['last_exit_post'])
        if k % rec_every == 0:
            rec['t'].append(t + h); rec['w'].append(w.clone()); rec['Pm'].append(Pm.clone()); rec['Pref'].append(Pref.clone())
    out = {k: v.numpy() for k, v in out.items()}
    out['Pm_end'] = Pm.numpy(); out['w_end'] = w.numpy()
    rec = {k: (np.array(v) if k == 't' else torch.stack(v, 1).numpy()) for k, v in rec.items()}
    return out, rec, p.Pm0.numpy()


if __name__ == '__main__':
    gains = np.array(json.load(open('baseline_ceiling_v13x.json'))['gains'])
    cases = []
    for loc, durs in ((1, (0.1, 0.5, 1.0, 2.0, 3.0)), (2, (0.5, 3.0))):
        for dur in durs:
            for rate in (0.0, 2.5, 0.5):
                cases.append(dict(loc=loc, t_fault=1.0, t_clear=1.0 + dur, rate=rate))
    o, rec, Pm0 = run(cases, gains)
    rows = []
    for i, s in enumerate(cases):
        j = s['loc'] - 1
        share_ok = abs(o['Pm_end'][i, j] - Pm0[j]) <= SHARE_TOL * Pm0[j]
        # first time from which G_j stays within tolerance of Pm0
        pmj = rec['Pm'][i, :, j]; okk = np.abs(pmj - Pm0[j]) <= SHARE_TOL * Pm0[j]
        stay = np.where(~okk)[0]
        t_share = (None if not okk[-1] else float(rec['t'][stay[-1] + 1] - s['t_clear']) if len(stay) else 0.0)
        rows.append(dict(machine=f"G{s['loc']}", duration=s['t_clear'] - s['t_fault'],
                         variant='none' if s['rate'] == 0 else f"ramp {s['rate']} pu/s",
                         pe_at_close=float(o['pe_at_close'][i]), pm_jump_at_close=float(o['jump_at_close'][i]),
                         peak_during_disconnection=float(o['peak_disc'][i]), peak_after_reconnection=float(o['peak_post'][i]),
                         ts_after_reconnection=float(max(o['last_exit_post'][i] - s['t_clear'], 0.0)),
                         Pm_end=o['Pm_end'][i].round(4).tolist(), share_regained=bool(share_ok),
                         t_share_from_reconnection=t_share, ref_total_mismatch=float(o['ref_mismatch'][i]),
                         rail_bind_fraction_post=float(o['rail_bind_steps'][i] / ((T_END - s['t_clear']) / H_STEP)),
                         w_end_maxdev=float(np.abs(o['w_end'][i] - 1).max())))
        r = rows[-1]
        print(f"{r['machine']} {r['duration']:.1f}s {r['variant']:13s} Pe@close {r['pe_at_close']:+.2e}  peak disc {r['peak_during_disconnection']:.4f}  "
              f"peak post {r['peak_after_reconnection']:.4f}  ts {r['ts_after_reconnection']:.2f}  share {r['share_regained']} ({r['t_share_from_reconnection']})  "
              f"Pm_end {r['Pm_end']}  mism {r['ref_total_mismatch']:+.3f}  rail {r['rail_bind_fraction_post']:.3f}")
    json.dump(dict(note='EXPLORATORY - not spec; awaiting Ashhar', T_END=T_END, share_tol=SHARE_TOL, cases=rows),
              open('reconnect_proto.json', 'w'), indent=1)
    np.savez('reconnect_proto.npz', t=rec['t'], w=rec['w'], Pm=rec['Pm'], Pref=rec['Pref'], Pm0=Pm0,
             cases=np.array([[s['loc'], s['t_fault'], s['t_clear'], s['rate']] for s in cases]))
