"""
sched_sim.py -- spec v2.0 scheduled-PID simulator (batched, torch) on the v1.4 plant.

Plant: plant_v2.PlantV2 with gen_mode='v13' (alpha < 1 runback with the unit's PID active, restored
at clearing; alpha = 1 permanent disconnection, S20), static Pm0 +- 1 band (S22), no rate limit.
Gains: piecewise constant per regime (S1 of spec v2.0), one gain set per scenario per regime:
    G3[b, r]  r = 0: R0 (pre-fault)   r = 1: R1 / R1d (faulted)   r = 2: R2 (post-clearing)
Regime at step start t:  2 if t >= t_sw2,  else 1 if t >= t_sw1,  else 0.
    t_sw1 = identification switch (oracle: t_fault; MLP: t_fault + delay)
    t_sw2 = t_clear for alpha < 1 events; +inf for alpha = 1 (no clearing, S20)
Bumpless transfer (S23/S24): integrator preloading, position form. At a step where the regime
changes AND the gains differ, the incoming integrator is set so that this step's commanded output
equals the previous step's:  I' = P_set(prev) - Kp' e - Kd' N_D (e - xf) - h Ki' e,  clamped to the
band. Any residual (clamp active) is the transition bump; |dP_set| at switch steps is logged next to
the largest |dP_set| at ordinary steps.
"""
import numpy as np
import torch

from plant_v2 import PlantV2, NG, LIMIT, H_STEP, N_D, T_END

BAND = 1e-3


def make_plant(device='cuda'):
    return PlantV2(device=device, rail=1.0, rate=float('inf'), gen_mode='v13')


def switch_times(scen, delay=0.0):
    t1 = np.array([s['t_fault'] + delay for s in scen])
    t2 = np.array([np.inf if (s['type'] == 'gen' and s['alpha'] >= 1 - 1e-12) else s['t_clear'] for s in scen])
    return t1, t2


@torch.no_grad()
def simulate_sched(p, scen, G3, delay=0.0, record=False, rec_every=10, t_start=None, t_sw=None):
    """scen: list of scenario dicts; G3: (B, 3, NG, 3) array/tensor. Returns dict of numpy arrays.
    t_sw: optional (t1, t2) per-scenario switch-time arrays overriding switch_times(scen, delay)."""
    c, dev, dt, h = p.c, p.device, p.dtype, H_STEP
    bt = p.batch(scen); B = bt['B']
    G3 = torch.as_tensor(np.asarray(G3), dtype=dt, device=dev)
    t1, t2 = switch_times(scen, delay) if t_sw is None else (np.asarray(t_sw[0], float), np.asarray(t_sw[1], float))
    t1 = torch.as_tensor(t1, dtype=dt, device=dev); t2 = torch.as_tensor(t2, dtype=dt, device=dev)
    if t_start is None:
        t_start = float(np.floor(min(s['t_fault'] for s in scen) / 0.05) * 0.05)
    k0, n = int(round(t_start / h)), int(round(T_END / h))
    d = c.delta0.expand(B, NG).clone(); w = torch.ones(B, NG, dtype=dt, device=dev)
    I = p.Pm0.expand(B, NG).clone(); xf = torch.zeros(B, NG, dtype=dt, device=dev)
    Pm = p.Pm0.expand(B, NG).clone(); Ps_prev = Pm.clone()
    lo, hi = p.lo, p.hi
    peak = torch.zeros(B, dtype=dt, device=dev); peak_x = peak.clone(); tv = peak.clone()
    le_x = peak.clone(); rail = peak.clone(); bump = peak.clone(); dps_ord = peak.clone(); rail_x = peak.clone()
    bump1 = peak.clone(); bump2 = peak.clone()                        # per switch kind: into R1 / into R2
    r_prev = torch.zeros(B, dtype=torch.long, device=dev)
    bidx = torch.arange(B, device=dev)
    rec = {'t': [], 'w': [], 'Pm': [], 'd': [], 'Pmech': []} if record else None
    if record and k0 == 0:                                            # include the t = 0 sample
        rec['t'].append(0.0); rec['w'].append(w.clone()); rec['Pm'].append(Pm.clone()); rec['d'].append(d.clone()); rec['Pmech'].append(Pm.clone())
    for k in range(k0, n):
        t = k * h; tm = t + 0.5 * h
        r = torch.where(t >= t2 - 1e-9, 2, torch.where(t >= t1 - 1e-9, 1, 0))
        g = G3[bidx, r]                                                   # (B,NG,3)
        Kp, Ki, Kd = g[..., 0], g[..., 1], g[..., 2]
        faulted = (tm >= bt['tf']) & (tm < bt['tc'])
        fz = bt['frozen'] & faulted.unsqueeze(1)
        e = 1.0 - w
        dterm = N_D * (e - xf)
        sw = (r != r_prev) & (G3[bidx, r] != G3[bidx, r_prev]).flatten(1).any(1)   # (B,)
        if sw.any():
            Ipre = Ps_prev - Kp * e - Kd * dterm - h * Ki * e
            I = torch.where(sw.unsqueeze(1), torch.minimum(torch.maximum(Ipre, lo), hi), I)
        xf_n = xf + h * N_D * (e - xf)
        I_n = torch.minimum(torch.maximum(I + h * Ki * e, lo), hi)
        Ps_raw = Kp * e + I_n + Kd * dterm
        Ps = torch.minimum(torch.maximum(Ps_raw, lo), hi)
        live_m = ~fz
        rail = rail + ((Ps != Ps_raw) & live_m).any(1).to(dt)
        rail_x = rail_x + ((Ps != Ps_raw) & live_m & ~bt['own']).any(1).to(dt)   # S28: survivors only
        dps = torch.where(live_m, (Ps - Ps_prev).abs(), torch.zeros_like(Ps)).amax(1)
        bump = torch.where(sw, torch.maximum(bump, dps), bump)
        bump1 = torch.where(sw & (r == 1), torch.maximum(bump1, dps), bump1)
        bump2 = torch.where(sw & (r == 2), torch.maximum(bump2, dps), bump2)
        dps_ord = torch.where(~sw & (k > k0), torch.maximum(dps_ord, dps), dps_ord)
        xf = torch.where(fz, xf, xf_n); I = torch.where(fz, I, I_n)
        Pm_new = torch.where(fz, Pm, Ps)
        tv = tv + (Pm_new - Pm).abs().sum(1); Pm = Pm_new
        Ps_prev = torch.where(fz, Ps_prev, Ps); r_prev = r
        Pmech = Pm - bt['trip'] * faulted.unsqueeze(1).to(dt)
        fm = faulted.view(B, 1, 1)
        M = torch.where(fm, bt['Mf'], p.M_pre); N = torch.where(fm, bt['Nf'], p.N_pre)
        live = (~(bt['dead'] & faulted.unsqueeze(1))).to(dt)
        k1d, k1w, _ = p._f(d, w, Pmech, M, N)
        k2d, k2w, _ = p._f(d + 0.5 * h * k1d, w + 0.5 * h * k1w, Pmech, M, N)
        k3d, k3w, _ = p._f(d + 0.5 * h * k2d, w + 0.5 * h * k2w, Pmech, M, N)
        k4d, k4w, _ = p._f(d + h * k3d, w + h * k3w, Pmech, M, N)
        d = d + live * (h / 6.0) * (k1d + 2 * k2d + 2 * k3d + k4d)
        w = w + live * (h / 6.0) * (k1w + 2 * k2w + 2 * k3w + k4w)
        adev = (w - 1.0).abs()
        peak = torch.maximum(peak, adev.amax(1))
        post = bt['own'] & (tm >= bt['tf']).unsqueeze(1)
        dx = torch.where(post, torch.zeros_like(adev), adev).amax(1)
        peak_x = torch.maximum(peak_x, dx)
        le_x = torch.where(dx > BAND, torch.full_like(dx, t + h), le_x)
        if record and k % rec_every == 0:
            rec['t'].append(t + h); rec['w'].append(w.clone()); rec['Pm'].append(Pm.clone())
            rec['d'].append(d.clone()); rec['Pmech'].append(Pmech.clone())
    tf = bt['tf']
    out = dict(peak=peak, peak_excl=peak_x, tv=tv, ts=torch.clamp(le_x - tf, min=0.0),
               rail_frac=rail / (n - k0), rail_frac_surv=rail_x / (n - k0), bump=bump, bump_r1=bump1, bump_r2=bump2, dps_ordinary=dps_ord,
               w_end=(w - 1).abs().amax(1))
    out = {k_: v.cpu().numpy() for k_, v in out.items()}
    if record:
        out['rec'] = {k_: (np.array(v) if k_ == 't' else torch.stack(v, 1).cpu().numpy()) for k_, v in rec.items()}
    return out


def const_G3(B, g):
    g = np.asarray(g)
    return np.broadcast_to(g, (B, 3, NG, 3)).copy()
