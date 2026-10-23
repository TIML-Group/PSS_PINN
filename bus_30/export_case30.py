"""
export_case30.py
Prepares everything the 30-bus Simulink workflow needs from case30_ieee.json:

  case30_simulink_data.mat  -> struct P used by the Simulink model
  case30_base.mat           -> base network + branch data for generating faults

Model: same as the 9-bus workflow (classical machines behind Xd', droop governor,
constant-impedance loads, Kron-reduced network), generalised to N machines.

Two things differ from the 9-bus exporter and are handled here:
  1. Buses 2, 5 and 8 carry both a machine and a load. The machine output is the
     net bus injection PLUS the local load (the load is already represented as an
     impedance in Y). Using only the net injection -- as the initialisation in
     cross_validation_datagen_governor.py does -- counts that load twice and the
     starting point is not an equilibrium.
  2. Machines 3-6 (buses 5, 8, 11, 13) have Pm0 = 0: they are synchronous
     condensers (reactive support only). By default they get no governor and
     cannot trip or receive a delta_p (see GOV_ON_CONDENSERS).
"""
import json
import numpy as np
from scipy.io import savemat
import cross_validation_datagen_governor as cv

# ----------------------------------------------------------------- USER CONFIG
PARAMS_JSON       = "case30_ieee.json"
GOV_R, GOV_TG     = cv.GOVERNOR_R, cv.GOVERNOR_TG
F0                = 60.0
GOV_ON_CONDENSERS = False   # True: every machine gets a governor (as in the Python DAE)
# -----------------------------------------------------------------------------


def col(a):
    return np.asarray(a).reshape(-1, 1)


def absorb_loads(Y_raw, sl, V_lf):
    Y = np.array(Y_raw, dtype=complex).copy()
    for k in range(Y.shape[0]):
        if abs(sl[k]) > 1e-10:
            Y[k, k] += np.conj(sl[k]) / abs(V_lf[k]) ** 2
    return Y


def kron_reduce(Y_ci, gen_buses, Xdp):
    NB, N = Y_ci.shape[0], len(gen_buses)
    yg = 1.0 / (1j * Xdp)
    Ybb = Y_ci.copy()
    Ybg = np.zeros((NB, N), dtype=complex)
    for i, b in enumerate(gen_buses):
        Ybb[b, b] += yg[i]
        Ybg[b, i] = -yg[i]
    Vrec = -np.linalg.solve(Ybb, Ybg)
    Yred = np.diag(yg) + Ybg.T @ Vrec
    return Yred, Vrec


def branch_contribution(l):
    """2x2 contribution [[Yff, Yft], [Ytf, Ytt]] of one branch (pi model, tap, shift)."""
    y = 1.0 / (l['r_pu'] + 1j * l['x_pu'])
    b = 1j * l['b_pu'] / 2
    tap = (l['ratio_raw'] if l['ratio_raw'] != 0 else 1.0) * np.exp(1j * np.deg2rad(l['shift_deg']))
    return np.array([[(y + b) / abs(tap) ** 2, -y / np.conj(tap)],
                     [-y / tap, y + b]])


def is_connected(branches, NB, skip):
    adj = {i: set() for i in range(NB)}
    for i, l in enumerate(branches):
        if i == skip or not l['status']:
            continue
        adj[l['from_bus']].add(l['to_bus'])
        adj[l['to_bus']].add(l['from_bus'])
    seen, stack = {0}, [0]
    while stack:
        u = stack.pop()
        for v in adj[u] - seen:
            seen.add(v)
            stack.append(v)
    return len(seen) == NB


def main():
    params = json.load(open(PARAMS_JSON))
    sd = cv.get_system_data(params)
    gb, NB, NG = list(sd['machine_buses']), sd['N_bus'], sd['N_gen']
    V_lf, S_lf = cv.run_loadflow(sd['Y_adm'], sd['bus_params'], NB)
    H, D, Xdp, _, _ = cv.get_machine_params(params)
    sl = sd['pl_ini'] + 1j * sd['ql_ini']

    # machine initial conditions (fix 1: add the local load back)
    Vg = V_lf[gb]
    Sg = S_lf[gb] + sl[gb]
    Ig = np.conj(Sg) / np.conj(Vg)
    Ep = Vg + 1j * Xdp * Ig
    delta0 = np.angle(Ep)
    rot = np.exp(-1j * (delta0 - np.pi / 2))
    Idq, Vdq = Ig * rot, Vg * rot
    Eq = Vdq.imag + Xdp * Idq.real
    Ed = Vdq.real - Xdp * Idq.imag
    Pm0 = Ed * Idq.real + Eq * Idq.imag
    Pm0[np.abs(Pm0) < 1e-9] = 0.0

    producing = Pm0 > 1e-6                                   # real generators
    gov = np.ones(NG, bool) if GOV_ON_CONDENSERS else producing
    invR = np.where(gov, 1.0 / GOV_R, 0.0)

    Y_pre = absorb_loads(sd['Y_adm'], sl, V_lf)
    Yr, Vr = kron_reduce(Y_pre, gb, Xdp)

    E0 = (Ed + 1j * Eq) * np.exp(1j * (delta0 - np.pi / 2))
    Pe0 = np.real(E0 * np.conj(Yr @ E0))
    print(f"{NG} machines at buses {np.array(gb) + 1}, {NB} buses")
    print("Pm0          :", np.round(Pm0, 4))
    print("producing    :", producing.astype(int), " governor:", gov.astype(int))
    print("equilibrium check  max|Pe0 - Pm0| =", np.max(np.abs(Pe0 - Pm0)),
          "  max|V_rec - V_lf| =", np.max(np.abs(Vr @ E0 - V_lf)))

    # branches: exact contributions, faultable lines
    L = params['network']['lines']
    fr = np.array([l['from_bus'] for l in L]) + 1
    to = np.array([l['to_bus'] for l in L]) + 1
    C = np.array([branch_contribution(l) for l in L])            # nbr x 2 x 2
    is_trafo = np.array([l['ratio_raw'] != 0 for l in L])
    on = np.array([bool(l['status']) for l in L])
    keeps_connected = np.array([is_connected(L, NB, i) for i in range(len(L))])
    faultable = on & ~is_trafo & keeps_connected
    print(f"branches: {len(L)}, lines: {np.sum(~is_trafo)}, faultable lines: {np.sum(faultable)}")

    P = dict(H=col(H), D=col(D), Xdp=col(Xdp), Ed=col(Ed), Eq=col(Eq),
             Pm0=col(Pm0), dPm0=col(np.zeros(NG)), delta0=col(delta0),
             invR=col(invR), R=float(GOV_R), Tg=float(GOV_TG), f0=float(F0),
             t_fault=0.2, t_clear=0.3, t_end=8.0,
             Yred_pre=Yr, Yred_fault=Yr, Yred_post=Yr,
             Vrec_pre=Vr, Vrec_fault=Vr, Vrec_post=Vr,
             dPc=col(np.zeros(NG)), t_corr=1e9, t_corr_end=1e9)
    savemat("case30_simulink_data.mat", P)
    print("wrote case30_simulink_data.mat")

    savemat("case30_base.mat", dict(
        Y_pre=Y_pre,
        gen_bus=col(np.asarray(gb) + 1),
        load_bus=col(np.flatnonzero(np.abs(sl) > 1e-10) + 1),
        S_load=col(sl), V_lf=col(V_lf),
        producing=col(producing.astype(float)), governor=col(gov.astype(float)),
        br_from=col(fr), br_to=col(to),
        br_Yff=col(C[:, 0, 0]), br_Yft=col(C[:, 0, 1]),
        br_Ytf=col(C[:, 1, 0]), br_Ytt=col(C[:, 1, 1]),
        br_faultable=col(faultable.astype(float)),
        br_is_trafo=col(is_trafo.astype(float))))
    print("wrote case30_base.mat")


if __name__ == "__main__":
    main()
