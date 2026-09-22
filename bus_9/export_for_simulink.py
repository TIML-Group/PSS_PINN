"""
export_for_simulink.py
Exports everything the Simulink model needs from cross_validation_datagen_governor.py
into  ieee9_gov_simulink_data.mat  (and optionally a Python reference trajectory).

Why this works: in the Python DAE the stator has Rs = 0, E' is constant and the
loads are converted to constant impedances, so the network is LINEAR.  The
algebraic states (Id, Iq, Vm, theta) can therefore be eliminated exactly by
Kron reduction to the generator internal nodes:

    I_gen = Yred * E',   Pe = Re(E' . conj(I_gen)),   V_bus = Vrec * E'

That turns the 33-state DAE into a 9-state ODE (delta, omega, Pm) with no
algebraic loop.  A switching event (fault / clearing) is just a change of Yred,
so no "_algebraic_correction" step is needed in Simulink.
"""
import json
import numpy as np
from scipy.io import savemat
import cross_validation_datagen_governor as cv

# ----------------------------------------------------------------- USER CONFIG
PARAMS_JSON = "ieee9_parameters.json"
FAULT_TYPE  = "gen"     # 'none' | 'gen' | 'load' | 'line'
FAULT_LOC   = 0         # 'gen': generator index, 'load': bus index (0-indexed)
FAULT_MAG   = 0.2       # 'gen': step in Pm0 [pu], 'load': admittance added to Y[loc,loc]
T_FAULT, T_CLEAR, T_END = 0.2, 1.2, 2.0
Y_FAULT_RAW = None      # 'line' only: 9x9 complex Ybus during the fault (before load absorption)
Y_POST_RAW  = None      # post-clearing 9x9 complex Ybus (None = original Ybus)
GOV_R, GOV_TG = cv.GOVERNOR_R, cv.GOVERNOR_TG
F0 = 60.0
RUN_PYTHON_REFERENCE = True   # also run the original DAE and save it for comparison
# -----------------------------------------------------------------------------


def col(a):
    """MATLAB wants column vectors (savemat saves 1-D arrays as rows)."""
    return np.asarray(a).reshape(-1, 1)


def absorb_loads(Y_raw, sl, V_lf):
    """Same constant-impedance conversion as Trapezoidal_dae_multitimestep.__init__."""
    Y = np.array(Y_raw, dtype=complex).copy()
    for k in range(Y.shape[0]):
        if abs(sl[k]) > 1e-10:
            Y[k, k] += np.conj(sl[k]) / abs(V_lf[k]) ** 2
    return Y


def kron_reduce(Y_ci, gen_buses, Xdp):
    """Reduce the network to generator internal nodes (behind jXd')."""
    NB, N = Y_ci.shape[0], len(gen_buses)
    yg = 1.0 / (1j * Xdp)
    Ybb = Y_ci.copy()
    Ybg = np.zeros((NB, N), dtype=complex)
    for i, b in enumerate(gen_buses):
        Ybb[b, b] += yg[i]
        Ybg[b, i] = -yg[i]
    Vrec = -np.linalg.solve(Ybb, Ybg)          # V_bus = Vrec @ E'
    Yred = np.diag(yg) + Ybg.T @ Vrec          # I_gen = Yred @ E'
    return Yred, Vrec


def initial_conditions(params):
    """Mirror of the initialisation block inside run_dae_simulation."""
    sd = cv.get_system_data(params)
    gb = sd['machine_buses']
    V_lf, S_lf = cv.run_loadflow(sd['Y_adm'], sd['bus_params'], sd['N_bus'])
    H, D, Xdp, _, _ = cv.get_machine_params(params)

    Vg, Sg = V_lf[gb], S_lf[gb]
    Ig = np.conj(Sg) / np.conj(Vg)
    Ep = Vg + 1j * Xdp * Ig
    delta0 = np.angle(Ep)
    rot = np.exp(-1j * (delta0 - np.pi / 2))
    Idq, Vdq = Ig * rot, Vg * rot
    Eq = Vdq.imag + Xdp * Idq.real
    Ed = Vdq.real - Xdp * Idq.imag
    Pm0 = Ed * Idq.real + Eq * Idq.imag
    sl = sd['pl_ini'] + 1j * sd['ql_ini']
    return sd, V_lf, H, D, Xdp, delta0, Ed, Eq, Pm0, sl


def main():
    params = json.load(open(PARAMS_JSON))
    sd, V_lf, H, D, Xdp, delta0, Ed, Eq, Pm0, sl = initial_conditions(params)
    Y0, gb, N = sd['Y_adm'], sd['machine_buses'], sd['N_gen']

    Y_pre  = absorb_loads(Y0, sl, V_lf)
    Y_post = absorb_loads(Y0 if Y_POST_RAW is None else Y_POST_RAW, sl, V_lf)
    dPm0 = np.zeros(N)
    if FAULT_TYPE in ("none", "gen"):
        Y_flt = Y_pre.copy()
        if FAULT_TYPE == "gen":
            dPm0[FAULT_LOC] = FAULT_MAG
    elif FAULT_TYPE == "load":
        Y_flt = Y_pre.copy()
        Y_flt[FAULT_LOC, FAULT_LOC] += FAULT_MAG
    elif FAULT_TYPE == "line":
        if Y_FAULT_RAW is None:
            raise ValueError("FAULT_TYPE='line' needs Y_FAULT_RAW")
        Y_flt = absorb_loads(Y_FAULT_RAW, sl, V_lf)
    else:
        raise ValueError(FAULT_TYPE)

    Yr_pre, Vr_pre = kron_reduce(Y_pre, gb, Xdp)
    Yr_flt, Vr_flt = kron_reduce(Y_flt, gb, Xdp)
    Yr_post, Vr_post = kron_reduce(Y_post, gb, Xdp)

    # Sanity check: the initial point must be an equilibrium
    E0 = (Ed + 1j * Eq) * np.exp(1j * (delta0 - np.pi / 2))
    Pe0 = np.real(E0 * np.conj(Yr_pre @ E0))
    print("Pm0          :", Pm0)
    print("Pe0 (reduced):", Pe0, " max mismatch =", np.max(np.abs(Pe0 - Pm0)))

    P = dict(H=col(H), D=col(D), Xdp=col(Xdp), Ed=col(Ed), Eq=col(Eq),
             Pm0=col(Pm0), dPm0=col(dPm0), delta0=col(delta0),
             R=float(GOV_R), Tg=float(GOV_TG), f0=float(F0),
             t_fault=float(T_FAULT), t_clear=float(T_CLEAR), t_end=float(T_END),
             dPc=col(np.zeros(N)), t_corr=1e9, t_corr_end=1e9,   # corrective delta_p (off)
             Yred_pre=Yr_pre, Yred_fault=Yr_flt, Yred_post=Yr_post,
             Vrec_pre=Vr_pre, Vrec_fault=Vr_flt, Vrec_post=Vr_post)
    savemat("ieee9_gov_simulink_data.mat", P)
    print("wrote ieee9_gov_simulink_data.mat")

    # Base network data for generate_dataset.m (builds a new fault per scenario)
    savemat("ieee9_base.mat", dict(
        Y_pre=Y_pre,                              # Ybus with loads as impedances
        gen_bus=col(np.asarray(gb) + 1),          # 1-indexed for MATLAB
        load_bus=col(np.flatnonzero(np.abs(sl) > 1e-10) + 1),
        S_load=col(sl), V_lf=col(V_lf)))
    print("wrote ieee9_base.mat")

    if RUN_PYTHON_REFERENCE:
        kw = dict(fault_time=T_FAULT, clear_time=T_CLEAR, total_time=T_END,
                  Y_bus_post=Y_POST_RAW, gov_R=GOV_R, gov_Tg=GOV_TG)
        if FAULT_TYPE == "line":
            out = cv.run_dae_with_clearing("none", 0, 0.0, params,
                                           Y_adm_fault=Y_FAULT_RAW, **kw)
        else:
            out = cv.run_dae_with_clearing(FAULT_TYPE, FAULT_LOC, FAULT_MAG,
                                           params, **kw)
        savemat("python_reference.mat",
                dict(t=col(out['time']), X=out['full_states']))
        print("wrote python_reference.mat")


if __name__ == "__main__":
    main()
