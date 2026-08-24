"""
verify_against_real_dae.py
=============================
Takes a ΔPm result (constant or time-varying) found by delta_pm_solver.py's
optimizer, and checks it against the REAL DAE (cross_validation_datagen_governor.py)
-- not our own differentiable rollout, which was used to FIND the answer.
This is the "trust but verify" step: does the correction actually work on
the ground-truth simulator, or only on our approximation of it?

HOW A TIME-VARYING ΔPm SEQUENCE IS REPLICATED IN THE REAL DAE
----------------------------------------------------------------
run_dae_simulation() recomputes Pm0_const FROM SCRATCH (a fresh load-flow
solve) at the start of every call, then adds fault_mag on top via the
'gen' contingency type. This means a time-varying ΔPm sequence can be
replicated by chaining many short calls -- one per timestep -- each
specifying the ABSOLUTE target ΔPm for that instant (not incremental,
since the baseline resets every call), continuing from the previous
segment's end state via custom_initial_state.

The whole scenario is built as:
  1. t=0 -> t_fault:        normal system (no correction needed yet)
  2. t_fault -> t_clear:    the actual fault is active (line/gen/load fault)
  3. t_clear -> t_clear+window: OUR ΔPm correction, chained step-by-step,
     on the REAL post-clearing topology (reverted per what we confirmed
     about line faults, or the original system for gen/load faults)

Usage: see the __main__ block at the bottom for a worked example
(transmission line fault, verifying a real optimizer result).
"""

import json
import numpy as np

from cross_validation_datagen_governor import (
    run_dae_simulation, get_system_data, GOVERNOR_R, GOVERNOR_TG)


def run_gen_fault_to_clearing(params, gen_idx, alpha, Pm0_gen, fault_time, clear_time):
    """Stage 1+2 for a gen-trip fault: fault_mag = -alpha * Pm0[gen_idx],
    matching generate_data_v8.py's exact formula. Topology is unchanged
    for gen faults (only Pm0_const shifts), so no Y_adm override needed."""
    data0 = run_dae_simulation(
        'none', 0, 0.0, params=params,
        fault_time=0.0, sim_end_time=fault_time,
        gov_R=GOVERNOR_R, gov_Tg=GOVERNOR_TG)
    state_at_fault = data0['full_states'][-1, :].copy()

    fault_mag = -alpha * Pm0_gen
    fault_duration = clear_time - fault_time
    data1 = run_dae_simulation(
        'gen', gen_idx, fault_mag, params=params,
        fault_time=0.0, sim_end_time=fault_duration,
        custom_initial_state=state_at_fault,
        gov_R=GOVERNOR_R, gov_Tg=GOVERNOR_TG)
    state_at_clear = data1['full_states'][-1, :].copy()

    sd = get_system_data(params)
    return state_at_clear, sd['Y_adm'].copy()   # gen faults: topology unchanged


def run_load_fault_to_clearing(params, bus_idx, fault_mag, fault_time, clear_time):
    """Stage 1+2 for a load fault: fault_mag added directly to Y_adm[bus,bus]
    (matching Load_Fault.update()). Topology reverts at clearing (same as
    confirmed for line faults), so returns the ORIGINAL Y_adm for use post-clear."""
    sd = get_system_data(params)
    Y_adm_original = sd['Y_adm'].copy()
    Y_adm_fault = Y_adm_original.copy()
    Y_adm_fault[bus_idx, bus_idx] += fault_mag

    data0 = run_dae_simulation(
        'none', 0, 0.0, params=params,
        fault_time=0.0, sim_end_time=fault_time,
        gov_R=GOVERNOR_R, gov_Tg=GOVERNOR_TG)
    state_at_fault = data0['full_states'][-1, :].copy()

    fault_duration = clear_time - fault_time
    data1 = run_dae_simulation(
        'none', 0, 0.0, params=params,
        fault_time=0.0, sim_end_time=fault_duration,
        custom_initial_state=state_at_fault,
        Y_adm_override=Y_adm_fault,
        gov_R=GOVERNOR_R, gov_Tg=GOVERNOR_TG)
    state_at_clear = data1['full_states'][-1, :].copy()

    return state_at_clear, Y_adm_original


def run_line_fault_to_clearing(params, bus_r, bus_s, fault_time, clear_time):
    """Stage 1+2: runs a transmission-line fault from t=0 through t_clear,
    using the REAL DAE (not the PINN). Returns the true state AT t_clear,
    ready to have a correction applied from there."""
    sd = get_system_data(params)
    Y_adm_original = sd['Y_adm'].copy()
    Y_adm_fault = Y_adm_original.copy()
    line_adm = Y_adm_fault[bus_r, bus_s]
    Y_adm_fault[bus_r, bus_s] = 0.0
    Y_adm_fault[bus_s, bus_r] = 0.0
    Y_adm_fault[bus_r, bus_r] += line_adm
    Y_adm_fault[bus_s, bus_s] += line_adm

    # stage 1: pre-fault, t=0 -> fault_time
    data0 = run_dae_simulation(
        'none', 0, 0.0, params=params,
        fault_time=0.0, sim_end_time=fault_time,
        gov_R=GOVERNOR_R, gov_Tg=GOVERNOR_TG)
    state_at_fault = data0['full_states'][-1, :].copy()

    # stage 2: faulted, fault_time -> clear_time
    fault_duration = clear_time - fault_time
    data1 = run_dae_simulation(
        'none', 0, 0.0, params=params,
        fault_time=0.0, sim_end_time=fault_duration,
        custom_initial_state=state_at_fault,
        Y_adm_override=Y_adm_fault,
        gov_R=GOVERNOR_R, gov_Tg=GOVERNOR_TG)
    state_at_clear = data1['full_states'][-1, :].copy()

    return state_at_clear, Y_adm_original


def run_gen_fault_to_clearing(params, gen_idx, alpha, fault_time, clear_time):
    """
    Gen-trip equivalent of run_line_fault_to_clearing -- follows the EXACT
    same pattern confirmed in run_dae_with_clearing's "Path A" (the
    standard 2-stage contingency path used for gen/load faults): one
    continuous simulation from t=0 to t_clear with the 'gen' contingency
    applied internally at fault_time, then an algebraic correction to get
    a clean handoff state.

    fault_mag follows the same formula confirmed early in this project:
    fault_mag = -alpha * Pm0[gen_idx] (negative, additive shift to the
    governor setpoint).
    """
    from cross_validation_datagen_governor import _algebraic_correction

    sd = get_system_data(params)
    Y_adm_original = sd['Y_adm'].copy()

    # need Pm0[gen_idx] for the fault_mag formula -- get it from a quick
    # reference call (fixed system property, independent of this scenario)
    ref = run_dae_simulation('none', 0, 0.0, params=params,
                              fault_time=0.0, sim_end_time=0.01,
                              gov_R=GOVERNOR_R, gov_Tg=GOVERNOR_TG)
    Pm0 = ref['Pm0']
    fault_mag = -alpha * Pm0[gen_idx]

    # single continuous call, t=0 -> clear_time, with the gen contingency
    # applied internally at fault_time (matches Path A exactly)
    data1 = run_dae_simulation(
        'gen', gen_idx, fault_mag, params=params,
        fault_time=fault_time, sim_end_time=clear_time,
        gov_R=GOVERNOR_R, gov_Tg=GOVERNOR_TG)
    faulted_state = data1['full_states'][-1, :].copy()

    # algebraic correction at the clearing switch (matches Path A)
    state_at_clear = _algebraic_correction(
        faulted_state, Y_adm_original, data1, params,
        gov_R=GOVERNOR_R, gov_Tg=GOVERNOR_TG)

    return state_at_clear, Y_adm_original


def run_load_fault_to_clearing(params, bus_idx, alpha, fault_time, clear_time):
    """
    Load-fault equivalent, same Path A pattern as run_gen_fault_to_clearing.

    fault_mag = alpha DIRECTLY (confirmed from generate_data_v9.py line
    217: fault_mag=float(alpha) -- no scaling formula, unlike gen faults).
    Load_Fault.update() adds this directly to Yadmittance[bus,bus] (a
    shunt admittance increase representing more load at that bus).
    """
    from cross_validation_datagen_governor import _algebraic_correction

    sd = get_system_data(params)
    Y_adm_original = sd['Y_adm'].copy()

    data1 = run_dae_simulation(
        'load', bus_idx, float(alpha), params=params,
        fault_time=fault_time, sim_end_time=clear_time,
        gov_R=GOVERNOR_R, gov_Tg=GOVERNOR_TG)
    faulted_state = data1['full_states'][-1, :].copy()

    state_at_clear = _algebraic_correction(
        faulted_state, Y_adm_original, data1, params,
        gov_R=GOVERNOR_R, gov_Tg=GOVERNOR_TG)

    return state_at_clear, Y_adm_original


def apply_delta_pm_sequence_real_dae(params, state_at_clear, Y_adm_post,
                                      delta_pm_sequence, target_gen, dt=0.05):
    """
    SINGLE-generator version -- chains one short REAL-DAE call per timestep
    via the built-in 'gen' contingency mechanism. Kept for simple
    single-generator corrections; see apply_delta_pm_sequence_multigen for
    corrections spanning multiple generators simultaneously (which the
    contingency mechanism can't express directly).
    """
    state = state_at_clear.copy()
    all_omega = []
    all_t = []
    t_cursor = 0.0

    for step, dpm in enumerate(delta_pm_sequence):
        data = run_dae_simulation(
            'gen', target_gen, float(dpm), params=params,
            fault_time=0.0, sim_end_time=dt,
            custom_initial_state=state,
            Y_adm_override=Y_adm_post,
            gov_R=GOVERNOR_R, gov_Tg=GOVERNOR_TG)
        state = data['full_states'][-1, :].copy()
        omega = data['full_states'][-1, [1, 3, 5]]
        all_omega.append(omega)
        t_cursor += dt
        all_t.append(t_cursor)

    return np.array(all_t), np.array(all_omega)


def apply_delta_pm_sequence_multigen(params, state_at_clear, Y_adm_post,
                                      delta_pm_sequence, dt=0.05, N_gen=3):
    """
    MULTI-generator version -- for corrections applied to more than one
    generator at once (which is what find_minimum_delta_pm_timevarying
    actually optimizes: a (n_steps, N_gen) array). The built-in 'gen'
    contingency mechanism only supports ONE location per call, so this
    bypasses it entirely: constructs Trapezoidal_dae_multitimestep directly
    with Pm0_const = original_baseline + delta_pm_vec (all N_gen entries
    at once), fresh each timestep, using the SAME fixed system properties
    (Eq_prime, Ed_prime, H, dampings, stator matrices, V_lf) pulled once
    from a real run_dae_simulation call.

    delta_pm_sequence: (n_steps, N_gen) array/tensor -- one row per
    timestep, one column per generator.
    """
    import torch as _torch
    from cross_validation_datagen_governor import (
        get_system_data, get_machine_params, Trapezoidal_dae_multitimestep, Machine_Models)

    sd = get_system_data(params)
    N_bus = sd['N_bus']
    machine_buses_py = sd['machine_buses']
    sl_array = sd['pl_ini'] + 1j * sd['ql_ini']
    H, dampings, Xd_p, Rs, stator_matrices = get_machine_params(params)

    # pull Eq_prime/Ed_prime/V_lf/Pm0_baseline once from a real call
    # (these are fixed system/operating-point properties, independent of
    # the correction being tested)
    ref = run_dae_simulation('none', 0, 0.0, params=params,
                              fault_time=0.0, sim_end_time=0.01,
                              gov_R=GOVERNOR_R, gov_Tg=GOVERNOR_TG)
    Eq_prime_const = ref['Eq_prime']
    Ed_prime_const = ref['Ed_prime']
    V_lf = ref['V_lf']
    Pm0_baseline = ref['Pm0']

    state = state_at_clear.copy()
    all_omega = []
    all_t = []
    t_cursor = 0.0

    for step in range(len(delta_pm_sequence)):
        dpm_vec = np.asarray(delta_pm_sequence[step], dtype=float)
        Pm0_corrected = Pm0_baseline + dpm_vec

        solver = Trapezoidal_dae_multitimestep(
            stator_matrices, Y_adm_post, Pm0_baseline, machine_buses_py,
            sl_array, V_lf,
            state, Eq_prime_const, Ed_prime_const,
            Pm0_corrected,          # <-- full multi-gen correction, direct
            dt, 0.01, "trapezoidal",
            gov_R=GOVERNOR_R, gov_Tg=GOVERNOR_TG)

        machine_model = Machine_Models(
            60, _torch.tensor(dampings, dtype=_torch.float64),
            _torch.tensor(H, dtype=_torch.float64),
            _torch.tensor(Eq_prime_const, dtype=_torch.float64),
            _torch.tensor(Ed_prime_const, dtype=_torch.float64),
            Pm0_const=_torch.tensor(Pm0_corrected, dtype=_torch.float64),
            gov_R=GOVERNOR_R, gov_Tg=GOVERNOR_TG)
        methods_map = machine_model.return_methods_mapping()
        solver.upload_dynamic_model(machine_model, "classical", methods_map)
        # no contingency applied -- Pm0_const is already the corrected target
        time_arr, states_arr = solver.iterations_trapezoidal()

        state = states_arr[-1, :].numpy().copy() if hasattr(states_arr[-1, :], 'numpy') else np.array(states_arr[-1, :])
        omega = state[[1, 3, 5]]
        all_omega.append(omega)
        t_cursor += dt
        all_t.append(t_cursor)

    return np.array(all_t), np.array(all_omega)


if __name__ == "__main__":
    with open('ieee9_parameters.json') as f:
        params = json.load(f)

    print(__doc__)
    print("This module is meant to be imported. See the accompanying")
    print("verify_line_46_example.py for a full worked verification.")
