"""
cross_validation_datagen_governor.py
DAE-based cross-validation data generator for the IEEE 9-bus PINN framework.
Extended with a first-order droop governor on each generator.

Governor model (droop + first-order lag)
Each generator gains one new differential state: Pm_i (mechanical power).

    dPm_i/dt = (1/Tg) * (Pm0_i - (omega_i - 1.0)/R - Pm_i)

where:
    Pm0_i   = initial (pre-fault) mechanical power [pu]  -- set-point
    omega_i = rotor speed [pu absolute]
    R       = droop coefficient (default 0.05, i.e. 5% regulation)
    Tg      = governor time constant [s]  (default 0.2 s)

State vector layout (33 states for 3-generator IEEE 9-bus)
OLD (30 states):
    [0:6]   differential  -- delta, omega  x3 generators
    [6:12]  algebraic     -- Id, Iq        x3 generators
    [12:30] network       -- Vm, theta     x9 buses

NEW (33 states):
    [0:6]   differential  -- delta, omega  x3 generators  (unchanged)
    [6:9]   governor      -- Pm            x3 generators  (NEW)
    [9:15]  algebraic     -- Id, Iq        x3 generators  (shifted +3)
    [15:33] network       -- Vm, theta     x9 buses       (shifted +3)

Design principle
This module is a **pure physics engine**.  All high-level decisions about
fault configurations, clearing strategies, and parameter sweeps are made in
``datagen.py``.  This module exposes two public entry points:

    run_dae_simulation(...)        — single-stage DAE integration
    run_dae_with_clearing(...)     — multi-stage DAE with fault clearing

Bug fixes carried over from previous version
BUG B — run_dae_with_clearing 3-stage line-fault path (unchanged).
BUG #2 — omega convention: absolute pu (equilibrium = 1.0).
BUG #3 — Pm clearing for type-3 faults (unchanged).
BUG #4 — hardcoded machine parameters replaced (unchanged).
"""

import numpy as np
import sys
import json
import torch
from torch.autograd.functional import jacobian
from abc import ABC, abstractmethod


# Governor default parameters

GOVERNOR_R  = 0.05   # droop coefficient (5% speed regulation)
GOVERNOR_TG = 0.2    # governor time constant [s]

# System data loader

def get_system_data(params: dict) -> dict:
    N_bus      = params['system']['buses']
    gen_buses  = params['system']['gen_buses_0indexed']
    load_buses = [i for i in range(N_bus) if i not in gen_buses]

    Y_real = np.array(params['network']['Y_bus']['real'])
    Y_imag = np.array(params['network']['Y_bus']['imag'])
    Y_adm  = Y_real + 1j * Y_imag

    N   = len(gen_buses)
    H   = np.array([params['generators'][f'gen_{i+1}']['H_sec']       for i in range(N)])
    D   = np.array([params['generators'][f'gen_{i+1}']['D_pu']        for i in range(N)])
    Xdp = np.array([params['generators'][f'gen_{i+1}']['Xd_prime_pu'] for i in range(N)])

    vm_ini = np.array([params['buses'][f'bus_{i+1}']['vm_pu']  for i in range(N_bus)])
    va_ini = np.array([params['buses'][f'bus_{i+1}']['va_rad'] for i in range(N_bus)])
    pg_ini = np.array([params['buses'][f'bus_{i+1}']['pg_pu']  for i in range(N_bus)])
    qg_ini = np.array([params['buses'][f'bus_{i+1}']['qg_pu']  for i in range(N_bus)])
    pl_ini = np.array([params['buses'][f'bus_{i+1}']['pl_pu']  for i in range(N_bus)])
    ql_ini = np.array([params['buses'][f'bus_{i+1}']['ql_pu']  for i in range(N_bus)])

    return {
        'N_bus': N_bus, 'N_gen': N,
        'gen_buses': gen_buses, 'load_buses': load_buses,
        'machine_buses': np.array(gen_buses),
        'Y_adm': Y_adm,
        'H': H, 'D': D, 'Xdp': Xdp,
        'vm_ini': vm_ini, 'va_ini': va_ini,
        'pg_ini': pg_ini, 'qg_ini': qg_ini,
        'pl_ini': pl_ini, 'ql_ini': ql_ini,
        'bus_params': params['buses'],
    }

# Machine parameters

def get_machine_params(params: dict):
    """
    Extract H, D (damping), Xd', Rs from the JSON params dict.

    Returns
    H              : (N_gen,) inertia constants [s]
    dampings       : (N_gen,) damping coefficients [pu]
    Xd_p           : (N_gen,) transient reactances [pu]
    Rs             : (N_gen,) stator resistances [pu]  — zero for classical model
    stator_matrices: (N_gen, 2, 2) inverse stator impedance matrices
    """
    N = len(params['system']['gen_buses_0indexed'])
    H        = np.array([params['generators'][f'gen_{i+1}']['H_sec']       for i in range(N)])
    dampings = np.array([params['generators'][f'gen_{i+1}']['D_pu']        for i in range(N)])
    Xd_p     = np.array([params['generators'][f'gen_{i+1}']['Xd_prime_pu'] for i in range(N)])
    Rs       = np.zeros(N)

    stator_matrices = np.zeros((N, 2, 2))
    for i in range(N):
        stator_matrix = np.array([[Rs[i],   -Xd_p[i]],
                                  [Xd_p[i],  Rs[i]  ]], dtype=np.float64)
        stator_matrices[i] = np.linalg.inv(stator_matrix)

    return H, dampings, Xd_p, Rs, stator_matrices


# Machine model (With governor) 
# pm is no longer a fixed constant

class Machine_Models:
    """
    Classical second-order machine model with first-order droop governor.

    Differential states per machine (3 total):
        delta  -- rotor angle [rad]
        omega  -- rotor speed [pu absolute, synchronous = 1.0]
        Pm     -- mechanical power [pu]  ← NEW governor state

    Equations:
        dδ/dt  = (ω − 1) · 2π · f₀
        dω/dt  = (Pm − Pe − D·(ω − 1)) / (2H)
        H is inertia constant
        dPm/dt = (1/Tg) · (Pm0 − (ω − 1)/R − Pm)

    omega convention: absolute per-unit (synchronous speed = 1.0 pu).
    """

    def __init__(self, sys_freq, mdamping, minertia,
                 Eq_prime_const, Ed_prime_const,
                 Pm0_const,
                 gov_R=GOVERNOR_R, gov_Tg=GOVERNOR_TG):
        self.freq           = sys_freq
        self.m_damping_vec  = mdamping          # (N_gen,) tensor
        self.m_inertia_vec  = minertia          # (N_gen,) tensor
        self.Eq_prime_const = Eq_prime_const    # (N_gen,) tensor
        self.Ed_prime_const = Ed_prime_const    # (N_gen,) tensor
        self.Pm0_const      = Pm0_const         # (N_gen,) tensor — governor set-point
        self.gov_R          = gov_R             # droop coefficient
        self.gov_Tg         = gov_Tg            # time constant [s]

    # *
    def return_methods_mapping(self):
        return {"classical": "classical_second_order_machine"}

    # *
    def set_pg_pf(self, pg_pf: torch.Tensor):
        # pg_pf is kept for interface compatibility (used by contingency classes)
        # but Pm is now a state variable — the governor drives it dynamically.
        self.pg_pf = pg_pf

    def classical_second_order_machine(self, diff_states, alg_states, no_machine):
        """
        Returns derivatives for (delta, omega, Pm) for generator no_machine.

        Parameters
        ----------
        diff_states : (3,) tensor  [delta, omega, Pm]
        alg_states  : (3,) tensor  [Id, Iq, Vm, Theta]
                      (Note: Theta not needed here but passed for consistency)
        no_machine  : int

        Returns
        -------
        (3,) tensor  [d_delta, d_omega, d_Pm]
        """
        delta, omega, Pm = diff_states[0], diff_states[1], diff_states[2]
        Id, Iq           = alg_states[0], alg_states[1]

        Eq_prime = self.Eq_prime_const[no_machine]
        Ed_prime = self.Ed_prime_const[no_machine]
        Pm0      = self.Pm0_const[no_machine]

        # Rotor angle
        d_delta = (omega - 1.0) * 2 * torch.pi * self.freq

        # Electrical power
        Pe = Ed_prime * Id + Eq_prime * Iq

        # Swing equation  (Pm is now a state, not the fixed pg_pf)
        d_omega = (Pm - Pe - self.m_damping_vec[no_machine] * (omega - 1.0)
                   ) / (2.0 * self.m_inertia_vec[no_machine])

        # Governor (droop + first-order lag)
        #   Pm_ref = Pm0 - (omega - 1) / R   (droop reference)
        #   dPm/dt = (Pm_ref - Pm) / Tg
        Pm_ref = Pm0 - (omega - 1.0) / self.gov_R
        d_Pm   = (Pm_ref - Pm) / self.gov_Tg

        return torch.stack([d_delta, d_omega, d_Pm])
    

# Contingency classes
# we have gen fault, load and line fault

class Contingency(ABC):
    @abstractmethod
    def update(self, obj, **kwargs): pass

class Generation_Fault(Contingency):
    def update(self, obj, **kwargs):
        ind         = kwargs["ind"]
        f_magnitude = kwargs["f_magnitude"]
        # With governor, Pm0 (set-point) shifts.  Update both Pm0 in machine
        # model and the initial pg_pf reference stored in the solver.
        obj.pg_pf_re[ind] += float(f_magnitude)
        obj.machine_class.Pm0_const[ind] += float(f_magnitude)
        obj.machine_class.set_pg_pf(obj.pg_pf_re)

class Load_Fault(Contingency):
    def update(self, obj, **kwargs):
        ind         = kwargs["ind"]
        f_magnitude = kwargs["f_magnitude"]
        obj.Yadmittance[ind, ind] += f_magnitude

class Topology_Change(Contingency):
    def update(self, obj, **kwargs):
        ind_s = kwargs["ind_s"]
        ind_r = kwargs["ind_r"]
        line_admittance = obj.Yadmittance[ind_r, ind_s]
        obj.Yadmittance[ind_r, ind_s] = 0.
        obj.Yadmittance[ind_s, ind_r] = 0.
        obj.Yadmittance[ind_r, ind_r] += line_admittance
        obj.Yadmittance[ind_s, ind_s] += line_admittance

# *
def contingency_compiler(type_f, time, magnitude, location):
    contingency_map = {"gen": "Generation_Fault", "load": "Load_Fault",
                       "topology": "Topology_Change"}
    if type_f == "none":
        return None, None
    if type_f not in contingency_map:
        raise ValueError(f"Fault type '{type_f}' not in {list(contingency_map)}")
    module_name       = sys.modules[__name__]
    contingency_class = getattr(module_name, contingency_map[type_f])
    contingency_info  = {"fault_type": type_f, "fault_time": time,
                         "fault_magnitude": magnitude, "fault_location": location}
    return contingency_class, contingency_info


# DAE solver  (updated for 33-state layout)
class Trapezoidal_dae_multitimestep():
    """
    Trapezoidal implicit DAE solver.

    State vector layout (N_gen=3, N_bus=9 → 33 states):
        [0 : 2*N]          differential  delta, omega     (N_gen pairs)
        [2*N : 3*N]        governor      Pm               (N_gen)   ← NEW
        [3*N : 5*N]        algebraic     Id, Iq           (N_gen pairs)
        [5*N : 5*N+2*N_bus] network      Vm, theta        (N_bus pairs)
    """

    def __init__(self, stator_clear_matrices, Y_adm, pg_pf_re, gen_indices,
                 pl_pf, V_lf, ini_cond,
                 Eq_prime_const, Ed_prime_const,
                 Pm0_const,
                 t_final, step_size, integration_scheme,
                 gov_R=GOVERNOR_R, gov_Tg=GOVERNOR_TG):
        torch.set_default_dtype(torch.float64)
        self.t_final       = torch.tensor(t_final,   dtype=torch.float64)
        self.step_size     = torch.tensor(step_size, dtype=torch.float64)
        self.initial_state = torch.tensor(ini_cond,  dtype=torch.float64)
        self.no_variables  = self.initial_state.shape[0]
        self.stator_clear_matrices = torch.tensor(stator_clear_matrices, dtype=torch.float64)

        # Constant-impedance load absorption (uses pre-fault V_lf)
        Y_adm_ci = Y_adm.copy()
        n_buses  = Y_adm.shape[0]
        pl_pf_np = np.array(pl_pf) if not isinstance(pl_pf, np.ndarray) else pl_pf
        for k in range(n_buses):
            if abs(pl_pf_np[k]) > 1e-10:
                Y_adm_ci[k, k] += np.conj(pl_pf_np[k]) / (abs(V_lf[k]) ** 2)

        self.Yadmittance  = torch.tensor(Y_adm_ci, dtype=torch.complex128)
        self.no_buses_sys = self.Yadmittance.shape[0]
        self.pg_pf_re     = torch.tensor(pg_pf_re, dtype=torch.float64)
        self.gen_indices  = torch.tensor(gen_indices, dtype=torch.int)
        self.no_gens      = self.gen_indices.shape[0]

        self.Eq_prime_const = torch.tensor(Eq_prime_const, dtype=torch.float64)
        self.Ed_prime_const = torch.tensor(Ed_prime_const, dtype=torch.float64)
        self.Pm0_const      = torch.tensor(Pm0_const,      dtype=torch.float64)

        self.gov_R  = gov_R
        self.gov_Tg = gov_Tg

        self.assign_vars_value()
        # Just builds index lookup tables 
        # so the solver knows where each state lives in the flat 33-vector.
        self.pl_pf = torch.zeros(n_buses, dtype=torch.complex128)
        self.integration_scheme = integration_scheme
        self.cntgcy_sim = False

    def upload_dynamic_model(self, generic_machine_model, selected_model, models_mapping):
        method_name = models_mapping.get(selected_model)
        if method_name is None:
            return generic_machine_model.fallback()
        self.machine_function = getattr(generic_machine_model, method_name, None)
        if self.machine_function is None:
            return generic_machine_model.fallback()
        self.machine_class = generic_machine_model
        self.set_active_generation_machines()

    def set_active_generation_machines(self):
        self.machine_class.set_pg_pf(self.pg_pf_re)

    @torch.enable_grad()
    def trapezoidal_rule_func(self, ini_dif_states, next_dif_states,
                               ini_alg_states, next_alg_states, func, no_machine):
        d_state_0 = func(ini_dif_states, ini_alg_states, no_machine)
        d_state_1 = func(next_dif_states, next_alg_states, no_machine)
        return next_dif_states - ini_dif_states - 0.5 * self.step_size * (d_state_0 + d_state_1)

    def rk_integration_scheme(self, ini_dif_states, next_dif_states,
                               ini_alg_states, next_alg_states, func, no_machine):
        return self.trapezoidal_rule_func(ini_dif_states, next_dif_states,
                                          ini_alg_states, next_alg_states, func, no_machine)

    def machine_stator_equations(self, next_dif_states, next_alg_states, no_machine):
        """
        Algebraic stator constraint.
        next_dif_states: [delta, omega, Pm]  — delta is index 0 (unchanged)
        """
        delta = next_dif_states[0]      # index 0 unchanged
        I_d, I_q, Vm, Theta = next_alg_states
        Eq_prime = self.Eq_prime_const[no_machine]
        Ed_prime = self.Ed_prime_const[no_machine]
        voltage_vector  = torch.stack([Ed_prime - Vm * torch.sin(delta - Theta),
                                       Eq_prime - Vm * torch.cos(delta - Theta)])
        stator_currents = torch.matmul(self.stator_clear_matrices[no_machine], voltage_vector)
        return torch.stack([I_d - stator_currents[0], I_q - stator_currents[1]])

    def update_sys_angles(self, states):
        return states # okay

    def assign_vars_value(self):
        """
        Build index arrays for the NEW 33-state layout.

        Layout:
            [0 : 2*N]            delta, omega   (differential)
            [2*N : 3*N]          Pm             (governor, differential)
            [3*N : 5*N]          Id, Iq         (algebraic)
            [5*N : 5*N + 2*N_bus] Vm, theta     (network)
        """
        N  = self.no_gens
        NB = self.no_buses_sys

        # --- differential state indices: [delta_i, omega_i, Pm_i]  shape (N, 3) ---
        self.dif_states_indices = torch.zeros((N, 3), dtype=torch.int32)
        for i in range(N):
            # delta at 2i, omega at 2i+1
            self.dif_states_indices[i, 0] = 2 * i        # delta
            self.dif_states_indices[i, 1] = 2 * i + 1    # omega
            self.dif_states_indices[i, 2] = 2 * N + i    # Pm  ← NEW

        # --- algebraic state indices: [Id, Iq, Vm_gen, Theta_gen]  shape (N, 4) ---
        # Id at 3N + 2i,  Iq at 3N + 2i+1
        # Vm, Theta for the generator bus (from network block at 5N + 2*bus_idx)
        self.alg_states_indices = torch.zeros((N, 4), dtype=torch.int32)
        for i in range(N):
            self.alg_states_indices[i, 0] = 3 * N + 2 * i          # Id
            self.alg_states_indices[i, 1] = 3 * N + 2 * i + 1      # Iq
            self.alg_states_indices[i, 2] = 5 * N + self.gen_indices[i] * 2       # Vm
            self.alg_states_indices[i, 3] = 5 * N + self.gen_indices[i] * 2 + 1   # Theta

        # Convenience aliases used in update_function_pure_physics
        self.currents_d_axis = torch.zeros(N,  dtype=torch.int32)
        self.currents_q_axis = torch.zeros(N,  dtype=torch.int32)
        self.deltas_local_rs = torch.zeros(N,  dtype=torch.int32)
        for i in range(N):
            self.currents_d_axis[i] = 3 * N + 2 * i          # Id
            self.currents_q_axis[i] = 3 * N + 2 * i + 1      # Iq
            self.deltas_local_rs[i] = 2 * i                   # delta

        self.Vm_indices    = torch.zeros(NB, dtype=torch.int32)
        self.Theta_indices = torch.zeros(NB, dtype=torch.int32)
        for i in range(NB):
            self.Vm_indices[i]    = 5 * N + i * 2
            self.Theta_indices[i] = 5 * N + i * 2 + 1

    def update_function_pure_physics(self, states_x0, states_x1):
        """
        Residual vector F(x0, x1) for the trapezoidal implicit step.
        Includes governor state Pm as a differential variable.
        """
        results_final = torch.zeros(self.no_variables)

        for no_mac in range(self.no_gens):
            # Differential states: [delta, omega, Pm]  (3 elements)
            dif_var_ini_m = states_x0[self.dif_states_indices[no_mac]]
            dif_var_new_m = states_x1[self.dif_states_indices[no_mac]]

            # Algebraic states for machine equations: [Id, Iq] only (2 elements)
            # (Vm, Theta also needed for stator equations — pass all 4)
            alg_var_ini_m = states_x0[self.alg_states_indices[no_mac][:-2]]   # [Id, Iq] at t0
            alg_var_new_m = states_x1[self.alg_states_indices[no_mac]]        # [Id, Iq, Vm, Theta]

            # Trapezoidal residual for [delta, omega, Pm]
            results_final[self.dif_states_indices[no_mac]] = self.rk_integration_scheme(
                dif_var_ini_m, dif_var_new_m, alg_var_ini_m, alg_var_new_m,
                self.machine_function, no_mac)

            # Stator algebraic equations (2 residuals for Id, Iq)
            results_final[self.alg_states_indices[no_mac][:2]] = self.machine_stator_equations(
                dif_var_new_m, alg_var_new_m, no_mac)

        # Network KCL residuals
        complex_vm   = states_x1[self.Vm_indices] * torch.exp(1j * states_x1[self.Theta_indices])
        currents_net = torch.matmul(self.Yadmittance, complex_vm)

        generator_currents = ((states_x1[self.currents_d_axis] + 1j * states_x1[self.currents_q_axis])
                              * torch.exp(1j * (states_x1[self.deltas_local_rs] - torch.pi / 2)))

        load_currents = torch.zeros(self.no_buses_sys, dtype=torch.complex128)
        gen_currents  = torch.zeros(self.no_buses_sys, dtype=torch.complex128)
        gen_currents[self.gen_indices] = generator_currents
        total_currents = gen_currents - load_currents - currents_net

        results_final[self.Vm_indices]    = torch.real(total_currents)
        results_final[self.Theta_indices] = torch.imag(total_currents)
        return results_final

    def trapezoidal_method(self, states_x0):
        states_x1          = states_x0.detach().clone()
        it_count           = 0
        residual_iteration = torch.ones(self.no_variables, dtype=torch.float64)
        num_max_iterations = 15
        tolerance_newton   = torch.tensor(1e-8, dtype=torch.float64)

        def wrapped_update_function(states_x1):
            return self.update_function_pure_physics(states_x0, states_x1)

        while (torch.max(torch.abs(residual_iteration)) > tolerance_newton
               and it_count < num_max_iterations):
            states_x1.requires_grad_(True)
            jacobian_matrix    = jacobian(wrapped_update_function, states_x1)
            states_x1.requires_grad_(False)
            inverse_matrix     = torch.linalg.inv(jacobian_matrix)
            residual_iteration = self.update_function_pure_physics(states_x0, states_x1)
            with torch.no_grad():
                x_increment = -torch.matmul(inverse_matrix, residual_iteration)
                states_x1  += x_increment
            it_count += 1

        states_x1 = self.update_sys_angles(states_x1)
        return states_x1

    def apply_contingency(self, contingency_class, contingency_information):
        if contingency_class is None or contingency_information is None:
            return
        self.cntgcy_sim      = True
        self.cntgcy_class    = contingency_class()
        self.cntgcy_time     = contingency_information["fault_time"]
        self.cntgcy_magn     = contingency_information["fault_magnitude"]
        self.cntgcy_location = contingency_information["fault_location"]

    def iterations_trapezoidal(self):
        no_steps = int(self.t_final / self.step_size) + 1

        if not self.cntgcy_sim:
            time_array   = torch.arange(no_steps) * self.step_size
            states_array = torch.zeros((no_steps, self.no_variables), dtype=torch.float64)
        else:
            step_dist    = int(self.cntgcy_time / self.step_size)
            no_steps    += 1
            time_raw     = torch.arange(no_steps - 1) * self.step_size
            time_array   = torch.cat((time_raw[:step_dist + 1],
                                      time_raw[step_dist:step_dist + 1],
                                      time_raw[step_dist + 1:]))
            states_array = torch.zeros((no_steps, self.no_variables), dtype=torch.float64)

        states_array[0, :] = self.initial_state
        current_state      = self.initial_state

        for i in range(1, no_steps):
            if self.cntgcy_sim and i == int(self.cntgcy_time / self.step_size) + 1:
                self.cntgcy_class.update(self, ind=self.cntgcy_location,
                                          f_magnitude=self.cntgcy_magn)
                temp_step      = self.step_size
                self.step_size = torch.tensor(0.)
                next_state     = self.trapezoidal_method(current_state)
                self.step_size = temp_step
            else:
                next_state = self.trapezoidal_method(current_state)

            states_array[i, :] = next_state
            current_state      = next_state

        return time_array.numpy(), states_array.numpy()



# Power flow (unchanged)

# finds steady state voltage across all 9 buses before any fault occurs
# that result becomes initial condition for the dynamic simulation
def run_loadflow(Y, bus_params, N_bus):
    Vm   = np.array([bus_params[f'bus_{i+1}']['vm_pu']   for i in range(N_bus)])
    Va   = np.array([bus_params[f'bus_{i+1}']['va_rad']  for i in range(N_bus)])
    Psch = np.array([bus_params[f'bus_{i+1}']['pg_pu']   for i in range(N_bus)]) \
         - np.array([bus_params[f'bus_{i+1}']['pl_pu']   for i in range(N_bus)])
    Qsch = np.array([bus_params[f'bus_{i+1}']['qg_pu']   for i in range(N_bus)]) \
         - np.array([bus_params[f'bus_{i+1}']['ql_pu']   for i in range(N_bus)])
    bt   = np.array([bus_params[f'bus_{i+1}']['bus_type'] for i in range(N_bus)])
    pq   = [i for i in range(N_bus) if bt[i] == 3]
    pv   = [i for i in range(N_bus) if bt[i] == 2]
    ns   = pv + pq

    for _ in range(20):
        V  = Vm * np.exp(1j * Va) # complex voltage
        Sc = V * np.conj(Y @ V) # apparent power
        dP = Psch[ns] - Sc.real[ns] # mismatch compute
        dQ = Qsch[pq] - Sc.imag[pq] #   "         "

        if np.max(np.abs(np.concatenate([dP, dQ]))) < 1e-9:
            break
        I    = Y @ V
        dVm  = np.diag(V);  dI = np.diag(I);  dVn = np.diag(V / np.abs(V))
        dSth = 1j * dVm @ np.conj(dI - Y @ dVm)
        dSV  = dVm @ np.conj(Y @ dVn) + np.conj(dI) @ dVn
        J    = np.block([[dSth.real[np.ix_(ns, ns)], dSV.real[np.ix_(ns, pq)]],
                         [dSth.imag[np.ix_(pq, ns)], dSV.imag[np.ix_(pq, pq)]]])
        dx   = np.linalg.solve(J, np.concatenate([dP, dQ]))
        Va[ns] += dx[:len(ns)]
        Vm[pq] += dx[len(ns):]
    V_lf = Vm * np.exp(1j * Va)
    S_lf = V_lf * np.conj(Y @ V_lf)
    return V_lf, S_lf


# Single-stage simulation runner

def run_dae_simulation(fault_type, fault_loc, fault_mag,
                       params: dict,
                       fault_time=0.2, sim_end_time=2.0,
                       custom_initial_state=None,
                       Y_adm_override=None,
                       Y_adm_fault=None,
                       gov_R=GOVERNOR_R, gov_Tg=GOVERNOR_TG):
    """
    Run one stage of the DAE simulation with governor dynamics.

    State vector is 33 elements (N_gen=3, N_bus=9):
        [0:6]   delta, omega  (differential)
        [6:9]   Pm            (governor, differential)  ← NEW
        [9:15]  Id, Iq        (algebraic)
        [15:33] Vm, theta     (network)

    Parameters (new)
    gov_R  : float
        Droop coefficient (default 0.05).
    gov_Tg : float
        Governor time constant [s] (default 0.2).

    All other parameters identical to the original run_dae_simulation.
    """
    sd               = get_system_data(params)
    Y_adm            = sd['Y_adm']
    N_bus            = sd['N_bus']
    N_gen            = sd['N_gen']
    Xdp              = sd['Xdp']
    machine_buses_py = sd['machine_buses']
    sl_array         = sd['pl_ini'] + 1j * sd['ql_ini']

    V_lf_all, S_lf_all = run_loadflow(Y_adm, sd['bus_params'], N_bus)
    final_volt_magn     = np.abs(V_lf_all)
    final_volt_angle    = np.angle(V_lf_all)
    final_active_p      = S_lf_all.real
    final_reactive_p    = S_lf_all.imag
    V_lf                = V_lf_all

    H, dampings, Xd_p, Rs, stator_matrices = get_machine_params(params)

    voltages_machines = V_lf[machine_buses_py]
    powers_machines_p = final_active_p[machine_buses_py]
    powers_machines_q = final_reactive_p[machine_buses_py]

    Ig_ic      = np.conj(powers_machines_p + 1j * powers_machines_q) / np.conj(voltages_machines)
    E_prime_ic = voltages_machines + 1j * Xdp * Ig_ic
    delta0_ic  = np.angle(E_prime_ic)
    omega0_ic  = np.ones(N_gen)   # absolute pu

    Id_Iq_complex = Ig_ic * np.exp(-1j * (delta0_ic - np.pi / 2))
    Id_ic = np.real(Id_Iq_complex)
    Iq_ic = np.imag(Id_Iq_complex)
    Vd_ic = np.real(voltages_machines * np.exp(-1j * (delta0_ic - np.pi / 2)))
    Vq_ic = np.imag(voltages_machines * np.exp(-1j * (delta0_ic - np.pi / 2)))

    Eq_prime_const = Vq_ic + Xd_p * Id_ic
    Ed_prime_const = Vd_ic - Xd_p * Iq_ic
    tm_array       = Ed_prime_const * Id_ic + Eq_prime_const * Iq_ic  # initial Pm

    # Build initial condition vector (33 states) 
    dif_vars_ini = np.zeros(N_gen * 2)    # delta, omega
    gov_vars_ini = tm_array.copy()         # Pm = Pm0 at t=0 (equilibrium)
    alg_vars_ini = np.zeros(N_gen * 2)    # Id, Iq

    for i in range(N_gen):
        dif_vars_ini[i * 2]     = delta0_ic[i]
        dif_vars_ini[i * 2 + 1] = omega0_ic[i]
        alg_vars_ini[i * 2]     = Id_ic[i]
        alg_vars_ini[i * 2 + 1] = Iq_ic[i]

    if custom_initial_state is not None:
        sim_variables = custom_initial_state.copy()
    else:
        voltage_angle_intercalats = np.ravel(np.column_stack((final_volt_magn, final_volt_angle)))
        # Layout: [delta,omega | Pm | Id,Iq | Vm,theta]
        sim_variables = np.concatenate([dif_vars_ini,
                                        gov_vars_ini,
                                        alg_vars_ini,
                                        voltage_angle_intercalats])

    machine_model = Machine_Models(
        60, dampings, H,
        Eq_prime_const, Ed_prime_const,
        Pm0_const  = tm_array,
        gov_R  = gov_R,
        gov_Tg = gov_Tg
    )
    # Convert to tensors so Machine_Models can do arithmetic
    machine_model.m_damping_vec  = torch.tensor(dampings,       dtype=torch.float64)
    machine_model.m_inertia_vec  = torch.tensor(H,              dtype=torch.float64)
    machine_model.Eq_prime_const = torch.tensor(Eq_prime_const, dtype=torch.float64)
    machine_model.Ed_prime_const = torch.tensor(Ed_prime_const, dtype=torch.float64)
    machine_model.Pm0_const      = torch.tensor(tm_array,       dtype=torch.float64)

    methods_map = machine_model.return_methods_mapping()

    contingency_lib, contingency_info = contingency_compiler(
        fault_type, fault_time, fault_mag, fault_loc)

    if Y_adm_fault is not None:
        Y_adm_for_solver = Y_adm_fault
    elif Y_adm_override is not None:
        Y_adm_for_solver = Y_adm_override
    else:
        Y_adm_for_solver = Y_adm

    solver = Trapezoidal_dae_multitimestep(
        stator_matrices, Y_adm_for_solver, tm_array, machine_buses_py,
        sl_array, V_lf,
        sim_variables, Eq_prime_const, Ed_prime_const,
        tm_array,              # Pm0_const
        sim_end_time, 0.01, "trapezoidal",
        gov_R=gov_R, gov_Tg=gov_Tg)

    solver.upload_dynamic_model(machine_model, "classical", methods_map)
    solver.apply_contingency(contingency_lib, contingency_info)
    time_array, states_array = solver.iterations_trapezoidal()

    # Remove duplicate point inserted at fault application
    if contingency_lib is not None:
        step_dist = int(fault_time / 0.01)
        dup_idx   = step_dist + 1
        keep      = np.ones(len(time_array), dtype=bool)
        keep[dup_idx] = False
        time_array   = time_array[keep]
        states_array = states_array[keep]

    return {
        'time'           : time_array,
        'full_states'    : states_array,
        'machine_buses'  : machine_buses_py,
        'Eq_prime'       : Eq_prime_const,
        'Ed_prime'       : Ed_prime_const,
        'V_lf'           : V_lf,
        'tm_array'       : tm_array,
        'sl_array'       : sl_array,
        'stator_matrices': stator_matrices,
        'H'              : H,
        'dampings'       : dampings,
        'Pm0'            : tm_array,    # governor set-point (for downstream use)
    }


# Internal algebraic correction helper

def _algebraic_correction(faulted_state: np.ndarray,
                           Y_adm_cleared: np.ndarray,
                           stage_result: dict,
                           params: dict,
                           gov_R=GOVERNOR_R, gov_Tg=GOVERNOR_TG) -> np.ndarray:
    """
    Enforce KCL on the next-stage network at an instantaneous switching event.
    Governor states (Pm) are differential and are carried through unchanged —
    only the algebraic states (Id, Iq, Vm, theta) are corrected.
    """
    sd       = get_system_data(params)
    V_lf     = stage_result['V_lf']
    sl_array = stage_result['sl_array']
    tm_array = stage_result['tm_array']
    Pm0      = stage_result.get('Pm0', tm_array)
    Eq_prime = stage_result['Eq_prime']
    Ed_prime = stage_result['Ed_prime']
    H        = stage_result['H']
    dampings = stage_result['dampings']
    stator_matrices = stage_result['stator_matrices']


    
    solver_clean = Trapezoidal_dae_multitimestep(
        stator_matrices, Y_adm_cleared, tm_array, sd['machine_buses'],
        sl_array, V_lf,
        faulted_state, Eq_prime, Ed_prime,
        Pm0,
        0.01, 0.01, "trapezoidal",
        gov_R=gov_R, gov_Tg=gov_Tg)

    machine_model = Machine_Models(
        60, dampings, H, Eq_prime, Ed_prime, Pm0,
        gov_R=gov_R, gov_Tg=gov_Tg)
    machine_model.m_damping_vec  = torch.tensor(dampings, dtype=torch.float64)
    machine_model.m_inertia_vec  = torch.tensor(H,        dtype=torch.float64)
    machine_model.Eq_prime_const = torch.tensor(Eq_prime, dtype=torch.float64)
    machine_model.Ed_prime_const = torch.tensor(Ed_prime, dtype=torch.float64)
    machine_model.Pm0_const      = torch.tensor(Pm0,      dtype=torch.float64)

    solver_clean.upload_dynamic_model(machine_model, "classical",
                                      machine_model.return_methods_mapping())

    current_state          = torch.tensor(faulted_state, dtype=torch.float64)
    solver_clean.step_size = torch.tensor(0.0, dtype=torch.float64)
    corrected_state        = solver_clean.trapezoidal_method(current_state)
    return corrected_state.detach().numpy()


# Multi-stage simulation with fault clearing

def run_dae_with_clearing(fault_type, fault_loc, fault_mag,
                          params: dict,
                          fault_time=0.2, clear_time=1.2, total_time=2.0,
                          Y_bus_post=None,
                          Y_adm_fault=None,
                          gov_R=GOVERNOR_R, gov_Tg=GOVERNOR_TG):
    """
    Run a simulation with fault application at fault_time and clearing at
    clear_time.  Governor dynamics included.

    Parameters identical to the original, plus:
    gov_R  : float  — droop coefficient (default 0.05)
    gov_Tg : float  — governor time constant [s] (default 0.2)

    Returns
    dict with keys: 'time', 'full_states', 'machine_buses',
                    'Eq_prime', 'Ed_prime'

    Note: full_states has shape (T, 33).
    State layout reminder:
        cols 0:6   — delta, omega
        cols 6:9   — Pm  (governor)
        cols 9:15  — Id, Iq
        cols 15:33 — Vm, theta
    """
    sd             = get_system_data(params)
    Y_adm_original = sd['Y_adm']
    Y_adm_cleared  = Y_bus_post if Y_bus_post is not None else Y_adm_original

    gov_kwargs = dict(gov_R=gov_R, gov_Tg=gov_Tg)

    
    # Path B — 3-stage line fault path
   
    if Y_adm_fault is not None:
        data0 = run_dae_simulation(
            'none', fault_loc, 0.0, params=params,
            fault_time=0.0, sim_end_time=fault_time, **gov_kwargs)
        prefault_end_state = data0['full_states'][-1, :].copy()

        corrected_at_fault = _algebraic_correction(
            prefault_end_state, Y_adm_fault, data0, params, **gov_kwargs)

        fault_duration = clear_time - fault_time
        data1 = run_dae_simulation(
            'none', fault_loc, 0.0, params=params,
            fault_time=0.0, sim_end_time=fault_duration,
            custom_initial_state=corrected_at_fault,
            Y_adm_override=Y_adm_fault, **gov_kwargs)
        faulted_end_state = data1['full_states'][-1, :].copy()

        corrected_at_clear = _algebraic_correction(
            faulted_end_state, Y_adm_cleared, data1, params, **gov_kwargs)

        remaining_time = total_time - clear_time
        data2 = run_dae_simulation(
            'none', fault_loc, 0.0, params=params,
            fault_time=0.0, sim_end_time=remaining_time,
            custom_initial_state=corrected_at_clear,
            Y_adm_override=Y_adm_cleared, **gov_kwargs)

        stage1_time_shifted = data1['time'][1:] + fault_time
        stage2_time_shifted = data2['time'][1:] + clear_time
        full_time   = np.concatenate([data0['time'], stage1_time_shifted, stage2_time_shifted])
        full_states = np.concatenate([data0['full_states'],
                                      data1['full_states'][1:],
                                      data2['full_states'][1:]])
        torch.set_default_dtype(torch.float32)
        return {
            'time'         : full_time,
            'full_states'  : full_states,
            'machine_buses': data0['machine_buses'],
            'Eq_prime'     : data0['Eq_prime'],
            'Ed_prime'     : data0['Ed_prime'],
        }

    # Path A — standard 2-stage contingency path
    
    data1 = run_dae_simulation(
        fault_type, fault_loc, fault_mag, params=params,
        fault_time=fault_time, sim_end_time=clear_time, **gov_kwargs)
    faulted_state = data1['full_states'][-1, :].copy()

    corrected_state_np = _algebraic_correction(
        faulted_state, Y_adm_cleared, data1, params, **gov_kwargs)

    remaining_time = total_time - clear_time
    data2 = run_dae_simulation(
        'none', fault_loc, 0.0, params=params,
        fault_time=0.0, sim_end_time=remaining_time,
        custom_initial_state=corrected_state_np,
        Y_adm_override=Y_adm_cleared, **gov_kwargs)

    stage2_time_shifted = data2['time'][1:] + clear_time
    full_time   = np.concatenate([data1['time'], stage2_time_shifted])
    full_states = np.concatenate([data1['full_states'], data2['full_states'][1:]])

    torch.set_default_dtype(torch.float32)
    return {
        'time'         : full_time,
        'full_states'  : full_states,
        'machine_buses': data1['machine_buses'],
        'Eq_prime'     : data1['Eq_prime'],
        'Ed_prime'     : data1['Ed_prime'],
    }