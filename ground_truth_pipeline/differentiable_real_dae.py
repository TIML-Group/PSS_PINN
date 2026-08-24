"""
differentiable_real_dae.py
=============================
The real fix: makes delta_pm differentiable THROUGH THE REAL DAE'S OWN
SOLVER (Trapezoidal_dae_multitimestep.trapezoidal_method /
update_function_pure_physics), not our own reimplementation of the physics.

WHY THIS EXISTS
----------------
Our own rollout (delta_pm_solver.py's FaithfulResidual) was built to
FAITHFULLY REPRODUCE the real DAE's equations -- and it does, closely, for
the UNCORRECTED case (matched to 5 decimal places). But once a correction
is optimized aggressively (pushed toward a threshold), verified real-DAE
checks showed the two could disagree substantially (up to 100%+ in some
cases) -- an unexplained gap that multiple fixes (safety margins,
limiting optimization thoroughness) only partially addressed.

This module removes the gap at its root: instead of optimizing against
OUR APPROXIMATION and hoping it transfers to the real DAE, delta_pm is
optimized DIRECTLY against the real DAE's own residual equations. There is
no separate "our simulator" anymore for the post-clearing window -- there
is only the real physics, made differentiable via the SAME implicit-
differentiation technique used for our own rollout (this is possible
because the real DAE's own solver is ALREADY built in torch, using the
exact same Newton-Raphson + jacobian pattern we replicated ourselves).

MECHANISM
----------
Exactly the same implicit-differentiation pattern as delta_pm_solver.py's
ImplicitStep, but calling the REAL solver's own
update_function_pure_physics directly (a public method) instead of our
own residual reimplementation. delta_pm enters via
solver.machine_class.Pm0_const (confirmed the exact attribute path used
internally: Pm0 = self.Pm0_const[no_machine] inside the machine model).

CRITICAL: this must be verified against finite differences before being
trusted for anything -- the earlier version of this technique (applied to
our own simulator) had a real bug (missing dx1/dx0 gradient chaining)
that a finite-difference check caught. Same rigor applies here.
"""

import torch
import numpy as np
from torch.autograd.functional import jacobian


def build_real_solver(stator_matrices, Y_adm, Pm0_baseline, machine_buses,
                       sl_array, V_lf, Eq_prime, Ed_prime, H, dampings,
                       gov_R, gov_Tg, dt=0.05, internal_step=0.01):
    """Constructs ONE Trapezoidal_dae_multitimestep instance whose
    machine_class.Pm0_const we'll overwrite before each step (matching
    exactly how verify_against_real_dae.py's apply_delta_pm_sequence_multigen
    already does it for the non-differentiable verification path)."""
    from cross_validation_datagen_governor import Trapezoidal_dae_multitimestep, Machine_Models

    solver = Trapezoidal_dae_multitimestep(
        stator_matrices, Y_adm, Pm0_baseline, machine_buses,
        sl_array, V_lf,
        np.zeros(5 * len(machine_buses) + 2 * Y_adm.shape[0]), Eq_prime, Ed_prime, Pm0_baseline,
        dt, internal_step, "trapezoidal",
        gov_R=gov_R, gov_Tg=gov_Tg)

    machine_model = Machine_Models(
        60, torch.tensor(dampings, dtype=torch.float64),
        torch.tensor(H, dtype=torch.float64),
        torch.tensor(Eq_prime, dtype=torch.float64),
        torch.tensor(Ed_prime, dtype=torch.float64),
        Pm0_const=torch.tensor(Pm0_baseline, dtype=torch.float64),
        gov_R=gov_R, gov_Tg=gov_Tg)
    methods_map = machine_model.return_methods_mapping()
    solver.upload_dynamic_model(machine_model, "classical", methods_map)
    return solver


class ImplicitStepReal(torch.autograd.Function):
    """
    One Newton-solved trapezoidal step through the REAL DAE's own solver,
    differentiable w.r.t. delta_pm AND x0 via implicit differentiation --
    NOT by unrolling the Newton loop, and NOT using our own approximate
    residual equations.
    """
    @staticmethod
    def forward(ctx, delta_pm_vec, x0, solver, Pm0_baseline):
        Pm0_corrected = Pm0_baseline + delta_pm_vec
        solver.machine_class.Pm0_const = Pm0_corrected.detach()

        # forward solve using their EXACT trapezoidal_method -- guarantees
        # identical forward numerics to the true DAE
        x1 = solver.trapezoidal_method(x0.detach())

        # dF/dx1 at the converged solution (needed for implicit diff)
        def resid_x1(xx1):
            solver.machine_class.Pm0_const = Pm0_corrected.detach()
            return solver.update_function_pure_physics(x0.detach(), xx1)
        J = jacobian(resid_x1, x1.detach())

        # dF/d(delta_pm) -- Pm0_const enters the graph here
        dpm = delta_pm_vec.detach().clone()
        def resid_wrt_dpm(p):
            solver.machine_class.Pm0_const = Pm0_baseline + p
            r = solver.update_function_pure_physics(x0.detach(), x1.detach())
            solver.machine_class.Pm0_const = Pm0_corrected.detach()  # restore
            return r
        dF_ddpm = jacobian(resid_wrt_dpm, dpm)

        # dF/dx0
        def resid_wrt_x0(xx0):
            solver.machine_class.Pm0_const = Pm0_corrected.detach()
            return solver.update_function_pure_physics(xx0, x1.detach())
        dF_dx0 = jacobian(resid_wrt_x0, x0.detach())

        ctx.save_for_backward(J, dF_ddpm, dF_dx0)
        return x1

    @staticmethod
    def backward(ctx, grad_output):
        J, dF_ddpm, dF_dx0 = ctx.saved_tensors
        lam = torch.linalg.solve(J.T, -grad_output)
        grad_delta_pm = torch.matmul(lam, dF_ddpm)
        grad_x0 = torch.matmul(lam, dF_dx0)
        return grad_delta_pm, grad_x0, None, None


def rollout_real(solver, x0_full, delta_pm_seq, Pm0_baseline, sub_steps=5):
    """
    Chains ImplicitStepReal across a (n_steps, N_gen) delta_pm sequence,
    directly against the real DAE's own equations.

    IMPORTANT: trapezoidal_method's dt is self.step_size (0.01 by default,
    per build_real_solver's internal_step), NOT the outer correction
    timestep (0.05s) -- confirmed by reading the residual formula directly
    (line ~324: uses self.step_size, not self.t_final). So each 0.05s
    correction-window entry in delta_pm_seq needs sub_steps=5 chained
    calls (0.05/0.01) at the SAME delta_pm value, not one call -- matching
    exactly what iterations_trapezoidal does internally for a 0.05s window
    with a 0.01s internal step (the same granularity
    apply_delta_pm_sequence_multigen already uses for verification).
    """
    x = x0_full
    traj = [x]
    for t in range(len(delta_pm_seq)):
        for _ in range(sub_steps):
            x = ImplicitStepReal.apply(delta_pm_seq[t], x, solver, Pm0_baseline)
        traj.append(x)
    return torch.stack(traj)


if __name__ == "__main__":
    print(__doc__)
