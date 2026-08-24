"""
find_minimum_delta_pm_real.py
================================
The true fix: optimizes delta_pm directly against the REAL DAE's own
equations (via differentiable_real_dae.py's validated implicit-
differentiation wrapper), not our approximation of it. No predicted-vs-
real gap to work around anymore -- there's only one physics.

Mirrors delta_pm_solver.py's find_minimum_delta_pm_timevarying in
structure (same hard-barrier loss, same best-iterate tracking, same
early stopping), but every rollout call is a real DAE evaluation.

COST WARNING: expected to be SLOWER per iteration than the approximate
version -- each correction timestep now does 5 real Newton solves (0.01s
internal steps) through the full 33-state system, plus 3 extra jacobian
calls per step for the implicit-differentiation backward pass, instead of
1 step through our simplified rollout.
"""

import time
import torch

from differentiable_real_dae import rollout_real
from delta_pm_solver import STABILITY_THRESHOLD


def stability_loss_real(traj, delta_pm, threshold=STABILITY_THRESHOLD, lam=1e-6,
                         grace_steps=4, big_offset=1.0):
    """Same hard-barrier design as delta_pm_solver.py's stability_loss:
    any stabilizing solution beats any non-stabilizing one regardless of
    size (fixes the loss-weighting bug found earlier in this project)."""
    omega_idx = torch.tensor([1, 3, 5])
    omega_traj = traj[1 + grace_steps:, omega_idx]
    max_dev = (omega_traj - 1.0).abs().max()
    violation = torch.relu(max_dev - threshold)
    is_violating = (violation > 0).float()
    penalty = lam * (delta_pm ** 2).sum()
    return is_violating * (big_offset + violation) + (1 - is_violating) * penalty


def find_minimum_delta_pm_real(solver, x0, Pm0_baseline, n_steps=40, n_gen=3,
                                lr=0.05, lam=1e-6, n_iters=60, patience=20,
                                grace_steps=4, init_delta_pm=None, verbose=True):
    """
    Gradient descent on delta_pm (n_steps, n_gen), evaluated via the REAL
    DAE at every iteration. Same best-iterate tracking and early stopping
    as the approximate version.

    init_delta_pm: optional (n_steps, n_gen) starting point, for resuming
    a search across multiple calls (real-DAE iterations are expensive
    enough that reaching a large iteration budget may need chunking).
    """
    if init_delta_pm is not None:
        delta_pm = torch.nn.Parameter(init_delta_pm.clone())
    else:
        delta_pm = torch.nn.Parameter(torch.zeros(n_steps, n_gen, dtype=torch.float64))
    optimizer = torch.optim.Adam([delta_pm], lr=lr)

    best_loss = float('inf')
    best_delta_pm = None
    best_max_dev = None
    best_iter = None
    patience_counter = 0
    history = []

    omega_idx = torch.tensor([1, 3, 5])
    t_start = time.time()
    for it in range(n_iters):
        optimizer.zero_grad()
        traj = rollout_real(solver, x0, delta_pm, Pm0_baseline)
        loss = stability_loss_real(traj, delta_pm, lam=lam, grace_steps=grace_steps)

        dpm_now = delta_pm.detach().clone()  # snapshot BEFORE the step
        loss.backward()
        optimizer.step()

        max_dev = (traj[1 + grace_steps:, omega_idx] - 1.0).abs().max().item()
        elapsed = time.time() - t_start
        history.append((it, loss.item(), max_dev, elapsed))
        if verbose:
            print(f"  iter {it:3d}  loss={loss.item():.6f}  max_dev={max_dev:.5f}  "
                  f"delta_pm range=[{dpm_now.min().item():.3f}, {dpm_now.max().item():.3f}]  "
                  f"({elapsed:.0f}s)")

        if loss.item() < best_loss - 1e-9:
            best_loss, best_delta_pm, best_max_dev, best_iter = (
                loss.item(), dpm_now, max_dev, it)
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            if verbose:
                print(f"  converged, stopping at iter {it}.")
            break

    return {
        'best_delta_pm': best_delta_pm, 'best_loss': best_loss,
        'best_max_dev': best_max_dev, 'best_iter': best_iter, 'history': history,
    }
