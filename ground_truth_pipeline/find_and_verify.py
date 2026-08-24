"""
find_and_verify.py
=====================
The trustworthy version of the ΔPm pipeline: instead of trusting our own
optimizer's prediction, this ALWAYS checks the result against the real DAE,
and if it doesn't hold up, scales the correction up and checks again --
repeating until the REAL system confirms stability, or giving up honestly
after a few tries.

WHY THIS EXISTS
----------------
Verified directly: our optimizer's "minimum correction that works"
objective causes it to land right on the edge of whatever target it's
given (0.005, or a tighter margin target) -- and results that hug ANY
edge that precisely turned out to be unreliable, regardless of which
edge. Multiple theories were tested (timestep resolution, a fixed safety
margin, optimization thoroughness) -- each partially explained the
mismatches but none fully did. Rather than keep guessing at the internal
cause, this makes the REAL DAE the final arbiter: nothing is reported as
"stabilized" unless the real solver confirms it.

SCALING, NOT RE-OPTIMIZING, FOR RETRIES
------------------------------------------
Re-running the full optimizer for each retry attempt is expensive (100+
seconds per attempt). Scaling the ALREADY-FOUND correction and re-checking
against the real DAE is cheap (just a handful of DAE calls, no gradient
descent) -- so retries here are fast even though the initial optimization
is not.

Usage: see the __main__ block for a worked example.
"""

import time
import numpy as np


def find_and_verify(params, res_fn, x0, Pm0_setpoint,
                     real_state_at_clear, real_Y_adm_post,
                     correction_gens=(0, 1, 2),
                     scale_factors=(1.0, 0.85, 1.15, 0.7, 1.3, 0.5, 1.5, 0.3, 1.75, 2.0),
                     n_iters=100, patience=30, dt=0.05, window=2.0,
                     threshold=None, verbose=True):
    """
    1. Find a candidate delta_pm sequence with our optimizer (ONCE).
    2. Verify it against the real DAE.
    3. If it doesn't hold up, scale the SAME sequence and re-verify (cheap
       -- no re-optimization) -- repeat through scale_factors.

    scale_factors search BOTH directions (below and above 1.0), not just
    upward -- verified directly that "more correction" is NOT reliably
    "closer to solved": one real test showed a 1.15x scale-up making the
    real result WORSE (0.00520 -> 0.00642), confirming a non-monotonic
    relationship between correction size and outcome (consistent with
    findings much earlier in this project: "shallow minimum" behavior
    where the optimal correction is a specific point, not "as much as
    possible"). Ordered to try close-to-1.0 factors first (cheapest
    deviations from what the optimizer already found), widening outward.

    Returns a dict with: verified (bool), scale_used, real_max_dev,
    predicted_max_dev (from our pipeline, for reference/comparison),
    delta_pm_seq (the final, verified sequence), n_dae_checks.
    """
    import torch
    from delta_pm_solver import find_minimum_delta_pm_timevarying, STABILITY_THRESHOLD
    from verify_against_real_dae import apply_delta_pm_sequence_multigen

    if threshold is None:
        threshold = STABILITY_THRESHOLD

    t0 = time.time()
    result = find_minimum_delta_pm_timevarying(
        res_fn, x0, Pm0_setpoint, correction_gens=list(correction_gens),
        n_iters=n_iters, patience=patience, dt=dt, window=window, verbose=False)
    base_seq = result['best_delta_pm_seq'].numpy()
    predicted_max_dev = result['best_max_dev']

    if verbose:
        print(f"  optimizer found candidate: predicted={predicted_max_dev:.5f} "
              f"({time.time()-t0:.0f}s)")

    n_steps = int(round(window / dt))
    n_checks = 0
    best_scale_seen = None
    best_real_dev_seen = float('inf')

    for scale in scale_factors:
        scaled_seq = base_seq * scale
        _, omega_arr = apply_delta_pm_sequence_multigen(
            params, real_state_at_clear, real_Y_adm_post, scaled_seq, dt=dt)
        real_max_dev = float(np.abs(omega_arr[4:] - 1.0).max())
        n_checks += 1

        if verbose:
            print(f"  scale={scale:.2f}x  real_max_dev={real_max_dev:.5f}  "
                  f"{'STABLE' if real_max_dev <= threshold else 'not stable'}")

        if real_max_dev < best_real_dev_seen:
            best_real_dev_seen = real_max_dev
            best_scale_seen = scale

        if real_max_dev <= threshold:
            return {
                'verified': True, 'scale_used': scale, 'real_max_dev': real_max_dev,
                'predicted_max_dev': predicted_max_dev, 'delta_pm_seq': scaled_seq,
                'n_dae_checks': n_checks, 'elapsed_s': time.time() - t0,
            }

    # nothing worked -- honest failure, report the BEST (lowest) real result
    # seen across ALL tried scales, not just the last one tried
    return {
        'verified': False, 'scale_used': best_scale_seen, 'real_max_dev': best_real_dev_seen,
        'predicted_max_dev': predicted_max_dev, 'delta_pm_seq': base_seq * best_scale_seen,
        'n_dae_checks': n_checks, 'elapsed_s': time.time() - t0,
    }


if __name__ == "__main__":
    print(__doc__)
    print("This module is meant to be imported. See verify_batch_gen_v2.py")
    print("for a full batch example using this pipeline.")
