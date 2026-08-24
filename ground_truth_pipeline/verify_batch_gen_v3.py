"""
verify_batch_gen_v3.py
=========================
Runs the same 10 gen-fault scenarios as verify_batch_gen.py and
verify_batch_gen_v2.py, but using find_minimum_delta_pm_real.py --
optimizing DIRECTLY against the real DAE. No PINN, no approximate
rollout, no separate verification step: the reported max_dev IS the real
DAE's own number, trustworthy by construction.

BUDGET: n_iters=180, patience=50 -- raised from an earlier 60/20 run
after observing some scenarios (e.g. Gen1 alpha=0.6) converge slowly but
steadily (roughly linear improvement per iteration, not stuck) and likely
need ~100-150 iterations to cross the stability threshold, not 60.

COST WARNING: at ~15-30s/iteration, easy scenarios (some converge in <15
iterations) finish quickly, but the hardest ones could use most/all of
the 180-iteration budget -- potentially 60-90+ minutes each in the worst
case. Total batch time could run several hours. Designed to run
unattended (progress saved incrementally, resumable).

Usage:
    python verify_batch_gen_v3.py

Requires, in the same folder: ieee9_parameters.json,
cross_validation_datagen_governor.py, delta_pm_solver.py (for
STABILITY_THRESHOLD), verify_against_real_dae.py,
differentiable_real_dae.py, find_minimum_delta_pm_real.py.

NOTE: does NOT need best_model_gen.pt or meta.json -- there's no PINN
involved in this version at all. The handoff state at t_clear comes
directly from the real DAE (run_gen_fault_to_clearing), same as the
verification scripts already did for the "real" side of the comparison.

Output: verify_results_gen_v3_bigbudget.csv
"""

import csv
import os
import time
import traceback
import json

import torch

from delta_pm_solver import STABILITY_THRESHOLD
from cross_validation_datagen_governor import get_system_data, get_machine_params, run_dae_simulation, GOVERNOR_R, GOVERNOR_TG
from verify_against_real_dae import run_gen_fault_to_clearing
from differentiable_real_dae import build_real_solver
from find_minimum_delta_pm_real import find_minimum_delta_pm_real

torch.set_default_dtype(torch.float64)

with open('ieee9_parameters.json') as f:
    PARAMS = json.load(f)

SAMPLE_SCENARIOS = [
    (0, 0.4, 2.5), (0, 0.6, 3.0), (0, 0.8, 3.5),
    (1, 0.4, 2.5), (1, 0.6, 2.0), (1, 0.6, 3.0), (1, 0.8, 3.5),
    (2, 0.4, 2.5), (2, 0.6, 3.0), (2, 0.8, 3.5),
]

# fixed system properties -- computed once, reused for every scenario
sd = get_system_data(PARAMS)
H, dampings, Xd_p, Rs, stator_matrices = get_machine_params(PARAMS)
sl_array = sd['pl_ini'] + 1j * sd['ql_ini']
ref = run_dae_simulation('none', 0, 0.0, params=PARAMS, fault_time=0.0, sim_end_time=0.01,
                          gov_R=GOVERNOR_R, gov_Tg=GOVERNOR_TG)
Eq_prime, Ed_prime, V_lf, Pm0 = ref['Eq_prime'], ref['Ed_prime'], ref['V_lf'], ref['Pm0']
Pm0_t = torch.tensor(Pm0, dtype=torch.float64)

OUTPUT_CSV = 'verify_results_gen_v3_bigbudget.csv'
done_keys = set()
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV) as f:
        for row in csv.DictReader(f):
            done_keys.add((row['gen_idx'], row['alpha'], row['t_clear']))
    print(f"Resuming -- {len(done_keys)} scenarios already completed, will skip those.")

write_header = not os.path.exists(OUTPUT_CSV)
csv_file = open(OUTPUT_CSV, 'a', newline='')
writer = csv.writer(csv_file)
if write_header:
    writer.writerow(['gen_idx', 'alpha', 't_clear', 'baseline_max_dev',
                      'real_max_dev', 'stabilized', 'n_iters_used', 'elapsed_s', 'error'])
    csv_file.flush()

t_batch_start = time.time()
for i, (gen_idx, alpha, t_clear) in enumerate(SAMPLE_SCENARIOS):
    key = (str(float(gen_idx)), str(alpha), str(t_clear))
    if key in done_keys:
        print(f"[{i+1}/{len(SAMPLE_SCENARIOS)}] SKIP (already done)")
        continue

    print(f"\n[{i+1}/{len(SAMPLE_SCENARIOS)}] Gen{gen_idx}, alpha={alpha}, t_clear={t_clear}")
    t0 = time.time()
    try:
        solver = build_real_solver(stator_matrices, sd['Y_adm'], Pm0, sd['machine_buses'],
                                    sl_array, V_lf, Eq_prime, Ed_prime, H, dampings,
                                    GOVERNOR_R, GOVERNOR_TG, dt=0.05)

        state_at_clear, Y_adm_orig = run_gen_fault_to_clearing(
            PARAMS, gen_idx, alpha, fault_time=0.2, clear_time=t_clear)
        x0 = torch.tensor(state_at_clear, dtype=torch.float64)

        # baseline (delta_pm=0) -- real DAE, for reference
        omega_idx = [1, 3, 5]
        baseline_traj = state_at_clear  # at t_clear itself; get true baseline via a zero-correction rollout
        from differentiable_real_dae import rollout_real
        zero_seq = torch.zeros(40, 3, dtype=torch.float64)
        with torch.no_grad():
            base_traj = rollout_real(solver, x0, zero_seq, Pm0_t)
        baseline_max_dev = (base_traj[1 + 4:, omega_idx] - 1.0).abs().max().item()

        if baseline_max_dev <= STABILITY_THRESHOLD:
            elapsed = time.time() - t0
            print(f"  -> already stable ({baseline_max_dev:.5f}) -- skipping optimization. ({elapsed:.0f}s)")
            writer.writerow([gen_idx, alpha, t_clear, baseline_max_dev,
                              baseline_max_dev, True, 0, elapsed, ''])
            csv_file.flush()
            continue

        result = find_minimum_delta_pm_real(
            solver, x0, Pm0_t, n_steps=40, n_gen=3,
            n_iters=180, patience=50, verbose=False)

        elapsed = time.time() - t0
        stabilized = result['best_max_dev'] <= STABILITY_THRESHOLD
        writer.writerow([gen_idx, alpha, t_clear, baseline_max_dev,
                          result['best_max_dev'], stabilized, result['best_iter'] + 1,
                          elapsed, ''])
        csv_file.flush()

        print(f"  -> baseline={baseline_max_dev:.5f}  real_max_dev={result['best_max_dev']:.5f}  "
              f"stabilized={stabilized}  (n_iters={result['best_iter']+1})  ({elapsed:.0f}s)")

    except Exception as e:
        elapsed = time.time() - t0
        err_msg = f"{type(e).__name__}: {e}"
        print(f"  -> FAILED: {err_msg}")
        traceback.print_exc()
        writer.writerow([gen_idx, alpha, t_clear, '', '', '', '', elapsed, err_msg])
        csv_file.flush()

    total_elapsed = time.time() - t_batch_start
    print(f"  (batch elapsed: {total_elapsed/60:.1f} min)")

csv_file.close()
print(f"\nDone. Results saved to {OUTPUT_CSV}")
