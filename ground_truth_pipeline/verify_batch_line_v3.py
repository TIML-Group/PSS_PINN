"""
verify_batch_line_v3.py
==========================
Line-fault equivalent of verify_batch_gen_v3.py: optimizes delta_pm
DIRECTLY against the real DAE. Reuses run_line_fault_to_clearing (built
and validated earlier in this project -- confirmed baseline match with
the real DAE to <1% back when we first tested it).

Line locations: (3,4), (3,5), (4,6), (5,8), (7,8) -- excludes (6,7),
which was excluded from PINN training as always/near-always unstable
(not directly relevant here since this version has no PINN, but kept
consistent with the rest of the project's scope decisions).

BUDGET: n_iters=180, patience=50 -- same generous budget as gen/load,
given some gen-fault scenarios needed 70-139 iterations.

Usage:
    python verify_batch_line_v3.py

Requires, in the same folder: ieee9_parameters.json,
cross_validation_datagen_governor.py, delta_pm_solver.py,
verify_against_real_dae.py, differentiable_real_dae.py,
find_minimum_delta_pm_real.py.

Output: verify_results_line_v3.csv
"""

import csv
import os
import time
import traceback
import json

import torch

from delta_pm_solver import STABILITY_THRESHOLD
from cross_validation_datagen_governor import get_system_data, get_machine_params, run_dae_simulation, GOVERNOR_R, GOVERNOR_TG
from verify_against_real_dae import run_line_fault_to_clearing
from differentiable_real_dae import build_real_solver, rollout_real
from find_minimum_delta_pm_real import find_minimum_delta_pm_real

torch.set_default_dtype(torch.float64)

with open('ieee9_parameters.json') as f:
    PARAMS = json.load(f)

# sample spanning the lines known to have real stable/unstable diversity
# (4,6) and (5,8) showed mixed behavior in earlier data generation;
# (3,4),(3,5),(7,8) were almost always stable -- include one of each
# plus a spread of clearing times for the "interesting" lines
SAMPLE_SCENARIOS = [
    # (bus_r, bus_s, t_clear)
    (4, 6, 0.55), (4, 6, 0.75), (4, 6, 2.45),
    (5, 8, 0.45), (5, 8, 1.45), (5, 8, 3.25),
    (3, 4, 2.45), (3, 5, 1.45), (7, 8, 2.15),
]

sd = get_system_data(PARAMS)
H, dampings, Xd_p, Rs, stator_matrices = get_machine_params(PARAMS)
sl_array = sd['pl_ini'] + 1j * sd['ql_ini']
ref = run_dae_simulation('none', 0, 0.0, params=PARAMS, fault_time=0.0, sim_end_time=0.01,
                          gov_R=GOVERNOR_R, gov_Tg=GOVERNOR_TG)
Eq_prime, Ed_prime, V_lf, Pm0 = ref['Eq_prime'], ref['Ed_prime'], ref['V_lf'], ref['Pm0']
Pm0_t = torch.tensor(Pm0, dtype=torch.float64)

OUTPUT_CSV = 'verify_results_line_v3.csv'
done_keys = set()
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV) as f:
        for row in csv.DictReader(f):
            done_keys.add((row['bus_r'], row['bus_s'], row['t_clear']))
    print(f"Resuming -- {len(done_keys)} scenarios already completed, will skip those.")

write_header = not os.path.exists(OUTPUT_CSV)
csv_file = open(OUTPUT_CSV, 'a', newline='')
writer = csv.writer(csv_file)
if write_header:
    writer.writerow(['bus_r', 'bus_s', 't_clear', 'baseline_max_dev',
                      'real_max_dev', 'stabilized', 'n_iters_used', 'elapsed_s', 'error'])
    csv_file.flush()

t_batch_start = time.time()
for i, (bus_r, bus_s, t_clear) in enumerate(SAMPLE_SCENARIOS):
    key = (str(float(bus_r)), str(float(bus_s)), str(t_clear))
    if key in done_keys:
        print(f"[{i+1}/{len(SAMPLE_SCENARIOS)}] SKIP (already done)")
        continue

    print(f"\n[{i+1}/{len(SAMPLE_SCENARIOS)}] Line({bus_r},{bus_s}), t_clear={t_clear}")
    t0 = time.time()
    try:
        solver = build_real_solver(stator_matrices, sd['Y_adm'], Pm0, sd['machine_buses'],
                                    sl_array, V_lf, Eq_prime, Ed_prime, H, dampings,
                                    GOVERNOR_R, GOVERNOR_TG, dt=0.05)

        state_at_clear, Y_adm_orig = run_line_fault_to_clearing(
            PARAMS, bus_r, bus_s, fault_time=0.2, clear_time=t_clear)
        x0 = torch.tensor(state_at_clear, dtype=torch.float64)

        omega_idx = [1, 3, 5]
        zero_seq = torch.zeros(40, 3, dtype=torch.float64)
        with torch.no_grad():
            base_traj = rollout_real(solver, x0, zero_seq, Pm0_t)
        baseline_max_dev = (base_traj[1 + 4:, omega_idx] - 1.0).abs().max().item()

        if baseline_max_dev <= STABILITY_THRESHOLD:
            elapsed = time.time() - t0
            print(f"  -> already stable ({baseline_max_dev:.5f}) -- skipping optimization. ({elapsed:.0f}s)")
            writer.writerow([bus_r, bus_s, t_clear, baseline_max_dev,
                              baseline_max_dev, True, 0, elapsed, ''])
            csv_file.flush()
            continue

        result = find_minimum_delta_pm_real(
            solver, x0, Pm0_t, n_steps=40, n_gen=3,
            n_iters=180, patience=50, verbose=False)

        elapsed = time.time() - t0
        stabilized = result['best_max_dev'] <= STABILITY_THRESHOLD
        writer.writerow([bus_r, bus_s, t_clear, baseline_max_dev,
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
        writer.writerow([bus_r, bus_s, t_clear, '', '', '', '', elapsed, err_msg])
        csv_file.flush()

    total_elapsed = time.time() - t_batch_start
    print(f"  (batch elapsed: {total_elapsed/60:.1f} min)")

csv_file.close()
print(f"\nDone. Results saved to {OUTPUT_CSV}")
