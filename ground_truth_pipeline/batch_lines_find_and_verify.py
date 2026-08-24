"""
batch_lines_find_and_verify.py
=================================
Full 190-scenario transmission-line-fault grid (5 lines x 38 t_clear
values, excludes line 6-7), using the PINN-based find_and_verify
pipeline: PINN + approximate rollout finds the correction, real DAE
verifies (and scales if needed) before anything is reported as
"stabilized".

Usage:
    python batch_lines_find_and_verify.py

Requires, in the same folder: best_model_line.pt, meta.json,
ieee9_parameters.json, delta_pm_solver.py,
cross_validation_datagen_governor.py, verify_against_real_dae.py,
find_and_verify.py.

Output: batch_results_lines_verified.csv
"""

import csv
import os
import time
import traceback
import json

import torch
import numpy as np

from delta_pm_solver import build_scenario, STABILITY_THRESHOLD, rollout
from verify_against_real_dae import run_line_fault_to_clearing, apply_delta_pm_sequence_multigen
from find_and_verify import find_and_verify

with open('ieee9_parameters.json') as f:
    PARAMS = json.load(f)

CHECKPOINT_PATH = 'best_model_line.pt'
LINE_LOCS = [(3, 4), (3, 5), (4, 6), (5, 8), (7, 8)]   # excludes (6,7)
T_CLEARS = [round(0.25 + 0.1 * i, 2) for i in range(38)]

SCENARIOS = []
for (bus_r, bus_s) in LINE_LOCS:
    for t_clear in T_CLEARS:
        loc_code = float(bus_r * 10 + bus_s)
        SCENARIOS.append({'bus_r': bus_r, 'bus_s': bus_s, 'loc_code': loc_code, 't_clear': t_clear})

print(f"Total scenarios queued: {len(SCENARIOS)} ({len(LINE_LOCS)} lines x {len(T_CLEARS)} t_clear values)")

OUTPUT_CSV = 'batch_results_lines_verified.csv'
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
                      'predicted_max_dev', 'real_max_dev', 'scale_used',
                      'n_dae_checks', 'stabilized', 'elapsed_s', 'error'])
    csv_file.flush()

t_batch_start = time.time()
for i, sc in enumerate(SCENARIOS):
    key = (str(float(sc['bus_r'])), str(float(sc['bus_s'])), str(sc['t_clear']))
    if key in done_keys:
        continue

    print(f"\n[{i+1}/{len(SCENARIOS)}] Line({sc['bus_r']},{sc['bus_s']}), t_clear={sc['t_clear']}")
    t0 = time.time()
    try:
        torch.set_default_dtype(torch.float32)  # reset -- real DAE forces float64 as a side effect

        res_fn, x0, Pm0_setpoint = build_scenario(
            CHECKPOINT_PATH, 'ieee9_parameters.json', 'meta.json',
            1.0, sc['loc_code'], sc['t_clear'])  # alpha=1.0 placeholder, ignored for lines

        omega_idx = torch.tensor([res_fn.dif_idx[m][1].item() for m in range(3)])
        zero_dpm = torch.zeros(3, dtype=torch.float64)
        n_steps = int(round(2.0 / 0.05))
        baseline_traj = rollout(res_fn, x0, zero_dpm, dt=0.05, n_steps=n_steps, Pm0=Pm0_setpoint)
        baseline_max_dev = (baseline_traj[1 + 4:, omega_idx] - 1.0).abs().max().item()

        state_at_clear, Y_adm_orig = run_line_fault_to_clearing(
            PARAMS, sc['bus_r'], sc['bus_s'], fault_time=0.2, clear_time=sc['t_clear'])

        if baseline_max_dev <= STABILITY_THRESHOLD:
            zero_seq = np.zeros((n_steps, 3))
            _, omega_real = apply_delta_pm_sequence_multigen(
                PARAMS, state_at_clear, Y_adm_orig, zero_seq, dt=0.05)
            real_baseline = float(np.abs(omega_real[4:] - 1.0).max())
            elapsed = time.time() - t0
            verified_stable = real_baseline <= STABILITY_THRESHOLD
            print(f"  -> pipeline says already stable ({baseline_max_dev:.5f}); "
                  f"real DAE: {real_baseline:.5f} "
                  f"({'confirmed' if verified_stable else 'DISAGREES'}) ({elapsed:.0f}s)")
            writer.writerow([sc['bus_r'], sc['bus_s'], sc['t_clear'], baseline_max_dev,
                              baseline_max_dev, real_baseline, 1.0, 1, verified_stable, elapsed, ''])
            csv_file.flush()
            continue

        result = find_and_verify(
            PARAMS, res_fn, x0, Pm0_setpoint, state_at_clear, Y_adm_orig,
            n_iters=100, patience=30, verbose=False)

        elapsed = time.time() - t0
        writer.writerow([sc['bus_r'], sc['bus_s'], sc['t_clear'], baseline_max_dev,
                          result['predicted_max_dev'], result['real_max_dev'],
                          result['scale_used'], result['n_dae_checks'], result['verified'],
                          elapsed, ''])
        csv_file.flush()

        status = "VERIFIED" if result['verified'] else "NOT VERIFIED (best effort)"
        print(f"  -> {status}  real_max_dev={result['real_max_dev']:.5f}  "
              f"scale={result['scale_used']}  ({elapsed:.0f}s)")

    except Exception as e:
        elapsed = time.time() - t0
        err_msg = f"{type(e).__name__}: {e}"
        print(f"  -> FAILED: {err_msg}")
        traceback.print_exc()
        writer.writerow([sc['bus_r'], sc['bus_s'], sc['t_clear'], '', '', '', '', '', '', elapsed, err_msg])
        csv_file.flush()

    if (i + 1) % 5 == 0:
        total_elapsed = time.time() - t_batch_start
        n_done = i + 1 - len(done_keys)
        rate = total_elapsed / max(n_done, 1)
        remaining = (len(SCENARIOS) - i - 1) * rate
        print(f"  (batch elapsed: {total_elapsed/60:.1f} min, est. remaining: {remaining/60:.0f} min)")

csv_file.close()
print(f"\nDone. Results saved to {OUTPUT_CSV}")
