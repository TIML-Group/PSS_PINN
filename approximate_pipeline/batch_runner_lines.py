"""
batch_runner_lines.py
========================
Runs the transmission-line-fault grid through the ΔPm optimizer: 5 valid
line locations (34, 35, 46, 58, 78 -- excludes 67, which was excluded from
training as always/near-always unstable) x 38 t_clear values (0.25 -> 3.95,
step 0.1, matching generate_data_line_only.py's grid) = 190 scenarios.

APPROACH (same as batch_runner_full.py, Option A): check baseline first,
skip if already stable, go straight to time-varying delta_pm for anything
unstable (skips the constant-delta_pm step, which rarely succeeds once
past "mild" -- confirmed on gen/load faults: 14/15 hardest cases resolved
by time-varying vs. 3/15 by constant).

Usage:
    python batch_runner_lines.py

Requires, in the same folder: best_model_line.pt, meta.json,
ieee9_parameters.json, delta_pm_solver.py.

Output: batch_results_lines.csv
"""

import csv
import os
import time
import traceback

import numpy as np
import torch

from delta_pm_solver import build_scenario, find_minimum_delta_pm_timevarying, rollout, STABILITY_THRESHOLD

CHECKPOINT_PATH = 'best_model_line.pt'
LINE_LOCS = [34.0, 35.0, 46.0, 58.0, 78.0]   # excludes 67.0 (excluded from training)
T_CLEARS = [round(0.25 + 0.1 * i, 2) for i in range(38)]  # matches training grid

SCENARIOS = []
for loc in LINE_LOCS:
    for t_clear in T_CLEARS:
        SCENARIOS.append({'fault_loc': loc, 't_clear': t_clear})

print(f"Total scenarios queued: {len(SCENARIOS)} "
      f"({len(LINE_LOCS)} lines x {len(T_CLEARS)} t_clear values)")

# ── resume support ──────────────────────────────────────────────────────
OUTPUT_CSV = 'batch_results_lines.csv'
done_keys = set()
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV) as f:
        for row in csv.DictReader(f):
            done_keys.add((row['fault_loc'], row['t_clear']))
    print(f"Resuming -- {len(done_keys)} scenarios already completed, will skip those.")

write_header = not os.path.exists(OUTPUT_CSV)
csv_file = open(OUTPUT_CSV, 'a', newline='')
writer = csv.writer(csv_file)
if write_header:
    writer.writerow(['fault_loc', 't_clear', 'baseline_max_dev', 'best_max_dev',
                      'best_loss', 'best_iter', 'stabilized', 'method', 'elapsed_s', 'error'])
    csv_file.flush()

# ── run ──────────────────────────────────────────────────────────────────
t_batch_start = time.time()
for i, sc in enumerate(SCENARIOS):
    key = (str(sc['fault_loc']), str(sc['t_clear']))
    if key in done_keys:
        continue  # quiet skip -- progress is in the CSV

    print(f"\n[{i+1}/{len(SCENARIOS)}] line fault, loc={sc['fault_loc']}, "
          f"t_clear={sc['t_clear']}")
    t0 = time.time()
    try:
        res_fn, x0, Pm0_setpoint = build_scenario(
            CHECKPOINT_PATH, 'ieee9_parameters.json', 'meta.json',
            1.0, sc['fault_loc'], sc['t_clear'])   # alpha=1.0 placeholder, ignored

        omega_idx = torch.tensor([res_fn.dif_idx[m][1].item() for m in range(res_fn.no_gens)])
        zero_dpm = torch.zeros(res_fn.no_gens, dtype=torch.float64)
        n_steps = int(round(2.0 / 0.05))
        baseline_traj = rollout(res_fn, x0, zero_dpm, dt=0.05, n_steps=n_steps, Pm0=Pm0_setpoint)
        baseline_max_dev = (baseline_traj[1 + 4:, omega_idx] - 1.0).abs().max().item()

        if baseline_max_dev <= STABILITY_THRESHOLD:
            elapsed = time.time() - t0
            print(f"  -> already stable ({baseline_max_dev:.5f}) -- skipping. ({elapsed:.0f}s)")
            writer.writerow([sc['fault_loc'], sc['t_clear'], baseline_max_dev,
                              baseline_max_dev, 0.0, '', True, 'none', elapsed, ''])
            csv_file.flush()
        else:
            result = find_minimum_delta_pm_timevarying(
                res_fn, x0, Pm0_setpoint, correction_gens=[0, 1, 2],
                n_iters=100, patience=30, verbose=False)
            stabilized = result['best_max_dev'] <= STABILITY_THRESHOLD
            elapsed = time.time() - t0

            writer.writerow([sc['fault_loc'], sc['t_clear'], baseline_max_dev,
                              result['best_max_dev'], result['best_loss'],
                              result['best_iter'], stabilized, 'timevarying', elapsed, ''])
            csv_file.flush()
            print(f"  -> baseline={baseline_max_dev:.5f}  best={result['best_max_dev']:.5f}  "
                  f"stabilized={stabilized}  ({elapsed:.0f}s)")

    except Exception as e:
        elapsed = time.time() - t0
        err_msg = f"{type(e).__name__}: {e}"
        print(f"  -> FAILED: {err_msg}")
        traceback.print_exc()
        writer.writerow([sc['fault_loc'], sc['t_clear'], '', '', '', '', '', '', elapsed, err_msg])
        csv_file.flush()

    if (i + 1) % 5 == 0:
        total_elapsed = time.time() - t_batch_start
        n_done = i + 1 - len(done_keys)
        rate = total_elapsed / max(n_done, 1)
        remaining = (len(SCENARIOS) - i - 1) * rate
        print(f"  (batch elapsed: {total_elapsed/60:.1f} min, "
              f"est. remaining: {remaining/60:.0f} min)")

csv_file.close()
print(f"\nDone. Results saved to {OUTPUT_CSV}")
