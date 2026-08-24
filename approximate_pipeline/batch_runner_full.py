"""
batch_runner_full.py
======================
Runs the ENTIRE gen-trip + load-fault grid (330 scenarios total: 150 gen
+ 180 load, read directly from meta.json so it matches your actual
training data exactly).

APPROACH (Option A, chosen after comparing on the 15 hardest scenarios
tested earlier: time-varying resolved 14/15, constant only resolved 3/15):
  1. Check baseline (delta_pm=0, cheap) for every scenario.
  2. Already stable at baseline -> record and skip (no optimization needed).
  3. Unstable at baseline -> go STRAIGHT to time-varying delta_pm (skip the
     constant-delta_pm step entirely -- it rarely succeeds once you're past
     "mild", so trying it first mostly just adds cost without changing the
     outcome for most scenarios).

COST WARNING -- READ THIS BEFORE RUNNING:
This is expected to take many hours (rough estimate: 10-20+ hours,
unattended, based on per-scenario costs observed on the 15-scenario time-
varying test: roughly 100-300s each for scenarios that need optimization).
With ~330 scenarios and likely over half needing real optimization, budget
for a long unattended run -- overnight, or across more than one session.

SAFETY FEATURES (same as the earlier batch runners):
- Saves each scenario's result to CSV as soon as it finishes.
- Resumable -- re-running skips scenarios already in the output CSV.
- Per-scenario error handling -- one failure doesn't kill the whole run.

Usage:
    python batch_runner_full.py

Requires, in the same folder: best_model_gen.pt, best_model_load.pt,
meta.json, ieee9_parameters.json, delta_pm_solver.py.

Output: batch_results_full.csv
"""

import csv
import json
import os
import time
import traceback

import numpy as np
import torch

from delta_pm_solver import (build_scenario, find_minimum_delta_pm_timevarying,
                              rollout, STABILITY_THRESHOLD)

with open('meta.json') as f:
    META = json.load(f)

# ── build the full scenario list directly from meta.json ───────────────────
SCENARIOS = []
for gen_idx in META['gen_buses']:
    for alpha in META['alpha_gen']:
        for t_clear in META['t_clears_gen']:
            SCENARIOS.append({
                'fault_type': 'gen', 'checkpoint': 'best_model_gen.pt',
                'alpha': alpha, 'fault_loc': float(gen_idx), 't_clear': t_clear,
                'loc_desc': f'Gen{gen_idx}',
            })
for bus in META['load_buses']:
    for alpha in META['alpha_load']:
        for t_clear in META['t_clears_load']:
            SCENARIOS.append({
                'fault_type': 'load', 'checkpoint': 'best_model_load.pt',
                'alpha': alpha, 'fault_loc': float(bus), 't_clear': t_clear,
                'loc_desc': f'Bus{bus}',
            })

print(f"Total scenarios queued: {len(SCENARIOS)} "
      f"({len(META['gen_buses'])*len(META['alpha_gen'])*len(META['t_clears_gen'])} gen-trip + "
      f"{len(META['load_buses'])*len(META['alpha_load'])*len(META['t_clears_load'])} load-fault)")

# ── resume support ──────────────────────────────────────────────────────
OUTPUT_CSV = 'batch_results_full.csv'
done_keys = set()
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV) as f:
        for row in csv.DictReader(f):
            done_keys.add((row['fault_type'], row['loc_desc'],
                            row['alpha'], row['t_clear']))
    print(f"Resuming -- {len(done_keys)} scenarios already completed, will skip those.")

write_header = not os.path.exists(OUTPUT_CSV)
csv_file = open(OUTPUT_CSV, 'a', newline='')
writer = csv.writer(csv_file)
if write_header:
    writer.writerow(['fault_type', 'loc_desc', 'alpha', 'fault_loc', 't_clear',
                      'baseline_max_dev', 'best_max_dev', 'best_loss', 'best_iter',
                      'stabilized', 'method', 'elapsed_s', 'error'])
    csv_file.flush()

# ── run ──────────────────────────────────────────────────────────────────
t_batch_start = time.time()
for i, sc in enumerate(SCENARIOS):
    key = (sc['fault_type'], sc['loc_desc'], str(sc['alpha']), str(sc['t_clear']))
    if key in done_keys:
        continue  # quiet skip -- printing all ~330 would be noisy; progress is in the CSV

    print(f"\n[{i+1}/{len(SCENARIOS)}] {sc['fault_type']} fault, {sc['loc_desc']} "
          f"(alpha={sc['alpha']:.3f}, t_clear={sc['t_clear']})")
    t0 = time.time()
    try:
        res_fn, x0, Pm0_setpoint = build_scenario(
            sc['checkpoint'], 'ieee9_parameters.json', 'meta.json',
            sc['alpha'], sc['fault_loc'], sc['t_clear'])

        # --- check baseline first; skip optimization entirely if already stable ---
        omega_idx = torch.tensor([res_fn.dif_idx[m][1].item() for m in range(res_fn.no_gens)])
        zero_dpm = torch.zeros(res_fn.no_gens, dtype=torch.float64)
        n_steps = int(round(2.0 / 0.05))
        baseline_traj = rollout(res_fn, x0, zero_dpm, dt=0.05, n_steps=n_steps, Pm0=Pm0_setpoint)
        baseline_max_dev = (baseline_traj[1 + 4:, omega_idx] - 1.0).abs().max().item()

        if baseline_max_dev <= STABILITY_THRESHOLD:
            elapsed = time.time() - t0
            print(f"  -> already stable ({baseline_max_dev:.5f}) -- skipping. ({elapsed:.0f}s)")
            writer.writerow([sc['fault_type'], sc['loc_desc'], sc['alpha'], sc['fault_loc'],
                              sc['t_clear'], baseline_max_dev, baseline_max_dev, 0.0, '',
                              True, 'none', elapsed, ''])
            csv_file.flush()
        else:
            result = find_minimum_delta_pm_timevarying(
                res_fn, x0, Pm0_setpoint, correction_gens=[0, 1, 2],
                n_iters=100, patience=30, verbose=False)
            stabilized = result['best_max_dev'] <= STABILITY_THRESHOLD
            elapsed = time.time() - t0

            writer.writerow([sc['fault_type'], sc['loc_desc'], sc['alpha'], sc['fault_loc'],
                              sc['t_clear'], baseline_max_dev, result['best_max_dev'],
                              result['best_loss'], result['best_iter'], stabilized,
                              'timevarying', elapsed, ''])
            csv_file.flush()
            print(f"  -> baseline={baseline_max_dev:.5f}  best={result['best_max_dev']:.5f}  "
                  f"stabilized={stabilized}  ({elapsed:.0f}s)")

    except Exception as e:
        elapsed = time.time() - t0
        err_msg = f"{type(e).__name__}: {e}"
        print(f"  -> FAILED: {err_msg}")
        traceback.print_exc()
        writer.writerow([sc['fault_type'], sc['loc_desc'], sc['alpha'], sc['fault_loc'],
                          sc['t_clear'], '', '', '', '', '', '', elapsed, err_msg])
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
