"""
batch_full_find_and_verify.py
================================
Full 330-scenario gen-trip + load-fault grid, using the PINN-based
find_and_verify pipeline: the PINN + approximate rollout FINDS the
correction (fast, deployable), then the correction is checked against
the REAL DAE and scaled (up or down) until confirmed, or honestly
reported as unresolved. Every "stabilized=True" in this batch has been
directly confirmed on the real DAE -- not just predicted.

Grid generated directly from meta.json (10 severities x 5 t_clears x 3
gens = 150, + 10 severities x 6 t_clears x 3 buses = 180 -> 330 total),
matching the original (unverified) full batch exactly.

COST WARNING: each scenario now costs a full optimizer run (~150-800s,
same as before) PLUS one or more real-DAE verification/scaling checks.
Expect this to take considerably longer than the original ~11.5-hour
batch. Designed to run unattended -- incremental saving, resumable.

Usage:
    python batch_full_find_and_verify.py

Requires, in the same folder: best_model_gen.pt, best_model_load.pt,
meta.json, ieee9_parameters.json, delta_pm_solver.py,
cross_validation_datagen_governor.py, verify_against_real_dae.py,
find_and_verify.py.

Output: batch_results_full_verified.csv
"""

import csv
import json
import os
import time
import traceback

import torch

from delta_pm_solver import build_scenario, STABILITY_THRESHOLD, rollout
from verify_against_real_dae import run_gen_fault_to_clearing, run_load_fault_to_clearing
from find_and_verify import find_and_verify
import torch as _torch

with open('ieee9_parameters.json') as f:
    PARAMS = json.load(f)
with open('meta.json') as f:
    META = json.load(f)

# ── build the full scenario list directly from meta.json ───────────────────
SCENARIOS = []
for gen_idx in META['gen_buses']:
    for alpha in META['alpha_gen']:
        for t_clear in META['t_clears_gen']:
            SCENARIOS.append({'fault_type': 'gen', 'checkpoint': 'best_model_gen.pt',
                               'loc': gen_idx, 'alpha': alpha, 't_clear': t_clear,
                               'loc_desc': f'Gen{gen_idx}'})
for bus in META['load_buses']:
    for alpha in META['alpha_load']:
        for t_clear in META['t_clears_load']:
            SCENARIOS.append({'fault_type': 'load', 'checkpoint': 'best_model_load.pt',
                               'loc': bus, 'alpha': alpha, 't_clear': t_clear,
                               'loc_desc': f'Bus{bus}'})

print(f"Total scenarios queued: {len(SCENARIOS)} "
      f"({len(META['gen_buses'])*len(META['alpha_gen'])*len(META['t_clears_gen'])} gen-trip + "
      f"{len(META['load_buses'])*len(META['alpha_load'])*len(META['t_clears_load'])} load-fault)")

# ── resume support ──────────────────────────────────────────────────────
OUTPUT_CSV = 'batch_results_full_verified.csv'
done_keys = set()
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV) as f:
        for row in csv.DictReader(f):
            done_keys.add((row['fault_type'], row['loc_desc'], row['alpha'], row['t_clear']))
    print(f"Resuming -- {len(done_keys)} scenarios already completed, will skip those.")

write_header = not os.path.exists(OUTPUT_CSV)
csv_file = open(OUTPUT_CSV, 'a', newline='')
writer = csv.writer(csv_file)
if write_header:
    writer.writerow(['fault_type', 'loc_desc', 'alpha', 't_clear',
                      'baseline_max_dev', 'predicted_max_dev', 'real_max_dev',
                      'scale_used', 'n_dae_checks', 'stabilized', 'elapsed_s', 'error'])
    csv_file.flush()

# ── run ──────────────────────────────────────────────────────────────────
t_batch_start = time.time()
for i, sc in enumerate(SCENARIOS):
    key = (sc['fault_type'], sc['loc_desc'], str(sc['alpha']), str(sc['t_clear']))
    if key in done_keys:
        continue  # quiet skip -- progress is in the CSV

    print(f"\n[{i+1}/{len(SCENARIOS)}] {sc['fault_type']} fault, {sc['loc_desc']} "
          f"(alpha={sc['alpha']:.3f}, t_clear={sc['t_clear']})")
    t0 = time.time()
    try:
        # dtype: the real DAE's solver forces global default dtype to
        # float64 as a side effect during verification -- reset before
        # each scenario's PINN loading (needs float32). Bug found and
        # fixed earlier in this project.
        torch.set_default_dtype(torch.float32)

        res_fn, x0, Pm0_setpoint = build_scenario(
            sc['checkpoint'], 'ieee9_parameters.json', 'meta.json',
            sc['alpha'], float(sc['loc']), sc['t_clear'])

        # baseline check first -- skip the expensive optimizer entirely if
        # no correction is needed (find_and_verify has no such shortcut
        # built in, and would otherwise waste a full optimizer run on
        # already-stable scenarios -- roughly 40% of this grid, based on
        # the earlier unverified batch)
        omega_idx = _torch.tensor([res_fn.dif_idx[m][1].item() for m in range(3)])
        zero_dpm = _torch.zeros(3, dtype=_torch.float64)
        n_steps = int(round(2.0 / 0.05))
        baseline_traj = rollout(res_fn, x0, zero_dpm, dt=0.05, n_steps=n_steps, Pm0=Pm0_setpoint)
        baseline_max_dev = (baseline_traj[1 + 4:, omega_idx] - 1.0).abs().max().item()

        if sc['fault_type'] == 'gen':
            state_at_clear, Y_adm_orig = run_gen_fault_to_clearing(
                PARAMS, sc['loc'], sc['alpha'], fault_time=0.2, clear_time=sc['t_clear'])
        else:
            state_at_clear, Y_adm_orig = run_load_fault_to_clearing(
                PARAMS, sc['loc'], sc['alpha'], fault_time=0.2, clear_time=sc['t_clear'])

        if baseline_max_dev <= STABILITY_THRESHOLD:
            # confirm the "already stable" baseline against the REAL DAE
            # too -- our own pipeline's baseline predictions have shown
            # real bias before (Section 7.3 of the project report), so
            # this shortcut still gets a genuine ground-truth check
            zero_seq_np = [[0.0, 0.0, 0.0]] * n_steps
            from verify_against_real_dae import apply_delta_pm_sequence_multigen
            import numpy as _np
            _, omega_real = apply_delta_pm_sequence_multigen(
                PARAMS, state_at_clear, Y_adm_orig, _np.array(zero_seq_np), dt=0.05)
            real_baseline = float(_np.abs(omega_real[4:] - 1.0).max())
            elapsed = time.time() - t0
            verified_stable = real_baseline <= STABILITY_THRESHOLD
            print(f"  -> pipeline says already stable ({baseline_max_dev:.5f}); "
                  f"real DAE: {real_baseline:.5f} "
                  f"({'confirmed' if verified_stable else 'DISAGREES'}) ({elapsed:.0f}s)")
            writer.writerow([sc['fault_type'], sc['loc_desc'], sc['alpha'], sc['t_clear'],
                              baseline_max_dev, baseline_max_dev, real_baseline,
                              1.0, 1, verified_stable, elapsed, ''])
            csv_file.flush()
            continue

        result = find_and_verify(
            PARAMS, res_fn, x0, Pm0_setpoint, state_at_clear, Y_adm_orig,
            n_iters=100, patience=30, verbose=False)

        elapsed = time.time() - t0
        writer.writerow([sc['fault_type'], sc['loc_desc'], sc['alpha'], sc['t_clear'],
                          baseline_max_dev, result['predicted_max_dev'], result['real_max_dev'],
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
        writer.writerow([sc['fault_type'], sc['loc_desc'], sc['alpha'], sc['t_clear'],
                          '', '', '', '', '', '', elapsed, err_msg])
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
