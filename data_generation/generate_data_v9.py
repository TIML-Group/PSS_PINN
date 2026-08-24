"""
generate_data.py  (Version 9)
==============================
Generates load faults and gen trips ONLY.

Line/transformer faults are handled by a SEPARATE script now:
generate_data_line_only.py -- which also fixes a real bug that used to
live here (t_clear=0.1 was before fault_time=0.2, causing "clear before
the fault even happened", which failed identically for all 6 lines every
time this file was run) and expands coverage to include generator
transformers. Removing line-fault generation from this file entirely
avoids maintaining two copies of the same logic and removes the bug at
its source (there's no line-fault code path here to be buggy).

Use merge_line_data.py to combine this file's output with
generate_data_line_only.py's output into one dataset.

Dataset design
--------------
Load faults:
    Buses  : 4, 5, 7 (0-indexed load buses)
    Alpha  : 0.1 → 1.5 pu (10 values)
    t_clear: 0.3, 0.5, 0.8, 1.5, 2.5, 3.5 s  ← EXTENDED (was 0.3, 0.5, 0.8)
             Short values → stable; longer values → unstable transients

Generator trips:
    Gens   : 0, 1, 2
    Alpha  : 0.1 → 1.0 (10 values)
    t_clear: 2.0, 2.5, 3.0, 3.5, 4.0 s (unchanged)

Changes from v6:
    - Load fault t_clear extended from 3 values to 6 values
    - Adds dynamic/unstable load fault scenarios the PINN never saw before
    - Load fault count: 90 → 180 scenarios

X_train columns: [t, alpha, fault_loc, fault_type_id, t_clear]
Y_train columns: [delta_1_rel, delta_2_rel, delta_3_rel,
                  omega_0, omega_1, omega_2, Pm_0, Pm_1, Pm_2,
                  Id_0, Id_1, Id_2, Iq_0, Iq_1, Iq_2]

Stability: max|omega-1| < 0.005 in window (t_clear, t_clear+2s)

Usage
-----
    python generate_data_v9.py --params ieee9_parameters.json
    python generate_data_v9.py --params ieee9_parameters.json --quick
"""

import argparse
import json
import os
import time
import numpy as np
import sys
import signal

sys.path.insert(0, '.')
from cross_validation_datagen_governor import (
    run_dae_with_clearing, get_system_data, GOVERNOR_R, GOVERNOR_TG)

# ── Timeout ──────────────────────────────────────────────────
class TimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutError("DAE solver timed out")

def run_with_timeout(func, timeout_sec, **kwargs):
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_sec)
    try:
        result = func(**kwargs)
        signal.alarm(0)
        return result
    except TimeoutError:
        signal.alarm(0)
        raise TimeoutError(f"Timed out after {timeout_sec}s")

# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--params',     type=str,   default='ieee9_parameters.json')
parser.add_argument('--outdir',     type=str,   default='training_data_v5')
parser.add_argument('--t_fault',    type=float, default=0.2)
parser.add_argument('--t_total',    type=float, default=8.0)
parser.add_argument('--downsample', type=int,   default=5)
parser.add_argument('--omega_tol',  type=float, default=0.005)
parser.add_argument('--timeout',    type=int,   default=120)
parser.add_argument('--quick',      action='store_true',
                    help='Quick: fewer scenarios for testing')
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Load parameters
# ──────────────────────────────────────────────────────────────
with open(args.params) as f:
    params = json.load(f)

buses_raw  = params['buses']
N_bus      = params['system']['buses']
gen_buses  = params['system']['gen_buses_0indexed']
load_buses = [i for i in range(N_bus) if buses_raw[f'bus_{i+1}']['pl_pu'] > 0]
Pm0        = np.array([params['generators'][f'gen_{i+1}']['Pm_pu']
                       for i in range(len(gen_buses))])
gov_kwargs = dict(gov_R=GOVERNOR_R, gov_Tg=GOVERNOR_TG)

print(f"Loaded: {args.params}")
print(f"Load buses : {load_buses}")
print(f"Gen buses  : {gen_buses}  Pm0: {Pm0}")

# ──────────────────────────────────────────────────────────────
# Sweep parameters
# ──────────────────────────────────────────────────────────────
if args.quick:
    alpha_load   = [0.3, 0.8, 1.5]
    t_clears_load= [0.3, 0.5]
    alpha_gen    = [0.3, 0.7, 1.0]
    t_clears_gen = [2.0, 3.0, 4.0]
else:
    alpha_load   = np.linspace(0.1, 1.5, 10).tolist()
    t_clears_load= [0.3, 0.5, 0.8, 1.5, 2.5, 3.5]   # extended: adds dynamic/unstable load faults
    alpha_gen    = np.linspace(0.1, 1.0, 10).tolist()
    t_clears_gen = [2.0, 2.5, 3.0, 3.5, 4.0]

# Defensive guard: no clear_time should ever be <= fault_time (this is what
# broke the old line-fault t_clear=0.1 value, back when line faults were
# generated in this file -- see generate_data_line_only.py, which now owns
# that code path and this same guard).
_MIN_MARGIN = 0.01
for _name, _clears in [('t_clears_load', t_clears_load),
                        ('t_clears_gen', t_clears_gen)]:
    _bad = [tc for tc in _clears if tc <= args.t_fault + _MIN_MARGIN]
    if _bad:
        raise ValueError(
            f"{_name} contains clear_time(s) {_bad} too close to or before "
            f"fault_time ({args.t_fault}) -- would clear before/immediately "
            f"at the fault. Use values >= fault_time + {_MIN_MARGIN}.")

n_load = len(load_buses) * len(alpha_load) * len(t_clears_load)
n_gen  = len(gen_buses)  * len(alpha_gen)  * len(t_clears_gen)
total  = n_load + n_gen

print(f"\nLoad fault scenarios : {n_load}")
print(f"Gen trip  scenarios : {n_gen}")
print(f"Total               : {total}")
print(f"t_total             : {args.t_total} s\n")

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def check_stable(omega, t, t_clear):
    """Check stability in window (t_clear, t_clear+2s)."""
    window = (t > t_clear) & (t < (t_clear + 2.0))
    if window.sum() == 0:
        window = t > t_clear
    dev = float(np.abs(omega[window] - 1.0).max())
    return dev < args.omega_tol, dev

def extract_states(result):
    t_arr  = result['time']
    states = result['full_states']
    idx    = np.arange(0, len(t_arr), args.downsample)
    t_arr  = t_arr[idx]
    states = states[idx]
    delta_abs = states[:, [0, 2, 4]]
    delta_rel = delta_abs - delta_abs[:, [0]]
    omega     = states[:, [1, 3, 5]]
    Pm        = states[:, [6, 7, 8]]
    Id        = states[:, [9, 11, 13]]   # Id0, Id1, Id2 (interleaved layout)
    Iq        = states[:, [10, 12, 14]]  # Iq0, Iq1, Iq2 (interleaved layout)
    return t_arr, delta_rel, omega, Pm, Id, Iq

def store(n_t, t_arr, alpha, loc, type_id, t_clear,
          delta_rel, omega, Pm, Id, Iq, stable):
    all_t.append(t_arr)
    all_alpha.append(np.full(n_t, alpha))
    all_loc.append(np.full(n_t, float(loc)))
    all_type_id.append(np.full(n_t, float(type_id)))
    all_t_clear.append(np.full(n_t, t_clear))
    all_delta.append(delta_rel)
    all_omega.append(omega)
    all_Pm.append(Pm)
    all_Id.append(Id)
    all_Iq.append(Iq)
    all_stable.append(np.full(n_t, float(stable)))

# ──────────────────────────────────────────────────────────────
# Storage
# ──────────────────────────────────────────────────────────────
all_t=[];all_alpha=[];all_loc=[];all_type_id=[]
all_t_clear=[];all_delta=[];all_omega=[];all_Pm=[]
all_Id=[];all_Iq=[]
all_stable=[];failed=[]
Eq_prime_ref=None; Ed_prime_ref=None
t_start=time.time();run_idx=0;n_stable=0;n_unstable=0

hdr = f"{'Run':<8}{'Type':<10}{'Loc':<6}{'Alpha':<8}{'tc':<7}{'Status':<12}{'dev':<10}{'Time'}"
sep = "-"*72

# ──────────────────────────────────────────────────────────────
# 1. Load faults (type_id = 0)
# ──────────────────────────────────────────────────────────────
print("=== Load faults ===")
print(hdr); print(sep)
for bus_idx in load_buses:
    for alpha in alpha_load:
        for t_clear in t_clears_load:
            run_idx += 1
            t0 = time.time()
            try:
                result = run_with_timeout(
                    run_dae_with_clearing, args.timeout,
                    fault_type='load', fault_loc=bus_idx,
                    fault_mag=float(alpha), params=params,
                    fault_time=args.t_fault, clear_time=t_clear,
                    total_time=args.t_total, **gov_kwargs)
                t_arr, dr, om, pm, Id, Iq = extract_states(result)
                if Eq_prime_ref is None:
                    Eq_prime_ref = result['Eq_prime']
                    Ed_prime_ref = result['Ed_prime']
                stable, dev = check_stable(om, t_arr, t_clear)
                store(len(t_arr), t_arr, alpha, bus_idx, 0,
                      t_clear, dr, om, pm, Id, Iq, stable)
                if stable: n_stable+=1
                else: n_unstable+=1
                mk = '✓ stable' if stable else '✗ UNSTABLE'
                print(f"  [{run_idx:3d}/{total}] load      {bus_idx:<6}{alpha:<8.2f}{t_clear:<7.1f}{mk:<12}{dev:.5f}  ({time.time()-t0:.0f}s)")
            except Exception as e:
                print(f"  [{run_idx:3d}/{total}] load      FAILED: {e}")
                failed.append({'type':'load','loc':bus_idx,'alpha':float(alpha),'t_clear':t_clear,'error':str(e)})

# ──────────────────────────────────────────────────────────────
# 2. Generator trips (type_id = 1)
# ──────────────────────────────────────────────────────────────
print("\n=== Generator trips ===")
print(hdr); print(sep)
for gen_idx in range(len(gen_buses)):
    for alpha in alpha_gen:
        for t_clear in t_clears_gen:
            run_idx += 1
            fault_mag = -float(alpha) * Pm0[gen_idx]
            t0 = time.time()
            try:
                result = run_with_timeout(
                    run_dae_with_clearing, args.timeout,
                    fault_type='gen', fault_loc=gen_idx,
                    fault_mag=fault_mag, params=params,
                    fault_time=args.t_fault, clear_time=t_clear,
                    total_time=args.t_total, **gov_kwargs)
                t_arr, dr, om, pm, Id, Iq = extract_states(result)
                if Eq_prime_ref is None:
                    Eq_prime_ref = result['Eq_prime']
                    Ed_prime_ref = result['Ed_prime']
                stable, dev = check_stable(om, t_arr, t_clear)
                store(len(t_arr), t_arr, alpha, gen_idx, 1,
                      t_clear, dr, om, pm, Id, Iq, stable)
                if stable: n_stable+=1
                else: n_unstable+=1
                mk = '✓ stable' if stable else '✗ UNSTABLE'
                print(f"  [{run_idx:3d}/{total}] gen={gen_idx}    {gen_idx:<6}{alpha:<8.2f}{t_clear:<7.1f}{mk:<12}{dev:.5f}  ({time.time()-t0:.0f}s)")
            except Exception as e:
                print(f"  [{run_idx:3d}/{total}] gen={gen_idx}    FAILED: {e}")
                failed.append({'type':'gen','loc':gen_idx,'alpha':float(alpha),'t_clear':t_clear,'error':str(e)})

# ──────────────────────────────────────────────────────────────
# Assemble & save
# ──────────────────────────────────────────────────────────────
t_all       = np.concatenate(all_t)
alpha_all   = np.concatenate(all_alpha)
loc_all     = np.concatenate(all_loc)
type_id_all = np.concatenate(all_type_id)
t_clear_all = np.concatenate(all_t_clear)
delta_all   = np.vstack(all_delta)
omega_all   = np.vstack(all_omega)
Pm_all      = np.vstack(all_Pm)
Id_all      = np.vstack(all_Id)
Iq_all      = np.vstack(all_Iq)
stable_all  = np.concatenate(all_stable)

X_train = np.column_stack([t_all, alpha_all, loc_all, type_id_all, t_clear_all])
Y_train = np.column_stack([delta_all, omega_all, Pm_all, Id_all, Iq_all])

print(f"\n-- Sanity checks --")
assert not np.any(np.isnan(X_train)), "NaN in X_train!"
assert not np.any(np.isnan(Y_train)), "NaN in Y_train!"
assert not np.any(np.isinf(Y_train)), "Inf in Y_train!"
print(f"  OK No NaN/Inf")
print(f"  OK Pm range: [{Pm_all.min():.4f}, {Pm_all.max():.4f}] pu")
print(f"  OK Stable  : {n_stable}  Unstable: {n_unstable}  "
      f"({n_unstable/(n_stable+n_unstable)*100:.1f}% unstable)")
print(f"\nX_train: {X_train.shape}  Y_train: {Y_train.shape}")

np.save(os.path.join(args.outdir, 'X_train.npy'), X_train)
np.save(os.path.join(args.outdir, 'Y_train.npy'), Y_train)
np.save(os.path.join(args.outdir, 'stable_labels.npy'), stable_all)

meta = {
    "X_train_cols"    : ["t", "alpha", "fault_loc", "fault_type_id", "t_clear"],
    "Y_train_cols"    : ["delta_1_rel","delta_2_rel","delta_3_rel",
                         "omega_0","omega_1","omega_2","Pm_0","Pm_1","Pm_2",
                         "Id_0","Id_1","Id_2","Iq_0","Iq_1","Iq_2"],
    "fault_type_ids"  : {"0":"load_fault","1":"gen_trip"},
    "delta_convention": "relative to Gen 0",
    "stability_label" : f"max|omega-1|<{args.omega_tol} in (t_clear, t_clear+2s)",
    "load_buses":load_buses,"gen_buses":gen_buses,"Pm0":Pm0.tolist(),
    "alpha_load":alpha_load if isinstance(alpha_load,list) else alpha_load.tolist(),
    "alpha_gen" :alpha_gen  if isinstance(alpha_gen, list) else alpha_gen.tolist(),
    "t_clears_load":t_clears_load,"t_clears_gen":t_clears_gen,
    "t_fault":args.t_fault,"t_total":args.t_total,
    "dt_out":0.01*args.downsample,
    "governor":{"R":GOVERNOR_R,"Tg":GOVERNOR_TG},
    "Eq_prime": Eq_prime_ref.tolist() if Eq_prime_ref is not None else None,
    "Ed_prime": Ed_prime_ref.tolist() if Ed_prime_ref is not None else None,
    "n_stable":n_stable,"n_unstable":n_unstable,
    "n_failed":len(failed),"failed":failed,
    "X_shape":list(X_train.shape),"Y_shape":list(Y_train.shape),
    "note": "load + gen data only -- line/transformer faults are generated "
            "separately by generate_data_line_only.py, then combined via "
            "merge_line_data.py.",
}
with open(os.path.join(args.outdir,'meta.json'),'w') as f:
    json.dump(meta,f,indent=2)

print(f"\nSaved to ./{args.outdir}/")
print(f"  X_train.npy       {X_train.shape}")
print(f"  Y_train.npy       {Y_train.shape}")
print(f"  stable_labels.npy {stable_all.shape}")
if failed: print(f"\n  WARNING: {len(failed)} failed")
print(f"\nTotal time: {time.time()-t_start:.1f} s")
