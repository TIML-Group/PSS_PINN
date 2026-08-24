"""
generate_data_line_only.py
=============================
Regenerates the line/transformer-fault portion of the dataset:
  - 6 transmission lines (as before) + 3 generator step-up transformers
    (NEW -- previously excluded as a different equipment category, now
    added per request)
  - t_clear grid: 0.25 -> 4.0 in steps of 0.1 (38 values, 342 scenarios
    total across 9 locations) -- finer than the original 9-value grid,
    still starting safely above fault_time (0.2) to avoid the t_clear=0.1
    bug from the original run (see generate_data_v9.py for that story).

Does NOT touch load faults or gen trips -- if your existing gen/load data
is already trusted, there's no need to regenerate those 330 scenarios just
to expand the line-fault set.

Output format is IDENTICAL to generate_data_v9.py's (same X_train_cols,
Y_train_cols, same meta.json structure) so this can be used as a drop-in
replacement for retraining just the line-fault model, OR merged with your
existing gen/load .npy files (see merge_line_data.py, generated alongside
this) to rebuild one combined dataset.

COST WARNING: 342 scenarios at ~35s each (observed rate) is roughly
3-3.5 hours. Consider running this unattended.

Usage:
    python generate_data_line_only.py --params ieee9_parameters.json

Requires cross_validation_datagen_governor.py and ieee9_parameters.json in
the same folder.

Output (in --outdir, default 'training_data_line_v2'):
    X_train.npy, Y_train.npy, stable_labels.npy, meta.json
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
parser.add_argument('--outdir',     type=str,   default='training_data_line_v2')
parser.add_argument('--t_fault',    type=float, default=0.2)
parser.add_argument('--t_total',    type=float, default=8.0)
parser.add_argument('--downsample', type=int,   default=5)
parser.add_argument('--omega_tol',  type=float, default=0.005)
parser.add_argument('--timeout',    type=int,   default=120)
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Load parameters
# ──────────────────────────────────────────────────────────────
with open(args.params) as f:
    params = json.load(f)

# IEEE 9-bus transmission lines (0-indexed bus pairs) -- unchanged from v9.
# These 6 branches have BOTH real and imaginary Y entries (resistance +
# reactance) -- the signature of actual transmission lines.
transmission_lines = [(3, 4), (3, 5), (4, 6), (5, 8), (6, 7), (7, 8)]

# NEW: generator step-up transformers. Confirmed via the Y_bus itself: each
# generator bus (0,1,2) connects to exactly ONE other bus (3,6,8), with a
# PURELY REACTIVE entry (Y_real=0.0) -- the signature of a transformer
# branch, not a transmission line. These were correctly excluded from the
# original "line fault" sweep (transformers are a different equipment
# category with different fault modes in real power systems), but are now
# being added as their own contingency locations per user request.
gen_transformers = [(0, 3), (1, 6), (2, 8)]

all_lines = transmission_lines + gen_transformers
lines = all_lines
print(f"Testing {len(transmission_lines)} transmission lines + "
      f"{len(gen_transformers)} generator transformers = {len(lines)} locations")

# Finer t_clear grid: 0.25 -> 4.0 in steps of 0.1 (38 values). Starts at
# 0.25 (not 0.2/0.1) to stay safely above fault_time=0.2 -- see
# generate_data_v9.py for the full story on why clear_time must exceed
# fault_time.
t_clears_line = [round(0.25 + 0.1 * i, 2) for i in range(38)]

# Defensive guard -- same as generate_data_v9.py, kept here too so this
# script can't silently reintroduce the same bug if edited later.
_MIN_MARGIN = 0.01
_bad = [tc for tc in t_clears_line if tc <= args.t_fault + _MIN_MARGIN]
if _bad:
    raise ValueError(
        f"t_clears_line contains clear_time(s) {_bad} too close to or "
        f"before fault_time ({args.t_fault}). Use values >= "
        f"fault_time + {_MIN_MARGIN}.")

n_line = len(lines) * len(t_clears_line)
print(f"Line fault scenarios: {n_line}")
print(f"t_clears_line: {t_clears_line}")
print(f"t_total: {args.t_total} s\n")


# ──────────────────────────────────────────────────────────────
# Helpers (identical to generate_data_v9.py)
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
    Id        = states[:, [9, 11, 13]]
    Iq        = states[:, [10, 12, 14]]
    return t_arr, delta_rel, omega, Pm, Id, Iq

def build_line_fault_Y_adm(bus_r, bus_s):
    sd    = get_system_data(params)
    Y_adm = sd['Y_adm'].copy()
    line_adm            = Y_adm[bus_r, bus_s]
    Y_adm[bus_r, bus_s] = 0.0
    Y_adm[bus_s, bus_r] = 0.0
    Y_adm[bus_r, bus_r]+= line_adm
    Y_adm[bus_s, bus_s]+= line_adm
    return Y_adm

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
# Line faults ONLY (type_id = 2, same encoding as generate_data_v9.py so
# this stays merge-compatible)
# ──────────────────────────────────────────────────────────────
print("=== Line faults (fixed grid) ===")
print(hdr); print(sep)
for (bus_r, bus_s) in lines:
    Y_adm_fault = build_line_fault_Y_adm(bus_r, bus_s)
    line_id = float(bus_r * 10 + bus_s)
    for t_clear in t_clears_line:
        run_idx += 1
        t0 = time.time()
        try:
            result = run_with_timeout(
                run_dae_with_clearing, args.timeout,
                fault_type='none', fault_loc=0,
                fault_mag=0.0, params=params,
                fault_time=args.t_fault, clear_time=t_clear,
                total_time=args.t_total,
                Y_adm_fault=Y_adm_fault,
                gov_R=GOVERNOR_R, gov_Tg=GOVERNOR_TG)
            t_arr, dr, om, pm, Id, Iq = extract_states(result)
            if Eq_prime_ref is None:
                Eq_prime_ref = result['Eq_prime']
                Ed_prime_ref = result['Ed_prime']
            stable, dev = check_stable(om, t_arr, t_clear)
            store(len(t_arr), t_arr, 1.0, line_id, 2,
                  t_clear, dr, om, pm, Id, Iq, stable)
            if stable: n_stable += 1
            else: n_unstable += 1
            mk = '✓ stable' if stable else '✗ UNSTABLE'
            print(f"  [{run_idx:3d}/{n_line}] line ({bus_r},{bus_s}) "
                  f"{line_id:<6.0f}{1.0:<8.2f}{t_clear:<7.2f}{mk:<12}{dev:.5f}  "
                  f"({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"  [{run_idx:3d}/{n_line}] line ({bus_r},{bus_s}) FAILED: {e}")
            failed.append({'type': 'line', 'loc': f'{bus_r}-{bus_s}',
                            't_clear': t_clear, 'error': str(e)})

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
denom = n_stable + n_unstable
pct = (n_unstable / denom * 100) if denom else 0.0
print(f"  OK Stable  : {n_stable}  Unstable: {n_unstable}  ({pct:.1f}% unstable)")
print(f"\nX_train: {X_train.shape}  Y_train: {Y_train.shape}")

np.save(os.path.join(args.outdir, 'X_train.npy'), X_train)
np.save(os.path.join(args.outdir, 'Y_train.npy'), Y_train)
np.save(os.path.join(args.outdir, 'stable_labels.npy'), stable_all)

meta = {
    "X_train_cols"    : ["t", "alpha", "fault_loc", "fault_type_id", "t_clear"],
    "Y_train_cols"    : ["delta_1_rel","delta_2_rel","delta_3_rel",
                         "omega_0","omega_1","omega_2","Pm_0","Pm_1","Pm_2",
                         "Id_0","Id_1","Id_2","Iq_0","Iq_1","Iq_2"],
    "fault_type_ids"  : {"0":"load_fault","1":"gen_trip","2":"line_fault"},
    "line_encoding"   : "fault_loc = bus_r*10 + bus_s",
    "lines_tested"    : [list(l) for l in lines],
    "delta_convention": "relative to Gen 0",
    "stability_label" : f"max|omega-1|<{args.omega_tol} in (t_clear, t_clear+2s)",
    "t_clears_line"   : t_clears_line,
    "t_fault":args.t_fault,"t_total":args.t_total,
    "dt_out":0.01*args.downsample,
    "governor":{"R":GOVERNOR_R,"Tg":GOVERNOR_TG},
    "Eq_prime": Eq_prime_ref.tolist() if Eq_prime_ref is not None else None,
    "Ed_prime": Ed_prime_ref.tolist() if Ed_prime_ref is not None else None,
    "n_stable":n_stable,"n_unstable":n_unstable,
    "n_failed":len(failed),"failed":failed,
    "X_shape":list(X_train.shape),"Y_shape":list(Y_train.shape),
    "note": "line-faults-only regeneration, fixed t_clear grid (0.1->0.25). "
            "Merge with existing gen/load .npy files using merge_line_data.py "
            "before retraining, or use standalone to retrain just the "
            "line-fault model.",
}
with open(os.path.join(args.outdir,'meta.json'),'w') as f:
    json.dump(meta,f,indent=2)

print(f"\nSaved to ./{args.outdir}/")
print(f"  X_train.npy       {X_train.shape}")
print(f"  Y_train.npy       {Y_train.shape}")
print(f"  stable_labels.npy {stable_all.shape}")
if failed: print(f"\n  WARNING: {len(failed)} failed")
else: print(f"\n  All {n_line} scenarios succeeded (0 failed)")
print(f"\nTotal time: {time.time()-t_start:.1f} s")
