# Physics-informed fault identification and frequency control for IEEE 9-bus and 30-bus systems

A receding-horizon neural controller for power-system transient stability.
From a short window of measurements it:

1. identifies the ongoing disturbance: which fault (generator trip, load step
   or line outage), where, and how severe;
2. rolls the swing equations forward through a differentiable physics model to
   predict whether the frequency will leave its ±0.005 pu envelope;
3. emits a corrective governor set-point change ΔP that keeps it inside.

Out-of-limit (OOL) prediction and ΔP are not learned labels. They are read off
the physics rollout of the identified system, so every prediction the model
makes can be checked against a simulation of the real grid.

The same architecture is trained on two systems:

| | [`bus_9/`](bus_9) | [`bus_30/`](bus_30) |
|---|---|---|
| System | IEEE 9-bus, 3 generators | IEEE 30-bus, 2 generators + 4 synchronous condensers |
| Dynamic states | 9 | 18 |
| Measured channels → features | 33 → 31 | 90 → 88 |
| Fault hypotheses (incl. "no fault") | 13 | 57 |
| Governed machines (ΔP width) | 3 | 2 (G1, G2) |
| Hidden width / LSTM layers / trunk layers | 128 / 3 / 4 | 384 / 5 / 6 |
| Parameters | 637,954 | 8,344,117 |

---

## Contents

- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Data](#data)
- [Running the 9-bus pipeline](#running-the-9-bus-pipeline)
- [Running the 30-bus pipeline](#running-the-30-bus-pipeline)
- [Evaluation protocol](#evaluation-protocol)
- [Results](#results)
- [Known issues and limitations](#known-issues-and-limitations)
- [Earlier work](#earlier-work)

---

## How it works

```
measured window (100 steps × 10 ms)
  → per-channel features (relative angles, ω, Pm, Id, Iq, Vm, relative θ)
  → learnable soft time mask             (window length L is a trained parameter)
  → bias-free linear projection          (masked steps contribute exactly zero)
  → residual LSTM stack with LayerNorm
  → learnable softmax pooling over time → MLP trunk
  → heads: fault hypothesis · severity α · initial-state correction · ΔP
  → differentiable network build (Kron reduction of the faulted admittance matrix)
  → RK4 rollout of the swing + governor equations
  → OOL readout (peak |ω − 1| over the judged window) and ΔP evaluation
```

**Physics.** Each machine has three states:

| State | Equation |
|---|---|
| rotor angle δ | dδ/dt = 2π·60·(ω − 1) |
| rotor speed ω | dω/dt = (Pm − Pe − D(ω − 1)) / 2H |
| mechanical power Pm | dPm/dt = (Pm_set − (ω − 1)/R − Pm) / Tg |

Pe comes from the Kron-reduced network, built for each fault hypothesis. The
ports in `ieee9_physics.py` and `case30_physics.py` reproduce the dataset
trajectories at the measurement-noise floor (see [Results](#results)).

**Identification branch.** The head's hypothesis probabilities build a
soft-weighted faulted network. The free rollout (ΔP = 0) is supervised against
the measured future, which is what trains identification through the physics.

**Control branch.** The context (trunk features, hypothesis, α, initial state,
network, set-point) is **detached** before it reaches the ΔP head, so the
control loss cannot reshape identification. Without this, the controller could
lower its loss by misidentifying the system as one that is easier to control.

**Cascade ΔP head** (`--dp_head cascade`). An LSTM cell and a small FC stack
refine ΔP over three steps. Each step rolls the current ΔP forward, describes
the resulting excursion (peaks, timing, sign), and feeds that back in.
Effectively it is a learned optimiser with the simulator in the loop.

**Receding horizon.** The rollout assumes the fault persists for the whole
horizon (1.5 s). A deployed controller would re-decide every ~20 ms and absorb
the uncertainty about when the fault clears. The evaluation in this repository
scores a single decision (see [limitations](#known-issues-and-limitations)).

---

## Repository layout

```
.
├── README.md
├── bus_9/
│   ├── DATASET.md                  dataset description (see "Data" for two errata)
│   ├── ieee9_base.mat              base admittance matrix, buses
│   ├── ieee9_gov_simulink_data.mat machine constants, initial state, reduced network
│   ├── compute_delta_p.m           MATLAB reference for faults and ΔP labels
│   ├── ieee9_physics.py            differentiable 9-bus model (Kron, faults, RK4)
│   ├── pinn_model.py               FaultEncoder + RolloutController
│   ├── convert_dataset.py          .mat → scenario-major float32 memmap
│   ├── train_receding.py           training
│   ├── refine_dp.py                constrained ||ΔP|| minimisation
│   ├── diagnose_dp.py              head vs oracle vs feasibility ceiling
│   └── verify_true.py              true-dynamics verification + plots
└── bus_30/
    ├── CASE30_DATASET.md           dataset description
    ├── case30_ieee.json            system definition
    ├── case30_base.mat             base admittance matrix, exact branch data
    ├── case30_simulink_data.mat    machine constants, initial state, reduced network
    ├── case30_lib.m                MATLAB reference (scenario construction, fast model)
    ├── case30_compute_delta_p.m    MATLAB reference for the ΔP labels
    ├── case30_physics.py           differentiable 30-bus model
    ├── verify_case30.py            physics port vs dataset (run first)
    ├── convert_case30.py           40 shards + labels → one memmap
    ├── check_data30.py             converted data sanity check
    ├── oracle_case30.py            trajectory error of a perfect identifier
    ├── pinn_model30.py             FaultEncoder + RolloutController, 30-bus dimensions
    ├── train_case30.py             training
    ├── case30_common.py            shared loader, batch builder, actionability gate
    ├── refine_dp30.py              constrained ||ΔP|| minimisation
    ├── diagnose_dp30.py            head vs oracle vs feasibility ceiling
    └── verify_true30.py            true-dynamics verification + plots
```

Each folder is self-contained. Run its scripts from inside that folder: the
default paths to the `.mat` support files are relative to the working
directory.

The 30-bus post-training scripts (`refine_dp30.py`, `diagnose_dp30.py`,
`verify_true30.py`) all import their batch builder and actionability gate
from `case30_common.py`. That module is tested draw-for-draw against
`train_case30.py`:

```
python case30_common.py --selftest --ckpt case30_ctrl/best.pt --data data30_mm
```

---

## Requirements

- Python 3.11
- PyTorch with CUDA for your GPU (developed on an RTX 5070 Ti Laptop, 12 GB)
- NumPy, SciPy, h5py, Matplotlib

```
pip install numpy scipy h5py matplotlib
# plus a CUDA build of torch from https://pytorch.org
```

Memory used during development:

| | 9-bus | 30-bus |
|---|---|---|
| Converted dataset | ~2.1 GB (5 ms) | 2.69 GB (10 ms) |
| Batch size used | 1024 | 512 |
| RAM | converted data should fit in page cache | same; 31 GB was enough |

---

## Data

The datasets are too large for git and are not included. Both were generated
in Simulink. Each has 10,000 scenarios of 8 s, stored at 1 ms (8001 steps),
with Gaussian measurement noise (ω: 5·10⁻⁵ pu, δ: 0.01 rad, Pm/Id/Iq:
0.005 pu, Vm: 0.001 pu, θ: 0.01 rad).

| | 9-bus | 30-bus |
|---|---|---|
| Files | `pinn_dataset.mat` | `case30_dataset.mat` (labels) + `case30_X_001…040.mat` (40 shards × 250 scenarios) |
| Size | — | 28.8 GB |
| Out of limit | ~45–50% | 37.9% |
| ΔP labels | 3 per scenario | 6 per scenario; only columns 1–2 (G1, G2) are nonzero |
| Not correctable within ±1 pu (`dp_status 3`) | ~14% of OOL | 16.0% of OOL |

**Errata in `bus_9/DATASET.md`:**
1. It says the data is stored every 10 ms. It is stored at **1 ms** (T = 8001).
2. Rows 4 and 5 of the `lines` table in §1 are swapped relative to the stored
   `lines` variable (the stored order is `[7 8]` then `[6 9]`). `meta_loc`
   indexes the **stored** table; the scripts always read it from the dataset.

**Recommended `.gitignore`:**

```
*.npy
data_mm/
data30_mm/
pinn_dataset.mat
case30_dataset.mat
case30_X_*.mat
case30_delta_p_labels.mat
**/best.pt
**/last.pt
**/refined/
**/verify/
__pycache__/
```

The small support files (`*_base.mat`, `*_simulink_data.mat`,
`case30_ieee.json`) are a few KB each and should be committed. The physics
models need them.

---

## Running the 9-bus pipeline

From `bus_9/`, with `pinn_dataset.mat` in the folder:

```
# 1. convert to a memmap (every 5th sample -> 5 ms)
python convert_dataset.py --mat pinn_dataset.mat --out data_mm

# 2. train identification + control
python train_receding.py --data data_mm --out receding_cascade ^
  --epochs 500 --batch_size 1024 --instants_per_scenario 4 ^
  --w_max 1.0 --dt_window 0.01 --init_window 0.5 --lam_window 0.05 ^
  --lstm_layers 3 --ctrl_start 100 ^
  --dp_head cascade --n_refine 3 --w_dp_norm 1.0 --margin 1e-3

# 3. is the ΔP head the bottleneck?
python diagnose_dp.py --ckpt receding_cascade/best.pt --data data_mm --n 512 --steps 400

# 4. minimise ||ΔP|| at constant rescue rate
python refine_dp.py --ckpt receding_cascade/best.pt --data data_mm --floor_tol 0

# 5. check the emitted ΔP in the true dynamics, and plot
python verify_true.py --ckpt receding_cascade/refined/best.pt --data data_mm --n 2048 --plots_per_type 3
```

(`^` is the PowerShell/cmd line continuation; use `\` on Linux/macOS.)

---

## Running the 30-bus pipeline

From `bus_30/`, with `case30_dataset.mat`, the 40 `case30_X_*.mat` shards and
the support files in the folder:

```
# 0. the physics port must reproduce the dataset at the noise floor
python verify_case30.py --mat case30_dataset_small.mat
#    expect: median omega RMSE ~5.0e-05, OOL agreement 100/100

# 1. convert the 40 shards (placed by scenario_ids, not file order)
python convert_case30.py --labels case30_dataset.mat --out data30_mm
#    optional: --agg mean averages each 10 samples instead of keeping one (~3x less noise)

# 2. check the conversion before spending GPU hours
python check_data30.py --data data30_mm
#    expect: OOL-from-X vs y_ool agreement ~0.997, 10000/10000 covered

# 3. optional: trajectory error a PERFECT identifier would get
python oracle_case30.py --data data30_mm --n 512

# 4a. identification only (control branch off)
python train_case30.py --data data30_mm --out case30_v3 ^
  --epochs 60 --batch_size 512 --instants_per_scenario 4 ^
  --w_max 1.0 --dt_window 0.01 --init_window 0.5 --lam_window 0 ^
  --lstm_layers 5 --ctrl_start 999 ^
  --dp_head cascade --n_refine 3 --w_dp_norm 1.0 --margin 1e-3

# 4b. add control on top (weights, feature statistics AND split carried over)
python train_case30.py --data data30_mm --out case30_ctrl --init case30_v3/best.pt ^
  --epochs 200 --batch_size 512 --instants_per_scenario 4 ^
  --w_max 1.0 --dt_window 0.01 --init_window 0.5 --lam_window 0.05 ^
  --lstm_layers 5 --ctrl_start 1 ^
  --dp_head cascade --n_refine 3 --w_dp_norm 1.0 --margin 1e-3

# 5. post-training
python diagnose_dp30.py --ckpt case30_ctrl/best.pt --data data30_mm
python refine_dp30.py   --ckpt case30_ctrl/best.pt --data data30_mm --floor_tol 0
python verify_true30.py --ckpt case30_ctrl/refined/best.pt --data data30_mm --split test
```

Notes:

- **Feature normalisation is on by default and is essential.** Without it the
  time-varying part of the input is ~6% of its magnitude (the 30 bus-voltage
  channels sit at a near-constant 0.98). Identification then collapses to a
  constant hypothesis. `--feat_norm 0` reproduces this, as an ablation.
- **`--init`** continues from a checkpoint. It reuses that checkpoint's
  weights, feature statistics and split, so a continued run can never
  validate on scenarios it trained on.
- **`refine_dp30.py`** does not start counting patience until a warm-up has
  passed (`--warmup 40`). Its multiplier starts high and needs time to decay
  before ‖ΔP‖ can fall. On the 9-bus run, ‖ΔP‖ sat above the baseline for the
  first 23 epochs.
- **`verify_true30.py`** plots each fault type, preferring cases rescued over
  the whole run. That makes them selected examples, not typical ones. Each
  title states which tier the case came from.

---

## Evaluation protocol

Most of the large metric changes during development came from flaws in the
evaluation protocol, not from the model. The protocol used by every script:

| Rule | Why |
|---|---|
| **Trajectory loss is masked to the valid horizon.** It stops at `t_clear` if a fault is active, and at `t_fault` if pre-fault. | The rollout holds the fault; the real system clears it. Supervising past the change penalises an unknowable event. (9-bus traj 0.0221 → 0.00109.) |
| **OOL is scored against the measured future over the same judged window**, not the whole-run label `y_ool`. | Like-for-like comparison. (9-bus OOL acc 0.853 → 0.966.) |
| **Decision instants are drawn from `t_fault + U(0, 0.2 s)` for a share `p_active`**, the rest uniformly. | Matches the 50 ms detection convention of the ΔP labels. |
| **Rescue is scored only on actionable instants**: fault active now, ω still inside the envelope now, non-empty judged window. | Instants after clearing, or already outside the envelope, are failures no controller could avoid. |
| **ΔP is held until the fault clears** (`--dp_hold clear`). | Same convention as `y_delta_p`. |
| **Governor grace period of 200 ms**: peaks inside it are not scored. | Tg = 0.2 s; no set-point change acts sooner. |
| **Primary control metrics are rescue rate and mean ‖ΔP‖.** Distance to `y_delta_p` is reported for reference only. | `y_delta_p` is the minimum-norm solution to a different question, judged to the end of the run. |

**Rescue rate** = fraction of actionable decision instants whose judged-window
peak |ω − 1| exceeds 0.005 without ΔP, and stays ≤ 0.005 with it.

Changing the gate changes the denominator. Rescue rates from before and after
a change to this definition are not comparable; always state the definition
alongside a figure.

---

## Results

As of 24 September 2026, single training run per system. "Model" figures are
measured inside the model's own identified dynamics. "True" figures apply the
emitted ΔP in a simulation of the real faulted system.

### Physics ports

| | 9-bus | 30-bus |
|---|---|---|
| ω RMSE vs dataset, 100 scenarios (noise floor 5.0·10⁻⁵) | median 5.008·10⁻⁵ | median 5.001·10⁻⁵, max 5.03·10⁻⁵ |
| OOL agreement | 100/100 | 100/100 |
| Reduced admittance matrix vs stored | 1.8·10⁻¹⁵ | 1.5·10⁻¹⁵ |

### 9-bus (`receding_cascade/best.pt`, epoch 151, test split)

| Metric | Value |
|---|---|
| Trajectory loss (ω RMSE) | 0.00464 (6.81·10⁻⁴) |
| Hypothesis accuracy (13 classes) | 0.9902 |
| OOL accuracy (TPR / TNR) | 0.9583 (0.923 / 0.974) |
| Severity α RMSE (range 0.1–1.0) | 0.0605 |
| Rescue rate, model dynamics | 0.9943 |
| Mean ‖ΔP‖ | 0.1463 pu |
| Learned window | 97.5 ms (86–107 ms across runs) |

- **Diagnostic:** the head (0.9947) is within half a point of the
  gradient-descent oracle and the feasibility ceiling (0.995–1.000 across
  runs), so the head is not the bottleneck.
- **Cascade vs linear ΔP head:** rescue 0.9570 vs 0.9603, a difference within
  noise (1 SE ≈ 0.0064). ‖ΔP‖ is 20% lower with the cascade (0.112 vs
  0.140 pu).
- **Refinement:** an earlier run reported a 26.2% ‖ΔP‖ reduction. See
  [known issues](#known-issues-and-limitations) before quoting it.

### 30-bus (`case30_ctrl/best.pt`, epoch 45)

**Identification and prediction** (test split, 500 scenarios):

| Metric | Value |
|---|---|
| Trajectory loss (ω RMSE) | 0.00908 (9.53·10⁻⁴) |
| Hypothesis accuracy (57 classes) | 0.9851 |
| — given a fault is active: exact class / type | 0.9907 / 0.9965 |
| — top-5 / fault detection | 0.9969 / 0.9858 |
| OOL accuracy (TPR / TNR) | 0.9610 (0.979 / 0.955) |
| Severity α RMSE | 0.0701 |
| Learned window | 128.3 ms (not converged; see limitations) |

A perfect identifier (true fault and severity, measured starting state) would
get trajectory loss 0.00031. With the starting state averaged over 5 samples
it would get 0.00014 (`oracle_case30.py`).

**Control, model dynamics** (test split): rescue 0.8671 on actionable cases,
overall ok 0.9572, mean ‖ΔP‖ 0.1657 pu.

**Head vs ceiling** (`diagnose_dp30.py`, test split, decisions within 200 ms
of onset, 166 cases needing rescue):

| | Rescue | Mean ‖ΔP‖ |
|---|---|---|
| Trained head | 0.9217 | 0.600 pu |
| Oracle, same objective | 0.8916 | 0.517 pu |
| Feasible, no effort penalty | 0.9398 | 0.728 pu |

The head is within 1.8 points of the feasibility ceiling. 10 of its 13 misses
are also missed by the feasibility oracle, and 9 of the 13 are
`dp_status 3`. The remaining failures are actuator limits: only G1 and G2 have
governors, and a large G1 trip leaves a deficit bigger than ±1 pu.

**True dynamics** (`verify_true30.py`, validation split, 2048 instants,
337 needing rescue):

| Metric | Value |
|---|---|
| Rescue, true dynamics | 0.8309 |
| Rescue, model dynamics (same cases) | 0.8843 |
| — load faults (n = 186) | 0.973 |
| — generator trips (n = 148) | 0.669 |
| Healthy instants pushed out of the envelope by ΔP | 0 of 1442 |
| Rescued through to the end of the run (with / without ΔP) | 0.700 / 0.000 |
| Mean ‖ΔP‖: needing rescue / not needing | 0.630 / 0.081 pu |

The model-internal rescue rate is 5.3 points optimistic, almost entirely on
generator trips and on the earliest decisions (0–50 ms: 0.77 true vs 0.89
model).

---

## Known issues and limitations

**Evaluation scope**
- **Single-decision evaluation.** Every figure scores one decision with ΔP held
  until the fault clears. The receding-horizon controller the design assumes
  (re-deciding every ~20 ms) has not been replayed in closed loop. This is the
  main reason the 30-bus end-of-run rescue (0.700) is below the in-window
  figure (0.831): faults last up to 3 s, beyond the 1.5 s horizon.
- **Single seed per configuration.** The cascade head's advantages (lower
  ‖ΔP‖, slightly better identification) come from one run per arm.
- **No independent Simulink check of the final ΔP.** Verification uses the
  Python port, which reproduces the Simulink data at the noise floor.
- **Validation vs test.** `verify_true30.py` defaults to the validation split;
  use `--split test` for reported numbers.

**30-bus**
- **Generator trips are the weak spot:** 0.669 true rescue. It is not yet
  known whether the cause is marginal solutions (the model's rescue just
  clears the margin, the real system just misses) or severity underestimated
  early (Pm reaches a new set-point only through the 0.2 s governor lag).
- **The safety margin roughly equals the model's own error.** The margin is
  1·10⁻³, while ω RMSE on the test set is 9.5·10⁻⁴. Refinement spends that
  margin, so refined results must be re-verified in true dynamics.
- **The learned window did not converge.** It fell monotonically from 500 ms
  and was still falling when early stopping fired. The mask edge (β = 26.9 ms)
  was also not fully annealed at the selected epoch. Treat 128.3 ms as
  provisional.
- **Refinement has not yet produced a cheaper feasible ΔP.** The first run
  stopped on patience during the multiplier's warm-up; `--warmup` fixes that.
  The diagnostic suggests the head is already near the best rescue/effort
  trade-off, so expect a small reduction, not the 9-bus 26%.

**9-bus scripts (fixed in the 30-bus versions, not yet back-ported)**
- **Mask edge overwritten on load.** `refine_dp.py`, `diagnose_dp.py` and
  `verify_true.py` set β to `--beta_min` after loading. The checkpoint was
  selected with a softer edge, so these scripts evaluate a slightly different
  window than training reported.
- **Refinement compares different splits.** `refine_dp.py` compares a
  validation baseline against a test result (the 26.2% figure is validation
  0.1043 → test 0.0770). It also saves the first epoch that meets the rescue
  floor even if its ‖ΔP‖ is higher than the baseline's. Re-measure on test
  before quoting.
- **No feature normalisation** in `train_receding.py`. It trains well without
  it on the 9-bus, but the two systems differ in this respect.
- **The batch builder and gate are duplicated** across four 9-bus scripts. They
  drifted apart twice during development. The 30-bus side uses one shared
  module with a self-test.

**Both systems**
- **Split.** The training scripts regenerate a 90/5/5 split with a fixed seed
  and store it in the checkpoint. They ignore the 80/10/10 `split` shipped in
  the datasets.
- **Sampling.** Both models read every 10th raw sample (10 ms). On the 9-bus
  this happens in two steps: the converter keeps every 5th sample, and the
  window reads every 2nd. Averaging instead of decimating (`--agg mean`) would
  cut noise ~3× and is an open ablation.
- **Line faults that peak after reconnection** (the `dp_status 3` line cases)
  cannot be corrected by a constant set-point change withdrawn at clearing.
  This is a limit of the actuator, not of the model.
- **No fault-time head.** Fault time is not predicted; the decision is
  anchored to the measurement window only.

---

## Earlier work

Before the receding-horizon formulation, a pointwise PINN mapped
`(t, α, fault location, t_clear) → 14 states` with an ODE-residual physics
loss (`pinn_multifault4.py`, `networks_seq.py`, `plot_validation.py`). It is
superseded by the pipelines above. If kept, it belongs in a separate `legacy/`
folder.

A generic system-agnostic module (`power_system.py`, `psys_physics.py`,
`system_io.py`, `verify_physics.py`, `validate_case30.py`) was drafted and
then abandoned in favour of the separate `bus_30/` pipeline. It is not used by
either pipeline and should not be committed alongside them.
