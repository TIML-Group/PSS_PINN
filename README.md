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

The same pipeline is built twice — once with LSTMs, once with size-matched
MLPs — as an ablation to test what recurrence actually buys.

The architecture is trained on two systems:

| | [`bus_9/`](bus_9) | [`bus_30/`](bus_30) |
|---|---|---|
| System | IEEE 9-bus, 3 generators | IEEE 30-bus, 2 generators + 4 synchronous condensers |
| Dynamic states | 9 | 18 |
| Measured channels → features | 33 → 31 | 90 → 88 |
| Fault hypotheses (incl. "no fault") | 13 | 57 |
| Governed machines (ΔP width) | 3 | 2 (G1, G2) |
| Hidden width / LSTM layers / trunk layers | 128 / 3 / 4 | 384 / 5 / 6 |
| Parameters (LSTM+LSTM) | 637,954 | 8,344,117 |

---

## Contents

- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Data](#data)
- [Running the 9-bus pipeline](#running-the-9-bus-pipeline)
- [Running the 30-bus pipeline](#running-the-30-bus-pipeline)
- [Ablation: LSTM vs size-matched MLP](#ablation-lstm-vs-size-matched-mlp)
- [Latency and CUDA-graph capture](#latency-and-cuda-graph-capture)
- [Evaluation protocol](#evaluation-protocol)
- [Results](#results)
- [Known issues and limitations](#known-issues-and-limitations)
- [Earlier work](#earlier-work)

---

## How it works

```
measured window (100 steps × 10 ms)
  → per-channel features (relative angles, ω, Pm, Id, Iq, Vm, relative θ)
  → per-channel normalisation (train-set statistics stored in the checkpoint)
  → learnable soft time mask             (window length L is a trained parameter)
  → bias-free linear projection          (masked steps contribute exactly zero)
  → residual LSTM stack with LayerNorm   (or a matched MLP encoder — see ablation)
  → learnable softmax pooling over time → MLP trunk
  → heads: fault hypothesis · severity α · initial-state correction · ΔP
  → differentiable network build (Kron reduction of the faulted admittance matrix)
  → RK4 rollout of the swing + governor equations
  → cascaded ΔP refinement (3 rounds, simulator in the loop)
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
The fault enters through the network and set-point rules (load fault: add α to
`Y[b,b]`; line fault: remove the exact branch entries; generator trip:
`Pset += −α·Pm0`), never through the state.

**Identification branch.** The head's hypothesis probabilities build a
soft-weighted faulted network. The free rollout (ΔP = 0) is supervised against
the measured future, which is what trains identification through the physics.

**Control branch.** The context (trunk features, hypothesis, α, initial state)
is **detached** before it reaches the ΔP head, so the control loss cannot
reshape identification (`Yred` and `Pset` are detached for the same reason).

**Cascade ΔP head** (`--dp_head cascade`). Three refinement rounds: each round
re-simulates with the current ΔP, describes the resulting excursion (peaks,
timing, sign), and feeds that back into the head. The LSTM/MLP ablation differs
in exactly one line: whether the cell's recurrent state `(h, c)` is threaded
through the rounds. Effectively it is a learned optimiser with the simulator
in the loop, trained by backpropagating through all 150 RK4 steps and all 3
rounds.

**Receding horizon.** The rollout assumes the fault persists for the whole
horizon (1.5 s); every loss and judgement is masked to the interval where the
identified network actually holds (the valid mask). A deployed controller
re-decides every ~20 ms and emits ΔP unconditionally; the reachability gate is
a metric-scope definition, not a deployment filter. ΔP is withdrawn when the
fault clears — not a design choice to be tuned but a physical requirement (see
the hold-policy study in [Results](#results)).

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
│   ├── refine_dp.py                constrained ||ΔP|| minimisation (stage 3, --cuda_graph)
│   ├── profile_latency.py          whole-decision latency profiler (eager vs graph, dt sweeps)
│   ├── verify_true.py              true-dynamics verification (stage 4, --cuda_graph)
│   ├── check_refined.py            refinement provenance check (tensor diff vs original)
│   ├── diagnose_dp.py              head vs oracle vs feasibility ceiling
│   ├── alpha_residual.py           signed severity error by fault type
│   ├── compare_models.py           cross-architecture comparison, bootstrap CIs
│   └── predict9.py                 rolling-window predictor
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
    ├── refine_dp30.py              constrained ||ΔP|| minimisation (stage 3, --cuda_graph)
    ├── profile_latency.py          whole-decision latency profiler (eager vs graph)
    ├── verify_true30.py            true-dynamics verification + reliability diagram (--cuda_graph)
    ├── diagnose_dp30.py            head vs oracle vs feasibility ceiling
    ├── alpha_residual30.py         signed severity error by fault type
    └── compare_models.py           cross-architecture comparison, bootstrap CIs
```

Each folder is self-contained. Run its scripts from inside that folder: the
default paths to the `.mat` support files are relative to the working
directory. `bus_30/verify_physics.py` imports the generic `psys_physics.py`
module and asserts that `PowerSystem.ieee9()` reproduces
`bus_9/ieee9_physics.py` bit-for-bit — the 9-bus-specific module and the
system-agnostic one are kept equivalent by that check, and both carry the
capture-safe batched solver used by `--cuda_graph` training.
`cross_validation_datagen_governor.py` is a confirmed remnant kept only
because the `.mat` exporters import its constants.

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

Non-CUDA machines: two ready-made requirement files ship with the repo —
`requirements-rocm.txt` (x86_64 Windows + AMD gfx1030: Windows ROCm torch
wheels exist from AMD's repo.radeon.com (ROCm 7.2.1) and community TheRock
builds — see that file for exact install commands; needs Python 3.12) and
`requirements-directml.txt` (Windows ARM64 / aarch64 via torch-directml, no
driver stack; also fine for any DX12 GPU on Windows). The package set
(numpy, scipy, h5py, matplotlib, torch) was derived from the script imports —
nothing else is third-party. The device-selection code (`train_*.py`,
`case30_common.py`) is written for CUDA; the ROCm builds surface as 'cuda'
via HIP so most code runs unchanged, while DirectML needs explicit
`torch_directml.device()` amendments.

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
graphify-out/
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
python train_receding.py --data data_mm --out receding_cascade \
  --epochs 500 --batch_size 1024 --instants_per_scenario 4 \
  --w_max 1.0 --dt_window 0.01 --init_window 0.5 --lam_window 0.05 \
  --lstm_layers 3 --ctrl_start 100 \
  --dp_head cascade --n_refine 3 --w_dp_norm 1.0 --margin 1e-3

# 3. is the ΔP head the bottleneck?
python diagnose_dp.py --ckpt receding_cascade/best.pt --data data_mm --n 512 --steps 400

# 4. minimise ||ΔP|| at constant rescue rate
python refine_dp.py --ckpt receding_cascade/best.pt --data data_mm --floor_tol 0
#    add --cuda_graph for a 2-4x faster refinement step (launch-overhead bound;
#    identical schedule and losses, see "Latency and CUDA-graph capture")

# 5. check the emitted ΔP in the true dynamics, and plot
python verify_true.py --ckpt receding_cascade/refined/best.pt --data data_mm --n 2048 --plots_per_type 3

# 6. cross-architecture comparison with bootstrap CIs and the α-bias overlay
python compare_models.py --data data_mm \
  --ckpt "LSTM+LSTM=receding_cascade/refined/best.pt" \
  --ckpt "MLP+MLP=receding_fullmlp/refined/best.pt"
```

(Use `\` on Linux/macOS; `^` under PowerShell/cmd.)

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
python train_case30.py --data data30_mm --out case30_v3 \
  --epochs 60 --batch_size 512 --instants_per_scenario 4 \
  --w_max 1.0 --dt_window 0.01 --init_window 0.5 --lam_window 0 \
  --lstm_layers 5 --ctrl_start 999 \
  --dp_head cascade --n_refine 3 --w_dp_norm 1.0 --margin 1e-3

# 4b. add control on top (weights, feature statistics AND split carried over)
python train_case30.py --data data30_mm --out case30_ctrl --init case30_v3/best.pt \
  --epochs 200 --batch_size 512 --instants_per_scenario 4 \
  --w_max 1.0 --dt_window 0.01 --init_window 0.5 --lam_window 0.05 \
  --lstm_layers 5 --ctrl_start 1 \
  --dp_head cascade --n_refine 3 --w_dp_norm 1.0 --margin 1e-3

# 5. post-training
python diagnose_dp30.py --ckpt case30_ctrl/best.pt --data data30_mm
python refine_dp30.py   --ckpt case30_ctrl/best.pt --data data30_mm --floor_tol 0
#    add --cuda_graph for the graph-captured training step (2.3x measured)
#    note: --warmup 40 (default) holds patience until the multiplier has decayed;
#    on the 9-bus the refinement needed --lam0 1.0 --lam_eta 30 to bind at all
python verify_true30.py --ckpt case30_ctrl/refined/best.pt --data data30_mm --split test

# 6. cross-architecture comparison (own-floor and --rescue_floor common-floor runs)
python compare_models.py --data data30_mm \
  --ckpt "LSTM+LSTM=case30_ctrl/refined_full/best.pt" \
  --ckpt "MLP+MLP=case30_fullmlp_ctrl/refined/best.pt"
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
  first 23 epochs; at the default λ settings the constraint is slack enough
  that nothing binds — the 9-bus runs needed `--lam0 1.0 --lam_eta 30`.
- **`verify_true30.py`** plots each fault type, preferring cases rescued over
  the whole run. That makes them selected examples, not typical ones. Each
  title states which tier the case came from. It also reports the `p_fail`
  reliability diagram and, with `--dp_hold window|run`, the hold-policy study.

---

## Ablation: LSTM vs size-matched MLP

Every architecture is built twice: LSTM encoder + LSTM ΔP head, and an MLP
encoder + MLP ΔP head whose **parameter count matches the LSTM variant**
(`matched_mlp_proj()` solves for the projection width). Any difference in
results therefore cannot be attributed to capacity.

The evaluation (`compare_models.py`) runs identical test cases across
architectures and reports 95% cluster-bootstrap CIs (resampling over unique
scenarios — each metric's effective N is its own scenario count, smaller for
rescue metrics). It also reports the signed α residual (gen+load only; the α
head is masked to active gen/load faults and line faults are binary in the
network build, so line-fault α is neither trained nor used).

**Read the two refinement settings as answering different questions:**

- **Common rescue floor** (`--rescue_floor`): both models held to the same bar —
  the controlled comparison; use for any architecture claim.
- **Each model's own floor** (`--floor_tol 0`, the default): each model held to
  *its own* baseline rescue — the better operating point; use for deployment
  numbers, not comparisons.

The floor distinction is not cosmetic. The first 30-bus comparison (each model
at its own floor) suggested architecture differences in rescue and ΔP effort;
re-refining at a common floor collapsed those differences into noise. When
floor values matter to a conclusion, the conclusion is about the floor, not
the architecture.

---

## Evaluation protocol

Most of the large metric changes during development came from flaws in the
evaluation protocol, not from the model. The protocol used by every script:

| Rule | Why |
|---|---|
| **Trajectory loss is masked to the valid horizon.** It stops at `t_clear` if a fault is active, and at `t_fault` if pre-fault. | The rollout holds the fault; the real system clears it. Supervising past the change penalises an unknowable event. (9-bus traj 0.0221 → 0.00109.) About half of scenarios clear inside the 1.5 s horizon (min 0.082, mean 1.572 s). |
| **OOL is scored against the measured future over the same judged window**, not the whole-run label `y_ool`. | Like-for-like comparison. (9-bus OOL acc 0.853 → 0.966.) The prediction side is the model's free (ΔP = 0) rollout over the same window. |
| **Decision instants are drawn from `t_fault + U(0, 0.2 s)` for a share `p_active`**, the rest uniformly; 4 instants per scenario per epoch. | Matches the 50 ms detection convention of the ΔP labels. |
| **Rescue is scored only on actionable instants**: fault active now, ω still inside the envelope now, non-empty judged window. | Instants after clearing, or already outside the envelope, are failures no controller could avoid. Two of the three gate conditions are label-derived — the gate defines which cases are *scored*, not which receive control; the unfiltered `dp_rescue_all` exists alongside. |
| **ΔP is held until the fault clears** (`--dp_hold clear`). | Physically required: after clearing, a residual set-point offset settles at (ω−1) = R·ΔP with R = 1/20, so any sustained ΔP above 0.100 pu is outside the envelope at steady state (needy-case mean ‖ΔP‖ 0.664 → 6.6× the limit). Verified by hold-policy experiments — see [Results](#results). |
| **Governor grace period of 200 ms**: peaks inside it are not scored. | Tg = 0.2 s; no set-point change acts sooner. |
| **Primary control metrics are rescue rate and mean ‖ΔP‖.** Distance to `y_delta_p` is reported for reference only. | `y_delta_p` is the minimum-norm solution to a different question (fixed 50 ms application, true fault known), judged to the end of the run. |

**Rescue rate** = fraction of actionable decision instants whose judged-window
peak |ω − 1| exceeds 0.005 without ΔP, and stays ≤ 0.005 with it.

Changing the gate changes the denominator. Rescue rates from before and after
a change to this definition are not comparable; always state the definition
alongside a figure.

---

## Results

Single training run per architecture per system (seed 11); the learned-window
result is the only one replicated across both systems. "Model" figures are
measured inside the model's own identified dynamics. "True" figures apply the
emitted ΔP in a simulation of the real faulted system. Test split: 4096
instants, 499 scenarios, 95% cluster-bootstrap CI.

### Physics ports

| | 9-bus | 30-bus |
|---|---|---|
| ω RMSE vs dataset, 100 scenarios (noise floor 5.0·10⁻⁵) | median 5.008·10⁻⁵ | median 5.001·10⁻⁵, max 5.03·10⁻⁵ |
| OOL agreement | 100/100 | 100/100 |
| Reduced admittance matrix vs stored | 1.8·10⁻¹⁵ | 1.5·10⁻¹⁵ |

### 9-bus refined, test split (receding_cascade/refined, `--lam0 1.0 --lam_eta 30`)

| Metric | LSTM+LSTM | MLP+MLP |
|---|---|---|
| Decision accuracy | 0.990 | 0.987 |
| Severity α RMSE (gen+load) | 0.059 | 0.063 |
| α signed bias | −0.002 (n.s.) | +0.014 (over, sig.) |
| OOL TPR | 0.913 | 0.950 |
| Rescue rate, TRUE dynamics | 0.977 | 0.987 |
| Mean ‖ΔP‖ | 0.103 pu | 0.132 pu |
| Learned window | 98 ms | 280 ms |

Refinement is a strict Pareto gain: ‖ΔP‖ falls 26% (LSTM) / 15% (MLP) with
rescue unchanged. Control-locality check: accuracy, α, TPR and window are
identical to the unrefined row — refinement touches only the ΔP head
(`check_refined.py` verifies this tensor-by-tensor). Identification metrics are
architecture-tied here only because refinement held each model at its own
floor; both baselines happened to be identical (0.9978), so the α-bias
comparison is clean: the over-estimating MLP genuinely spent more.

### 30-bus, two refinement settings (test split)

**Common rescue floor** — the controlled comparison (use for architecture claims):

| Metric | LSTM+LSTM | MLP+MLP |
|---|---|---|
| Decision accuracy | 0.990 | 0.990 |
| Severity α RMSE | 0.060 | 0.082 |
| OOL TPR | 0.987 | 0.962 |
| Rescue rate, TRUE dynamics | 0.758 | 0.747 |
| Mean ‖ΔP‖ | 0.115 pu | 0.111 pu |
| Learned window | 128 ms | 277 ms |

**Each model's own floor** — the better operating point (deployment numbers;
not an architecture comparison — each model is held to its own baseline, and
the LSTM's was the higher):

| Metric | LSTM+LSTM | MLP+MLP |
|---|---|---|
| Rescue rate, TRUE dynamics | 0.864 | 0.830 |
| Mean ‖ΔP‖ | 0.136 pu | 0.133 pu |

Identification metrics are bit-identical between the settings (refinement
trains only the ΔP head). Refinement history: the first own-floor LSTM run
crashed mid-way (an unconverged ‖ΔP‖ of 0.151/0.166); the completed run
(best epoch 191) reduced ‖ΔP‖ 0.166 → 0.137 while *raising* rescue — Pareto,
not trade, in both architectures. After refinement the ‖ΔP‖ gap is 0.003
(own floor) / 0.004 (common floor): **there is no control-effort economy
difference between the architectures on the 30-bus**, and the rescue/ΔP
differences of the original own-floor comparison were an artefact of the
unmatched floors.

### What is defensible

- **Identification is architecture-agnostic.** Decision accuracy ≈ 0.99 for
  both architectures on both systems.
- **The learned window is the only replicating architectural effect**:
  the LSTM needs 98–128 ms of history, the MLP 277–280 ms — a 2–3× difference
  in the same direction on two independent systems, with a mechanistic
  explanation (recurrence compresses history into its state).
- **The α-bias mechanism is demonstrated on the 9-bus**: the over-estimating
  MLP spent more ΔP for more rescue, with no confound. On the 30-bus both
  models under-estimate severity (−0.009 / −0.023), but the control-loop
  consequences are inside the noise. Within-run z-scores establish each bias;
  direction survival across reseeds is untested.
- **Generator-trip failure on the 30-bus is a feasibility limit, not a head
  failure** at the severe end, but a genuine head failure at the mild end
  (see below).

### Generator trips on the 30-bus (cross-tabulated, own-floor checkpoint)

Of the missed generator trips: severe ones (α 0.70–0.90, 49 cases, rescue
0.571) fail at the **actuator rail** — 78.6% had ΔP saturated at ±dp_max, so
no set-point change within the limits could have saved them (9 of 42 left
authority unused). Below α 0.70, by contrast, 10 misses (7 in α 0.30–0.50,
3 in 0.50–0.70) show **no rail saturation** — a genuine head failure on mild
faults; that is the specific residual defect. Control effort is
non-monotonic in α (peaks at α 0.50–0.70), not inverted at the top.

### p_fail calibration (reliability diagram, own-floor checkpoint)

`p_fail` — the bounded surrogate the control loss minimises — tracks its event
to within 0.010 (ECE 0.028 vs its own rollout event): **the surrogate is
calibrated; `--dp_tau` needs no change.** The 4.7-point gap between model-
internal (0.884) and true rescue (0.837) lives in the rollout, concentrated
exactly on the needy cases (model 0.116 vs true 0.163). The safety margin
fails in both directions at once: near-certain-failure cases actually fail
rarely (over-protected by ×2–4) while 6.4% of confident-safe needy cases fail
(under-protected ×6). A uniform margin trades one failure mode for the other; a
severity-scaled margin is untested future work.

### ΔP hold policy (withdrawal at clearing is physically required)

| `--dp_hold` | end-of-run rescue |
|---|---|
| `clear` (withdrawn at clearing) | **0.668** |
| `window` | 0.456 |
| `run` (held to the end) | **0.056** |

Holding longer is catastrophically worse, and droop explains it: after the
fault clears the network is healthy, so a residual set-point offset settles at
(ω−1) = R·ΔP (R = 1/20) — any sustained ΔP above 0.100 pu is outside the
envelope at steady state, by construction. This validates the
`HOLD_MODE = 'until_clear'` choice in `compute_delta_p.m`. The residual gap
between in-window (0.837) and end-of-run (0.668) rescue — about a fifth of
saved cases drift out during ring-down — is the remaining open question; the
leading candidate is the withdrawal transient (a ramped withdrawal would test
it, as a new control policy). Without ΔP the end-of-run rate is 0.000.

### Head vs ceiling (30-bus, `diagnose_dp30.py`, 166 needy cases)

| | Rescue | Mean ‖ΔP‖ |
|---|---|---|
| Trained head | 0.9217 | 0.600 pu |
| Oracle, same objective | 0.8916 | 0.517 pu |
| Feasible, no effort penalty | 0.9398 | 0.728 pu |

---

## Latency and CUDA-graph capture

The decision latency matters for a deployed controller: the unrolled 150-step
RK4 rollout plus the 3-round cascade ΔP head submit thousands of tiny
kernels, so the *eager* step is launch-bound — batch size barely matters
(447.6 ms at batch 1, 464.9 ms at batch 256 on the 9-bus).

Two independent fixes ship in the repo:

1. **Whole-decision CUDA-graph capture** (`profile_latency.py --graph`, and
   `--cuda_graph` on `refine_dp*.py` / `verify_true*.py`): replay of the whole
   decision removes launch overhead entirely on ROCm/HIP.
   - At the baseline spacing (dt_roll = 10 ms): eager 126.6 ms → graph replay
     **20.0 ms** (7×) at batch 256 on the 9-bus; 126.6 → 96.8 ms with
     `n_refine 2`; 64.3 ms with `n_roll 75`; 49.1 ms with both.
   - The refine *training* step is graph-captured too: the 9-bus refines at
     ~17 s/epoch vs ~57 s eager (3.4×), the 30-bus at 2.3× on a 3-epoch A/B.
   - Same-seed graph reruns are bit-identical; vs eager, rescue differs ≤0.7 pt
     and ‖ΔP‖ ≤2% (the capture-safe batched solver — plain `linalg.solve`
     cannot run inside an ROCm capture — introduces ~1.6e-5 ΔP noise that
     flips borderline rescue cases at float32). **Reported paper numbers come
     from the eager exact-solve paths**, which are untouched.
2. **Sample-spacing lever** (`--dt_roll`): rescue is invariant across dt_roll
   5–210 ms at fixed horizon (checked to four decimals); the RK4 stability
   edge sits at 213 ms and the operating point at dt = 100 ms, where the
   whole-decision graph replay is **20.0 ms** with rescue 0.9003 — the
   spacing + graph levers compound near-multiplicatively.
   `verify_true*.py --dt_roll_override` re-scores any checkpoint at another
   spacing (30-bus dt=100 ms test split: ω RMSE +1.6%).

Capture on ROCm/Windows needs the pitfalls the scripts already handle
(large-stack capture thread, capture-safe solver, λ via a device tensor,
`os._exit` shutdown); see the module docstrings of `refine_dp.py` and
`case30_physics.py` for the full list. `train_*.py` are NOT graph-captured
(the `--ctrl_start` phase flip changes the backward graph mid-run); port
pending unless a retrain happens.

---

## Known issues and limitations

**Evaluation scope**
- **Single-decision evaluation.** Every figure scores one decision with ΔP held
  until the fault clears. The receding-horizon controller the design assumes
  (re-deciding every ~20 ms) has not been replayed in closed loop. This is the
  main reason the 30-bus end-of-run rescue (0.668) is below the in-window
  figure (0.837): faults last up to 3 s, beyond the 1.5 s horizon.
- **Single seed per configuration.** Run-to-run variance was measured once and
  is large (two near-identical 30-bus runs differed in TPR 0.979 vs 0.937):
  gaps ≲0.04 cannot be attributed to architecture from one seed. The
  window result does not depend on this.
- **No independent Simulink check of the final ΔP.** Verification uses the
  Python port, which reproduces the Simulink data at the noise floor.
- **Validation vs test.** `verify_true30.py` defaults to the validation split;
  use `--split test` for reported numbers.
- **The hold-policy `window`/`run` rows were measured on an interrupted
  checkpoint** and not re-run on the completed one; the ordering (and the
  droop arithmetic) are far outside any plausible refinement effect, but
  re-measure before quoting them numerically.

**30-bus**
- **Generator trips below α 0.70 are the residual defect:** 10 misses with no
  rail saturation — the head fails on mild faults. At the severe end the same
  cell is actuator-limited (see cross-tab above): more capacity would not help;
  more actuated machines or a higher `dp_max` would.
- **α saturates at the top of its range, in the unsafe direction.** Every
  configuration under-estimates severity in the α 0.90–1.01 bin (9-bus LSTM
  −0.037, MLP −0.038; 30-bus LSTM −0.056, MLP −0.100, 96% of cases under).
  `α = 0.1 + 0.9·sigmoid(raw)` reaches 1.0 only asymptotically.
  `--alpha_hi 1.1` (a supported training flag) would put α = 1.0 in the
  sigmoid's responsive band; untested (requires retraining).
- **The safety margin roughly equals the model's own error.** The margin is
  1·10⁻³ while ω RMSE on the test set is 9.5·10⁻⁴, and the model's own rollout
  is ~5 points optimistic against true dynamics on needy cases. Refinement
  spends that margin; re-verify refined results in true dynamics.
- **The learned window did not fully converge** on the 30-bus. The mask edge
  (β = 26.9 ms) was not fully annealed at the selected epoch. Treat 128 ms as
  provisional (plateau phrasing: "LSTM windows settle near 100–150 ms, MLP
  near 280 ms").

**Both systems**
- **Split.** The training scripts regenerate a 90/5/5 split with a fixed seed
  and store it in the checkpoint. They ignore the 80/10/10 `split` shipped in
  the datasets. Reason: more training data, with 500 scenarios judged
  sufficient for val/test — a project decision, not a defect in the stored
  split.
- **Sampling.** Both models read every 10th raw sample (10 ms). On the 9-bus
  this happens in two steps: the converter keeps every 5th sample, and the
  window reads every 2nd. Averaging instead of decimating (`--agg mean`) would
  cut noise ~3× and is an open ablation.
- **Line faults that peak after reconnection** (the `dp_status 3` line cases)
  cannot be corrected by a constant set-point change withdrawn at clearing.
  This is a limit of the actuator, not of the model.
- **No fault-time head.** Fault time is not predicted; the decision is
  anchored to the measurement window only.
- **Line-fault α is neither trained nor used** (the α loss is masked to
  gen/load and line faults are binary in the network build). Any α statistic
  pooling line faults measures a free parameter; the reported α RMSE already
  masks to gen+load.

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