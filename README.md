# PINN-Based Corrective Control for Power System Transient Stability

Finds the **minimum generator power-setpoint adjustment (ΔPm)** needed to restore
frequency stability after a fault on the IEEE 9-bus test system, with results
**directly verified against the system's own ground-truth DAE simulator** 

## Objective

Given a fault (generator trip, load fault, or transmission line fault), find the
smallest possible ΔPm, a corrective adjustment to each generator's governor
setpoint such that:

```
max|ω_i − 1| < 0.005 pu   for all generators i, in the window (t_clear, t_clear + 2s)
```

## Two pipelines

This repo contains two generations of the method:

| | **Approximate pipeline** | **Ground-truth pipeline** |
|---|---|---|
| Handoff state | Predicted by a PINN | Computed directly by the real DAE |
| ΔPm search | Gradient descent through our own hand-built rollout | Gradient descent through the real DAE's own solver, OR the approximate rollout + real-DAE verification |
| Trustworthy? | Not on its own (see below) | Yes — every reported result is checked against, or found directly by, the real solver |
| Speed | Fast (deployable in real time) | Slower (used for verification / offline analysis) |

**Why both exist:** the approximate pipeline is fast enough for real-time use,
but a verification exercise found it could be meaningfully wrong by up to ~190%
error in some cases specifically for corrections that land close to the
stability threshold. Rather than trusting the approximate pipeline blindly, the
ground-truth pipeline checks (or in some cases fully replaces) its answers
against the real solver before anything is reported as a success.

**Chosen production approach:** PINN finds a candidate correction (fast), the
real DAE verifies it and retries with scaled corrections if needed (`find_and_verify.py`).
Every result reported from this pipeline is a real, ground-truth-confirmed number.

## Results (ground-truth verified)

Across the full scenario grid (330 gen-trip/load-fault scenarios + 190
transmission-line scenarios = 520 total), verified directly against the real DAE:

| Fault type | Verified success rate |
|---|---|
| Generator trip | 128/150 (85.3%) |
| Load fault | 156/180 (86.7%) |
| Transmission line fault | 183/190 (96.3%) |
| **Total** | **467/520 (89.8%)** |

Failures concentrate almost entirely on **Generator 1** (the largest generator
in the system rated as 1.63 pu (gen 1) vs. 0.72 (gen 0) and 0.85 pu (gen 2) for the other two) at high
severity, and on the most severe/longest-duration


## Repository structure

```
systems/                Core system definition (IEEE 9-bus DAE solver + parameters) —
                        cross_validation_datagen_governor.py, ieee9_parameters.json
data_generation/       Scripts that generate the training data (DAE simulations
                        across the full fault/severity/clearing-time grid)
training/               PINN training script (one model per fault type)
approximate_pipeline/   The fast, PINN-based ΔPm optimizer (not independently
                        trustworthy near the stability boundary — see below)
ground_truth_pipeline/  Real-DAE verification and fully ground-truth optimization
```

All scripts import from `systems/` — e.g.
`from cross_validation_datagen_governor import ...` — so `systems/` should be
on the Python path (or its contents placed alongside the script being run) for
anything here to import cleanly.

### `data_generation/`

| File | Purpose |
|---|---|
| `generate_data_v9.py` | Generates gen-trip and load-fault training data (150 + 180 scenarios). |
| `generate_data_line_only.py` | Generates transmission-line and transformer-fault data. Includes a fix for an off-by-timing bug in the original clearing-time grid (a value shorter than the fault's own onset time caused every line to fail identically during generation). |
| `merge_line_data.py` | Combines the line/transformer data with the existing gen/load data into one dataset. |
| `split_line_and_transformer.py` | Separates transmission-line and transformer scenarios into independent training categories. (Training them together corrupted the shared loss — transformer faults are 5–10x more severe than transmission-line faults, since a transformer failing severs a generator's *only* grid connection, while a transmission line failing just removes one of several parallel paths.) |

### `training/`

| File | Purpose |
|---|---|
| `pinn_multifault2.py` | Trains one PINN per fault type (load, gen, transmission-line, transformer) to predict the pre-clearing trajectory / handoff state. |

### `approximate_pipeline/`

| File | Purpose |
|---|---|
| `delta_pm_solver.py` | The core ΔPm optimizer: loads a trained PINN, builds a differentiable rollout of the post-clearing dynamics (matching the real DAE's equations), and optimizes ΔPm via gradient descent. Supports both a single constant correction and a time-varying correction (one value per 0.05s timestep — strictly more expressive, and found to resolve substantially more scenarios). |
| `batch_runner_full.py` | Runs the full 330-scenario gen-trip/load-fault grid through the approximate pipeline. |
| `batch_runner_lines.py` | Runs the full 190-scenario transmission-line grid through the approximate pipeline. |

**Important:** results from this pipeline alone are *predictions*, not
verified facts. Use the ground-truth pipeline below to confirm any result
before trusting it.

### `ground_truth_pipeline/`

| File | Purpose |
|---|---|
| `verify_against_real_dae.py` | Runs the actual fault (gen, load, or line) through the real DAE directly — no PINN — to get the true handoff state, and applies a candidate ΔPm sequence to the real solver to see what actually happens. |
| `differentiable_real_dae.py` | Makes ΔPm differentiable **through the real DAE's own solver** (not an approximation of it), via implicit differentiation applied to the solver's actual residual equations. Gradient correctness verified against finite differences (agreement to 8+ significant figures) before being trusted for optimization. |
| `find_and_verify.py` | **The chosen production pipeline.** Finds a candidate ΔPm with the fast approximate optimizer, then verifies it against the real DAE — scaling the correction up or down and re-checking (cheaply, without re-optimizing) if the first attempt doesn't hold up. Only reports success once the real DAE confirms it. |
| `find_minimum_delta_pm_real.py` | A fully ground-truth optimizer — ΔPm is found by gradient descent directly against the real DAE's own equations, with no PINN and no approximation anywhere in the loop. Slower (each iteration is a real DAE evaluation), but requires no verification step, since the result is already a ground-truth number. |
| `batch_full_find_and_verify.py` | Runs the full 330-scenario gen-trip/load-fault grid through the production (`find_and_verify`) pipeline. |
| `batch_lines_find_and_verify.py` | Runs the full 190-scenario transmission-line grid through the production pipeline. |
| `verify_batch_gen_v3.py` / `verify_batch_load_v3.py` / `verify_batch_line_v3.py` | Sample batches (~10 scenarios each) using the fully ground-truth optimizer (`find_minimum_delta_pm_real.py`) instead of the PINN-based production pipeline — demonstrates the fully-differentiable-through-the-real-solver approach on a smaller scale. |

## Methodology

### System and fault model

Classical IEEE 9-bus, 3-generator test case. Each generator: a second-order
swing equation plus a first-order droop governor (3 differential states per
generator — δ, ω, Pm), coupled to the network through the stator algebraic
equations and the network admittance matrix (Y-bus).

Three fault types, all sharing a common structure: occuring at `t_fault = 0.2s`,
held until `t_clear`, then **reverting to the original system** (confirmed both
by direct inspection of the DAE solver's code and empirically: post-fault
frequency settles back to almost exactly 1.0 pu, the signature of the original
equilibrium being restored):

- **Generator trip:** governor setpoint shifted by `fault_mag = −α·Pm0[gen_idx]`, α ∈ [0.1, 1.0]
- **Load fault:** shunt admittance of magnitude α added to `Y_admittance[bus, bus]`, α ∈ [0.1, 1.5]
- **Line / transformer fault:** the branch's off-diagonal admittance is removed
  and redistributed to both endpoints' diagonal entries — no severity
  parameter (a branch is either connected or not)

### Governing equations

Per generator *i*:

```
dδ_i/dt = (ω_i − 1) · 2πf₀
dω_i/dt = (Pm_i − Pe_i − D_i(ω_i − 1)) / (2H_i)
dPm_i/dt = (1/Tg) · (Pm0_i − (ω_i − 1)/R − Pm_i)
```

where `Pe_i = Ed'_i·Id_i + Eq'_i·Iq_i`, with Id, Iq obtained by solving the
stator equations jointly with the network's KCL equations.

### Numerical integration

Implicit trapezoidal rule, solved at each step via Newton-Raphson:

```
F(x₀, x₁) = x₁,diff − x₀,diff − (Δt/2)·(f(x₀) + f(x₁)) = 0
x₁ ← x₁ − J⁻¹F(x₀, x₁),   J = ∂F/∂x₁
```

### Differentiating through the solve

Backpropagating through the Newton iterations directly would be expensive and
unstable. Instead, gradients are obtained via the implicit function theorem
(adjoint method), applied at the converged solution only:

```
λ = −(∂F/∂x₁)⁻ᵀ · (∂L/∂x₁)
∂L/∂ΔPm = λᵀ · (∂F/∂ΔPm)
∂L/∂x₀  = λᵀ · (∂F/∂x₀)
```

This same mathematics is used twice in this repo: once around a hand-built
approximation of the DAE (`approximate_pipeline/delta_pm_solver.py`), and once
around the real DAE's own residual function directly
(`ground_truth_pipeline/differentiable_real_dae.py`) — the latter validated
against finite differences before being trusted.

### Stability loss (hard-barrier design)

```
violation     = ReLU(max_dev − threshold)
is_violating  = 1 if violation > 0 else 0
loss          = is_violating · (1.0 + violation) + (1 − is_violating) · λ · Σ(ΔPm²)
```

This guarantees any genuinely stabilizing solution scores strictly better than
any non-stabilizing one, regardless of size — an earlier, simpler loss allowed
a small non-stabilizing correction to sometimes outscore a larger, genuinely
stabilizing one.

### Why the ground-truth pipeline exists

The approximate rollout was verified to match the real DAE's trajectory to
five decimal places for the **uncorrected** case. But once a real, optimized
correction is applied — especially one that lands close to the 0.005 stability
boundary, which the minimal-correction objective naturally drives it toward —
the two could disagree substantially (up to ~190% in one case). Several fixes
were tried (a fixed safety margin, limiting optimization thoroughness) and each
only partially closed the gap; the optimizer would simply re-converge onto
whatever new boundary it was given, just as precisely. The gap was closed
completely only by removing the approximation itself — either by verifying
every candidate against the real solver, or by optimizing against the real
solver directly.

## Requirements

```
torch
numpy
```

## Typical usage

```bash
# 1. Generate training data (systems/ files must be importable — e.g. run
#    from the repo root with systems/ added to PYTHONPATH, or copy its
#    contents alongside these scripts)
python data_generation/generate_data_v9.py --params systems/ieee9_parameters.json
python data_generation/generate_data_line_only.py --params systems/ieee9_parameters.json
python data_generation/merge_line_data.py --old_dir <gen_load_data> --line_dir <line_data> --out_dir training_data_merged
python data_generation/split_line_and_transformer.py --data_dir training_data_merged --out_dir training_data_split

# 2. Train PINNs
python training/pinn_multifault2.py --data training_data_split --fault_type 0   # load
python training/pinn_multifault2.py --data training_data_split --fault_type 1   # gen
python training/pinn_multifault2.py --data training_data_split --fault_type 2   # transmission line

# 3. Run the ground-truth-verified production pipeline
python ground_truth_pipeline/batch_full_find_and_verify.py
python ground_truth_pipeline/batch_lines_find_and_verify.py
```

Each batch script saves progress incrementally and is resumable — safe to
interrupt and continue across multiple sessions.
