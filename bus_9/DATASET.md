# IEEE 9-bus fault dataset for the PINN-LSTM

Simulated transient-stability dataset generated from the Simulink model
`ieee9_governor`, which reproduces the Python DAE in
`cross_validation_datagen_governor.py`. Each sample is one 8-second fault
scenario. The network is trained to detect the fault, decide whether the
frequency will leave its limits, and output the corrective governor action
that prevents it.

Files:

| File | Content |
|---|---|
| `pinn_dataset.mat` | the dataset (inputs, labels, metadata) - HDF5 / MATLAB v7.3 |
| `delta_p_labels.mat` | the ΔP labels only (same values, kept separately as a backup) |
| `ieee9_gov_simulink_data.mat` | machine constants, initial state, Kron-reduced network |
| `ieee9_base.mat` | base admittance matrix, generator/load buses |

---

## 1. Power system

IEEE 9-bus, 3 classical generators, 60 Hz, 100 MVA base, all values in per unit.

- **Generator buses:** 1, 2, 3 (generator index 1, 2, 3)
- **Load buses:** 5, 6, 8
- **Transmission lines** (between non-generator buses): 4-5, 4-6, 5-7, 6-9, 7-8, 8-9
- **Transformer branches** 1-4, 2-7, 3-9 are never faulted (opening one would island a machine)

Each generator has 3 states, so 9 differential states in total:

| State | Equation |
|---|---|
| rotor angle δ | dδ/dt = 2π·60·(ω − 1) |
| rotor speed ω | dω/dt = (Pm − Pe − D(ω − 1)) / 2H |
| mechanical power Pm | dPm/dt = (Pm_set − (ω − 1)/R − Pm) / Tg |

with droop R = 0.05 and governor time constant Tg = 0.2 s. H, D and Xd′ per
generator are in `ieee9_gov_simulink_data.mat`. Loads are constant impedances
folded into the admittance matrix, so the network is linear and Pe follows
algebraically from the three angles:

```
E    = (Ed + j·Eq) · exp(j(δ − π/2))      internal EMF
I    = Yred · E                            generator currents
Pe   = Re(E · conj(I))
Vbus = Vrec · E                            all nine bus voltages
```

`Yred` (3×3) and `Vrec` (9×3) are the Kron-reduced network matrices; the
pre-fault pair is stored in `ieee9_gov_simulink_data.mat`, and the faulted pair
for any scenario can be rebuilt from `ieee9_base.mat` (see
`scenario_P` in `compute_delta_p.m`).

---

## 2. Scenario generation

Each of the 10,000 scenarios draws its parameters from a uniform distribution.
The fault type is drawn with equal probability.

| Parameter | Range |
|---|---|
| fault type | `gen`, `load`, `line` (≈1/3 each) |
| fault time | 0.5 - 4.0 s |
| fault duration | 0.05 - 3.0 s (clearing time = fault time + duration, ≤ 7 s) |
| α, `gen` | 0.10 - 1.00 |
| α, `load` | 0.10 - 1.00 |
| α, `line` | always 1 |
| simulation length | 8 s, solver step 0.001 s (`ode14x`) |

**`gen` - generator trip.** The set-point of one generator drops by α·Pm0, so
α = 1 takes that unit to zero output. Restored at clearing. Pm falls, so ω
falls.

**`load` - load fault.** Extra shunt admittance at a load bus:
`Y(bus,bus) += α`. Removed at clearing. Demand rises, so ω falls.

**`line` - transmission line fault.** The line is disconnected for the duration
of the fault, then reconnected. The post-fault network is weaker, so power
reroutes and the machines swing against each other.

---

## 3. Contents of `pinn_dataset.mat`

N = 10000 scenarios, T = 8001 time steps (0 to 8 s, every 0.001 s; the solver ran
at 0.001 s and every step was stored).

### Network input

| Variable | Shape | Description |
|---|---|---|
| `X` | N × T × 33 (single) | the full simulated measurement, with Gaussian noise |
| `t` | T × 1 | time vector, 0 : 0.01 : 8 |

The 33 channels, in order (also in `state_names`):

| Index | Channel |
|---|---|
| 1, 3, 5 | δ1, δ2, δ3 (rad, unwrapped) |
| 2, 4, 6 | ω1, ω2, ω3 (pu) |
| 7 - 9 | Pm1, Pm2, Pm3 (pu) |
| 10, 12, 14 | Id1, Id2, Id3 (pu) |
| 11, 13, 15 | Iq1, Iq2, Iq3 (pu) |
| 16, 18, ... 32 | Vm at buses 1 - 9 (pu) |
| 17, 19, ... 33 | θ at buses 1 - 9 (rad, unwrapped) |

This is the same ordering as the Python DAE state vector, so the physics
equations can be applied to `X` directly.

**Measurement noise** (added to the clean simulation, per sample, independent
per channel; std in `noise_std`):

| Channel | Std |
|---|---|
| δ | 0.01 rad |
| ω | 5·10⁻⁵ pu |
| Pm | 0.005 pu |
| Id, Iq | 0.005 pu |
| Vm | 0.001 pu |
| θ | 0.01 rad |

Note that ω is the key signal and its swings (about 0.002 - 0.02) are only a
few hundred times its noise, whereas δ moves by order 1. Normalise every
channel using training-set statistics.

### Labels

| Variable | Shape | Description |
|---|---|---|
| `y_type` | N × 1 (int8) | 1 = `gen`, 2 = `load`, 3 = `line` (`type_names`) |
| `y_t_fault` | N × 1 | fault time [s] |
| `y_ool` | N × 1 (logical) | true if \|ω − 1\| > 0.005 for any generator at any time |
| `y_delta_p` | N × 3 | corrective set-point change per generator [pu] |
| `dp_status` | N × 1 (int8) | how `y_delta_p` was obtained (see below) |

The out-of-limit flag uses the **clean** ω, so noise cannot flip a label.
Roughly 45 - 50% of scenarios are out of limit; per type it is about 49%
(`gen`), 50% (`load`) and 40% (`line`).

### Metadata

| Variable | Description |
|---|---|
| `meta_alpha` | severity α |
| `meta_loc` | `gen`: generator index; `load`: bus number; `line`: row of `lines` |
| `lines` | 6 × 2, [from_bus to_bus] of each faultable line |
| `meta_t_clear` | clearing time [s] |
| `meta_maxdev` | peak \|ω − 1\| over the run, without correction |
| `t_apply`, `t_remove` | when ΔP is applied / withdrawn [s] |
| `base_maxdev_window` | peak \|ω − 1\| in the judged window, without ΔP |
| `best_maxdev_window` | the same with the chosen ΔP |
| `verify_maxdev` | peak from the Simulink re-simulation (NaN if not checked) |
| `split` | 1 = train (80%), 2 = validation (10%), 3 = test (10%) |
| `failed`, `omega_limit`, `dt_solver`, `noise_std`, `dp_settings` | run settings and flags |

---

## 4. The ΔP label

ΔP is a constant change added to the three governor set-points:
`Pm_set = Pm0 + ΔP`. It is the smallest such change (minimum Euclidean norm)
that keeps \|ω − 1\| within the limit. Timing:

| Moment | |
|---|---|
| fault occurs | `y_t_fault` |
| ΔP applied | `t_apply` = fault time + **0.05 s** detection delay |
| stability judged from | `t_apply` + **0.2 s** (governor lag Tg; nothing can act sooner) |
| ΔP withdrawn | at clearing (`t_remove` = `meta_t_clear`) |

ΔP is withdrawn at clearing because the fault itself disappears then; holding it
longer would turn the correction into a surplus and push ω the other way.
Each ΔP is limited to ±1 pu per generator and targets 0.0049 rather than
0.005, leaving a small margin.

`dp_status` values:

| Value | Meaning | `y_delta_p` |
|---|---|---|
| 0 | not out of limit, no action needed | zeros |
| 1 | corrected: this ΔP keeps ω inside the limit | the solution |
| 2 | out of limit only before ΔP could act, fine afterwards | zeros |
| 3 | no ΔP within ±1 pu fixes it | best effort (closest achievable) |

Status 3 is mostly line faults. Their peak arrives after the line reconnects,
when ΔP has already been withdrawn, and it is mainly the machines swinging
against each other rather than a system-wide frequency drift; a constant
set-point shift cannot remove that.

---

## 5. What the model is expected to do

**Input.** A window of the 33 measured channels: `X[k, t0:t1, :]`, i.e. the
measured state of the system up to the moment of the decision. The window is
trainable (learned length / attention over the sequence), but see the leakage
note below.

**Outputs.**

| Head | Type | Target | Loss |
|---|---|---|---|
| fault type | 3-class classification | `y_type` | cross-entropy |
| fault time | regression, seconds | `y_t_fault` | MSE / Huber |
| out of limit | binary classification | `y_ool` | binary cross-entropy |
| ΔP | 3-dim regression, pu | `y_delta_p` | MSE, masked (see below) |

In production the model watches the measurements; once it has detected a fault
and predicts that the frequency will leave its limits, it emits a non-zero ΔP,
which the governors apply until the fault clears.

### Notes for training

1. **Causality / leakage.** `X` is the simulation **without** correction. Once
   ΔP is applied in reality the trajectory changes, so anything after
   `t_apply` in `X` is not what the controller would actually see. Predict ΔP
   from data up to `t_apply` (fault time + 0.05 s) only. The out-of-limit flag
   and ΔP both refer to the future, which is the point: they are predicted, not
   measured.
2. **Which samples carry a ΔP target.** Train the ΔP head on `dp_status == 1`.
   Statuses 0 and 2 are genuine zeros and can be included as such. Status 3 is
   a best-effort value, not a solution: exclude it, or learn it as a separate
   "not correctable" class.
3. **Physics loss.** The equations in section 1 apply directly to the channels
   of `X`. The residuals of the three state equations can be formed for any
   window; Pe needs the Kron-reduced matrices, or it can be taken from the
   measured Id, Iq as `Pe = Ed·Id + Eq·Iq`.
4. **Split.** Use `split` so that the same scenario never appears in two sets.
5. **Scaling.** Per-channel normalisation from the training set; ω benefits
   from being centred on 1 and scaled by about 0.01.

---

## 6. Loading in Python

MATLAB v7.3 files are HDF5, and h5py returns the dimensions reversed.

```python
import h5py, numpy as np

with h5py.File("pinn_dataset.mat", "r") as f:
    X       = np.array(f["X"]).transpose(2, 1, 0)        # (10000, 801, 33)
    t       = np.array(f["t"]).ravel()                    # (801,)
    y_type  = np.array(f["y_type"]).ravel().astype(int) - 1   # 0, 1, 2
    y_tf    = np.array(f["y_t_fault"]).ravel()
    y_ool   = np.array(f["y_ool"]).ravel().astype(bool)
    y_dp    = np.array(f["y_delta_p"]).T                  # (10000, 3)
    status  = np.array(f["dp_status"]).ravel().astype(int)
    t_apply = np.array(f["t_apply"]).ravel()
    split   = np.array(f["split"]).ravel().astype(int)

# window that the controller could actually have seen
k = 0
n_end = int(round(t_apply[k] / (t[1] - t[0]))) + 1
window = X[k, :n_end, :]
```

---

## 7. Scripts that produced this

| Script | Role |
|---|---|
| `export_for_simulink.py` | initial conditions and Kron-reduced network from `ieee9_parameters.json` |
| `build_ieee9_gov_model.m` | builds `ieee9_governor.slx` |
| `generate_dataset.m` | runs the 10,000 randomised scenarios, writes `X` and the labels |
| `compute_delta_p.m` | finds ΔP for every out-of-limit scenario, adds the ΔP labels |
| `plot_delta_p_examples.m` | plots random scenarios with and without ΔP |
| `compare_with_python.m` | checks the Simulink model against the original Python DAE |

To regenerate with different settings, edit the SETTINGS block at the top of
`generate_dataset.m` (ranges, noise, sampling) or `compute_delta_p.m`
(detection delay, hold mode, ΔP limits), then rerun that script.
