# Era-2 datagen notes — 30-bus (plant v2-nogov, actuator hash 69af36e693dd)

## 1. Plant checks (2026-11-21)
- **No-fault equilibrium** in both Simulink (`case30_pidv2.slx`) and python (`plant_v2.py`): max|ω−1| = 0, |Pm−Pe| ≤ 4e-15, integrator stays at Pm0.
- **Simulink vs python**, max|Δω|:
  - gen trip: 7e-16
  - load: 2.4e-6 and 2.5e-6
  - line: 9.9e-6

  The residue comes from network switching inside the solver step (Simulink switches per minor step; python uses the step midpoint). It is below the 5e-5 ω noise floor.
- Open-loop v2 plant (all gains 0): min ζ ≈ −0.05 to −0.08 in every network. The governor-less classical model with transfer conductances is slightly unstable, so the PID is required for stability, not only for rescue.

## 2. Pole placement (Q9 stage 1) and the generalist (Q7)
- The linearised loop reduces exactly: the integral of the speed error equals angle/ωb, so Ki acts as extra synchronising torque. A = [[0, ωb I], [−M⁻¹(K + Ki/ωb), −M⁻¹(D + Kp)]] with M = 2H + Kd.
- Target region: all poles with Re ≤ −0.5 s⁻¹ and ζ ≥ 0.3, reached with minimum gains, robust over every linearisation in the cell.
  - ζ ≥ 0.7 needs Kp ≈ 150–220. On 600 scenarios that design gave the same in-envelope rate (0.885 vs 0.883), so the cheaper design was kept.
  - Gen cell: the tripped machine's own mode cannot be placed by the other machines' loops (ζ stays near 0.04). That is a property of the plant, and stage 2 handles it.
- **Generalist** (frozen, pure pole placement): Kp 50/62/72/77/81/44, Ki 2.2/32/44/45/24/40, Kd ≈ 0 except G5 2.5 (G1…G6). Full values are in `seeds_v2.json`.
- **Generalist acceptance** (`baseline_generalist.json`, 1200 era-1-mix scenarios, seed 7): in-envelope **0.850** overall.
  - By type: gen 0.588, load 1.000, line 0.979.
  - Settled below 1e-3 at 8 s: 0.977.
  - The threshold of 0.85 was set before this run, and the result lands exactly on it. Borderline: flag for confirmation.

## 3. Line pre-screen (§3 plan; `line_screen.json`)
- 33 lines × 60 durations (0.05–3.0 s) under the generalist: **64 of 1980 cells are needy, from 2 of 33 lines**.
  - **Line 1 (bus 1–2): needy at every duration ≥ 0.10 s.** Short faults peak at 0.008–0.011 and recover. A 3 s fault loses synchronism (relative angle ~200 rad, peak 0.13).
  - **Line 5:** needy only for some durations ≥ 0.25 s (peak ≤ 0.0057).
  - The other 31 lines are never needy.
- Consequence: the ≥30 needy-line-instant quota is reachable only by oversampling lines 1 and 5. Class support for the encoder is 2 line locations.

## 4. Effort-weight sweep (Q6; `lam_sweep_v2.json`, 20 anchors, CMA-ES pop 16 × 60 generations)
- Effort term: E = TV(Pm) + mean(θ²), where θ is the log-deviation of the gains from the seed. The first sweep used TV only (`lam_sweep_tvonly.json`); without a gain term the gains drifted up to ~200× along flat directions of the objective.
- The settling time t_s uses a band of 1e-3 pu. This is needed because the spec's "last exit of the envelope" is 0 by construction once the run stays inside the envelope.

| λ | feasible /20 | median TV | gain-mag median | median max multiplier | edge riders (margin < 5 %) |
|---|---|---|---|---|---|
| 0 | 15 | 4.34 | 2.09 | 24.2× | 0 |
| **0.03** | **15** | **2.48** | **0.35** | **4.9×** | **0** |
| 0.1 | 15 | 2.35 | 0.24 | 4.0× | 0 |
| 0.3 | 15 | 2.34 | 0.22 | 3.6× | 0 |
| 1.0 | 15 | 2.30 | 0.19 | 3.5× | 2 |

- Seed (untuned): 14/20 feasible.
- **λ = 0 chatters** on the hard anchors (TV 23 / 46 / 79 pu on the line-1 0.15 s, gen-1 α 0.6 and gen-1 α 0.95 anchors). At λ = 0.03 those fall to 5 / 7.5 / 16.
- λ = 0.3 starts costing t_s (line-1 0.6 s anchor: 1.74 → 4.59 s). λ = 1 adds edge riders and loses peak margin.
- **Proposal: λ = 0.03, with the settling band 1e-3 pu.** This is the smallest λ at which the chattering and gain drift disappear without a material t_s cost. Not pinned yet: per Q6 it goes into the spec (dated) before the 300-anchor run.

## 5. Generalist ceiling tune (Q7 rev 2; `ceiling_tune.py` → `baseline_ceiling.json`)
- **Protocol.** One constant gain set, started from the pole-placement generalist. CMA-ES (seed 20261122, σ0 0.5, population 16, 80 generations, no early stopping).
  - Objective: maximise the worst-class in-envelope rate over the whole run, with small tie-breaks on exceedance and TV that can never outweigh one rescue.
  - Tuning sample: 600 scenarios, 200 per class (seeds 101–103), separate from the seed-7 acceptance sample.
- **Result: plateau.** The worst class (gen) stayed at 0.63 on the tuning sample for all 80 generations (load 1.00, line 0.95).
- **Acceptance sample** (1,200 scenarios, seed 7, `baseline_ceiling_eval.json`):

  | Gains | Overall | gen | load | line |
  |---|---|---|---|---|
  | Ceiling-tuned | 0.848 | 0.583 | 1.000 | 0.979 |
  | Pole placement | 0.850 | 0.588 | 1.000 | 0.979 |

  - Gen trips, ceiling gains: G1 0.247, G2 0.969; α < 0.4 0.837, 0.4–0.7 0.476, ≥ 0.7 0.407.
  - Median max deviation: gen 0.0043, load 0.0014, line 8.6e-5 pu.
- **Why it plateaus: the actuator, not the gains.**
  - Multiplying every gain by ×3, ×10 or ×30 does not raise gen-trip rescue: 0.585, 0.566, 0.542. For G1 trips with α ≥ 0.7 it stays at 0.015.
  - At ×30, which in effect drives every survivor at the 2.5 pu/s rate limit, the largest G1 trip held in the envelope is α ≈ 0.2–0.5 (≈ 0.5–1.3 pu lost), depending on duration. G2 trips are held up to α 0.8.
  - So most G1 trips are beyond any controller under the frozen actuator, and scheduling cannot exceed it either.
- **Line screen re-run** against the ceiling gains (`line_screen_ceiling.json`): 65 of 1980 cells needy, still only line 1 (59 cells) and line 5 (6 cells).
- **Decision (2026-11-21, Noor): freeze the POLE-PLACEMENT generalist** (`seeds_v2.json` 'generalist') as the baseline. The ceiling tune found no better fixed gain set: it is equal within noise and marginally lower on the acceptance sample. The ceiling result is kept as the plateau evidence.
- Pre-registered acceptance (seed-7 sample, whole run): **0.850 overall — gen 0.588 / load 1.000 / line 0.979**. The needy definition and the line screen (`line_screen.json`) use this baseline.

## 6. S9 work order: the 2×2 screen matrix (rate limiter removed, v1.2) — run in Simulink
Sample: 1,200 scenarios (seed 7), whole run. Every number below comes from Simulink (`run_matrix_sl.m` on `case30_pidv2.slx`, fast restart). The ceiling gains were tuned in Python (`ceiling_tune.py A|B`, same protocol as §5) and scored in Simulink. The runner matched python to 6 significant figures on the spot check. Summary: `matrix_summary.json`.

| Round / gains | Overall | gen (G1 / G2) | load | line | Median max dev, gen/load/line [pu] | Median t_s, gen/load [s] | Median TV, gen/load [pu] | Largest G1 trip held (0.1 / 0.5 / 3 s) |
|---|---|---|---|---|---|---|---|---|
| A no rail / pole | 0.854 | 0.600 (0.251 / 1.000) | 1.000 | 0.979 | 0.0042 / 0.0011 / 8e-5 | 2.25 / 0.61 | 2.91 / 1.09 | 0.5 / 0.3 / 0.3 |
| A no rail / ceiling | 0.848 | 0.588 (0.247 / 0.979) | 1.000 | 0.972 | 0.0044 / 0.0013 / 7e-5 | 2.37 / 1.37 | 3.30 / 1.87 | 0.5 / 0.3 / 0.3 |
| B rail ±1 / pole | 0.854 | 0.600 (0.251 / 1.000) | 1.000 | 0.979 | 0.0042 / 0.0011 / 8e-5 | 2.25 / 0.61 | 2.91 / 1.09 | 0.5 / 0.3 / 0.3 |
| B rail ±1 / ceiling | 0.841 | 0.585 (0.247 / 0.974) | 1.000 | 0.954 | 0.0043 / 0.0013 / 7e-5 | 2.26 / 1.46 | 2.96 / 1.71 | 0.5 / 0.3 / 0.3 |

- **The rail costs nothing here.** The pole-placement gains never reach ±1 pu, so rows A and B are identical. Removing the rate limit (v1.2) lifts G2 trips from 0.969 to 1.000; G1 trips do not move.
- **The ceiling tunes plateau again:** gen stays at 0.65 on the tuning sample in both rounds. On the acceptance sample they are slightly worse than pole placement.
- **Why G1 trips fail: the tripped machine's OWN speed.** With no rail and no rate limit, gains ×10 / ×30 hold every survivor within 0.0016 / 0.0008 pu, but G1 itself still peaks at 0.0104 (α 0.7) and 0.0146 (α 1.0). Per Q8, G1's PID is frozen during its trip, so no controller on the other machines can hold G1's speed. The envelope is max over machines, so the case fails.
  - This is a definition question, not an actuator one: does the tripped machine's speed count toward the envelope?

## 7. S13 (v1.3) screen: C-hybrid gen faults (2026-11-21)
Plant: v1.2 actuator (rail ±1.0, no rate limit), pole-placement generalist, `gen_mode='v13'` (actuator hash ae1a45b349ab).
- α < 1: connected runback, the machine's own PID stays active, restored at clearing.
- α = 1: permanent disconnection at t_fault. `kron_drop` removes the machine's internal node; its row and column are exactly 0 and the rest equals the 5-machine reduction (checked to 0.0). The machine's swing and PID are frozen.

Cases: exactly the §6 set (`matrix_scen.mat`): the same 1,200 seed-7 scenarios, split, fault mix and α values (every gen α < 1, so all are runbacks), plus the 57-point G1 α grid. Full disconnections are a separate group, `disc`: every gen case of the 1,200 re-run at α = 1 (417 cases, same machine and t_fault), reported separately.

Two scorings are reported:
- **counted:** every machine in the network.
- **excluded:** the faulted machine is dropped from t_fault onward.

Python (`screen_v13.py` → `screen_v13.json`) and Simulink (`run_v13_sl.m` → `v13sl_*.mat`, `compare_v13_sl.py` → `screen_v13_simulink.json`) agree: max |Δpeak| is 7.6e-6 pu, with **0 pass/fail flips** across 1,674 cases. Python also reproduces §6 B/pole exactly under the legacy `q8` mode (0.854, G1 0.251).

| 1,200 eval set | Overall | gen (G1 / G2) | load | line |
|---|---|---|---|---|
| §6 reference (Q8: frozen PID, counted) | 0.854 | 0.600 (0.251 / 1.000) | 1.000 | 0.979 |
| v1.3 counted | **0.871** | 0.647 (0.341 / 1.000) | 1.000 | 0.979 |
| v1.3 excluded | **0.921** | 0.791 (0.610 / 1.000) | 1.000 | 0.979 |

- G1 runbacks by α:
  - counted: α < 0.39 **1.000**, 0.39–0.7 0.012, ≥ 0.7 0.015;
  - excluded: 1.000 / 0.704 / 0.074.

  The 0.39 edge matches Ashhar's prediction (rail +1.0 / Pm0 2.58 = 0.388). Below it, the unit's own PID fully replaces the loss.
- Largest G1 α held at each duration (0.1 / 0.5 / 3 s): counted 0.55 / 0.35 / 0.35, excluded 0.95 / 0.60 / 0.60. For comparison, §6 gave 0.5 / 0.3 / 0.3.
- Load and line are unchanged; the gen semantics don't touch them.
- **α = 1 disconnection group (separate):**
  - G2 1.000 (peak 1.0e-3);
  - **G1 0.000** (peak 8.5e-3 on the survivors, t_s about 3.9 s, back inside the 1e-3 band by 8 s).

  With constant gains the outcome depends only on the machine, so in effect this group is 2 cases.
- **Gain probe** (`gain_probe_v13.log`, G1, all gains multiplied):
  - G1 disconnection peak: ×1 0.0085, ×3 0.0045, ×10 0.0017, ×30 0.0011. The rail binds ≤ 2 % of the time, so it is **rescuable with more gain: needy, not capped.**
  - Deep runbacks, excluded scoring: ×3 brings α 0.9 to 0.0030, so rescuable.
  - Deep runbacks, counted scoring: α 0.9 stays at 0.0078–0.0083 even at ×30. The unit's own speed is rail-capped, so under counted scoring these cases are unrescuable regardless of gains.
- Implication for the ruling: under **excluded**, gen-trip rescue is fully open (deep runbacks and the G1 disconnection are needy and actionable). Under **counted**, G1 runbacks above α ≈ 0.39 stay a metric cap, as in §6. Ashhar rules which scoring is frozen.
- Code change, minor: TV now counts only actual Pm moves. Before, a frozen machine's would-be moves leaked into TV in python; Simulink always used the actual Pm. Peaks are unaffected. `gen_mode` defaults to `'q8'` so the §2–§6 scripts reproduce unchanged.

## 8. S16 speed probe (2026-11-21; `speed_probe.py`, `speed_probe_sl.m` → `speed_probe_py.json`, `speed_probe_sl.mat`)
Cases: eval idx 1–64 of `matrix_scen.mat`, v1.3, rail 1.0, no rate limit, pole gains, the full 8 s at 1 ms on both runners. Peaks agree to 8e-8 pu.

| Runner | 1 scenario | 64 batch | Scaling |
|---|---|---|---|
| Simulink, fast restart (warm) | **0.16 s** | 8.7 s (7.4 cases/s) | linear, one trajectory per sim; 3–4 MATLAB processes in parallel ≈ 5.8 cases/s each (§7 runs) |
| Simulink, no fast restart | 6.4 s | — | model load 7.5 s, first fast-restart compile 0.6 s |
| Python CPU (float64) | 1.9 s | **4.4 s (14.4 cases/s)** | 256 → 8.0 s; 1,200 → 16.8 s (71 cases/s) |
| Python CUDA (float64) | 12.3 s | 12.3 s (5.2 cases/s) | flat up to 10k (launch-bound); 1,200 → 12.6 s (95 cases/s) |

- **Single run:** Simulink is fastest (12× faster than Python CPU).
- **64 batch:** Python CPU is 2× faster than Simulink, and CUDA is slower than CPU.
- **≥ 1,000 cases:** Python CUDA wins (≈ 13–19× Simulink serial).

The §16 division stands: Python for batch and search work, Simulink for spot checks and ground truth.

## 9. S15 ceiling re-tune on v1.3-excluded, partial losses only (2026-11-21)
Protocol: `ceiling_tune.py v13x`, the same frozen protocol as §5. CMA-ES seed 20261122, σ0 0.5, population 16, 80 generations, worst-class objective, tuning sample 600 (seeds 101–103). Plant settings: v13 semantics, rail 1.0, no rate limit, **excluded scoring**. Gen cases are all α < 1 (asserted); α = 1 is held for Ashhar's reconnection ruling. Runtime 16 min.

- **Tuning sample:** 1.00 / 1.00 / 1.00 (gen / load / line) from generation 5 on, against 0.86 / 1.00 / 0.955 at the start.
- **Gains** (`baseline_ceiling_v13x.json`, `ceiling_v13x.mat`), ratio to pole placement for Kp / Ki / Kd:

  | | Kp | Ki | Kd |
  |---|---|---|---|
  | G1 | ×4.7 | ×2.65 | ×0.82 |
  | G2 | ×2.6 | ×0.48 | ×0.31 |
  | G3 | ×1.26 | ×0.17 | ×0.74 |
  | G4 | ×1.47 | ×0.18 | ×2.4 |
  | G5 | ×1.01 | ×0.22 | ×0.59 |
  | G6 | ×2.1 | ×0.39 | ×1.25 |

  In short: Kp is up and Ki is down on every machine except G1. Kp 237 / 163 / 91 / 113 / 82 / 94; Ki 5.8 / 15.5 / 7.5 / 8.1 / 5.2 / 15.6.

Same 1,200 seed-7 eval cases, excluded scoring (`eval_ceiling_v13x.py`). Python and Simulink are identical: max |Δpeak| 5.4e-6, **0 flips**.

| Gains | Overall | gen (G1 / G2) | load | line | G1 α <0.39 / 0.39–0.7 / ≥0.7 | Median max dev g/l/l [pu] | p95 max dev g/l/l | Median TV g/l |
|---|---|---|---|---|---|---|---|---|
| Pole placement | 0.921 | 0.791 (0.610 / 1.000) | 1.000 | 0.979 | 1.00 / 0.70 / 0.07 | 1.2e-3 / 1.1e-3 / 8e-5 | 7.3e-3 / 2.5e-3 / 3.0e-3 | 2.08 / 1.09 |
| **Ceiling re-tune** | **1.000** | **1.000 (1.000 / 1.000)** | 1.000 | **1.000** | 1.00 / 1.00 / 1.00 | 6.3e-4 / 8.2e-4 / 6e-5 | 4.2e-3 / 1.9e-3 / 2.7e-3 | 1.88 / 1.09 |

- **Margins:** the worst case under the re-tune is 0.00495 pu, and 12 of 1,200 cases lie within 10 % of the limit (17 under pole placement). The faulted unit's own peak (auxiliary) median is 1.8e-3 (pole 2.9e-3).
- **G1 runback α grid** (54 points with α < 1): pole 40/54, re-tune **54/54** (max 0.0044).
- **Line screen** (33 lines × 60 durations, `line_screen_v13x.json`):
  - pole: 58 needy cells, all on line 1. Line 5 is no longer needy without the limiter.
  - **re-tune: 9 needy cells, all on line 1** (max 0.0053), with 30 cells within 10 % of the limit.
- **Consequence:** under excluded scoring, one fixed gain set rescues every partial-loss and load case in the sample. The re-tuned baseline leaves almost nothing needy: 0 of 1,200 eval cases, and only 9 line-1 cells.
  - The needy pool would then be only line-1 long faults and (held) G1 α = 1 disconnections.
  - The ≥ 30 needy-line-instant quota is likely unreachable against this baseline.
  - This changes what the scheduler can be shown to add, so it needs Ashhar's ruling before the baseline is re-pinned.

## 10. EXPLORATORY: α = 1 reconnection and handoff prototype (2026-11-21; not spec, awaiting Ashhar)
Script `reconnect_proto.py` → `reconnect_proto.json/.npz`, plots `reconnect_proto_G1_1.0s.png`, `_G1_3.0s.png`, `_G2_3.0s.png`. Setup: §18 baseline gains (ceiling re-tune), rail 1.0, T = 12 s (longer than the dataset's 8 s so the handoff can settle). Cases: G1 durations 0.1/0.5/1/2/3 s and G2 durations 0.5/3 s, t_fault 1.0, each run with no return, a 2.5 pu/s ramp, and a 0.5 pu/s ramp.
- **Reconnection.** The machine is resynced to ω = 1, with its EMF angle aligned with the 5-machine terminal-bus voltage. Pm is initialised to the Pe it takes on at closing: G1 −0.010 pu, G2 +0.039 pu. The angle alignment does not give exactly 0, because the Thevenin impedance is not purely reactive. The PID restarts bumpless, so the command is continuous.
- **Handoff.** The survivors' integrator-held extra is absorbed into their references (bumpless). The machine's reference ramps from Pe to Pm0 at R. Each survivor's reference returns to its own Pm0 by the same fraction s(t). The final references sum to the pre-fault dispatch, so the common integrator offset settles to 0 and the old shares are restored exactly in the limit. During the ramp the references are not zero-sum: at closing the survivors' extra is mostly proportional action (Kp·e, a speed sag), not integrator-held.
- **Limits** (proposal, flagged): the ±1 pu limit applies to the PID correction around the moving reference, |P_set − P_ref(t)| ≤ 1. Survivors also keep their absolute band Pm0 ± 1. G1's band follows its reference, capped at Pm0 + 1, because G1 cannot satisfy |P_set − Pm0| ≤ 1 below 1.58 pu. The 2.5 pu/s rate applies to the reference ramp only, not to the PID corrections (v1.2 has no actuator rate limit). The rail never bound in any case.

| Case (G1) | Peak while disconnected | Peak after reconnection | Over-frequency from ramp | ω error at 12 s | G1 Pm at 12 s | Share regained (±2 %) |
|---|---|---|---|---|---|---|
| no return | 0.0051–0.0052 | 0.0037–0.0051 (the inherited sag at closing) | none | 1.4–1.5e-3 (still sagging) | 0.41–0.49 | no |
| ramp 2.5 pu/s | same | same | +0.9–1.1e-3 | 8e-5 | 2.554 | **yes, 1.37 s after reconnection** |
| ramp 0.5 pu/s | same | same | +0.6–0.7e-3 | 3.5e-4 | 2.51 | no (≈ 2.7 % short at 12 s; integrator tail) |

- **G2** (Pm0 0.46): peaks stay ≤ 0.001 in every variant. No return leaves G2 at 0.12–0.14 pu, and G1 carries the difference. With the 2.5 / 0.5 ramps G2 is back at its share in 0.35–0.45 s / 0.96 s.
- **The peak after reconnection is the sag inherited at closing** and is identical across variants. The handoff only adds a small over-frequency (≤ 1.1e-3) and removes the long tail.
- **Without a planned return**, the low-Ki baseline (Kp/Ki ≈ 15–40 s) leaves a 1.5e-3 pu frequency error at 12 s. G1 sits at about 0.45 pu, pushed up only by its own proportional response to the sag.
- **Side finding:** G1 full trips already reach 0.0051–0.0052 pu *during* the disconnection under the §18 baseline. The baseline was tuned on partial losses only, so α = 1 G1 is needy against it.

## 11. S20 (v1.4): α = 1 permanent disconnection, re-measured against the S18 baseline (2026-11-21; `disc_v14.py` → `disc_v14.json`)
Plant: v13 (α = 1 never cleared within the 8 s run). Survivors' references stay at Pm0, so the S20 floating band equals the v1.2 rail. Cases: all 417 gen cases of the 1,200 at α = 1 (same machine and t_fault). Scoring: excluded. Simulink (`run_v13_sl('disc', ·, 2, 'v13x')`) matches Python: max |Δpeak| 1.4e-8, 0 flips.

| Gains | G1 (223) in envelope | G1 peak | G1 t_s (1e-3 band) | G1 ω error at 8 s | G2 (194) | G2 peak |
|---|---|---|---|---|---|---|
| **S18 baseline (ceiling re-tune)** | **0.000** | **0.00515** | 5.75 s | 3.3e-3 | 1.000 | 0.00080 |
| Pole placement (reference) | 0.000 | 0.00848 | 3.85 s | 0.9e-3 | 1.000 | 0.00100 |

- **G1 α = 1 is needy against the baseline, but only marginally.** Its peak is 3 % over the limit, and every case has the same peak (t_fault does not matter with constant gains). The rail never binds. G2 α = 1 is never needy.
- **Actionable.** Scaling the S18 gains up gives these G1 peaks:

  | Gain scale | G1 peak | Rail binds |
  |---|---|---|
  | ×1.5 | 0.0038 | no |
  | ×2 | 0.0030 | no |
  | ×3 | 0.0021 | 1 % of the time |
  | ×5 | 0.0016 | — |
  | ×10 | 0.0013 | 2.7 % of the time |

  The binding fraction stays ≤ 2.7 %, so the scheduler has headroom even with a modest gain change.
- **The slow tail matters more than the peak.** Under the S18 gains (low Ki), G1 α = 1 is still 3.3e-3 pu off nominal at 8 s, with t_s 5.75 s. G1's 2.58 pu is carried mostly by proportional action, and the integrators take tens of seconds to take it over. Up to ×3 the gains stay outside the 1e-3 band through 8 s; ×5 settles in 0.63 s. This tail is part of the deviation and settling headline for the α = 1 class.
- The α = 1 class therefore carries a small rescue-rate story (G1: 0 % → 100 % needs only about ×1.5), and a larger deviation and settling story.
