# Era-2 library build: proposals for the spec v2.0 §9 pre-build items (DRAFT, 2026-11-22, Noor)

Basis: spec v2.0 + §15–§22. Evidence: `bus_30/prebuild_evidence.log`; linearisation from `pole_place.py` (z = δ/ωb, so Ki acts as synchronising torque); runs in `DATAGEN_NOTES.md` §7–§11.

## Evidence these proposals rest on

1. **The binding closed-loop mode is the common (frequency) mode, not a network mode.**
   - Its pole is real and well estimated by λc ≈ −ΣKi / Σ(D + Kp).
   - §18 baseline: λc = −0.072 s⁻¹ (time constant 14 s). Pole placement: −0.46 to −0.50 s⁻¹.
   - Under the §18 gains, the slowest pole is −0.071 to −0.073 across all 33 line configurations, and −0.064 / −0.095 in R1d (G2 / G1).
   - This is the 8 s tail seen in §21 (3.3e-3 pu residual at 8 s).
2. **Load configurations barely depend on α.**
   - K (= dPe/dδ) changes by at most 2.2 % between α = 0.1 and α = 1.0 (median 1.9 %).
   - The slowest decay moves by at most 0.0012 s⁻¹.
   - In the linearisation, a runback (α < 1) is the pre-fault network with a set-point offset, so α does not change it either.
3. **Settling in the 1e-3 band comes from either of two levers.**
   - A stiff proportional droop: a residual ΔP/ΣKp below the band.
   - A fast common mode: larger ΣKi.
   - Scaling all gains together leaves λc unchanged; ×5 settles G1 α = 1 in 0.63 s only through droop (2.58 / ΣKp ≈ 6.6e-4 < 1e-3).
   - So a t_s target has to name the lever.

## 1. Gain-library enumeration (configuration list)

| Regime | Configurations | Count | Basis |
|---|---|---|---|
| R0 no-fault | 1 global | 1 | — |
| R1 line | one per faultable line (switched Yred) | 33 | distinct networks |
| R1 load | one per load bus, **α collapsed** | 21 | evidence 2: K varies ≤ 2.2 % over α |
| R1 runback (α < 1) | one per machine (G1, G2), **α collapsed** | 2 | same network; α only shifts the offset |
| R1d disconnection (α = 1) | G1, G2 (5-machine reductions) | 2 | §22; only G1 is needy |
| R2 post-clearing | one per preceding family {line, load, runback-G1, runback-G2} | 4 | restored network is always pre-fault; the state at clearing differs by family |

- **Total: 63 entries × 18 gains** (Kp, Ki, Kd × 6 machines).
- **α bands are added only on demand.** If verification (item 4) shows an α-collapsed entry losing to the baseline in some α range, that entry splits into α bands. A split is a repair, not part of the base enumeration.
- **R2 may collapse to a single entry** if the four family entries are within verification tolerance of each other. That would be checked and reported.
- **R0 = the §18 baseline gains.** Pre-fault the plant sits at equilibrium, so R0 gains do not affect any metric. Making R0 identical to the generalist means the scheduled controller equals the baseline until it identifies a fault, which keeps the §6a oracle-gap comparison clean.

## 2. Dominant-pole targets per regime (frozen before the build)

Two target sets per regime:
- the **common mode** λc, which sets the tail and settling;
- the five **inter-machine (electromechanical) modes**, which set peak and ringing.

All targets must hold under the ±1 pu rail. True-dynamics verification confirms them.

| Regime | Objective (spec §3) | Inter-machine modes | Common mode λc | Notes |
|---|---|---|---|---|
| R0 | — | — | — | = §18 baseline (not designed) |
| R1 (line / load / runback) | min peak deviation | ζ ≥ 0.3, Re ≤ −0.5 s⁻¹ (as the existing pole placement) | Re ≤ −0.3 s⁻¹ | peak driven by Kp; λc kept moderate to limit integrator overshoot |
| R1d (G1, G2) | min peak deviation; the loss persists to 8 s | ζ ≥ 0.3, Re ≤ −0.5 s⁻¹ | **Re ≤ −1.0 s⁻¹** | the loss is permanent, so the tail counts for the whole run; e^{−1·t} brings 5e-3 to 1e-3 in 1.6 s |
| R2 | min t_s (1e-3 band) | **ζ ≥ 0.5**, Re ≤ −0.5 s⁻¹ | **Re ≤ −1.0 s⁻¹** | less ringing for settling |

- Ranking within a target region: smallest gains (gain-norm penalty, as `pole_place.py` does now).
- **Proposed check before freezing:** a quick sweep of λc ∈ {−0.3, −0.5, −1.0, −2.0} and ζ ∈ {0.3, 0.5} on a handful of configurations from each family. It measures true-dynamics peak, t_s and rail-binding, so the numbers above are frozen from data rather than assumed. Cost: minutes on CUDA.

## 3. Bumpless transfer implementation

**Recommend integrator preloading (conditioning), position form.** The PID is already position form with a rail-clamped integrator and anti-windup. The velocity form would change the anti-windup semantics and the integrator clamp that §10–§22 were measured with.

At each switch instant t_sw, with incoming gains (Kp′, Ki′, Kd′):

    I′ = P_set(t_sw⁻) − Kp′·e(t_sw) − Kd′·N_D·(e(t_sw) − xf(t_sw))

- The derivative-filter state xf carries over unchanged, so P_set(t_sw⁺) = P_set(t_sw⁻) exactly.
- I′ is then clamped to the rail.
- If the clamp is active (Kp′·e alone pushes past the band), the residual commanded step is **logged as the transition bump**. That is the "commanded-output continuity" metric in spec §6a.
- Switch instants are events: identification time (not t_fault), t_clear for α < 1, and no clearing switch for α = 1 (§20). There is no 20 Hz grid.
- The §19 prototype already used this preload at reconnection; the commanded step was 0 and the rail never bound.

## 4. Verification-driven repair budget

- **Verification set per entry:**
  - its configuration's true-dynamics cases on a fixed grid: durations 0.05–3 s (6 values) × α (5 values where α applies) × 2 fault instants;
  - plus the matching seed-7 cases from the 1,200;
  - Python CUDA, 8 s, with the entry's R2 follow-on for α < 1.
- **An entry fails verification if any of these hold:**
  - it is out of envelope where the baseline is inside;
  - its median peak deviation is worse than the baseline's;
  - rail-binding exceeds 5 % of the run;
  - its transition bump exceeds 0.05 pu.
- **Repair** uses the frozen CMA-ES protocol (population 16, **40 generations**, fixed seed). It starts from the pole-placed entry and uses the regime's objective plus a λ effort term; λ is re-swept on first use, per §15.
- **Budget:** at most 20 % of entries (≈ 12) and at most 3 h on CUDA in total.
  - If more than 20 % fail, stop and revisit the item-2 targets instead of polishing. Mass failure means the targets are wrong, not the entries.
  - An α-band split (item 1) counts as one repair.

## 5. Identification-head window profile

- **Keep the era-1 set: 32 / 64 / 128 ms plus no-window,** with the learned final window (era-1 w_max mechanism).
- Report time-to-correct-identification against window (spec §6a item 2).
- Also report the **switching delay cost**: the deviation lost between t_fault and the identification switch, measured with the oracle entry. That ties window length to the deviation headline.
- No new window lengths are proposed until those curves exist.

## 6. α = 1 case count for the dataset

- The era-1 builder uses N_SCEN = 10,000 with a uniform type mix (≈ 3,333 gen rows). In a uniform α draw, α = 1 has probability zero.
- **Proposal:** a separate α = 1 block of **1,000 rows (500 G1 + 500 G2)**, alongside the unchanged main mix.
  - The main split, fault mix and α ranges stay unchanged; the block is reported separately, as in §14.
  - Rows vary t_fault ~ U[0.5, 4] s and era-1 noise; t_clear = N/A.
- **Eval split:** at least 100 per machine (≫ the effective-N floor of 10) so per-machine deviation statistics are stable.
- Rationale: only 2 physical configurations exist, so extra rows mostly train the encoder's disconnection detection under timing and noise variation. They are cheap.

## Open for Ashhar

1. R2 keyed per preceding family (4), with collapse allowed.
2. The λc targets (R1 −0.3, R1d / R2 −1.0 s⁻¹), pending the item-2 sweep.
3. R0 = the §18 baseline.
4. The repair fail criteria and the 20 % stop rule.
5. The α = 1 block size of 1,000.
