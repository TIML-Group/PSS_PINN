# Era-2 identification training — results (2026-11-22, training spec v1.0)

## Setup
- **Code:** `train_ident_v2.py` (training), `ident_model.py` (model and features), `eval_ident_v2.py` (true-dynamics evaluation), `sched_sim.py`.
- **Checkpoints:** `ident_ckpt/` holds 10 models (cls / reg × profile 32, 64, 128, none = 256 ms, learned). Each is about 9.5 MB.
- **Results:** `ident_eval.json`, `ident_eval_summary.md`.
- **Data and splits:** `data_v2` (X recorded under R0 gains). Splits as generated: main 8000 / 1000 / 1000, α = 1 per machine 300 / 100 / 100. The test split is 1,000 main + 200 α = 1.
- **Decision protocol:** one latched switch at t_d = t_fault + L. R2 switches at the relay-supplied t_clear; there is no clearing for α = 1. If t_d > t_clear, the run goes straight to R2 at t_d.
  - **Assumption: the fault onset instant is known**, as a trigger, the same way t_clear is relay-supplied. Ashhar needs to confirm this.
- **Input:** an onset-aligned 256 ms window at 2 ms (128 steps) with the era-1 88-feature featurizer, normalised on the train split. A mask over elapsed time since onset hides everything after t_d.
  - Fixed profiles use a hard mask.
  - `learned` uses the era-1 sigmoid-edge mechanism with penalty 0.05·L/L_max and β annealed 10 → 2 ms.
  - Pre-fault windows (2 per scenario) form the abstain class.
- **Model:** bias-free per-step projection 88 → 32, flatten, era-1 MLPBlock (384 × 6, tanh).
  - **V-CLS:** 59 classes (abstain + 58 R1/R1d configurations), class-weighted cross-entropy (n^-1/2) + L1 on α.
  - **V-REG:** L1 on 36 log-gains (R1 and R2).
- **Training:** Adam with cosine LR 1e-3 → 1e-5, batch 512, 300 epochs, patience 40. Each model trains in 30–100 s on CUDA.

## Results (test split, true dynamics; median excluded peak ×1e-4 pu)
Each cell is scheduled (delayed oracle / oracle / generalist), then the scheduled in-envelope rate. See `ident_eval_summary.md`.

| Model | L | ID | gen runback | load | line | α=1 G1 | α=1 G2 | bar 1 | bar 2 | bar 3 |
|---|---|---|---|---|---|---|---|---|---|---|
| **cls_32** | 32 | 99.2 % | 4.37 (4.37/4.34/6.48) 100 % | 6.87 (6.87/5.97/8.50) | 0.63 (0.63/0.60/0.61) | **37.7 (37.7/9.9/51.5) 100 %** | 3.30 (3.30/0.25/7.96) | fail: line (+2.8 %, 1.7e-6 pu) | **100 %** | fail (delay) |
| **cls_learned** | 57 | 99.6 % | 4.40 | 7.37 | 0.60 | 48.3 (48.3/9.9/51.5) 100 % | 5.16 | **pass** | **100 %** | fail (delay) |
| cls_64 | 64 | 99.5 % | 4.67 | 7.50 | 0.59 | 49.6 100 % | 5.58 | pass | 100 % | fail (delay) |
| cls_128 | 128 | 99.6 % | 6.26 | 8.45 | 0.58 | 50.9 **0 %** | 7.75 | pass | 0 % | fail |
| cls_none | 256 | 99.4 % | 6.45 | 8.50 | 0.56 | 51.5 0 % | 7.96 | fail | 0 % | fail |
| reg_32 … reg_none | | 0–30 % of gains within 5 % | | | | | | fail | 0–100 % | fail |

**Findings**
1. **V-CLS identification works.**
   - Test configuration accuracy is 99.2–99.6 % (59 classes); α-MAE is 0.017–0.027.
   - Almost every error is an abstention on a line whose fault has no measurable effect (line 20, 29, 32, 40). The wrong-schedule cost is about 0: **scheduled = delayed oracle** in every class.
2. **The whole scheduled-vs-oracle gap is switching delay.** The oracle switches at t_fault (zero delay); any window delays the switch. For α = 1 G1 the peak by switching delay is:

   | Switching delay | Peak |
   |---|---|
   | 0 (oracle) | 0.00099 |
   | 32 ms | 0.00377 |
   | 57 ms | 0.00483 |
   | 64 ms | 0.00496 |
   | 128 ms | 0.00509 (out of envelope) |
   | 256 ms | 0.00515 (= generalist) |

   - The disconnection peak builds in the first ~100–250 ms, so windows ≥ 128 ms lose the rescue.
   - **Bar 3 (within 10 % of the oracle) is unreachable by construction** for any L > 0 on load / α = 1. Proposal for Ashhar: judge bar 3 against the **delayed oracle** (the library switched at the same t_d), which isolates identification quality. V-CLS meets that version everywhere, since scheduled = delayed oracle.
3. **Bars 1 and 2.**
   - cls_learned (57 ms) and cls_64 pass bar 1 (never worse per class: median peak and in-envelope) and bar 2 (α = 1 G1 100 %).
   - cls_32 passes bar 2 with the best peaks. It fails bar 1 only on line, by 1.7e-6 pu in median (line peaks are about 1 % of the limit).
   - 128 / 256 ms fail bar 2.
4. **Headline (generalist → scheduled median peak), cls_32:**

   | Class | Change |
   |---|---|
   | Runback | −32.5 % |
   | Load | −19.2 % |
   | Line | +2.8 % (≈ equal) |
   | α = 1 G1 | −26.8 % (0.00515 → 0.00377; in envelope 0 → 100 %) |
   | α = 1 G2 | −58.6 % |

   cls_learned: −32.1 / −13.3 / −2.1 / −6.2 / −35.1 %.
5. **Learned window:** V-CLS learned L = 57 ms; V-REG learned L = 86 ms.
6. **V-REG fails the ablation.**
   - Only 0–30 % of test cases get gains within 5 % of the target (log-gain MAE 0.3–1.5).
   - It is never better than V-CLS, it fails bar 1 on line or runback, and survivors' rail-binding reaches 0.55–0.78 on some line cases.
   - A 1,500-epoch retrain (diagnostic, `ident_ckpt_reg1500/`) was unstable: four of five runs early-stopped worse than at 300 epochs, and reg_learned collapsed to L = 0 (a constant gain set that rescues α = 1 G1 but fails line, 98 % in envelope).
   - **V-CLS is the design of record.**
7. **Per-scenario note.** With any switching delay ≥ 8 ms, about 55 % of runback cases end slightly worse than the generalist (median excess 1e-5 pu, max 1.4e-4), although the median is 32 % better. The zero-delay oracle is never worse. Bar 1 is per class, so it passes; this is recorded as a diagnostic.
8. **Library robustness item: R2_line under delayed entry.**
   - One of 337 line test cases (line 1, a 72 ms fault) has survivors at the rail 84 % of the time when R2_line takes over from a delayed or generalist trajectory. The peak stays in envelope (0.0027).
   - The R2_line entry has Kp about 1,000 on G1 / G2 and was tuned on oracle-entry states.
   - It also happens with no identifier at all (R2 at clearing after generalist R1), so it is a library-verification gap, not identification.

## Open for Ashhar
- (a) Bar 3 against the delayed oracle vs the zero-delay oracle.
- (b) The known-onset trigger assumption.
- (c) The design-of-record profile: cls_32 (best peaks; line bar 1 fails by 1.7e-6 pu) or cls_learned (57 ms, passes bars 1–2).
- (d) Re-verify the R2 entries under delayed entry states.
- (e) Checkpoint policy: commit all 10 (about 95 MB) or only the design of record.

## §31 End-to-end decision latency (2026-11-22, work order §31)
- **Machine:** NVIDIA RTX 5070 Ti Laptop GPU. Every column was measured on this machine with the era-1 protocol: warm state, 100 reps, host-synced eager, whole-decision CUDA-graph replay.
- **Scope:** controller architecture only; the observation window (57 ms) is **not** added.
- **Script:** `profile_latency_e2e.py` (results in `profile_latency_e2e.json`). It times one decision for the design of record, cls_learned, along this path:
  1. onset trigger: index into an on-GPU 2 ms ring buffer;
  2. window gather (128 × 90);
  3. era-1 featurize and normalise;
  4. IdentNet;
  5. argmax and library lookup (R1 + R2 family entry);
  6. PID command update at the switch step (bumpless integrator preload + one PID step, 6 machines).
- **Graph check:** max |diff| between graph and eager is 0.

| Decision (batch 1) | Eager [ms] | Graph replay [ms] | Batch 256, eager / graph [ms] |
|---|---|---|---|
| **Era 2, cls_learned e2e (30-bus)** | **1.17** (median 1.14, p90 1.36) | **0.133** (p90 0.148) | 1.15 / 0.56 |
| Era 1, 30-bus full decision, dt_roll = 100 ms | 88.1 | 11.8 | 101.7 / 35.1 |
| Era 1, 30-bus full decision, dt_roll = 10 ms | 685.5 | 61.5 | 708.5 / 87.6 |
| Era 1, 9-bus full decision, dt_roll = 100 ms (wiki: 85.6 / 20.0 at B256, other machine) | 74.5 | 8.7 | 80.4 / 12.7 |
| Era 1, 9-bus full decision, dt_roll = 10 ms (wiki: 464.9 / 126.6 at B256) | 712.1 | 59.5 | 730.5 / 69.9 |
| Era 1, 30-bus encoder-only dP (CUDA-event median) | 16.7 | 4.5 | 30.8 / 27.6 |

- **Era-2 stage split** (eager, synced median, batch 1), in ms:

  | Stage | Time [ms] |
  |---|---|
  | Gather | 0.05 |
  | Featurize | 0.17 |
  | MLP | 0.50 |
  | Lookup | 0.03 |
  | PID update | 0.29 |

  Eager is launch-bound (about 1 ms of kernel launches); graph replay removes that, leaving 0.13 ms.
- **Against the era-1 30-bus full decision at its operating point** (dt_roll = 100 ms, batch 1): era 2 is 75× faster eager (88.1 → 1.17 ms) and 89× faster in graph replay (11.8 → 0.133 ms). It has no rollout and no RK4 share; era 1 spends 93–99 % of its time in rollouts.
- **Notes.**
  - The era-1 profilers (`bus_9`, `bus_30` `profile_latency.py`) crash in the graph section when it is run without `--dt_roll` / `--n_roll` overrides: a duplicate `eager_ms` keyword. The 10 ms columns were run with explicit `--dt_roll 0.01 --n_roll 150` (9-bus) or a scratch copy with the duplicate removed (30-bus). The repo files are unchanged.
  - §31 item 3: the 30-bus era-1 profiler's loader (`case30_common.Setup`) already works.

## Spec v1.1 trunk-shrink ablations (2026-11-23)
V-CLS only (V-REG failed in v1.0), with the same splits, profiles, latch protocol, loss, optimiser and epochs as v1.0. One definition in `ident_model.py` (`TRUNKS`), with `train_ident_v2.py --trunk`. Checkpoints: `ident_ckpt_trunk/`. True-dynamics evaluation: `ident_eval_trunk.json`. Latency: `ident_ckpt_trunk/*_latency.json` (the §31 harness).

**Trunks**
- **A, temporal pooling:** a shared per-step MLP (192 × 4) on [32-dim projection | elapsed / L_max], then mask-weighted mean + soft-masked max over the observed steps, then the heads. **143,613 parameters (16× fewer).**
- **B, narrow flatten, the spec shape 128 × 3:** **567,997 parameters.** The spec's "~0.25 M" doesn't fit this shape, because the first layer alone is 4096 × 128 = 524 k.
- **B64, flatten at 64 × 3:** 277,245 parameters, which matches the ~0.25 M target. Trained at the learned profile only.

**Results** (test split, true dynamics). Bar 3 is judged against the delayed oracle (§30). Headline = generalist → scheduled change in median peak, per class (runback / load / line / α = 1 G1 / α = 1 G2).

| Model | Params | L [ms] | ID acc | α-MAE | Bar 1 | Bar 2 (α = 1 G1) | Bar 3 (delayed) | Headline |
|---|---|---|---|---|---|---|---|---|
| cls_learned (base, design of record) | 2.34 M | 57 | 99.58 % | 0.018 | pass | 100 % | pass | −32.1 / −13.3 / −2.1 / −6.2 / −35.1 % |
| **cls_learned_A** | **0.14 M** | **28** | **100 %** | **0.008** | line (+3.3 %, 2e-6 pu, the same class as cls_32) | **100 %** | pass | **−32.5 / −20.8 / +3.3 / −32.2 / −63.0 %** |
| cls_64_A | 0.14 M | 64 | 100 % | 0.006 | pass | 100 % | pass | = cls_64 |
| cls_32_A / 128_A / none_A | 0.14 M | 32 / 128 / 256 | 99.92 / 100 / 99.92 % | ≤ 0.008 | as base at the same L | as base (100 / 0 / 0 %) | pass | = base at the same L |
| cls_learned_B | 0.57 M | **78** | 99.33 % | 0.017 | pass | **0 %** | pass | −19.5 / −8.4 / −5.6 / −1.4 / −20.7 % |
| cls_32/64/128/none_B | 0.57 M | fixed | 99.33–100 % | ≤ 0.019 | as base | as base | pass | = base at the same L |
| cls_learned_B64 | 0.28 M | **75** | 99.67 % | 0.019 | pass | **0 %** | pass | −21.3 / −8.9 / −5.2 / −1.6 / −22.5 % |

**e2e latency** (batch 1, the §31 harness, same GPU):

| Model | Eager [ms] | Graph replay [ms] |
|---|---|---|
| base | 1.17 | 0.133 |
| A | 1.14 | 0.137 |
| B | 0.91 | 0.090 |
| B64 | 0.88 | 0.098 |

**Findings**
1. **The trunk size doesn't limit identification.** A (16× smaller) is the most accurate model trained: 99.9–100 %, zero abstentions, and α-MAE 0.006–0.008, which is 2–3× better than the 384 × 6 base. The pooling plus elapsed-time input suits the onset-aligned task.
2. **Peaks depend on L only.** Every model has scheduled = delayed oracle, so at a fixed profile all trunks give identical peaks. The trunk matters only through accuracy and through the window the learned profile settles on.
3. **The learned window is not trunk-invariant at λ_win = 0.05.** It settles at A 28 ms, base 57 ms, B 78 ms, B64 75 ms.
   - At 75–78 ms the α = 1 G1 rescue is lost (peak 0.00507–0.00508 > limit), so **learned-B and learned-B64 fail bar 2**.
   - learned-A's 28 ms gives the best peaks of any model (α = 1 G1 0.00349, −32 %), with the same 2e-6 pu line bar-1 miss as cls_32.
4. **Candidates per the spec's tie rule:**
   - **cls_64_A** ties the base exactly on bars 1–3, with 16× fewer parameters.
   - **cls_learned_A** passes bars 2–3, misses bar 1 on line by the margin §30 called rounding noise, and has better peaks than the design of record.
5. **Latency barely moves** (0.09–0.14 ms graph across 0.14–2.34 M parameters). The decision is launch-bound, so the trunk-size argument is parameters and accuracy, not wall-clock.

**New finding (affects §29 too): the bumpless transfer is not bumpless under delayed entry.**
- **Oracle switching (L = 0):** the switch-step jump is 0 in every class, so bumpless is exact, as verified in §27.
- **Switching at L = 57 ms (the same for every trunk):** the incoming integrator preload hits the static band clamp (Pm0 ± 1). A large e times stiff R1 gains puts I′ outside the band, so the commanded power jumps in one 1 ms step.

| Class (L = 57 ms) | Switch-step jump, median / max [pu] | Ordinary-step change, median / max [pu] |
|---|---|---|
| α = 1 G1 | **0.90 / 0.90** | 0.29 / 0.29 |
| Gen runback | 0 / 0.94 | 0.03 / 0.89 |
| Line | 0 / 0.97 | 0 / 1.37 |

- This is above §23's 0.05 bump criterion. It was in `ident_eval.json` (`bump_max`) but not reported in §29.
- It doesn't change any peak reported: the simulations include it. It is the same library-verification gap as the R2_line item (entries verified on oracle-entry states only), and it extends §30(d)'s re-verification to the **R1 entries** under delayed entry.

## §34(c) Delayed-entry re-verification and repair → library v2.2 (2026-11-23)

**Method**
- `reverify_delayed.py` reruns the §24 grid (same cases and references) with the R1 switch at t_f + L. R2 switches at max(t_clear, t_f + L), and never for α = 1.
- Scoring is `build_library.score` with §28 survivors' rail-binding. Bump is the entry's own transfer: R1/R1d entries the switch into R1, R2 entries the switch into R2. `sched_sim` now logs `bump_r1` and `bump_r2` separately; the old single `bump` mixed the two.
- Acceptance states are L = 0 (oracle entry) and L = 28 ms (cls_learned_A). L = 32, 57 and 64 ms are diagnostics only.
- `repair_delayed.py` runs CMA-ES from the current entry (log-multipliers) with the frozen §26 budget: population 16 × 40 generations, σ0 0.3, ≤ 60 stride cases.
  - Every case is simulated at both acceptance states.
  - Hard terms: median floor, envelope, needy rule, rail ≤ 5 %, own bump ≤ 0.05.
  - Main term: R1 median peak at 28 ms + 0.1 × at L = 0; R2 mean t_s.
  - No softer-gains fallback and no slew.
  - R1/R1d entries are repaired first, then the R2 entries still failing against the repaired R1 entries.
  - An entry is accepted only if its full-grid failure count drops.
- Runtime 71 min.

**v2.1 re-verification** (62 entries)

| L | failing | which |
|---|---|---|
| 0 | 1 | R2_runback_G1 t_s 1.192 vs reference 1.173 s (pre-existing: §27 finalize fell back R1_runback_G1 → generalist without re-verifying R2) |
| 28 ms | 7 | bump > 0.05: R1_line_1 0.785, R1_line_7 0.983, R1_line_10 0.945, R1_runback_G2 0.945, R1d_G1 0.883, R1d_G2 0.531; + R2_runback_G1 |
| 57 ms (diag.) | 10 | + R1_line_9 (envelope loss, worse), R2_line (rail 0.588, bump 0.575), R2_runback_G2 (bump 0.926) |

**Repair** (grid entry median excluded peak in pu; R2 rows mean t_s in s)

| Entry | Result | bump at 28 ms | median at L = 0 | median at 28 ms |
|---|---|---|---|---|
| R1_line_1 | accepted | 0.785 → 0.000 | 2.548e-3 → 2.574e-3 (+1 %) | 2.738e-3 → 2.719e-3 |
| R1_line_7 | accepted | 0.983 → 0.049 | 9.52e-4 → 1.054e-3 (+11 %, ref 1.059e-3) | 1.048e-3 → 8.34e-4 (−20 %) |
| R1_runback_G2 | accepted | 0.945 → 0.000 | 3.02e-4 → 2.53e-4 (−16 %) | 2.55e-4 → 2.34e-4 |
| R1d_G2 | accepted | 0.531 → 0.044 | 2.5e-5 → 4.2e-5 (ref 7.96e-4) | 2.94e-4 → 3.07e-4 (+4 %) |
| R2_runback_G1 | accepted | 0 → 0 | t_s 1.192 → 1.123 (ref 1.173) | same |
| R2_runback_G2 | accepted (L = 0 fixed) | 1.941 → **0.104** | t_s 0 → 0 | bump at L = 0 1.737 → 0 |
| R1_line_10 | **rejected** | 0.945 → 0.945 | — | — |
| R1d_G1 | **rejected** | 0.883 → 0.603 (still > 0.05) | — | — |

- R2_runback_G2 entered stage B only because repairing R1_runback_G2 created a new R1 → R2 bump. It passed at L = 0 on v2.1.

**Final v2.2 verification**
- **L = 0:** 0 failing, bump max 0.000.
- **L = 28 ms:** 3 failing, all on bump: R1_line_10 0.945, R1d_G1 0.883, R2_runback_G2 0.104.
- **L = 57 ms (diag.):** 7 failing. R1_line_7 and R1d_G2 bump again because they were repaired at 28 ms only; R1_line_9, R2_line, R1_line_10, R1d_G1, R2_runback_G2 also fail.

**Is the remaining bump reachable by gains?** A uniform scale of the entry gains at L = 28 ms, all cases of the entry, gives:

| Entry | ×0 | ×0.1 | ×0.3 | ×1 | ×3 |
|---|---|---|---|---|---|
| R1_line_10 (23 cases, all > 0.05 at ×1) | 0 | 0 | 0.945 | 0.945 | 0.984 |
| R1d_G1 (α = 1 G1, 4 cases) | 0 | 0.416 | 0.846 | 0.883 | 0.941 |
| R2_runback_G2 (254 cases) | 0 | 0 | 0 | 0.104 (1 %) | 0.842 |

- **R1d_G1 is structural.** 28 ms after the G1 trip, Kp·e already exceeds the integrator band headroom, so the preload clamps. Even 0.1× gains bump 0.42 pu. Meeting 0.05 pu means effectively switching the controller off.
  - Under §34's rules (no slew, no softer-gains fallback), this entry cannot pass at 28 ms. It needs a decision.
- **R1_line_10** needs ≤ 0.1× gains, which is far outside the σ0 0.3 search (about 8σ) and would almost certainly lose the peak floor.
- **R2_runback_G2** is a near miss (0.104 pu in 1 % of cases; ×0.3 is clean). More budget would very likely fix it.

**cls_learned_A in true dynamics, library v2.1 → v2.2** (test split, oracle re-simulated with v2.2; `eval_ident_v2.py --lib --glob`, outputs `ident_eval_v22check_*.json`)

| Class | median peak (dOrc) | bar 1 / # worse than gen | bump max (frac > 0.05) |
|---|---|---|---|
| Gen runback | 0.00044 → 0.00043 (= dOrc) | yes / 203 → 1 | 0.944 (47 %) → **0.158 (1 %)**, now R2 |
| Load | 0.00067 (= dOrc) | yes / 13 → 5 | 0 → 0 |
| Line | 0.00006 (= dOrc) | no (same 2e-6 pu miss) / 85 → 85 | 0.948 (12 %) → 0.945 (3 %) |
| α = 1 G1 | 0.00349 (= dOrc) | yes / 0 | 0.883 (100 %) → 0.883 (100 %) |
| α = 1 G2 | 0.00029 → 0.00031 | yes / 0 | 0.531 (100 %) → **0.044 (0 %)** |

- Accuracy 100 %, bar 2 100 %, and bar 3 is scheduled = delayed oracle in every class.
- Peaks are unchanged to within 2e-5 pu.
- cls_64_A (64 ms) on v2.2 also keeps all its bars. Its α = 1 G2 bump stays at 0.95 because 64 ms is not an acceptance state. Line rail improves 0.69 → 0.02.

**Open for the wiki owner**
1. **R1d_G1 at 28 ms.** Gains cannot satisfy bump ≤ 0.05 without disabling control. Choose one:
   - an exemption for α = 1 disconnections;
   - a mechanism change: slew, or a preload that tolerates the band;
   - accept and disclose.
2. **R1_line_10 and R2_runback_G2.** Allow a wider or longer repair (σ0, generations) for these two, or disclose.
3. **The 57 ms diagnostics** (R1_line_9 envelope loss, R2_line rail 0.59) are outside the acceptance states. They matter only if a 57 ms variant is ever deployed.

## §36(b) Focused repair of R2_runback_G2 → library v2.3; correction to §35 (2026-11-23)

**Focused repair** (`repair_focused.py`)
- Same objective and constraints as §35: hard terms at L = 0 and 28 ms, no slew, no softer gains. Starts from the v2.2 entry.
- Tuned on **all 254 cases** of the entry. §35's stride subset (51 cases) missed the 1 % that still bumped.
- Budget: σ0 0.6, population 24, 60 generations; 47 min.
- **Accepted.** Own bump at 28 ms 0.104 → 0.049; at L = 0 it stays 0. t_s stays 0 (ref 0) and in-envelope stays 100 %.
- The CMA objective reported viol 0 from generation 0 even on all cases, so the starting entry's 0.104 case was in the grid but not the tuning set. The search moved off it within the hard-term plateau.

**Correction: 6-dp gain rounding in the library writer**
- `repair_delayed.py` wrote gains with `np.round(·, 6)`. That moved the two generalist-fallback entries (R1_load_30, R1_runback_G1) 4.8e-7 off g18.
- With gains no longer identical, the scheduler registers a switch at t1. The switch re-preloads the integrator (I′ = P_prev − Kp·e − …), which unwinds a rail-clamped integrator, so the dynamics change.
- §35's verification used the in-memory gains, so its library tables are correct. **Its cls_learned_A table was run on the rounded JSON and is not.**
- Fixed:
  - writers now store full precision;
  - R1_load_30 and R1_runback_G1 are restored to the exact v2.1 values in `library_v22.json` and `library_v23.json`;
  - both libraries were re-verified from JSON (`reverify_v22.json`, `reverify_v23.json`);
  - `ident_eval_v22check_library_v22.*` is replaced by `ident_eval_check_v22.*`.
- The corrected v2.2 verification reproduces §35 exactly: L = 0 → 0 fails; 28 ms → R1_line_10, R1d_G1, R2_runback_G2.

**cls_learned_A on v2.2, corrected vs as reported in §35** (runback n = 364, load n = 299)

| Class | Median peak | # worse than gen | Bump max (frac > 0.05) |
|---|---|---|---|
| Gen runback | 0.000437 (reported 0.000434) | **147** (reported 1; v2.1 203) | 0.158 (0.5 %) — unchanged |
| Load | 0.000673 | **13** (reported 5; v2.1 13) | 0 |
| Line / α = 1 G1 / α = 1 G2 | unchanged | unchanged | unchanged |

- Bars are unchanged (bar 1 pass on runback and load, scheduled = delayed oracle).
- **The "203 → 1" in §35, cited in §36(1), is wrong. Corrected: 203 → 147.**
- Side observation: an identical-gains re-preload at the decision time acts as an integrator reset and lowers runback peaks. It is not used; it would be a mechanism change.

**Library v2.3** (from JSON; `reverify_v23.json`)

| L | Failing | With the §36(a) R1d exemption and the §36(b) R1_line_10 disclosure |
|---|---|---|
| 0 | 0 (bump max 0.000) | 0 |
| **28 ms** | 2: R1_line_10 bump 0.945, R1d_G1 0.883 | **0** |
| 57 ms (diag.) | 7: R1_line_7, R1_line_9, R1_line_10, R1d_G1, R1d_G2, R2_line, R2_runback_G2 (0.101) | — |

- R1d disclosed transfer at 28 ms: R1d_G1 0.883, R1d_G2 0.044.

**cls_learned_A on v2.3** (test split, true dynamics)
- 100 % ID, bar 2 100 %, scheduled = delayed oracle in every class.
- **Gen runback:** median 0.000437 → 0.000453 (+3.7 %, the cost of the R2_runback_G2 repair). Still bar 1 pass; 148 worse than gen.
- **Gen runback bump:** 0.158 → **0.051**, 0.3 % of cases above 0.05. The grid max is 0.049, so the test split has one case 0.001 over.
- **Other classes:** load, line, α = 1 G1 and α = 1 G2 unchanged from v2.2.
- **cls_64_A:** gen runback 0.000438 → 0.000495, bump 0.164 → 0.070; other classes unchanged.

## Needy-line re-count on the test split (2026-11-24, §24/§38 chain item)

`needy_recount.py`. Needy means the §18 generalist (g18, no switching) leaves the envelope: excluded peak > 0.005 pu. Counts come from dataset v2 meta; re-simulating the generalist reproduces meta exactly (max |diff| 0). Oracle and cls_learned_A (28 ms) are on library v2.3.

| Line faults | n | Needy | Near miss (0.9–1.0 × limit) |
|---|---|---|---|
| **test** | 337 | **0** | 11 |
| val | 326 | 4 | 5 |
| train | 2605 | 4 | 41 |
| all | 3268 | 8 | 57 |

Test context, needy counts in the other classes:

| Test class | n | Needy |
|---|---|---|
| Gen runback | 364 | 0 |
| Load | 299 | 0 |
| α = 1 G1 | 100 | 100 |
| α = 1 G2 | 100 | 0 |

- **Every needy and near-miss line case is line 1.**
  - The 8 needy cases are long faults, 2.65–2.98 s, with generalist peaks 0.00504–0.00527.
  - The near misses run 1.16–2.58 s; the test-split maximum is 0.004988.
- **The oracle and cls_learned_A rescue all of them** (oracle 0.00078, cls_learned_A 0.00198, identification correct in every case). The peak falls before the duration matters, so it is the same for every line-1 case. Non-test cases are in-sample for the model.
- **Test-split line in-envelope is 100 %** for generalist, oracle and cls_learned_A alike. The line class has no rescue-rate story on the test split; it is a deviation story, matching §18's retirement of the quota.
- The 8 needy line cases sit in train/val only, a consequence of the seeded split. The only needy test population is α = 1 G1 (100/100), rescued 100 % (bar 2).

## §40 ask: 5070 Ti re-run of the final controller (2026-11-24)

**Machine and protocol**
- RTX 5070 Ti Laptop on **AC power**. A first attempt on battery was throttled (eager 4.05 ms, p90 6.95 ms) and is discarded.
- Same scripts as Ashhar's runs: `profile_latency_e2e_v23_A.py` and `profile_latency_e2e_opt.py --burn 6000`. cls_learned_A on library v2.3, reps 100.

**1. Base A config** (`profile_latency_e2e_A.json`)

| Batch | Eager mean / median / p90 [ms] | Graph mean / median / p90 [ms] | max\|diff\| |
|---|---|---|---|
| 1 | 1.296 / 1.250 / 1.475 | **0.141 / 0.134 / 0.156** | 0 |
| 256 | 2.609 / 2.573 / 2.792 | 3.252 / 2.676 / 3.542 | 0 |

- Batch-1 stages: gather 0.055, featurize 0.174, mlp 0.552, lookup 0.026, pid_update 0.243.
- At batch 256 the A trunk's per-step pooling MLP dominates (mlp 2.09 ms), and graph shows no gain over eager.

**2. Fused probe, warmed** (`profile_latency_e2e_opt_5070.json` / `.log`; 6000 burn-in decisions per batch)

| Batch 1 | Eager mean / median / p90 [ms] | Graph mean / median / p90 [ms] | max\|diff\| | CUDA launches |
|---|---|---|---|---|
| gpu_base | 1.417 / 1.382 / 1.640 | 0.130 / 0.136 / 0.151 | 0 | 87 |
| **gpu_fused** | **0.851 / 0.842 / 0.947** | **0.098 / 0.099 / 0.127** | 2.2e-9 | **54** |
| cpu_pid (discarded) | 0.825 / 0.811 / 0.933 | 0.268 / 0.262 / 0.284 (graph + host) | 7.3e-12 | 40 (GPU arm) |

| Batch 256 | Eager mean / median / p90 [ms] | Graph mean / median / p90 [ms] | max\|diff\| | CUDA launches |
|---|---|---|---|---|
| gpu_base | 2.950 / 2.917 / 3.133 | 2.391 / 2.427 / 2.513 | 0 | 84 |
| gpu_fused | 2.689 / 2.655 / 2.890 | 2.369 / 2.449 / 2.776 | 4.8e-7 | 52 |
| cpu_pid | 2.615 / 2.586 / 2.770 | 2.550 / 2.521 / 2.677 | 4.8e-7 | 36 |

- **Numerics:**
  - Batch 1: feature maxdiff 4.2e-4, logits 3.0e-3, 0 argmax mismatches, min top1–top2 margin 8.93.
  - Batch 256: logits 5.7e-3, 0 mismatches, margin 2.78.
- **Launches are measured directly on CUDA.** The profiler emits `cudaLaunchKernel` here, so no proxy is needed. Base 87 → fused 54 (−38 %); kernel-issuing aten ops 285 → 186.
- **Fused batch-1 stages:**

  | Stage | Base [ms] | Fused [ms] |
  |---|---|---|
  | gather | 0.038 | 0.052 |
  | featurize | 0.187 | 0.081 |
  | mlp | 0.647 | 0.371 |
  | lookup | 0.026 | 0.018 |
  | pid | 0.309 | 0.201 |

- End-of-batch base recheck: batch 1 1.222 / 0.136, batch 256 2.989 / 2.448, consistent with the start.

**Findings**
- **Batch 1:** fused is −39 % eager (1.38 → 0.84 median) and −27 % graph (0.136 → 0.099 median). The final controller's decision is **~0.1 ms as one CUDA graph** on this machine, about 3.6× faster than the 6800 XT's 0.35–0.37 ms.
- **CPU PID:** on CUDA it edges the fused path on eager at batch 1 (0.811 vs 0.842 median). As a graph plus host round trip it loses (0.262 vs 0.099). The §40 discard ruling holds: the shipped graph is fused-GPU.
- **Batch 256:** the variants converge (graph ≈ 2.4–2.5 ms) because the A trunk's pooling MLP dominates. Launch overhead matters only at batch 1, which is the operating mode.

## §42 (wiki) Single-machine latency comparison: era-1 one-pass ΔP head on GPU + era-2 stage split, CPU vs GPU (2026-11-25, Noor)

**Scope.** All numbers are from one machine, the RTX 5070 Ti Laptop (CPU: Ryzen AI 9 HX 370, 12 threads).
- Settings: torch 2.12.0.dev20260408+cu128, AC power, batch 1 (the operating mode).
- Era 1 had no GPU-graph timing for its no-rollout variant. This section adds it, then splits the era-2 decision into encoder / lookup / PID on GPU and on CPU.

**1. Comparison table** (30-bus, batch 1, RTX 5070 Ti)

| Controller | Eager [ms] | CUDA graph [ms] | Source |
|---|---|---|---|
| Era 1, LSTM + RK4 cascade, dt_roll = 100 ms | 88.1* | 11.8* | §32 |
| Era 1, MLP encoder + one-pass ΔP head (no rollout) | **1.362** (mean 1.414, p90 1.516) | **0.184** (mean 0.191, p90 0.206) | new |
| Era 2, cls_learned_A fused + gain lookup + PID | **0.842** (mean 0.851, p90 0.947) | **0.099** (mean 0.098, p90 0.127) | §41 |

\*The era-1 profiler reports means. The other two rows are medians; for an all-means table use 1.414 / 0.191 and 0.851 / 0.098.

9-bus one-pass row, for reference: 1.211 / 0.105 ms (median), 1.293 / 0.105 ms (mean).

**2. Era-1 one-pass measurement** (`latency_era1_oneshot_graph.py`)

| | 30-bus | 9-bus |
|---|---|---|
| Encoder | frozen MLP encoder of the MLP+MLP teacher, `case30_fullmlp_ctrl/refined` (ep 170): 6,927,695 params | frozen MLP encoder of the MLP+MLP teacher, `receding_fullmlp/refined`: 476,442 params |
| Head (final checkpoint) | `oneshot_dp30/run_novmask`: 75,778 params | `oneshot_dp/run_teacher_v3_novmask`: 36,355 params |

- **Decision:** the raw window is already on the GPU. Then: featurise (normalised on 30-bus) → x_meas → encoder → context [z, p, α, x0] → one-pass head → ΔP. There is no RK4.
- **Protocol (as era 2 §32):** 300 test windows (frozen seed-42 draw) after 30 warm-ups; eager is host-synced per decision; the whole decision is captured as one CUDA graph.
- **Numerics:** graph vs eager max |diff| 0. The GPU-featurise path vs the repo's numpy path (`time30.py` / `time_neural.py`): 2.9e-7 (30-bus), 1.3e-5 (9-bus).
- **Earlier timings** were CPU only (0.74 / 1.72 ms) plus one 9-bus GPU eager run (1.20 ms, with host featurise, H2D/D2H and the older `run_label_tapply` head). They are kept but not comparable to this table.
- The `oneshot_dp` / `oneshot_dp30` folders are imported read-only and unchanged.

**3. Era-2 stage split anchored to the table** (§41 run, RTX 5070 Ti, medians)

| Stage | GPU eager [ms] | GPU CUDA graph [ms] |
|---|---|---|
| Encoder (gather 0.052 + featurize 0.081 + IdentNet 0.371) | 0.504 (60 %) | ≈ 0.068 (~69 %) |
| Gain lookup | 0.018 (2 %) | ≈ 0.010 (~10 %) |
| PID update | 0.201 (24 %) | ≈ 0.021 (~21 %) |
| Output copies + dispatch (not stage-attributed) | 0.119 (14 %) | — |
| **Whole decision** | **0.842** | **0.099** |

- GPU eager stages are the §41 per-stage medians.
- §41 timed the graph only as a whole. The graph split applies the per-segment shares of the split run (table 4) to 0.099 ms, hence "≈".

**4. Split run, CPU vs GPU** (`latency_era2_stage_split.py`; batch 1, 300 reps after 3,000 burn-in decisions, medians [ms])

The script executes the committed `profile_latency_e2e_opt.py` fused stage functions unchanged; only the device is switched.

| Stage | CPU eager | GPU eager | GPU graph (per segment) |
|---|---|---|---|
| **Encoder** | **0.325** | **0.493** | **0.089** |
| ↳ gather | 0.018 | 0.044 | — |
| ↳ featurize | 0.035 | 0.062 | — |
| ↳ IdentNet forward | 0.240 | 0.338 | — |
| **Gain lookup** | 0.008 | 0.022 | 0.013 |
| **PID update** | **0.061** | **0.219** | **0.027** |
| **Whole decision** | **0.423** | **0.762** | **0.096** |
| Encoder / PID share | 77 % / 14 % | 65 % / 29 % | ~69 % / ~21 % |

- **Numerics:** segmented vs whole max |diff| 0; graph vs eager 0.
- **Graph column:** the per-segment graphs sum to 0.129 ms against 0.096 ms whole, because each replay carries ~10 µs of fixed overhead. Use the whole number, and the segments for shares only.
- **Whole-graph agreement:** 0.096 ms here vs 0.099 ms in §41.
- **Eager disagreement:** GPU eager is 0.762 vs 0.842 ms. Eager clocks jitter run to run (here 3,000 burn / 300 reps; §41 6,000 / 100). The slide uses §41.

**Findings**
1. **Dropping the rollout is what makes era 1 fast.** The one-pass variant takes 0.184 ms against the cascade's 11.8 ms (64× in graph replay).
2. **Era 2 is still 1.9× faster in graph replay** (0.099 vs 0.184 ms) and 1.6× faster eager. Its network has about 50× fewer parameters (0.14 M vs 6.9 M encoder).
3. **The encoder dominates the era-2 decision** (60–77 %), then the PID update (14–29 %). The gain lookup is negligible.
4. **At batch 1 the CPU beats GPU eager** (0.423 vs 0.762 ms), because GPU eager is launch-bound. The GPU graph remains the fastest path, 4.4× faster than the CPU. The CPU column is the whole decision on the CPU; it is not §40's discarded cpu_pid split.

Files (`PSS_PINN/era2_sched_pid/bus_30/`):
- Scripts: `latency_era1_oneshot_graph.py`, `latency_era2_stage_split.py`.
- Results: `latency_era1_oneshot_graph_30.json` / `_9.json`, `latency_era2_stage_split_gpu.json` / `_cpu.json`.
- Wiki record: pid-scheduling-datagen-progress §42; comparison row added to wiki decision-latency-profile.
