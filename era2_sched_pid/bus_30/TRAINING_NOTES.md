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
