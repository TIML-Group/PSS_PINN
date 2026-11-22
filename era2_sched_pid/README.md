# era2_sched_pid — Era 2: MLP gain-scheduled PID (governor-less plant)

Spec: PINN_Wiki `specs/pid-scheduling-spec.md` (amended 2026-11-21) and
`specs/pid-scheduling-answers.md` (Q1–Q9). Plant tag **`v2-nogov`**. Every artifact here
carries the actuator hash from `plant_v2.actuator_hash()`.
The era-1 code in `bus_30/` and `bus_9/` is read-only from here: this folder imports
`bus_30/case30_physics.py` and the `.mat` network data, and changes nothing there.
The 30-bus system comes first; the 9-bus port follows the same layout under `bus_9/`.

## bus_30/

| File | Role |
|---|---|
| `plant_v2.py` | Batched torch plant: era-1 swing + network, governor deleted. Digital per-machine PID on all 6 machines (Q2), deviation rail \|P_set−Pm0\| ≤ 1.0, rate limit 2.5 pu/s, integrator clamped at the rail, ZOH gains on a global 20 Hz grid, gen trip as an actuator-bypassing disturbance with the tripped PID frozen (Q8). |
| `case30_pidv2_build.m` / `case30_pidv2.slx` | The same plant in Simulink (governor removed; discrete PID block, ode4 at 1 ms). |
| `case30_pidv2_check.m` + `check_plant_v2.py` | No-fault equilibrium plus a Simulink-vs-python cross-check on load, line and gen faults and on time-varying gains. |
| `pole_place.py` → `seeds_v2.json` | Q9 stage 1: per-class pole-placement seeds (post / gen / load α-bands / line) and the generalist (Q7, one gain set). |
| `baseline_generalist.py` → `baseline_generalist.json` | Acceptance of the generalist on a representative 1200-scenario sample. |
| `screen_lines.py` → `line_screen.json` | §3 plan: all 33 lines × durations under the generalist. |
| `scenarios_v2.py` | Era-1 scenario generators (same ranges). |
| `tune_anchors.py` | Q9 stage 2: deterministic CMA-ES anchor tuner (fixed seed / budget / initialisation at the seed). |
| `lam_sweep.py` → `lam_sweep.json` | Q6: effort-weight sweep on 20 anchors, run before λ is pinned. |

## Order of operations
1. `matlab -batch "run('case30_pidv2_build.m'); run('case30_pidv2_check.m')"` then `python check_plant_v2.py`
2. `python pole_place.py`, then `python baseline_generalist.py`. Freeze the baseline and log it in the wiki.
3. `python screen_lines.py`
4. `python lam_sweep.py`. Pin λ and the settling band in the spec (dated amendment).
5. Anchor grid and the full anchor tuning run. *(not started: waits on step 4's pin)*
6. Label interpolation, dataset v2 generation, and freezing the s42 draw with the ≥30 needy-line check. *(not started)*
