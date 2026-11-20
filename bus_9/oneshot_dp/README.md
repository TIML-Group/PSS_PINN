# One-shot ΔP experiment (9-bus), 2026-11-19

Neural-only controller: measurement window -> frozen MLP encoder (receding_fullmlp/refined/best.pt)
-> OneShotHead (2x128 tanh MLP on [z, p, alpha, x0]) -> ΔP = tanh-bounded ±1 pu. No RK4 / solver at decision time.

- Targets: dataset y_delta_p (no per-window RK4-teacher ΔP was ever saved). Windows end at t_apply = t_fault + 0.05 s;
  9000 train / 500 val scenarios from the teacher checkpoint's split. Windows before t = 0 are padded with the first sample
  (the encoder's learned mask, ~0.28 s, zeroes them).
- Eval: verify_oneshot.py = verify_true.py with only the ΔP source swapped. Frozen seed-42 test draw
  (compare_out_s42/instants.json, 2048 instants, 397 needy), dp_hold clear (immediate shutoff at clearing).
- Koopman: koopman_rerun/ = regenerated fit-edmd fullchan + fit-kf + sweep --row koopman_gov (original _koopman_out was
  not on disk; osqp 1.1.3 installed). Reproduces 0.4811 / harmed 0.44% / warm 0.81 ms.

Files: oneshot_common.py, train_oneshot.py, verify_oneshot.py, time_neural.py, time_koopman.py,
run_label_tapply/{head.pt, train_results.json, verify_s42/}, teacher_verify_s42/, latency_neural*.json, latency_koopman.json, log_*.txt
