#!/bin/bash
set -u
cd /mnt/d/Ongoing/PINN/PSS_PINN/bus_9
CK=receding_cascade/refined_common/best.pt
# isolate SAMPLING from HORIZON: all three runs have the SAME 0.75 s horizon
echo "=== A: dt_roll=10ms n_roll=75 (re-run of the 0.742 row, control) ==="
../.venv/Scripts/python.exe -u verify_true.py --ckpt $CK --data data_mm \
  --batch 256 --cuda_graph --n_roll 75 --dt_roll 0.010 \
  --out receding_cascade/refined_common/verify_dt10_nroll75
echo "=== B: dt_roll=20ms n_roll=38 (same horizon, coarser spacing) ==="
../.venv/Scripts/python.exe -u verify_true.py --ckpt $CK --data data_mm \
  --batch 256 --cuda_graph --dt_roll 0.020 \
  --out receding_cascade/refined_common/verify_dt20_h075
echo "=== C: dt_roll=5ms n_roll=150 (same horizon, finer spacing) ==="
../.venv/Scripts/python.exe -u verify_true.py --ckpt $CK --data data_mm \
  --batch 256 --cuda_graph --dt_roll 0.005 \
  --out receding_cascade/refined_common/verify_dt5_h075
echo SWEEP4-DONE
