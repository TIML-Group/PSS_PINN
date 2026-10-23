#!/bin/bash
set -u
cd /mnt/d/Ongoing/PINN/PSS_PINN/bus_9
CK=receding_cascade/refined_common/best.pt
echo "=== B2: dt_roll=20ms n_roll=38 (0.76s horizon, coarser spacing) ==="
../.venv/Scripts/python.exe -u verify_true.py --ckpt $CK --data data_mm \
  --batch 256 --cuda_graph --dt_roll 0.020 --n_roll 38 \
  --out receding_cascade/refined_common/verify_dt20_nroll38
echo "=== C2: dt_roll=5ms n_roll=150 (0.75s horizon, finer spacing) ==="
../.venv/Scripts/python.exe -u verify_true.py --ckpt $CK --data data_mm \
  --batch 256 --cuda_graph --dt_roll 0.005 --n_roll 150 \
  --out receding_cascade/refined_common/verify_dt5_nroll150
echo SWEEP4B-DONE
