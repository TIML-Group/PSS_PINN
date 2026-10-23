#!/bin/bash
set -u
cd /mnt/d/Ongoing/PINN/PSS_PINN/bus_9
CK=receding_cascade/refined_common/best.pt
echo "=== n_refine=2 n_roll=75 (combined) ==="
../.venv/Scripts/python.exe -u verify_true.py --ckpt $CK --data data_mm \
  --batch 256 --cuda_graph --n_refine 2 --n_roll 75 \
  --out receding_cascade/refined_common/verify_nref2_nroll75
echo SWEEP3-DONE
