#!/bin/bash
set -u
cd /mnt/d/Ongoing/PINN/PSS_PINN/bus_9
CK=receding_cascade/refined_common/best.pt
for R in 1 2 4; do
  echo "=== n_refine=$R (n_roll=150) ==="
  ../.venv/Scripts/python.exe -u verify_true.py --ckpt $CK --data data_mm \
    --batch 256 --cuda_graph --n_refine $R \
    --out receding_cascade/refined_common/verify_nref$R
  grep -E "rescue|mean.dP" /dev/null 2>/dev/null || true
done
for K in 25 50 75; do
  echo "=== n_roll=$K (n_refine=3) ==="
  ../.venv/Scripts/python.exe -u verify_true.py --ckpt $CK --data data_mm \
    --batch 256 --cuda_graph --n_roll $K \
    --out receding_cascade/refined_common/verify_nroll$K
done
echo SWEEP-DONE
