#!/bin/bash
set -u
cd /mnt/d/Ongoing/PINN/PSS_PINN/bus_9
CK=receding_cascade/refined_common/best.pt
# coarse end of the spacing sweep, full 1.5 s horizon:
echo "=== dt_roll=30ms n_roll=50 (1.5s) ==="
../.venv/Scripts/python.exe -u verify_true.py --ckpt $CK --data data_mm \
  --batch 256 --cuda_graph --dt_roll 0.030 --n_roll 50 \
  --out receding_cascade/refined_common/verify_dt30_nroll50
echo "=== dt_roll=40ms n_roll=38 (1.52s) ==="
../.venv/Scripts/python.exe -u verify_true.py --ckpt $CK --data data_mm \
  --batch 256 --cuda_graph --dt_roll 0.040 --n_roll 38 \
  --out receding_cascade/refined_common/verify_dt40_nroll38
echo SWEEP5-DONE
