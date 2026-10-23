#!/bin/bash
set -u
cd /mnt/d/Ongoing/PINN/PSS_PINN/bus_9
CK=receding_cascade/refined_common/best.pt
echo "=== COMPARE RMSE @ dt10 reference (ckpt default), TEST split ==="
../.venv/Scripts/python.exe -u compare_models.py --ckpt $CK --data data_mm --split test --n 4096 --n_true 2048 --batch 256 --dp_hold clear --out receding_cascade/refined_common/compare_dt10_test
echo "=== COMPARE RMSE @ dt100, TEST split (dt_roll override) ==="
../.venv/Scripts/python.exe -u compare_models.py --ckpt $CK --data data_mm --split test --n 4096 --n_true 2048 --batch 256 --dp_hold clear --dt_roll 0.1 --out receding_cascade/refined_common/compare_dt100_test
echo CONFIRM-DONE
