#!/bin/bash
set -u
cd /mnt/d/Ongoing/PINN/PSS_PINN/bus_9
CK=receding_cascade/refined_common/best.pt
echo "=== VERIFY dt=100ms TEST split ==="
../.venv/Scripts/python.exe -u verify_true.py --ckpt $CK --data data_mm --batch 256 --cuda_graph --dt_roll 0.1 --n_roll 15 --split test --out receding_cascade/refined_common/verify_dt100_test
echo "=== VERIFY dt=100ms VAL split confirm (same frozen draw as sweep) ==="
../.venv/Scripts/python.exe -u verify_true.py --ckpt $CK --data data_mm --batch 256 --cuda_graph --dt_roll 0.1 --n_roll 15 --split val --out receding_cascade/refined_common/verify_dt100_val_confirm
echo "=== COMPARE RMSE @ dt10 reference (ckpt default), TEST split ==="
../.venv/Scripts/python.exe -u compare_models.py --ckpt $CK --data data_mm --split test --n 4096 --n_true 2048 --batch 256 --dp_hold clear --out receding_cascade/refined_common/compare_dt10_test
echo "=== COMPARE RMSE @ dt100, TEST split (dt_roll override) ==="
../.venv/Scripts/python.exe -u compare_models.py --ckpt $CK --data data_mm --split test --n 4096 --n_true 2048 --batch 256 --dp_hold clear --dt_roll 0.1 --out receding_cascade/refined_common/compare_dt100_test
echo CONFIRM-DONE
