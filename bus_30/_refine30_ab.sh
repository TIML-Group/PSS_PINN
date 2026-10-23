#!/bin/bash
set -u
cd /mnt/d/Ongoing/PINN/PSS_PINN/bus_30
PY=../.venv/Scripts/python.exe
CK=case30_ctrl/best.pt

echo "=== A/B eager (3 epochs) ==="
$PY -u refine_dp30.py --ckpt $CK --data data30_mm --epochs 3 --seed 7 --floor_tol 0.005 --out receding_cascade_ab/refcg_ab_off

echo "=== A/B cuda_graph (3 epochs) ==="
$PY -u refine_dp30.py --ckpt $CK --data data30_mm --epochs 3 --seed 7 --floor_tol 0.005 --cuda_graph --out receding_cascade_ab/refcg_ab_on

echo "=== compare A/B ==="
$PY -u _ab_compare.py
echo REFINE30-AB-DONE
