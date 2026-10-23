#!/bin/bash
set -u
cd /mnt/d/Ongoing/PINN/PSS_PINN/bus_30
PY=../.venv/Scripts/python.exe
CK=case30_ctrl/best.pt

echo "=== A/B eager (3 epochs) ==="
$PY -u refine_dp30.py --ckpt $CK --data data30_mm --epochs 3 --seed 7 \
    --floor_tol 0.005 --out receding_cascade_ab/refcg_ab_off

echo "=== A/B cuda_graph (3 epochs) ==="
$PY -u refine_dp30.py --ckpt $CK --data data30_mm --epochs 3 --seed 7 \
    --floor_tol 0.005 --cuda_graph --out receding_cascade_ab/refcg_ab_on

echo "=== compare A/B ==="
$PY -u -c "
import json
o=json.load(open('receding_cascade_ab/refcg_ab_off/refine_pareto.json'))
n=json.load(open('receding_cascade_ab/refcg_ab_on/refine_pareto.json'))
print('epoch | eager rescue/norm | CG rescue/norm')
for a,b in zip(o,n):
    print(f"{a['epoch']:5d} | {a['rescue']:.6f}/{a['norm']:.6f} | {b['rescue']:.6f}/{b['norm']:.6f}")
d=max(abs(a['rescue']-b['rescue']) for a,b in zip(o,n)), max(abs(a['norm']-b['norm']) for a,b in zip(o,n))
print('max diffs:', d[0], d[1])
"
echo REFINE30-AB-DONE
