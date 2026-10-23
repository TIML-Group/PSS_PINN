#!/bin/bash
set -u
cd /mnt/d/Ongoing/PINN/PSS_PINN/bus_9
CK=receding_cascade/refined_common/best.pt
echo "=== VERIFY dt=0.05 n_roll=30 ( horizon 1.5s ) ==="
../.venv/Scripts/python.exe -u verify_true.py --ckpt $CK --data data_mm \
  --batch 256 --cuda_graph --dt_roll 0.05 --n_roll 30 \
  --out receding_cascade/refined_common/verify_dt050
echo "=== LATENCY dt=0.05 ==="
../.venv/Scripts/python.exe -u profile_latency.py --graph --batch 256 --reps 60 \
  --dt_roll 0.05 --n_roll 30 --out profile_latency_out/dt050
echo VP-dt050-DONE
