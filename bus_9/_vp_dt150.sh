#!/bin/bash
set -u
cd /mnt/d/Ongoing/PINN/PSS_PINN/bus_9
CK=receding_cascade/refined_common/best.pt
echo "=== VERIFY dt=0.15 n_roll=10 ( horizon 1.5s ) ==="
../.venv/Scripts/python.exe -u verify_true.py --ckpt $CK --data data_mm \
  --batch 256 --cuda_graph --dt_roll 0.15 --n_roll 10 \
  --out receding_cascade/refined_common/verify_dt150
echo "=== LATENCY dt=0.15 ==="
../.venv/Scripts/python.exe -u profile_latency.py --graph --batch 256 --reps 60 \
  --dt_roll 0.15 --n_roll 10 --out profile_latency_out/dt150
echo VP-dt150-DONE
