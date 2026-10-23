#!/bin/bash
set -u
cd /mnt/d/Ongoing/PINN/PSS_PINN/bus_9
CK=receding_cascade/refined_common/best.pt
echo "=== VERIFY dt=0.2 n_roll=8 ==="
../.venv/Scripts/python.exe -u verify_true.py --ckpt $CK --data data_mm \
  --batch 256 --cuda_graph --dt_roll 0.2 --n_roll 8 \
  --out receding_cascade/refined_common/verify_dt200
echo "=== LATENCY dt=0.2 ==="
../.venv/Scripts/python.exe -u profile_latency.py --graph --batch 256 --reps 60 \
  --dt_roll 0.2 --n_roll 8 --out profile_latency_out/dt200
echo VP-dt200-DONE
