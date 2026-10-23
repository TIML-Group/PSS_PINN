#!/bin/bash
set -u
cd /mnt/d/Ongoing/PINN/PSS_PINN/bus_9
../.venv/Scripts/python.exe -u profile_latency.py --graph --batch 256 --reps 60 --dt_roll 0.020 --n_roll 75 --out profile_latency_out/coarse20
../.venv/Scripts/python.exe -u profile_latency.py --graph --batch 256 --reps 60 --dt_roll 0.030 --n_roll 50 --out profile_latency_out/coarse30
../.venv/Scripts/python.exe -u profile_latency.py --graph --batch 256 --reps 60 --dt_roll 0.040 --n_roll 38 --out profile_latency_out/coarse40
echo LAT-DONE
