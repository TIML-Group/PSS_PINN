#!/bin/bash
set -u
cd /mnt/d/Ongoing/PINN/PSS_PINN/bus_9
rm -rf receding_cascade/refined_common/verify_dt200 receding_cascade/refined_common/verify_dt300
bash _vp_dt200.sh > out_dt200.txt 2>&1 & bash _vp_dt300.sh > out_dt300.txt 2>&1 & wait
echo PAIR-200-300-DONE
bash _vp_dt400.sh > out_dt400.txt 2>&1 & bash _vp_dt500.sh > out_dt500.txt 2>&1 & wait
echo PAIR-400-500-DONE
