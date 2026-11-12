#!/bin/bash
#SBATCH -J mwr-julia
#SBATCH -A mishra-sl3-cpu
#SBATCH -p icelake
#SBATCH -N1 -n1 -c8
#SBATCH -t 01:00:00
#SBATCH -o /rds/user/da579/hpc-work/MathWorld_Reviews/runs/logs/julia_%j.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=da579@cam.ac.uk
set -uo pipefail
. /rds/user/da579/hpc-work/MathWorld_Reviews/env/env.sh
export TMPDIR=/dev/shm/da579_mwr_${SLURM_JOB_ID}; mkdir -p "$TMPDIR"
echo "node $(hostname) start $(date)"
bash "$MWR_ROOT/env/julia_setup.sh"
rc=$?
rm -rf "$TMPDIR"
echo "julia_setup exit $rc $(date)"
[ $rc -eq 0 ] && echo "JULIA OK"
exit $rc
