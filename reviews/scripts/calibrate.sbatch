#!/bin/bash
#SBATCH -J mwr-calib
#SBATCH -A mishra-sl3-cpu
#SBATCH -p icelake
#SBATCH -N1 -n1 -c4
#SBATCH -t 01:30:00
#SBATCH -o /rds/user/da579/hpc-work/MathWorld_Reviews/runs/logs/calib_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=da579@cam.ac.uk
set -uo pipefail
. /rds/user/da579/hpc-work/MathWorld_Reviews/env/env.sh
RUN=$MWR_ROOT/runs/calibration/calib_${SLURM_JOB_ID}
. "$MWR_ROOT/scripts/stage.sh"
cd "$STAGE/work"
python "$MWR_ROOT/scripts/calibrate_grind.py" --gate "${CONJ_REFUTE_PROVER}" --out "$RUN/calibration.json"
rc=$?
echo "calibration exit $rc"
exit $rc
