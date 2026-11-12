#!/bin/bash
# sbatch --array=0-39 --dependency=afterany:<eval array> --export=ALL,WAVE=sweep1 scripts/wave_analysis.sbatch
# index = arm * 4 + dataset; every arm in tasks.tsv except c00_centre, sorted.
# Same two steps as the centre: analysis.sbatch (raw metrics) then analysis_readable.sbatch (vs Zenodo CSV).
#SBATCH -J mwr-wave-anal
#SBATCH -A mishra-sl3-cpu
#SBATCH -p icelake
#SBATCH -N1 -n1 -c4
#SBATCH -t 02:00:00
#SBATCH -o /rds/user/da579/hpc-work/MathWorld_Reviews/runs/logs/wave_analysis_%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=da579@cam.ac.uk
set -uo pipefail
ROOT=/rds/user/da579/hpc-work/MathWorld_Reviews
mapfile -t ARMS < <(awk -F'\t' 'NR>1 && $2!="c00_centre"{print $2}' "$ROOT/runs/$WAVE/tasks.tsv" | sort -u)
ARM=${ARMS[$(( SLURM_ARRAY_TASK_ID / 4 ))]}
DS_IDX=$(( SLURM_ARRAY_TASK_ID % 4 ))
export ARM WAVE SKIP_BLOCKS=1
echo "arm=$ARM dataset_index=$DS_IDX start $(date -Is)"
SLURM_ARRAY_TASK_ID=$DS_IDX bash "$ROOT/scripts/analysis.sbatch"; rc=$?
if [ "$rc" -ne 0 ]; then echo "wave analysis stopped after raw metrics: arm=$ARM rc=$rc $(date -Is)"; exit "$rc"; fi
SLURM_ARRAY_TASK_ID=$DS_IDX bash "$ROOT/scripts/analysis_readable.sbatch"; rc=$?
echo "wave analysis done arm=$ARM rc=$rc $(date -Is)"
exit "$rc"
