#!/bin/bash
# sbatch --array=0-3 --export=ALL,WAVE=sweep1,ARM=c00_centre scripts/analysis_readable.sbatch   (index = D0..D3)
# Needs the raw CSVs written by analysis.sbatch.
#SBATCH -J mwr-readable
#SBATCH -A mishra-sl3-cpu
#SBATCH -p icelake
#SBATCH -N1 -n1 -c4
#SBATCH -t 02:00:00
#SBATCH -o /rds/user/da579/hpc-work/MathWorld_Reviews/runs/logs/readable_%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=da579@cam.ac.uk
set -uo pipefail
. /rds/user/da579/hpc-work/MathWorld_Reviews/env/env.sh
DSS=(D0 D1 D2 D3)
DS=${DSS[$SLURM_ARRAY_TASK_ID]}
export CONJ_REFUTE_DATASET_PROFILE=$DS
BASE=$MWR_ROOT/runs/$WAVE/analysis/$ARM/$DS
RUN=$BASE/readable
. "$MWR_ROOT/scripts/stage.sh"
export TMPDIR=$STAGE/tmp
mkdir -p "$RUN/csv" "$RUN/logs" "$TMPDIR"
cd "$STAGE/work"
SRC=$STAGE/src
export PYTHONPATH=$SRC
case $DS in
  D0) PREMISES=(rank-nullity connected orientable) ;;
  D1) PREMISES=(rank-nullity connected) ;;
  *)  PREMISES=(rank-nullity) ;;
esac
PF=(); for p in "${PREMISES[@]}"; do PF+=(--premise "$p"); done
PAPER=$BASE/csv/paper_${DS}_M0_eval.csv
fail=0
NAMES=(${DS}_${ARM}_s0 ${DS}_${ARM}_s1 ${DS}_${ARM}_pooled)
for n in "${NAMES[@]}"; do
  python "$SRC/analyze_episodes.py" --mode convert-one --input-file "$BASE/csv/$n.csv" --output-dir "$RUN/csv" \
    > "$RUN/logs/convert_$n.log" 2>&1 || { echo "convert failed: $n"; fail=1; }
  python "$MWR_ROOT/analysis/check_readable_roundtrip.py" --raw "$BASE/csv/$n.csv" \
    --readable "$RUN/csv/${n}_readable.csv" --paper "$PAPER" "${PF[@]}" \
    > "$RUN/logs/roundtrip_$n.log" 2>&1 || { echo "roundtrip failed: $n"; fail=1; }
  tail -1 "$RUN/logs/roundtrip_$n.log" | cut -c1-100
done
if [ "$fail" -ne 0 ]; then echo "readable analysis stopped: fail=1 $(date -Is)"; exit 1; fi
if [ -z "${SKIP_BLOCKS:-}" ]; then
  for C in "$PAPER" "$RUN/csv/${NAMES[0]}_readable.csv" "$RUN/csv/${NAMES[1]}_readable.csv" "$RUN/csv/${NAMES[2]}_readable.csv"; do
    n=$(basename "$C" .csv)
    python "$SRC/analyze_blocks.py" --csv-name "$C" "${PF[@]}" --output-root "$RUN/blocks" \
      > "$RUN/logs/blocks_$n.log" 2>&1 || { echo "analyze_blocks failed: $n"; fail=1; }
  done
fi
for n in "${NAMES[@]}"; do
  python "$MWR_ROOT/analysis/episode_metrics.py" --csv-a "$RUN/csv/${n}_readable.csv" --csv-b "$PAPER" "${PF[@]}" \
    --out "$RUN/episode_metrics_${n}_vs_paper.json" > "$RUN/logs/metrics_$n.log" 2>&1 \
    || { echo "episode metrics failed: $n"; fail=1; }
done
echo "readable analysis done fail=$fail $(date -Is)"
exit $fail
