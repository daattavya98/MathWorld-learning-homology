#!/bin/bash
# sbatch --array=0-3 --export=ALL,WAVE=sweep1,ARM=c00_centre scripts/analysis.sbatch   (index = D0..D3)
#SBATCH -J mwr-anal
#SBATCH -A mishra-sl3-cpu
#SBATCH -p icelake
#SBATCH -N1 -n1 -c4
#SBATCH -t 03:00:00
#SBATCH -o /rds/user/da579/hpc-work/MathWorld_Reviews/runs/logs/analysis_%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=da579@cam.ac.uk
set -uo pipefail
. /rds/user/da579/hpc-work/MathWorld_Reviews/env/env.sh
DSS=(D0 D1 D2 D3)
DS=${DSS[$SLURM_ARRAY_TASK_ID]}
export CONJ_REFUTE_DATASET_PROFILE=$DS
SEEDS=(${SEED_LIST:-0 1})
RUN=$MWR_ROOT/runs/$WAVE/analysis/$ARM/$DS
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
fail=0

PAPER=$RUN/csv/paper_${DS}_M0_eval.csv
cp "$MWR_ROOT/data_src/zenodo_eval/evaluations/Fig 5/${DS}_M0_eval.csv" "$PAPER"
CSVS=("$PAPER")
for s in "${SEEDS[@]}"; do
  R=$MWR_ROOT/runs/$WAVE/rev_${WAVE}_${ARM}_${DS}_s${s}
  if [ ! -f "$R/eval_e10_s50.pkl" ]; then
    python "$MWR_ROOT/scripts/merge_eval_parts.py" "$R" 10 "$R/eval_e10_s50.pkl" || { echo "merge failed s$s"; fail=1; }
  fi
  python "$MWR_ROOT/scripts/upload_eval_wandb.py" --run-dir "$R" --arm "$ARM" --dataset "$DS" --seed "$s" \
    > "$RUN/logs/wandb_s${s}.log" 2>&1 || { echo "wandb upload failed s$s"; fail=1; }
  tail -1 "$RUN/logs/wandb_s${s}.log" | cut -c1-140
  C=$RUN/csv/${DS}_${ARM}_s${s}.csv
  python "$SRC/conj_refute_env/depickle_and_table.py" --input-file "$R/eval_e10_s50.pkl" --output-csv "$C" > /dev/null || fail=1
  CSVS+=("$C")
done
POOLED=$RUN/csv/${DS}_${ARM}_pooled.csv
python - "$POOLED" "${CSVS[@]:1}" <<'PY'
import sys
import pandas as pd
frames = []
for i, p in enumerate(sys.argv[2:]):
    d = pd.read_csv(p)
    d["episode"] = d["episode"] + 1000 * i
    frames.append(d)
pd.concat(frames, ignore_index=True).to_csv(sys.argv[1], index=False)
PY
CSVS+=("$POOLED")

if [ "${SKIP_BLOCKS:-0}" != 1 ]; then
  for C in "${CSVS[@]:1}"; do
    n=$(basename "$C" .csv)
    echo "analyze_blocks $n $(date -Is)"
    python "$SRC/analyze_blocks.py" --csv-name "$C" "${PF[@]}" --output-root "$RUN/blocks" \
      > "$RUN/logs/blocks_${n}.log" 2>&1 || { echo "analyze_blocks failed: $n"; fail=1; }
  done
fi
for C in "${CSVS[@]:1}"; do
  n=$(basename "$C" .csv)
  echo "episode metrics $n $(date -Is)"
  python "$MWR_ROOT/analysis/episode_metrics.py" --csv-a "$C" "${PF[@]}" \
    --out "$RUN/episode_metrics_${n}.json" > "$RUN/logs/metrics_${n}.log" 2>&1 \
    || { echo "episode metrics failed: $n"; fail=1; }
done
echo "analysis done fail=$fail $(date -Is)"
exit $fail
