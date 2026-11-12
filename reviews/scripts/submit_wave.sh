#!/bin/bash
# WAVE=sweep1 TRAIN_WALL=10:00:00 EVAL_WALL=04:00:00 SEEDS=0,1 [DATASETS=D0,D1,D2,D3 ONLY_ARMS=..] scripts/submit_wave.sh
# Two sbatch calls, 130 s apart: ONE train array, ONE eval array chained per index with aftercorr.
set -uo pipefail
. /rds/user/da579/hpc-work/MathWorld_Reviews/env/env.sh
set -e
: "${WAVE:?}" "${TRAIN_WALL:?}" "${EVAL_WALL:?}" "${SEEDS:?}"
ARGS=(--wave "$WAVE" --seeds "$SEEDS" --datasets "${DATASETS:-D0,D1,D2,D3}")
[ -n "${ONLY_ARMS:-}" ] && ARGS+=(--only-arms "$ONLY_ARMS")
python "$MWR_ROOT/scripts/make_wave.py" "${ARGS[@]}"
N=$(( $(wc -l < "$MWR_ROOT/runs/$WAVE/tasks.tsv") - 2 ))
RANGE=${ARRAY:-0-$N}
TJ=$(sbatch --parsable --array=${RANGE}%${TRAIN_MAX:-14} -t "$TRAIN_WALL" --export=ALL,WAVE="$WAVE" \
     "$MWR_ROOT/scripts/wave_train.sbatch")
echo "train array $TJ ($RANGE) wall=$TRAIN_WALL $(date -Is)" | tee -a "$MWR_ROOT/runs/$WAVE/submitted.txt"
sleep 130
EJ=$(sbatch --parsable --array=${RANGE}%${EVAL_MAX:-14} -t "$EVAL_WALL" --dependency=aftercorr:"$TJ" \
     --export=ALL,WAVE="$WAVE" "$MWR_ROOT/scripts/wave_eval.sbatch")
echo "eval array $EJ ($RANGE, aftercorr:$TJ) wall=$EVAL_WALL $(date -Is)" | tee -a "$MWR_ROOT/runs/$WAVE/submitted.txt"
