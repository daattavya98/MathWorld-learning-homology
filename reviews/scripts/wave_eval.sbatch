#!/bin/bash
# sbatch --array=0-N[%K] -t WALL --dependency=aftercorr:<train array> --export=ALL,WAVE=<wave> scripts/wave_eval.sbatch
# EPISODES single-episode processes, EVAL_PAR at a time (PYTHON_JULIACALL_THREADS each), merged into one pickle.
#SBATCH -J mwr-eval
#SBATCH -A mishra-sl3-cpu
#SBATCH -p icelake
#SBATCH -N1 -n1 -c20
#SBATCH -o /rds/user/da579/hpc-work/MathWorld_Reviews/runs/logs/eval_%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=da579@cam.ac.uk
set -uo pipefail
. /rds/user/da579/hpc-work/MathWorld_Reviews/env/env.sh
EPISODES=${EPISODES:-10}; EVAL_STEPS=${EVAL_STEPS:-50}; EVAL_PAR=${EVAL_PAR:-5}
TASKS=$MWR_ROOT/runs/$WAVE/tasks.tsv
IFS=$'\t' read -r IDX ARM DS SEED FLAGS < <(awk -F'\t' -v i="$SLURM_ARRAY_TASK_ID" 'NR>1 && $1==i' "$TASKS")
[ "$IDX" = "$SLURM_ARRAY_TASK_ID" ] || { echo "no task row $SLURM_ARRAY_TASK_ID"; exit 2; }
export CONJ_REFUTE_DATASET_PROFILE=$DS
RUN_ID=rev_${WAVE}_${ARM}_${DS}_s${SEED}
RUN=$MWR_ROOT/runs/$WAVE/$RUN_ID
OUT=$RUN/eval_e${EPISODES}_s${EVAL_STEPS}.pkl
[ -f "$RUN/ckpt/${RUN_ID}_marl.pt" ] || { echo "no checkpoint for $RUN_ID"; exit 4; }
if [ -f "$OUT" ]; then echo "done already: $OUT"; exit 0; fi
. "$MWR_ROOT/scripts/stage.sh"
mkdir -p "$RUN/eval_parts"
echo "eval task $IDX $RUN_ID episodes=$EPISODES steps=$EVAL_STEPS par=$EVAL_PAR start $(date -Is)"
cd "$STAGE/work"
rc=0; active=0
for ep in $(seq 0 $((EPISODES - 1))); do
  PART=$RUN/eval_parts/ep${ep}.pkl
  [ -f "$PART" ] && continue
  if [ "$active" -ge "$EVAL_PAR" ]; then wait -n || rc=1; active=$((active - 1)); fi
  /usr/bin/time -v python "$STAGE/src/conj_refute_env/pickle_my_model.py" \
    --run-id "$RUN_ID" --model-profile M0 --model-dir "$RUN/ckpt" --config-dir "$RUN/run_configs" \
    --config-id base_run_config --episodes 1 --max-steps "$EVAL_STEPS" \
    --seed $(( 1000 + SEED * 100 + ep )) --output-file "$PART" --wandb-mode "${WMODE:-disabled}" \
    --skip-cleanup > "$RUN/eval_parts/ep${ep}.log" 2>&1 &
  active=$((active + 1))
done
while [ "$active" -gt 0 ]; do wait -n || rc=1; active=$((active - 1)); done
echo "episodes finished rc=$rc $(date -Is)"
python - "$RUN" "$EPISODES" "$OUT" <<'PY'
import pickle, sys
from pathlib import Path
run, n, out = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
recs, missing = [], []
for ep in range(n):
    p = run / "eval_parts" / f"ep{ep}.pkl"
    if not p.is_file():
        missing.append(ep)
        continue
    for r in pickle.loads(p.read_bytes()):
        r["episode"] = ep
        recs.append(r)
if missing:
    sys.exit(f"missing episodes {missing}; not merging")
Path(out).write_bytes(pickle.dumps(recs))
print(f"merged {len(recs)} records from {n} episodes -> {out}")
PY
merge_rc=$?
if [ "$merge_rc" -eq 0 ]; then
  python "$MWR_ROOT/scripts/upload_eval_wandb.py" --run-dir "$RUN" --pickle "$(basename "$OUT")" \
    --arm "$ARM" --dataset "$DS" --seed "$SEED" 2>&1 | tail -1 | cut -c1-140
fi
exit $(( rc || merge_rc ))
