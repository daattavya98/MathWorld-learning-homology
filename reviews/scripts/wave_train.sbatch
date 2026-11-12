#!/bin/bash
# sbatch --array=0-N[%K] -t WALL --export=ALL,WAVE=<wave> scripts/wave_train.sbatch
#SBATCH -J mwr-train
#SBATCH -A mishra-sl3-cpu
#SBATCH -p icelake
#SBATCH -N1 -n1 -c28
#SBATCH -o /rds/user/da579/hpc-work/MathWorld_Reviews/runs/logs/train_%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=da579@cam.ac.uk
set -uo pipefail
. /rds/user/da579/hpc-work/MathWorld_Reviews/env/env.sh
TASKS=$MWR_ROOT/runs/$WAVE/tasks.tsv
IFS=$'\t' read -r IDX ARM DS SEED FLAGS < <(awk -F'\t' -v i="$SLURM_ARRAY_TASK_ID" 'NR>1 && $1==i' "$TASKS")
[ "$IDX" = "$SLURM_ARRAY_TASK_ID" ] || { echo "no task row $SLURM_ARRAY_TASK_ID"; exit 2; }
export CONJ_REFUTE_DATASET_PROFILE=$DS
RUN_ID=rev_${WAVE}_${ARM}_${DS}_s${SEED}
RUN=$MWR_ROOT/runs/$WAVE/$RUN_ID
if [ -f "$RUN/ckpt/${RUN_ID}_marl.pt" ]; then echo "done already: $RUN_ID"; exit 0; fi
. "$MWR_ROOT/scripts/stage.sh"
cd "$STAGE/work"
echo "task $IDX arm=$ARM ds=$DS seed=$SEED flags=[$FLAGS] start $(date -Is)"
# shellcheck disable=SC2086
/usr/bin/time -v python "$STAGE/src/conj_refute_env/train_maddpg.py" \
  --run-id "$RUN_ID" --model-profile M0 --num-envs 6 --max-steps "${MAX_STEPS:-600}" --train-block 50 \
  --seed "$SEED" --save-dir "$RUN/ckpt" --interim-save-dir "$RUN/interim" --config-dir "$RUN/run_configs" \
  --records-file "$RUN/train_records.jsonl" --wandb-mode "${WMODE:-online}" --skip-cleanup $FLAGS \
  > "$RUN/train.log" 2>&1
rc=$?
echo "train exit $rc $(date -Is)"
exit $rc
