#!/bin/bash
#SBATCH -J mwr-smoke
#SBATCH -A mishra-sl3-cpu
#SBATCH -p icelake
#SBATCH -N1 -n1 -c28
#SBATCH -t 03:00:00
#SBATCH --array=0-3
#SBATCH -o /rds/user/da579/hpc-work/MathWorld_Reviews/runs/logs/smoke_%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=da579@cam.ac.uk
set -uo pipefail
. /rds/user/da579/hpc-work/MathWorld_Reviews/env/env.sh
PROFILES=(D0 D1 D2 D3)
P=${PROFILES[$SLURM_ARRAY_TASK_ID]}
export CONJ_REFUTE_DATASET_PROFILE=$P
TAG=${SMOKE_TAG:-smoke}
RUN_ID=rev_${TAG}_${P}_s0
RUN=$MWR_ROOT/runs/$TAG/$RUN_ID
NUM_ENVS=6; MAX_STEPS=72; EVAL_STEPS=5
WMODE=offline; [ "$P" = D0 ] && WMODE=online
. "$MWR_ROOT/scripts/stage.sh"
cd "$STAGE/work"

/usr/bin/time -v python "$STAGE/src/conj_refute_env/train_maddpg.py" \
  --run-id "$RUN_ID" --model-profile M0 --num-envs $NUM_ENVS --max-steps $MAX_STEPS --train-block 50 \
  --seed 0 --save-dir "$RUN/ckpt" --interim-save-dir "$RUN/interim" --config-dir "$RUN/run_configs" \
  --records-file "$RUN/train_records.jsonl" --wandb-mode $WMODE --skip-cleanup > "$RUN/train.log" 2>&1
TR=$?; echo "train exit $TR $(date -Is)"
du -sh outputs 2>/dev/null; ls "$STAGE/src/conj_refute_env/manifold_data" | grep -c '^blk_'

/usr/bin/time -v python "$STAGE/src/conj_refute_env/pickle_my_model.py" \
  --run-id "$RUN_ID" --model-profile M0 --model-dir "$RUN/ckpt" --config-dir "$RUN/run_configs" \
  --config-id base_run_config --episodes 1 --max-steps $EVAL_STEPS --seed 0 \
  --output-file "$RUN/eval.pkl" --wandb-mode $WMODE --skip-cleanup > "$RUN/eval.log" 2>&1
EV=$?; echo "eval exit $EV $(date -Is)"

python "$MWR_ROOT/scripts/smoke_check.py" --run-dir "$RUN" --train-rc $TR --eval-rc $EV \
  --num-envs $NUM_ENVS --max-steps $MAX_STEPS --eval-steps $EVAL_STEPS
