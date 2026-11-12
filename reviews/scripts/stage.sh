# Source after env/env.sh inside a job. Needs RUN (per-task output dir on /rds).
STAGE=/dev/shm/da579_mwr_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$STAGE/work" "$RUN"
rsync -a --exclude __pycache__ --exclude 'manifold_data/blk_*' "$MWR_SRC/" "$STAGE/src/"
export PYTHONPATH=$STAGE/src
export WANDB_DIR=$RUN
git -C "$MWR_CLONE" rev-parse HEAD > "$RUN/code.rev"
git -C "$MWR_CLONE" diff HEAD > "$RUN/code.diff"
git -C "$MWR_CLONE" status --porcelain > "$RUN/code.status"
{ echo "node=$(hostname) job=${SLURM_JOB_ID:-} task=${SLURM_ARRAY_TASK_ID:-} cpus=${SLURM_CPUS_PER_TASK:-}"
  echo "prover=$CONJ_REFUTE_PROVER julia_threads=$PYTHON_JULIACALL_THREADS profile=${CONJ_REFUTE_DATASET_PROFILE:-}"
  echo "start=$(date -Is)"; } > "$RUN/job.meta"

(
  while sleep 300; do
    find "$STAGE/work/outputs" -mindepth 1 -maxdepth 1 -mmin +20 -exec rm -rf {} + 2>/dev/null
    find "$STAGE/src/conj_refute_env/manifold_data" -maxdepth 1 -name 'blk_*' -mmin +60 -delete 2>/dev/null
  done
) &
JANITOR=$!
stage_cleanup() { kill "$JANITOR" 2>/dev/null; du -sh "$STAGE" 2>/dev/null; rm -rf "$STAGE"; }
trap stage_cleanup EXIT

t0=$(date +%s)
find "$MWR_LEAN/.lake/packages" -path '*/.lake/build/lib/*' -type f -print0 | xargs -0 -P 8 -n 64 cat > /dev/null
echo "lean page-cache warm $(( $(date +%s) - t0 ))s"
t0=$(date +%s)
if (cd "$MWR_LEAN" && timeout 900 lake build --no-build Mathlib.LinearAlgebra.FiniteDimensional.Defs > /dev/null \
    && timeout 900 lake env lean template.lean); then
  echo "lean warm ok $(( $(date +%s) - t0 ))s"
else
  echo "lean warm FAILED (stale build or broken toolchain); aborting"; exit 3
fi
