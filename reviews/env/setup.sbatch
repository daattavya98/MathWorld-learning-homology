#!/bin/bash
#SBATCH -J mwr-setup
#SBATCH -A mishra-sl3-cpu
#SBATCH -p icelake
#SBATCH -N1 -n1 -c32
#SBATCH -t 02:30:00
#SBATCH -o /rds/user/da579/hpc-work/MathWorld_Reviews/runs/logs/setup_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=da579@cam.ac.uk
set -uo pipefail
PYTHON_JULIAPKG_OFFLINE=no
. /rds/user/da579/hpc-work/MathWorld_Reviews/env/env.sh
export TMPDIR=/dev/shm/da579_mwr_${SLURM_JOB_ID:-$$}; mkdir -p "$TMPDIR"
echo "node $(hostname) cores $(nproc) start $(date)"

venv_step() {
  set -e
  [ -x "$VIRTUAL_ENV/bin/python" ] || uv venv --python 3.11 "$VIRTUAL_ENV"
  uv pip install --python "$VIRTUAL_ENV/bin/python" -c "$MWR_ROOT/env/constraints.txt" \
    "torch==2.5.1+cpu" --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
  uv pip install --python "$VIRTUAL_ENV/bin/python" -c "$MWR_ROOT/env/constraints.txt" \
    sympy scipy matplotlib numpy pettingzoo beartype isort pysr toponetx networkx pandas \
    tensorboard leanclient agilerl debugpy tabulate ipdb pytest ruff wandb
  uv pip list --python "$VIRTUAL_ENV/bin/python" 2>/dev/null \
    | grep -iE "^(agilerl|pysr|torch|juliacall|juliapkg|numpy|pettingzoo|leanclient|toponetx|wandb|gymnasium|pandas) "
  bash "$MWR_ROOT/env/julia_setup.sh"
}

lean_step() {
  set -e
  elan toolchain list | grep -q "v4.24.0" || elan toolchain install leanprover/lean4:v4.24.0
  cd "$MWR_LEAN"
  lake --version
  echo "=== resolve deps from manifest + build Mathlib closure of the template import"
  timeout 7200 lake build Mathlib.LinearAlgebra.FiniteDimensional.Defs 2>&1 | tail -25
  lake build 2>&1 | tail -5
  echo "=== header elaborates"
  time lake env lean template.lean
  echo "oleans: $(find .lake/packages/mathlib/.lake/build -name '*.olean' | wc -l)"
  touch "$MWR_LEAN/.lake/.mwr_ready"
}

mkdir -p "$MWR_ROOT/runs/logs"
if [ -f "$MWR_ROOT/.venv/.mwr_ready" ]; then echo "venv ready, skip"; V=0; else
  ( venv_step ) > "$MWR_ROOT/runs/logs/setup_venv_${SLURM_JOB_ID}.log" 2>&1 & VP=$!; fi
if [ -f "$MWR_LEAN/.lake/.mwr_ready" ]; then echo "lean ready, skip"; L=0; else
  ( lean_step ) 2>&1; L=$?; echo "lean_step exit $L"; fi
if [ -n "${VP:-}" ]; then wait $VP; V=$?; echo "venv_step exit $V"; cat "$MWR_ROOT/runs/logs/setup_venv_${SLURM_JOB_ID}.log" | tail -40; fi
rm -rf "$TMPDIR"
echo "finished $(date)  venv=$V lean=$L"
[ "$V" -eq 0 ] && [ "$L" -eq 0 ] && echo "SETUP OK"
