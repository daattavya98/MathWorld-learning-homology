# Source before any job step: . /rds/user/da579/hpc-work/MathWorld_Reviews/env/env.sh
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-icl gcc/11 ceuadmin/uv/0.12.5

export MWR_ROOT=/rds/user/da579/hpc-work/MathWorld_Reviews
export MWR_CLONE=$MWR_ROOT/MathWorld-learning-homology
export MWR_SRC=$MWR_CLONE/conj-refute/src
export MWR_LEAN=$MWR_CLONE/formal_response_grind

export VIRTUAL_ENV=$MWR_ROOT/.venv
export PATH=$VIRTUAL_ENV/bin:$PATH
export PYTHONPATH=$MWR_SRC
export UV_CACHE_DIR=$MWR_ROOT/.uv_cache
export UV_PYTHON_INSTALL_DIR=$MWR_ROOT/.uv_python

export JULIA_DEPOT_PATH=$MWR_ROOT/julia_depot
export PYTHON_JULIAPKG_OFFLINE=${PYTHON_JULIAPKG_OFFLINE:-yes}
export PYTHON_JULIACALL_THREADS=${PYTHON_JULIACALL_THREADS:-4}

export ELAN_HOME=/rds/user/da579/hpc-work/elan
export PATH=$ELAN_HOME/bin:$PATH
export MATHLIB_NO_CACHE_ON_UPDATE=1

export CONJ_REFUTE_PROVER=${CONJ_REFUTE_PROVER:-grind}
export CONJ_REFUTE_FORMAL_RESPONSE_DIR=$MWR_LEAN
export CONJ_REFUTE_LEAN_INITIAL_BUILD=0

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_ENTITY=science_ai WANDB_PROJECT=conj_refute_maddpg
if [ -z "${WANDB_API_KEY:-}" ] && [ -f /rds/user/da579/hpc-work/MathWorld/.env ]; then
  export WANDB_API_KEY=$(grep -m1 '^WANDB_API_KEY=' /rds/user/da579/hpc-work/MathWorld/.env | cut -d= -f2- | tr -d "\"'")
fi
