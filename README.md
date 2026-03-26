# MathWorld
A multi agent reinforcement learning architecture for discovering mathematical concepts: [arXiv](https://arxiv.org/pdf/2603.04528)

## Installation
### 1. Install Poetry and elan
```bash
curl -sSL https://install.python-poetry.org | python3 -
```
Add it to your PATH, then verify:
```bash
poetry --version
```
Next install elan:

```bash
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
```
Restart the terminal and verify with
```bash
lake --version
```

### 2. Build the python and lean projects
Navigate to the root of the python project and install:
```bash
cd conj-refute
poetry install --no-root
```

Navigate to the root of the lean project and install:
```bash
cd ../formal_response
lake build
```

### 3. Verify
Navigate to the python project and run
```bash
poetry run pytest -q
```

### 4. Workflow

Set two experiment selectors before running:

1. Dataset profile via `CONJ_REFUTE_DATASET_PROFILE`: `D0`, `D1`, `D2`, `D3`
2. Model profile via `--model-profile`: `M0`, `M1`, `M0_NOISE`, `M2`

- `M0`: Full system (formal response enabled, no Gaussian noise, skeptic active)
- `M1`: No formal response
- `M0_NOISE`: Full system with Gaussian noise in formal response
- `M2`: No skeptic (all datapoints have weight = `1.0` throughout)

More details on the model profiles can be found in the paper.

If you pass both a model profile and explicit manual flags (for example
`--formal-response-enabled`), the explicit manual flags override the profile defaults.

### Training example (Poetry)

```bash
cd conj-refute
export CONJ_REFUTE_DATASET_PROFILE=D1
poetry run python src/conj_refute_env/train_maddpg.py \
	--run-id demo_m0 \
	--model-profile M0 \
	--max-steps 50 \
	--wandb-mode disabled
```

### Evaluation example (Poetry)

```bash
cd conj-refute
export CONJ_REFUTE_DATASET_PROFILE=D1
poetry run python src/conj_refute_env/pickle_my_model.py \
	--run-id demo_m0 \
	--model-profile M0 \
	--episodes 1 \
	--max-steps 10 \
	--wandb-mode disabled
```

# Artifacts
## Contents

The code for generating the datasets can be found at: [surface_triangulations](https://github.com/daattavya98/surface_triangulations)

All artifacts can be found at: [artifacts](https://zenodo.org/records/19219927).

This includes checkpoints for trained models, evaluation results and a collection of formalized conjectures along with a suggested proof. These are not required to run the code but are provided to cross verify with the paper.
