# MathWorld_Reviews — handoff (written 2026-11-06, from a session run in /rds/user/da579/hpc-work/MathWorld)

## 1. Goal
The user has received reviews of the MathWorld paper (arXiv:2603.04528, "Discovering mathematical concepts
through a multi-agent system"). This directory exists to run **a few small, new experiments** for the response.
- **Hyperparameter sensitivity.** Sweep the **RL optimiser** settings (lr_actor, lr_critic, gamma, tau,
  batch size) and the **exploration noise** settings (OU noise on/off, expl_noise), across **ALL dataset
  profiles D0–D3**.
- **A "direct LLM" baseline.** The user will design the protocol later. For now, build the capability to test
  **Claude, GPT and DeepSeek/Qwen models through their APIs** (no weight downloads).

Scope rule from the user: bring over the generic MathWorld learnings and analysis tooling, **and none of the
knot-specific material** (no hyperbolic volume, Chern–Simons, sigma/mirror, Lobachevsky, tetrahedra, knot
datasets).

## 2. What exists on disk right now
```
/rds/user/da579/hpc-work/MathWorld_Reviews/
  handoff.md                      <- this file
  MathWorld-learning-homology/    <- git clone of github.com/daattavya98/MathWorld-learning-homology
                                     at ad4c4d7 (3 commits, 35 files). UNTOUCHED: no branch, no edits.
```
Nothing else has been done yet:
- no Python environment
- no datasets copied in
- no Lean build
- no Zenodo download
- no jobs submitted

## 3. Decisions the user has made (do not re-ask)
| Topic | Decision |
|---|---|
| Datasets | **Map from the public `surface_triangulations` repo.** The mapping was verified on the data itself; see §5.9. |
| Lean prover | **Use the `grind` fallback, without LeanCopilot.** The user accepts that this makes results not directly comparable to the paper's LeanCopilot numbers, so every comparison must stay **within the grind prover**, including a re-run of the default config as the reference point. |
| W&B | **Keep the paper's destination:** entity `science_ai`, project `conj_refute_maddpg`. Prefix review run names (e.g. `rev_…`). |
| Sweep groups | **RL optimiser + exploration noise only.** No reward weights; no search/data knobs. |
| Sweep centre | **What actually ran**: the AgileRL 2.2.8 constructor defaults (see §5.2), not the `INIT_HP` dict. Make every hyperparameter an explicit, logged CLI flag whose default equals what ran. The paper's HP table may need correcting; tell the user which entries. |
| Datasets for the sweep | **All types**, i.e. D0, D1, D2 and D3. |
| LLM baseline | **Protocol to be designed by the user later.** Build a provider-agnostic harness for Claude, GPT and DeepSeek/Qwen through APIs. |
| Sweep design (proposed; user did not object) | One-at-a-time around the centre (e.g. ×0.5 / ×2), profile **M0**, a few seeds, grind prover throughout. |

## 4. The paper codebase in one screen
- Python project `MathWorld-learning-homology/conj-refute/` uses **Poetry** and Python **3.11 only**. The
  package `conj_refute_env` lives in `src/`. The Lean project `MathWorld-learning-homology/formal_response/` uses
  Lean/Mathlib **v4.24.0** plus **LeanCopilot**.
- Agents: `conjecturer` and `refuter` (= the "skeptic"); `AsyncPettingZooVecEnv`, `context="spawn"`. Each step:
  - PySR atomic blocks are fitted over 10 blocks (`blk_1..blk_10`), followed by a scaffold GA (implies /
    conjunction / logical_neg).
  - The resulting statement goes through `translate_fol_to_lean` → Lean.
  - The formal response is 0/1.
- Model profiles (`src/conj_refute_env/cli_utils.py:4-25`):

  | Profile | Meaning |
  |---|---|
  | M0 | full system |
  | M1 | `formal_response_enabled=False` |
  | M0_NOISE | M0 plus Gaussian noise on the formal response |
  | M2 | `identity_refuter_enabled=True` (no skeptic) |

- Dataset profiles come from the env var `CONJ_REFUTE_DATASET_PROFILE=D0|D1|D2|D3` (`env/utils.py:63-92`). Each
  file contributes `CONJ_REFUTE_ROWS_PER_DATASET` rows (default 350, max 500), resampled at every `reset()`.
- Train: `python src/conj_refute_env/train_maddpg.py --run-id X --model-profile M0 [--num-envs 6 --max-steps 600 --train-block 50 --wandb-mode ...]`.
  It reads `--config-dir ./run_configs` / `--env-config base_run_config`, **relative to the cwd**. A run lock
  is written to `run_configs/` (`write_or_validate_run_lock`).
- Eval: `python src/conj_refute_env/pickle_my_model.py --run-id X --model-profile M0 --episodes 15 --max-steps 50 --output-file ...`.
  It writes a pickle of per-step records.
- Analysis already in the repo:
  - `src/analyze_episodes.py`: cluster bootstrap CIs, ratio CIs, σ-effective, chi/homology breadth and
    proven rates.
  - `src/analyze_blocks.py`: tiered detection of the Euler characteristic and Betti numbers via
    rank-nullity, baselines `baseline_alternating_sum` / `baseline_b1`, bootstrap diff CIs.
  - `src/analysis_utils.py`
  - `src/conj_refute_env/depickle_and_table.py`

  **MathWorld's `analyze_campaign.py` was itself modelled on `analyze_episodes.py`**, so the core statistics
  already exist here. Port only what is missing.

## 5. Findings (verified in this session; file refs are relative to `MathWorld-learning-homology/conj-refute/src/`)

### 5.1 Python and Julia environment
- `poetry.lock` pins:

  | Package | Version |
  |---|---|
  | agilerl | **2.2.8** |
  | pysr | **1.5.8** |
  | torch | 2.5.1 |
  | juliacall | 0.9.25 |
  | numpy | 1.26.4 |
  | pettingzoo | 1.25.0 |
  | leanclient | 0.1.14 |
  | toponetx | 0.2.0 |
  | wandb | 0.17.9 |
  | gymnasium | 1.2.0 |

- MathWorld runs pysr 1.5.10, agilerl 2.5.0 and torch 2.9.0. **Do NOT reuse MathWorld's `.venv` or the shared
  depot `/rds/user/da579/hpc-work/julia_depot_fleet`.** The SymbolicRegression.jl versions differ, and an
  online juliapkg resolve against a shared Manifest is exactly the hazard that broke the MathWorld fleet.
  Build a separate venv and give it its own `JULIA_DEPOT_PATH` (e.g. `MathWorld_Reviews/julia_depot`).
- Poetry is not on CSD3. Use `module load ceuadmin/uv/0.12.5` with Python 3.11 and install with `uv pip`,
  constrained to the poetry.lock versions (export every name/version pair from `poetry.lock` to a constraints
  file).
- Install the torch CPU wheel (`--index-url https://download.pytorch.org/whl/cpu`, still 2.5.1). The nodes are
  CPU-only, and the default wheel drags in GBs of CUDA libraries.
- Also run `module load gcc/11`, because Julia needs GLIBCXX_3.4.26.
- Run the install inside an sbatch job, not on the login node.
- `poetry install --no-root` means the package is not installed, so set `PYTHONPATH=<clone>/conj-refute/src`
  in the job scripts.

### 5.2 `INIT_HP` is never passed to MADDPG
`train_maddpg.py:51-64` defines `INIT_HP`, but `MADDPG(...)` at `:331-338` receives only spaces, ids,
`net_config`, device and `vect_noise_dim`. The effective values are the AgileRL 2.2.8 constructor defaults,
verified from the 2.2.8 wheel (`agilerl/algorithms/maddpg.py` `__init__`):

| knob | INIT_HP says | what actually ran |
|---|---|---|
| O_U_noise | False | **True** (theta 0.15, dt 1e-2, mean_noise 0.0) |
| expl_noise | 0.3 | **0.1** |
| batch_size | 32 | **64** (the loop gates on `agent.batch_size`, `:497`) |
| lr_actor / lr_critic | 1e-3 / 1e-2 | 1e-3 / 1e-2 (same) |
| gamma / tau | 0.95 / 0.01 | 0.95 / 0.01 (same) |
| LEARN_STEP 5 | "learn every 5" | used only as a warm-up (`memory.counter > 5`, `:356,497`); `agent.learn` then runs **every** vec-step |

`NET_CONFIG` (encoder [32, 32], head [32]) and `MEMORY_SIZE` 100k ARE used. The sweep centre is the right-hand
column (user decision). Tell the user which paper-table entries this affects (OU noise, expl_noise, batch size).

### 5.3 MathWorld's replay-buffer bug (#9) does NOT apply here
In AgileRL 2.2.8, `DeterministicActor.forward` rescales to action-space units (`networks/actors.py:161-174`),
and `learn()` calls `actor(states)` in those same units. Storing `cont_actions` (env units) in the buffer is
therefore consistent. **Do not "fix" it.**

### 5.4 The grind prover path
- What exists:
  - `env/conjecture_to_lean_statement.py:250` `translate_fol_to_lean_with_grind`, which replaces
    `search_proof` with `simp_all [sub_eq_add_neg] <;> ring_nf at * <;> grind`
  - `env/check_lean_theorem.py:247` `check_theorem_no_leancopilot`
- What has to change:
  - The env hard-wires the LeanCopilot pair: `env/conj_refute_env.py:16,22` imports `translate_fol_to_lean`
    and `check_theorem`, and `compute_reward_terms` (`:268-325`) calls them.
  - `formal_response/template.lean` does `import LeanCopilot`.
  - `formal_response/lakefile.toml` requires LeanCopilot and adds `moreLinkArgs` for ctranslate2.
  - Plan: add a backend switch (config key or env var, default = the paper's LeanCopilot behaviour), a
    grind template without `import LeanCopilot`, and a lakefile variant without LeanCopilot.
- **Correctness hazard to calibrate before ANY wave.** `check_theorem_no_leancopilot` returns 1 whenever no
  severity-1 / "unsolved goals" diagnostic shows up within `TIMEOUT = 15 s` (`check_lean_theorem.py:16,
  310-324`). A successful proof emits no diagnostic either. If leanclient's `get_diagnostics()` does not
  block until elaboration finishes, then **slow equals "proved"** (a false positive).
  - Write a calibration script that exits nonzero unless known truths score 1 and known falsehoods / type
    errors score 0, and that times each check. Known truths: Euler characteristic
    `h1 - w1 + w2 = …` / rank-nullity Betti statements from `analysis_utils.py`.
  - The MathWorld rule is strict-equality hits backed by a calibration script that fails loudly.
- Cold start: `fc.open_file(timeout=15)` on a cold Mathlib import will time out. The exception is caught and
  scored 0, so early steps get silently lost. MathWorld's fix is to warm the page cache once with a single
  import before any worker starts.
- Cache: the env calls `check_theorem(..., use_cache=False)`, so no proof cache is read. The LeanCopilot path
  appends to `formal_response/proof_cache_final.jsonl` (override with `CONJ_REFUTE_PROOF_CACHE_FINAL_PATH`);
  the grind path writes nothing when `use_cache=False`. The Zenodo `proof_cache_mega.json` holds
  **LeanCopilot verdicts**. Never feed it to grind runs.

### 5.5 Lean on CSD3 (from the MathWorld experience)
- elan is installed at `/rds/user/da579/hpc-work/elan`, and `~/.elan` is a symlink to it. The only toolchain
  built so far is v4.34; add v4.24 with `elan toolchain install leanprover/lean4:v4.24.0` inside an sbatch job
  (compute nodes DO have outbound network).
- Lean's bundled clang needs GLIBC_2.29 and RHEL8 has 2.28, so **native targets fail**. Pure `.olean`
  elaboration is fine. For anything native, use the `LEAN_CC` shim `/rds/user/da579/hpc-work/MathWorld/scratch/lean_cc.sh`
  (see `scratch/lean_env.sh` there), and set `TMPDIR=/dev/shm/...`.
- `lake exe cache get` is broken here: the cache CLI is native, and curl fails with exit 35. **Build Mathlib
  v4.24 from source** in sbatch. In MathWorld that took about 35 min on 32 cores for about 2k modules. The
  grind template imports only `Mathlib.LinearAlgebra.FiniteDimensional.Defs`; check that `ring_nf` / `simp_all`
  resolve under that import.
- **Never run a bare `lake update`.** Mathlib's post_update hook runs `cache get` and burns an hour. Use
  `lake update <pkg>` with `MATHLIB_NO_CACHE_ON_UPDATE=1`, and back up `lake-manifest.json` first.
- Never build or check Lean on a login node.

### 5.6 Concurrency hazards (many jobs in ONE checkout clobber each other)
- Each step writes `blk_conjectures_{run}_{env}_{t}.csv` into `conj_refute_env/manifold_data/`, the same
  directory the datasets live in (`conj_refute_env.py:994-999`).
- At the end of training, `clean_extra_datafiles` deletes **every** `blk_*` there, `clean_extra_leanfiles`
  deletes **every** `formal_response/lean_files/conjecture_*`, and `manage_caches` merges and deletes the temp
  caches (`train_maddpg.py:597-605`, `env/utils.py:387-460`). Array tasks must pass `--skip-cleanup` or use a
  per-task staged copy (MathWorld stages per-job scratch on /dev/shm).
- `pickle_my_model.py` defaults to `--output-file step_data.pkl`, so concurrent evals overwrite each other.
  Always pass a per-run path.
- `--config-id` defaults to `form_response_official`, **which is not in the repo**. Pass
  `--config-id base_run_config`.
- `run_configs/` and its run-lock files are cwd-relative.

### 5.7 Seeding
There is **no `--seed`** in `train_maddpg.py` or `pickle_my_model.py`. `env.reset(seed=None)`
(`conj_refute_env.py:642-664`) resamples dataset rows with the unseeded global `np.random`. For seed
replicates as ONE SLURM array (index = seed), add a seed flag that seeds numpy, torch, Python `random`, the
env resets and the PySR `random_state`, if PySR runs deterministic. Without one, replicates are only
"independent draws".

### 5.8 Other config facts
- `src/base_run_config.json` and `src/run_configs/base_run_config.json` differ in whitespace only.
- Reward weights: `alpha_conj` 150, `beta_conj` 3, `beta_ref` 3, `gamma_ref` 0.5, `repeat_penalty` 0.3,
  `uniform_conj` 0.8, `formal_response_enabled` true.
- Env defaults when a key is absent: `alpha_ref` 5, `max_data_weight` 5, env `gamma` 0.99, `no_of_blocks` 10
  (`conj_refute_env.py:393-406`).
- PySR runs with `niterations` 50 (`env/ga_formula.py:328,376`).
- maxsize is clipped to [7, 30] (`conj_refute_env.py:902-915`).
- W&B is hard-coded in `train_maddpg.py:281-287` (`science_ai` / `conj_refute_maddpg`, mode from
  `--wandb-mode` or `WANDB_MODE`, default online). Not yet verified: whether the CSD3 W&B login has write
  access to the `science_ai` entity. Smoke-test it. `wandb.init` is not wrapped in try/except here, so a
  failure is loud, which is good.

### 5.9 Datasets: verified mapping from `surface_triangulations`
Source: `https://raw.githubusercontent.com/daattavya98/surface_triangulations/HEAD/surface_triangulations/data_gen/incidence_matrix_dataframes/<file>`.
Target: `MathWorld-learning-homology/conj-refute/src/conj_refute_env/manifold_data/<target>` (gitignored; the
loader resolves `project_root/manifold_data`).

Verification: for every file, V = height_δ1, E = width_δ1, F = width_δ2, χ = V − E + F, b0 = V − rank_δ1,
b2 = nullity_δ2. **Every row of every file has exactly the expected signature:**

| target (env name) | source file | rows | (χ, b0, b2) |
|---|---|---|---|
| sphere_data.csv | sphere_data.csv | 566 | (2,1,1) |
| torus_data.csv | torus_data.csv | 573 | (0,1,1) |
| klein_data.csv | klein_bottle_data.csv | 500 | (0,1,0) |
| sphere_plus_sphere_data.csv | genus_0_plus_0_dataset_combined_final.csv | 526 | (4,2,2) |
| torus_plus_sphere_data.csv | genus_1_plus_0_dataset_combined.csv | 523 | (2,2,2) |
| torus_plus_torus_data.csv | genus_1_plus_1_dataset_combined.csv | 548 | (0,2,2) |
| klein_plus_klein_data.csv | klein_plus_klein_dataset.csv | 500 | (0,2,0) |
| klein_plus_torus_data.csv | klein_plus_torus_dataset.csv | 514 | (0,2,1) |
| klein_plus_sphere_data.csv | klein_plus_sphere_dataset.csv | 513 | (2,2,1) |
| klein_plus_torus_plus_sphere_data.csv | klein_plus_torus_plus_sphere_data.csv | 510 | (2,3,2) |

- Unused: `genus_0_plus_1_dataset_cleaned` (75 rows, too few), `genus_2_dataset` (225 rows), `sphere_dataset` /
  `torus_dataset` (216 / 225 rows), the `extra_*` files and every `*_chain_complex_violated*` file (the public
  workflow refuses those, `env/utils.py:184-194`).
- The loader keeps only the 8 `width/height/rank/nullity_δ{1,2}` columns, so extra columns (b0/b1/b2,
  `Unnamed: 0`) do no harm.
- Caveat to state in the paper response if asked: these are the public files. They have not been checked to
  be byte-identical to the paper's local `manifold_data`.

### 5.10 Zenodo artifacts (record 19219927), not yet downloaded
`checkpoints.zip` 213 MB, `evaluations.zip` 0.3 MB, `proof_cache_mega.json` 40.3 MB. **No datasets.**
- Useful as a cheap anchor: re-evaluate the paper's own checkpoints under the grind prover (eval only). That
  separates "prover changed" from "hyperparameters changed".
- Check that the checkpoint layout matches what `pickle_my_model.py` expects (`--model-dir`, `{run_id}_marl.pt`
  plus the run lock).

## 6. Proposed plan (next session)
1. **Layout.** Keep the clone's upstream history and do all changes on a branch (`git switch -c reviews`) so
   that `git diff ad4c4d7` shows exactly what changed versus the paper code. Add alongside:
   - `MathWorld_Reviews/CLAUDE.md`, built from §7 plus the review-specific facts above
   - `env/`: venv build, Lean build and lean_env sbatch scripts
   - `scripts/`: SLURM templates and submit helpers
   - `analysis/`
   - `llm_baseline/`
   - a gitignored `runs/`

   Decide with the user whether `MathWorld_Reviews` itself should be a git repo.
2. **Environment job** (sbatch): uv venv with Python 3.11 pinned to poetry.lock, CPU torch, own Julia depot,
   first PySR import resolved online inside the job, then `PYTHON_JULIAPKG_OFFLINE=yes` for every later job.
   Mirror `check_env.sh` from MathWorld: import-order check, which libstdc++ loads.
3. **Datasets**: fetch and copy per §5.9, then re-run the signature check.
4. **Lean job**: toolchain v4.24, a lakefile without LeanCopilot, Mathlib v4.24 from source, the grind
   template.
5. **Prover switch + calibration script** (§5.4). Gate everything on it.
6. **HP plumbing** in `train_maddpg.py`:
   - flags `--lr-actor --lr-critic --gamma --tau --batch-size --ou-noise/--no-ou-noise --expl-noise`
     (plus theta / dt / mean-noise), with defaults = the right-hand column of §5.2, passed into `MADDPG(...)`
   - make the loop's batch gate use the same value
   - log the resolved values to the W&B config and a sidecar JSON
   - add `--seed`
   - leave `INIT_HP` in place but mark it as unused, or delete it; ask the user
7. **Smoke + timing job** for one short train and eval per dataset profile. Then give the user a **core-hour
   estimate before any wave** (they are cost-conscious).
   - Rough arm count: centre + lr_actor×2 + lr_critic×2 + gamma×2 + tau×2 + batch×2 + OU off + expl_noise×2
     = **14 configs × 4 datasets × 3 seeds = 168 train + 168 eval**.
   - The paper quotes ~8–10 h each for train and eval on an M3 MacBook. This may need trimming (fewer seeds,
     fewer levels, shorter eval); present options with the numbers.
8. **Wave**: ONE `--array` per phase (index → (config, dataset, seed) via a lookup table), eval chained with
   `--dependency=aftercorr:<train array>`, right-sized `WALL`, `--mail-user=da579@cam.ac.uk`.
9. **LLM harness** (`llm_baseline/`):
   - one interface, `propose(messages) -> text`, with backends for Anthropic (Claude), OpenAI (GPT), DeepSeek
     (OpenAI-compatible API) and Qwen (OpenAI-compatible: DashScope, OpenRouter or Together)
   - keys from env vars only, never committed
   - log model id, sampling params, prompt, raw response and timestamp for every call
   - parse the output into the env's conjecture grammar, then score it with the **identical**
     `translate_fol_to_lean_with_grind` + `check_theorem_no_leancopilot` path and the same analysis tiers
   - load the `claude-api` skill before writing the Anthropic code. Current IDs: `claude-opus-5-5`,
     `claude-sonnet-5-5`, `claude-haiku-5-5`, `claude-fable-5-1`
   - check that compute nodes reach the API endpoints
   - leave the prompt/protocol pluggable: **the user designs it**
10. **Analysis gaps to fill.** MathWorld has no script for either of these:
    - a Fisher / chi-square / Mann-Whitney comparison between arms (`scratch/layer3_compare.py` in MathWorld
      has a pure-numpy Mann-Whitney U)
    - sacct → core-hours (needed for a compute-cost table)

## 7. Durable learnings carried over from MathWorld (put these in the new CLAUDE.md)
Sources: `/rds/user/da579/hpc-work/MathWorld/CLAUDE.md` and the user's MathWorld auto-memory. **The new session
will NOT load either automatically**, because memory is keyed by project path.

**Cluster rules (CSD3 admin complaint 2026-09-09 named Claude explicitly):**
- Login-node work is SECONDS long: ≤ 4 CPUs, ≤ 20 GB, `OMP_NUM_THREADS=1`, no production work, no IO churn on
  /rds. A watchdog kills heavy login processes, and it has killed Claude Code itself.
- Never on a login node: pytest, anything importing pysr/juliacall/torch, Julia precompile, Lean builds or
  checks. Use sbatch (or a short `srun -A mishra-sl3-cpu -p icelake …`).
- Slurm client calls (`squeue` / `sacct` / `sbatch`) must be **≥ 2 min apart, never in loops or watchers**.
  Submit sweeps and seed replicates as **ONE `--array`**, never a loop of sbatch.
- Use sbatch, not a background srun, for anything that must outlive the turn: a background srun dies with its
  shell.
- Accounts: `mishra-sl3-cpu` is free but capped (~200k core-h/quarter, 448-core per-user cap, medium
  priority); `mishra-sl4-cpu` is free, lowest priority, backfill only. Both have a 12 h wall.
  **Show a core-hour estimate before scaling.**
- Right-size `WALL` per job; backfill fills short gaps first. Size it from the worst observed rate, never the
  median.
- Job email: SLURM `--mail-user=da579@cam.ac.uk`. Mail from user processes on nodes is blocked. Never name a
  shell variable `MAIL`.
- A job showing RUNNING with no processes can mean the node rebooted under it; check the node's BootTime.
  `SBATCH_EXCLUDE` is a no-op; use `--exclude`.
- Verify state with `squeue` / `sacct` / files before resubmitting. An empty or buffered log does not mean
  nothing happened.

**Python, Julia and PySR:**
- uv is a module (`module load ceuadmin/uv/0.12.5`); if it vanishes, check `module avail ceuadmin/uv`. Also
  `module load gcc/11` for GLIBCXX_3.4.26.
- **Import PySR before torch** (torch first segfaults, PyTorch #78829).
- **`faulthandler.enable()` must run BEFORE Julia starts.** Enabled after `import pysr`, it hijacks Julia's
  GC-safepoint signal, and every thread "segfaults" at the first GC.
- Julia depot: build it once in a job, then run jobs with `PYTHON_JULIAPKG_OFFLINE=yes`. An online resolve
  inside many jobs rewrites the shared `Manifest.toml` and installs into ephemeral depots, which produces mass
  "failed to find source of parent package" failures. Never run the full pytest without
  `PYTHON_JULIAPKG_OFFLINE=yes`.
- Per-job tmpfs copies count against the job's memory cgroup; prefer a shared read-only depot plus a small
  per-job scratch.

**Experimental-design lessons (domain-agnostic):**
- **Diff the configs before theorising.** MathWorld's agent path silently ran a different search than its
  own baseline ten times: budget knobs drivable to 0, dead boost knobs, constants off, a different maxsize box,
  a parameter only the agent could tune, the replay-buffer scale. §5.2 here is the same class of bug.
- A baseline must sample **exactly the box the agent can reach**, with the same constant settings, the same
  pinned knobs and the same fit budget. Compare best-of-K against best-of-K, never one fit against an episode.
- Replicate the unit, don't pool. A rare success is tested by re-running THAT checkpoint with fresh nuisance
  seeds; a pooled rate across heterogeneous policies hid a real 13× per-policy effect. A checkpoint selected
  for having succeeded inflates the effect size.
- Hit rule = strict equality, backed by a calibration script that exits nonzero if ground truth stops scoring
  1. Never add a tolerance.
- CSVs that round scores hide near misses. Count hits from the raw records or the `terminated` flag.
- Ablation semantics:
  - skeptic off = weights fixed at 1 in train AND eval
  - prover off = no provability reward in training, real prover at eval
- Keep train and eval episodes the same length (`train_block = num_envs × max_timesteps`) so a training
  episode is an eval episode.
- A queued job reads the source when it STARTS. Editing code while jobs are pending changes what they run;
  use flags or env vars for per-submission settings.
- Write per-step training records to NFS (JSONL). Files in tmpfs and the PySR hall of fame die with the job.
- Report per-cell n. Don't rerun failed or wall-killed cells without asking.

**User preferences:**
- Sparse code comments: add the code, skip the explanatory prose.
- Use standard terms, defining a transform by its formula and standard name rather than repo shorthand.
- Cost-conscious: estimate first.
- Never propose rewards that name the answer.

## 8. MathWorld files worth porting (adapt; do not copy knot logic)
All paths are under `/rds/user/da579/hpc-work/MathWorld/`. Generic, port nearly as-is:
- `scripts/campaign_seeds.sbatch`: seed array, eval chained per seed with `aftercorr`
- `scripts/wave_digest.sh`: sacct final state and elapsed time
- `scripts/notify_when_done.sh`: an `afterany` sentinel job that triggers the mail
- `scripts/check_env.sh`
- `scripts/warm_shared_depot.sh`: depot builder (change the paths)
- `scripts/submit_campaign.sh`: WALL / DEP / EXCLUDE / NOTIFY_EMAIL handling (strip the variants)
- `scripts/smoke_campaign.sh`: a PASS/FAIL `check()` harness with a fake date tag

Generic patterns that need rewrites against the homology CLI:
- `run_pysr_baseline.py`: `sample_ga_params` plus the resumable loop. Reference only; not requested.
- `run_campaign_evals.py`: fast exit if done, resume from per-episode partials, `--out-tag` re-eval with
  fresh seeds
- `campaign_common.py`: only `stage_julia_depot`, `setup_scratch`, `run_name`
- `scratch/reeval_hit_policy_100926.sbatch`: fresh-seed re-eval of one checkpoint
- `scratch/layer3_compare.py`: `mannwhitney_u`
- `scripts/compare_sig_baseline.py`: best-of-episode quantiles, all-pairs win %

Do NOT port:
- `scripts/monitor_arms.sh`: it calls `squeue` every 60 s, which breaks the admin rule
- `scratch/submit_080926_wave.sh`: a loop of sbatch calls
- anything with `calibrate_*`, `l5`, `cs`, `sigma`, `mirror`, `knot`, `lean_*` or `layer*` in its name

## 9. Open questions for the user
1. Should `MathWorld_Reviews` itself be a git repo, or only the clone, on the `reviews` branch?
2. Delete or keep the unused `INIT_HP` dict after the HP flags are added?
3. Sweep levels per knob (×0.5 / ×2?), the number of seeds, and eval length. Decide after the timing job's
   core-hour estimate.
4. Should the paper's checkpoints be re-evaluated under grind (Zenodo) as the prover anchor? (Recommended:
   cheap.)
5. LLM baseline protocol: the user will specify later. Which API keys / providers are available on CSD3
   (DeepSeek direct vs OpenRouter / Together for Qwen)?
