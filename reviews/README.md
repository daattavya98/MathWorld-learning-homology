# Review experiments: hyperparameter sweep (sweep1)

Prover: Lean `grind` (`formal_response_grind/`). 11 arms (`results/sweep1/tasks.tsv`, `scripts/sweep_arms.tsv`) x D0-D3 x 2 seeds, 600 training steps, 10 eval episodes of up to 50 steps per run.

- `scripts/`: SLURM pipeline (train -> eval -> `analysis.sbatch` -> `analysis_readable.sbatch`; `wave_analysis.sbatch` runs both per arm/dataset).
- `analysis/`: `episode_metrics.py` calls `analyze_episodes.compute_all_metrics_with_cis` unchanged; `check_readable_roundtrip.py` checks the readable conversion; `collect_vs_paper.py` builds the summary.
- `env/`: environment setup (no credentials).

## Results: `results/sweep1/analysis/`
- `vs_paper_summary.csv`: long-form, every arm x dataset x scope (s0, s1, pooled, paper) x metric, with bootstrap CI and P(ours > paper). Not normalised.
- `<arm>/<D>/csv/`: raw depickled eval CSVs per seed, pooled (seed i episodes offset by 1000*i), and the paper's Zenodo `paper_<D>_M0_eval.csv`.
- `<arm>/<D>/readable/csv/`: readable-form CSVs; `readable/episode_metrics_*_vs_paper.json`: ours (A) vs paper (B).
- `<arm>/<D>/episode_metrics_*.json`: raw-form metrics, ours only.
- `code.rev`, `job.meta`: code revision and job info per analysis.
