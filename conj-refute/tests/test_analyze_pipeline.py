import os
from pathlib import Path

import pandas as pd
import sympy as sp

from conftest import load_module_from_src


os.environ.setdefault("CONJ_REFUTE_DATASET_PROFILE", "D0")


analyze_blocks = load_module_from_src("analyze_blocks.py", "analyze_blocks_under_test")
analyze_episodes = load_module_from_src(
    "analyze_episodes.py", "analyze_episodes_under_test"
)


def _symbols():
    return {
        "height_D1": sp.Symbol("height_D1"),
        "width_D1": sp.Symbol("width_D1"),
        "width_D2": sp.Symbol("width_D2"),
        "height_D2": sp.Symbol("height_D2"),
        "rank_D1": sp.Symbol("rank_D1"),
        "rank_D2": sp.Symbol("rank_D2"),
        "nullity_D1": sp.Symbol("nullity_D1"),
        "nullity_D2": sp.Symbol("nullity_D2"),
    }


def _write_episode_csv(path: Path):
    df = pd.DataFrame(
        {
            "episode": [0, 0, 0, 1],
            "step": [1, 2, 3, 1],
            "conjecture": [
                "equals(height_D2, width_D2)",
                "equals(nullity_D1, rank_D2)",
                "equals(rank_D1, nullity_D2)",
                "equals(height_D1, width_D1)",
            ],
            "formal_response": [1, 0, 1, 1],
        }
    )
    df.to_csv(path, index=False)


def test_extract_equals_blocks_nested():
    text = "conjunction(equals(a,b),equals(equals(c,d),e))"
    blocks = analyze_blocks.extract_equals_blocks(text)
    assert "equals(a,b)" in blocks
    assert "equals(equals(c,d),e)" in blocks
    assert "equals(c,d)" in blocks


def test_split_args_respects_nested_parens():
    args = "plus(a,b), times(c,minus(d,e))"
    left, right = analyze_blocks.split_args(args)
    assert left == "plus(a,b)"
    assert right == "times(c,minus(d,e))"


def test_parse_expr_underscore_substitution():
    syms = {"height_D1": sp.Symbol("height_D1")}
    expr = analyze_blocks.parse_expr("height_D1 + _1", syms)
    assert str(sp.simplify(expr - (syms["height_D1"] + 1))) == "0"


def test_check_alternating_sum_positive_and_negative():
    syms = _symbols()
    h = syms["height_D1"]
    w1 = syms["width_D1"]
    w2 = syms["width_D2"]

    good = 2 * h - 2 * w1 + 2 * w2 - 3
    bad = h * w1 + w2

    assert analyze_blocks.check_alternating_sum(good, h, w1, w2)
    assert not analyze_blocks.check_alternating_sum(bad, h, w1, w2)


def test_check_betti_numbers_detects_b0_and_b1():
    syms = _symbols()

    b0_expr = syms["height_D1"] - syms["rank_D1"]
    b1_expr = syms["nullity_D1"] - syms["rank_D2"]

    assert analyze_blocks.check_betti_numbers(b0_expr, syms) == {"b0"}
    assert analyze_blocks.check_betti_numbers(b1_expr, syms) == {"b1"}


def test_compute_episode_stats_objects_smoke(tmp_path: Path):
    csv_path = tmp_path / "episodes.csv"
    _write_episode_csv(csv_path)

    episodes = analyze_episodes.compute_episode_stats_objects(str(csv_path))

    assert len(episodes) == 2
    assert episodes[0].steps == 3
    assert episodes[1].steps == 1


def test_bootstrap_helpers_are_deterministic():
    x = [0.1, 0.2, 0.3, 0.4]
    y = [0.05, 0.1, 0.2, 0.25]

    mean_hat, ci, z_sigma = analyze_episodes.bootstrap_mean_ci_with_sigma(
        x, B=200, seed=0
    )
    ratio_hat, ratio_ci, z_log, p_improve, sigma_eff = (
        analyze_episodes.bootstrap_ratio_ci(x, y, B=200, seed=0)
    )

    assert mean_hat > 0
    assert ci[0] <= ci[1]
    assert z_sigma == z_sigma
    assert ratio_hat > 1.0
    assert ratio_ci[0] <= ratio_ci[1]
    assert z_log == z_log
    assert 0.0 <= p_improve <= 1.0
    assert sigma_eff == sigma_eff


def test_cluster_bootstrap_ratio_ci_smoke(tmp_path: Path):
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    _write_episode_csv(csv_a)
    _write_episode_csv(csv_b)

    episodes_a = analyze_episodes.compute_episode_stats_objects(str(csv_a))
    episodes_b = analyze_episodes.compute_episode_stats_objects(str(csv_b))

    metric_fn = analyze_episodes.chi_usage_pooled
    ratio_hat, ratio_ci, _, p_improve, _ = analyze_episodes.cluster_bootstrap_ratio_ci(
        episodes_a, episodes_b, metric_fn=metric_fn, B=100, seed=0
    )

    assert ratio_hat == ratio_hat
    assert ratio_ci[0] <= ratio_ci[1]
    assert 0.0 <= p_improve <= 1.0
