import pandas as pd
import numpy as np
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set
import os
from scipy.stats import norm
from analysis_utils import (
    load_conjecture_dataframe,
    build_symbols_with_width_height_fallback,
    extract_equals_blocks,
    atomic_block_to_poly,
    check_alternating_sum,
    alternating_sum_with_rank_nullity,
    check_betti_numbers,
    betti_with_rank_nullity,
    bootstrap_mean_ci,
)

try:
    from .conj_refute_env.env.utils import conjecture_to_readable_latex
except ImportError:
    from conj_refute_env.env.utils import conjecture_to_readable_latex


@dataclass
class EpisodeStats:
    episode_id: int

    total_blocks_occ: int
    chi_blocks_occ: int
    homology_blocks_occ: int

    unique_blocks: Set[str]
    unique_chi_blocks: Set[str]
    unique_homology_blocks: Set[str]

    proven_total: int
    proven_with_ch_or_hom: int

    proven_with_chi: int
    proven_with_homology: int

    proven_unique_blocks: Set[str]
    proven_unique_ch_or_hom_blocks: Set[str]

    proven_unique_chi_blocks: set
    proven_unique_homology_blocks: set

    steps: int

    # Conjectures in this episode that contain chi-matching atomic blocks
    chi_conjecture_hits: List[dict]


def compute_episode_stats_objects(
    csv_file_path: str, premises=None, proof_threshold: int = 1
) -> List[EpisodeStats]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df, _ = load_conjecture_dataframe(
        csv_file_path,
        script_dir=script_dir,
        required_columns=["formal_response"],
    )

    if "episode" not in df.columns:
        df["episode"] = 0  # Single episode if none provided

    # Build symbols once
    symbols_dict, h1_symbol, w1_symbol, w2_symbol, _ = (
        build_symbols_with_width_height_fallback(df)
    )

    episodes: Dict[int, List[Tuple[str, int]]] = {}
    for _, row in df.iterrows():
        ep = int(row["episode"])
        episodes.setdefault(ep, []).append(
            (row["conjecture"], int(row["formal_response"]))
        )

    out: List[EpisodeStats] = []

    for ep, rows in episodes.items():
        total_blocks_occ = 0
        chi_blocks_occ = 0
        homology_blocks_occ = 0

        unique_blocks: Set[str] = set()
        unique_chi_blocks: Set[str] = set()
        unique_homology_blocks: Set[str] = set()

        proven_total = 0
        proven_with_ch_or_hom = 0
        proven_unique_blocks: Set[str] = set()
        proven_unique_ch_or_hom_blocks: Set[str] = set()
        proven_with_chi = 0
        proven_with_homology = 0
        proven_unique_chi_blocks = set()
        proven_unique_homology_blocks = set()
        chi_conjecture_hits = []

        for step_num, (conjecture_str, formal_response) in enumerate(rows, start=1):
            blocks = extract_equals_blocks(conjecture_str)

            # occurrence denominator for usage rate
            total_blocks_occ += len(blocks)

            # unique denominator for breadth
            unique_blocks.update(blocks)

            # Determine which blocks in this conjecture hit chi/homology
            conj_has_ch_or_hom = False
            conj_ch_or_hom_blocks: Set[str] = set()
            conj_has_chi = False
            conj_has_homology = False
            conj_chi_blocks = set()
            conj_hom_blocks = set()

            for block_str in blocks:
                diff = atomic_block_to_poly(
                    block_str, symbols_dict, w1_symbol, premises=premises
                )
                if diff is False or diff is None:
                    continue

                chi_hit = check_alternating_sum(
                    diff, h1_symbol, w1_symbol, w2_symbol
                ) or alternating_sum_with_rank_nullity(
                    diff, symbols_dict, premises=premises
                )
                hom_hit = bool(
                    check_betti_numbers(
                        diff,
                        symbols_dict,
                        premises=premises,
                        include_b2=True,
                    )
                ) or betti_with_rank_nullity(
                    diff,
                    symbols_dict,
                    premises=premises,
                    include_b2=True,
                )

                if chi_hit:
                    chi_blocks_occ += 1
                    unique_chi_blocks.add(block_str)
                    conj_has_ch_or_hom = True
                    conj_has_chi = True
                    conj_ch_or_hom_blocks.add(block_str)
                    conj_chi_blocks.add(block_str)

                if hom_hit:
                    homology_blocks_occ += 1
                    unique_homology_blocks.add(block_str)
                    conj_has_ch_or_hom = True
                    conj_has_homology = True
                    conj_ch_or_hom_blocks.add(block_str)
                    conj_hom_blocks.add(block_str)

            # proven stats
            if formal_response >= proof_threshold:
                proven_total += 1
                proven_unique_blocks.update(blocks)
                if conj_has_ch_or_hom:
                    proven_with_ch_or_hom += 1
                    proven_unique_ch_or_hom_blocks.update(conj_ch_or_hom_blocks)
                if conj_has_chi:
                    proven_with_chi += 1
                    proven_unique_chi_blocks.update(conj_chi_blocks)
                if conj_has_homology:
                    proven_with_homology += 1
                    proven_unique_homology_blocks.update(conj_hom_blocks)

            if conj_has_chi:
                chi_conjecture_hits.append(
                    {
                        "episode": ep,
                        "step": step_num,
                        "conjecture": conjecture_str,
                        "chi_blocks": sorted(conj_chi_blocks),
                    }
                )

        out.append(
            EpisodeStats(
                episode_id=ep,
                total_blocks_occ=total_blocks_occ,
                chi_blocks_occ=chi_blocks_occ,
                homology_blocks_occ=homology_blocks_occ,
                unique_blocks=unique_blocks,
                unique_chi_blocks=unique_chi_blocks,
                unique_homology_blocks=unique_homology_blocks,
                proven_total=proven_total,
                proven_with_ch_or_hom=proven_with_ch_or_hom,
                proven_with_chi=proven_with_chi,
                proven_with_homology=proven_with_homology,
                proven_unique_blocks=proven_unique_blocks,
                proven_unique_ch_or_hom_blocks=proven_unique_ch_or_hom_blocks,
                proven_unique_chi_blocks=proven_unique_chi_blocks,
                proven_unique_homology_blocks=proven_unique_homology_blocks,
                steps=len(rows),
                chi_conjecture_hits=chi_conjecture_hits,
            )
        )

    return out


def sigma_effective_from_p(p, cap=5.0):
    """
    Convert dominance probability to one-sided Gaussian-equivalent sigma.
    """
    if p <= 0.0:
        return -cap
    if p >= 1.0:
        return cap
    return float(norm.ppf(p))


def bootstrap_mean_ci_with_sigma(x, B=10000, alpha=0.05, seed=0):
    return bootstrap_mean_ci(x, B=B, alpha=alpha, seed=seed, return_sigma=True)


def bootstrap_ratio_ci(a, b, B=10000, alpha=0.05, seed=0, eps=1e-12):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]

    rng = np.random.default_rng(seed)
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return np.nan, (np.nan, np.nan), np.nan

    ratios = np.empty(B)
    for i in range(B):
        a_star = rng.choice(a, size=n1, replace=True)
        b_star = rng.choice(b, size=n2, replace=True)

        mean_a = a_star.mean()
        mean_b = b_star.mean()

        ratios[i] = mean_a / max(mean_b, eps)

    ratio_hat = a.mean() / max(b.mean(), eps)
    lo = np.quantile(ratios, alpha / 2)
    hi = np.quantile(ratios, 1 - alpha / 2)

    # P(A > B)
    p_improve = float((ratios > 1.0).mean())
    sigma_eff = sigma_effective_from_p(p_improve)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratios = np.log(ratios)
        z_sigma_log = np.nanmean(log_ratios) / np.nanstd(log_ratios, ddof=1)

    return ratio_hat, (lo, hi), z_sigma_log, p_improve, sigma_eff


def cluster_bootstrap_ci_with_sigma(episodes, metric_fn, B=10000, alpha=0.05, seed=0):
    """
    Resamples episodes with replacement and recomputes a scalar metric each time.
    """
    rng = np.random.default_rng(seed)
    E = len(episodes)
    if E == 0:
        return np.nan, (np.nan, np.nan)

    vals = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, E, size=E)
        sample = [episodes[i] for i in idx]
        vals[b] = metric_fn(sample)

    point = metric_fn(episodes)
    lo = np.nanquantile(vals, alpha / 2)
    hi = np.nanquantile(vals, 1 - alpha / 2)

    sigma_hat = np.std(vals, ddof=1)
    z_sigma = point / sigma_hat if sigma_hat > 0 else np.nan

    return point, (lo, hi), z_sigma


def cluster_bootstrap_ratio_ci(
    episodes_a,
    episodes_b,
    metric_fn,
    B=10000,
    alpha=0.05,
    seed=0,
    eps=1e-12,
):
    rng = np.random.default_rng(seed)
    Ea, Eb = len(episodes_a), len(episodes_b)
    if Ea == 0 or Eb == 0:
        return np.nan, (np.nan, np.nan), np.nan

    ratios = np.empty(B)
    for b in range(B):
        idx_a = rng.integers(0, Ea, size=Ea)
        idx_b = rng.integers(0, Eb, size=Eb)

        samp_a = [episodes_a[i] for i in idx_a]
        samp_b = [episodes_b[i] for i in idx_b]

        val_a = metric_fn(samp_a)
        val_b = metric_fn(samp_b)

        ratios[b] = val_a / max(val_b, eps)

    ratio_hat = metric_fn(episodes_a) / max(metric_fn(episodes_b), eps)
    lo = np.quantile(ratios, alpha / 2)
    hi = np.quantile(ratios, 1 - alpha / 2)

    p_improve = float((ratios > 1.0).mean())
    sigma_eff = sigma_effective_from_p(p_improve)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratios = np.log(ratios)
        z_sigma_log = np.nanmean(log_ratios) / np.nanstd(log_ratios, ddof=1)

    return ratio_hat, (lo, hi), z_sigma_log, p_improve, sigma_eff


def episode_level_arrays(episodes: List[EpisodeStats]):
    chi_usage = []
    chi_breadth = []
    hom_usage = []
    hom_breadth = []

    proven_rate = []
    proven_diverse_ch_or_hom_breadth = (
        []
    )

    # split proven metrics
    proven_with_chi_rate = []
    proven_with_hom_rate = []
    proven_diverse_chi_breadth = []
    proven_diverse_hom_breadth = []

    steps = []

    for ep in episodes:
        chi_usage.append(
            ep.chi_blocks_occ / ep.total_blocks_occ
            if ep.total_blocks_occ > 0
            else np.nan
        )
        hom_usage.append(
            ep.homology_blocks_occ / ep.total_blocks_occ
            if ep.total_blocks_occ > 0
            else np.nan
        )

        denom_unique = len(ep.unique_blocks)
        chi_breadth.append(
            len(ep.unique_chi_blocks) / denom_unique if denom_unique > 0 else np.nan
        )
        hom_breadth.append(
            len(ep.unique_homology_blocks) / denom_unique
            if denom_unique > 0
            else np.nan
        )

        proven_rate.append(
            ep.proven_with_ch_or_hom / ep.proven_total
            if ep.proven_total > 0
            else np.nan
        )
        proven_with_chi_rate.append(
            ep.proven_with_chi / ep.proven_total if ep.proven_total > 0 else np.nan
        )
        proven_with_hom_rate.append(
            ep.proven_with_homology / ep.proven_total if ep.proven_total > 0 else np.nan
        )

        denom_proven_unique = len(ep.proven_unique_blocks)
        proven_diverse_ch_or_hom_breadth.append(
            len(ep.proven_unique_ch_or_hom_blocks) / denom_proven_unique
            if denom_proven_unique > 0
            else np.nan
        )
        proven_diverse_chi_breadth.append(
            len(ep.proven_unique_chi_blocks) / denom_proven_unique
            if denom_proven_unique > 0
            else np.nan
        )
        proven_diverse_hom_breadth.append(
            len(ep.proven_unique_homology_blocks) / denom_proven_unique
            if denom_proven_unique > 0
            else np.nan
        )

        steps.append(ep.steps)

    return {
        "chi_usage_ep": np.array(chi_usage, float),
        "chi_breadth_ep": np.array(chi_breadth, float),
        "hom_usage_ep": np.array(hom_usage, float),
        "hom_breadth_ep": np.array(hom_breadth, float),
        "proven_rate_ep": np.array(proven_rate, float),
        "proven_diverse_ch_or_hom_breadth_ep": np.array(
            proven_diverse_ch_or_hom_breadth, float
        ),
        # split metrics
        "proven_with_chi_rate_ep": np.array(proven_with_chi_rate, float),
        "proven_with_hom_rate_ep": np.array(proven_with_hom_rate, float),
        "proven_diverse_chi_breadth_ep": np.array(proven_diverse_chi_breadth, float),
        "proven_diverse_hom_breadth_ep": np.array(proven_diverse_hom_breadth, float),
        "steps": np.array(steps, float),
    }


def chi_usage_pooled(episodes: List[EpisodeStats]) -> float:
    K = sum(ep.chi_blocks_occ for ep in episodes)
    N = sum(ep.total_blocks_occ for ep in episodes)
    return K / N if N > 0 else np.nan


def hom_usage_pooled(episodes: List[EpisodeStats]) -> float:
    K = sum(ep.homology_blocks_occ for ep in episodes)
    N = sum(ep.total_blocks_occ for ep in episodes)
    return K / N if N > 0 else np.nan


def chi_discovery_breadth_global(episodes: List[EpisodeStats]) -> float:
    U = set()
    C = set()
    for ep in episodes:
        U |= ep.unique_blocks
        C |= ep.unique_chi_blocks
    return len(C) / len(U) if len(U) > 0 else np.nan


def hom_discovery_breadth_global(episodes: List[EpisodeStats]) -> float:
    U = set()
    H = set()
    for ep in episodes:
        U |= ep.unique_blocks
        H |= ep.unique_homology_blocks
    return len(H) / len(U) if len(U) > 0 else np.nan


def proven_with_ch_or_hom_rate_global(episodes: List[EpisodeStats]) -> float:
    P = sum(ep.proven_total for ep in episodes)
    Q = sum(ep.proven_with_ch_or_hom for ep in episodes)
    return Q / P if P > 0 else np.nan


def proven_diverse_ch_or_hom_breadth_global(episodes: List[EpisodeStats]) -> float:
    U = set()
    CH = set()
    for ep in episodes:
        U |= ep.proven_unique_blocks
        CH |= ep.proven_unique_ch_or_hom_blocks
    return len(CH) / len(U) if len(U) > 0 else np.nan


def proven_with_chi_rate_global(episodes):
    P = sum(ep.proven_total for ep in episodes)
    Q = sum(ep.proven_with_chi for ep in episodes)
    return Q / P if P > 0 else np.nan


def proven_with_homology_rate_global(episodes):
    P = sum(ep.proven_total for ep in episodes)
    Q = sum(ep.proven_with_homology for ep in episodes)
    return Q / P if P > 0 else np.nan


def proven_diverse_chi_breadth_global(episodes):
    U, C = set(), set()
    for ep in episodes:
        U |= ep.proven_unique_blocks
        C |= ep.proven_unique_chi_blocks
    return len(C) / len(U) if len(U) > 0 else np.nan


def proven_diverse_hom_breadth_global(episodes):
    U, H = set(), set()
    for ep in episodes:
        U |= ep.proven_unique_blocks
        H |= ep.proven_unique_homology_blocks
    return len(H) / len(U) if len(U) > 0 else np.nan


def median_iqr(x: np.ndarray):
    x = np.asarray(x, float)
    return float(np.median(x)), (
        float(np.quantile(x, 0.25)),
        float(np.quantile(x, 0.75)),
    )


def compute_all_metrics_with_cis(
    csv_a: str,
    csv_b: Optional[str] = None,
    premises=None,
    premises_a=None,
    premises_b=None,
    B=10000,
    alpha=0.05,
    seed=0,
    proof_threshold=1,
):

    if premises_a is None:
        premises_a = premises
    if premises_b is None:
        premises_b = premises

    eps_a = compute_episode_stats_objects(
        csv_a, premises=premises_a, proof_threshold=proof_threshold
    )
    arr_a = episode_level_arrays(eps_a)

    # Episode-length summary
    med_steps_a, (q1_a, q3_a) = median_iqr(arr_a["steps"])

    # Episode-average metrics + CIs (bootstrap over episodes, unweighted)
    ep_mean_ci_a = {
        "chi_usage_ep": bootstrap_mean_ci_with_sigma(
            arr_a["chi_usage_ep"], B=B, alpha=alpha, seed=seed
        ),
        "chi_breadth_ep": bootstrap_mean_ci_with_sigma(
            arr_a["chi_breadth_ep"], B=B, alpha=alpha, seed=seed
        ),
        "hom_usage_ep": bootstrap_mean_ci_with_sigma(
            arr_a["hom_usage_ep"], B=B, alpha=alpha, seed=seed
        ),
        "hom_breadth_ep": bootstrap_mean_ci_with_sigma(
            arr_a["hom_breadth_ep"], B=B, alpha=alpha, seed=seed
        ),
        "proven_rate_ep": bootstrap_mean_ci_with_sigma(
            arr_a["proven_rate_ep"], B=B, alpha=alpha, seed=seed
        ),
        "proven_with_hom_rate_ep": bootstrap_mean_ci_with_sigma(
            arr_a["proven_with_hom_rate_ep"], B=B, alpha=alpha, seed=seed
        ),
        "proven_with_chi_rate_ep": bootstrap_mean_ci_with_sigma(
            arr_a["proven_with_chi_rate_ep"], B=B, alpha=alpha, seed=seed
        ),
        "proven_diverse_chi_breadth_ep": bootstrap_mean_ci_with_sigma(
            arr_a["proven_diverse_chi_breadth_ep"], B=B, alpha=alpha, seed=seed
        ),
        "proven_diverse_hom_breadth_ep": bootstrap_mean_ci_with_sigma(
            arr_a["proven_diverse_hom_breadth_ep"], B=B, alpha=alpha, seed=seed
        ),
        "proven_diverse_ch_or_hom_breadth_ep": bootstrap_mean_ci_with_sigma(
            arr_a["proven_diverse_ch_or_hom_breadth_ep"], B=B, alpha=alpha, seed=seed
        ),
    }

    # Cluster bootstrap CIs for pooled/union estimands
    pooled_ci_a = {
        "chi_usage_pooled": cluster_bootstrap_ci_with_sigma(
            eps_a, chi_usage_pooled, B=B, alpha=alpha, seed=seed
        ),
        "hom_usage_pooled": cluster_bootstrap_ci_with_sigma(
            eps_a, hom_usage_pooled, B=B, alpha=alpha, seed=seed
        ),
        "chi_breadth_global": cluster_bootstrap_ci_with_sigma(
            eps_a, chi_discovery_breadth_global, B=B, alpha=alpha, seed=seed
        ),
        "hom_breadth_global": cluster_bootstrap_ci_with_sigma(
            eps_a, hom_discovery_breadth_global, B=B, alpha=alpha, seed=seed
        ),
        "proven_rate_global": cluster_bootstrap_ci_with_sigma(
            eps_a, proven_with_ch_or_hom_rate_global, B=B, alpha=alpha, seed=seed
        ),
        "proven_with_chi_rate_global": cluster_bootstrap_ci_with_sigma(
            eps_a, proven_with_chi_rate_global, B=B, alpha=alpha, seed=seed
        ),
        "proven_with_hom_rate_global": cluster_bootstrap_ci_with_sigma(
            eps_a, proven_with_homology_rate_global, B=B, alpha=alpha, seed=seed
        ),
        "proven_diverse_chi_breadth_global": cluster_bootstrap_ci_with_sigma(
            eps_a, proven_diverse_chi_breadth_global, B=B, alpha=alpha, seed=seed
        ),
        "proven_diverse_hom_breadth_global": cluster_bootstrap_ci_with_sigma(
            eps_a, proven_diverse_hom_breadth_global, B=B, alpha=alpha, seed=seed
        ),
        "proven_diverse_ch_or_hom_breadth_global": cluster_bootstrap_ci_with_sigma(
            eps_a, proven_diverse_ch_or_hom_breadth_global, B=B, alpha=alpha, seed=seed
        ),
    }

    result = {
        "A": {
            "episode_stats": ep_mean_ci_a,
            "pooled_stats": pooled_ci_a,
            "steps_median": med_steps_a,
            "steps_q1_q3": (q1_a, q3_a),
            "num_episodes": len(eps_a),
        }
    }

    if csv_b is None:
        return result

    eps_b = compute_episode_stats_objects(
        csv_b, premises=premises_b, proof_threshold=proof_threshold
    )
    arr_b = episode_level_arrays(eps_b)
    med_steps_b, (q1_b, q3_b) = median_iqr(arr_b["steps"])

    ep_mean_ci_b = {
        "chi_usage_ep": bootstrap_mean_ci_with_sigma(
            arr_b["chi_usage_ep"], B=B, alpha=alpha, seed=seed
        ),
        "chi_breadth_ep": bootstrap_mean_ci_with_sigma(
            arr_b["chi_breadth_ep"], B=B, alpha=alpha, seed=seed
        ),
        "hom_usage_ep": bootstrap_mean_ci_with_sigma(
            arr_b["hom_usage_ep"], B=B, alpha=alpha, seed=seed
        ),
        "hom_breadth_ep": bootstrap_mean_ci_with_sigma(
            arr_b["hom_breadth_ep"], B=B, alpha=alpha, seed=seed
        ),
        "proven_rate_ep": bootstrap_mean_ci_with_sigma(
            arr_b["proven_rate_ep"], B=B, alpha=alpha, seed=seed
        ),
        "proven_with_chi_rate_ep": bootstrap_mean_ci_with_sigma(
            arr_b["proven_with_chi_rate_ep"], B=B, alpha=alpha, seed=seed
        ),
        "proven_with_hom_rate_ep": bootstrap_mean_ci_with_sigma(
            arr_b["proven_with_hom_rate_ep"], B=B, alpha=alpha, seed=seed
        ),
        "proven_diverse_chi_breadth_ep": bootstrap_mean_ci_with_sigma(
            arr_b["proven_diverse_chi_breadth_ep"], B=B, alpha=alpha, seed=seed
        ),
        "proven_diverse_hom_breadth_ep": bootstrap_mean_ci_with_sigma(
            arr_b["proven_diverse_hom_breadth_ep"], B=B, alpha=alpha, seed=seed
        ),
        "proven_diverse_ch_or_hom_breadth_ep": bootstrap_mean_ci_with_sigma(
            arr_b["proven_diverse_ch_or_hom_breadth_ep"], B=B, alpha=alpha, seed=seed
        ),
    }

    pooled_ci_b = {
        "chi_usage_pooled": cluster_bootstrap_ci_with_sigma(
            eps_b, chi_usage_pooled, B=B, alpha=alpha, seed=seed
        ),
        "hom_usage_pooled": cluster_bootstrap_ci_with_sigma(
            eps_b, hom_usage_pooled, B=B, alpha=alpha, seed=seed
        ),
        "chi_breadth_global": cluster_bootstrap_ci_with_sigma(
            eps_b, chi_discovery_breadth_global, B=B, alpha=alpha, seed=seed
        ),
        "hom_breadth_global": cluster_bootstrap_ci_with_sigma(
            eps_b, hom_discovery_breadth_global, B=B, alpha=alpha, seed=seed
        ),
        "proven_rate_global": cluster_bootstrap_ci_with_sigma(
            eps_b, proven_with_ch_or_hom_rate_global, B=B, alpha=alpha, seed=seed
        ),
        "proven_with_chi_rate_global": cluster_bootstrap_ci_with_sigma(
            eps_b, proven_with_chi_rate_global, B=B, alpha=alpha, seed=seed
        ),
        "proven_with_hom_rate_global": cluster_bootstrap_ci_with_sigma(
            eps_b, proven_with_homology_rate_global, B=B, alpha=alpha, seed=seed
        ),
        "proven_diverse_chi_breadth_global": cluster_bootstrap_ci_with_sigma(
            eps_b, proven_diverse_chi_breadth_global, B=B, alpha=alpha, seed=seed
        ),
        "proven_diverse_hom_breadth_global": cluster_bootstrap_ci_with_sigma(
            eps_b, proven_diverse_hom_breadth_global, B=B, alpha=alpha, seed=seed
        ),
        "proven_diverse_ch_or_hom_breadth_global": cluster_bootstrap_ci_with_sigma(
            eps_b, proven_diverse_ch_or_hom_breadth_global, B=B, alpha=alpha, seed=seed
        ),
    }

    deltas_ep = {
        "delta_chi_usage_ep": bootstrap_ratio_ci(
            arr_a["chi_usage_ep"], arr_b["chi_usage_ep"], B=B, alpha=alpha, seed=seed
        ),
        "delta_chi_breadth_ep": bootstrap_ratio_ci(
            arr_a["chi_breadth_ep"],
            arr_b["chi_breadth_ep"],
            B=B,
            alpha=alpha,
            seed=seed,
        ),
        "delta_hom_usage_ep": bootstrap_ratio_ci(
            arr_a["hom_usage_ep"], arr_b["hom_usage_ep"], B=B, alpha=alpha, seed=seed
        ),
        "delta_hom_breadth_ep": bootstrap_ratio_ci(
            arr_a["hom_breadth_ep"],
            arr_b["hom_breadth_ep"],
            B=B,
            alpha=alpha,
            seed=seed,
        ),
        # combined proven metrics (episode mean)
        "delta_proven_rate_ep": bootstrap_ratio_ci(
            arr_a["proven_rate_ep"],
            arr_b["proven_rate_ep"],
            B=B,
            alpha=alpha,
            seed=seed,
        ),
        "delta_proven_diverse_ch_or_hom_breadth_ep": bootstrap_ratio_ci(
            arr_a["proven_diverse_ch_or_hom_breadth_ep"],
            arr_b["proven_diverse_ch_or_hom_breadth_ep"],
            B=B,
            alpha=alpha,
            seed=seed,
        ),
        "delta_proven_with_chi_rate_ep": bootstrap_ratio_ci(
            arr_a["proven_with_chi_rate_ep"],
            arr_b["proven_with_chi_rate_ep"],
            B=B,
            alpha=alpha,
            seed=seed,
        ),
        "delta_proven_with_hom_rate_ep": bootstrap_ratio_ci(
            arr_a["proven_with_hom_rate_ep"],
            arr_b["proven_with_hom_rate_ep"],
            B=B,
            alpha=alpha,
            seed=seed,
        ),
        "delta_proven_diverse_chi_breadth_ep": bootstrap_ratio_ci(
            arr_a["proven_diverse_chi_breadth_ep"],
            arr_b["proven_diverse_chi_breadth_ep"],
            B=B,
            alpha=alpha,
            seed=seed,
        ),
        "delta_proven_diverse_hom_breadth_ep": bootstrap_ratio_ci(
            arr_a["proven_diverse_hom_breadth_ep"],
            arr_b["proven_diverse_hom_breadth_ep"],
            B=B,
            alpha=alpha,
            seed=seed,
        ),
    }

    deltas_pooled = {
        "delta_chi_usage_pooled": cluster_bootstrap_ratio_ci(
            eps_a, eps_b, chi_usage_pooled, B=B, alpha=alpha, seed=seed
        ),
        "delta_hom_usage_pooled": cluster_bootstrap_ratio_ci(
            eps_a, eps_b, hom_usage_pooled, B=B, alpha=alpha, seed=seed
        ),
        "delta_chi_breadth_global": cluster_bootstrap_ratio_ci(
            eps_a, eps_b, chi_discovery_breadth_global, B=B, alpha=alpha, seed=seed
        ),
        "delta_hom_breadth_global": cluster_bootstrap_ratio_ci(
            eps_a, eps_b, hom_discovery_breadth_global, B=B, alpha=alpha, seed=seed
        ),
        # combined proven metrics (pooled/union)
        "delta_proven_rate_global": cluster_bootstrap_ratio_ci(
            eps_a, eps_b, proven_with_ch_or_hom_rate_global, B=B, alpha=alpha, seed=seed
        ),
        "delta_proven_diverse_ch_or_hom_breadth_global": cluster_bootstrap_ratio_ci(
            eps_a,
            eps_b,
            proven_diverse_ch_or_hom_breadth_global,
            B=B,
            alpha=alpha,
            seed=seed,
        ),
        "delta_proven_with_chi_rate_global": cluster_bootstrap_ratio_ci(
            eps_a, eps_b, proven_with_chi_rate_global, B=B, alpha=alpha, seed=seed
        ),
        "delta_proven_with_hom_rate_global": cluster_bootstrap_ratio_ci(
            eps_a, eps_b, proven_with_homology_rate_global, B=B, alpha=alpha, seed=seed
        ),
        "delta_proven_diverse_chi_breadth_global": cluster_bootstrap_ratio_ci(
            eps_a, eps_b, proven_diverse_chi_breadth_global, B=B, alpha=alpha, seed=seed
        ),
        "delta_proven_diverse_hom_breadth_global": cluster_bootstrap_ratio_ci(
            eps_a, eps_b, proven_diverse_hom_breadth_global, B=B, alpha=alpha, seed=seed
        ),
    }

    result["B"] = {
        "episode_stats": ep_mean_ci_b,
        "pooled_stats": pooled_ci_b,
        "steps_median": med_steps_b,
        "steps_q1_q3": (q1_b, q3_b),
        "num_episodes": len(eps_b),
    }
    result["deltas_episode_mean"] = deltas_ep
    result["deltas_pooled_union"] = deltas_pooled

    return result


def convert_conj_to_readable(file_name: str, output_dir: Optional[str] = None):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    resolved = (
        file_name if os.path.isabs(file_name) else os.path.join(script_dir, file_name)
    )
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"CSV not found at '{resolved}'")

    df = pd.read_csv(resolved)
    source_col = None
    if "conjecture" in df.columns:
        source_col = "conjecture"
    elif "LaTeX conjecture" in df.columns:
        source_col = "LaTeX conjecture"

    if source_col is None:
        raise ValueError(
            "CSV must contain either 'conjecture' or 'LaTeX conjecture' column."
        )

    successful_rows = []
    for _, row in df.iterrows():
        conj_str = row[source_col]
        if pd.isna(conj_str):
            continue

        try:
            readable = conjecture_to_readable_latex(conj_str)
        except Exception:
            continue

        updated_row = row.copy()
        updated_row[source_col] = readable
        successful_rows.append(updated_row)

    if output_dir is None:
        output_dir = os.path.join(script_dir, "readable_conjectures")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(
        output_dir,
        f"{os.path.splitext(os.path.basename(resolved))[0]}_readable.csv",
    )
    pd.DataFrame(successful_rows, columns=df.columns).to_csv(output_file, index=False)
    return output_file, len(successful_rows)


def convert_all_model_evaluation_csvs_to_readable(
    model_evaluations_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if model_evaluations_dir is None:
        model_evaluations_dir = os.path.join(script_dir, "model_evaluations")
    if output_dir is None:
        output_dir = os.path.join(script_dir, "readable_conjectures")

    if not os.path.isdir(model_evaluations_dir):
        raise FileNotFoundError(f"Directory not found: '{model_evaluations_dir}'")

    csv_files = sorted(
        f for f in os.listdir(model_evaluations_dir) if f.lower().endswith(".csv")
    )

    if not csv_files:
        return []

    results = []
    for csv_file in csv_files:
        csv_path = os.path.join(model_evaluations_dir, csv_file)
        try:
            output_file, row_count = convert_conj_to_readable(
                csv_path, output_dir=output_dir
            )
            results.append(
                {
                    "input": csv_path,
                    "output": output_file,
                    "rows_written": row_count,
                    "status": "ok",
                }
            )
        except Exception as exc:
            results.append(
                {
                    "input": csv_path,
                    "output": None,
                    "rows_written": 0,
                    "status": f"error: {exc}",
                }
            )

    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Analyze episode-level outputs and convert conjectures to readable form."
    )
    parser.add_argument(
        "--mode",
        choices=["convert-one", "convert-all"],
        default="convert-one",
        help="Run one-file conversion or convert all files from model_evaluations/",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="model_evaluations/D1_280226_run1.csv",
        help="Input CSV used in convert-one mode",
    )
    parser.add_argument(
        "--model-evaluations-dir",
        type=str,
        default=None,
        help="Directory scanned in convert-all mode",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="readable_conjectures",
        help="Output directory for generated readable CSV files",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.mode == "convert-all":
        results = convert_all_model_evaluation_csvs_to_readable(
            model_evaluations_dir=args.model_evaluations_dir,
            output_dir=args.output_dir,
        )
        print(f"Converted {len(results)} files")
        return results

    output_file, row_count = convert_conj_to_readable(
        args.input_file,
        output_dir=args.output_dir,
    )
    print(f"Converted {row_count} rows to: {output_file}")
    return output_file, row_count


if __name__ == "__main__":
    main()
