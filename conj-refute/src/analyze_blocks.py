import pandas as pd
from collections import Counter
import argparse
import os
from typing import List, Tuple
import shutil
from conj_refute_env.env.conjecture_to_lean_statement import (
    translate_fol_to_lean_with_grind,
)
from conj_refute_env.env.check_lean_theorem import (
    check_theorem_no_leancopilot,
)
import numpy as np
from conj_refute_env.env.utils import fix_conj_str
from analysis_utils import (
    extract_equals_blocks,
    load_conjecture_dataframe,
    build_symbols_with_width_height_fallback,
    is_formal_response_positive,
    atomic_block_to_poly,
    check_alternating_sum,
    alternating_sum_with_rank_nullity,
    check_betti_numbers,
    betti_with_rank_nullity,
    bootstrap_mean_ci,
)
from analysis_utils import split_args as _split_args
from analysis_utils import parse_expr as _parse_expr

split_args = _split_args
parse_expr = _parse_expr


def _build_basic_stats_dataframe(total_blocks, unique_block_count, tier_block_counters):
    """Builds a summary table with totals and percentages for each tier."""
    rows = [
        {
            "tier": "overall",
            "total_blocks": total_blocks,
            "unique_blocks": unique_block_count,
            "pct_total_blocks": 100.0 if total_blocks else 0.0,
            "pct_unique_blocks": 100.0 if unique_block_count else 0.0,
        }
    ]

    for tier in ("gold", "silver", "bronze"):
        counter = tier_block_counters.get(tier, {})
        tier_total = sum(counter.values()) if counter else 0
        tier_unique = len(counter)
        rows.append(
            {
                "tier": tier,
                "total_blocks": tier_total,
                "unique_blocks": tier_unique,
                "pct_total_blocks": (
                    (tier_total / total_blocks * 100.0) if total_blocks else 0.0
                ),
                "pct_unique_blocks": (
                    (tier_unique / unique_block_count * 100.0)
                    if unique_block_count
                    else 0.0
                ),
            }
        )

    df = pd.DataFrame(rows)
    return df[
        [
            "tier",
            "total_blocks",
            "unique_blocks",
            "pct_total_blocks",
            "pct_unique_blocks",
        ]
    ]


def _build_conjectures_dataframe(tier_records):
    """Creates a dataframe listing conjectures associated with each tier."""
    columns = [
        "tier",
        "episode",
        "step",
        "atomic_block",
        "conjecture",
        "formal_response",
    ]
    rows = []
    for tier in ("gold", "silver", "bronze"):
        for record in tier_records.get(tier, []):
            rows.append(
                {
                    "tier": tier,
                    "episode": record.get("episode"),
                    "step": record.get("step"),
                    "atomic_block": record.get("atomic_block"),
                    "conjecture": record.get("conjecture"),
                    "formal_response": record.get("formal_response"),
                }
            )

    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows)
    sort_keys = [key for key in ("tier", "episode", "step") if key in df.columns]
    if sort_keys:
        df = df.sort_values(sort_keys).reset_index(drop=True)
    return df[columns]


def _build_trajectory_tables(df, tier_records):
    """Collects per-positive-response trajectory tables keyed by tier, episode, and step."""
    required_cols = [
        "episode",
        "step",
        "conjecture",
        "rew_conj_length",
        "rew_conj_formal",
        "reward_ref",
    ]
    tables = {}
    seen_keys = set()

    for tier, records in tier_records.items():
        for record in records:
            if not _formal_response_is_positive(record.get("formal_response")):
                continue

            episode = record.get("episode")
            step = record.get("step")
            if pd.isna(episode) or pd.isna(step):
                continue

            key = (tier, episode, step)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            episode_df = df[df["episode"] == episode]
            if episode_df.empty:
                continue
            if any(col not in episode_df.columns for col in required_cols):
                continue

            episode_slice = episode_df[required_cols].copy()
            episode_slice = episode_slice[episode_slice["step"] <= step]
            episode_slice = episode_slice.sort_values("step")
            episode_slice = episode_slice.rename(
                columns={
                    "conjecture": "statement",
                    "rew_conj_length": "diversity reward",
                    "rew_conj_formal": "formal reward",
                    "reward_ref": "refuter reward",
                }
            )
            table_key = f"{tier}_episode_{episode}_step_{step}"
            tables[table_key] = episode_slice[
                [
                    "episode",
                    "step",
                    "statement",
                    "diversity reward",
                    "formal reward",
                    "refuter reward",
                ]
            ]

    return tables


def _save_analysis_outputs(
    csv_basename,
    df,
    total_blocks,
    unique_block_count,
    tier_block_counters,
    tier_records,
    script_dir,
    output_root=None,
):
    """Persists analysis artifacts into tier-specific directories."""
    base_dir = output_root or script_dir
    output_dirs = {
        "basic_stats_chi": os.path.join(base_dir, "basic_stats_chi"),
        "conjectures_chi": os.path.join(base_dir, "conjectures_chi"),
        "trajectories": os.path.join(base_dir, "trajectories"),
    }

    for path in output_dirs.values():
        os.makedirs(path, exist_ok=True)

    basic_stats_df = _build_basic_stats_dataframe(
        total_blocks, unique_block_count, tier_block_counters
    )
    basic_stats_df = basic_stats_df.round(4)
    basic_stats_df.to_csv(
        os.path.join(output_dirs["basic_stats_chi"], f"result_analysis_{csv_basename}"),
        index=False,
    )

    conjectures_df = _build_conjectures_dataframe(tier_records)
    conjectures_df.to_csv(
        os.path.join(output_dirs["conjectures_chi"], f"result_analysis_{csv_basename}"),
        index=False,
    )

    trajectory_tables = _build_trajectory_tables(df, tier_records)
    trajectory_root = os.path.join(
        output_dirs["trajectories"], f"result_analysis_{csv_basename}"
    )
    if os.path.isdir(trajectory_root):
        shutil.rmtree(trajectory_root)
    elif os.path.isfile(trajectory_root):
        os.remove(trajectory_root)

    os.makedirs(trajectory_root, exist_ok=True)

    if not trajectory_tables:
        empty_columns = [
            "episode",
            "step",
            "statement",
            "diversity reward",
            "formal reward",
            "refuter reward",
        ]
        pd.DataFrame(columns=empty_columns).to_csv(
            os.path.join(trajectory_root, "no_positive_responses.csv"),
            index=False,
        )
    else:
        for table_key, table_df in trajectory_tables.items():
            output_path = os.path.join(trajectory_root, f"{table_key}.csv")
            table_df.to_csv(output_path, index=False)


def _build_betti_conjectures_dataframe(tier_records):
    """Creates a dataframe for Betti-related conjectures."""
    columns = [
        "tier",
        "betti_type",
        "episode",
        "step",
        "atomic_block",
        "conjecture",
        "formal_response",
    ]
    rows = []
    for tier in ("gold", "silver", "bronze"):
        for record in tier_records.get(tier, []):
            rows.append(
                {
                    "tier": tier,
                    "betti_type": record.get("betti_type"),
                    "episode": record.get("episode"),
                    "step": record.get("step"),
                    "atomic_block": record.get("atomic_block"),
                    "conjecture": record.get("conjecture"),
                    "formal_response": record.get("formal_response"),
                }
            )

    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows)
    sort_keys = [
        key for key in ("tier", "betti_type", "episode", "step") if key in df.columns
    ]
    if sort_keys:
        df = df.sort_values(sort_keys).reset_index(drop=True)
    return df[columns]


def _save_betti_outputs(
    csv_basename,
    total_blocks,
    unique_block_count,
    tier_block_counters,
    tier_records,
    script_dir,
    output_root,
    label,
    premises=None,
):
    """Persists Betti statistics and conjectures into dedicated directories."""
    base_dir = output_root or script_dir
    directory_map = {
        "b0": ("basic_stats_b0", "conjectures_b0"),
        "b1": ("basic_stats_b1", "conjectures_b1"),
        "homology": ("basic_stats_homology", "conjectures_homology"),
    }

    stats_dir_name, conjectures_dir_name = directory_map[label]
    stats_dir = os.path.join(base_dir, stats_dir_name)
    conjectures_dir = os.path.join(base_dir, conjectures_dir_name)

    os.makedirs(stats_dir, exist_ok=True)
    os.makedirs(conjectures_dir, exist_ok=True)

    stats_path = os.path.join(stats_dir, f"result_analysis_{csv_basename}")
    conjectures_path = os.path.join(conjectures_dir, f"result_analysis_{csv_basename}")

    if label == "b0" and premises and "connected" in premises:
        message_df = pd.DataFrame(
            {"message": ["connectedness is in the premises, b0 is not counted"]}
        )
        message_df.to_csv(stats_path, index=False)
        message_df.to_csv(conjectures_path, index=False)
        return

    basic_stats_df = _build_basic_stats_dataframe(
        total_blocks, unique_block_count, tier_block_counters
    )
    basic_stats_df = basic_stats_df.round(4)
    basic_stats_df.to_csv(stats_path, index=False)

    conjectures_df = _build_betti_conjectures_dataframe(tier_records)
    conjectures_df.to_csv(conjectures_path, index=False)


_formal_response_is_positive = is_formal_response_positive


def analyze_statistics(
    csv_file_name, premises=None, save_outputs=False, output_root=None
):
    """
    Main function to load CSV and perform all analyses.

    Parameters
    ----------
    csv_file_name : str
        Name of the CSV file relative to this script's directory.
    premises : list, optional
        Premises applied when simplifying the conjectures.
    save_outputs : bool, optional
        When True, persist summary data into tier-specific directories.
    output_root : str, optional
        Base directory for saved artifacts. Defaults to this script's directory.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        df, csv_path = load_conjecture_dataframe(csv_file_name, script_dir=script_dir)
        print(f"Attempting to load file from: {csv_path}\n")

        symbols_dict, h, w1, w2, _ = build_symbols_with_width_height_fallback(
            df,
            require_height_and_width2=True,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading CSV: {exc}")
        return
    except Exception as exc:
        print(f"Error loading CSV: {exc}")
        return

    # --- Process all rows ---
    all_atomic_blocks = []
    alternating_sum_count = 0
    gold_alternating_sum_conjectures = (
        set()
    )
    silver_alternating_sum_conjectures = (
        set()
    )
    bronze_alternating_sum_conjectures = (
        set()
    )

    gold_atomic_blocks = {}
    silver_atomic_blocks = {}
    bronze_atomic_blocks = {}
    tier_records = {"gold": [], "silver": [], "bronze": []}

    betti_atomic_blocks = {
        "b0": {"gold": {}, "silver": {}, "bronze": {}},
        "b1": {"gold": {}, "silver": {}, "bronze": {}},
    }
    betti_records = {
        "b0": {"gold": [], "silver": [], "bronze": []},
        "b1": {"gold": [], "silver": [], "bronze": []},
    }
    betti_conjectures = {
        "b0": {"gold": set(), "silver": set(), "bronze": set()},
        "b1": {"gold": set(), "silver": set(), "bronze": set()},
    }

    formalised_statement_count = 0
    formalised_with_chi_or_betti = 0

    def _record_betti_match(
        betti_label, tier_label, block_str, episode, step, conjecture, response
    ):
        counter = betti_atomic_blocks[betti_label][tier_label]
        counter[block_str] = counter.get(block_str, 0) + 1
        betti_conjectures[betti_label][tier_label].add((conjecture, response))
        betti_records[betti_label][tier_label].append(
            {
                "betti_type": betti_label,
                "episode": episode,
                "step": step,
                "atomic_block": block_str,
                "conjecture": conjecture,
                "formal_response": response,
            }
        )

    for index, row in df.iterrows():
        conjecture_str = row.get("conjecture")
        formal_response = row.get("formal_response")
        if pd.isna(conjecture_str):
            continue

        episode_val = row.get("episode")
        step_val = row.get("step")

        is_formalized = _formal_response_is_positive(formal_response)

        row_blocks = extract_equals_blocks(conjecture_str)
        all_atomic_blocks.extend(row_blocks)

        found_match_in_row = False
        found_match_in_row_silver = False
        found_match_in_row_bronze = False
        row_has_betti = False
        for block_str in row_blocks:
            diff = atomic_block_to_poly(block_str, symbols_dict, w1, premises=premises)
            if diff is False or diff is None:
                continue

            if check_alternating_sum(diff, h, w1, w2):
                alternating_sum_count += 1
                if block_str not in gold_atomic_blocks:
                    gold_atomic_blocks[block_str] = 0
                gold_atomic_blocks[block_str] += 1
                tier_records["gold"].append(
                    {
                        "episode": episode_val,
                        "step": step_val,
                        "atomic_block": block_str,
                        "conjecture": conjecture_str,
                        "formal_response": formal_response,
                    }
                )
                found_match_in_row = True
            else:
                if alternating_sum_with_rank_nullity(
                    diff, symbols_dict, tier="silver", premises=premises
                ):
                    silver_alternating_sum_conjectures.add(
                        (conjecture_str, formal_response)
                    )
                    if block_str not in silver_atomic_blocks:
                        silver_atomic_blocks[block_str] = 0
                    silver_atomic_blocks[block_str] += 1
                    tier_records["silver"].append(
                        {
                            "episode": episode_val,
                            "step": step_val,
                            "atomic_block": block_str,
                            "conjecture": conjecture_str,
                            "formal_response": formal_response,
                        }
                    )
                    found_match_in_row_silver = True
                else:
                    if alternating_sum_with_rank_nullity(
                        diff, symbols_dict, tier="bronze", premises=premises
                    ):
                        bronze_alternating_sum_conjectures.add(
                            (conjecture_str, formal_response)
                        )
                        if block_str not in bronze_atomic_blocks:
                            bronze_atomic_blocks[block_str] = 0
                        bronze_atomic_blocks[block_str] += 1
                        tier_records["bronze"].append(
                            {
                                "episode": episode_val,
                                "step": step_val,
                                "atomic_block": block_str,
                                "conjecture": conjecture_str,
                                "formal_response": formal_response,
                            }
                        )
                        found_match_in_row_bronze = True

            betti_gold_matches = check_betti_numbers(
                diff, symbols_dict, premises=premises
            )
            betti_matches_by_tier = {
                "gold": betti_gold_matches,
                "silver": set(),
                "bronze": set(),
            }
            unmatched_betti = {"b0", "b1"} - betti_gold_matches

            if unmatched_betti:
                silver_matches = betti_with_rank_nullity(
                    diff, symbols_dict, tier="silver", premises=premises
                )
                silver_matches = silver_matches - betti_matches_by_tier["gold"]
                betti_matches_by_tier["silver"] = silver_matches
                unmatched_betti -= silver_matches

            if unmatched_betti:
                bronze_matches = betti_with_rank_nullity(
                    diff, symbols_dict, tier="bronze", premises=premises
                )
                bronze_matches = (
                    bronze_matches
                    - betti_matches_by_tier["gold"]
                    - betti_matches_by_tier["silver"]
                )
                betti_matches_by_tier["bronze"] = bronze_matches

            for tier_label in ("gold", "silver", "bronze"):
                for betti_label in betti_matches_by_tier[tier_label]:
                    _record_betti_match(
                        betti_label,
                        tier_label,
                        block_str,
                        episode_val,
                        step_val,
                        conjecture_str,
                        formal_response,
                    )

            row_has_betti = row_has_betti or any(
                betti_matches_by_tier[tier_label]
                for tier_label in ("gold", "silver", "bronze")
            )

        if found_match_in_row:
            gold_alternating_sum_conjectures.add((conjecture_str, formal_response))
        elif found_match_in_row_silver:
            silver_alternating_sum_conjectures.add((conjecture_str, formal_response))
        elif found_match_in_row_bronze:
            bronze_alternating_sum_conjectures.add((conjecture_str, formal_response))

        if is_formalized:
            formalised_statement_count += 1
            if (
                found_match_in_row
                or found_match_in_row_silver
                or found_match_in_row_bronze
                or row_has_betti
            ):
                formalised_with_chi_or_betti += 1

    homology_atomic_counters = {
        tier: Counter() for tier in ("gold", "silver", "bronze")
    }
    homology_records = {"gold": [], "silver": [], "bronze": []}
    homology_conjectures = {"gold": set(), "silver": set(), "bronze": set()}

    for betti_label in ("b0", "b1"):
        for tier_label in ("gold", "silver", "bronze"):
            homology_atomic_counters[tier_label].update(
                betti_atomic_blocks[betti_label][tier_label]
            )
            homology_records[tier_label].extend(betti_records[betti_label][tier_label])
            homology_conjectures[tier_label].update(
                betti_conjectures[betti_label][tier_label]
            )

    # --- Task 1: Total Number of Atomic Blocks ---
    total_blocks = len(all_atomic_blocks)
    print("--- Formalized Statement Counts ---")
    print(f"Total formalized statements: {formalised_statement_count}")
    print(
        f"Formalized statements with chi or Betti blocks: {formalised_with_chi_or_betti}\n"
    )

    print("--- Task 1: Total Atomic Blocks ---")
    print(f"Total number of atomic blocks found: {total_blocks}\n")

    # --- Task 2: Frequency of All Atomic Blocks ---
    block_frequency = Counter(all_atomic_blocks)
    print("--- Task 2: Atomic Block Frequency (Top 20) ---")
    if total_blocks == 0:
        print("No atomic blocks found.")
    else:
        print(f"{'Count':>5} {'Prob.':>7}   {'Atomic Block'}")
        print("-" * 60)
        for block, count in block_frequency.most_common(20):
            probability = count / total_blocks
            print(f"{count:>5} ({probability:>6.2%}) : {block}")
    print("\n")

    # --- Task 3: Alternating Sum Frequency ---
    print("--- Task 3: Alternating Sum Frequency ---")
    print(f"Found {alternating_sum_count} atomic blocks representing the form:")
    print("k * (height_D1 - width_D1 + width_D2) = C")
    print("(where C/k does not contain any of height_D1, width_D1, width_D2)\n")

    print("--- Full Conjectures Containing Alternating Sum ---")
    if not gold_alternating_sum_conjectures:
        print("Gold not found.")
    else:
        print(
            f"Found {len(gold_alternating_sum_conjectures)} unique 'gold'-conjectures containing this form:"
        )
        for i, (conj_str, formal_response) in enumerate(
            sorted(list(gold_alternating_sum_conjectures))
        ):
            print(f"\n{i+1}: {conj_str}, formal = {formal_response}")
    print("\n")
    if not silver_alternating_sum_conjectures:
        print("Silver not found.")
    else:
        print(
            f"Found {len(silver_alternating_sum_conjectures)} unique 'silver'-conjectures containing this form:"
        )
        for i, (conj_str, formal_response) in enumerate(
            sorted(list(silver_alternating_sum_conjectures))
        ):
            print(f"\n{i+1}: {conj_str}, formal = {formal_response}")
    print("\n")
    if not bronze_alternating_sum_conjectures:
        print("Bronze not found.")
    else:
        print(
            f"Found {len(bronze_alternating_sum_conjectures)} unique 'bronze'-conjectures containing this form:"
        )
        for i, (conj_str, formal_response) in enumerate(
            sorted(list(bronze_alternating_sum_conjectures))
        ):
            print(f"\n{i+1}: {conj_str}, formal = {formal_response}")
    print("\n")

    # Frequency of gold conjectures
    print("--- Frequency of Gold Atomic Blocks ---")
    if not gold_atomic_blocks:
        print("No gold atomic blocks found.")
    else:
        total_gold_blocks = sum(gold_atomic_blocks.values())
        print(f"Total gold atomic blocks found: {total_gold_blocks}")
        print("Gold atomic block frequencies:")
        for block_str, count in gold_atomic_blocks.items():
            print(f" - {block_str}: {count}")
        print("\n")
        print(
            f"Overall Frequencies: {(total_gold_blocks/total_blocks)*100:.2f}% of all atomic blocks were gold.\n"
        )
    # Frequency of silver conjectures
    print("--- Frequency of Silver Atomic Blocks ---")
    if not silver_atomic_blocks:
        print("No silver atomic blocks found.")
    else:
        total_silver_blocks = sum(silver_atomic_blocks.values())
        print(f"Total silver atomic blocks found: {total_silver_blocks}")
        print("Silver atomic block frequencies:")
        for block_str, count in silver_atomic_blocks.items():
            print(f" - {block_str}: {count}")
        print("\n")
        print(
            f"Overall Frequencies: {(total_silver_blocks/total_blocks)*100:.2f}% of all atomic blocks were silver.\n"
        )
    # Frequency of bronze conjectures
    print("--- Frequency of Bronze Atomic Blocks ---")
    if not bronze_atomic_blocks:
        print("No bronze atomic blocks found.")
    else:
        total_bronze_blocks = sum(bronze_atomic_blocks.values())
        print(f"Total bronze atomic blocks found: {total_bronze_blocks}")
        print("Bronze atomic block frequencies:")
        for block_str, count in bronze_atomic_blocks.items():
            print(f" - {block_str}: {count}")
        print("\n")
        print(
            f"Overall Frequencies: {(total_bronze_blocks/total_blocks)*100:.2f}% of all atomic blocks were bronze.\n"
        )

    betti_label_descriptions = {
        "b0": "height_D1 - rank_D1",
        "b1": "nullity_D1 - rank_D2",
    }

    print("--- Betti Detection ---")
    for betti_label in ("b0", "b1"):
        tier_totals = {
            tier_label: sum(betti_atomic_blocks[betti_label][tier_label].values())
            for tier_label in ("gold", "silver", "bronze")
        }
        total_matches = sum(tier_totals.values())
        description = betti_label_descriptions[betti_label]
        print(f"{betti_label.upper()} ({description}):")

        if total_matches == 0:
            print("  No detections.\n")
            continue

        unique_blocks = len(
            {
                block
                for tier_label in ("gold", "silver", "bronze")
                for block in betti_atomic_blocks[betti_label][tier_label]
            }
        )
        conjecture_union = set()
        for tier_label in ("gold", "silver", "bronze"):
            conjecture_union.update(betti_conjectures[betti_label][tier_label])

        print(f"  Total atomic blocks: {total_matches}")
        print(f"  Unique atomic blocks: {unique_blocks}")
        print(f"  Unique conjectures: {len(conjecture_union)}")
        for tier_label in ("gold", "silver", "bronze"):
            tier_count = tier_totals[tier_label]
            if tier_count:
                print(
                    f"  {tier_label.title()} blocks: {tier_count}, {(tier_count/total_matches)*100:.2f}% of {betti_label}"
                )
        print("")

    print("--- Homology Summary (b0 + b1) ---")
    homology_tier_totals = {
        tier_label: sum(homology_atomic_counters[tier_label].values())
        for tier_label in ("gold", "silver", "bronze")
    }
    total_homology_blocks = sum(homology_tier_totals.values())

    if total_homology_blocks == 0:
        print("No homology detections.\n")
    else:
        unique_homology_blocks = len(
            {
                block
                for tier_label in ("gold", "silver", "bronze")
                for block in homology_atomic_counters[tier_label]
            }
        )
        homology_conjecture_union = set()
        for tier_label in ("gold", "silver", "bronze"):
            homology_conjecture_union.update(homology_conjectures[tier_label])

        print(f"Total homology atomic blocks: {total_homology_blocks}")
        print(f"Unique homology atomic blocks: {unique_homology_blocks}")
        print(f"Unique homology conjectures: {len(homology_conjecture_union)}")
        for tier_label in ("gold", "silver", "bronze"):
            tier_count = homology_tier_totals[tier_label]
            if tier_count:
                print(f"{tier_label.title()} homology blocks: {tier_count}")
        print("\n")

    unique_block_count = len(block_frequency)
    if save_outputs:
        csv_basename = os.path.basename(csv_file_name)
        tier_block_counters = {
            "gold": gold_atomic_blocks,
            "silver": silver_atomic_blocks,
            "bronze": bronze_atomic_blocks,
        }
        _save_analysis_outputs(
            csv_basename,
            df,
            total_blocks,
            unique_block_count,
            tier_block_counters,
            tier_records,
            script_dir,
            output_root=output_root,
        )

        _save_betti_outputs(
            csv_basename,
            total_blocks,
            unique_block_count,
            betti_atomic_blocks["b0"],
            betti_records["b0"],
            script_dir,
            output_root,
            "b0",
            premises=premises,
        )

        _save_betti_outputs(
            csv_basename,
            total_blocks,
            unique_block_count,
            betti_atomic_blocks["b1"],
            betti_records["b1"],
            script_dir,
            output_root,
            "b1",
            premises=premises,
        )

        homology_atomic_blocks = {
            tier_label: dict(homology_atomic_counters[tier_label])
            for tier_label in ("gold", "silver", "bronze")
        }

        _save_betti_outputs(
            csv_basename,
            total_blocks,
            unique_block_count,
            homology_atomic_blocks,
            homology_records,
            script_dir,
            output_root,
            "homology",
            premises=premises,
        )


def analyze_combined_statistics(
    csv_file_names,
    combined_csv_name,
    premises=None,
    save_outputs=False,
    output_root=None,
):
    """Row-wise combine several CSVs, persist the merge, and run the analysis once."""

    if not csv_file_names:
        print("No CSV files provided for combination.")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataframes = []

    for name in csv_file_names:
        csv_path = os.path.join(script_dir, name)
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"Error: The file was not found at '{csv_path}'.")
            return
        except Exception as exc:
            print(f"Error loading '{csv_path}': {exc}")
            return

        df = df.copy()
        df["source_csv"] = name
        dataframes.append(df)

    if not dataframes:
        print("No data could be loaded; aborting combination.")
        return

    combined_df = pd.concat(dataframes, ignore_index=True)

    if not combined_csv_name.lower().endswith(".csv"):
        combined_csv_name = f"{combined_csv_name}.csv"

    combined_dir = os.path.join(script_dir, "combined_inputs")
    os.makedirs(combined_dir, exist_ok=True)

    combined_path = os.path.join(combined_dir, combined_csv_name)
    combined_df.to_csv(combined_path, index=False)
    print(f"Combined CSV written to: {combined_path}")

    relative_path = os.path.relpath(combined_path, script_dir)
    analyze_statistics(
        relative_path,
        premises=premises,
        save_outputs=save_outputs,
        output_root=output_root,
    )


def _prepare_baseline_context(csv_file_path, required_columns=None):
    """Load baseline CSV and return shared symbol context."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df, _ = load_conjecture_dataframe(
        csv_file_path,
        script_dir=script_dir,
        required_columns=required_columns,
    )
    symbols_dict, h, w1, w2, _ = build_symbols_with_width_height_fallback(
        df,
        require_height_and_width2=True,
    )
    return df, symbols_dict, h, w1, w2


def _build_tier_summary_df(
    tier_counts,
    tier_conjectures,
    tier_atomic_block_sets,
    total_atomic_blocks,
    total_unique_atomic_blocks,
    total_conjectures,
):
    summary_rows = []
    for tier in ("gold", "silver", "bronze"):
        atomic_blocks = tier_counts[tier]
        unique_blocks = len(tier_atomic_block_sets[tier])
        unique_conjs = len(tier_conjectures[tier])
        summary_rows.append(
            {
                "tier": tier,
                "atomic_blocks": atomic_blocks,
                "pct_atomic_blocks": (
                    (atomic_blocks / total_atomic_blocks * 100.0)
                    if total_atomic_blocks
                    else 0.0
                ),
                "unique_atomic_blocks": unique_blocks,
                "pct_unique_atomic_blocks": (
                    (unique_blocks / total_unique_atomic_blocks * 100.0)
                    if total_unique_atomic_blocks
                    else 0.0
                ),
                "unique_conjectures": unique_conjs,
                "pct_unique_conjectures": (
                    (unique_conjs / total_conjectures * 100.0)
                    if total_conjectures
                    else 0.0
                ),
            }
        )

    return pd.DataFrame(summary_rows)[
        [
            "tier",
            "atomic_blocks",
            "pct_atomic_blocks",
            "unique_atomic_blocks",
            "pct_unique_atomic_blocks",
            "unique_conjectures",
            "pct_unique_conjectures",
        ]
    ].round(4)


def _run_tiered_baseline_scan(df, symbols_dict, w1, tier_detector, premises=None):
    """Shared tier-scanning loop used by baseline analyses."""
    premises = premises or []
    tier_counts = Counter({"gold": 0, "silver": 0, "bronze": 0})
    tier_conjectures = {tier: set() for tier in ("gold", "silver", "bronze")}
    tier_atomic_block_sets = {tier: set() for tier in ("gold", "silver", "bronze")}
    total_atomic_blocks = 0
    unique_atomic_blocks = set()

    for conj in df["conjecture"]:
        blocks = extract_equals_blocks(conj)
        total_atomic_blocks += len(blocks)
        unique_atomic_blocks.update(blocks)

        found_gold = found_silver = found_bronze = False
        for block_str in blocks:
            diff = atomic_block_to_poly(block_str, symbols_dict, w1, premises=premises)
            if diff is False or diff is None:
                continue

            tier = tier_detector(diff)
            if tier is None:
                continue

            tier_counts[tier] += 1
            tier_atomic_block_sets[tier].add(block_str)
            if tier == "gold":
                found_gold = True
            elif tier == "silver":
                found_silver = True
            elif tier == "bronze":
                found_bronze = True

        if found_gold:
            tier_conjectures["gold"].add(conj)
        elif found_silver:
            tier_conjectures["silver"].add(conj)
        elif found_bronze:
            tier_conjectures["bronze"].add(conj)

    total_conjectures = len(df)
    total_unique_atomic_blocks = len(unique_atomic_blocks)
    summary_df = _build_tier_summary_df(
        tier_counts,
        tier_conjectures,
        tier_atomic_block_sets,
        total_atomic_blocks,
        total_unique_atomic_blocks,
        total_conjectures,
    )

    return {
        "tier_counts": tier_counts,
        "tier_conjectures": tier_conjectures,
        "tier_atomic_block_sets": tier_atomic_block_sets,
        "total_atomic_blocks": total_atomic_blocks,
        "total_unique_atomic_blocks": total_unique_atomic_blocks,
        "total_conjectures": total_conjectures,
        "summary": summary_df,
    }


def baseline_alternating_sum(csv_file_path, premises=None, output_csv_path=None):
    df, symbols_dict, h, w1, w2 = _prepare_baseline_context(csv_file_path)

    def _tier_detector(diff):
        if check_alternating_sum(diff, h, w1, w2):
            return "gold"
        if alternating_sum_with_rank_nullity(
            diff, symbols_dict, tier="silver", premises=premises
        ):
            return "silver"
        if alternating_sum_with_rank_nullity(
            diff, symbols_dict, tier="bronze", premises=premises
        ):
            return "bronze"
        return None

    result = _run_tiered_baseline_scan(
        df,
        symbols_dict,
        w1,
        _tier_detector,
        premises=premises,
    )
    tier_counts = result["tier_counts"]
    tier_conjectures = result["tier_conjectures"]
    tier_atomic_block_sets = result["tier_atomic_block_sets"]
    total_atomic_blocks = result["total_atomic_blocks"]
    total_unique_atomic_blocks = result["total_unique_atomic_blocks"]
    total_conjectures = result["total_conjectures"]
    summary_df = result["summary"]

    print("--- Alternating Sum Baseline ---")
    print(f"Total atomic blocks: {total_atomic_blocks}")
    print(f"Unique atomic blocks: {total_unique_atomic_blocks}")
    print(f"Total conjectures: {total_conjectures}")
    for tier in ("gold", "silver", "bronze"):
        block_count = tier_counts[tier]
        block_pct = (
            (block_count / total_atomic_blocks * 100.0) if total_atomic_blocks else 0.0
        )
        unique_block_count = len(tier_atomic_block_sets[tier])
        unique_block_pct = (
            (unique_block_count / total_unique_atomic_blocks * 100.0)
            if total_unique_atomic_blocks
            else 0.0
        )
        unique_conj_count = len(tier_conjectures[tier])
        unique_conj_pct = (
            (unique_conj_count / total_conjectures * 100.0)
            if total_conjectures
            else 0.0
        )
        print(
            f"{tier.title()}: {block_count} blocks ({block_pct:.2f}%), "
            f"{unique_block_count} unique blocks ({unique_block_pct:.2f}%), "
            f"{unique_conj_count} conjectures ({unique_conj_pct:.2f}%)"
        )
    print("")

    if output_csv_path:
        output_dir = os.path.dirname(output_csv_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        summary_df.to_csv(output_csv_path, index=False)

    return {
        "total_atomic_blocks": total_atomic_blocks,
        "total_conjectures": total_conjectures,
        "summary": summary_df,
    }


def baseline_b1(csv_file_path, premises=None, output_csv_path=None):
    df, symbols_dict, _, w1, _ = _prepare_baseline_context(csv_file_path)

    def _tier_detector(diff):
        if "b1" in check_betti_numbers(diff, symbols_dict, premises=premises):
            return "gold"
        if "b1" in betti_with_rank_nullity(
            diff,
            symbols_dict,
            tier="silver",
            premises=premises,
        ):
            return "silver"
        if "b1" in betti_with_rank_nullity(
            diff,
            symbols_dict,
            tier="bronze",
            premises=premises,
        ):
            return "bronze"
        return None

    result = _run_tiered_baseline_scan(
        df,
        symbols_dict,
        w1,
        _tier_detector,
        premises=premises,
    )
    tier_counts = result["tier_counts"]
    tier_conjectures = result["tier_conjectures"]
    tier_atomic_block_sets = result["tier_atomic_block_sets"]
    total_atomic_blocks = result["total_atomic_blocks"]
    total_unique_atomic_blocks = result["total_unique_atomic_blocks"]
    total_conjectures = result["total_conjectures"]
    summary_df = result["summary"]

    print("--- Betti b1 Baseline ---")
    print(f"Total atomic blocks: {total_atomic_blocks}")
    print(f"Unique atomic blocks: {total_unique_atomic_blocks}")
    print(f"Total conjectures: {total_conjectures}")
    for tier in ("gold", "silver", "bronze"):
        block_count = tier_counts[tier]
        block_pct = (
            (block_count / total_atomic_blocks * 100.0) if total_atomic_blocks else 0.0
        )
        unique_block_count = len(tier_atomic_block_sets[tier])
        unique_block_pct = (
            (unique_block_count / total_unique_atomic_blocks * 100.0)
            if total_unique_atomic_blocks
            else 0.0
        )
        unique_conj_count = len(tier_conjectures[tier])
        unique_conj_pct = (
            (unique_conj_count / total_conjectures * 100.0)
            if total_conjectures
            else 0.0
        )
        print(
            f"{tier.title()}: {block_count} blocks ({block_pct:.2f}%), "
            f"{unique_block_count} unique blocks ({unique_block_pct:.2f}%), "
            f"{unique_conj_count} conjectures ({unique_conj_pct:.2f}%)"
        )
    print("")

    if output_csv_path:
        output_dir = os.path.dirname(output_csv_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        summary_df.to_csv(output_csv_path, index=False)

    return {
        "total_atomic_blocks": total_atomic_blocks,
        "total_unique_atomic_blocks": total_unique_atomic_blocks,
        "total_conjectures": total_conjectures,
        "summary": summary_df,
    }


def formalize_conjecture(conjecture: str, run_ts: int, env_id: str = "ga") -> int:
    """
    Formalize a conjecture using the predefined formalization tool,
    i.e, convert to lean and check with `grind' tactic.
    ----------
    conjecture : str
        conjecture to formalize
    """
    conjecture = fix_conj_str(conjecture)
    try:
        lean_stmt = translate_fol_to_lean_with_grind(fol_str=conjecture)
    except Exception as e:
        print(f"Error translating conjecture '{conjecture}': {e}\n Applying custom fix")
        form_response = 0
        return form_response

    form_response = check_theorem_no_leancopilot(
        lean_stmt=lean_stmt, env_id=env_id, run_ts=run_ts
    )
    return form_response


def count_formal_response(csv_file_path: str) -> List[int]:
    """
    For baseline ga csv's, try to formalize each conjecture and return the formal responses.
    ----------
    csv_file_path : str
        baseline ga csv to compute formal responses for
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    resolved_csv_path = (
        csv_file_path
        if os.path.isabs(csv_file_path)
        else os.path.join(script_dir, csv_file_path)
    )
    if not os.path.isfile(resolved_csv_path):
        raise FileNotFoundError(f"CSV not found at '{resolved_csv_path}'")

    df = pd.read_csv(resolved_csv_path)
    if "conjecture" not in df.columns:
        raise ValueError("CSV must contain a 'conjecture' column.")
    df = df.dropna(subset=["conjecture"])
    if df.empty:
        raise ValueError("CSV contains no conjectures.")

    formal_responses = []
    for conj in df["conjecture"]:
        run_ts = len(formal_responses)
        response = formalize_conjecture(conj, run_ts=run_ts, env_id="ga")
        formal_responses.append(response)

    return formal_responses


def count_formal_response_with_chi_and_homology(
    csv_file_path: str, premises=None
) -> Tuple[int, List[int]]:
    """
    For baseline ga csv's with their formal responses, check which of the formalized conjectures
    have atomic blocks corresponding to chi or homology.
    Return a list indicating which conjectures are formalized and have chi/homology blocks.
    1 indicates formalized with chi/homology blocks, 0 otherwise.
    ----------
    csv_file_path : str
        baseline ga with formal csv to check
    premises : List[str]
        list of premises to use in the analysis
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df, _ = load_conjecture_dataframe(
        csv_file_path,
        script_dir=script_dir,
        required_columns=["formal_response"],
    )
    symbols_dict, h1_symbol, w1_symbol, w2_symbol, _ = (
        build_symbols_with_width_height_fallback(df)
    )
    formal_responses = []
    num_formalized = 0
    for index, row in df.iterrows():
        conjecture_str = row["conjecture"]
        formal_response = row["formal_response"]
        is_formalized = _formal_response_is_positive(formal_response)
        has_chi_or_homology = 0
        if is_formalized:
            num_formalized += 1
            blocks = extract_equals_blocks(conjecture_str)
            for block_str in blocks:
                diff = atomic_block_to_poly(
                    block_str, symbols_dict, w1_symbol, premises=premises
                )
                if diff is False or diff is None:
                    continue
                if check_alternating_sum(diff, h1_symbol, w1_symbol, w2_symbol):
                    has_chi_or_homology = 1
                    break
                betti_matches = check_betti_numbers(
                    diff, symbols_dict, premises=premises
                )
                if betti_matches:
                    has_chi_or_homology = 1
                    break
        formal_responses.append(has_chi_or_homology)

    return num_formalized, formal_responses


def bootstrap_diff_ci(a, b, B=10000, alpha=0.05, seed=0):
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    b = np.asarray(b, dtype=float)
    b = b[~np.isnan(b)]
    rng = np.random.default_rng(seed)
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return np.nan, (np.nan, np.nan)

    deltas = np.empty(B)
    for i in range(B):
        a_star = rng.choice(a, size=n1, replace=True)
        b_star = rng.choice(b, size=n2, replace=True)
        deltas[i] = a_star.mean() - b_star.mean()

    delta_hat = a.mean() - b.mean()
    lo = np.quantile(deltas, alpha / 2)
    hi = np.quantile(deltas, 1 - alpha / 2)
    return delta_hat, (lo, hi)


def compute_episode_statistics(
    csv_file_path: str, premises=None
) -> Tuple[List[int], List[int], List[int], List[int]]:
    # Compute per episode occurence of alternating sum atomic blocks and homology atomic blocks
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df, _ = load_conjecture_dataframe(
        csv_file_path,
        script_dir=script_dir,
        required_columns=["episode", "formal_response"],
    )

    # Get episode-wise data
    episode_data = {}
    for index, row in df.iterrows():
        episode = row["episode"]
        conjecture_str = row["conjecture"]
        if episode not in episode_data:
            episode_data[episode] = []
        episode_data[episode].append(conjecture_str)

    episode_alternating_sum_counts = []
    episode_homology_counts = []
    episode_proven_with_chi_or_homology = []
    episode_unique_atomic_blocks = []

    symbols_dict, h1_symbol, w1_symbol, w2_symbol, _ = (
        build_symbols_with_width_height_fallback(df)
    )

    for episode, conjectures in episode_data.items():
        alternating_sum_count = 0
        homology_count = 0
        proven_with_chi_or_homology_in_episode = 0
        unique_atomic_blocks_in_episode = set()
        for conjecture_str in conjectures:
            has_chi_or_homology = False
            blocks = extract_equals_blocks(conjecture_str)
            unique_atomic_blocks_in_episode.update(blocks)
            for block_str in blocks:
                diff = atomic_block_to_poly(
                    block_str, symbols_dict, w1_symbol, premises=premises
                )
                if diff is False or diff is None:
                    continue
                if check_alternating_sum(diff, h1_symbol, w1_symbol, w2_symbol):
                    alternating_sum_count += 1
                    has_chi_or_homology = True
                elif alternating_sum_with_rank_nullity(
                    diff, symbols_dict, premises=premises
                ):
                    alternating_sum_count += 1
                    has_chi_or_homology = True
                betti_matches = check_betti_numbers(
                    diff, symbols_dict, premises=premises
                )
                if betti_matches:
                    homology_count += 1
                    has_chi_or_homology = True
                elif betti_with_rank_nullity(diff, symbols_dict, premises=premises):
                    homology_count += 1
                    has_chi_or_homology = True
            if (
                has_chi_or_homology
                and df.loc[
                    df["conjecture"] == conjecture_str, "formal_response"
                ].values[0]
                == 1
            ):
                proven_with_chi_or_homology_in_episode += 1

        episode_alternating_sum_counts.append(alternating_sum_count)
        episode_homology_counts.append(homology_count)
        episode_proven_with_chi_or_homology.append(
            proven_with_chi_or_homology_in_episode
        )
        episode_unique_atomic_blocks.append(len(unique_atomic_blocks_in_episode))

    no_of_episodes = len(episode_data)
    no_of_formals_per_episode = np.array(
        [sum(df.loc[df["episode"] == ep, "formal_response"]) for ep in episode_data]
    )
    episode_lengths = np.asarray(
        [len(df.loc[df["episode"] == ep]) for ep in episode_data], dtype=float
    )

    return (
        episode_unique_atomic_blocks,
        episode_alternating_sum_counts,
        episode_homology_counts,
        episode_proven_with_chi_or_homology,
        episode_unique_atomic_blocks,
        no_of_episodes,
        no_of_formals_per_episode,
        episode_lengths,
    )


def compute_confidence_interval(
    csv_file_a: str, csv_file_b: str = None, premises=None, confidence_level=0.95
) -> Tuple[float, float]:
    # Compute per episode occurence of alternating sum atomic blocks, homology atomic blocks and confidence intervals
    script_dir = os.path.dirname(os.path.abspath(__file__))
    resolved_csv_path_a = (
        csv_file_a
        if os.path.isabs(csv_file_a)
        else os.path.join(script_dir, csv_file_a)
    )
    if not os.path.isfile(resolved_csv_path_a):
        raise FileNotFoundError(f"CSV not found at '{resolved_csv_path_a}'")
    df_a = pd.read_csv(resolved_csv_path_a)
    if "conjecture" not in df_a.columns:
        raise ValueError("CSV must contain a 'conjecture' column.")
    df_a = df_a.dropna(subset=["conjecture"])
    if df_a.empty:
        raise ValueError("CSV contains no conjectures.")
    if "episode" not in df_a.columns:
        raise ValueError("CSV must contain an 'episode' column.")

    if csv_file_b is not None:
        resolved_csv_path_b = (
            csv_file_b
            if os.path.isabs(csv_file_b)
            else os.path.join(script_dir, csv_file_b)
        )
        if not os.path.isfile(resolved_csv_path_b):
            raise FileNotFoundError(f"CSV not found at '{resolved_csv_path_b}'")
        df_b = pd.read_csv(resolved_csv_path_b)
        if "conjecture" not in df_b.columns:
            raise ValueError("CSV must contain a 'conjecture' column.")
        df_b = df_b.dropna(subset=["conjecture"])
        if df_b.empty:
            raise ValueError("CSV contains no conjectures.")
        if "episode" not in df_b.columns:
            raise ValueError("CSV must contain an 'episode' column.")

    # Get episode-wise data
    (
        episode_unique_atomic_blocks_a,
        episode_alternating_sum_counts_a,
        episode_homology_counts_a,
        episode_proven_with_chi_or_homology_a,
        episode_unique_atomic_blocks_a,
        no_of_episodes_a,
        no_of_formals_per_episode_a,
        episode_lengths_a,
    ) = compute_episode_statistics(csv_file_a, premises=premises)

    if csv_file_b is not None:
        (
            episode_unique_atomic_blocks_b,
            episode_alternating_sum_counts_b,
            episode_homology_counts_b,
            episode_proven_with_chi_or_homology_b,
            episode_unique_atomic_blocks_b,
            no_of_episodes_b,
            no_of_formals_per_episode_b,
            episode_lengths_b,
        ) = compute_episode_statistics(csv_file_b, premises=premises)

    # Stats for CSV A
    median_steps_per_episode_a = np.median(episode_lengths_a)
    q1_a = np.percentile(episode_lengths_a, 25)
    q3_a = np.percentile(episode_lengths_a, 75)
    per_episode_chi_freq_a = np.array(episode_alternating_sum_counts_a) / (
        np.array(episode_unique_atomic_blocks_a)
    )
    per_episode_homology_freq_a = np.array(episode_homology_counts_a) / (
        np.array(episode_unique_atomic_blocks_a)
    )
    per_episode_proven_with_chi_or_homology_freq_a = np.array(
        episode_proven_with_chi_or_homology_a
    ) / (np.array(no_of_formals_per_episode_a))
    mean_chi_a, (ci_chi_lower_a, ci_chi_upper_a) = bootstrap_mean_ci(
        per_episode_chi_freq_a, B=10000
    )
    mean_homology_a, (ci_homology_lower_a, ci_homology_upper_a) = bootstrap_mean_ci(
        per_episode_homology_freq_a, B=10000
    )
    proven_mean_a, (ci_proven_lower_a, ci_proven_upper_a) = bootstrap_mean_ci(
        per_episode_proven_with_chi_or_homology_freq_a, B=10000
    )
    cov_np_a = np.cov(episode_unique_atomic_blocks_a, per_episode_chi_freq_a, ddof=1)[
        0, 1
    ]
    corr_np_a = np.corrcoef(episode_unique_atomic_blocks_a, per_episode_chi_freq_a)[
        0, 1
    ]
    chi_pooled_a = np.sum(per_episode_chi_freq_a) / np.sum(
        episode_unique_atomic_blocks_a
    )
    decomposition_gap_a = chi_pooled_a - np.mean(per_episode_chi_freq_a)
    prediction_gap_a = cov_np_a / np.mean(episode_unique_atomic_blocks_a)

    # Stats for CSV B
    if csv_file_b is not None:
        median_steps_per_episode_b = np.median(episode_lengths_b)
        q1_b = np.percentile(episode_lengths_b, 25)
        q3_b = np.percentile(episode_lengths_b, 75)
        per_episode_chi_freq_b = np.array(episode_alternating_sum_counts_b) / (
            np.array(episode_unique_atomic_blocks_b)
        )
        per_episode_homology_freq_b = np.array(episode_homology_counts_b) / (
            np.array(episode_unique_atomic_blocks_b)
        )
        per_episode_proven_with_chi_or_homology_freq_b = np.array(
            episode_proven_with_chi_or_homology_b
        ) / (np.array(no_of_formals_per_episode_b))
        mean_chi_b, (ci_chi_lower_b, ci_chi_upper_b) = bootstrap_mean_ci(
            per_episode_chi_freq_b, B=10000
        )
        mean_homology_b, (ci_homology_lower_b, ci_homology_upper_b) = bootstrap_mean_ci(
            per_episode_homology_freq_b, B=10000
        )
        proven_mean_b, (ci_proven_lower_b, ci_proven_upper_b) = bootstrap_mean_ci(
            per_episode_proven_with_chi_or_homology_freq_b, B=10000
        )
        cov_np_b = np.cov(
            episode_unique_atomic_blocks_b, per_episode_chi_freq_b, ddof=1
        )[0, 1]
        corr_np_b = np.corrcoef(episode_unique_atomic_blocks_b, per_episode_chi_freq_b)[
            0, 1
        ]
        chi_pooled_b = np.sum(per_episode_chi_freq_b) / np.sum(
            episode_unique_atomic_blocks_b
        )
        decomposition_gap_b = chi_pooled_b - np.mean(per_episode_chi_freq_b)
        prediction_gap_b = cov_np_b / np.mean(episode_unique_atomic_blocks_b)

    # Difference statistics
    if csv_file_b is not None:
        delta_chi, (delta_chi_lower, delta_chi_upper) = bootstrap_diff_ci(
            per_episode_chi_freq_a, per_episode_chi_freq_b, B=10000
        )
        delta_homology, (delta_homology_lower, delta_homology_upper) = (
            bootstrap_diff_ci(
                per_episode_homology_freq_a, per_episode_homology_freq_b, B=10000
            )
        )
        delta_proven, (delta_proven_lower, delta_proven_upper) = bootstrap_diff_ci(
            per_episode_proven_with_chi_or_homology_freq_a,
            per_episode_proven_with_chi_or_homology_freq_b,
            B=10000,
        )

    if csv_file_b is not None:
        return (
            (mean_chi_a, ci_chi_lower_a, ci_chi_upper_a),
            (mean_homology_a, ci_homology_lower_a, ci_homology_upper_a),
            (proven_mean_a, ci_proven_lower_a, ci_proven_upper_a),
            (cov_np_a, corr_np_a, chi_pooled_a, decomposition_gap_a, prediction_gap_a),
            median_steps_per_episode_a,
            (q1_a, q3_a),
            (mean_chi_b, ci_chi_lower_b, ci_chi_upper_b),
            (mean_homology_b, ci_homology_lower_b, ci_homology_upper_b),
            (proven_mean_b, ci_proven_lower_b, ci_proven_upper_b),
            (cov_np_b, corr_np_b, chi_pooled_b, decomposition_gap_b, prediction_gap_b),
            median_steps_per_episode_b,
            (q1_b, q3_b),
            (delta_chi, delta_chi_lower, delta_chi_upper),
            (delta_homology, delta_homology_lower, delta_homology_upper),
            (delta_proven, delta_proven_lower, delta_proven_upper),
        )
    else:
        return (
            (mean_chi_a, ci_chi_lower_a, ci_chi_upper_a),
            (mean_homology_a, ci_homology_lower_a, ci_homology_upper_a),
            (proven_mean_a, ci_proven_lower_a, ci_proven_upper_a),
            (cov_np_a, corr_np_a, chi_pooled_a, decomposition_gap_a, prediction_gap_a),
            median_steps_per_episode_a,
            (q1_a, q3_a),
        )


def parse_args(argv=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output_root = os.path.dirname(script_dir)

    parser = argparse.ArgumentParser(
        description="Analyze block-level statistics for a model-evaluation CSV."
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="eval_noise_orig_data_same_formal_reward.csv",
        help="CSV file path or file name to analyze",
    )
    parser.add_argument(
        "--premise",
        dest="premises",
        action="append",
        default=None,
        help="Premise to apply (repeatable); defaults to rank-nullity, connected, orientable",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=default_output_root,
        help="Directory where analysis outputs are saved",
    )
    parser.add_argument(
        "--no-save-outputs",
        action="store_true",
        help="Do not save output artifacts to disk",
    )
    parser.add_argument(
        "--skip-formalized-summary",
        action="store_true",
        help="Skip counting formalized conjectures with chi/homology blocks",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    premises = args.premises or ["rank-nullity", "connected", "orientable"]

    analyze_statistics(
        args.csv_name,
        premises=premises,
        save_outputs=not args.no_save_outputs,
        output_root=args.output_root,
    )

    if args.skip_formalized_summary:
        return

    num_formalized, formal_resp_with_chi_homology = (
        count_formal_response_with_chi_and_homology(args.csv_name, premises=premises)
    )
    num_formalized_with_chi_homology = len(
        [r for r in formal_resp_with_chi_homology if r == 1]
    )
    pct_formalized_with_chi_homology = (
        (num_formalized_with_chi_homology / len(formal_resp_with_chi_homology)) * 100.0
        if formal_resp_with_chi_homology
        else 0.0
    )
    print(f"Number of formalised conjectures: {num_formalized}\\n")
    print(
        f"Number of formalised conjectures with chi/homology blocks: {num_formalized_with_chi_homology}\\n"
    )
    print(
        f"Percentage of formalised conjectures with chi/homology blocks: {pct_formalized_with_chi_homology:.2f}%\\n"
    )


if __name__ == "__main__":
    main()
