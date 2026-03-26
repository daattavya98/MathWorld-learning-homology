#!/usr/bin/env python3
import os
import json
import argparse
import pickle
import numpy as np
import torch
import wandb
from tqdm import tqdm
import pandas as pd

from conj_refute_env import (
    MathWorld,
    manage_caches,
    PROOF_CACHE_PATH,
    TEMP_CACHE_DIR,
    clean_extra_datafiles,
    clean_extra_leanfiles,
    LEAN_DATA_DIR,
    atomic_eval_stats,
)
from conj_refute_env.env.utils import enforce_run_lock, assert_formal_response_assets
from conj_refute_env.cli_utils import (
    normalize_config_id,
    normalize_model_profile,
    apply_model_profile_defaults,
    add_optional_bool_arg,
    apply_env_mode_overrides,
)
from agilerl.algorithms.maddpg import MADDPG

# Force CPU‐only & single‐thread BLAS
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def parse_args():
    p = argparse.ArgumentParser(
        description="Eval MathWorld; record per-step data (including LaTeX conjectures) and pickle"
    )
    p.add_argument("--model-dir", type=str, default="./model_checkpoints")
    p.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Must match the run-id used in train_maddpg.",
    )
    p.add_argument("--config-id", type=str, default="form_response_official")
    p.add_argument(
        "--model-profile",
        type=str,
        default="M0",
        help="Model profile to apply (M0, M1, M0_NOISE, M2)",
    )
    p.add_argument("--config-dir", type=str, default="./run_configs")
    p.add_argument("--episodes", type=int, default=15)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--output-file", type=str, default="step_data.pkl")
    p.add_argument(
        "--wandb-mode",
        type=str,
        default=os.environ.get("WANDB_MODE", "online"),
        choices=["online", "offline", "disabled"],
        help="Weights & Biases mode",
    )
    p.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Skip cleanup and cache merge for smoke/debug runs",
    )

    add_optional_bool_arg(
        p,
        "formal-response-enabled",
        "Enable Lean formal-response evaluation",
    )
    add_optional_bool_arg(
        p,
        "noise-experiment-enabled",
        "Enable noise injection on formal response",
    )
    add_optional_bool_arg(
        p,
        "strict-proof-check",
        "Use strict proof check (formal_response == 1.0)",
    )
    p.add_argument(
        "--formal-response-policy",
        type=str,
        choices=["auto", "always_on", "always_off"],
        default=None,
        help="Policy controlling when formal response is evaluated",
    )
    p.add_argument(
        "--formal-response-reward-mode",
        type=str,
        choices=["standard", "disabled"],
        default=None,
        help="Control whether formal response contributes to rewards",
    )
    p.add_argument(
        "--formal-response-noise-level",
        type=float,
        default=None,
        help="Noise level used when noise experiment is enabled",
    )
    p.add_argument(
        "--formal-response-proof-threshold",
        type=float,
        default=None,
        help="Threshold used when strict-proof-check is disabled",
    )
    return p.parse_args()


def main():
    args = parse_args()
    model_profile = normalize_model_profile(args.model_profile)
    device = torch.device("cpu")

    lean_assets = assert_formal_response_assets(create_lean_dir=True)
    print(f"🧠 Lean assets ready at: {lean_assets['formal_response_root']}")

    lock = enforce_run_lock(
        config_dir=args.config_dir,
        run_id=args.run_id,
        model_profile=model_profile,
    )
    print("🔒 Using locked experiment settings:")
    print(f"   • dataset_profile: {lock['dataset_profile']}")
    print(f"   • rows_per_dataset: {lock['rows_per_dataset']}")
    print(f"   • model_profile: {lock['model_profile']}")

    wandb.require("core")
    # ----- WANDB CONFIGURATION -----
    # Initialize wandb
    wandb.init(
        project="conj_refute_maddpg", name=f"{args.run_id}_eval", mode=args.wandb_mode
    )

    # Load trained MADDPG
    ckpt = os.path.join(args.model_dir, f"{args.run_id}_marl.pt")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    agent = MADDPG.load(ckpt, device)

    # Load the correct config
    cfg_path = os.path.join(
        args.config_dir, f"{normalize_config_id(args.config_id)}.json"
    )
    with open(cfg_path, "r") as f:
        env_cfg = json.load(f)
    env_cfg = apply_model_profile_defaults(env_cfg, model_profile)
    env_cfg = apply_env_mode_overrides(env_cfg, args)

    all_records = []

    # Outer progress bar for episodes
    for ep in tqdm(range(args.episodes), desc="Episodes"):
        env = MathWorld(env_config=env_cfg)
        state, infos = env.reset()
        cum = {"conjecturer": 0.0, "refuter": 0.0}

        # Inner progress bar for steps in this episode
        for step in tqdm(range(args.max_steps), desc=f" Ep {ep} steps ", leave=False):
            # 1) Actions from agent
            cont_a, disc_a = agent.get_action(state, training=False, infos=infos)
            raw = disc_a if agent.discrete_actions else cont_a

            # 2) To numpy / int
            actions = {}
            for aid, a in raw.items():
                arr = a.detach().cpu().numpy() if hasattr(a, "detach") else np.array(a)
                arr = np.squeeze(arr)
                actions[aid] = int(arr) if agent.discrete_actions else arr

            # 3) Pad conjecturer
            if "conjecturer" in actions:
                D = env.action_space("conjecturer").shape[0]
                act = np.asarray(actions["conjecturer"], dtype=np.float64)
                if act.shape[0] == D - 1:
                    actions["conjecturer"] = np.concatenate([act, [0.0]])
                elif act.shape[0] != D:
                    raise RuntimeError(
                        f"Conjecturer action length {act.shape[0]} != {D}"
                    )

            # 4) Reshape refuter actions
            flat_ref = np.asarray(actions["refuter"], dtype=np.float64)
            # The refuter action space is now (no_of_blocks * 40 * 2) dimensional
            # Each block has 40 (idx, weight) pairs, so reshape to (no_of_blocks, 40, 2)
            no_of_blocks = env.no_of_blocks
            ref_actions = flat_ref.reshape((no_of_blocks, 40, 2))

            # 5) Step env
            next_state, rewards, term, trunc, infos = env.step(actions)

            refuter_union_vector = infos["refuter"]["refuter_datapoint_vector"]

            # 6) Cumulate rewards
            cum["conjecturer"] += rewards["conjecturer"]
            cum["refuter"] += rewards["refuter"]

            # 7) Get & convert conjecture
            info_c = infos["conjecturer"]
            info_r = infos["refuter"]

            wandb.log(
                {
                    "episode": ep,
                    "step": step,
                    "cum_reward_conj": cum["conjecturer"],
                    "cum_reward_ref": cum["refuter"],
                    "rew_conj_formal": info_c["rew_conj_formal"],
                    "rew_conj_div": info_c["rew_conj_div"],
                    "rew_ref_formal": info_r["rew_ref_formal"],
                    "rew_ref_data": info_r["rew_ref_data"],
                    "rew_ref_spread": info_r["rew_ref_spread"],
                    "formal_response": info_c["formal_response"],
                    "terminated_by_proof": info_c["formal_response"] == 1,
                    "truncated": any(trunc.values()),
                }
            )

            record = {
                "episode": ep,
                "step": step,
                "cum_reward_conj": cum["conjecturer"],
                "cum_reward_ref": cum["refuter"],
                # from infos
                "raw_conjecture": info_c["conjecture"],
                "latex_conjecture": info_c["conjecture"],
                "rew_conj_formal": info_c["rew_conj_formal"],
                "rew_conj_uniform": info_c["rew_conj_uniform"],
                "rew_conj_div": info_c["rew_conj_div"],
                "rew_ref_formal": info_r["rew_ref_formal"],
                "rew_ref_data": info_r["rew_ref_data"],
                "rew_ref_spread": info_r["rew_ref_spread"],
                "formal_response": info_c["formal_response"],
                "uniform_block_val": info_c["uniform_block_val"],
                "repeat_block_val": info_c["repeat_block_val"],
                # other state/action info
                "ref_actions": ref_actions.copy(),
                "conjecturer_state": next_state["conjecturer"].copy(),
                "refuter_vector": refuter_union_vector.copy(),
                # termination flags
                "terminated_by_proof": info_c["formal_response"] == 1,
                "truncated": any(trunc.values()),
            }
            all_records.append(record)

            state = next_state
            if any(term.values()) or any(trunc.values()):
                break

        env.close()

    # Save to pickle
    with open(args.output_file, "wb") as f:
        pickle.dump(all_records, f)

    print(f"\nSaved {len(all_records)} records to {args.output_file}")

    if args.skip_cleanup:
        print("🧪 Skipping cleanup and cache merge (--skip-cleanup enabled)")
        return

    manage_caches(PROOF_CACHE_PATH, TEMP_CACHE_DIR)

    manifold_data_dir = os.path.join(os.path.dirname(__file__), "manifold_data")
    atomic_stats_dir = os.path.join(os.path.dirname(__file__), "atomic_eval_stats")
    atomic_stats = atomic_eval_stats(manifold_data_dir)
    # Calculate the average of all values in atomic_stats
    all_values = []
    for stats in atomic_stats.values():
        all_values.extend(
            [
                v
                for v in stats.values()
                if isinstance(v, (int, float)) and not pd.isna(v)
            ]
        )
    avg_atomic = np.mean(all_values) if all_values else float("nan")
    atomic_stats["average_all_values"] = avg_atomic
    # Save atomic_stats dictionary to a JSON file
    atomic_stats_path = os.path.join(
        atomic_stats_dir, f"eval_{args.run_id}_atomic_stats.json"
    )
    with open(atomic_stats_path, "w") as f:
        json.dump(atomic_stats, f, indent=2)
    print(f"✅ Atomic eval stats saved to: {atomic_stats_path}")

    # Clean up manifold data directory
    clean_extra_datafiles(manifold_data_dir)

    # Clean up lean file directory
    clean_extra_leanfiles(LEAN_DATA_DIR)


if __name__ == "__main__":
    main()
    # manifold_data_dir = os.path.join(os.path.dirname(__file__), "manifold_data")
    # clean_extra_datafiles(manifold_data_dir)
    # clean_extra_leanfiles(LEAN_DATA_DIR)
