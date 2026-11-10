"""
File for training a multi-agent deep reinforcement learning model using MADDPG on the
MathWorld environment.
"""

import os
import json
import argparse
import random
import uuid
import time
from pathlib import Path
from copy import deepcopy
import wandb

from conj_refute_env import MathWorld, manage_caches
import numpy as np
from pandas.core.computation.scope import DEFAULT_GLOBALS
import torch
from tqdm import trange, tqdm
from conj_refute_env.env.utils import (
    scrub,
    clean_extra_datafiles,
    clean_extra_leanfiles,
    LEAN_DATA_DIR,
    assert_formal_response_assets,
    get_active_experiment_settings,
    write_or_validate_run_lock,
)
from conj_refute_env.env.check_lean_theorem import PROOF_CACHE_PATH, TEMP_CACHE_DIR
from conj_refute_env.cli_utils import (
    normalize_config_id,
    normalize_model_profile,
    apply_model_profile_defaults,
    add_optional_bool_arg,
    apply_env_mode_overrides,
)

# 2)import AgileRL components
from agilerl.algorithms.maddpg import MADDPG
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv

NET_CONFIG = {
    "encoder_config": {
        "hidden_size": [32, 32],
    },
    "head_config": {"hidden_size": [32]},  # Network head hidden size
}

# ——— Pure MADDPG hyperparameters ———
# Only MEMORY_SIZE and LEARN_STEP are read; the MADDPG knobs come from the CLI
# flags below (defaults = AgileRL 2.2.8 constructor defaults, which is what ran).
INIT_HP = {
    "BATCH_SIZE": 32,
    "O_U_NOISE": False,
    "EXPL_NOISE": 0.3,
    "MEAN_NOISE": 0.0,
    # "THETA":        0.15,
    # "DT":           0.01,
    "LR_ACTOR": 1e-3,
    "LR_CRITIC": 1e-2,
    "GAMMA": 0.95,
    "MEMORY_SIZE": 100_000,
    "LEARN_STEP": 5,
    "TAU": 0.01,
}

DEFAULT_GLOBALS["nan"] = np.nan

os.environ["OMP_NUM_THREADS"] = "1"  # Disable OpenMP parallelism
os.environ["MKL_NUM_THREADS"] = "1"  # Disable MKL parallelism


SRC_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = SRC_DIR / "base_run_config.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./model_checkpoints",
        help="Where to save model checkpoints",
    )
    parser.add_argument(
        "--run-id", type=str, default=None, help="Run ID for wandb and file name"
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default="base_run_config",
        help="Environment config id used under config-dir",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default=str(DEFAULT_BASE_CONFIG),
        help="Template config that is edited between runs",
    )
    parser.add_argument(
        "--model-profile",
        type=str,
        default="M0",
        help="Model profile to apply (M0, M1, M0_NOISE, M2)",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="./run_configs",
        help="Run config file to use; overrides base_config",
    )
    parser.add_argument(
        "--interim-save-dir",
        type=str,
        default="./interim_checkpoints",
        help="Where to save interim model checkpoints",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to resume from",
    )
    parser.add_argument(
        "--resume-step", type=int, default=0, help="Step number to resume from"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=6,
        help="Number of parallel vectorized environments",
    )
    parser.add_argument(
        "--max-steps", type=int, default=600, help="Total environment steps"
    )
    parser.add_argument(
        "--train-block",
        type=int,
        default=50,
        help="Environment steps per training block",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=198,
        help="Save an interim checkpoint every N steps",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default=os.environ.get("WANDB_MODE", "online"),
        choices=["online", "offline", "disabled"],
        help="Weights & Biases mode",
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Skip cleanup of manifold/lean files and proof cache merge",
    )

    add_optional_bool_arg(
        parser,
        "formal-response-enabled",
        "Enable Lean formal-response evaluation",
    )
    add_optional_bool_arg(
        parser,
        "noise-experiment-enabled",
        "Enable noise injection on formal response",
    )
    add_optional_bool_arg(
        parser,
        "strict-proof-check",
        "Use strict proof check (formal_response == 1.0)",
    )
    parser.add_argument(
        "--formal-response-policy",
        type=str,
        choices=["auto", "always_on", "always_off"],
        default=None,
        help="Policy controlling when formal response is evaluated",
    )
    parser.add_argument(
        "--formal-response-reward-mode",
        type=str,
        choices=["standard", "disabled"],
        default=None,
        help="Control whether formal response contributes to rewards",
    )
    parser.add_argument(
        "--formal-response-noise-level",
        type=float,
        default=None,
        help="Noise level used when noise experiment is enabled",
    )
    parser.add_argument(
        "--formal-response-proof-threshold",
        type=float,
        default=None,
        help="Threshold used when strict-proof-check is disabled",
    )

    rl = parser.add_argument_group("MADDPG (defaults = AgileRL 2.2.8 MADDPG defaults)")
    rl.add_argument("--lr-actor", type=float, default=1e-3)
    rl.add_argument("--lr-critic", type=float, default=1e-2)
    rl.add_argument("--gamma", type=float, default=0.95)
    rl.add_argument("--tau", type=float, default=0.01)
    rl.add_argument("--batch-size", type=int, default=64)
    rl.add_argument("--ou-noise", action=argparse.BooleanOptionalAction, default=True)
    rl.add_argument("--expl-noise", type=float, default=0.1)
    rl.add_argument("--mean-noise", type=float, default=0.0)
    rl.add_argument("--theta", type=float, default=0.15)
    rl.add_argument("--dt", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--records-file",
        type=str,
        default=None,
        help="Append one JSON line per env per vec-step",
    )
    return parser.parse_args()


def maddpg_kwargs(args):
    return {
        "lr_actor": args.lr_actor,
        "lr_critic": args.lr_critic,
        "gamma": args.gamma,
        "tau": args.tau,
        "batch_size": args.batch_size,
        "O_U_noise": args.ou_noise,
        "expl_noise": args.expl_noise,
        "mean_noise": args.mean_noise,
        "theta": args.theta,
        "dt": args.dt,
    }


def resolved_maddpg_hparams(agent):
    return {
        "lr_actor": agent.lr_actor,
        "lr_critic": agent.lr_critic,
        "gamma": agent.gamma,
        "tau": agent.tau,
        "batch_size": agent.batch_size,
        "O_U_noise": agent.O_U_noise,
        "expl_noise": float(agent.expl_noise[0].flatten()[0]),
        "mean_noise": float(agent.mean_noise[0].flatten()[0]),
        "theta": agent.theta,
        "dt": agent.dt,
    }


def _jsonable(x):
    if isinstance(x, np.generic):
        return x.item()
    return x


def write_step_records(path, total_steps, infos, rewards, term, trunc, num_envs, agent_ids):
    ic = infos.get("conjecturer", {})
    keys = [
        "conjecture",
        "formal_response",
        "rew_conj_formal",
        "rew_conj_div",
        "rew_conj_uniform",
        "rew_ref_formal",
        "rew_ref_data",
        "rew_ref_spread",
        "uniform_block_val",
        "repeat_block_val",
        "arith_number",
        "env_id",
    ]
    with open(path, "a", encoding="utf-8") as f:
        for i in range(num_envs):
            rec = {"total_steps": total_steps, "env": i, "t_wall": time.time()}
            for k in keys:
                if k in ic:
                    rec[k] = _jsonable(ic[k][i])
            for a in agent_ids:
                rec[f"reward_{a}"] = _jsonable(np.asarray(rewards[a])[i])
                rec[f"term_{a}"] = bool(np.asarray(term[a])[i])
                rec[f"trunc_{a}"] = bool(np.asarray(trunc[a])[i])
            f.write(json.dumps(rec) + "\n")


def make_env(env_config, num_envs=3):
    """Create a MathWorld environment with specific parameters."""
    return AsyncPettingZooVecEnv(
        [lambda: MathWorld(env_config=env_config) for _ in range(num_envs)],
        context="spawn",  # Use spawn for better compatibility with multiprocessing
        # context="forkserver"
    )


def main():
    args = parse_args()
    start_time = time.time()
    model_profile = normalize_model_profile(args.model_profile)

    if args.num_envs < 1:
        raise ValueError("--num-envs must be >= 1")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be >= 1")
    if args.train_block < args.num_envs:
        raise ValueError(
            "--train-block must be >= --num-envs to run at least one step per block"
        )
    if args.checkpoint_interval < 1:
        raise ValueError("--checkpoint-interval must be >= 1")

    config_id = normalize_config_id(args.env_config)
    # Generate run ID if not provided
    run_id = args.run_id or str(uuid.uuid4().hex)

    print("Starting MADDPG Training")
    print(f"Run ID: {run_id}")
    print("─" * 60)

    lean_assets = assert_formal_response_assets(create_lean_dir=True)
    print(f"Lean assets at: {lean_assets['formal_response_root']}")

    os.makedirs(args.config_dir, exist_ok=True)
    lock_path = write_or_validate_run_lock(
        config_dir=args.config_dir,
        run_id=run_id,
        config_id=args.env_config,
        model_profile=model_profile,
    )
    experiment_settings = get_active_experiment_settings()
    print("Locked experiment settings:")
    print(f"   • dataset_profile: {experiment_settings['dataset_profile']}")
    print(f"   • rows_per_dataset: {experiment_settings['rows_per_dataset']}")
    print(f"   • model_profile: {model_profile}")
    print(f"   • lock file: {lock_path}")
    config_path = os.path.join(args.config_dir, f"{config_id}.json")
    if not os.path.exists(config_path):
        with open(args.base_config, "r") as f:
            cfg = json.load(f)

        run_cfg_path = os.path.join(args.config_dir, f"{config_id}.json")
        with open(run_cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
    else:
        with open(config_path, "r") as f:
            cfg = json.load(f)
    cfg = apply_model_profile_defaults(cfg, model_profile)
    cfg = apply_env_mode_overrides(cfg, args)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    def reset_seed(block_idx):
        if args.seed is None:
            return None
        return args.seed * 1_000_000 + block_idx * args.num_envs

    run_meta = {
        "run_id": run_id,
        "seed": args.seed,
        "prover_backend": os.environ.get("CONJ_REFUTE_PROVER", "leancopilot"),
        "julia_threads": os.environ.get("PYTHON_JULIACALL_THREADS"),
        "dataset_profile": experiment_settings["dataset_profile"],
        "rows_per_dataset": experiment_settings["rows_per_dataset"],
        "model_profile": model_profile,
        "memory_size": INIT_HP["MEMORY_SIZE"],
        "learning_delay": INIT_HP["LEARN_STEP"],
    }

    # 3) build a single full_config for W&B: RL + env
    full_config = {
        **{f"maddpg_{k}": v for k, v in maddpg_kwargs(args).items()},
        **run_meta,
        **NET_CONFIG,
        "num_envs": args.num_envs,
        "max_steps": args.max_steps,
        "train_block": args.train_block,
        "checkpoint_interval": args.checkpoint_interval,
        **cfg,
    }
    print("Initializing Weights & Biases...")
    wandb.require("core")
    # ----- WANDB CONFIGURATION -----
    # Initialize wandb
    wandb.init(
        project="conj_refute_maddpg",
        entity="science_ai",
        name=f"{run_id}",
        config=full_config,
        mode=args.wandb_mode,
    )
    wandb.config.update({"num_envs": full_config["num_envs"]})

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_envs = wandb.config.num_envs

    # ------ Prepare a table for per-env metrics --------
    metrics_table = wandb.Table(
        columns=[
            "step",
            "env",
            "conj",
            "arith_number",
            "formal_response",
            "rew_conj_formal",
            "rew_conj_div",
            "rew_conj_uniform",
            "rew_ref_formal",
            "rew_ref_data",
            "rew_ref_spread",
        ]
    )

    # Create the environment with specific parameters
    print("Creating environment...")
    env = make_env(cfg, num_envs)
    obs, infos = env.reset(seed=reset_seed(0))

    # Convert observations to float32
    obs = {aid: arr.astype(np.float32) for aid, arr in obs.items()}

    # ——— Observation & action spaces ———
    # one space *per agent* in the parallel env
    observation_spaces = [env.single_observation_space(a) for a in env.agents]
    action_spaces = [env.single_action_space(a) for a in env.agents]

    # ——— Agent IDs & hyperparameters ———
    agent_ids = deepcopy(env.agents)
    INIT_HP["AGENT_IDS"] = agent_ids

    # ——— Instantiate one MADDPG for all agents ———
    # note: we pass agent_ids here too, so no Evo‐HPO will run
    print("Initializing MADDPG agent...")
    agent = MADDPG(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=agent_ids,
        net_config=NET_CONFIG,
        device=device,
        vect_noise_dim=num_envs,
        **maddpg_kwargs(args),
    )
    resolved = resolved_maddpg_hparams(agent)
    print(f"MADDPG hyperparameters: {resolved}")
    wandb.config.update({f"resolved_{k}": v for k, v in resolved.items()})
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, f"{run_id}_hparams.json"), "w") as f:
        json.dump(
            {
                "maddpg": resolved,
                **run_meta,
                "net_config": NET_CONFIG,
                "num_envs": args.num_envs,
                "max_steps": args.max_steps,
                "train_block": args.train_block,
                "env_config": cfg,
            },
            f,
            indent=2,
        )

    # Resume from checkpoint if provided
    if args.resume_checkpoint:
        print(f"Loading checkpoint: {args.resume_checkpoint}")
        agent.load_checkpoint(args.resume_checkpoint)

    # ——— Replay buffer ———
    print("Setting up replay buffer...")
    memory = MultiAgentReplayBuffer(
        INIT_HP["MEMORY_SIZE"],
        field_names=["obs", "action", "reward", "next_obs", "done"],
        agent_ids=agent_ids,
        device=device,
    )

    # ——— Training parameters ———
    max_steps = args.max_steps
    learning_delay = INIT_HP["LEARN_STEP"]  # steps before learning starts
    train_block = args.train_block
    checkpoint_interval = args.checkpoint_interval

    print("Training configuration:")
    print(f"   • Max steps: {max_steps}")
    print(f"   • Learning delay: {learning_delay}")
    print(f"   • Train block size: {train_block}")
    print(f"   • Checkpoint interval: {checkpoint_interval}")
    print(f"   • Number of environments: {num_envs}")
    print(f"   • Device: {device}")
    print(f"   • W&B mode: {args.wandb_mode}")
    print("─" * 60)

    total_steps = args.resume_step if args.resume_checkpoint else 0
    episodes_completed = 0
    total_proven_theorems = 0
    last_checkpoint_time = time.time()

    # Main progress bar
    pbar = trange(
        max_steps, unit="step", desc="Training Progress", initial=total_steps
    )

    # MAIN TRAINING LOOP
    print("Starting training...")
    block_idx = 0
    while total_steps < max_steps:
        block_idx += 1
        # reset env (vectorised)
        obs, infos = env.reset(seed=reset_seed(block_idx))
        obs = {aid: o.astype(np.float32) for aid, o in obs.items()}

        scores = np.zeros((num_envs, len(agent_ids)))
        completed = []

        # Number of env steps this block will run (clip for final partial block)
        block_steps = train_block // num_envs

        for step_in_block in tqdm(
            range(block_steps), desc="Training block", leave=False, position=1
        ):
            # get actions
            cont_actions, disc_actions = agent.get_action(
                obs=obs, training=True, infos=infos
            )
            actions = disc_actions if agent.discrete_actions else cont_actions

            # ─── ← NEW: Log conjecturer actions (block selection) ───
            raw_conj = cont_actions["conjecturer"]
            if hasattr(raw_conj, "detach"):
                raw_conj = raw_conj.detach().cpu().numpy()
            else:
                raw_conj = np.asarray(raw_conj, dtype=np.float32)

            # Extract block selection parameters (first two elements)
            config_idx_raw = raw_conj[:, 0]

            wandb.log(
                {
                    "conjecturer/config_idx_raw_mean": config_idx_raw.mean(),
                    "conjecturer/config_idx_raw_min": config_idx_raw.min(),
                    "conjecturer/config_idx_raw_max": config_idx_raw.max(),
                    "conjecturer/config_idx_raw_hist": wandb.Histogram(
                        config_idx_raw.flatten()
                    ),
                },
                step=total_steps,
            )

            raw_ref = cont_actions["refuter"]
            if hasattr(raw_ref, "detach"):
                raw_ref = raw_ref.detach().cpu().numpy()
            else:
                raw_ref = np.asarray(raw_ref, dtype=np.float32)

            # Also log aggregate statistics across all blocks
            all_indx_values = raw_ref[:, 0::2]
            all_cont_bits = raw_ref[:, 1::2]

            wandb.log(
                {
                    "refuter/all_blocks/cont_bits_mean": all_cont_bits.mean(),
                    "refuter/all_blocks/cont_bits_min": all_cont_bits.min(),
                    "refuter/all_blocks/cont_bits_max": all_cont_bits.max(),
                    "refuter/all_blocks/cont_bits_hist": wandb.Histogram(
                        all_cont_bits.flatten()
                    ),
                },
                step=total_steps,
            )

            wandb.log(
                {
                    "refuter/all_blocks/indx_values_mean": all_indx_values.mean(),
                    "refuter/all_blocks/indx_values_min": all_indx_values.min(),
                    "refuter/all_blocks/indx_values_max": all_indx_values.max(),
                    "refuter/all_blocks/indx_values_hist": wandb.Histogram(
                        all_indx_values.flatten()
                    ),
                },
                step=total_steps,
            )

            # Handle nan's before they happen
            for aid in actions:
                actions[aid] = scrub(actions[aid])

            # EOF file error handling
            try:
                # step
                next_obs, rewards, term, trunc, infos = env.step(actions)
                next_obs = {
                    aid: arr.astype(np.float32) for aid, arr in next_obs.items()
                }
            except EOFError:
                print(
                    f"[Recovery] Worker died at step {total_steps}; "
                    f"saving checkpoint and rebuilding env…"
                )
                # 3.1) save your agent weights
                os.makedirs(args.interim_save_dir, exist_ok=True)
                ckpt = os.path.join(
                    args.interim_save_dir, f"{run_id}_marl_step_{total_steps}.pt"
                )
                agent.save_checkpoint(ckpt)

                # Tear down old env
                env.close(terminate=True)

                # Rebuild and reset
                env = make_env(cfg, num_envs=num_envs)
                obs, infos = env.reset()
                obs = {aid: o.astype(np.float32) for aid, o in obs.items()}

                continue

            if args.records_file:
                write_step_records(
                    args.records_file,
                    total_steps + num_envs,
                    infos,
                    rewards,
                    term,
                    trunc,
                    num_envs,
                    agent_ids,
                )

            # store in replay buffer
            memory.save_to_memory(
                obs, actions, rewards, next_obs, term, is_vectorised=True
            )

            # learning updates every LEARN_STEP
            if memory.counter > learning_delay and len(memory) >= agent.batch_size:
                batch = memory.sample(agent.batch_size)
                agent.learn(batch)

            obs = next_obs
            step_rewards = np.array(list(rewards.values())).T
            scores += step_rewards
            total_steps += num_envs
            pbar.update(num_envs)

            # Periodic checkpointing
            if total_steps % checkpoint_interval == 0:
                os.makedirs(args.interim_save_dir, exist_ok=True)
                ckt_path = os.path.join(
                    args.interim_save_dir, f"{run_id}_marl_step_{total_steps}.pt"
                )
                agent.save_checkpoint(ckt_path)

                # Update progress bar with checkpoint info
                checkpoint_time = time.time()
                time_since_last = checkpoint_time - last_checkpoint_time
                last_checkpoint_time = checkpoint_time

                pbar.set_postfix_str(f"Checkpoint saved | {time_since_last:.1f}s")

            # Log step rewards to wandb
            wandb.log(
                {
                    "step": total_steps,
                    **{
                        f"reward/env{i}/agent{j}": step_rewards[i, j]
                        for i in range(num_envs)
                        for j in range(len(agent_ids))
                    },
                },
                step=total_steps,
            )

            # detect episode ends & record
            term_arr = np.array(list(term.values())).T
            trunc_arr = np.array(list(trunc.values())).T
            for i in range(num_envs):
                if np.any(term_arr[i]) or np.any(trunc_arr[i]):
                    completed.append(scores[i].copy())
                    agent.scores.append(scores[i].copy())
                    scores[i] = 0
                    episodes_completed += 1

            # reset noise for done envs
            done_idxs = [
                i
                for i in range(num_envs)
                if np.any(term_arr[i]) or np.any(trunc_arr[i])
            ]
            agent.reset_action_noise(done_idxs)

            if total_steps >= max_steps:
                break

        if completed:
            avg_score = np.mean(np.array(completed), axis=0)
            pbar.set_postfix_str(f"Avg scores: {avg_score}")
        else:
            pbar.set_postfix_str("No episodes completed")

    # Training complete
    total_time = time.time() - start_time

    print("\n" + "─" * 60)
    print("TRAINING COMPLETE!")
    print(f"Total training time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Steps completed: {total_steps:,}")
    print(f"Episodes completed: {episodes_completed}")
    print(f"Average steps per second: {total_steps/total_time:.1f}")

    # Upload final metrics
    wandb.log({"env_metrics_table": metrics_table})
    wandb.log(
        {
            "training/total_time": total_time,
            "training/total_steps": total_steps,
            "training/episodes_completed": episodes_completed,
            "training/proven_theorems": total_proven_theorems,
            "training/steps_per_second": total_steps / total_time,
        }
    )

    # Save final model
    print("Saving final model...")
    os.makedirs(args.save_dir, exist_ok=True)
    final_model_path = os.path.join(args.save_dir, f"{run_id}_marl.pt")
    agent.save_checkpoint(final_model_path)

    print(f"Final model saved to: {final_model_path}")
    print("─" * 60)

    pbar.close()
    env.close()
    wandb.finish()

    if args.skip_cleanup:
        print("Skipping cleanup and cache merge (--skip-cleanup enabled)")
        return

    # Clean up manifold_data directory
    manifold_data_dir = os.path.join(os.path.dirname(__file__), "manifold_data")
    clean_extra_datafiles(manifold_data_dir)
    clean_extra_leanfiles(LEAN_DATA_DIR)
    manage_caches(PROOF_CACHE_PATH, TEMP_CACHE_DIR)


if __name__ == "__main__":
    main()
