from gymnasium.spaces import Box
from gymnasium.utils import seeding
import numpy as np
import pandas as pd
import os
from pettingzoo import ParallelEnv
from copy import copy
from pathlib import Path
from .utils import (
    load_datafile,
    fix_conj_str,
    get_active_dataset_profile,
    get_rows_for_dataset,
    get_profile_dataset_filenames,
)
from .conjecture_to_lean_statement import translate_fol_to_lean
from .ga_formula import (
    atomic_from_atomic_ga_params,
    scaffolded_conj_from_ga_params,
    evaluate_conjecture_on_dataset,
)
from .check_lean_theorem import check_theorem
from typing import List, Tuple, Dict
import uuid
from scipy.special import comb


proj_root = Path(__file__).resolve().parent.parent


def _get_profile_or_raise() -> str:
    profile = get_active_dataset_profile()
    if profile not in {"D0", "D1", "D2", "D3"}:
        raise ValueError(
            "Active dataset profile must be set to one of D0, D1, D2, D3 before running the experiment."
        )
    return profile


def _sample_rows(filename: str) -> np.ndarray:
    data = load_datafile(project_root=proj_root, filename=filename)
    required_rows = get_rows_for_dataset(filename)
    if data.shape[0] < required_rows:
        raise ValueError(
            f"Dataset '{filename}' has {data.shape[0]} rows, but {required_rows} random datapoints are required."
        )
    idx = np.random.choice(data.shape[0], size=required_rows, replace=False)
    return data[idx]


def _build_experiment_feature_matrix() -> np.ndarray:
    profile = _get_profile_or_raise()
    datasets = get_profile_dataset_filenames(profile)
    pieces = [_sample_rows(filename) for filename in datasets]
    return np.concatenate(pieces, axis=0)


_initial_feature_matrix = _build_experiment_feature_matrix()
datapoint_length = _initial_feature_matrix.shape[0]


def get_config_index_range(num_blocks: int, total_blocks: int) -> Tuple[int, int]:
    """
    Get the start and end indices for configurations with a specific number of blocks.

    Parameters
    ----------
    num_blocks : int
        Number of blocks to select (1 to total_blocks)
    total_blocks : int
        Total number of available blocks

    Returns
    -------
    tuple
        (start_idx, end_idx) where start_idx is inclusive and end_idx is exclusive
    """
    if num_blocks < 1 or num_blocks > total_blocks:
        raise ValueError(f"num_blocks must be between 1 and {total_blocks}")

    start_idx = 0
    for k in range(1, num_blocks):
        start_idx += int(comb(total_blocks, k, exact=True))

    end_idx = start_idx + int(comb(total_blocks, num_blocks, exact=True))
    return start_idx, end_idx


def config_index_to_blocks(config_idx: int, total_blocks: int) -> List[int]:
    """
    Convert a configuration index to the actual block indices in lexicographic order.

    Parameters
    ----------
    config_idx : int
        The configuration index
    total_blocks : int
        Total number of available blocks

    Returns
    -------
    list
        List of block indices (0-indexed)
    """
    max_config_idx = (2**total_blocks) - 2
    if config_idx < 0 or config_idx > max_config_idx:
        raise ValueError(
            f"config_idx {config_idx} is out of range. Valid range is [0, {max_config_idx}] for {total_blocks} blocks"
        )

    # Find which num_blocks this config_idx corresponds to
    cumulative = 0
    for num_blocks in range(1, total_blocks + 1):
        next_cumulative = cumulative + int(comb(total_blocks, num_blocks, exact=True))
        if config_idx < next_cumulative:
            relative_idx = config_idx - cumulative
            from itertools import combinations

            combinations_list = list(combinations(range(total_blocks), num_blocks))
            return list(combinations_list[relative_idx])
        cumulative = next_cumulative

    raise ValueError(
        f"config_idx {config_idx} is out of range for {total_blocks} blocks"
    )


def blocks_to_config_index(block_indices: List[int], total_blocks: int) -> int:
    """
    Convert block indices to configuration index.

    Parameters
    ----------
    block_indices : list
        List of block indices (0-indexed)
    total_blocks : int
        Total number of available blocks

    Returns
    -------
    int
        The configuration index
    """
    num_blocks = len(block_indices)
    if num_blocks == 0:
        raise ValueError("block_indices cannot be empty")

    # Get the starting index for this num_blocks
    start_idx, _ = get_config_index_range(num_blocks, total_blocks)

    # Generate all combinations and find the position
    from itertools import combinations

    combinations_list = list(combinations(range(total_blocks), num_blocks))
    block_tuple = tuple(sorted(block_indices))

    try:
        relative_idx = combinations_list.index(block_tuple)
        return start_idx + relative_idx
    except ValueError:
        raise ValueError(f"Invalid block indices: {block_indices}")


def compute_uniformity_and_repeats(
    self,
    original_block_names: None | Dict[str, str],
    blk_counts: None | Dict[str, int],
    repeat_penalty: float,
) -> Tuple[float, float, Dict[str, float]]:

    blk_evals = {}
    blk_names = {}
    repeat_counts = {}
    data_used = float(len(self.all_datapoint_indices))
    uniform_block_val = 0.0
    repeat_block_val = 0.0
    if blk_counts is None or blk_counts == {}:
        print(
            f"No blocks selected, returning {self.no_of_blocks} for uniform and repeat values."
        )
        return self.no_of_blocks, 0.0
    else:
        full_dataset_scalar = np.ones(int(data_used), dtype=np.float64)
        for blk_key, count in blk_counts.items():
            atomic_conj = blk_key
            blk_evals[blk_key] = (
                np.sum(
                    evaluate_conjecture_on_dataset(
                        conjecture=atomic_conj, data_vector_to_use=full_dataset_scalar
                    )
                )
                / data_used
            )
        for key, val in blk_evals.items():
            if val == 0.0 or val == 1.0:
                uniform_block_val += 1.0
        # Find repeats for each block
        for key, val in blk_counts.items():
            if val == 1:
                repeat_counts[key] = 0
            elif val > 1:
                repeat_counts[key] = val - 1

        # Repeat penalty: For each block if number of repeats is n, the penalty is x^(-r^0 -r^1..-r^n)
        for r in repeat_counts.values():
            if r > 0:
                for i in range(0, r):
                    repeat_block_val += repeat_penalty ** (i + 1)

    for key, val in original_block_names.items():
        if val in blk_evals:
            blk_names[key] = blk_evals[val]

    return uniform_block_val, repeat_block_val, blk_evals, blk_names


def check_overlap_between_blocks(
    blk_counts: None | Dict[str, int],
    ref_data_vector: np.ndarray,
) -> bool:
    if not blk_counts:
        return False

    ref_union_non_zero_scalar = np.where(ref_data_vector != 0.0, 1.0, 0.0)
    block_truths: Dict[str, np.ndarray] = {}
    for blk_key in blk_counts.keys():
        eval_result = evaluate_conjecture_on_dataset(
            conjecture=blk_key, data_vector_to_use=ref_union_non_zero_scalar
        )
        block_truths[blk_key] = eval_result != 0.0

    block_keys = list(block_truths.keys())
    for i in range(len(block_keys) - 1):
        lhs = block_truths[block_keys[i]]
        for j in range(i + 1, len(block_keys)):
            rhs = block_truths[block_keys[j]]
            if np.any(np.logical_and(lhs, rhs)):
                return True
    return False


def add_noise(formal_response: float, noise_level: float = 0.1) -> float:
    """
    Add Gaussian noise to the formal response for simulating uncertainty.

    Parameters
    ----------
    formal_response : float
        The original formal response (0 or 1).
    noise_level : float
        The standard deviation of the Gaussian noise to be added.

    Returns
    -------
    float
        The noisy formal response, clipped to [0, 1].
    """
    if formal_response not in [0, 1]:
        raise ValueError("formal_response must be either 0 or 1.")
    if formal_response == 1:
        noisy_response = formal_response - np.random.normal(0.0, noise_level)
    else:
        noisy_response = formal_response + np.random.normal(0.0, noise_level)
    noisy_response = np.clip(noisy_response, 0.0, 1.0)
    return noisy_response


def compute_reward_terms(
    conj: None | str,
    len_conj: None | int,
    blk_counts: None | Dict[str, int],
    run_id,
    env_id: str,
    timestep: int = 0,
    formal_response_enabled: bool = True,
) -> Tuple[str, List[int], Dict[str, int]]:
    """
    This function computes the reward terms for the conjecturer and refuter. These are the formal response returned by lean based on the conjecturer's parameters
    and the refuter's datapoint vector and the length of the outputted conjecture. We use a modified (FOL) symbolic regressor to produce a conjecture based on the
    genetic algorithm parameters. The data dependence of the fitness function for the GA depends on the refuter's datapoint vector. The produced conjecture is then
    passed into a translation function that produces a translated lean theorem. The theorem is then passed into the lean server through leanclient. Lean Copilot
    attempts to produce a proof for the generated theorem. If the proof is successful, the formal response is 1, otherwise it is 0.

    Parameters
    ----------
    ga_params : dict
        The changeable parameters for the genetic algorithm as decided by the conjecturer.
    ref_data_vector : np.ndarray
        The indices of the datapoints as chosen by the refuter.

    Returns
    -------
    bool
        The computed formal response. 1 if Lean is successful, otherwise 0.
    """
    if not conj or len_conj == 0:
        print(f"[timestep {timestep}] GA produced no equations – skipping reward.")
        return 0

    if formal_response_enabled:
        conj = fix_conj_str(conj)
        # # Translate the conjecture into a lean theorem
        try:
            lean_statement = translate_fol_to_lean(
                fol_str=conj,
                theorem_name=f"conjecture_run_{run_id}_timestep_{timestep}",
            )
        except Exception as e:
            print(f"[timestep {timestep}] Problematic conjecture: {conj}, error:{e}")
            return 0

        # # Send the translated theorem to Lean and get the formal response
        try:
            formal_response = check_theorem(
                lean_stmt=lean_statement, env_id=env_id, run_ts=timestep
            )
        except Exception as e:
            print("Lean proof unsuccessful: ", str(e))
            return 0
    else:
        return 0

    return formal_response


class MathWorld(ParallelEnv):
    metadata = {
        "name": "MathWorld",
        "render_modes": ["vectors"],
    }

    def __init__(self, render_mode="vectors", *, env_config: Dict = None):
        """
        Initialize the MathWorld environment with provided arguments.
        Args:
        - timestamp
        - possible_agent_names
        - all_datapoint_indices
        - ga_params
        - refuter_datapoint_vectors
        - formal_response
        - action_spaces
        - observation_spaces
        - run_id
        - reward_HPs
        - current_conjecture
        - no_of_blocks
        """
        super().__init__()
        # Initialize an env-id
        self.env_id = uuid.uuid4().hex

        cfg = env_config or {}
        self.no_of_blocks = cfg.get(
            "no_of_blocks", 10
        )
        self.block_configs = {}
        self.scaffolding_config = cfg.get("scaffolding", None)
        for i in range(self.no_of_blocks):
            block_key = f"blk_{i+1}"
            self.block_configs[block_key] = cfg.get(block_key, None)

        # Runtime-configurable formal response behavior.
        self.formal_response_default_enabled = cfg.get("formal_response_enabled", True)
        self.formal_response_enabled = self.formal_response_default_enabled
        self.formal_response_policy = cfg.get("formal_response_policy", "auto")
        self.formal_response_reward_mode = cfg.get(
            "formal_response_reward_mode", "standard"
        )
        self.noise_experiment_enabled = cfg.get("noise_experiment_enabled", False)
        self.identity_refuter_enabled = cfg.get("identity_refuter_enabled", False)
        self.formal_response_noise_level = float(
            cfg.get("formal_response_noise_level", 0.12)
        )
        self.strict_proof_check = cfg.get("strict_proof_check", True)
        self.formal_response_proof_threshold = float(
            cfg.get("formal_response_proof_threshold", 0.5)
        )

        if self.formal_response_policy not in {"auto", "always_on", "always_off"}:
            raise ValueError(
                "formal_response_policy must be one of: auto, always_on, always_off"
            )
        if self.formal_response_reward_mode not in {"standard", "disabled"}:
            raise ValueError(
                "formal_response_reward_mode must be one of: standard, disabled"
            )
        if not 0.0 <= self.formal_response_proof_threshold <= 1.0:
            raise ValueError("formal_response_proof_threshold must be in [0.0, 1.0]")
        if self.formal_response_noise_level < 0.0:
            raise ValueError("formal_response_noise_level must be >= 0.0")

        # Pull in standard hyperparameters
        self.γ = cfg.get("gamma", 0.99)  # discount factor
        self.α_conj = cfg.get("alpha_conj", 10.0)  # terminal bonus for conjecturer
        self.β_conj = cfg.get("beta_conj", 0.1)  # length penalty for conjecturer
        self.uniform_conj = cfg.get(
            "uniform_conj", 0.5
        )  # uniformity bonus for conjecturer
        self.α_ref = cfg.get("alpha_ref", 5.0)  # terminal bonus for refuter
        self.β_ref = cfg.get("beta_ref", 4.5)  # datapoint boost for refuter
        self.γ_ref = cfg.get("gamma_ref", 1.0)  # spread boost for refuter
        self.repeat_penalty = cfg.get(
            "repeat_penalty", 0.5
        )
        self.max_data_weight = cfg.get(
            "max_data_weight", 5.0
        )  # maximum weight for a datapoint
        self.run_id = 0
        self.timestep = 0
        self.all_datapoint_indices = np.arange(datapoint_length)
        self.genetic_algorithm_int_parameters = {}
        self.genetic_algorithm_float_parameters = {}
        self.refuter_datapoint_vectors = None
        self.refuter_datapoint_max_vector = None
        self.formal_response = 0
        self.chosen_no_of_blocks = cfg.get(
            "chosen_no_of_blocks", self.no_of_blocks
        )
        self.possible_agents = ["conjecturer", "refuter"]
        self._int_keys = []
        self._float_keys = []
        self.last_conjecture = None
        self.atomic_conjectures = (
            []
        )
        self.selected_blocks = []
        self.previous_selected_blocks = (
            []
        )

        # Uniform block value measurements
        self.uniform_block_values = np.zeros(self.no_of_blocks, dtype=np.float32)
        self.union_uniform_block_val = 0.0
        self.prev_union_uniform_block_val = 0.0

        if render_mode != "vectors":
            raise ValueError(
                f"Unsupported render_mode {render_mode!r}, only 'vectors' is allowed"
            )
        self.render_mode = render_mode

        self._action_spaces = {agent: None for agent in self.possible_agents}
        self._observation_spaces = {agent: None for agent in self.possible_agents}

        self._build_spaces()

    def _build_spaces(self):
        n_data = len(self.all_datapoint_indices) * self.no_of_blocks

        int_keys = {}
        float_keys = {}
        for i in range(self.no_of_blocks):
            int_keys[f"blk_{i+1}"] = ["maxsize"]
            float_keys[f"blk_{i+1}"] = [
                "complexityWeight",
                "equalsUsedBoost",
                "plusUsedBoost",
                "timesUsedBoost",
                "minusUsedBoost",
            ]
        int_keys["scaffolding"] = ["maxsize"]
        float_keys["scaffolding"] = [
            "complexityWeight",
            "logicalNegUsedBoost",
            "impliesUsedBoost",
            "conjunctUsedBoost",
        ]

        n_int = len(int_keys["blk_1"]) * self.no_of_blocks + len(
            int_keys["scaffolding"]
        )
        n_float = len(float_keys["blk_1"]) * self.no_of_blocks + len(
            float_keys["scaffolding"]
        )

        # ——— build the observation‐space ———
        # flat vector: [ conj_float (n_float) | conj_int (n_int) | ref_vec (n_data) ]
        # Add the uniform block values to the observation spaces
        obs_low_conj = np.concatenate(
            [
                -np.inf * np.ones(n_float),
                -np.inf * np.ones(n_int),
                [0.0],
                np.zeros(n_data),
                np.zeros(self.no_of_blocks),
                [0.0],
            ]
        )
        obs_high_conj = np.concatenate(
            [
                +np.inf * np.ones(n_float),
                +np.inf * np.ones(n_int),
                [
                    float(2**self.no_of_blocks - 2)
                ],
                self.max_data_weight * np.ones(n_data),
                np.ones(self.no_of_blocks),
                [float(self.no_of_blocks)],
            ]
        )
        obs_space_conj = Box(
            low=obs_low_conj.astype(np.float32),
            high=obs_high_conj.astype(np.float32),
            dtype=np.float32,
        )
        self._observation_spaces["conjecturer"] = obs_space_conj

        # Refuter's observation space now sees real float weights in [0, max]
        obs_low_ref = np.concatenate(
            [
                -np.inf * np.ones(n_float),
                -np.inf * np.ones(n_int),
                [0.0],
                np.zeros(n_data),
                np.zeros(self.no_of_blocks),
                [0.0],
            ]
        )
        obs_high_ref = np.concatenate(
            [
                +np.inf * np.ones(n_float),
                +np.inf * np.ones(n_int),
                [
                    float(2**self.no_of_blocks - 2)
                ],
                self.max_data_weight * np.ones(n_data),
                np.ones(self.no_of_blocks),
                [float(self.no_of_blocks)],
            ]
        )
        self._observation_spaces["refuter"] = Box(
            low=obs_low_ref.astype(np.float32),
            high=obs_high_ref.astype(np.float32),
            dtype=np.float32,
        )

        # ——— build the action‑space for the conjecturer ———
        # flat vector: [ num_blocks (1…no_of_blocks) | config_idx (0…max_config_idx) | delta_int (n_int) ∈ [–5,5] | delta_float (n_float) ∈ [–2.5,2.5] ]
        # We encode both as continuous values, but will round/clip in step().

        # The maximum config_idx is sum_{k=1}^{no_of_blocks} C(no_of_blocks, k) - 1
        # max_config_idx = int(sum(comb(self.no_of_blocks, k, exact=True) for k in range(1, self.no_of_blocks + 1)) - 1)

        act_low = np.concatenate(
            [
                [-1.0],  # Just learn a step for the index
                -5 * np.ones(n_int),
                -2.5 * np.ones(n_float),
            ]
        )
        act_high = np.concatenate([[1.0], +5 * np.ones(n_int), +2.5 * np.ones(n_float)])
        self._action_spaces["conjecturer"] = Box(
            low=act_low.astype(np.float32),
            high=act_high.astype(np.float32),
            dtype=np.float32,
        )

        # ——— build the action‐space for the refuter ———
        # a n-dim Box encoding 40 (idx,weight) pairs:
        #   idx ∈ [0…n_data−i−1], weight ∈ [−1.0, +1.0]
        idx_lows = []
        idx_highs = []
        # For each block, create 40 (idx, weight) pairs
        for block in range(self.no_of_blocks):
            for i in range(40):
                idx_lows.append(0.0)
                idx_highs.append(
                    float(len(self.all_datapoint_indices) - i - 1)
                )  # indices still range [0 … n_data−i−1]
                # now one continuous “weight” per index, in [−1, +1]
                idx_lows.append(-1.0)
                idx_highs.append(+1.0)
        low = np.array(idx_lows, dtype=np.float32)
        high = np.array(idx_highs, dtype=np.float32)
        self._action_spaces["refuter"] = Box(low=low, high=high, dtype=np.float32)

    def get_observations(self):
        """
        Must return a dict[str, np.ndarray] or nested dict/tuple of arrays
        matching self._observation_spaces exactly.
        """

        conj_float_values = []
        for block_key in self.genetic_algorithm_float_parameters:
            for param_name in self.genetic_algorithm_float_parameters[block_key]:
                conj_float_values.append(
                    self.genetic_algorithm_float_parameters[block_key][param_name]
                )

        conj_int_values = []
        for block_key in self.genetic_algorithm_int_parameters:
            for param_name in self.genetic_algorithm_int_parameters[block_key]:
                conj_int_values.append(
                    self.genetic_algorithm_int_parameters[block_key][param_name]
                )

        conj_float = np.array(conj_float_values, dtype=np.float64)
        conj_int = np.array(conj_int_values, dtype=np.int32)
        conj_block_config = blocks_to_config_index(
            self.selected_blocks, total_blocks=self.no_of_blocks
        )
        self.chosen_no_of_blocks = len(self.selected_blocks)

        # Now ref_vec is the continuous weight vectors for each of the blocks
        ref_vec = np.concatenate(
            [
                self.refuter_datapoint_vectors[block].astype(np.float64)
                for block in range(self.no_of_blocks)
            ]
        )

        zero_float = np.zeros_like(conj_float)
        zero_int = np.zeros_like(conj_int, dtype=np.float64)
        zero_ref = np.zeros_like(ref_vec)

        flat_conj = np.concatenate(
            [
                conj_float,
                conj_int.astype(np.float64),
                [float(conj_block_config)],
                zero_ref,
                self.uniform_block_values.astype(np.float64),
                [self.union_uniform_block_val],
            ]
        )
        flat_ref = np.concatenate(
            [
                zero_float,
                zero_int,
                [0.0],
                ref_vec,
                self.uniform_block_values.astype(np.float64),
                [self.union_uniform_block_val],
            ]
        )
        return {
            "conjecturer": flat_conj,
            "refuter": flat_ref,
        }

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
        """
        # Initialize state
        if options and "run_id" in options:
            self.run_id = options["run_id"]
        else:
            self.run_id = str(uuid.uuid4().hex)
        if options and "env_id" in options:
            self.env_id = options["env_id"]
        else:
            self.env_id = str(uuid.uuid4().hex)

        self.np_random, _ = seeding.np_random(seed)

        self.agents = copy(self.possible_agents)
        self.timestep = 0
        self.formal_response = 0
        self.last_conjecture = None

        # Load in data to ensure datapoint indices are initalized correctly
        feature_matrix = _build_experiment_feature_matrix()
        datapoint_length = feature_matrix.shape[0]

        self.all_datapoint_indices = np.arange(datapoint_length)
        self.refuter_datapoint_vectors = []
        for block in range(self.no_of_blocks):
            if self.identity_refuter_enabled:
                ref_vector = np.ones(datapoint_length, dtype=np.float64)
            else:
                # Initialize refuter_datapoint_vectors with random indices
                indices = self.np_random.choice(
                    self.all_datapoint_indices,
                    size=int(0.6 * datapoint_length),
                    replace=False,
                )
                ref_vector = np.zeros(datapoint_length, dtype=np.float64)
                ref_vector[indices] = 1.0
            self.refuter_datapoint_vectors.append(ref_vector)

        # Initialize refuter_datapoint_max_vector and compute max values
        self.refuter_datapoint_max_vector = np.zeros(datapoint_length, dtype=np.float64)
        for i in self.all_datapoint_indices:
            ith_elements = [ref_vec[i] for ref_vec in self.refuter_datapoint_vectors]
            self.refuter_datapoint_max_vector[i] = np.maximum.reduce(ith_elements)

        # ----- Apply ga overrides or use fallbacks otherwise -----
        # Initialize genetic algorithm parameters with new structure
        self.genetic_algorithm_int_parameters = {}
        self.genetic_algorithm_float_parameters = {}

        # Process block configurations
        for i in range(self.no_of_blocks):
            block_key = f"blk_{i+1}"
            if (
                block_key in self.block_configs
                and self.block_configs[block_key] is not None
            ):
                block_config = self.block_configs[block_key]
                self.genetic_algorithm_int_parameters[block_key] = block_config.get(
                    "genetic_algorithm_int_parameters", {}
                )

                self.genetic_algorithm_float_parameters[block_key] = {
                    param_name: np.random.uniform(0.5 * value, 1.5 * value)
                    for param_name, value in block_config.get(
                        "genetic_algorithm_float_parameters", {}
                    ).items()
                }
            else:
                raise ValueError(
                    f"No configuration provided for {block_key} in env_config"
                )

        if self.scaffolding_config is not None:
            self.genetic_algorithm_int_parameters["scaffolding"] = (
                self.scaffolding_config.get("genetic_algorithm_int_parameters", {})
            )
            self.genetic_algorithm_float_parameters["scaffolding"] = {
                param_name: np.random.uniform(0.8 * value, 1.2 * value)
                for param_name, value in self.scaffolding_config.get(
                    "genetic_algorithm_float_parameters", {}
                ).items()
            }
        else:
            raise ValueError("No scaffolding configuration provided in env_config")

        # Build 2D key structures: [block/scaffolding, param_name]
        self._int_keys = []
        self._float_keys = []

        for block_key in self.genetic_algorithm_int_parameters:
            for param_name in self.genetic_algorithm_int_parameters[block_key]:
                self._int_keys.append([block_key, param_name])

        for block_key in self.genetic_algorithm_float_parameters:
            for param_name in self.genetic_algorithm_float_parameters[block_key]:
                self._float_keys.append([block_key, param_name])

        self.chosen_no_of_blocks = np.random.randint(1, self.no_of_blocks)
        # Randomly select blocks for the conjecturer
        chosen_idx_range = get_config_index_range(
            self.chosen_no_of_blocks, self.no_of_blocks
        )
        if chosen_idx_range[0] == chosen_idx_range[1]:
            # If the range is empty, just select the only valid index
            chosen_idx = chosen_idx_range[0]
        else:
            chosen_idx = np.random.randint(chosen_idx_range[0], chosen_idx_range[1])
        # Convert the chosen index to actual block indices
        self.selected_blocks = config_index_to_blocks(chosen_idx, self.no_of_blocks)

        # Build one shared Dict-space for both agents
        n_int = len(self._int_keys)
        n_float = len(self._float_keys)
        n_data = len(self.all_datapoint_indices) * self.no_of_blocks

        # Per block uniform value measurements
        self.uniform_block_values = np.ones(self.no_of_blocks, dtype=np.float32)
        self.union_uniform_block_val = float(np.sum(self.uniform_block_values))

        low = np.concatenate(
            [
                np.full(n_float, -np.inf, dtype=np.float64),
                np.full(n_int, -np.inf, dtype=np.float64),
                [1.0],
                np.zeros(n_data, dtype=np.float64),
                np.zeros(self.no_of_blocks, dtype=np.float32),
                [0.0],
            ]
        )
        high = np.concatenate(
            [
                np.full(n_float, +np.inf, dtype=np.float64),
                np.full(n_int, +np.inf, dtype=np.float64),
                [
                    float(2**self.no_of_blocks - 2)
                ],
                self.max_data_weight * np.ones(n_data, dtype=np.float64),
                np.ones(self.no_of_blocks, dtype=np.float32),
                [float(self.no_of_blocks)],
            ]
        )

        self._observation_spaces["conjecturer"] = Box(
            low=low, high=high, dtype=np.float32
        )
        self._observation_spaces["refuter"] = Box(low=low, high=high, dtype=np.float32)

        # ——— build the action‐space for the refuter ———
        # a n-dim Box encoding 40 (idx,weight) pairs:
        #   idx ∈ [0…n_data−i−1], weight ∈ [−1.0, +1.0]
        idx_lows = []
        idx_highs = []
        # For each block, create 40 (idx, weight) pairs
        for block in range(self.no_of_blocks):
            for i in range(40):
                idx_lows.append(0.0)
                idx_highs.append(
                    float(len(self.all_datapoint_indices) - i - 1)
                )
                idx_lows.append(-1.0)
                idx_highs.append(+1.0)
        low = np.array(idx_lows, dtype=np.float32)
        high = np.array(idx_highs, dtype=np.float32)
        self._action_spaces["refuter"] = Box(low=low, high=high, dtype=np.float32)

        # build per‑agent action space for conjecturer as ONE flat Box:
        #   [num_blocks (1…no_of_blocks), config_idx (0…max_config_idx), delta_int (n_int) ∈ [–5, +5], delta_float (n_float) ∈ [–2.5, +2.5]]
        # max_config_idx = int(sum(comb(self.no_of_blocks, k, exact=True) for k in range(1, self.no_of_blocks + 1)) - 1)
        low = np.concatenate(
            [
                [-1.0],
                np.full(n_int, -5, dtype=np.float32),
                np.full(n_float, -2.5, dtype=np.float32),
            ]
        )
        high = np.concatenate(
            [
                [1.0],
                np.full(n_int, +5, dtype=np.float32),
                np.full(n_float, +2.5, dtype=np.float32),
            ]
        )
        self._action_spaces["conjecturer"] = Box(low=low, high=high, dtype=np.float32)

        observations = self.get_observations()

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {"formal_response": self.formal_response} for a in self.agents}

        return observations, infos

    def step(self, actions):
        """
        This method accepts an action, computes the state of the environment after applying the action,
        returns a tuple of the next observation, the resulting reward, if the environment has terminated,
        if the environment has truncated, and any additional information.

        Parameters
        ----------
        actions : _type_
            The actions taken by the agents.
        """

        # 1) capture the agents _before_ any removal
        old_agents = list(self.agents)

        # Store previous selected blocks
        self.previous_selected_blocks = self.selected_blocks.copy()

        # Process conjecturer actions: [num_blocks, config_idx, delta_int..., delta_float...]
        flat = np.asarray(actions["conjecturer"], dtype=np.float64)

        # Extract block selection parameters
        config_idx_step = flat[0]

        current_index = blocks_to_config_index(self.selected_blocks, self.no_of_blocks)
        max_config_idx = (
            2**self.no_of_blocks - 2
        )
        new_index = int(
            np.clip(np.round(current_index + config_idx_step), 0, max_config_idx)
        )

        # Round and clip to valid ranges
        # Compute the number of blocks for the new_index using get_config_index_range
        num_blocks = None
        for k in range(1, self.no_of_blocks + 1):
            start_idx, end_idx = get_config_index_range(k, self.no_of_blocks)
            if start_idx <= new_index < end_idx:
                num_blocks = k
                break

        if num_blocks is None:
            print(f"Invalid config index: {new_index}")
            num_blocks = (
                self.chosen_no_of_blocks
            )

        self.chosen_no_of_blocks = num_blocks
        self.selected_blocks = config_index_to_blocks(new_index, self.no_of_blocks)

        # Extract delta updates for GA parameters
        n_int = len(self._int_keys)
        conj_int_arr = flat[
            1 : 1 + n_int
        ]  # Skip first 2 elements (num_blocks, config_idx)
        conj_float_arr = flat[1 + n_int :]  # Remaining elements are float deltas

        # 3) rebuild your key→delta maps with 2D structure
        int_updates = {}
        for i, (block_key, param_name) in enumerate(self._int_keys):
            int_updates[(block_key, param_name)] = int(conj_int_arr[i])

        float_updates = {}
        for i, (block_key, param_name) in enumerate(self._float_keys):
            float_updates[(block_key, param_name)] = float(conj_float_arr[i])

        for (block_key, param_name), d in int_updates.items():
            floor = 7 if param_name == "maxsize" else 1
            if param_name == "maxsize":
                updated_value = (
                    self.genetic_algorithm_int_parameters[block_key][param_name] + d
                )
                self.genetic_algorithm_int_parameters[block_key][param_name] = min(
                    30, max(7, updated_value)
                )
            else:
                self.genetic_algorithm_int_parameters[block_key][param_name] = max(
                    floor,
                    self.genetic_algorithm_int_parameters[block_key][param_name] + d,
                )

        # 5) apply conjecturer's float updates
        for (block_key, param_name), d in float_updates.items():
            self.genetic_algorithm_float_parameters[block_key][param_name] = max(
                0.0, self.genetic_algorithm_float_parameters[block_key][param_name] + d
            )

        # 5) unpack the flat 80-vec for the refuter
        if self.identity_refuter_enabled:
            for block in range(self.no_of_blocks):
                self.refuter_datapoint_vectors[block].fill(1.0)
        else:
            raw_ref = np.asarray(
                actions["refuter"], dtype=np.float64
            )
            # “idxs” must be rounded+clipped; “weights” can be negative or positive
            # Each block gets 80 elements: 40 (idx, weight) pairs
            for block in range(self.no_of_blocks):
                start = block * 80
                end = start + 80
                block_raw_ref = raw_ref[start:end]
                idxs = np.round(block_raw_ref[0::2]).astype(int)
                weights = block_raw_ref[1::2]
                for global_choice, delta_w in zip(idxs, weights):
                    gi = int(
                        np.clip(global_choice, 0, len(self.all_datapoint_indices) - 1)
                    )
                    new_val = self.refuter_datapoint_vectors[block][gi] + float(delta_w)
                    self.refuter_datapoint_vectors[block][gi] = min(
                        max(new_val, 0.0), self.max_data_weight
                    )

        # For reward and downstream code, you may want to aggregate all blocks' vectors
        ref_vect_to_use = [
            self.refuter_datapoint_vectors[blk_idx] for blk_idx in self.selected_blocks
        ]
        for i in self.all_datapoint_indices:
            ith_elements = [ref_vec[i] for ref_vec in ref_vect_to_use]
            self.refuter_datapoint_max_vector[i] = np.maximum.reduce(ith_elements)

        # Reset atomic_conjectures list for this step to ensure fresh conjectures
        self.atomic_conjectures = []

        for chsn_blk_idx in self.selected_blocks:
            chsn_blk_key = (
                f"blk_{chsn_blk_idx + 1}"
            )
            combined_atomic_ga_params = {}
            combined_atomic_ga_params.update(
                self.genetic_algorithm_float_parameters[chsn_blk_key]
            )
            combined_atomic_ga_params.update(
                self.genetic_algorithm_int_parameters[chsn_blk_key]
            )
            atomic_conj, len_atomic_conj = atomic_from_atomic_ga_params(
                atomic_ga_params=combined_atomic_ga_params,
                data_vector=self.refuter_datapoint_vectors[chsn_blk_idx],
            )
            if atomic_conj is not None:
                self.atomic_conjectures.append([atomic_conj, len_atomic_conj])

        total_atomic_conj_len = sum(len_conj for _, len_conj in self.atomic_conjectures)

        atomic_conjecture_results = {}
        # Create new datafile for the scaffold ga
        for idx, blk in enumerate(self.selected_blocks):
            if idx < len(self.atomic_conjectures):
                atomic_conj_to_use = (
                    self.atomic_conjectures[idx][0]
                    if self.atomic_conjectures[idx][0] is not None
                    else None
                )
                data_vector_to_use = self.refuter_datapoint_vectors[blk]
                column_data = evaluate_conjecture_on_dataset(
                    conjecture=atomic_conj_to_use, data_vector_to_use=data_vector_to_use
                )
                atomic_conjecture_results[atomic_conj_to_use] = column_data
        atomic_conj_data_file = pd.DataFrame(atomic_conjecture_results)
        output_file = os.path.join(
            proj_root,
            "manifold_data",
            f"blk_conjectures_{self.run_id}_{self.env_id}_{self.timestep}.csv",
        )
        atomic_conj_data_file.to_csv(output_file, index=False)

        scaffold_ga_params = {}
        scaffold_ga_params.update(
            self.genetic_algorithm_float_parameters["scaffolding"]
        )
        scaffold_ga_params.update(self.genetic_algorithm_int_parameters["scaffolding"])

        # Now build the scaffold conjecture from the atomic conjectures
        step_id = f"{self.run_id}_{self.env_id}_{self.timestep}"
        conj, len_conj, blk_counts, original_block_names = (
            scaffolded_conj_from_ga_params(
                scaffolding_ga_params=scaffold_ga_params,
                ref_data_vector=self.refuter_datapoint_max_vector,
                id=step_id,
                atomic_block_length=total_atomic_conj_len,
            )
        )
        self.last_conjecture = conj

        self.prev_union_uniform_block_val = self.union_uniform_block_val

        # compute uniform block val and repeat block val
        uniform_block_val, repeat_block_val, blk_evals, blk_names = (
            compute_uniformity_and_repeats(
                self,
                original_block_names=original_block_names,
                blk_counts=blk_counts,
                repeat_penalty=self.repeat_penalty,
            )
        )

        eps = 1e-9
        no_uniform_or_repeat = (
            abs(uniform_block_val) <= eps and abs(repeat_block_val) <= eps
        )
        if no_uniform_or_repeat:
            total_blocks = int(np.sum(list(blk_counts.values()))) if blk_counts else 0
            if total_blocks > 1:
                has_overlap = check_overlap_between_blocks(
                    blk_counts=blk_counts,
                    ref_data_vector=self.refuter_datapoint_max_vector,
                )
            else:
                has_overlap = True  # Only one block, so trivially has overlap
        self.formal_response_enabled = self._resolve_formal_response_enabled(
            no_uniform_or_repeat=no_uniform_or_repeat,
            has_overlap=has_overlap,
        )

        self.formal_response = compute_reward_terms(
            conj=conj,
            len_conj=len_conj,
            blk_counts=blk_counts,
            timestep=self.timestep,
            run_id=self.run_id,
            env_id=self.env_id,
            formal_response_enabled=self.formal_response_enabled,
        )

        if self.noise_experiment_enabled and self.formal_response_enabled:
            self.formal_response = add_noise(
                formal_response=self.formal_response,
                noise_level=self.formal_response_noise_level,
            )

        # Per-block uniform value measurements
        for key, val in blk_names.items():
            blk_idx = int(key.split("_")[1])
            if 0 <= blk_idx < self.no_of_blocks:
                self.uniform_block_values[blk_idx] = (
                    1.0 if val == 0.0 or val == 1.0 else 0.0
                )
            else:
                print(
                    f"Warning: Block index {blk_idx} out of range for uniform value measurement."
                )
        # For all other blocks not in blk_names, set uniform value to 0.0
        for i in range(self.no_of_blocks):
            if f"blk_{i}" not in blk_names:
                self.uniform_block_values[i] = 0.0

        self.union_uniform_block_val = float(np.sum(self.uniform_block_values))

        if blk_counts:
            c_form, c_div, c_uniform, r_form, r_data, r_spread = self.get_reward_parts(
                len_conj=total_atomic_conj_len,
                prev_uniform_block_val=self.prev_union_uniform_block_val,
                uniform_block_val=uniform_block_val,
                repeat_block_val=repeat_block_val,
                no_uniform_or_repeat=no_uniform_or_repeat,
                has_overlap=has_overlap,
            )
        else:
            c_form, c_div, c_uniform, r_form, r_data, r_spread = self.get_reward_parts(
                len_conj=total_atomic_conj_len,
                prev_uniform_block_val=self.prev_union_uniform_block_val,
                uniform_block_val=uniform_block_val,
                repeat_block_val=repeat_block_val,
                no_uniform_or_repeat=no_uniform_or_repeat,
            )

        if self.formal_response_reward_mode == "disabled":
            r_conj = c_div + c_uniform
            r_ref = r_data + r_spread - c_div
        else:
            r_conj = c_form + c_div + c_uniform
            r_ref = r_form + r_data + r_spread - c_div

        rewards = {"conjecturer": r_conj, "refuter": r_ref}

        # build terminations & truncations
        proof_done = self._proof_satisfied(self.formal_response)
        time_done = self.timestep >= 50  # max steps per episode

        if not self.formal_response_enabled:
            proof_done = False

        terminate = proof_done and uniform_block_val == 0.0 and repeat_block_val == 0.0

        terminations = {a: terminate for a in old_agents}

        # Treat timeouts as truncations
        truncations = {a: time_done for a in old_agents}

        # finally collect observations & infos
        observations = self.get_observations()

        # Filter to old_agents
        observations = {a: observations[a] for a in old_agents}
        infos = {a: {"formal_response": self.formal_response} for a in old_agents}

        # if terminal, clear out agents so parallel_api_test knows we're done
        self.timestep += 1
        if any(terminations.values()):
            self.agents = []

        for a in old_agents:
            if self.last_conjecture is not None:
                arith_num = (
                    self.last_conjecture.count("+")
                    + self.last_conjecture.count("-")
                    + self.last_conjecture.count("*")
                )
            else:
                arith_num = 0
            infos[a].update(
                {
                    "rew_conj_formal": c_form,
                    "rew_conj_div": c_div,
                    "rew_conj_uniform": c_uniform,
                    "rew_ref_formal": r_form,
                    "rew_ref_data": r_data,
                    "rew_ref_spread": r_spread,
                    "uniform_block_val": uniform_block_val,
                    "repeat_block_val": repeat_block_val,
                    "conjecture": (
                        self.last_conjecture if self.last_conjecture is not None else ""
                    ),
                    "refuter_datapoint_vector": self.refuter_datapoint_max_vector,
                    "arith_number": arith_num,
                    "env_id": self.env_id,
                    "blk_evals": blk_evals,
                }
            )

        return observations, rewards, terminations, truncations, infos

    def render(self):
        """
        Renders the environment.
        """
        pass  # No rendering needed for this environment

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def get_last_conjecture(self) -> str:
        """
        Return the last conjecture produced by the GA

        Returns
        -------
        str
            The last best conjecture produced by the GA.
        """
        return self.last_conjecture

    def get_reward_parts(
        self,
        len_conj: int = 0,
        prev_uniform_block_val: float = 0.0,
        uniform_block_val: float = 0.0,
        repeat_block_val: float = 0.0,
        no_uniform_or_repeat: bool = False,
        has_overlap: bool = True,
    ) -> Tuple[List[float], Dict[str, float]]:
        """
        Return the discounted reward components in order
        (rew_conj_formal, rew_conj_div, rew_conj_uniform,
         rew_ref_formal,  rew_ref_data,  rew_ref_spread)
        """

        active = np.nonzero(self.refuter_datapoint_max_vector > 0)[0]
        spread = active.std(ddof=0) if active.size > 1 else 0.0
        data_used = float(np.count_nonzero(self.refuter_datapoint_max_vector))

        # 3) discount
        disc = self.γ**self.timestep

        # Adding potential based uniform block value reward
        phi_new = (
            0.5
            * (
                (len(self.selected_blocks) - uniform_block_val)
                / len(self.selected_blocks)
            )
            if len(self.selected_blocks) > 0
            else 0.0
        )
        phi_old = (
            0.5
            * (
                (len(self.previous_selected_blocks) - prev_uniform_block_val)
                / len(self.previous_selected_blocks)
            )
            if len(self.previous_selected_blocks) > 0
            else 0.0
        )
        c_uniform = disc * phi_new - phi_old

        fr = 1 if self._proof_satisfied(self.formal_response) else 0
        c_form = (
            self.α_conj * self.formal_response
            if (no_uniform_or_repeat and has_overlap and fr == 1)
            else 0.0
        )
        c_div = (
            (self.β_conj * len_conj if self.last_conjecture else 0.0)
            if (no_uniform_or_repeat and has_overlap)
            else 0.0
        )

        r_form = -self.α_ref * self.formal_response if fr == 1 else 0.0
        r_data = self.β_ref * data_used
        r_sprd = self.γ_ref * spread

        return [
            c_form * disc,
            c_div * disc,
            c_uniform * disc,
            r_form * disc,
            r_data * disc,
            r_sprd * disc,
        ]

    def _resolve_formal_response_enabled(
        self, no_uniform_or_repeat: bool, has_overlap: bool
    ) -> bool:
        if self.formal_response_policy == "always_on":
            return True
        if self.formal_response_policy == "always_off":
            return False
        return bool(
            self.formal_response_default_enabled
            and no_uniform_or_repeat
            and has_overlap
        )

    def _proof_satisfied(self, formal_response_value: float) -> bool:
        if self.strict_proof_check:
            return float(formal_response_value) == 1.0
        return float(formal_response_value) >= self.formal_response_proof_threshold
