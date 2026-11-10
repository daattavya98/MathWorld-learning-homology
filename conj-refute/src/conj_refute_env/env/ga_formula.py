# atomic_formula_genetic_algorithm.py
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import pandas as pd
import re
import sympy
from pysr import PySRRegressor, jl

from .utils import (
    load_datafile,
    get_active_dataset_profile,
    get_rows_for_dataset,
    get_profile_dataset_filenames,
)

# ────────────────────────────────────────────────────────────────────────────────
# 1) Paths: where to find your Julia snippets (including signum_loss.jl)
# ────────────────────────────────────────────────────────────────────────────────
JULIA_FOLDER = "julia_snippets"
LOSS_FILE_ATOMIC = "loss_atomic_formulae.jl"
LOSS_FILE_SCAFFOLDING = "loss_scaffolding.jl"
project_root = Path(__file__).resolve().parent.parent  # Go up one more level

# Build the full path to the Julia file:
_jl_loss_path = os.path.join(project_root, JULIA_FOLDER, LOSS_FILE_ATOMIC)
if not os.path.exists(_jl_loss_path):
    raise FileNotFoundError(
        f"Cannot find {LOSS_FILE_ATOMIC} in {os.path.dirname(_jl_loss_path)}"
    )
_jl_scaffolding_path = os.path.join(project_root, JULIA_FOLDER, LOSS_FILE_SCAFFOLDING)
if not os.path.exists(_jl_scaffolding_path):
    raise FileNotFoundError(
        f"Cannot find {LOSS_FILE_SCAFFOLDING} in {os.path.dirname(_jl_scaffolding_path)}"
    )

# ────────────────────────────────────────────────────────────────────────────────
# 2) Include signum_loss.jl exactly once and initialize Julia’s `ref_weights`
# ────────────────────────────────────────────────────────────────────────────────
jl.include(_jl_loss_path)
jl.include(_jl_scaffolding_path)
# Make data_weights a mutable global in Main (no `const`), so Python can overwrite it:
jl.seval("data_weights = Float64[]")
jl.seval("ref_weights = Float64[]")

# ────────────────────────────────────────────────────────────────────────────────
# Define equals operator
jl.seval(
    """
        function equals(x::T, y::T) where T
            if x == y
                return T(1.0)
            else
                return T(0.0)
            end
        end
    """
)

# Define implication operator
jl.seval(
    """
         function implies(x::T, y::T) where T
            if (x == T(1.0) && y == T(0.0))
                return T(0.0)
            elseif (x == T(0.0) && y == T(1.0)) || (x == T(1.0) && y == T(1.0)) || (x == T(0.0) && y == T(0.0))
                return T(1.0)
            else
                return T(NaN)
            end
         end
         """
)

# Define logical negation operator
jl.seval(
    """
        function logical_neg(x::T) where T
            if x == T(0.0)
                return T(1.0)
            elseif x == T(1.0)
                return T(0.0)
            else
                return T(NaN)
            end
        end
         """
)
# Define conjunction operator
jl.seval(
    """
        function conjunction(x::T, y::T) where T
            if x == T(1.0) && y == T(1.0)
                return T(1.0)
            elseif x == T(0.0) || y == T(0.0)
                return T(0.0)
            else
                return T(NaN)
            end
        end
    """
)


class sympy_p(sympy.Function):
    pass


def _get_profile_or_raise() -> str:
    profile = get_active_dataset_profile()
    if profile not in {"D0", "D1", "D2", "D3"}:
        raise ValueError(
            "Active dataset profile must be set to one of D0, D1, D2, D3 before running the experiment."
        )
    return profile


def _sample_rows(filename: str) -> np.ndarray:
    data = load_datafile(project_root=project_root, filename=filename)
    required_rows = get_rows_for_dataset(filename)
    if data.shape[0] < required_rows:
        raise ValueError(
            f"Dataset '{filename}' has {data.shape[0]} rows, but {required_rows} random datapoints are required."
        )
    idx = np.random.choice(data.shape[0], size=required_rows, replace=False)
    return data[idx]


def _build_atomic_feature_matrix() -> np.ndarray:
    profile = _get_profile_or_raise()
    datasets = get_profile_dataset_filenames(profile)
    pieces = [_sample_rows(filename) for filename in datasets]
    return np.concatenate(pieces, axis=0)


# ────────────────────────────────────────────────────────────────────────────────
# 3) Helper: call Julia’s `make_loss_structure(...)` (JIT‐compiled once)
# ────────────────────────────────────────────────────────────────────────────────
def _atomic_loss_with_params(
    complexityWeight: float,
    equalsUsedBoost: float,
    plusUsedBoost: float,
    timesUsedBoost: float,
    minusUsedBoost: float,
    outerEqualsPenalty: float,
    equalsSemanticPenalty: float,
    dataErrorPenalty: float,
) -> Any:
    """
    Call Julia's make_loss_structure(...) with the 22 penalty parameters.
    Because Julia JIT‐compiles this function once, subsequent calls run at Julia speed.
    The global `ref_weights` (in Julia) is used as the `weights` field.
    """
    return jl.make_atomic_loss_structure(
        complexityWeight,
        equalsUsedBoost,
        plusUsedBoost,
        timesUsedBoost,
        minusUsedBoost,
        outerEqualsPenalty,
        equalsSemanticPenalty,
        dataErrorPenalty,
    )


def _make_scaffolding_loss_with_params(
    complexityWeight: float,
    logicalNegUsedBoost: float,
    impliesUsedBoost: float,
    conjunctionUsedBoost: float,
    outerNodePenalty: float,
    impliesSemanticErrorPenalty: float,
    logicalFalsePenalty: float,
    repeatPenalty: float,
    uniformPenalty: float,
    dataErrorPenalty: float,
    atomicBlockLength: float,
) -> Any:
    """
    Call Julia’s make_loss_structure(...) with the 22 penalty parameters.
    Because Julia JIT‐compiles this function once, subsequent calls run at Julia speed.
    The global `ref_weights` (in Julia) is used as the `weights` field.
    """
    return jl.make_scaffolding_loss_structure(
        complexityWeight,
        logicalNegUsedBoost,
        impliesUsedBoost,
        conjunctionUsedBoost,
        outerNodePenalty,
        impliesSemanticErrorPenalty,
        logicalFalsePenalty,
        repeatPenalty,
        uniformPenalty,
        dataErrorPenalty,
        atomicBlockLength,
    )


# ────────────────────────────────────────────────────────────────────────────────
# 4) Cache: keep the last‐used tuple of penalty‐values and its Julia loss object
# ────────────────────────────────────────────────────────────────────────────────


_last_atomic_obj: Any = None
_last_atomic_ga_values: Optional[Tuple[float, ...]] = None
_last_scaffolding_loss_obj: Any = None
_last_scaffolding_ga_values: Optional[Tuple[float, ...]] = None


def _get_loss_for_atomic_ga(atomic_ga_params: Dict[str, float]) -> Any:
    global _last_atomic_obj, _last_atomic_ga_values

    # 1) Defaults in the exact same order as make_loss_structure’s signature:
    defaults = {
        "complexityWeight": 1.5,
        "equalsUsedBoost": 1.0,
        "plusUsedBoost": 1.0,
        "timesUsedBoost": 1.0,
        "minusUsedBoost": 1.0,
        "outerEqualsPenalty": 75.0,
        "equalsSemanticPenalty": 50.0,
        "dataErrorPenalty": 125.0,
    }

    # 2) Build the tuple in field‐order:
    atomic_ga_tuple = (
        atomic_ga_params.get("complexityWeight", defaults["complexityWeight"]),
        atomic_ga_params.get("equalsUsedBoost", defaults["equalsUsedBoost"]),
        atomic_ga_params.get("plusUsedBoost", defaults["plusUsedBoost"]),
        atomic_ga_params.get("timesUsedBoost", defaults["timesUsedBoost"]),
        atomic_ga_params.get("minusUsedBoost", defaults["minusUsedBoost"]),
        atomic_ga_params.get("outerEqualsPenalty", defaults["outerEqualsPenalty"]),
        atomic_ga_params.get(
            "equalsSemanticPenalty", defaults["equalsSemanticPenalty"]
        ),
        atomic_ga_params.get("dataErrorPenalty", defaults["dataErrorPenalty"]),
    )

    # 3) If identical to last, return cached object:
    if _last_atomic_ga_values == atomic_ga_tuple and _last_atomic_obj is not None:
        return _last_atomic_obj

    # 4) Otherwise, build a new Julia loss object (fast now) and cache it:
    new_loss = _atomic_loss_with_params(*atomic_ga_tuple)
    _last_atomic_obj = new_loss
    _last_atomic_ga_values = atomic_ga_tuple
    return new_loss


def _get_loss_scaffolding_for_ga(scaffolding_ga_params: Dict[str, float]) -> Any:
    """
    Given ga_params, build or reuse a Julia SignumLossWithWeights.
    We order the 22 parameters in a fixed sequence; if that tuple matches
    the last tuple, we return the cached loss object. Otherwise, we call
    make_loss_structure(...) (fast now that it’s compiled), cache it, and return it.
    """
    global _last_scaffolding_loss_obj, _last_scaffolding_ga_values

    # 1) Defaults in the exact same order as make_loss_structure’s signature:
    defaults = {
        "complexityWeight": 1.5,
        "logicalNegUsedBoost": 0.25,
        "impliesUsedBoost": 2.0,
        "conjunctionUsedBoost": 1.0,
        "outerNodePenalty": 10.0,
        "impliesSemanticErrorPenalty": 10.0,
        "logicalFalsePenalty": 3.5,
        "dataErrorPenalty": 1.25,
        "repeatPenalty": 7.0,
        "uniformPenalty": 10.0,
        "atomicBlockLength": 0.0,
    }

    # 2) Build the tuple in field‐order:
    scaffolding_ga_tuple = (
        scaffolding_ga_params.get("complexityWeight", defaults["complexityWeight"]),
        scaffolding_ga_params.get(
            "logicalNegUsedBoost", defaults["logicalNegUsedBoost"]
        ),
        scaffolding_ga_params.get("impliesUsedBoost", defaults["impliesUsedBoost"]),
        scaffolding_ga_params.get(
            "conjunctionUsedBoost", defaults["conjunctionUsedBoost"]
        ),
        scaffolding_ga_params.get("outerNodePenalty", defaults["outerNodePenalty"]),
        scaffolding_ga_params.get(
            "impliesSemanticErrorPenalty", defaults["impliesSemanticErrorPenalty"]
        ),
        scaffolding_ga_params.get(
            "logicalFalsePenalty", defaults["logicalFalsePenalty"]
        ),
        scaffolding_ga_params.get("repeatPenalty", defaults["repeatPenalty"]),
        scaffolding_ga_params.get("uniformPenalty", defaults["uniformPenalty"]),
        scaffolding_ga_params.get("dataErrorPenalty", defaults["dataErrorPenalty"]),
        scaffolding_ga_params.get("atomicBlockLength", defaults["atomicBlockLength"]),
    )

    # 3) If identical to last, return cached object:
    if (
        _last_scaffolding_ga_values == scaffolding_ga_tuple
        and _last_scaffolding_loss_obj is not None
    ):
        return _last_scaffolding_loss_obj

    # 4) Otherwise, build a new Julia loss object (fast now) and cache it:
    new_loss = _make_scaffolding_loss_with_params(*scaffolding_ga_tuple)
    _last_scaffolding_loss_obj = new_loss
    _last_scaffolding_ga_values = scaffolding_ga_tuple
    return new_loss


# ────────────────────────────────────────────────────────────────────────────────
# 5) Use the user’s original create_regressor_model function verbatim:
# ────────────────────────────────────────────────────────────────────────────────


def create_atomic_regressor_model(
    loss_func: Any,
    atomic_ga_params: dict = None,
    maxsize: int = 15,
    warmup_maxsize_by: float = 1.5,
) -> PySRRegressor:

    default_params = {
        "maxsize": maxsize,
        "warmup_maxsize_by": warmup_maxsize_by,
        "populations": 40,
        "niterations": 50,
        "ncycles_per_iteration": 500,
        "binary_operators": ["+", "*", "-", "equals"],
        "loss_function": loss_func,
        "extra_sympy_mappings": {
            "equals": sympy_p,
        },
        "nested_constraints": {
            "+": {"equals": 0},
            "*": {"equals": 0},
            "-": {"equals": 0},
            "equals": {"equals": 0},
        },
    }

    # Override defaults with any atomic_ga_params that match:
    if atomic_ga_params is not None:
        for key, val in atomic_ga_params.items():
            if key not in default_params:
                raise KeyError(f"Unknown GA parameter {key}")
        default_params.update(
            {k: v for k, v in atomic_ga_params.items() if k in default_params}
        )

    model = PySRRegressor(
        **default_params,
        complexity_of_constants=20,
        weight_simplify=0.0,
        should_optimize_constants=False,
        weight_mutate_constant=0.0,
        optimize_probability=0.0,
        timeout_in_seconds=30.0,
        population_size=60,
    )

    return model


def create_scaffolding_regressor_model(
    loss_func: Any,
    scaffolding_ga_params: dict = None,
    maxsize: int = 15,
    warmup_maxsize_by: float = 1.5,
) -> PySRRegressor:
    default_params = {
        "maxsize": maxsize,
        "warmup_maxsize_by": warmup_maxsize_by,
        "populations": 15,
        "niterations": 50,
        "ncycles_per_iteration": 500,
        "binary_operators": ["implies", "conjunction"],
        "unary_operators": ["logical_neg"],
        "loss_function": loss_func,
        "extra_sympy_mappings": {
            "logical_neg": sympy_p,
            "implies": sympy_p,
            "conjunction": sympy_p,
        },
        "nested_constraints": {"logical_neg": {"logical_neg": 0}},
        # "deterministic": True,
        # "random_state": 42,
        # "parallelism": "serial"
    }
    # Override defaults with any scaffolding_ga_params that match:
    if scaffolding_ga_params is not None:
        for key, val in scaffolding_ga_params.items():
            if key not in default_params:
                raise KeyError(f"Unknown GA parameter {key}")
        default_params.update(
            {k: v for k, v in scaffolding_ga_params.items() if k in default_params}
        )

    scaffolding_model = PySRRegressor(
        **default_params,
        complexity_of_constants=100,
        should_optimize_constants=False,
        weight_mutate_constant=0.0,
        timeout_in_seconds=30.0,
        population_size=60,
    )

    return scaffolding_model


def parse_conjecture(conjecture: str, actual_blocks: List) -> str:
    """
    Parse the conjecture string and replace variable names with actual block values.
    """
    for i, block in enumerate(actual_blocks):
        conjecture = conjecture.replace(f"blk_{i}", str(block))
    return conjecture


# ────────────────────────────────────────────────────────────────────────────────
# 6) Main API: atomic_from_atomic_ga_params
# ────────────────────────────────────────────────────────────────────────────────
def atomic_from_atomic_ga_params(
    atomic_ga_params: Optional[Dict[str, float]] = None,
    data_vector: Optional[np.ndarray] = None,
) -> Tuple[Optional[str], int]:
    """
    Create a PySRRegressor model for atomic formulae using the given GA parameters.

    :param atomic_ga_params: Dictionary of GA parameters for atomic formulae.
    :param data_vector: Optional data vector for training.
    :return: Configured PySRRegressor model.
    """
    # -------------------------------------------------------------------------
    # A) Build (or reuse) the Julia loss object for these ga_params:
    # -------------------------------------------------------------------------
    atomic_ga_params = atomic_ga_params or {}
    atomic_loss_keys = {
        "complexityWeight",
        "equalsUsedBoost",
        "plusUsedBoost",
        "timesUsedBoost",
        "minusUsedBoost",
        "outerEqualsPenalty",
        "equalsSemanticPenalty",
        "dataErrorPenalty",
    }

    # Extract the values for Julia’s loss:
    atomic_loss_params = {
        k: v for k, v in atomic_ga_params.items() if k in atomic_loss_keys
    }
    atomic_loss_func = _get_loss_for_atomic_ga(atomic_loss_params)

    # The remaining keys go to create_regressor_model’s `atomic_ga_params` argument:
    atomic_model_params = {
        k: v for k, v in atomic_ga_params.items() if k not in atomic_loss_keys
    }
    # -------------------------------------------------------------------------
    # B) Overwrite Julia’s global data_weights in place:
    # -------------------------------------------------------------------------

    if data_vector is not None:
        # Convert to float64 array → PythonCall turns it into Julia Array{Float64}
        jl.data_weights = np.array(data_vector, dtype=np.float64).ravel()
    else:
        # Means “all weights = 1” inside Julia
        jl.data_weights = np.zeros((0,), dtype=np.float64)

    # -------------------------------------------------------------------------
    # C) Load/build the feature matrix X:
    # -------------------------------------------------------------------------
    X = _build_atomic_feature_matrix()
    # Insert the two constant columns:
    X = np.insert(X, 8, 0, axis=1)
    X = np.insert(X, 9, 1, axis=1)
    # X = np.insert(X, 10, 2, axis=1)

    # -------------------------------------------------------------------------
    # D) Build the label vector y: y[i] = 1.0 if data_vector[i] > 0, else 0.0
    # -------------------------------------------------------------------------
    if data_vector is not None:
        if data_vector.shape[0] != X.shape[0]:
            raise ValueError(
                f"data_vector length {data_vector.shape[0]} does not match number of rows in X ({X.shape[0]})"
            )
        y = (data_vector > 0).astype(float)
    else:
        y = np.ones((X.shape[0],), dtype=float)

    # -------------------------------------------------------------------------
    # E) Fit via the user’s create_regressor_model (so we don’t change that logic):
    # -------------------------------------------------------------------------
    atomic_model = create_atomic_regressor_model(
        loss_func=atomic_loss_func,
        atomic_ga_params=atomic_model_params,
    )
    variable_names = [
        "width_D1",
        "width_D2",
        "height_D1",
        "height_D2",
        "rank_D1",
        "rank_D2",
        "nullity_D1",
        "nullity_D2",
        "_0",
        "_1",
    ]
    atomic_model.fit(X, y, variable_names=variable_names)

    # -------------------------------------------------------------------------
    # F) Extract the best equation and its complexity:
    # -------------------------------------------------------------------------
    best_df = atomic_model.equations_

    if best_df.empty:
        print("No equations were found for this set of parameters.")
        return None, 0

    conj = atomic_model.get_best().equation
    len_conj = atomic_model.get_best().complexity
    return conj, len_conj


def scaffolded_conj_from_ga_params(
    scaffolding_ga_params: Optional[Dict[str, float]] = None,
    ref_data_vector: Optional[np.ndarray] = None,
    id: str = None,
    baseline_check: bool = False,
    atomic_block_length: int = 0,
) -> Tuple[Optional[str], int]:
    """
    1. Build or reuse a Julia SignumLossWithWeights for the given ga_params.
    2. Overwrite Julia’s `ref_weights` with the new ref_data_vector (numpy float64).
    3. Load or build X, compute y = (ref_data_vector > 0).
    4. Call the user’s create_regressor_model(log_spec, loss_func, ga_params, …)
       which will internally set `loss_function=loss_func` and construct PySRRegressor.
    5. Fit the model on (X, y). Extract and return (equation, complexity).
    """
    # -------------------------------------------------------------------------
    # A) Build (or reuse) the Julia loss object for these ga_params:
    # -------------------------------------------------------------------------
    scaffolding_ga_params = dict(scaffolding_ga_params or {})
    scaffolding_ga_params["atomicBlockLength"] = float(atomic_block_length)

    # Define the 22 keys that belong to Julia’s loss function:
    loss_keys = {
        "complexityWeight",
        "logicalNegUsedBoost",
        "impliesUsedBoost",
        "conjunctionUsedBoost",
        "outerNodePenalty",
        "impliesSemanticErrorPenalty",
        "logicalFalsePenalty",
        "dataErrorPenalty",
        "repeatPenalty",
        "uniformPenalty",
        "atomicBlockLength",
    }

    # Extract the values for Julia’s loss:
    scaffolding_loss_params = {
        k: v for k, v in scaffolding_ga_params.items() if k in loss_keys
    }

    loss_func = _get_loss_scaffolding_for_ga(scaffolding_loss_params)

    # The remaining keys go to create_regressor_model’s `ga_params` argument:
    scaffolding_model_params = {
        k: v for k, v in scaffolding_ga_params.items() if k not in loss_keys
    }

    # -------------------------------------------------------------------------
    # B) Overwrite Julia’s global ref_weights in place:
    # -------------------------------------------------------------------------
    if ref_data_vector is not None:
        # Convert to float64 array → PythonCall turns it into Julia Array{Float64}
        jl.ref_weights = np.array(ref_data_vector, dtype=np.float64).ravel()
    else:
        # Means “all weights = 1” inside Julia
        jl.ref_weights = np.zeros((0,), dtype=np.float64)

    # -------------------------------------------------------------------------
    # C) Load/build the feature matrix X:
    # -------------------------------------------------------------------------
    if baseline_check:
        path_to_data = os.path.join(
            project_root, "manifold_data", "baseline_ga", f"blk_conjectures_{id}.csv"
        )
    else:
        path_to_data = os.path.join(
            project_root, "manifold_data", f"blk_conjectures_{id}.csv"
        )
    # X = load_datafile(project_root=path_to_data, filename=f"blk_conjectures_{id}.csv")
    X = pd.read_csv(path_to_data, header=None).to_numpy()

    # -------------------------------------------------------------------------
    # D) Build the label vector y: y[i] = 1.0 if ref_data_vector[i] > 0, else 0.0
    # -------------------------------------------------------------------------
    if ref_data_vector is not None:
        if ref_data_vector.shape[0] != (X.shape[0] - 1):
            raise ValueError(
                f"ref_data_vector length {ref_data_vector.shape[0]} does not match number of rows in X ({X.shape[0] - 1})"
            )
        y = (ref_data_vector > 0).astype(float)
    else:
        y = np.ones((X.shape[0] - 1,), dtype=float)

    # -------------------------------------------------------------------------
    # E) Fit via the user’s create_regressor_model (so we don’t change that logic):
    # -------------------------------------------------------------------------
    # log_spec = logger_spec  # same as your original
    model = create_scaffolding_regressor_model(
        loss_func=loss_func,
        scaffolding_ga_params=scaffolding_model_params,
    )

    actual_blocks = list(X[0])
    no_of_blocks = X.shape[1]

    # Now all of the blocks need to be evaluated on the new ref_data_vector
    # The new blk_evaluations are concatenated into a numpy array as the new X
    union_evaluations = np.ndarray((X.shape[0] - 1, X.shape[1]))

    for i in range(no_of_blocks):
        blk_new_eval = evaluate_conjecture_on_dataset(
            conjecture=actual_blocks[i], data_vector_to_use=ref_data_vector
        )
        union_evaluations[:, i] = np.array(blk_new_eval)
    union_evaluations = np.array(union_evaluations)

    variable_names = [f"blk_{i}" for i in range(no_of_blocks)]
    # X = X[1:, :]  # Skip the first row which is the header
    # model.fit(X, y, variable_names=variable_names)
    # union_evaluations = union_evaluations[1:, :]  # Skip the first row which is the header

    model.fit(union_evaluations, y, variable_names=variable_names)

    # -------------------------------------------------------------------------
    # F) Extract the best equation and its complexity:
    # -------------------------------------------------------------------------
    best_df = model.equations_

    if best_df.empty:
        print("No equations were found for this set of parameters.")
        return None, 0, {}, {}

    conj = model.get_best().equation
    len_conj = model.get_best().complexity

    # Get the blocks that were actually used in the conjecture
    block_counts = {}
    original_block_names = {}
    conj_str = str(conj)

    # Use regex to find all occurrences of blk_X with word boundaries
    block_matches = re.findall(r"\bblk_(\d+)\b", conj_str)

    # Count occurrences of each block
    for block_id in block_matches:
        block_name = f"blk_{block_id}"
        # I also want to save the original block names, that is, before parsing in the next step
        original_block_name_key = block_name
        block_name = parse_conjecture(
            conjecture=block_name, actual_blocks=actual_blocks
        )
        original_block_names[original_block_name_key] = block_name
        if block_name in block_counts:
            block_counts[block_name] += 1
        else:
            block_counts[block_name] = 1

    # Parse the conjecture to replace variable names with actual block values
    conj = parse_conjecture(conjecture=conj, actual_blocks=actual_blocks)

    return conj, len_conj, block_counts, original_block_names


def evaluate_conjecture_on_dataset(
    conjecture,
    data_vector_to_use: np.ndarray,
    variable_names: List[str] = [
        "width_D1",
        "width_D2",
        "height_D1",
        "height_D2",
        "rank_D1",
        "rank_D2",
        "nullity_D1",
        "nullity_D2",
        "_0",
        "_1",
    ],
) -> np.ndarray:
    """
    Evaluate a sympy conjecture expression on the full dataset and multiply element-wise with data_vector_to_use.
    All conjectures have the form equals(left_expr, right_expr).

    :param conjecture: Sympy expression representing the conjecture
    :param data_vector_to_use: 1D array of weights to multiply with evaluation results
    :param variable_names: List of variable names corresponding to columns in dataset
    :return: Array of evaluation results multiplied element-wise with data_vector_to_use
    """
    # Load the full dataset internally
    X = _build_atomic_feature_matrix()
    # Insert the two constant columns:
    X = np.insert(X, 8, 0, axis=1)
    X = np.insert(X, 9, 1, axis=1)
    # X = np.insert(X, 10, 2, axis=1)

    # Validate data_vector_to_use matches dataset size
    if len(data_vector_to_use) != X.shape[0]:
        raise ValueError(
            f"data_vector_to_use length {len(data_vector_to_use)} does not match dataset size {X.shape[0]}"
        )

    # Create a mapping from variable names to column indices
    var_to_col = {name: i for i, name in enumerate(variable_names)}

    # Convert the sympy expression to a string for parsing
    conj_str = str(conjecture)

    # Check if this is a properly formed equals expression
    if not conj_str.startswith("equals("):
        print(f"Conjecture must start with 'equals(', got: {conj_str}")
        return [0] * X.shape[0]  # Return zero vector if not valid

    # Extract the left and right expressions from equals(left, right)
    # Find the matching parentheses and comma
    # Remove the outer equals( and the final )
    inner_expr = conj_str[7:-1]  # Remove "equals(" and ")"

    # Find the comma that separates left and right expressions
    # We need to be careful about nested parentheses
    paren_count = 0
    comma_pos = -1

    for i, char in enumerate(inner_expr):
        if char == "(":
            paren_count += 1
        elif char == ")":
            paren_count -= 1
        elif char == "," and paren_count == 0:
            comma_pos = i
            break

    if comma_pos == -1:
        raise ValueError(f"Could not find separating comma in: {conj_str}")

    left_expr = inner_expr[:comma_pos].strip()
    right_expr = inner_expr[comma_pos + 1 :].strip()

    # print(f"    Evaluating: equals({left_expr}, {right_expr})")

    # Evaluate conjecture on entire dataset to get 0/1 vector
    evaluation_results = []

    for row_idx in range(X.shape[0]):
        try:
            # Evaluate both expressions for this row
            val1 = evaluate_expression(left_expr, X[row_idx], var_to_col)
            val2 = evaluate_expression(right_expr, X[row_idx], var_to_col)

            # Check if they are equal (with small tolerance for floating point)
            result = 1 if abs(val1 - val2) < 1e-10 else 0
            evaluation_results.append(result)

        except Exception:
            # print(f"    Warning: Failed to evaluate row {row_idx}: {e}")
            evaluation_results.append(0)

    # Convert to numpy array and multiply element-wise with data_vector_to_use
    evaluation_array = np.array(evaluation_results)
    final_result = evaluation_array * data_vector_to_use

    return final_result


def evaluate_expression(expr_str: str, row_data: np.ndarray, var_to_col: dict) -> float:
    """
    Evaluate a mathematical expression string using the row data.

    :param expr_str: String expression to evaluate (arithmetic with variables)
    :param row_data: Single row of data
    :param var_to_col: Mapping from variable names to column indices
    :return: Evaluated result as float
    """
    # Handle basic cases first
    expr_str = expr_str.strip()

    # Debug: Check the structure of row_data
    if not isinstance(row_data, np.ndarray):
        raise ValueError(
            f"row_data must be numpy array, got {type(row_data)}: {row_data}"
        )
    if row_data.ndim == 0:
        raise ValueError(f"row_data is scalar: {row_data}, expected 1D array")
    if row_data.ndim != 1:
        raise ValueError(f"row_data must be 1D array, got shape {row_data.shape}")

    # If it's just a variable name, return its value
    if expr_str in var_to_col:
        col_idx = var_to_col[expr_str]
        if col_idx >= len(row_data):
            raise ValueError(
                f"Column index {col_idx} for variable '{expr_str}' is out of bounds for row_data of length {len(row_data)}"
            )
        return float(row_data[col_idx])

    # Try to parse as a number
    try:
        return float(expr_str)
    except ValueError:
        pass

    # For complex expressions, substitute variables with their values
    eval_expr = expr_str

    # Sort variable names by length (longest first) to avoid partial replacements
    # e.g., replace "width_D1" before "width_D" to avoid conflicts
    sorted_vars = sorted(var_to_col.keys(), key=len, reverse=True)

    for var_name in sorted_vars:
        if var_name in eval_expr:
            # Use word boundaries to ensure we don't replace parts of other variables
            import re

            pattern = r"\b" + re.escape(var_name) + r"\b"
            col_idx = var_to_col[var_name]
            if col_idx >= len(row_data):
                raise ValueError(
                    f"Column index {col_idx} for variable '{var_name}' is out of bounds for row_data of length {len(row_data)}"
                )
            eval_expr = re.sub(pattern, str(float(row_data[col_idx])), eval_expr)

    # Evaluate the mathematical expression
    try:
        # Use eval with a restricted namespace for safety
        allowed_names = {
            "__builtins__": {},
            "abs": abs,
            "min": min,
            "max": max,
        }
        result = eval(eval_expr, allowed_names)
        return float(result)
    except Exception as e:
        raise ValueError(f"Could not evaluate '{expr_str}' -> '{eval_expr}': {e}")
