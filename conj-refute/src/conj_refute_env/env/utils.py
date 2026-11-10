"Utility functions for the ConjRefute environment."

import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import re
from typing import Dict, Set, List
from datetime import datetime, timezone

# from pysr import TensorBoardLoggerSpec

project_root = Path(__file__).resolve().parent.parent
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

# leancopilot = paper prover; grind_legacy = paper's check_theorem_no_leancopilot;
# grind = same tactic, scored only on a completed `#print axioms` witness.
PROVER_BACKENDS = {
    "leancopilot": "formal_response",
    "grind_legacy": "formal_response_grind",
    "grind": "formal_response_grind",
}


def get_prover_backend() -> str:
    backend = os.environ.get("CONJ_REFUTE_PROVER", "leancopilot").strip().lower()
    if backend not in PROVER_BACKENDS:
        raise ValueError(
            f"CONJ_REFUTE_PROVER={backend!r}; allowed: {', '.join(PROVER_BACKENDS)}"
        )
    return backend


def resolve_formal_response_root() -> Path:
    override = os.environ.get("CONJ_REFUTE_FORMAL_RESPONSE_DIR")
    if override:
        return Path(override).resolve()
    return REPO_ROOT / PROVER_BACKENDS[get_prover_backend()]


PROJECT_ROOT = resolve_formal_response_root()
PROOF_CACHE_PATH = PROJECT_ROOT / "proof_cache_new.json"
TEMP_CACHE_DIR = PROJECT_ROOT / "temp_proof_caches"
LEAN_DATA_DIR = PROJECT_ROOT / "lean_files"
LEAN_TEMPLATE_PATH = PROJECT_ROOT / "template.lean"
LEAN_TOOLCHAIN_PATH = PROJECT_ROOT / "lean-toolchain"


DATASET_FILE_ALIASES: Dict[str, str] = {
    "sphere_data": "sphere_data.csv",
    "torus_data": "torus_data.csv",
    "klein_data": "klein_data.csv",
    "sphere_plus_sphere_data": "sphere_plus_sphere_data.csv",
    "torus_plus_sphere_data": "torus_plus_sphere_data.csv",
    "torus_plus_torus_data": "torus_plus_torus_data.csv",
    "klein_plus_klein_data": "klein_plus_klein_data.csv",
    "klein_plus_torus_data": "klein_plus_torus_data.csv",
    "klein_plus_sphere_data": "klein_plus_sphere_data.csv",
    "klein_plus_torus_plus_sphere_data": "klein_plus_torus_plus_sphere_data.csv",
    "sphere+sphere": "sphere_plus_sphere_data.csv",
    "torus+torus": "torus_plus_torus_data.csv",
    "torus+sphere": "torus_plus_sphere_data.csv",
    "klein+klein": "klein_plus_klein_data.csv",
    "klein+torus": "klein_plus_torus_data.csv",
    "klein+sphere": "klein_plus_sphere_data.csv",
    "klein+torus+sphere": "klein_plus_torus_plus_sphere_data.csv",
    "sphere_data.csv": "sphere_data.csv",
    "torus_data.csv": "torus_data.csv",
    "klein_data.csv": "klein_data.csv",
    "sphere_plus_sphere_data.csv": "sphere_plus_sphere_data.csv",
    "torus_plus_sphere_data.csv": "torus_plus_sphere_data.csv",
    "torus_plus_torus_data.csv": "torus_plus_torus_data.csv",
    "klein_plus_klein_data.csv": "klein_plus_klein_data.csv",
    "klein_plus_torus_data.csv": "klein_plus_torus_data.csv",
    "klein_plus_sphere_data.csv": "klein_plus_sphere_data.csv",
    "klein_plus_torus_plus_sphere_data.csv": "klein_plus_torus_plus_sphere_data.csv",
    "sphere_dataset": "sphere_data.csv",
    "torus_dataset": "torus_data.csv",
    "klein_bottle_dataset": "klein_data.csv",
    "sphere_dataset.csv": "sphere_data.csv",
    "torus_dataset.csv": "torus_data.csv",
    "klein_bottle_dataset.csv": "klein_data.csv",
}

FULL_ROW_DATASETS: Set[str] = set()

DATASET_PROFILES: Dict[str, Set[str]] = {
    "D0": {
        "sphere_data.csv",
        "torus_data.csv",
    },
    "D1": {
        "sphere_data.csv",
        "torus_data.csv",
        "klein_data.csv",
    },
    "D2": {
        "sphere_data.csv",
        "torus_data.csv",
        "sphere_plus_sphere_data.csv",
        "torus_plus_sphere_data.csv",
        "torus_plus_torus_data.csv",
    },
    "D3": {
        "sphere_data.csv",
        "torus_data.csv",
        "klein_data.csv",
        "sphere_plus_sphere_data.csv",
        "torus_plus_torus_data.csv",
        "klein_plus_klein_data.csv",
        "torus_plus_sphere_data.csv",
        "klein_plus_torus_data.csv",
        "klein_plus_sphere_data.csv",
        "klein_plus_torus_plus_sphere_data.csv",
    },
}

_ACTIVE_DATASET_PROFILE = os.environ.get("CONJ_REFUTE_DATASET_PROFILE")
_DEFAULT_ROWS_PER_DATASET = 350
_MAX_ROWS_PER_DATASET = 500


def _parse_rows_per_dataset(value: str | int | None) -> int:
    if value is None:
        rows = _DEFAULT_ROWS_PER_DATASET
    elif isinstance(value, int):
        rows = value
    else:
        value = str(value).strip()
        if value == "":
            rows = _DEFAULT_ROWS_PER_DATASET
        else:
            try:
                rows = int(value)
            except ValueError as exc:
                raise ValueError(
                    "CONJ_REFUTE_ROWS_PER_DATASET must be an integer between 1 and 500"
                ) from exc

    if rows < 1 or rows > _MAX_ROWS_PER_DATASET:
        raise ValueError(
            f"rows_per_dataset must be between 1 and {_MAX_ROWS_PER_DATASET}; got {rows}"
        )
    return rows


_ACTIVE_ROWS_PER_DATASET = _parse_rows_per_dataset(
    os.environ.get("CONJ_REFUTE_ROWS_PER_DATASET")
)


def set_active_dataset_profile(dataset_profile: str | None) -> None:
    """
    Select a dataset profile once at experiment start.

    Allowed values: "D0", "D1", "D2", "D3", or None (no profile restriction).
    """
    global _ACTIVE_DATASET_PROFILE
    if dataset_profile is None:
        _ACTIVE_DATASET_PROFILE = None
        os.environ.pop("CONJ_REFUTE_DATASET_PROFILE", None)
        return

    if dataset_profile not in DATASET_PROFILES:
        allowed = ", ".join(sorted(DATASET_PROFILES.keys()))
        raise ValueError(
            f"Unknown dataset profile '{dataset_profile}'. Allowed: {allowed}"
        )
    _ACTIVE_DATASET_PROFILE = dataset_profile
    os.environ["CONJ_REFUTE_DATASET_PROFILE"] = dataset_profile


def get_active_dataset_profile() -> str | None:
    return _ACTIVE_DATASET_PROFILE


def get_profile_dataset_filenames(dataset_profile: str | None = None) -> List[str]:
    profile = (
        dataset_profile if dataset_profile is not None else get_active_dataset_profile()
    )
    if profile is None:
        raise ValueError(
            "Active dataset profile is not set. Set CONJ_REFUTE_DATASET_PROFILE to one of D0, D1, D2, D3."
        )
    if profile not in DATASET_PROFILES:
        allowed = ", ".join(sorted(DATASET_PROFILES.keys()))
        raise ValueError(f"Unknown dataset profile '{profile}'. Allowed: {allowed}")

    canonical_files = {
        canonical_name
        for canonical_name in (
            _resolve_dataset_filename(name) for name in DATASET_PROFILES[profile]
        )
    }
    return sorted(canonical_files)


def set_rows_per_dataset(rows_per_dataset: int | None) -> None:
    global _ACTIVE_ROWS_PER_DATASET
    _ACTIVE_ROWS_PER_DATASET = _parse_rows_per_dataset(rows_per_dataset)
    os.environ["CONJ_REFUTE_ROWS_PER_DATASET"] = str(_ACTIVE_ROWS_PER_DATASET)


def get_rows_per_dataset() -> int:
    return _ACTIVE_ROWS_PER_DATASET


def is_violated_chain_complex_dataset(filename: str) -> bool:
    canonical_filename = _resolve_dataset_filename(filename)
    return "chain_complex_violated" in canonical_filename


def get_rows_for_dataset(filename: str) -> int:
    if is_violated_chain_complex_dataset(filename):
        raise ValueError(
            "Violated chain-complex datasets are disabled in this public workflow"
        )
    return get_rows_per_dataset()


def get_active_experiment_settings() -> Dict[str, int | str]:
    profile = get_active_dataset_profile()
    if profile is None:
        raise ValueError(
            "Active dataset profile is not set. Set CONJ_REFUTE_DATASET_PROFILE to one of D0, D1, D2, D3."
        )
    if profile not in DATASET_PROFILES:
        allowed = ", ".join(sorted(DATASET_PROFILES.keys()))
        raise ValueError(
            f"Unknown active dataset profile '{profile}'. Allowed: {allowed}"
        )

    rows = get_rows_per_dataset()
    return {
        "dataset_profile": profile,
        "rows_per_dataset": rows,
    }


def assert_formal_response_assets(create_lean_dir: bool = True) -> Dict[str, str]:
    """Validate formal_response files required by Lean-dependent evaluation.

    Returns a small path summary for logging/diagnostics.
    """
    required_paths = [
        PROJECT_ROOT,
        LEAN_TEMPLATE_PATH,
        LEAN_TOOLCHAIN_PATH,
    ]
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing formal_response assets required for Lean workflow: "
            + ", ".join(missing)
        )

    if create_lean_dir:
        LEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)
        TEMP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    return {
        "formal_response_root": str(PROJECT_ROOT),
        "lean_template": str(LEAN_TEMPLATE_PATH),
        "lean_toolchain": str(LEAN_TOOLCHAIN_PATH),
        "lean_data_dir": str(LEAN_DATA_DIR),
        "temp_cache_dir": str(TEMP_CACHE_DIR),
    }


def _run_lock_path(config_dir: str, run_id: str) -> Path:
    return Path(config_dir) / f"{run_id}_dataset_lock.json"


def _normalize_run_lock(
    lock: Dict[str, int | str], active: Dict[str, int | str]
) -> tuple[Dict[str, int | str], bool]:
    """Backfill legacy lock fields and coerce numeric settings to ints."""
    normalized = dict(lock)
    mutated = False

    for key in ["rows_per_dataset"]:
        value = normalized.get(key)
        if value is None:
            normalized[key] = int(active[key])
            mutated = True
            continue
        if not isinstance(value, int):
            normalized[key] = int(value)
            if normalized[key] != value:
                mutated = True

    model_profile = normalized.get("model_profile")
    if model_profile is None:
        normalized["model_profile"] = str(active["model_profile"])
        mutated = True
    elif not isinstance(model_profile, str):
        normalized["model_profile"] = str(model_profile)
        if normalized["model_profile"] != model_profile:
            mutated = True

    return normalized, mutated


def write_or_validate_run_lock(
    config_dir: str,
    run_id: str,
    config_id: str | None = None,
    model_profile: str = "M0",
) -> Path:
    lock_path = _run_lock_path(config_dir=config_dir, run_id=run_id)
    active = get_active_experiment_settings()
    active = dict(active)
    active["model_profile"] = model_profile
    expected = {
        "run_id": run_id,
        "config_id": config_id,
        "dataset_profile": active["dataset_profile"],
        "rows_per_dataset": active["rows_per_dataset"],
        "model_profile": model_profile,
    }

    if lock_path.exists():
        with open(lock_path, "r") as f:
            existing = json.load(f)

        existing, mutated = _normalize_run_lock(existing, active)

        for key in ["dataset_profile", "rows_per_dataset", "model_profile"]:
            existing_value = existing.get(key)
            if existing_value != expected[key]:
                raise ValueError(
                    f"Run lock mismatch for run_id='{run_id}' at '{lock_path}'. "
                    f"Existing {key}={existing_value!r}, current {key}={expected[key]!r}."
                )

        if mutated:
            with open(lock_path, "w") as f:
                json.dump(existing, f, indent=2)
        return lock_path

    payload = {
        **expected,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "lock_version": 1,
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as f:
        json.dump(payload, f, indent=2)
    return lock_path


def enforce_run_lock(
    config_dir: str, run_id: str, model_profile: str = "M0"
) -> Dict[str, int | str]:
    lock_path = _run_lock_path(config_dir=config_dir, run_id=run_id)
    if not lock_path.exists():
        raise FileNotFoundError(
            f"Run lock file not found for run_id='{run_id}': {lock_path}. "
            "Train this run first (or keep its lock file) before evaluation."
        )

    with open(lock_path, "r") as f:
        lock = json.load(f)

    active = get_active_experiment_settings()
    active = dict(active)
    active["model_profile"] = model_profile
    lock, mutated = _normalize_run_lock(lock, active)
    if mutated:
        with open(lock_path, "w") as f:
            json.dump(lock, f, indent=2)

    for key in ["dataset_profile", "rows_per_dataset", "model_profile"]:
        lock_value = lock.get(key)
        if lock_value != active[key]:
            raise ValueError(
                f"Evaluation settings mismatch for run_id='{run_id}'. "
                f"Locked {key}={lock_value!r}, current {key}={active[key]!r}."
            )
    return lock


def _resolve_dataset_filename(filename: str) -> str:
    return DATASET_FILE_ALIASES.get(filename, filename)


def _validate_dataset_allowed(filename: str) -> None:
    if _ACTIVE_DATASET_PROFILE is None:
        return

    allowed_files = DATASET_PROFILES[_ACTIVE_DATASET_PROFILE]
    allowed_canonical_files = {
        _resolve_dataset_filename(name) for name in allowed_files
    }
    if filename not in allowed_canonical_files:
        allowed_sorted = ", ".join(sorted(allowed_canonical_files))
        raise ValueError(
            f"Dataset '{filename}' is not allowed for active profile '{_ACTIVE_DATASET_PROFILE}'. "
            f"Allowed datasets: {allowed_sorted}"
        )


# Create logger
# global logger_spec
# logger_spec = TensorBoardLoggerSpec(
#     log_dir=os.path.join(os.getcwd(), "ga_logs/runs"),
#     log_interval=10,
# )


def manage_caches(proof_cache_path, temp_cache_dir):
    # Load the main proof cache if it exists
    if proof_cache_path.exists():
        try:
            main_data = json.loads(proof_cache_path.read_text())
        except json.JSONDecodeError:
            main_data = {}
    else:
        main_data = {}

    # Write all per-env temp caches to the main cache
    for temp_file in temp_cache_dir.glob("*.json"):
        try:
            temp_data = json.loads(temp_file.read_text())
        except json.JSONDecodeError:
            temp_data = {}
        # Merge only new keys
        for k, v in temp_data.items():
            if k not in main_data:
                main_data[k] = v

        # Cleanup this temp file
        temp_file.unlink(missing_ok=True)

    # Write the updated main cache back to disk
    proof_cache_path.write_text(json.dumps(main_data, indent=2))


def clean_extra_datafiles(manifold_data_dir: Path):
    """
    Clean up any extra data files that are not needed.
    """
    print(f"Cleaning up manifold_data directory: {manifold_data_dir}")
    if os.path.isdir(manifold_data_dir):
        for fname in os.listdir(manifold_data_dir):
            if fname.startswith("blk_"):
                try:
                    os.remove(os.path.join(manifold_data_dir, fname))
                except Exception as e:
                    print(f"[Warning] Could not delete {fname}: {e}")


def clean_output_dir(output_dir: Path):
    """
    Clean up the output directory by removing all files.
    """
    print(f"Cleaning up output directory: {output_dir}")
    if output_dir.exists() and output_dir.is_dir():
        for item in output_dir.iterdir():
            if item.is_dir():
                try:
                    for subitem in item.iterdir():
                        if subitem.is_file():
                            subitem.unlink()
                        elif subitem.is_dir():
                            subitem.rmdir()
                    item.rmdir()
                except Exception as e:
                    print(f"[Warning] Could not delete directory {item}: {e}")


def clean_extra_leanfiles(lean_data_dir: Path):
    """
    Clean up any extra data files that are not needed.
    """
    print(f"Cleaning up lean_data directory: {lean_data_dir}")
    if os.path.isdir(lean_data_dir):
        for fname in os.listdir(lean_data_dir):
            if fname.startswith("conjecture_"):
                try:
                    os.remove(os.path.join(lean_data_dir, fname))
                except Exception as e:
                    print(f"[Warning] Could not delete {fname}: {e}")


def compute_statistics(file_path: Path, current_eval: dict):
    """
    Compute statistics for the given data file.
    """
    # Load CSV data
    df = pd.read_csv(file_path)

    # Compute average for each column
    averages = df.mean().to_dict()

    # Use the first column header as the conjecture name
    conjecture = df.columns[0]

    # Add to current_eval if not already present
    if conjecture not in current_eval:
        current_eval[conjecture] = averages

    return current_eval


def atomic_eval_stats(manifold_data_dir: Path):
    """
    Compute atomic evaluation statistics for the given manifold data directory.
    """
    print(f"Computing atomic evaluation statistics for: {manifold_data_dir}")
    current_eval = {}
    if os.path.isdir(manifold_data_dir):
        for fname in os.listdir(manifold_data_dir):
            if fname.startswith("blk_"):
                # Load the data and compute statistics
                compute_statistics(os.path.join(manifold_data_dir, fname), current_eval)

    return current_eval


def scrub(x):
    x = np.asarray(x, dtype=np.float32)
    # replace any NaNs or infs with zero
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def fix_conj_str(conjecture: str) -> str:
    """
    Remove stray constants to fix formed conjecture.

    This removes ONLY the stray trailing digits `0`, `1`, `2` that appear
    immediately after a closing parenthesis, e.g. `...)2,` or `...)0)` or `...)1` (EOF).
    It does NOT modify `_0`, `_1`, `_2`, or any variable names like `width_D1`.
    """
    if conjecture is None:
        return conjecture

    # Remove a single trailing digit in {0,1,2,3,4} that directly follows ')'
    # and is followed by ',', ')' or end-of-string.
    conjecture = re.sub(r"\)([01234])(?=[,)])", ")", conjecture)
    conjecture = re.sub(r"\)([01234])$", ")", conjecture)

    return conjecture


class _ConjNode:
    def __init__(self, func: str, args: List["_ConjNode"] | None = None):
        self.func = func
        self.args = args if args is not None else []


def _tokenize_conjecture(s: str) -> List[str]:
    tokens = []
    token = ""
    i = 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            if token:
                tokens.append(token)
                token = ""
            i += 1
            continue
        if c in "(),+-*[]":
            if token:
                tokens.append(token)
                token = ""
            tokens.append(c)
            i += 1
            continue
        token += c
        i += 1
    if token:
        tokens.append(token)
    return tokens


def _parse_conjecture_expression(s: str) -> _ConjNode:
    tokens = _tokenize_conjecture(s)
    node = _parse_sum(tokens)
    if tokens:
        raise ValueError(f"Extra tokens remaining after parsing: {tokens[:10]}")
    return node


def _parse_sum(tokens: List[str]) -> _ConjNode:
    node = _parse_term(tokens)
    while tokens and tokens[0] in {"+", "-"}:
        op = tokens.pop(0)
        right = _parse_term(tokens)
        node = _ConjNode("plus" if op == "+" else "minus", [node, right])
    return node


def _parse_term(tokens: List[str]) -> _ConjNode:
    node = _parse_factor(tokens)
    while tokens and tokens[0] == "*":
        tokens.pop(0)
        right = _parse_factor(tokens)
        node = _ConjNode("times", [node, right])
    return node


def _parse_factor(tokens: List[str]) -> _ConjNode:
    if not tokens:
        raise ValueError("Unexpected end of tokens in factor.")

    if tokens[0] == "(":
        tokens.pop(0)
        node = _parse_sum(tokens)
        if not tokens or tokens[0] != ")":
            raise ValueError("Expected ')' in factor.")
        tokens.pop(0)
        return node

    token = tokens.pop(0)
    if tokens and tokens[0] == "(":
        tokens.pop(0)
        args: List[_ConjNode] = []
        if tokens and tokens[0] != ")":
            while True:
                args.append(_parse_sum(tokens))
                if tokens and tokens[0] == ",":
                    tokens.pop(0)
                else:
                    break
        if not tokens or tokens[0] != ")":
            raise ValueError("Expected ')' after function call arguments.")
        tokens.pop(0)
        return _ConjNode(token, args)
    return _ConjNode(token)


def _is_const(node: _ConjNode, value: int) -> bool:
    if node.args:
        return False
    if value == 0:
        return node.func in {"_0", "0"}
    if value == 1:
        return node.func in {"_1", "1"}
    return node.func == str(value)


def _nodes_equal(a: _ConjNode, b: _ConjNode) -> bool:
    if a.func != b.func or len(a.args) != len(b.args):
        return False
    return all(_nodes_equal(x, y) for x, y in zip(a.args, b.args))


def _simplify_conjecture_ast(node: _ConjNode) -> _ConjNode:
    if not node.args:
        return node

    simplified_args = [_simplify_conjecture_ast(arg) for arg in node.args]
    func = node.func

    if func == "plus":
        left, right = simplified_args
        if _is_const(left, 0):
            return right
        if _is_const(right, 0):
            return left
        if _nodes_equal(left, right):
            return _ConjNode("times", [_ConjNode("2"), left])
        return _ConjNode("plus", [left, right])

    if func == "minus":
        left, right = simplified_args
        if _is_const(right, 0):
            return left
        return _ConjNode("minus", [left, right])

    if func == "times":
        left, right = simplified_args
        if _is_const(left, 0) or _is_const(right, 0):
            return _ConjNode("_0")
        if _is_const(left, 1):
            return right
        if _is_const(right, 1):
            return left
        return _ConjNode("times", [left, right])

    if func == "logical_neg" and len(simplified_args) == 1:
        inner = simplified_args[0]
        if inner.func == "equals" and len(inner.args) == 2:
            return _ConjNode("not_equals", [inner.args[0], inner.args[1]])
        return _ConjNode("logical_neg", [inner])

    return _ConjNode(func, simplified_args)


def _render_readable_math(node: _ConjNode) -> str:
    symbol_mapping = {
        "width_D1": "w(D1)",
        "width_D2": "w(D2)",
        "height_D1": "h(D1)",
        "height_D2": "h(D2)",
        "rank_D1": "r(D1)",
        "rank_D2": "r(D2)",
        "nullity_D1": "n(D1)",
        "nullity_D2": "n(D2)",
        "_0": "0",
        "_1": "1",
        "_2": "2",
    }

    if not node.args:
        return symbol_mapping.get(node.func, node.func)

    if node.func == "equals":
        return f"({_render_readable_math(node.args[0])} = {_render_readable_math(node.args[1])})"
    if node.func == "not_equals":
        return f"({_render_readable_math(node.args[0])} ≠ {_render_readable_math(node.args[1])})"
    if node.func == "logical_neg":
        return f"¬({_render_readable_math(node.args[0])})"
    if node.func == "conjunction":
        return f"({_render_readable_math(node.args[0])} ∧ {_render_readable_math(node.args[1])})"
    if node.func == "implies":
        return f"({_render_readable_math(node.args[0])} ⇒ {_render_readable_math(node.args[1])})"
    if node.func == "plus":
        return f"({_render_readable_math(node.args[0])} + {_render_readable_math(node.args[1])})"
    if node.func == "minus":
        return f"({_render_readable_math(node.args[0])} - {_render_readable_math(node.args[1])})"
    if node.func == "times":
        return f"({_render_readable_math(node.args[0])} * {_render_readable_math(node.args[1])})"

    args_str = ", ".join(_render_readable_math(arg) for arg in node.args)
    return f"{node.func}({args_str})"


def conjecture_to_readable_latex(conjecture: str) -> str:
    """
    Convert conjecture text (prefix-style function form) to readable math text.

    Rules:
    - equals -> '='
    - logical_neg(equals(a,b)) -> 'a ≠ b'
    - conjunction -> '∧'
    - implies -> '⇒'
    - basic algebraic simplification: x+0=x, x-0=x, x*0=0, x*1=x, x+x=2*x

    Notes:
    - This function intentionally raises parse errors for malformed inputs.
    - Output is unicode-readable math text (not strict LaTeX command strings).
    """
    expr = _parse_conjecture_expression(conjecture)
    simplified = _simplify_conjecture_ast(expr)
    return _render_readable_math(simplified)


def load_datafile(project_root: str, filename: str) -> np.ndarray:
    """
    Load the dataset from the specified file.

    Args:
        filepath (str): The path to the file containing the dataset.

    Returns:
        np.ndarray: The dataset.
    """
    canonical_filename = _resolve_dataset_filename(filename)
    _validate_dataset_allowed(canonical_filename)

    filepath = os.path.join(project_root, "manifold_data", canonical_filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset file '{canonical_filename}' not found at '{filepath}'. "
            f"Original requested name: '{filename}'"
        )

    df = pd.read_csv(filepath)

    feature_columns = [
        "width_δ1",
        "width_δ2",
        "height_δ1",
        "height_δ2",
        "rank_δ1",
        "rank_δ2",
        "nullity_δ1",
        "nullity_δ2",
    ]

    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset '{canonical_filename}' is missing required feature columns: {missing}"
        )

    dataset = df.loc[:, feature_columns].to_numpy()
    return dataset
