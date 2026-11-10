import re
import os
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import sympy as sp


_READABLE_TOKEN = re.compile(r"\s*([wrnh]\(D[12]\)|\d+\.\d+|\d+|[()=≠∧⇒¬+\-*])")
_READABLE_SYMBOLS = {
    "w(D1)": "width_D1",
    "w(D2)": "width_D2",
    "h(D1)": "height_D1",
    "h(D2)": "height_D2",
    "r(D1)": "rank_D1",
    "r(D2)": "rank_D2",
    "n(D1)": "nullity_D1",
    "n(D2)": "nullity_D2",
}
_READABLE_LOGIC = {"=": "equals", "∧": "conjunction", "⇒": "implies"}


def is_readable_conjecture(text):
    """True for the form written by env.utils.conjecture_to_readable_latex."""
    return "equals(" not in text and re.search(r"[=≠∧⇒¬]", text) is not None


def _tokenize_readable(text):
    tokens, pos = [], 0
    text = text.rstrip()
    while pos < len(text):
        match = _READABLE_TOKEN.match(text, pos)
        if match is None:
            raise ValueError(f"Unexpected text at {pos} in readable conjecture: {text!r}")
        tokens.append(match.group(1))
        pos = match.end()
    return tokens


def readable_to_raw(text):
    """Inverse of conjecture_to_readable_latex's rendering (without undoing its simplification)."""
    tokens = _tokenize_readable(text)
    pos = 0

    def expect(token):
        nonlocal pos
        if pos >= len(tokens) or tokens[pos] != token:
            raise ValueError(f"Expected {token!r} at token {pos} in {text!r}")
        pos += 1

    def node():
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError(f"Unexpected end of readable conjecture: {text!r}")
        token = tokens[pos]
        if token == "¬":
            pos += 1
            expect("(")
            inner = node()
            expect(")")
            return f"logical_neg({inner})"
        if token == "(":
            pos += 1
            left = node()
            if pos >= len(tokens):
                raise ValueError(f"Unexpected end of readable conjecture: {text!r}")
            op = tokens[pos]
            pos += 1
            right = node()
            expect(")")
            if op in _READABLE_LOGIC:
                return f"{_READABLE_LOGIC[op]}({left}, {right})"
            if op == "≠":
                return f"logical_neg(equals({left}, {right}))"
            if op in {"+", "-", "*"}:
                return f"({left} {op} {right})"
            raise ValueError(f"Unknown operator {op!r} in {text!r}")
        pos += 1
        if token in _READABLE_SYMBOLS:
            return _READABLE_SYMBOLS[token]
        if token.isdigit():
            return f"_{token}" if token in {"0", "1", "2"} else token
        if re.fullmatch(r"\d+\.\d+", token):
            return token
        raise ValueError(f"Unexpected token {token!r} in {text!r}")

    raw = node()
    if pos != len(tokens):
        raise ValueError(f"Trailing tokens in readable conjecture: {text!r}")
    return raw


def find_matching_paren(text, start_index):
    """Find the index of the matching closing parenthesis."""
    depth = 1
    for i in range(start_index, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def extract_equals_blocks(text):
    """Recursively find all equals(arg1, arg2) blocks in a string."""
    blocks = []
    start_index = 0
    while True:
        try:
            start = text.index("equals(", start_index)
        except ValueError:
            break

        end = find_matching_paren(text, start + 7)
        if end == -1:
            start_index = start + 7
            continue

        block_str = text[start : end + 1]
        blocks.append(block_str)
        blocks.extend(extract_equals_blocks(text[start + 7 : end]))
        start_index = end + 1

    return blocks


def get_all_symbols(df, col_name):
    """Find all potential variable names in the conjecture column."""
    all_text = " ".join(df[col_name].dropna())
    keywords = {"equals", "implies", "conjunction", "logical_neg"}
    symbols = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", all_text))
    variable_names = {s for s in symbols if s not in keywords and not s.startswith("_")}
    return variable_names


def load_conjecture_dataframe(
    csv_file_path: str,
    script_dir: Optional[str] = None,
    required_columns: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, str]:
    """Resolve, load, and validate a conjecture CSV in a single helper."""
    resolved_csv_path = (
        csv_file_path
        if os.path.isabs(csv_file_path)
        else os.path.join(script_dir or os.getcwd(), csv_file_path)
    )
    if not os.path.isfile(resolved_csv_path):
        raise FileNotFoundError(f"CSV not found at '{resolved_csv_path}'")

    df = pd.read_csv(resolved_csv_path)
    if "conjecture" not in df.columns:
        raise ValueError("CSV must contain a 'conjecture' column.")

    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

    df = df.dropna(subset=["conjecture"])
    if df.empty:
        raise ValueError("CSV contains no conjectures.")

    readable = df["conjecture"].map(is_readable_conjecture)
    if readable.any():
        df = df.copy()
        df.loc[readable, "conjecture"] = df.loc[readable, "conjecture"].map(
            readable_to_raw
        )

    return df, resolved_csv_path


def build_symbols_with_width_height_fallback(
    df: pd.DataFrame,
    col_name: str = "conjecture",
    require_height_and_width2: bool = False,
):
    """Build symbols dict and normalize width_D1/height_D2 fallback."""
    all_vars = get_all_symbols(df, col_name)
    symbols_dict = {var: sp.Symbol(var) for var in all_vars}

    if require_height_and_width2:
        required = {"height_D1", "width_D2"}
        missing = required - symbols_dict.keys()
        if missing:
            raise ValueError(f"Missing required variables: {missing}")

    w1_symbol = symbols_dict.get("width_D1")
    h2_symbol = symbols_dict.get("height_D2")
    if w1_symbol is None and h2_symbol is None:
        raise ValueError("Need either 'width_D1' or 'height_D2' in the data.")
    if w1_symbol is None:
        w1_symbol = h2_symbol
        symbols_dict["width_D1"] = w1_symbol
    if h2_symbol is None:
        symbols_dict["height_D2"] = w1_symbol

    h1_symbol = symbols_dict.get("height_D1")
    w2_symbol = symbols_dict.get("width_D2")
    h2_symbol = symbols_dict.get("height_D2")

    return symbols_dict, h1_symbol, w1_symbol, w2_symbol, h2_symbol


def is_formal_response_positive(value) -> bool:
    """Return True when the value represents a positive formal response."""
    try:
        return float(value) == 1.0
    except (TypeError, ValueError):
        return False


def bootstrap_mean_ci(x, B=10000, alpha=0.05, seed=0, return_sigma=False):
    """Bootstrap mean and confidence interval with optional sigma estimate."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    n = len(x)
    if n == 0:
        if return_sigma:
            return np.nan, (np.nan, np.nan), np.nan
        return np.nan, (np.nan, np.nan)

    means = np.empty(B)
    for b in range(B):
        means[b] = rng.choice(x, size=n, replace=True).mean()

    mean_hat = x.mean()
    lo = np.quantile(means, alpha / 2)
    hi = np.quantile(means, 1 - alpha / 2)

    if not return_sigma:
        return mean_hat, (lo, hi)

    sigma_hat = np.std(means, ddof=1)
    z_sigma = mean_hat / sigma_hat if sigma_hat > 0 else np.nan
    return mean_hat, (lo, hi), z_sigma


def split_args(args_str):
    """Split arg1,arg2 while respecting nested parentheses."""
    depth = 0
    for i, char in enumerate(args_str):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            return args_str[:i].strip(), args_str[i + 1 :].strip()
    return None, None


def parse_expr(expr_str, symbols_dict):
    """Parse expression and convert constants like _1 to 1."""
    processed_str = re.sub(r"_(0|1|2)(?!\d)", lambda m: m.group(1), expr_str)
    return sp.parse_expr(processed_str, local_dict=symbols_dict)


def premise_substitutions(diff, symbols_dict, premises):
    """Apply configured premise substitutions to a symbolic difference."""
    if "connected" in premises:
        height_d1 = symbols_dict.get("height_D1")
        rank_d1 = symbols_dict.get("rank_D1")
        if height_d1 is not None and rank_d1 is not None:
            diff = diff.subs(height_d1 - rank_d1, 1)
    if "orientable" in premises:
        nullity_d2 = symbols_dict.get("nullity_D2")
        if nullity_d2 is not None:
            diff = diff.subs(nullity_d2, 1)
    return diff


def atomic_block_to_poly(block_str, symbols_dict, w1, premises=None):
    """Convert an equals(...) atomic block to a sympy difference expression."""
    args_str = block_str[7:-1]
    arg1_str, arg2_str = split_args(args_str)

    if arg1_str is None:
        return False

    try:
        expr1 = parse_expr(arg1_str, symbols_dict)
        expr2 = parse_expr(arg2_str, symbols_dict)
    except Exception:
        return False

    diff = expr1 - expr2

    height_d2 = symbols_dict.get("height_D2")
    if height_d2 is not None:
        diff = diff.subs(height_d2, w1)

    if premises is not None:
        diff = premise_substitutions(diff, symbols_dict, premises)

    return diff


def _expressions_match_up_to_constant(lhs, rhs):
    """Return True when lhs is a non-zero constant multiple of rhs."""
    if rhs is None:
        return False

    lhs_simplified = sp.simplify(lhs)
    rhs_simplified = sp.simplify(rhs)

    if lhs_simplified is None or rhs_simplified is None:
        return False
    if lhs_simplified.is_zero:
        return False

    try:
        ratio = sp.simplify(lhs_simplified / rhs_simplified)
    except (ZeroDivisionError, TypeError):
        return False

    if ratio.is_number:
        return bool(ratio)

    free_symbols = getattr(ratio, "free_symbols", set())
    if free_symbols:
        return False

    simplified_ratio = sp.simplify(ratio)
    if simplified_ratio.is_number:
        return bool(simplified_ratio)

    return simplified_ratio != 0


def check_alternating_sum(diff, h, w1, w2):
    """Check if expression matches k * (h - w1 + w2) = C."""
    poly = diff.as_poly(h, w1, w2)
    if poly is not None and poly.is_linear:
        coeff_h = poly.coeff_monomial(h)
        coeff_w1 = poly.coeff_monomial(w1)
        coeff_w2 = poly.coeff_monomial(w2)

        if coeff_h == -coeff_w1 and coeff_h == coeff_w2:
            simplified_coeff_h = sp.simplify(coeff_h)
            if not simplified_coeff_h.is_zero:
                constant_term = sp.simplify(poly.coeff_monomial(1))
                if constant_term is not None:
                    disallowed_symbols = {h, w1, w2}
                    if constant_term.free_symbols.isdisjoint(disallowed_symbols):
                        return True

    return False


def alternating_sum_with_rank_nullity(diff, symbols_dict, tier="silver", premises=None):
    """Check alternating sum match after rank-nullity substitutions."""
    rank_d2 = symbols_dict.get("rank_D2")
    nullity_d1 = symbols_dict.get("nullity_D1")
    nullity_d2 = symbols_dict.get("nullity_D2")
    rank_d1 = symbols_dict.get("rank_D1")

    width_d2 = sp.Symbol("width_D2")
    width_d1 = sp.Symbol("width_D1")

    if tier == "silver":
        modified_diff = diff
        if all(v is None for v in [rank_d2, nullity_d2, rank_d1, nullity_d1]):
            return False
        if rank_d2 is not None:
            replacement_expr = width_d2 - nullity_d2
            modified_diff = diff.subs(rank_d2, replacement_expr)
            if premises is not None:
                modified_diff = premise_substitutions(
                    modified_diff, symbols_dict, premises
                )
        elif nullity_d2 is not None and rank_d2 is None:
            replacement_expr = width_d2 - rank_d2
            modified_diff = diff.subs(nullity_d2, replacement_expr)
            if premises is not None:
                modified_diff = premise_substitutions(
                    modified_diff, symbols_dict, premises
                )

        if modified_diff != diff:
            if rank_d1 is not None:
                replacement_expr = width_d1 - nullity_d1
                modified_diff = modified_diff.subs(rank_d1, replacement_expr)
                if premises is not None:
                    modified_diff = premise_substitutions(
                        modified_diff, symbols_dict, premises
                    )
            elif nullity_d1 is not None and rank_d1 is None:
                replacement_expr = width_d1 - rank_d1
                modified_diff = modified_diff.subs(nullity_d1, replacement_expr)
                if premises is not None:
                    modified_diff = premise_substitutions(
                        modified_diff, symbols_dict, premises
                    )

    elif tier == "bronze":
        if all(v is None for v in [rank_d2, nullity_d2]) or all(
            v is None for v in [rank_d1, nullity_d1]
        ):
            return False
        if rank_d2 is not None:
            replacement_expr = width_d2 - nullity_d2
            modified_diff = diff.subs(rank_d2, replacement_expr)
            if premises is not None:
                modified_diff = premise_substitutions(
                    modified_diff, symbols_dict, premises
                )
        elif nullity_d2 is not None and rank_d2 is None:
            replacement_expr = width_d2 - rank_d2
            modified_diff = diff.subs(nullity_d2, replacement_expr)
            if premises is not None:
                modified_diff = premise_substitutions(
                    modified_diff, symbols_dict, premises
                )

        if rank_d1 is not None:
            replacement_expr = width_d1 - nullity_d1
            modified_diff = modified_diff.subs(rank_d1, replacement_expr)
            if premises is not None:
                modified_diff = premise_substitutions(
                    modified_diff, symbols_dict, premises
                )
        elif nullity_d1 is not None and rank_d1 is None:
            replacement_expr = width_d1 - rank_d1
            modified_diff = modified_diff.subs(nullity_d1, replacement_expr)
            if premises is not None:
                modified_diff = premise_substitutions(
                    modified_diff, symbols_dict, premises
                )
    else:
        return False

    h = symbols_dict["height_D1"]
    return check_alternating_sum(modified_diff, h, width_d1, width_d2)


def check_betti_numbers(diff, symbols_dict, premises=None, include_b2=False):
    """Check if diff corresponds to Betti expressions (b0/b1 and optionally b2)."""
    if diff is False or diff is None:
        return set()

    height_d1 = symbols_dict.get("height_D1")
    rank_d1 = symbols_dict.get("rank_D1")
    nullity_d1 = symbols_dict.get("nullity_D1")
    rank_d2 = symbols_dict.get("rank_D2")
    nullity_d2 = symbols_dict.get("nullity_D2")

    premises = premises or []
    skip_b0 = "connected" in premises
    skip_b2 = "orientable" in premises

    targets = {}
    if not skip_b0 and height_d1 is not None and rank_d1 is not None:
        targets["b0"] = height_d1 - rank_d1
    if nullity_d1 is not None and rank_d2 is not None:
        targets["b1"] = nullity_d1 - rank_d2
    if include_b2 and (not skip_b2) and nullity_d2 is not None:
        targets["b2"] = nullity_d2

    if not targets:
        return set()

    simplified_diff = sp.simplify(diff)
    if premises:
        simplified_diff = sp.simplify(
            premise_substitutions(simplified_diff, symbols_dict, premises)
        )
    if simplified_diff.is_zero:
        return set()

    matches = set()
    for label, target_expr in targets.items():
        if _expressions_match_up_to_constant(simplified_diff, target_expr):
            matches.add(label)
    return matches


def _betti_rank_nullity_variant(
    diff, symbols_dict, use_d1=False, use_d2=False, premises=None
):
    """Return diff after selected rank-nullity substitutions for Betti checks."""
    if diff is False or diff is None:
        return None

    expr = diff
    substitutions = []

    if use_d1:
        nullity_d1 = symbols_dict.get("nullity_D1")
        rank_d1 = symbols_dict.get("rank_D1")
        if nullity_d1 is None or rank_d1 is None:
            return None
        width_d1 = symbols_dict.get("width_D1", sp.Symbol("width_D1"))
        substitutions.append((nullity_d1, width_d1 - rank_d1))

    if use_d2:
        nullity_d2 = symbols_dict.get("nullity_D2")
        rank_d2 = symbols_dict.get("rank_D2")
        if nullity_d2 is None or rank_d2 is None:
            return None
        width_d2 = symbols_dict.get("width_D2", sp.Symbol("width_D2"))
        substitutions.append((nullity_d2, width_d2 - rank_d2))

    for old, new in substitutions:
        expr = expr.subs(old, new)

    if premises is not None:
        expr = premise_substitutions(expr, symbols_dict, premises)

    return sp.simplify(expr)


def betti_with_rank_nullity(
    diff, symbols_dict, tier="silver", premises=None, include_b2=False
):
    """Check Betti expressions after tier-based rank-nullity substitutions."""
    if diff is False or diff is None:
        return set()

    if tier == "silver":
        variants = [
            _betti_rank_nullity_variant(
                diff, symbols_dict, use_d1=True, use_d2=False, premises=premises
            ),
            _betti_rank_nullity_variant(
                diff, symbols_dict, use_d1=False, use_d2=True, premises=premises
            ),
        ]
    elif tier == "bronze":
        variants = [
            _betti_rank_nullity_variant(
                diff, symbols_dict, use_d1=True, use_d2=True, premises=premises
            )
        ]
    else:
        return set()

    matches = set()
    for variant in variants:
        if variant is None:
            continue
        matches.update(
            check_betti_numbers(
                variant,
                symbols_dict,
                premises=premises,
                include_b2=include_b2,
            )
        )
    return matches
