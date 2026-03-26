import re
import time
import json
import os
from pathlib import Path

import leanclient as lc
from leanclient.utils import DocumentContentChange

# ─── CONFIG ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent.parent.parent.parent / "formal_response"
)
TEMPLATE_FILE = PROJECT_ROOT / "template.lean"
LEAN_SUBDIR = "lean_files"
TIMEOUT = 15.0  # seconds to wait for diagnostics
PROOF_CACHE_PATH = PROJECT_ROOT / "proof_cache_new.json"

TEMP_CACHE_DIR = PROJECT_ROOT / "temp_proof_caches"
TEMP_CACHE_DIR.mkdir(exist_ok=True)

marker_line = "--  === BEGIN CONJECTURES (edited in place) ==="
header_text = TEMPLATE_FILE.read_text()


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def _conj_path_for(env_id: str) -> Path:
    """
    Ensure the per-env conjecture file exists with exactly one header+marker,
    and return its Path.
    """
    out = PROJECT_ROOT / LEAN_SUBDIR / f"conjecture_{env_id}.lean"
    if not out.exists():
        out.write_text(header_text.rstrip() + "\n\n" + marker_line + "\n")
    return out


# ─── GLOBALS ─────────────────────────────────────────────────────────────────
_client = lc.LeanLSPClient(str(PROJECT_ROOT))
_fileclients = {}


def _make_cache_key(stmt: str) -> str:
    # Create a unique cache key based on the environment ID, timestamp, and statement.
    return re.sub(r"^theorem\s+\S+", "", stmt).strip()


def _read_cache(cache_path: Path = PROOF_CACHE_PATH) -> dict:
    try:
        return json.loads(cache_path.read_text())
    except Exception:
        return {}


def _write_cache(c: dict, cache_path: Path) -> None:
    cache_path.write_text(json.dumps(c, indent=2))


def _append_cache_entry(key: str, entry: dict, cache_path: Path) -> None:
    payload = {"key": key, **entry}
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _store_cache_entry(
    key: str, entry: dict, cache_path: Path, use_cache: bool
) -> None:
    if use_cache:
        c = _read_cache(cache_path=cache_path)
        c[key] = entry
        _write_cache(c, cache_path=cache_path)
        return

    _append_cache_entry(key=key, entry=entry, cache_path=cache_path)


def _get_proof_cache_final_path() -> Path:
    """Resolve where non-temporary proof outcomes are appended.
    """

    override = os.environ.get("CONJ_REFUTE_PROOF_CACHE_FINAL_PATH")
    path = Path(override) if override else (PROJECT_ROOT / "proof_cache_final.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def check_theorem(
    lean_stmt: str, env_id: str, run_ts: int, use_cache: bool = False
) -> int:
    key = _make_cache_key(lean_stmt)

    temp_path = TEMP_CACHE_DIR / f"{env_id}.json"
    write_cache_path = temp_path if use_cache else _get_proof_cache_final_path()

    if use_cache:
        cache = _read_cache(cache_path=PROOF_CACHE_PATH)
        if key in cache:
            result = cache[key]["result"]
            return result

        if temp_path.exists():
            try:
                temp_cache = json.loads(temp_path.read_text())
            except json.JSONDecodeError:
                temp_cache = {}
        else:
            temp_cache = {}

        if key in temp_cache:
            result = temp_cache[key]["result"]
            return result

    conj_path = _conj_path_for(env_id)
    if env_id not in _fileclients:
        fc = _client.create_file_client(file_path=f"{LEAN_SUBDIR}/{conj_path.name}")
        fc.open_file(timeout=TIMEOUT)
        _fileclients[env_id] = fc
    else:
        fc = _fileclients[env_id]

    baseline_diags = fc.get_diagnostics()

    new_blk = lean_stmt.rstrip() + "\n"
    old_lines = conj_path.read_text().splitlines(keepends=True)

    for i, ln in enumerate(old_lines):
        if ln.strip() == marker_line:
            marker_idx = i
            break
    else:
        marker_idx = len(old_lines) - 1

    conj_path.write_text("".join(old_lines[: marker_idx + 1]) + new_blk)

    start_line = marker_idx + 1
    end_line = len(old_lines) - 1
    end_char = len(old_lines[end_line])

    change = DocumentContentChange(
        text=new_blk,
        start=[start_line, 0],
        end=[end_line, end_char],
    )
    fc.update_file(changes=[change], timeout=TIMEOUT)

    diag_start, diags = time.time(), []
    while time.time() - diag_start < TIMEOUT:
        diags = fc.get_diagnostics()
        if len(diags) != len(baseline_diags):
            break
        time.sleep(0.1)

    errs = [
        d
        for d in diags
        if d.get("severity") == 1 or "type mismatch" in d.get("message", "")
    ]
    if errs:
        entry = {"statement": lean_stmt, "suggested_proof": "type error", "result": 0}
        _store_cache_entry(
            key=key, entry=entry, cache_path=write_cache_path, use_cache=use_cache
        )
        return 0

    if len(diags) == len(baseline_diags):
        full = conj_path.read_text()
        full_lines = full.splitlines(keepends=True)

        change = DocumentContentChange(
            text=full, start=[0, 0], end=[len(full_lines) - 1, len(full_lines[-1])]
        )
        fc.update_file(changes=[change], timeout=TIMEOUT)

        # wait a little more
        t1 = time.time()
        while time.time() - t1 < 20.0:
            diags = fc.get_diagnostics()
            if len(diags) != len(baseline_diags):
                break
            time.sleep(0.1)

    if len(diags) == len(baseline_diags):
        entry = {
            "statement": lean_stmt,
            "suggested_proof": "Copilot times out",
            "result": 0,
        }
        _store_cache_entry(
            key=key, entry=entry, cache_path=write_cache_path, use_cache=use_cache
        )
        return 0

    messages = [d.get("message", "") for d in diags]
    suggestion = next((m for m in messages if "Try this:" in m), None)
    if not suggestion:
        entry = {
            "statement": lean_stmt,
            "suggested_proof": "no suggestion",
            "result": 0,
        }
        _store_cache_entry(
            key=key, entry=entry, cache_path=write_cache_path, use_cache=use_cache
        )
        return 0
    if "sorry" in suggestion.lower():
        entry = {"statement": lean_stmt, "suggested_proof": "sorry", "result": 0}
        _store_cache_entry(
            key=key, entry=entry, cache_path=write_cache_path, use_cache=use_cache
        )
        return 0

    tail = suggestion.split("Try this:", 1)[1]
    tactics = "; ".join(ln.strip() for ln in tail.splitlines() if ln.strip()).lstrip(
        "; "
    )
    final = re.sub(r"\bsearch_proof\b", tactics, conj_path.read_text())
    conj_path.write_text(final)

    final_lines = final.splitlines(keepends=True)

    # Force a full-file update to keep Lean's buffer in sync
    full = conj_path.read_text()
    full_lines = full.splitlines(keepends=True)

    change2 = DocumentContentChange(
        text=full,
        start=[0, 0],
        end=[len(final_lines) - 1, len(final_lines[-1])],
    )
    fc.update_file(changes=[change2], timeout=TIMEOUT)

    all_lines = final.splitlines()

    term_goal = fc.get_term_goal(line=len(all_lines) - 1, character=len(all_lines[-1]))
    result = 1 if term_goal is None else 0

    # 8) store in cache and return
    entry = {"statement": lean_stmt, "suggested_proof": tactics, "result": result}
    _store_cache_entry(
        key=key, entry=entry, cache_path=write_cache_path, use_cache=use_cache
    )

    return result


# ─── MAIN ENTRYPOINT ────────────────────────────────────────────────────────────
def check_theorem_no_leancopilot(
    lean_stmt: str, env_id: str, run_ts: int, use_cache: bool = False
) -> int:
    key = _make_cache_key(lean_stmt)

    if use_cache:
        # 1a) check the global proof cache under lock
        cache = _read_cache(cache_path=PROOF_CACHE_PATH)
        if key in cache:
            result = cache[key]["result"]
            return result

        # Make the per-env temp cache
        temp_path = TEMP_CACHE_DIR / f"{env_id}.json"

        # 1b) if not in cache, check the temp cache
        if temp_path.exists():
            try:
                temp_cache = json.loads(temp_path.read_text())
            except json.JSONDecodeError:
                temp_cache = {}
        else:
            temp_cache = {}

        if key in temp_cache:
            return temp_cache[key]["result"]

    # 2) get (or create) our per-env file & client
    conj_path = _conj_path_for(env_id)
    if env_id not in _fileclients:
        fc = _client.create_file_client(file_path=f"{LEAN_SUBDIR}/{conj_path.name}")
        fc.open_file(timeout=TIMEOUT)
        _fileclients[env_id] = fc
    else:
        fc = _fileclients[env_id]

    baseline_diags = fc.get_diagnostics()

    new_blk = lean_stmt.rstrip() + "\n"

    old_lines = conj_path.read_text().splitlines(keepends=True)

    for i, ln in enumerate(old_lines):
        if ln.strip() == marker_line:
            marker_idx = i
            break
    else:
        marker_idx = len(old_lines) - 1

    conj_path.write_text("".join(old_lines[: marker_idx + 1]) + new_blk)

    start_line = marker_idx + 1
    end_line = len(old_lines) - 1
    end_char = len(old_lines[end_line])

    change = DocumentContentChange(
        text=new_blk,
        start=[start_line, 0],
        end=[end_line, end_char],
    )
    fc.update_file(changes=[change], timeout=TIMEOUT)

    # 4) wait for any diagnostics
    diag_start, diags = time.time(), []
    while time.time() - diag_start < TIMEOUT:
        diags = fc.get_diagnostics()
        if len(diags) != len(baseline_diags):
            break
        time.sleep(0.1)

    errs = [d for d in diags if d.get("severity") == 1]
    errs += [d for d in diags if "unsolved goals" in d.get("message", "").lower()]

    if errs:
        if True:
            result = 0
    else:
        result = 1

    if result:
        entry = {"statement": lean_stmt, "suggested_proof": "grind", "result": result}
    else:
        entry = {
            "statement": lean_stmt,
            "suggested_proof": "no proof",
            "result": result,
        }

    if use_cache:
        c = _read_cache(cache_path=temp_path)
        c[key] = entry
        _write_cache(c, cache_path=temp_path)

    return result
