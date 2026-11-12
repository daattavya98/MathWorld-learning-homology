"""Grind-prover calibration.

Ground truth for the CHECKER is a batch compile (`lake env lean`) of the identical file: no LSP, no
silence timeout. Exit 0 only if the gated backend never scores 1 where batch Lean says 0, never scores 0
where batch Lean says 1 (except the deliberate slow probes), the cold open scores a truth 1, and the
core identities are proved. Mathematical truth vs batch Lean is reported as tactic coverage."""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import uuid

os.environ.setdefault("CONJ_REFUTE_DATASET_PROFILE", "D3")

from conj_refute_env.env import utils  # noqa: E402
from conj_refute_env.env import conj_refute_env as cre  # noqa: E402
from conj_refute_env.env import check_lean_theorem as clt  # noqa: E402
from conj_refute_env.env.conjecture_to_lean_statement import (  # noqa: E402
    translate_fol_to_lean_with_grind,
    FIXED_PREMISES_BY_PROFILE,
)

COMMON_TRUE = {
    "rank_nullity_1": "equals(width_D1, rank_D1 + nullity_D1)",
    "rank_nullity_2": "equals(width_D2, rank_D2 + nullity_D2)",
    "euler_poincare": "equals(height_D1 - width_D1 + width_D2, height_D1 - rank_D1 - nullity_D1 + rank_D2 + nullity_D2)",
    "dims_agree": "equals(height_D2, width_D1)",
    "implication": "implies(equals(nullity_D1, _0), equals(width_D1, rank_D1))",
    "conj_premise": "implies(conjunction(equals(rank_D1, _1), equals(nullity_D1, _1)), equals(width_D1, _2))",
}
CORE = {"rank_nullity_1", "rank_nullity_2", "euler_poincare"}
COMMON_FALSE = {
    "chi_eq_2": "equals(height_D1 - width_D1 + width_D2, _2)",
    "width_eq_rank": "equals(width_D1, rank_D1)",
    "rank_eq_rank": "equals(rank_D1, rank_D2)",
    "neg_rank_nullity": "logical_neg(equals(width_D1, rank_D1 + nullity_D1))",
}
B2_IS_1 = "equals(nullity_D2, _1)"
B0_IS_1 = "equals(height_D1 - rank_D1, _1)"
PROFILE_CASES = {
    "D0": {B2_IS_1: 1, B0_IS_1: 1},
    "D1": {B2_IS_1: 0, B0_IS_1: 1},
    "D2": {B2_IS_1: 0, B0_IS_1: 0},
    "D3": {B2_IS_1: 0, B0_IS_1: 0},
}
RAW = {
    "type_error": ('theorem cal_type_error_{s} : (1 : ℤ) = "a" := by\n  grind', 0, False),
    "slow_false": ("theorem cal_slow_false_{s} : (1 : ℤ) = 2 := by\n  sleep 25000\n  grind", 0, True),
    "sorry": ("theorem cal_sorry_{s} : (1 : ℤ) = 2 := by\n  sorry", 0, False),
    "slow_true": ("theorem cal_slow_true_{s} : (1 : ℤ) = 1 := by\n  sleep 25000\n  rfl", 1, True),
}


def timed(fn, *a, **k):
    t0 = time.time()
    out = fn(*a, **k)
    return out, time.time() - t0


def file_text(stmt):
    name = clt._theorem_name(stmt)
    return clt.header_text.rstrip() + "\n\n" + clt.marker_line + "\n" + stmt.rstrip() + f"\n\n#print axioms {name}\n"


def batch_verdict(stmt, tag):
    name = clt._theorem_name(stmt)
    rel = f"{clt.LEAN_SUBDIR}/calib_batch_{tag}.lean"
    path = clt.PROJECT_ROOT / rel
    path.write_text(file_text(stmt))
    t0 = time.time()
    try:
        p = subprocess.run(["lake", "env", "lean", rel], cwd=clt.PROJECT_ROOT, capture_output=True,
                           text=True, timeout=300)
        out, rc = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired:
        out, rc = "TIMEOUT", -1
    dt = time.time() - t0
    path.unlink(missing_ok=True)
    has_axioms = f"'{name}' depends on axioms" in out or f"'{name}' does not depend on any axioms" in out
    ok = rc == 0 and not re.search(r": error", out) and has_axioms and "sorryAx" not in out
    return int(ok), dt, out[-300:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", default=os.environ.get("CONJ_REFUTE_PROVER", "grind"))
    ap.add_argument("--profiles", default="D0,D1,D2,D3")
    ap.add_argument("--skip-legacy", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    assert cre.PROVER_BACKEND == args.gate, (cre.PROVER_BACKEND, args.gate)
    utils.assert_formal_response_assets(create_lean_dir=True)
    print(f"gate backend={args.gate} lean project={clt.PROJECT_ROOT}", flush=True)
    rows = []
    run_id = uuid.uuid4().hex[:8]
    env_gate, env_legacy = f"cal_gate_{run_id}", f"cal_legacy_{run_id}"

    warm, dt = timed(cre.compute_reward_terms, conj=COMMON_TRUE["dims_agree"], len_conj=1, blk_counts={},
                     run_id=run_id, env_id=env_gate, timestep=0)
    cold = {"profile": "-", "case": "cold_open_dims_agree", "expect": 1, "batch": 1, "slow": False,
            "gate": int(warm), "gate_s": round(dt, 2)}
    print(json.dumps(cold), flush=True)
    if not args.skip_legacy:
        _, dt = timed(clt.check_theorem_no_leancopilot,
                      translate_fol_to_lean_with_grind(COMMON_TRUE["dims_agree"], theorem_name="cal_warm"),
                      env_id=env_legacy, run_ts=0)
        print(f"legacy cold open {dt:.1f}s", flush=True)

    t = 1
    for p in args.profiles.split(","):
        utils.set_active_dataset_profile(p)
        cases = {**{k: (v, 1) for k, v in COMMON_TRUE.items()},
                 **{k: (v, 0) for k, v in COMMON_FALSE.items()},
                 **{("b2_is_1" if f == B2_IS_1 else "b0_is_1"): (f, e) for f, e in PROFILE_CASES[p].items()}}
        for name, (fol, expect) in cases.items():
            t += 1
            got, dt = timed(cre.compute_reward_terms, conj=fol, len_conj=1, blk_counts={},
                            run_id=run_id, env_id=env_gate, timestep=t)
            stmt = cre.translate_statement(fol_str=utils.fix_conj_str(fol),
                                           theorem_name=f"conjecture_run_{run_id}_timestep_{t}")
            bv, bdt, bout = batch_verdict(stmt, f"{run_id}_{t}")
            row = {"profile": p, "case": name, "fol": fol, "expect": expect, "slow": False,
                   "gate": int(got), "gate_s": round(dt, 2), "batch": bv, "batch_s": round(bdt, 1)}
            if bv != expect:
                row["batch_tail"] = bout
            if not args.skip_legacy:
                lstmt = translate_fol_to_lean_with_grind(
                    fol, fixed_premises=FIXED_PREMISES_BY_PROFILE[p], theorem_name=f"cal_legacy_{t}")
                lg, dl = timed(clt.check_theorem_no_leancopilot, lstmt, env_id=env_legacy, run_ts=t)
                row.update(legacy=int(lg), legacy_s=round(dl, 2))
            rows.append(row)
            print(json.dumps(row), flush=True)

    for name, (tmpl, expect, slow) in RAW.items():
        stmt = tmpl.format(s=run_id)
        if args.gate == "grind":
            (got, diags), dt = timed(clt.check_theorem_grind, stmt, env_id=env_gate, run_ts=0, return_diags=True)
        else:
            got, dt = timed(cre.check_statement, stmt, env_id=env_gate, run_ts=0)
            diags = None
        bv, bdt, _ = batch_verdict(stmt, f"{run_id}_{name}")
        row = {"profile": "-", "case": name, "fol": stmt, "expect": expect, "slow": slow,
               "gate": int(got), "gate_s": round(dt, 2), "batch": bv, "batch_s": round(bdt, 1),
               "gate_diags": [str(d.get("message", d))[:160] for d in (diags or [])]}
        if not args.skip_legacy:
            lg, dl = timed(clt.check_theorem_no_leancopilot, stmt.replace("cal_", "cal_lg_"),
                           env_id=env_legacy, run_ts=0)
            row.update(legacy=int(lg), legacy_s=round(dl, 2))
        rows.append(row)
        print(json.dumps(row), flush=True)

    false_pos = [r for r in rows if r["gate"] == 1 and r["batch"] == 0]
    false_neg = [r for r in rows if r["gate"] == 0 and r["batch"] == 1 and not r["slow"]]
    core_miss = [r for r in rows if r["case"] in CORE and (r["batch"] != 1 or r["gate"] != 1)]
    fails = false_pos + false_neg + core_miss + ([cold] if cold["gate"] != 1 else [])
    key = lambda r: (r["profile"], r["case"])  # noqa: E731
    legacy_s = sorted(r["legacy_s"] for r in rows if "legacy_s" in r)
    summary = {
        "gate": args.gate,
        "n": len(rows),
        "cold_open": cold,
        "gate_false_positive_vs_batch": [key(r) for r in false_pos],
        "gate_false_negative_vs_batch": [key(r) for r in false_neg],
        "core_not_proved": [key(r) for r in core_miss],
        "slow_probe_misses": [key(r) for r in rows if r["slow"] and r["gate"] != r["batch"]],
        "tactic_coverage_gaps": [key(r) for r in rows if r["batch"] != r["expect"]],
        "legacy_false_positive_vs_batch": [key(r) for r in rows if r.get("legacy") == 1 and r["batch"] == 0],
        "legacy_false_negative_vs_batch": [key(r) for r in rows if r.get("legacy") == 0 and r["batch"] == 1],
        "gate_s_max_nonslow": max(r["gate_s"] for r in rows if not r["slow"]),
        "legacy_s_median": legacy_s[len(legacy_s) // 2] if legacy_s else None,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "cold_open": cold, "rows": rows}, f, indent=2)
    print("SUMMARY", json.dumps(summary, indent=2))
    print("CALIBRATION PASS" if not fails else "CALIBRATION FAIL")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
