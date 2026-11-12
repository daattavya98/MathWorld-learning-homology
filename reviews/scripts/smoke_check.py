import argparse
import json
import math
import pickle
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

CENTRE = {
    "lr_actor": 1e-3, "lr_critic": 1e-2, "gamma": 0.95, "tau": 0.01, "batch_size": 64,
    "O_U_noise": True, "expl_noise": 0.1, "mean_noise": 0.0, "theta": 0.15, "dt": 1e-2,
}
LOG_PATTERNS = {
    "lean_exception": r"Lean proof unsuccessful",
    "lsp_timeout": r"_wait_for_diagnostics` timed out",
    "problematic_conjecture": r"Problematic conjecture",
    "no_equations": r"GA produced no equations",
    "traceback": r"Traceback \(most recent call last\)",
    "worker_died": r"\[Recovery\] Worker died",
}


def q(xs):
    if not xs:
        return None
    xs = sorted(xs)
    return {"n": len(xs), "min": round(xs[0], 1), "median": round(st.median(xs), 1), "max": round(xs[-1], 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--train-rc", type=int, required=True)
    ap.add_argument("--eval-rc", type=int, required=True)
    ap.add_argument("--num-envs", type=int, required=True)
    ap.add_argument("--max-steps", type=int, required=True)
    ap.add_argument("--eval-steps", type=int, required=True)
    a = ap.parse_args()
    run = Path(a.run_dir)
    run_id = run.name
    checks, info = {}, {}

    checks["train_rc0"] = a.train_rc == 0
    checks["eval_rc0"] = a.eval_rc == 0
    checks["checkpoint"] = (run / "ckpt" / f"{run_id}_marl.pt").is_file()

    hp_path = run / "ckpt" / f"{run_id}_hparams.json"
    if hp_path.is_file():
        hp = json.loads(hp_path.read_text())
        def same(a, b):
            if isinstance(b, bool) or a is None:
                return a == b
            return math.isclose(float(a), float(b), rel_tol=1e-6)

        diff = {k: (hp["maddpg"].get(k), v) for k, v in CENTRE.items() if not same(hp["maddpg"].get(k), v)}
        checks["hparams_centre"] = not diff
        info["hparams_diff"] = diff
        info["prover_backend"] = hp.get("prover_backend")
        info["julia_threads"] = hp.get("julia_threads")
    else:
        checks["hparams_centre"] = False

    rec_path = run / "train_records.jsonl"
    recs = [json.loads(l) for l in rec_path.read_text().splitlines()] if rec_path.is_file() else []
    vec_steps = a.max_steps // a.num_envs
    checks["train_records_count"] = len(recs) == vec_steps * a.num_envs
    by_step = defaultdict(list)
    for r in recs:
        by_step[r["total_steps"]].append(r)
    stamps = sorted(min(r["t_wall"] for r in rs) for rs in by_step.values())
    durs = [b - a_ for a_, b in zip(stamps, stamps[1:])]
    fr = [r.get("formal_response") for r in recs]
    info["train"] = {
        "records": len(recs),
        "vec_step_s": q(durs),
        "formal_response_counts": {str(v): fr.count(v) for v in sorted(set(map(str, fr)))} if fr else {},
        "proved": sum(1 for v in fr if v == 1),
        "empty_conjecture": sum(1 for r in recs if not r.get("conjecture")),
        "terminated": sum(1 for r in recs if r.get("term_conjecturer")),
        "examples": [r.get("conjecture") for r in recs if r.get("formal_response") == 1][:3],
    }

    ev_path = run / "eval.pkl"
    if ev_path.is_file():
        ev = pickle.loads(ev_path.read_bytes())
        ts = [r["t_wall"] for r in ev]
        info["eval"] = {
            "records": len(ev),
            "step_s": q([b - a_ for a_, b in zip(ts, ts[1:])]),
            "proved": sum(1 for r in ev if r["formal_response"] == 1),
            "terminated_by_proof": sum(1 for r in ev if r["terminated_by_proof"]),
        }
        checks["eval_records"] = 1 <= len(ev) <= a.eval_steps
    else:
        checks["eval_records"] = False

    for name in ["train.log", "eval.log"]:
        p = run / name
        txt = p.read_text(errors="replace") if p.is_file() else ""
        info[name] = {k: len(re.findall(v, txt)) for k, v in LOG_PATTERNS.items()}
        m = re.search(r"Elapsed \(wall clock\) time.*?: (\S+)", txt)
        info[name]["wall"] = m.group(1) if m else None
        m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", txt)
        info[name]["max_rss_gb"] = round(int(m.group(1)) / 1e6, 2) if m else None
        checks[f"{name}_no_traceback"] = info[name]["traceback"] == 0
    checks["no_lean_exceptions"] = info["train.log"]["lean_exception"] + info["eval.log"]["lean_exception"] == 0

    ok = all(checks.values())
    out = {"run_id": run_id, "pass": ok, "checks": checks, "info": info}
    (run / "smoke_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("SMOKE PASS" if ok else "SMOKE FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
