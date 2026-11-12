"""Collect every arm's episode_metrics_*_vs_paper.json and eval CSV counts into one long-form CSV."""

import argparse
import csv
import json
from pathlib import Path

DATASETS = ["D0", "D1", "D2", "D3"]
SCOPES = ["s0", "s1", "pooled"]
CAP = 50


def counts(path):
    eps = {}
    for row in csv.DictReader(open(path)):
        if row["conjecture"]:
            eps.setdefault(row["episode"], []).append(float(row["formal_response"]))
    return {
        "episodes": len(eps),
        "statements": sum(len(v) for v in eps.values()),
        "proofs": sum(sum(v) for v in eps.values()),
        "eps_ended_early": sum(1 for v in eps.values() if len(v) < CAP),
    }


def metric_rows(base, side_stats):
    for section in ("pooled_stats", "episode_stats"):
        for name, (value, ci, *_rest) in side_stats[section].items():
            yield {**base, "metric": name, "value": value, "lo": ci[0], "hi": ci[1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.wave_dir) / "analysis"
    fields = ["arm", "dataset", "scope", "side", "episodes", "statements", "proofs", "eps_ended_early",
              "steps_median", "metric", "value", "lo", "hi", "p_ours_gt_paper"]
    rows, missing = [], []
    for arm_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        arm = arm_dir.name
        for ds in DATASETS:
            for scope in SCOPES:
                name = f"{ds}_{arm}_{scope}"
                jf = arm_dir / ds / "readable" / f"episode_metrics_{name}_vs_paper.json"
                if not jf.is_file():
                    missing.append(name)
                    continue
                j = json.loads(jf.read_text())
                ours = {"arm": arm, "dataset": ds, "scope": scope, "side": "ours",
                        "steps_median": j["A"]["steps_median"],
                        **counts(arm_dir / ds / "csv" / f"{name}.csv")}
                ours["statements"] = counts(arm_dir / ds / "readable" / "csv" / f"{name}_readable.csv")["statements"]
                rows.extend(metric_rows(ours, j["A"]))
                if scope != "pooled":
                    continue
                paper = {"arm": arm, "dataset": ds, "scope": "paper", "side": "paper",
                         "steps_median": j["B"]["steps_median"],
                         **counts(arm_dir / ds / "csv" / f"paper_{ds}_M0_eval.csv")}
                rows.extend(metric_rows(paper, j["B"]))
                for section in ("deltas_pooled_union", "deltas_episode_mean"):
                    for metric, v in j[section].items():
                        rows.append({**ours, "side": "ratio_ours_over_paper", "metric": metric, "value": v[0],
                                     "lo": v[1][0], "hi": v[1][1],
                                     "p_ours_gt_paper": v[3] if len(v) > 3 else ""})
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {a.out}; missing: {len(missing)}")
    for m in missing:
        print("  missing", m)


if __name__ == "__main__":
    main()
