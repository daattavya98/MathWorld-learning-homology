"""Write runs/<wave>/tasks.tsv: one row per (arm, dataset, seed); array index = row index."""

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", required=True)
    ap.add_argument("--arms", default=str(ROOT / "scripts/sweep_arms.tsv"))
    ap.add_argument("--only-arms", default=None, help="comma-separated arm ids")
    ap.add_argument("--datasets", default="D0,D1,D2,D3")
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()

    with open(a.arms, newline="") as f:
        arms = [(r["arm"], r["flags"] or "") for r in csv.DictReader(f, delimiter="\t")]
    if a.only_arms:
        keep = set(a.only_arms.split(","))
        arms = [x for x in arms if x[0] in keep]
    rows = [
        (arm, ds, int(s), flags)
        for arm, flags in arms
        for ds in a.datasets.split(",")
        for s in a.seeds.split(",")
    ]
    out = ROOT / "runs" / a.wave / "tasks.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = "idx\tarm\tdataset\tseed\tflags\n" + "".join(
        f"{i}\t{arm}\t{ds}\t{s}\t{flags}\n" for i, (arm, ds, s, flags) in enumerate(rows)
    )
    if out.exists() and out.read_text() != text:
        raise SystemExit(f"{out} exists with different contents; use a new wave name")
    out.write_text(text)
    print(f"{len(rows)} tasks -> {out}  (--array=0-{len(rows) - 1})")


if __name__ == "__main__":
    main()
