import csv, sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
dst = root / "MathWorld-learning-homology/conj-refute/src/conj_refute_env/manifold_data"
bad = 0
for line in (root / "env/datasets_map.tsv").read_text().splitlines():
    target, _, chi, b0, b2 = line.split("\t")
    want = (int(chi), int(b0), int(b2))
    with open(dst / target, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    sigs = set()
    for r in rows:
        V, E, F = int(r["height_δ1"]), int(r["width_δ1"]), int(r["width_δ2"])
        sigs.add((V - E + F, V - int(r["rank_δ1"]), int(r["nullity_δ2"])))
    ok = sigs == {want}
    bad += not ok
    print(f"{'OK ' if ok else 'BAD'} {target:40s} rows={len(rows):4d} sig={sorted(sigs)} want={want}")
sys.exit(1 if bad else 0)
