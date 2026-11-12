"""Calls analyze_episodes.compute_all_metrics_with_cis unchanged and writes its output as JSON."""

import argparse
import json

import numpy as np

from analyze_episodes import compute_all_metrics_with_cis


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-a", required=True)
    ap.add_argument("--csv-b", default=None)
    ap.add_argument("--premise", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = compute_all_metrics_with_cis(a.csv_a, a.csv_b, premises=a.premise)
    with open(a.out, "w") as f:
        json.dump(clean(res), f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
