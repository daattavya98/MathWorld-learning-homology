"""Exit nonzero unless raw and readable forms of the same CSV give identical episode statistics,
and every row of the given paper CSV parses. Rows convert_conj_to_readable drops are listed and
excluded from the raw side; only rows with a negative decimal literal may be dropped."""

import argparse
import re
import sys
import tempfile

import pandas as pd

from analysis_utils import is_readable_conjecture, readable_to_raw
from analyze_episodes import compute_episode_stats_objects

FIELDS = ["total_blocks_occ", "chi_blocks_occ", "homology_blocks_occ", "proven_total",
          "proven_with_ch_or_hom", "proven_with_chi", "proven_with_homology", "steps"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--readable", required=True)
    ap.add_argument("--paper", required=True)
    ap.add_argument("--premise", action="append", required=True)
    a = ap.parse_args()

    bad = 0
    paper = pd.read_csv(a.paper).dropna(subset=["conjecture"])
    for c in paper["conjecture"]:
        if not is_readable_conjecture(c):
            print("paper row not readable form:", c)
            bad += 1
            continue
        raw = readable_to_raw(c)
        if "(" not in raw:
            print("paper row did not convert:", c)
            bad += 1
    print(f"paper rows parsed: {len(paper)}")

    raw_df = pd.read_csv(a.raw)
    readable_df = pd.read_csv(a.readable)
    keep = pd.MultiIndex.from_frame(raw_df[["episode", "step"]]).isin(
        pd.MultiIndex.from_frame(readable_df[["episode", "step"]]))
    dropped = raw_df[~keep]
    print(f"rows dropped by converter: {len(dropped)} of {len(raw_df)}")
    for _, r in dropped.iterrows():
        print(f"  dropped episode {r['episode']} step {r['step']}: {r['conjecture']}")
        if not re.search(r"-\d+\.\d", str(r["conjecture"])):
            print("  dropped row has no negative decimal literal")
            bad += 1

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        raw_df[keep].to_csv(f.name, index=False)
        kept_raw = f.name

    ra = {e.episode_id: e for e in compute_episode_stats_objects(kept_raw, premises=a.premise)}
    rd = {e.episode_id: e for e in compute_episode_stats_objects(a.readable, premises=a.premise)}
    if set(ra) != set(rd):
        print("episode ids differ:", sorted(ra), sorted(rd))
        bad += 1
    for ep in sorted(set(ra) & set(rd)):
        for f in FIELDS:
            x, y = getattr(ra[ep], f), getattr(rd[ep], f)
            if x != y:
                print(f"episode {ep} {f}: raw={x} readable={y}")
                bad += 1
    print("ROUNDTRIP PASS" if not bad else f"ROUNDTRIP FAIL ({bad})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
