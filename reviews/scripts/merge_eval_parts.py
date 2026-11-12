"""Same merge as wave_eval.sbatch: eval_parts/ep{0..n-1}.pkl -> one pickle, refusing if any episode is missing."""

import pickle
import sys
from pathlib import Path

run, n, out = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
recs, missing = [], []
for ep in range(n):
    p = run / "eval_parts" / f"ep{ep}.pkl"
    if not p.is_file():
        missing.append(ep)
        continue
    for r in pickle.loads(p.read_bytes()):
        r["episode"] = ep
        recs.append(r)
if missing:
    sys.exit(f"missing episodes {missing}; not merging")
Path(out).write_bytes(pickle.dumps(recs))
print(f"merged {len(recs)} records from {n} episodes -> {out}")
