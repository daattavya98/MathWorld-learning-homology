"""One W&B run per merged eval pickle, same metric keys as pickle_my_model.py."""

import argparse
import os
import pickle
from pathlib import Path

import wandb

KEYS = [
    "episode", "step", "cum_reward_conj", "cum_reward_ref", "rew_conj_formal", "rew_conj_div",
    "rew_ref_formal", "rew_ref_data", "rew_ref_spread", "formal_response", "terminated_by_proof",
    "truncated",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--pickle", default="eval_e10_s50.pkl")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()

    run = Path(a.run_dir)
    marker = run / "wandb_eval_uploaded.txt"
    if marker.exists():
        print(f"already uploaded: {marker.read_text().strip()}")
        return
    recs = pickle.loads((run / a.pickle).read_bytes())
    recs.sort(key=lambda r: (r["episode"], r["step"]))
    wandb.require("core")
    w = wandb.init(
        project="conj_refute_maddpg",
        entity="science_ai",
        name=f"{run.name}_eval",
        mode="online",
        job_type="eval",
        tags=["merged_eval"],
        config={
            "run_id": run.name, "arm": a.arm, "dataset_profile": a.dataset, "seed": a.seed,
            "episodes": len({r["episode"] for r in recs}), "model_profile": "M0",
            "prover_backend": os.environ.get("CONJ_REFUTE_PROVER"),
        },
    )
    for r in recs:
        wandb.log({k: r[k] for k in KEYS})
    url = w.url
    wandb.finish()
    marker.write_text(url + "\n")
    print(f"uploaded {len(recs)} steps -> {url}")


if __name__ == "__main__":
    main()
