# depickle_and_table.py

import argparse
import pickle
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Load step_data.pkl, build DataFrame, print Markdown table, and save CSV"
    )
    p.add_argument("--input-file", type=str, default="step_data.pkl")
    p.add_argument("--output-csv", type=str, default="step_data.csv")
    return p.parse_args()


def load_records(input_file):
    with open(input_file, "rb") as f:
        return pickle.load(f)


def records_to_dataframe(records):
    rows = []
    for r in records:
        rows.append(
            {
                "episode": r["episode"],
                "step": r["step"],
                "conjecture": r["raw_conjecture"],
                "uniform_block_val": r["uniform_block_val"],
                "repeat_block_val": r["repeat_block_val"],
                "formal_response": r["formal_response"],
                "rew_conj_formal": r["rew_conj_formal"],
                "rew_conj_uniform": r["rew_conj_uniform"],
                "rew_conj_length": r["rew_conj_div"],
                "reward_ref": r["cum_reward_ref"],
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()

    records = load_records(args.input_file)
    df = records_to_dataframe(records)

    print(df.to_markdown(index=False))
    df.to_csv(args.output_csv, index=False)
    print(f"Saved CSV → {args.output_csv}")


if __name__ == "__main__":
    main()
