import pickle
from pathlib import Path

import pandas as pd

from conftest import load_module_from_src


depickle_and_table = load_module_from_src(
    "conj_refute_env/depickle_and_table.py", "depickle_and_table_under_test"
)
load_records = depickle_and_table.load_records
records_to_dataframe = depickle_and_table.records_to_dataframe
main = depickle_and_table.main


REQUIRED_COLUMNS = [
    "episode",
    "step",
    "conjecture",
    "uniform_block_val",
    "repeat_block_val",
    "formal_response",
    "rew_conj_formal",
    "rew_conj_uniform",
    "rew_conj_length",
    "reward_ref",
]


def sample_records():
    return [
        {
            "episode": 1,
            "step": 2,
            "raw_conjecture": "equals(a,b)",
            "uniform_block_val": 0.1,
            "repeat_block_val": 0.2,
            "formal_response": 1,
            "rew_conj_formal": 3.0,
            "rew_conj_uniform": 4.0,
            "rew_conj_div": 5.0,
            "cum_reward_ref": 6.0,
        }
    ]


def test_records_to_dataframe_has_expected_columns():
    df = records_to_dataframe(sample_records())
    assert list(df.columns) == REQUIRED_COLUMNS
    assert df.iloc[0]["conjecture"] == "equals(a,b)"


def test_load_records_round_trip(tmp_path: Path):
    input_file = tmp_path / "step_data.pkl"
    records = sample_records()
    with input_file.open("wb") as f:
        pickle.dump(records, f)

    loaded = load_records(str(input_file))
    assert loaded == records


def test_main_writes_csv(tmp_path: Path, monkeypatch):
    input_file = tmp_path / "step_data.pkl"
    output_csv = tmp_path / "step_data.csv"
    with input_file.open("wb") as f:
        pickle.dump(sample_records(), f)

    monkeypatch.setattr(
        "sys.argv",
        [
            "depickle_and_table.py",
            "--input-file",
            str(input_file),
            "--output-csv",
            str(output_csv),
        ],
    )

    main()
    assert output_csv.exists()

    df = pd.read_csv(output_csv)
    assert list(df.columns) == REQUIRED_COLUMNS
    assert len(df) == 1
