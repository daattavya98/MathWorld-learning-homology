import json

import pytest

from conftest import load_module_from_src


def _load_utils(monkeypatch):
    monkeypatch.setenv("CONJ_REFUTE_DATASET_PROFILE", "D0")
    monkeypatch.setenv("CONJ_REFUTE_ROWS_PER_DATASET", "350")
    return load_module_from_src("conj_refute_env/env/utils.py", "env_utils_model_lock")


def test_run_lock_writes_model_profile(tmp_path, monkeypatch):
    utils = _load_utils(monkeypatch)

    lock_path = utils.write_or_validate_run_lock(
        config_dir=str(tmp_path),
        run_id="run_1",
        config_id="base_run_config",
        model_profile="M2",
    )
    payload = json.loads(lock_path.read_text())

    assert payload["model_profile"] == "M2"
    lock = utils.enforce_run_lock(
        config_dir=str(tmp_path),
        run_id="run_1",
        model_profile="M2",
    )
    assert lock["model_profile"] == "M2"


def test_run_lock_rejects_model_profile_mismatch(tmp_path, monkeypatch):
    utils = _load_utils(monkeypatch)

    utils.write_or_validate_run_lock(
        config_dir=str(tmp_path),
        run_id="run_2",
        config_id="base_run_config",
        model_profile="M0",
    )

    with pytest.raises(ValueError, match="model_profile"):
        utils.enforce_run_lock(
            config_dir=str(tmp_path),
            run_id="run_2",
            model_profile="M1",
        )


def test_legacy_lock_backfills_model_profile(tmp_path, monkeypatch):
    utils = _load_utils(monkeypatch)

    legacy_path = tmp_path / "run_3_dataset_lock.json"
    legacy_payload = {
        "run_id": "run_3",
        "config_id": "base_run_config",
        "dataset_profile": "D0",
        "rows_per_dataset": 350,
        "created_at_utc": "2026-03-25T00:00:00+00:00",
        "lock_version": 1,
    }
    legacy_path.write_text(json.dumps(legacy_payload, indent=2))

    lock = utils.enforce_run_lock(
        config_dir=str(tmp_path),
        run_id="run_3",
        model_profile="M0",
    )

    assert lock["model_profile"] == "M0"
    updated_payload = json.loads(legacy_path.read_text())
    assert updated_payload["model_profile"] == "M0"
