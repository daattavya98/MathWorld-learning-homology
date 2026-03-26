from pathlib import Path
from types import SimpleNamespace

from conftest import load_module_from_src


ROOT = Path(__file__).resolve().parents[1]


def _build_override_args():
    return SimpleNamespace(
        formal_response_enabled=False,
        noise_experiment_enabled=True,
        strict_proof_check=False,
        formal_response_policy="always_on",
        formal_response_reward_mode="disabled",
        formal_response_noise_level=0.125,
        formal_response_proof_threshold=0.8,
    )


def _base_cfg():
    return {
        "formal_response_enabled": True,
        "noise_experiment_enabled": False,
        "identity_refuter_enabled": False,
        "strict_proof_check": True,
        "formal_response_policy": "auto",
        "formal_response_reward_mode": "standard",
        "formal_response_noise_level": 0.0,
        "formal_response_proof_threshold": 1.0,
    }


def test_train_override_precedence():
    cli_utils = load_module_from_src("conj_refute_env/cli_utils.py", "cli_utils_train")
    apply_overrides = cli_utils.apply_env_mode_overrides

    cfg = _base_cfg()
    updated = apply_overrides(cfg, _build_override_args())

    assert updated["formal_response_enabled"] is False
    assert updated["noise_experiment_enabled"] is True
    assert updated["strict_proof_check"] is False
    assert updated["formal_response_policy"] == "always_on"
    assert updated["formal_response_reward_mode"] == "disabled"
    assert updated["formal_response_noise_level"] == 0.125
    assert updated["formal_response_proof_threshold"] == 0.8


def test_pickle_override_precedence():
    cli_utils = load_module_from_src("conj_refute_env/cli_utils.py", "cli_utils_pickle")
    apply_overrides = cli_utils.apply_env_mode_overrides

    cfg = _base_cfg()
    updated = apply_overrides(cfg, _build_override_args())

    assert updated["formal_response_enabled"] is False
    assert updated["noise_experiment_enabled"] is True
    assert updated["strict_proof_check"] is False
    assert updated["formal_response_policy"] == "always_on"
    assert updated["formal_response_reward_mode"] == "disabled"
    assert updated["formal_response_noise_level"] == 0.125
    assert updated["formal_response_proof_threshold"] == 0.8


def test_model_profiles_apply_expected_defaults():
    cli_utils = load_module_from_src("conj_refute_env/cli_utils.py", "cli_utils_profiles")

    cfg = _base_cfg()
    m0 = cli_utils.apply_model_profile_defaults(cfg, "M0")
    assert m0["formal_response_enabled"] is True
    assert m0["noise_experiment_enabled"] is False
    assert m0["identity_refuter_enabled"] is False

    m1 = cli_utils.apply_model_profile_defaults(cfg, "M1")
    assert m1["formal_response_enabled"] is False
    assert m1["noise_experiment_enabled"] is False
    assert m1["identity_refuter_enabled"] is False

    m0_noise = cli_utils.apply_model_profile_defaults(cfg, "M0_NOISE")
    assert m0_noise["formal_response_enabled"] is True
    assert m0_noise["noise_experiment_enabled"] is True
    assert m0_noise["identity_refuter_enabled"] is False

    m2 = cli_utils.apply_model_profile_defaults(cfg, "M2")
    assert m2["formal_response_enabled"] is True
    assert m2["noise_experiment_enabled"] is False
    assert m2["identity_refuter_enabled"] is True


def test_model_profile_alias_normalization():
    cli_utils = load_module_from_src(
        "conj_refute_env/cli_utils.py", "cli_utils_profile_alias"
    )
    assert cli_utils.normalize_model_profile("m0+noise") == "M0_NOISE"


def test_manual_overrides_win_over_model_profile():
    cli_utils = load_module_from_src(
        "conj_refute_env/cli_utils.py", "cli_utils_profile_precedence"
    )

    cfg = _base_cfg()
    cfg = cli_utils.apply_model_profile_defaults(cfg, "M0")
    updated = cli_utils.apply_env_mode_overrides(cfg, _build_override_args())

    assert updated["formal_response_enabled"] is False
    assert updated["noise_experiment_enabled"] is True
    assert updated["strict_proof_check"] is False


def test_analyze_blocks_cli_main_smoke(monkeypatch):
    monkeypatch.setenv("CONJ_REFUTE_DATASET_PROFILE", "D0")
    mod = load_module_from_src("analyze_blocks.py", "analyze_blocks_cli_smoke")
    calls = {"analyze": None, "count": None}

    def _fake_analyze_statistics(csv_name, premises, save_outputs, output_root):
        calls["analyze"] = {
            "csv_name": csv_name,
            "premises": premises,
            "save_outputs": save_outputs,
            "output_root": output_root,
        }

    def _fake_count(csv_name, premises):
        calls["count"] = {"csv_name": csv_name, "premises": premises}
        return 2, [1, 0]

    monkeypatch.setattr(mod, "analyze_statistics", _fake_analyze_statistics)
    monkeypatch.setattr(
        mod,
        "count_formal_response_with_chi_and_homology",
        _fake_count,
    )

    mod.main(
        [
            "--csv-name",
            "demo.csv",
            "--premise",
            "rank-nullity",
            "--premise",
            "connected",
            "--output-root",
            "out",
            "--no-save-outputs",
        ]
    )

    assert calls["analyze"] == {
        "csv_name": "demo.csv",
        "premises": ["rank-nullity", "connected"],
        "save_outputs": False,
        "output_root": "out",
    }
    assert calls["count"] == {
        "csv_name": "demo.csv",
        "premises": ["rank-nullity", "connected"],
    }


def test_analyze_blocks_cli_skip_formalized_summary(monkeypatch):
    monkeypatch.setenv("CONJ_REFUTE_DATASET_PROFILE", "D0")
    mod = load_module_from_src(
        "analyze_blocks.py", "analyze_blocks_cli_skip_formalized"
    )

    monkeypatch.setattr(
        mod,
        "analyze_statistics",
        lambda csv_name, premises, save_outputs, output_root: None,
    )

    called = {"count": False}

    def _fake_count(csv_name, premises):
        called["count"] = True
        return 0, []

    monkeypatch.setattr(mod, "count_formal_response_with_chi_and_homology", _fake_count)

    mod.main(["--csv-name", "demo.csv", "--skip-formalized-summary"])

    assert called["count"] is False


def test_analyze_episodes_cli_modes(monkeypatch):
    monkeypatch.setenv("CONJ_REFUTE_DATASET_PROFILE", "D0")
    mod = load_module_from_src("analyze_episodes.py", "analyze_episodes_cli_smoke")

    calls = {"one": None, "all": None}

    def _fake_convert_one(input_file, output_dir=None):
        calls["one"] = {"input_file": input_file, "output_dir": output_dir}
        return "out.csv", 3

    def _fake_convert_all(model_evaluations_dir=None, output_dir=None):
        calls["all"] = {
            "model_evaluations_dir": model_evaluations_dir,
            "output_dir": output_dir,
        }
        return [{"status": "ok"}]

    monkeypatch.setattr(mod, "convert_conj_to_readable", _fake_convert_one)
    monkeypatch.setattr(
        mod,
        "convert_all_model_evaluation_csvs_to_readable",
        _fake_convert_all,
    )

    out_file, rows = mod.main(
        ["--mode", "convert-one", "--input-file", "in.csv", "--output-dir", "rd"]
    )
    assert (out_file, rows) == ("out.csv", 3)
    assert calls["one"] == {"input_file": "in.csv", "output_dir": "rd"}

    res = mod.main(
        [
            "--mode",
            "convert-all",
            "--model-evaluations-dir",
            "me",
            "--output-dir",
            "rd2",
        ]
    )
    assert isinstance(res, list)
    assert calls["all"] == {"model_evaluations_dir": "me", "output_dir": "rd2"}
