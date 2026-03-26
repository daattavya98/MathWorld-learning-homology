import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
BASE_CFG_PATH = ROOT / "src" / "base_run_config.json"


def _base_cfg() -> dict:
    return json.loads(BASE_CFG_PATH.read_text())


def _load_modules(monkeypatch):
    monkeypatch.setenv("CONJ_REFUTE_DATASET_PROFILE", "D0")

    import conj_refute_env.cli_utils as cli_utils
    import conj_refute_env.env.utils as env_utils

    env_utils.set_active_dataset_profile("D0")

    import conj_refute_env.env.conj_refute_env as env_mod

    return cli_utils, env_mod


def _make_env_for_profile(monkeypatch, profile: str):
    cli_utils, env_mod = _load_modules(monkeypatch)
    cfg = cli_utils.apply_model_profile_defaults(_base_cfg(), profile)
    env = env_mod.MathWorld(env_config=cfg)
    env.reset(seed=7)
    return env, env_mod


def test_m0_enables_formal_checks_and_refuter_is_active(monkeypatch):
    env, _ = _make_env_for_profile(monkeypatch, "M0")

    assert env.formal_response_default_enabled is True
    assert env.noise_experiment_enabled is False
    assert env.identity_refuter_enabled is False

    assert env._resolve_formal_response_enabled(True, True) is True
    assert env._resolve_formal_response_enabled(False, True) is False

    v0 = env.refuter_datapoint_vectors[0]
    expected_active = int(0.6 * v0.size)
    assert np.count_nonzero(v0) == expected_active
    assert np.any(v0 == 0.0)
    assert np.any(v0 == 1.0)


def test_m1_disables_formal_checks(monkeypatch):
    env, _ = _make_env_for_profile(monkeypatch, "M1")

    assert env.formal_response_default_enabled is False
    assert env.noise_experiment_enabled is False
    assert env.identity_refuter_enabled is False

    assert env._resolve_formal_response_enabled(True, True) is False


def test_m2_identity_refuter_starts_on_and_stays_on(monkeypatch):
    env, env_mod = _make_env_for_profile(monkeypatch, "M2")

    assert env.formal_response_default_enabled is True
    assert env.identity_refuter_enabled is True

    for vec in env.refuter_datapoint_vectors:
        assert np.all(vec == 1.0)

    monkeypatch.setattr(
        env_mod,
        "atomic_from_atomic_ga_params",
        lambda atomic_ga_params, data_vector: ("equals(1,1)", 1),
    )
    monkeypatch.setattr(
        env_mod,
        "evaluate_conjecture_on_dataset",
        lambda conjecture, data_vector_to_use: np.ones_like(data_vector_to_use),
    )
    monkeypatch.setattr(
        env_mod,
        "scaffolded_conj_from_ga_params",
        lambda scaffolding_ga_params, ref_data_vector, id, atomic_block_length: (
            "equals(1,1)",
            1,
            {"equals(1,1)": 1},
            {"blk_0": "equals(1,1)"},
        ),
    )
    monkeypatch.setattr(env_mod, "compute_reward_terms", lambda **kwargs: 0)
    monkeypatch.setattr(
        env_mod,
        "compute_uniformity_and_repeats",
        lambda self, original_block_names, blk_counts, repeat_penalty: (
            0.0,
            0.0,
            {"equals(1,1)": 0.5},
            {"blk_0": 0.5},
        ),
    )
    monkeypatch.setattr(env_mod.pd.DataFrame, "to_csv", lambda self, path, index=False: None)

    ref_dim = env.action_space("refuter").shape[0]
    conj_dim = env.action_space("conjecturer").shape[0]
    ref_action = np.zeros(ref_dim, dtype=np.float64)
    ref_action[0::2] = 0.0
    ref_action[1::2] = -1.0

    actions = {
        "conjecturer": np.zeros(conj_dim, dtype=np.float64),
        "refuter": ref_action,
    }

    env.step(actions)

    for vec in env.refuter_datapoint_vectors:
        assert np.all(vec == 1.0)


def test_m0_noise_enables_noise_and_numeric_noise_behavior(monkeypatch):
    env, env_mod = _make_env_for_profile(monkeypatch, "M0_NOISE")

    assert env.formal_response_default_enabled is True
    assert env.noise_experiment_enabled is True
    assert env.identity_refuter_enabled is False

    np.random.seed(0)
    noisy_from_one = env_mod.add_noise(formal_response=1, noise_level=0.12)
    noisy_from_zero = env_mod.add_noise(formal_response=0, noise_level=0.12)

    assert noisy_from_one == pytest.approx(0.788313717, rel=1e-6, abs=1e-6)
    assert noisy_from_zero == pytest.approx(0.048018865, rel=1e-6, abs=1e-6)
