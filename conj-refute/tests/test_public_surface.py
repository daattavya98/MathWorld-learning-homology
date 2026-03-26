from pathlib import Path

from conftest import load_module_from_src


def test_required_runtime_files_exist():
    root = Path(__file__).resolve().parents[1]
    workspace_root = root.parent

    required_paths = [
        root / "src" / "conj_refute_env" / "train_maddpg.py",
        root / "src" / "conj_refute_env" / "pickle_my_model.py",
        root / "src" / "conj_refute_env" / "depickle_and_table.py",
        root / "src" / "conj_refute_env" / "env" / "conj_refute_env.py",
        root / "src" / "analyze_episodes.py",
        root / "src" / "analyze_blocks.py",
        root / "src" / "base_run_config.json",
        workspace_root / "formal_response" / "lean-toolchain",
        workspace_root / "formal_response" / "template.lean",
    ]

    missing = [str(p) for p in required_paths if not p.exists()]
    assert not missing, f"Missing required files: {missing}"


def test_train_script_uses_parse_args_in_main():
    root = Path(__file__).resolve().parents[1]
    train_script = root / "src" / "conj_refute_env" / "train_maddpg.py"
    text = train_script.read_text(encoding="utf-8")

    assert "def parse_args():" in text
    assert "args = parse_args()" in text


def test_workflow_scripts_define_expected_cli_arguments():
    root = Path(__file__).resolve().parents[1]

    train_text = (root / "src" / "conj_refute_env" / "train_maddpg.py").read_text(
        encoding="utf-8"
    )
    pickle_text = (root / "src" / "conj_refute_env" / "pickle_my_model.py").read_text(
        encoding="utf-8"
    )
    depickle_text = (
        root / "src" / "conj_refute_env" / "depickle_and_table.py"
    ).read_text(encoding="utf-8")
    analyze_blocks_text = (root / "src" / "analyze_blocks.py").read_text(
        encoding="utf-8"
    )
    analyze_episodes_text = (root / "src" / "analyze_episodes.py").read_text(
        encoding="utf-8"
    )

    for flag in [
        "--env-config",
        "--base-config",
        "--model-profile",
        "--config-dir",
        "--num-envs",
        "--max-steps",
        "--train-block",
        "--checkpoint-interval",
        "--wandb-mode",
        "formal-response-enabled",
        "noise-experiment-enabled",
        "--formal-response-policy",
        "--formal-response-reward-mode",
        "--formal-response-noise-level",
        "--formal-response-proof-threshold",
        "strict-proof-check",
    ]:
        assert flag in train_text

    for flag in [
        "--run-id",
        "--config-id",
        "--model-profile",
        "--episodes",
        "--max-steps",
        "--output-file",
        "--wandb-mode",
        "formal-response-enabled",
        "noise-experiment-enabled",
        "--formal-response-policy",
        "--formal-response-reward-mode",
        "--formal-response-noise-level",
        "--formal-response-proof-threshold",
        "strict-proof-check",
    ]:
        assert flag in pickle_text

    for flag in ["--input-file", "--output-csv"]:
        assert flag in depickle_text

    for marker in ["def parse_args", "def main", "--csv-name", "--premise"]:
        assert marker in analyze_blocks_text

    for marker in ["def parse_args", "def main", "--mode", "convert-all"]:
        assert marker in analyze_episodes_text


def test_required_julia_loss_snippets_exist():
    root = Path(__file__).resolve().parents[1]
    snippets_dir = root / "src" / "conj_refute_env" / "julia_snippets"

    required = {
        "loss_atomic_formulae.jl",
        "loss_scaffolding.jl",
    }

    missing = [name for name in sorted(required) if not (snippets_dir / name).exists()]
    assert not missing, f"Missing required Julia snippet files: {missing}"


def test_dataset_profiles_resolve_to_existing_manifold_files():
    root = Path(__file__).resolve().parents[1]
    manifold_data_dir = root / "src" / "conj_refute_env" / "manifold_data"

    utils = load_module_from_src("conj_refute_env/env/utils.py", "env_utils_profiles")

    all_resolved: set[str] = set()
    for profile in ["D0", "D1", "D2", "D3"]:
        filenames = utils.get_profile_dataset_filenames(profile)
        assert filenames, f"Profile {profile} resolved to no dataset files"
        all_resolved.update(filenames)

    missing = [
        name for name in sorted(all_resolved) if not (manifold_data_dir / name).exists()
    ]
    assert not missing, f"Missing manifold dataset files referenced by profiles: {missing}"
