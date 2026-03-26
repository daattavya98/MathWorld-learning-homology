import argparse


MODEL_PROFILES = {
    "M0": {
        "formal_response_enabled": True,
        "noise_experiment_enabled": False,
        "identity_refuter_enabled": False,
    },
    "M1": {
        "formal_response_enabled": False,
        "noise_experiment_enabled": False,
        "identity_refuter_enabled": False,
    },
    "M0_NOISE": {
        "formal_response_enabled": True,
        "noise_experiment_enabled": True,
        "identity_refuter_enabled": False,
    },
    "M2": {
        "formal_response_enabled": True,
        "noise_experiment_enabled": False,
        "identity_refuter_enabled": True,
    },
}

MODEL_PROFILE_ALIASES = {
    "M0+NOISE": "M0_NOISE",
}


def normalize_config_id(config_id: str) -> str:
    return config_id[:-5] if config_id.endswith(".json") else config_id


def normalize_model_profile(model_profile: str | None) -> str:
    if model_profile is None:
        return "M0"

    candidate = str(model_profile).strip().upper()
    candidate = MODEL_PROFILE_ALIASES.get(candidate, candidate)
    if candidate not in MODEL_PROFILES:
        allowed = ", ".join(sorted(MODEL_PROFILES.keys()))
        raise ValueError(
            f"Unknown model profile '{model_profile}'. Allowed: {allowed}"
        )
    return candidate


def apply_model_profile_defaults(cfg: dict, model_profile: str | None) -> dict:
    cfg = dict(cfg)
    normalized_profile = normalize_model_profile(model_profile)
    cfg.update(MODEL_PROFILES[normalized_profile])
    cfg["model_profile"] = normalized_profile
    return cfg


def add_optional_bool_arg(parser: argparse.ArgumentParser, name: str, help_text: str):
    group = parser.add_mutually_exclusive_group()
    dest = name.replace("-", "_")
    group.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
    group.add_argument(
        f"--no-{name}",
        dest=dest,
        action="store_false",
        help=f"Disable {help_text.lower()}",
    )
    parser.set_defaults(**{dest: None})


def apply_env_mode_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    cfg = dict(cfg)
    overrides = {
        "formal_response_enabled": args.formal_response_enabled,
        "noise_experiment_enabled": args.noise_experiment_enabled,
        "strict_proof_check": args.strict_proof_check,
    }
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value

    if args.formal_response_policy is not None:
        cfg["formal_response_policy"] = args.formal_response_policy
    if args.formal_response_reward_mode is not None:
        cfg["formal_response_reward_mode"] = args.formal_response_reward_mode
    if args.formal_response_noise_level is not None:
        cfg["formal_response_noise_level"] = args.formal_response_noise_level
    if args.formal_response_proof_threshold is not None:
        cfg["formal_response_proof_threshold"] = args.formal_response_proof_threshold
    return cfg
