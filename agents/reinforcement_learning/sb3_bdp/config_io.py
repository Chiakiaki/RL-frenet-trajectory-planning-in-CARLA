"""YAML config loading and reproduction saving for the SB3 BDP runner."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Dict, Optional


def resolve_local_path(path: str, base_path: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = base_path / candidate
    return candidate


def load_yaml_config(config_file: Path) -> Dict[str, Any]:
    import yaml

    with config_file.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"SB3 config must be a YAML mapping: {config_file}")
    return data


def flatten_config_dict(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert the readable nested YAML config into argparse-style keys.

    The runner still uses argparse internally, so the YAML leaves are named
    after CLI arguments.  A few aliases make the file more explicit, especially
    the legacy CARLA config path, which is stored but only used for CARLA runs.
    """

    flattened: Dict[str, Any] = {}
    for section, value in config.items():
        if isinstance(value, dict):
            for key, nested_value in value.items():
                if section == "legacy_carla" and key == "cfg_file":
                    flattened["legacy_carla_cfg_file"] = nested_value
                elif section == "algorithm" and key == "name":
                    flattened["sb3_algorithm"] = nested_value
                else:
                    flattened[key] = nested_value
        else:
            flattened[section] = value

    if "carla_cfg_file" in flattened and "legacy_carla_cfg_file" not in flattened:
        flattened["legacy_carla_cfg_file"] = flattened["carla_cfg_file"]
    return flattened


def find_auto_test_config(
    config_file: Optional[str],
    test: bool,
    log_path: Optional[str],
    base_path: Path,
) -> Optional[Path]:
    if config_file is not None:
        return resolve_local_path(config_file, base_path)
    if not test or log_path is None:
        return None

    candidate = resolve_local_path(log_path, base_path) / "config.yaml"
    return candidate if candidate.exists() else None


def parser_destinations(parser: argparse.ArgumentParser) -> set[str]:
    return {
        action.dest
        for action in parser._actions
        if action.dest is not argparse.SUPPRESS
    }


def load_config_defaults(parser: argparse.ArgumentParser, config_file: Optional[Path]) -> None:
    if config_file is None:
        return
    if not config_file.exists():
        raise FileNotFoundError(f"SB3 config file does not exist: {config_file}")

    flattened = flatten_config_dict(load_yaml_config(config_file))
    valid_destinations = parser_destinations(parser)
    defaults = {key: value for key, value in flattened.items() if key in valid_destinations}
    defaults["config_file"] = str(config_file)
    parser.set_defaults(**defaults)


def build_resolved_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "environment": {
            "env_source": args.env_source,
            "env": args.env,
            "gym_make_kwargs": args.gym_make_kwargs,
            "max_candidates": args.max_candidates,
        },
        "legacy_carla": {
            "cfg_file": args.legacy_carla_cfg_file,
            "carla_host": args.carla_host,
            "carla_port": args.carla_port,
            "tm_port": args.tm_port,
            "carla_res": args.carla_res,
            "planner_mode": args.planner_mode,
            "is_finish_traj": args.is_finish_traj,
            "use_lidar": args.use_lidar,
            "num_traj": args.num_traj,
            "scale_yaw": args.scale_yaw,
            "scale_v": args.scale_v,
            "bdp_debug": args.bdp_debug,
            "short_hard": args.short_hard,
            "env_change": args.env_change,
        },
        "algorithm": {
            "name": args.sb3_algorithm,
            "learning_rate": args.learning_rate,
            "num_timesteps": args.num_timesteps,
            "trpo_timesteps_per_batch": args.trpo_timesteps_per_batch,
            "batch_size": args.batch_size,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "target_kl": args.target_kl,
            "cg_max_steps": args.cg_max_steps,
            "cg_damping": args.cg_damping,
            "n_critic_updates": args.n_critic_updates,
            "sub_sampling_factor": args.sub_sampling_factor,
            "ppo_epochs": args.ppo_epochs,
            "ppo_clip_range": args.ppo_clip_range,
            "seed": args.seed,
            "device": args.device,
            "n_envs": args.n_envs,
            "vec_env": args.vec_env,
        },
        "model_architecture": {
            "policy_class": "BDPBoltzmannPolicy",
            "policy_layers": list(args.policy_layers),
            "value_layers": list(args.value_layers),
            "activation": args.activation,
            "candidate_sampler": "DiscreteOneHotExternalSampler",
        },
        "logging": {
            "log_path": args.log_path,
            "log_interval": args.log_interval,
            "save_freq": args.save_freq,
            "eval_freq": args.eval_freq,
            "n_eval_episodes": args.n_eval_episodes,
        },
        "render": {
            "play_mode": args.play_mode,
            "render_mode": args.render_mode,
            "render_train": args.render_train,
            "render_freq": args.render_freq,
        },
        "testing": {
            "num_test_episode": args.num_test_episode,
            "test_model": args.test_model,
            "test_last": args.test_last,
        },
    }


def write_resolved_config(args: argparse.Namespace, log_dir: Path, base_path: Path) -> None:
    import yaml

    if args.config_file is not None:
        source_config = resolve_local_path(args.config_file, base_path)
        target_config = log_dir / "config.yaml"
        if source_config.exists() and source_config.resolve() != target_config.resolve():
            shutil.copyfile(source_config, target_config)
        elif not target_config.exists():
            with target_config.open("w") as f:
                yaml.safe_dump(build_resolved_config(args), f, sort_keys=False)
    else:
        with (log_dir / "config.yaml").open("w") as f:
            yaml.safe_dump(build_resolved_config(args), f, sort_keys=False)

    with (log_dir / "resolved_config.yaml").open("w") as f:
        yaml.safe_dump(build_resolved_config(args), f, sort_keys=False)
