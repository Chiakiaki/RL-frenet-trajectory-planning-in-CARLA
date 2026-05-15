"""SB3 algorithm/model factory helpers for the BDP runner."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Type

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import NatureCNN
from torch import nn

from .policies import BDPBoltzmannPolicy


def activation_from_name(name: str) -> Type[nn.Module]:
    return {"tanh": nn.Tanh, "relu": nn.ReLU}[name]


def get_algorithm_class(name: str) -> Type[Any]:
    if name == "TRPO":
        try:
            from sb3_contrib import TRPO
        except ImportError as exc:
            raise ImportError(
                "TRPO for Stable-Baselines3 is provided by sb3-contrib. "
                "Install it with `pip install sb3-contrib`, or run with "
                "`--sb3_algorithm PPO` for a PPO smoke test."
            ) from exc
        return TRPO
    if name == "PPO":
        return PPO
    raise ValueError(f"Unsupported SB3 algorithm: {name}")


def policy_net_arch(args: argparse.Namespace) -> Any:
    if args.policy_layers is None and args.value_layers is None:
        return None
    return dict(pi=list(args.policy_layers or []), vf=list(args.value_layers or []))


def policy_kwargs_for(args: argparse.Namespace, policy_mode: str) -> dict[str, Any]:
    kwargs = dict(
        net_arch=policy_net_arch(args),
        activation_fn=activation_from_name(args.activation),
        normalize_images=args.builtin_policy == "CnnPolicy",
    )
    if args.builtin_policy == "CnnPolicy":
        kwargs["features_extractor_class"] = NatureCNN
    return kwargs


def make_model(args: argparse.Namespace, env: Any, model_dir: Path) -> Any:
    algorithm_class = get_algorithm_class(args.sb3_algorithm)
    policy_mode = getattr(args, "policy_mode", "bdp")
    if policy_mode == "builtin":
        policy = args.builtin_policy
    else:
        if args.builtin_policy not in ("MlpPolicy", "CnnPolicy"):
            raise ValueError(
                "BDP mode currently supports builtin_policy=MlpPolicy or CnnPolicy. "
                f"Got {args.builtin_policy}."
            )
        policy = BDPBoltzmannPolicy
    policy_kwargs = policy_kwargs_for(args, policy_mode)

    # n_steps is the rollout collection length for the inner data-collection loop of on-policy algorithms like PPO/TRPO
    # in sb3,rollout batch size should be n_steps * n_envs
    # n_steps is often much smaller for A2C. SB3’s default A2C value is commonly n_steps=5, while PPO often uses larger values like 2048.
    common_kwargs = dict(
        policy=policy,
        env=env,
        learning_rate=args.learning_rate,
        n_steps=args.trpo_timesteps_per_batch,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        tensorboard_log=str(model_dir),
        verbose=1,
        seed=args.seed,
        device=args.device,
        policy_kwargs=policy_kwargs,
    )

    if args.sb3_algorithm == "TRPO":
        if policy_mode == "bdp" and args.sub_sampling_factor != 1:
            raise ValueError(
                "SB3-contrib TRPO sub_sampling_factor > 1 is not supported by this "
                "Dict-observation candidate wrapper. Use the default value of 1."
            )
        common_kwargs.update(
            dict(
                cg_max_steps=args.cg_max_steps,
                cg_damping=args.cg_damping,
                n_critic_updates=args.n_critic_updates,
                target_kl=args.target_kl,
                sub_sampling_factor=args.sub_sampling_factor,
            )
        )
    else:
        common_kwargs.update(
            dict(
                n_epochs=args.ppo_epochs,
                clip_range=args.ppo_clip_range,
                target_kl=args.target_kl,
            )
        )

    return algorithm_class(**common_kwargs)
