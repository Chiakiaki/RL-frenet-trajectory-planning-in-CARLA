"""SB3 callback helpers for the BDP runner."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback


def callback_freq_from_timesteps(freq: int, n_envs: int) -> int:
    """
    Convert a desired total-timestep frequency into SB3 callback calls.

    SB3 callbacks are called once per vectorized env step.  With ``n_envs=4``,
    one callback call advances ``model.num_timesteps`` by 4.  Using ``ceil``
    keeps the actual save/eval point at or just after the requested total
    timestep count.
    """

    return max(int(math.ceil(int(freq) / max(int(n_envs), 1))), 1)


def make_training_callback(
    args: argparse.Namespace,
    model_dir: Path,
    eval_env: Optional[object] = None,
) -> Optional[BaseCallback]:
    """
    Build off-the-shelf SB3 callbacks for periodic and best-model saving.

    ``CheckpointCallback`` saves periodic ``step_*`` models.  ``EvalCallback``
    evaluates less frequently and saves only one ``best_model.zip`` when the
    evaluation mean reward improves.  This avoids the old per-step best-save
    behavior that produced many ``best_<timestep>.zip`` files.
    """

    callbacks = []
    n_envs = int(getattr(args, "n_envs", 1))

    if int(args.save_freq) > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=callback_freq_from_timesteps(args.save_freq, n_envs),
                save_path=str(model_dir),
                name_prefix="step",
            )
        )

    if int(args.eval_freq) > 0 and eval_env is not None:
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(model_dir),
                log_path=str(model_dir / "eval"),
                eval_freq=callback_freq_from_timesteps(args.eval_freq, n_envs),
                n_eval_episodes=args.n_eval_episodes,
                deterministic=True,
                render=False,
            )
        )

    if not callbacks:
        return None
    if len(callbacks) == 1:
        return callbacks[0]
    return CallbackList(callbacks)
