#!/usr/bin/env python3
"""
Standalone Stable-Baselines3/PyTorch runner for the BDP/TRPO candidate-action
planner.

This file intentionally does not modify the original TensorFlow Stable
Baselines implementation.  The migration strategy is:

1. Keep the existing CARLA/Frenet environment and its ``external_sampler()``.
2. Wrap it as a Gymnasium environment for SB3.
3. Put the sampled candidate trajectories into the observation.
4. Train a discrete SB3 policy whose logits are produced by a PyTorch
   state-action goodness network, matching the old Boltzmann candidate scoring
   idea from ``BDPL``/``TRPO_bdp``.

The implementation is split under ``agents/reinforcement_learning/sb3_bdp``:

    callbacks.py  checkpoint callback
    envs.py       CARLA/generic Gymnasium candidate-action wrappers
    model.py      SB3 TRPO/PPO factory
    policies.py   PyTorch BDP Boltzmann policy
    samplers.py   external candidate samplers
    spaces.py     Gym/Gymnasium space conversion helpers

Important difference from the TensorFlow implementation:

The old code flattens all state-action pairs in a rollout into one long list:

    (s0, a0), (s0, a1), (s1, a0), (s2, a0), (s2, a1), ...

Then it needs ``grouping_mn`` to say which pairs belong to the same original
state, so TensorFlow can apply softmax inside each candidate set.

In SB3 we keep every candidate set attached to its own observation:

    obs["candidates"]     -> shape (max_candidates, candidate_dim)
    obs["candidate_mask"] -> shape (max_candidates,)

After batching, PyTorch sees logits with shape (batch, max_candidates), so the
Categorical distribution naturally applies one softmax per observation row.
Invalid padded slots are masked out.  That is why there is no explicit grouping
matrix in this migration.

TRPO itself is provided by ``sb3-contrib`` in the SB3 ecosystem:

    pip install sb3-contrib

CartPole smoke-test launch shape:

python3 ./run_BDPL_sb3.py \
  --env_source=gymnasium \
  --env=CartPole-v1 \
  --sb3_algorithm=PPO \
  --num_timesteps=20000
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Tuple

from agents.reinforcement_learning.sb3_bdp.callbacks import make_training_callback
from agents.reinforcement_learning.sb3_bdp.config_io import (
    find_auto_test_config,
    load_config_defaults,
    write_resolved_config,
)
from agents.reinforcement_learning.sb3_bdp.envs import make_sb3_env
from agents.reinforcement_learning.sb3_bdp.model import get_algorithm_class, make_model

CURRENT_PATH = Path(__file__).resolve().parent
cfg = None


def make_run_timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def parse_args_cfgs() -> Tuple[argparse.Namespace, Any]:
    global cfg
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config_file", type=str, default=None)
    pre_parser.add_argument("--test", default=False, action="store_true")
    pre_parser.add_argument("--log_path", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    config_file = find_auto_test_config(
        pre_args.config_file,
        pre_args.test,
        pre_args.log_path,
        CURRENT_PATH,
    )

    parser = argparse.ArgumentParser(
        description="Train/test the BDP candidate planner with Stable-Baselines3."
    )
    parser.add_argument("--config_file", type=str, default=None, help="SB3 runner YAML config.")
    parser.add_argument("--cfg_file", type=str, default=None, help="Config YAML used by the CARLA env.")
    parser.add_argument(
        "--legacy_carla_cfg_file",
        type=str,
        default=None,
        help="Stored CARLA config path used only when --env_source=carla and --cfg_file is omitted.",
    )
    parser.add_argument("--env", type=str, default="CarlaGymEnv-v5", help="Registered CARLA Gym env id.")
    parser.add_argument(
        "--env_source",
        choices=("carla", "gymnasium", "gym"),
        default="carla",
        help="Use the CARLA planner env, a Gymnasium env, or a legacy Gym env.",
    )
    parser.add_argument(
        "--gym_make_kwargs",
        type=json.loads,
        default={},
        help='Extra kwargs for gym.make(), as JSON. Prefer YAML environment.gym_make_kwargs.',
    )
    parser.add_argument("--agent_id", type=int, default=None)
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--num_timesteps", type=float, default=1e7)
    parser.add_argument("--num_test_episode", type=int, default=2200)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--play_mode", type=int, default=0)
    parser.add_argument(
        "--render_mode",
        choices=("human", "rgb_array"),
        default=None,
        help="Gymnasium render mode for generic envs. --play_mode sets this to human when omitted.",
    )
    parser.add_argument(
        "--render_train",
        default=False,
        action="store_true",
        help="Call env.render() during training. Intended for generic envs with --n_envs=1.",
    )
    parser.add_argument("--render_freq", type=int, default=1, help="Render every N callback calls during training.")
    parser.add_argument("--verbosity", type=int, default=0)
    parser.add_argument("--test", default=False, action="store_true")
    parser.add_argument("--env_change", type=str, default="None")
    parser.add_argument("--test_model", type=str, default="")
    parser.add_argument("--test_last", default=False, action="store_true")

    parser.add_argument("--carla_host", metavar="H", default="127.0.0.1")
    parser.add_argument("-p", "--carla_port", metavar="P", default=2000, type=int)
    parser.add_argument("--tm_port", default=8000, type=int)
    parser.add_argument("--carla_res", metavar="WIDTHxHEIGHT", default="1280x720")

    parser.add_argument("--learning_rate", type=float, default=7e-4)
    parser.add_argument("--planner_mode", type=str, default="bdp")
    parser.add_argument("--is_finish_traj", type=int, default=1)
    parser.add_argument("--use_lidar", type=int, default=0)
    parser.add_argument("--num_traj", type=int, default=3)
    parser.add_argument("--scale_yaw", type=float, default=40.0)
    parser.add_argument("--scale_v", type=float, default=0.01)
    parser.add_argument("--bdp_debug", type=int, default=0)
    parser.add_argument("--trpo_timesteps_per_batch", type=int, default=1024)
    parser.add_argument("--short_hard", type=int, default=0)

    parser.add_argument(
        "--sb3_algorithm",
        choices=("TRPO", "PPO"),
        default="TRPO",
        help="TRPO requires sb3-contrib. PPO is available for smoke tests/comparisons.",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.98)
    parser.add_argument("--target_kl", type=float, default=0.01)
    parser.add_argument("--cg_max_steps", type=int, default=10)
    parser.add_argument("--cg_damping", type=float, default=1e-2)
    parser.add_argument("--n_critic_updates", type=int, default=3)
    parser.add_argument("--sub_sampling_factor", type=int, default=1)
    parser.add_argument("--ppo_epochs", type=int, default=10)
    parser.add_argument("--ppo_clip_range", type=float, default=0.2)
    parser.add_argument("--save_freq", type=int, default=5000)
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=10000,
        help="Evaluate and save SB3 best_model.zip every N total timesteps. Use 0 to disable.",
    )
    parser.add_argument("--n_eval_episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--n_envs",
        type=int,
        default=1,
        help="Number of parallel training environments for generic Gym/Gymnasium envs.",
    )
    parser.add_argument(
        "--vec_env",
        choices=("dummy", "subproc"),
        default="dummy",
        help="Vector-env backend when --n_envs > 1. Use subproc for process-level parallelism.",
    )
    parser.add_argument("--max_candidates", type=int, default=None)
    parser.add_argument(
        "--policy_mode",
        choices=("bdp", "builtin"),
        default="bdp",
        help=(
            "bdp uses the candidate-action Dict observation and BDPBoltzmannPolicy. "
            "builtin uses the raw Gym/Gymnasium env and an ordinary SB3 policy."
        ),
    )
    parser.add_argument(
        "--builtin_policy",
        type=str,
        default="MlpPolicy",
        help="SB3 policy name/class used only when --policy_mode=builtin, e.g. MlpPolicy or CnnPolicy.",
    )
    parser.add_argument("--policy_layers", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--value_layers", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--activation", choices=("tanh", "relu"), default="tanh")

    load_config_defaults(parser, config_file) # config file, override previous defaults 
    args = parser.parse_args() # command line
    args.num_timesteps = int(args.num_timesteps)

    if (args.play_mode or args.render_train) and args.render_mode is None and args.env_source != "carla":
        args.render_mode = "human"
    if args.render_train and args.env_source != "carla" and args.n_envs != 1:
        raise ValueError("--render_train for generic envs requires --n_envs=1")

    if args.legacy_carla_cfg_file is None and args.cfg_file is not None:
        args.legacy_carla_cfg_file = args.cfg_file
    if args.env_source == "carla" and args.cfg_file is None:
        args.cfg_file = args.legacy_carla_cfg_file
    if args.env_source != "carla":
        args.cfg_file = None

    if args.test and args.cfg_file is None and args.env_source == "carla":
        if args.agent_id is not None:
            log_dir = CURRENT_PATH / "logs" / f"agent_{args.agent_id}"
        elif args.test_dir is not None:
            log_dir = CURRENT_PATH / "logs" / args.test_dir
        else:
            raise ValueError("--test without --cfg_file needs --agent_id or --test_dir")

        candidates = sorted(log_dir.glob("*.yaml"))
        if not candidates:
            raise FileNotFoundError(f"No YAML config found in {log_dir}")
        args.cfg_file = str(candidates[0])

    if args.cfg_file is None and args.env_source == "carla":
        raise ValueError("--cfg_file is required for training")

    if args.env_source == "carla" and args.cfg_file is not None:
        from config import cfg as loaded_cfg
        from config import cfg_from_yaml_file

        cfg = loaded_cfg
        cfg_from_yaml_file(args.cfg_file, cfg)
        cfg.TAG = Path(args.cfg_file).stem
        cfg.EXP_GROUP_PATH = "/".join(args.cfg_file.split("/")[1:-1])

    return args, cfg


def prepare_log_dirs(args: argparse.Namespace) -> Tuple[Path, Path]:
    if args.agent_id is not None:
        log_dir = CURRENT_PATH / "logs" / f"agent_{args.agent_id}"
    elif args.test and args.test_dir is not None:
        log_dir = CURRENT_PATH / "logs" / args.test_dir
    elif args.log_path is not None:
        log_dir = Path(args.log_path)
        if not log_dir.is_absolute():
            log_dir = CURRENT_PATH / log_dir
        if not args.test:
            log_dir = log_dir / make_run_timestamp()
    else:
        timestamp = make_run_timestamp()
        log_dir = CURRENT_PATH / "logs" / f"sb3_{timestamp}"

    model_dir = log_dir / "models"

    if args.test:
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    else:
        if log_dir.exists() and args.agent_id is not None:
            raise FileExistsError(f"Refusing to overwrite existing log directory: {log_dir}")
        model_dir.mkdir(parents=True, exist_ok=False)
        copy_reproduction_info(args, log_dir)

    return log_dir, model_dir


def copy_reproduction_info(args: argparse.Namespace, log_dir: Path) -> None:
    try:
        import git

        repo = git.Repo(search_parent_directories=False)
        commit_id = repo.head.object.hexsha
    except Exception as exc:
        commit_id = f"unavailable ({exc})"

    with (log_dir / "reproduction_info.txt").open("w") as f:
        f.write(f"Git commit id: {commit_id}\n\n")
        f.write(f"Program arguments:\n\n{args}\n\n")
        if cfg is not None:
            f.write(f"Configuration file:\n\n{cfg}\n")
        else:
            f.write("Configuration file:\n\nNone; generic Gym/Gymnasium environment.\n")
        f.write("\nMigration runner: run_BDPL_sb3.py\n")
        f.write("\nResolved SB3 runner config: config.yaml\n")
        f.write("Effective SB3 runner config after CLI overrides: resolved_config.yaml\n")

    write_resolved_config(args, log_dir, CURRENT_PATH)

    if args.env_source == "carla" and args.cfg_file is not None:
        shutil.copyfile(args.cfg_file, log_dir / Path(args.cfg_file).name)


def parse_step_from_name(path: Path) -> int:
    digits = "".join(ch if ch.isdigit() else " " for ch in path.stem).split()
    return int(digits[-1]) if digits else -1


def resolve_model_path(args: argparse.Namespace, model_dir: Path) -> Path:
    if args.test_model:
        path = Path(args.test_model)
        if not path.is_absolute():
            path = model_dir / path
        if path.suffix != ".zip" and path.with_suffix(".zip").exists():
            path = path.with_suffix(".zip")
        return path

    if not args.test_last:
        best_model = model_dir / "best_model.zip"
        if best_model.exists():
            return best_model

    pattern = "step_*_steps.zip" if args.test_last else "best_*.zip"
    candidates = sorted(model_dir.glob(pattern), key=parse_step_from_name)
    if args.test_last and not candidates:
        candidates = sorted(model_dir.glob("step_*.zip"), key=parse_step_from_name)
    if not candidates:
        candidates = sorted(model_dir.glob("*final_model.zip"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No SB3 model found in {model_dir}")
    return candidates[-1]


def run_training(args: argparse.Namespace) -> None:
    log_dir, model_dir = prepare_log_dirs(args)
    env = make_sb3_env(args, log_dir, is_train=True)
    model = make_model(args, env, model_dir)
    eval_env = None
    if args.eval_freq > 0:
        if args.env_source == "carla":
            print("SB3 EvalCallback best-model saving is skipped for CARLA training.")
        else:
            eval_env = make_sb3_env(args, log_dir, is_train=False)
    callback = make_training_callback(args, model_dir, eval_env=eval_env)

    model_tag = "BDP" if args.policy_mode == "bdp" else "builtin"
    final_model = model_dir / f"{args.sb3_algorithm}_{model_tag}_final_model"
    try:
        print("Model is created")
        print("Training started")
        model.learn(
            total_timesteps=args.num_timesteps,
            callback=callback,
            log_interval=args.log_interval,
            tb_log_name=f"{args.sb3_algorithm}_{model_tag}",
        )
    finally:
        print("*" * 100)
        print("FINISHED TRAINING; saving model...")
        print("*" * 100)
        model.save(str(final_model))
        env.close()
        if eval_env is not None:
            eval_env.close()
        print(f"Model has been saved to {final_model}.zip")


def run_test(args: argparse.Namespace) -> None:
    log_dir, model_dir = prepare_log_dirs(args)
    env = make_sb3_env(args, log_dir, is_train=False)
    model_path = resolve_model_path(args, model_dir)
    algorithm_class = get_algorithm_class(args.sb3_algorithm)

    print(f"{model_path.name} is loading...")
    model = algorithm_class.load(str(model_path), env=env, device=args.device)
    model_env = model.get_env()
    if model_env is None:
        raise RuntimeError("Loaded SB3 model does not have an environment")
    print("Model is loaded")

    try:
        obs = model_env.reset()
        episode_count = 0
        while episode_count < args.num_test_episode:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = model_env.step(action)
            if args.play_mode != 0:
                model_env.render()
            for done in dones:
                if done:
                    episode_count += 1
                    print(f"{episode_count}/{args.num_test_episode} finished")
                    if episode_count >= args.num_test_episode:
                        break
    finally:
        model_env.close()


if __name__ == "__main__":
    parsed_args, _ = parse_args_cfgs()
    print("Env is starting")
    if parsed_args.test:
        run_test(parsed_args)
    else:
        run_training(parsed_args)
