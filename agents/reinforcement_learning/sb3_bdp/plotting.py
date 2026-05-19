"""Reward-curve plotting utilities for SB3 BDP experiment folders."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class RunLog:
    """One timestamped training run selected for plotting."""

    path: Path
    label: str


@dataclass(frozen=True)
class RewardCurve:
    """Smoothed reward curve for one timestamped training run."""

    label: str
    timesteps: np.ndarray
    rewards: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def sanitize_filename_component(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)
    return cleaned.strip("_") or "all"


def resolve_logdir(logdir: str | Path, base_path: Path | None = None) -> Path:
    """Resolve a user log prefix.

    A leading ``/logs/...`` path is treated as repo-relative shorthand, because
    that is how the training configs often describe log locations.
    """

    base_path = Path.cwd() if base_path is None else base_path
    path = Path(logdir).expanduser()
    if path.is_absolute():
        if len(path.parts) >= 2 and path.parts[1] == "logs":
            return base_path / Path(*path.parts[1:])
        return path
    return base_path / path


def path_matches_filters(path: Path, filters: Sequence[str] | None) -> bool:
    if not filters:
        return True
    path_text = str(path).lower()
    return all(item.lower() in path_text for item in filters if item)


def find_training_monitor_files(run_dir: Path) -> list[Path]:
    """Return train monitor files below one run, excluding test monitors."""

    monitors = []
    for path in sorted(run_dir.rglob("monitor.csv")):
        if "test" in path.relative_to(run_dir).parts:
            continue
        monitors.append(path)
    return monitors


def label_for_run(run_dir: Path, prefix: Path) -> str:
    try:
        relative = run_dir.relative_to(prefix)
    except ValueError:
        relative = run_dir

    parts = relative.parts
    if len(parts) >= 2:
        # New layout:
        #   <env>/<policy_mode>_<algorithm>_<policy>_<n_envs>env/<timestamp>
        # Keep the legend compact but still distinguish repeated timestamps.
        return str(Path(*parts[-2:]))
    return str(relative)


def discover_run_logs(prefix: str | Path, filters: Sequence[str] | None = None) -> list[RunLog]:
    """Select timestamped run folders under ``prefix``.

    A run folder is detected as the parent of a training ``monitor.csv`` file.
    Path filters are substring matches applied to the full run path; all filters
    must match.  For example, ``["PPO"]`` selects PPO runs and
    ``["builtin", "PPO"]`` selects built-in PPO runs.
    """

    prefix = Path(prefix)
    run_dirs = set()
    for monitor_path in prefix.rglob("monitor.csv"):
        if "test" in monitor_path.relative_to(prefix).parts:
            continue
        run_dirs.add(monitor_path.parent.parent)

    runs = []
    for run_dir in sorted(run_dirs):
        if not path_matches_filters(run_dir, filters):
            continue
        if not find_training_monitor_files(run_dir):
            continue
        runs.append(RunLog(path=run_dir, label=label_for_run(run_dir, prefix)))
    return runs


def read_monitor_rows(monitor_path: Path) -> list[tuple[float, int, float]]:
    rows = []
    with monitor_path.open("r", newline="") as f:
        first_line = f.readline()
        if not first_line.startswith("#"):
            f.seek(0)
        reader = csv.DictReader(f)
        for row in reader:
            try:
                reward = float(row["r"])
                length = int(float(row["l"]))
                time_value = float(row.get("t", 0.0))
            except (KeyError, TypeError, ValueError):
                continue
            rows.append((reward, length, time_value))
    return rows


def rolling_mean_std(values: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    window_size = max(1, int(window_size))
    means = np.zeros_like(values, dtype=np.float64)
    stds = np.zeros_like(values, dtype=np.float64)
    for index in range(values.shape[0]):
        start = max(0, index - window_size + 1)
        window = values[start : index + 1]
        means[index] = np.mean(window)
        stds[index] = np.std(window)
    return means, stds


def load_run_curve(
    run: RunLog,
    window_size: int = 20,
    max_timesteps: float | None = None,
) -> RewardCurve:
    """Load all training monitors for one run and build a smoothed curve."""

    rows = []
    for monitor_path in find_training_monitor_files(run.path):
        rows.extend(read_monitor_rows(monitor_path))
    if not rows:
        raise ValueError(f"No training monitor rows found in {run.path}")

    # Interleave vector-env episode completions by monitor time, then accumulate
    # episode lengths to approximate total SB3 env transitions.
    rows.sort(key=lambda item: item[2])
    rewards = np.asarray([item[0] for item in rows], dtype=np.float64)
    lengths = np.asarray([item[1] for item in rows], dtype=np.int64)
    timesteps = np.cumsum(lengths)

    if max_timesteps is not None:
        keep = timesteps <= max_timesteps
        if np.any(keep):
            rewards = rewards[keep]
            timesteps = timesteps[keep]

    mean, std = rolling_mean_std(rewards, window_size)
    return RewardCurve(
        label=run.label,
        timesteps=timesteps,
        rewards=rewards,
        mean=mean,
        std=std,
    )


def default_output_path(logdir: str | Path, filters: Sequence[str] | None = None) -> Path:
    suffix = "all" if not filters else "_".join(sanitize_filename_component(item) for item in filters)
    return Path(logdir) / f"reward_comparison_{suffix}.png"


def plot_reward_curves(
    curves: Iterable[RewardCurve],
    output_path: str | Path,
    title: str | None = None,
    alpha: float = 0.2,
    show: bool = False,
) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    curves = list(curves)
    if not curves:
        raise ValueError("No reward curves to plot.")

    fig, ax = plt.subplots(figsize=(12, 7))
    for curve in curves:
        line = ax.plot(curve.timesteps, curve.mean, linewidth=2.0, label=curve.label)[0]
        color = line.get_color()
        ax.fill_between(
            curve.timesteps,
            curve.mean - curve.std,
            curve.mean + curve.std,
            color=color,
            alpha=alpha,
            linewidth=0,
        )

    ax.set_title(title or "SB3 BDP Reward Comparison")
    ax.set_xlabel("Total Environment Timesteps")
    ax.set_ylabel("Episode Reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot SB3 BDP reward curves from monitor.csv logs.")
    parser.add_argument(
        "--logdir",
        "--log_dir",
        required=True,
        help="Log prefix to search, e.g. logs/sb3, /logs/sb3, or logs/sb3/CarRacing-v3.",
    )
    parser.add_argument(
        "--filter",
        dest="filters",
        nargs="*",
        default=[],
        help="Path substring filters. All filters must match, e.g. --filter PPO CnnPolicy.",
    )
    parser.add_argument("--window_size", type=int, default=20, help="Rolling reward window in episodes.")
    parser.add_argument("--alpha", type=float, default=0.2, help="Std-band opacity.")
    parser.add_argument("--n_steps", type=float, default=None, help="Optional max total env timesteps to plot.")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path. Default is inside --logdir.")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--show", action="store_true", help="Also show the matplotlib window.")
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logdir = resolve_logdir(args.logdir)
    runs = discover_run_logs(logdir, filters=args.filters)
    if not runs:
        raise SystemExit(f"No training monitor logs found under {logdir} with filters {args.filters}")

    curves = [load_run_curve(run, args.window_size, args.n_steps) for run in runs]
    output_path = Path(args.output) if args.output is not None else default_output_path(logdir, args.filters)
    output_path = plot_reward_curves(curves, output_path, args.title, args.alpha, args.show)

    print("Selected runs:")
    for run in runs:
        print(f"  {run.path}")
    print(f"Saved plot: {output_path}")
    return output_path


if __name__ == "__main__":
    main()
