from pathlib import Path

import numpy as np

from agents.reinforcement_learning.sb3_bdp.plotting import (
    RunLog,
    default_output_path,
    discover_run_logs,
    load_run_curve,
    resolve_logdir,
)


def write_monitor_csv(path: Path, rows: list[tuple[float, int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ['#{"t_start": 0.0, "env_id": "None"}', "r,l,t"]
    lines.extend(f"{reward},{length},{time}" for reward, length, time in rows)
    path.write_text("\n".join(lines) + "\n")


def make_run(prefix: Path, relative_run: str) -> Path:
    run_dir = prefix / relative_run
    (run_dir / "config.yaml").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text("environment: {}\n")
    write_monitor_csv(run_dir / "env_0" / "monitor.csv", [(1.0, 10, 1.0)])
    return run_dir


def test_discover_run_logs_uses_prefix_and_path_filters(tmp_path: Path) -> None:
    prefix = tmp_path / "logs" / "sb3"
    make_run(prefix, "CarRacing-v3/bdp_PPO_CnnPolicy_24env/20260101_000000")
    make_run(prefix, "CarRacing-v3/bdp_TRPO_CnnPolicy_24env/20260101_000000")
    make_run(prefix, "CarRacing-v3/builtin_PPO_CnnPolicy_24env/20260101_000000")

    all_runs = discover_run_logs(prefix)
    ppo_runs = discover_run_logs(prefix, filters=["PPO"])
    builtin_ppo_runs = discover_run_logs(prefix, filters=["builtin", "PPO"])

    assert len(all_runs) == 3
    assert len(ppo_runs) == 2
    assert len(builtin_ppo_runs) == 1
    assert "builtin_PPO_CnnPolicy_24env" in builtin_ppo_runs[0].label


def test_load_run_curve_merges_env_monitors_and_excludes_test(tmp_path: Path) -> None:
    run_dir = tmp_path / "logs" / "sb3" / "CarRacing-v3" / "bdp_PPO_CnnPolicy_2env" / "20260101_000000"
    write_monitor_csv(run_dir / "env_0" / "monitor.csv", [(1.0, 10, 1.0), (3.0, 30, 3.0)])
    write_monitor_csv(run_dir / "env_1" / "monitor.csv", [(2.0, 20, 2.0), (4.0, 40, 4.0)])
    write_monitor_csv(run_dir / "test" / "monitor.csv", [(999.0, 100, 5.0)])

    curve = load_run_curve(RunLog(path=run_dir, label="demo"), window_size=2)

    np.testing.assert_allclose(curve.timesteps, np.array([10, 30, 60, 100]))
    np.testing.assert_allclose(curve.rewards, np.array([1.0, 2.0, 3.0, 4.0]))
    np.testing.assert_allclose(curve.mean, np.array([1.0, 1.5, 2.5, 3.5]))
    np.testing.assert_allclose(curve.std, np.array([0.0, 0.5, 0.5, 0.5]))


def test_default_output_path_uses_filter_name(tmp_path: Path) -> None:
    output = default_output_path(tmp_path / "logs" / "sb3", filters=["PPO", "CnnPolicy"])

    assert output == tmp_path / "logs" / "sb3" / "reward_comparison_PPO_CnnPolicy.png"


def test_resolve_logdir_accepts_repo_style_absolute_shorthand(tmp_path: Path) -> None:
    base = tmp_path / "repo"
    (base / "logs" / "sb3").mkdir(parents=True)

    assert resolve_logdir("/logs/sb3", base) == base / "logs" / "sb3"
