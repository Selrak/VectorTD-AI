from __future__ import annotations

import argparse
import os
from pathlib import Path
import statistics
import time
from typing import Any

import gymnasium as gym

try:
    import torch
    import sb3_contrib
    import stable_baselines3 as sb3
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper
except Exception as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "SB3 2.7.x, sb3-contrib 2.7.x, and torch are required for MaskablePPO benchmarks."
    ) from exc

from vectortd.ai.config_loader import apply_overrides, deep_merge, load_json_config
from vectortd.ai.env import VectorTDEventEnv
from vectortd.ai.rewards import RewardConfig


def _resolve_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parents[3]


def _next_log_path(run_dir: Path, *, prefix: str) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    max_idx = 0
    for path in run_dir.glob(f"{prefix}_*.txt"):
        stem = path.stem
        if not stem.startswith(f"{prefix}_"):
            continue
        suffix = stem[len(prefix) + 1 :]
        if suffix.isdigit():
            max_idx = max(max_idx, int(suffix))
    return run_dir / f"{prefix}_{max_idx + 1}.txt"


def _latest_log_path(run_dir: Path, *, prefix: str) -> Path | None:
    max_idx = 0
    latest = None
    for path in run_dir.glob(f"{prefix}_*.txt"):
        stem = path.stem
        if not stem.startswith(f"{prefix}_"):
            continue
        suffix = stem[len(prefix) + 1 :]
        if suffix.isdigit():
            idx = int(suffix)
            if idx > max_idx:
                max_idx = idx
                latest = path
    return latest


def _normalize_report(report: str) -> str:
    lines = []
    for line in report.splitlines():
        if line.startswith("timestamp="):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _format_timing_block(title: str, timing: dict[str, float]) -> list[str]:
    lines = [f"{title} timing:"]
    for key in sorted(timing.keys()):
        lines.append(f"  {key}: {timing[key]:.6f}")
    return lines


def _aggregate_timings(timings: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    total: dict[str, dict[str, float]] = {"env": {}, "engine": {}}
    for entry in timings:
        for scope in ("env", "engine"):
            scope_data = entry.get(scope, {}) or {}
            for key, value in scope_data.items():
                total[scope][key] = total[scope].get(key, 0.0) + float(value)
    return total


def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


class TimingBridge(gym.Wrapper):
    def get_timing_snapshot(self) -> dict[str, Any]:
        if hasattr(self.env, "get_timing_snapshot"):
            return self.env.get_timing_snapshot()
        base = getattr(self.env, "unwrapped", None)
        if base is not None and hasattr(base, "get_timing_snapshot"):
            return base.get_timing_snapshot()
        return {}


class TimedVecEnv(VecEnvWrapper):
    def __init__(self, venv):
        super().__init__(venv)
        self.step_wait_time_total = 0.0
        self.step_wait_calls = 0
        self.reset_time_total = 0.0
        self.reset_calls = 0

    def step_wait(self):
        start = time.perf_counter()
        obs, rewards, dones, infos = self.venv.step_wait()
        self.step_wait_time_total += time.perf_counter() - start
        self.step_wait_calls += 1
        return obs, rewards, dones, infos

    def reset(self, *args, **kwargs):
        start = time.perf_counter()
        obs = self.venv.reset(*args, **kwargs)
        self.reset_time_total += time.perf_counter() - start
        self.reset_calls += 1
        return obs


class TimingTracker:
    def __init__(self) -> None:
        self.rollout_times: list[float] = []
        self.train_times: list[float] = []


class RolloutTimingCallback(BaseCallback):
    def __init__(self, tracker: TimingTracker):
        super().__init__(verbose=0)
        self._tracker = tracker
        self._rollout_start: float | None = None

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        self._rollout_start = time.perf_counter()

    def _on_rollout_end(self) -> None:
        if self._rollout_start is None:
            return
        self._tracker.rollout_times.append(time.perf_counter() - self._rollout_start)
        self._rollout_start = None


class TimedMaskablePPO(MaskablePPO):
    def __init__(self, *args, tracker: TimingTracker, **kwargs):
        super().__init__(*args, **kwargs)
        self._timing_tracker = tracker

    def train(self) -> None:
        start = time.perf_counter()
        super().train()
        self._timing_tracker.train_times.append(time.perf_counter() - start)


def _build_report(stats: dict[str, Any], timing: dict[str, Any], vec_timing: dict[str, float]) -> str:
    elapsed = float(stats["elapsed"])
    steps = int(stats["timesteps"])
    steps_per_sec = steps / elapsed if elapsed > 0 else 0.0

    lines = [
        "VectorTD SB3 MaskablePPO throughput benchmark",
        f"timestamp={time.strftime('%Y-%m-%d %H:%M:%S')}",
        (
            "cfg={cfg} map={map} seed={seed} action_space_kind={action_space_kind}".format(
                cfg=stats["cfg_path"],
                map=stats["map"],
                seed=stats["seed"],
                action_space_kind=stats["action_space_kind"],
            )
        ),
        "vec_env={vec_env} num_envs={num_envs} n_steps={n_steps} batch_size={batch_size}".format(
            vec_env=stats["vec_env"],
            num_envs=stats["num_envs"],
            n_steps=stats["n_steps"],
            batch_size=stats["batch_size"],
        ),
        "policy={policy} n_epochs={n_epochs} device={device}".format(
            policy=stats["policy"],
            n_epochs=stats["n_epochs"],
            device=stats["device"],
        ),
        "timesteps={steps} elapsed_sec={elapsed:.6f} steps_per_sec={rate:.2f}".format(
            steps=steps,
            elapsed=elapsed,
            rate=steps_per_sec,
        ),
    ]

    learn_time = float(stats.get("learn_time_total", 0.0))
    rollout_time = float(stats.get("rollout_time_total", 0.0))
    train_time = float(stats.get("train_time_total", 0.0))
    overhead = learn_time - rollout_time - train_time
    if overhead < 0:
        overhead = 0.0

    lines.append("timing_breakdown:")
    lines.append(f"  learn_sec={learn_time:.6f}")
    lines.append(
        "  rollout_sec={sec:.6f} rollout_calls={calls} rollout_avg_sec={avg:.6f}".format(
            sec=rollout_time,
            calls=int(stats.get("rollout_calls", 0)),
            avg=(rollout_time / stats["rollout_calls"]) if stats.get("rollout_calls") else 0.0,
        )
    )
    lines.append(
        "  train_sec={sec:.6f} train_calls={calls} train_avg_sec={avg:.6f}".format(
            sec=train_time,
            calls=int(stats.get("train_calls", 0)),
            avg=(train_time / stats["train_calls"]) if stats.get("train_calls") else 0.0,
        )
    )
    lines.append(f"  overhead_sec={overhead:.6f}")
    if learn_time > 0:
        lines.append(
            "  rollout_pct={:.1f}% train_pct={:.1f}% overhead_pct={:.1f}%".format(
                (rollout_time / learn_time) * 100.0,
                (train_time / learn_time) * 100.0,
                (overhead / learn_time) * 100.0,
            )
        )

    if vec_timing:
        lines.append("vecenv_timing:")
        step_wait = float(vec_timing.get("step_wait_time_total", 0.0))
        step_calls = float(vec_timing.get("step_wait_calls", 0.0))
        reset_time = float(vec_timing.get("reset_time_total", 0.0))
        reset_calls = float(vec_timing.get("reset_calls", 0.0))
        if step_calls > 0:
            lines.append(f"  step_wait_sec_total={step_wait:.6f} step_wait_avg_sec={step_wait / step_calls:.6f}")
            lines.append(f"  step_wait_calls={int(step_calls)}")
        if reset_calls > 0:
            lines.append(f"  reset_sec_total={reset_time:.6f} reset_avg_sec={reset_time / reset_calls:.6f}")
            lines.append(f"  reset_calls={int(reset_calls)}")

    env_timing = timing.get("env", {}) if isinstance(timing, dict) else {}
    engine_timing = timing.get("engine", {}) if isinstance(timing, dict) else {}
    if env_timing:
        lines.extend(_format_timing_block("env", env_timing))
        step_calls = env_timing.get("step_calls", 0.0)
        step_time = env_timing.get("step_time_total", 0.0)
        if step_calls > 0:
            lines.append(f"  env_step_avg_sec={(step_time / step_calls):.6f}")
        wave_calls = env_timing.get("wave_sim_calls", 0.0)
        wave_time = env_timing.get("wave_sim_time_total", 0.0)
        wave_ticks_total = env_timing.get("wave_sim_ticks_total", 0.0)
        if wave_calls > 0:
            lines.append(f"  env_wave_avg_sec={(wave_time / wave_calls):.6f}")
        if wave_ticks_total > 0:
            lines.append(f"  env_tick_avg_sec={(wave_time / wave_ticks_total):.9f}")

    if engine_timing:
        lines.extend(_format_timing_block("engine", engine_timing))
        step_calls = engine_timing.get("step_calls", 0.0)
        step_time = engine_timing.get("step_time_total", 0.0)
        step_ticks = engine_timing.get("step_ticks", 0.0)
        if step_calls > 0:
            lines.append(f"  engine_step_avg_sec={(step_time / step_calls):.6f}")
        if step_ticks > 0:
            lines.append(f"  engine_tick_avg_sec={(step_time / step_ticks):.9f}")

    if stats.get("rollout_stats") is not None:
        rollout_stats = stats["rollout_stats"]
        lines.append(
            "rollout_stats count={count} mean={mean:.6f} median={median:.6f} min={min:.6f} max={max:.6f}".format(
                **rollout_stats
            )
        )
    if stats.get("train_stats") is not None:
        train_stats = stats["train_stats"]
        lines.append(
            "train_stats count={count} mean={mean:.6f} median={median:.6f} min={min:.6f} max={max:.6f}".format(
                **train_stats
            )
        )

    return "\n".join(lines) + "\n"


def _make_env_factory(
    rank: int,
    *,
    map_path: str,
    base_seed: int,
    max_build_actions: int,
    max_wave_ticks: int,
    action_space_kind: str,
    reward_config,
    place_cell_top_k: int | None,
):
    def _init():
        env = VectorTDEventEnv(
            default_map=map_path,
            action_space_kind=action_space_kind,
            max_build_actions=max_build_actions,
            max_wave_ticks=max_wave_ticks,
            reward_config=reward_config,
            place_cell_top_k=place_cell_top_k,
            timing_enabled=True,
            log_interval_sec=0.0,
        )
        env._initial_seed = base_seed + rank
        return TimingBridge(env)

    return _init


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/switchback_maskableppo.json")
    ap.add_argument("--map", default="switchback")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--n-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--total-timesteps", type=int, default=None)
    ap.add_argument("--vec-env", choices=["subproc", "dummy"], default="subproc")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--log-dir", default=None)
    args = ap.parse_args()

    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)

    default_cfg = {
        "schema_version": 1,
        "run": {"total_timesteps": 50_000, "num_envs": max(1, (os.cpu_count() or 1) - 1)},
        "env": {"action_space_kind": "legacy", "max_build_actions": 100, "max_wave_ticks": 20_000},
        "masking": {"place_cell_top_k": None},
        "reward": {},
        "sb3": {},
    }

    cfg_json = load_json_config(args.cfg)
    cfg = deep_merge(default_cfg, cfg_json)
    cfg["run"]["map"] = args.map
    cfg["run"]["seed"] = int(args.seed)
    cfg = apply_overrides(cfg, args.override)

    if args.num_envs is not None:
        cfg["run"]["num_envs"] = int(args.num_envs)
    if args.n_steps is not None:
        cfg["sb3"]["n_steps"] = int(args.n_steps)
    if args.batch_size is not None:
        cfg["sb3"]["batch_size"] = int(args.batch_size)
    if args.total_timesteps is not None:
        cfg["run"]["total_timesteps"] = int(args.total_timesteps)

    num_envs = int(cfg["run"]["num_envs"])
    total_timesteps = int(cfg["run"]["total_timesteps"])
    map_path = str(cfg["run"].get("map", args.map))
    seed = int(cfg["run"].get("seed", args.seed))
    action_space_kind = cfg["env"].get("action_space_kind") or "legacy"
    action_space_kind = str(action_space_kind)
    max_build_actions = int(cfg["env"].get("max_build_actions", 100))
    max_wave_ticks = int(cfg["env"].get("max_wave_ticks", 20_000))
    place_cell_top_k = cfg.get("masking", {}).get("place_cell_top_k")
    if place_cell_top_k is not None:
        place_cell_top_k = int(place_cell_top_k)

    reward_cfg = RewardConfig(**(cfg.get("reward") or {}))

    env_fns = [
        _make_env_factory(
            i,
            map_path=map_path,
            base_seed=seed,
            max_build_actions=max_build_actions,
            max_wave_ticks=max_wave_ticks,
            action_space_kind=action_space_kind,
            reward_config=reward_cfg,
            place_cell_top_k=place_cell_top_k,
        )
        for i in range(num_envs)
    ]
    if args.vec_env == "dummy":
        vec_env = DummyVecEnv(env_fns)
    else:
        vec_env = SubprocVecEnv(env_fns)
    timed_env = TimedVecEnv(vec_env)

    policy_kwargs_cfg = dict((cfg.get("sb3") or {}).get("policy_kwargs") or {})
    activation_name = policy_kwargs_cfg.get("activation_fn")
    if activation_name is not None and hasattr(torch.nn, str(activation_name)):
        policy_kwargs_cfg["activation_fn"] = getattr(torch.nn, str(activation_name))
    policy_kwargs = policy_kwargs_cfg
    policy = (cfg.get("sb3") or {}).get("policy", "MlpPolicy")
    sb3_cfg = dict(cfg.get("sb3") or {})
    sb3_cfg.pop("algo", None)
    sb3_cfg.pop("policy", None)
    sb3_cfg.pop("policy_kwargs", None)

    tracker = TimingTracker()
    callback = RolloutTimingCallback(tracker)
    model = TimedMaskablePPO(
        policy,
        timed_env,
        tensorboard_log=None,
        policy_kwargs=policy_kwargs,
        device=args.device,
        seed=seed,
        verbose=0,
        tracker=tracker,
        **sb3_cfg,
    )

    start_time = time.perf_counter()
    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=True, callback=callback)
    elapsed = time.perf_counter() - start_time

    timings = timed_env.env_method("get_timing_snapshot")
    timed_env.close()

    stats = {
        "cfg_path": str(Path(args.cfg)),
        "map": map_path,
        "seed": seed,
        "action_space_kind": action_space_kind,
        "vec_env": args.vec_env,
        "num_envs": num_envs,
        "n_steps": int((cfg.get("sb3") or {}).get("n_steps", 0) or 0),
        "batch_size": int((cfg.get("sb3") or {}).get("batch_size", 0) or 0),
        "n_epochs": int((cfg.get("sb3") or {}).get("n_epochs", 0) or 0),
        "policy": policy,
        "device": str(model.device),
        "timesteps": int(getattr(model, "num_timesteps", 0) or 0),
        "elapsed": elapsed,
        "learn_time_total": elapsed,
        "rollout_time_total": sum(tracker.rollout_times),
        "rollout_calls": len(tracker.rollout_times),
        "train_time_total": sum(tracker.train_times),
        "train_calls": len(tracker.train_times),
        "rollout_stats": _stats(tracker.rollout_times),
        "train_stats": _stats(tracker.train_times),
    }
    vec_timing = {
        "step_wait_time_total": timed_env.step_wait_time_total,
        "step_wait_calls": timed_env.step_wait_calls,
        "reset_time_total": timed_env.reset_time_total,
        "reset_calls": timed_env.reset_calls,
    }
    timing = _aggregate_timings(timings)
    report = _build_report(stats, timing, vec_timing)

    root_dir = _resolve_root()
    bench_dir = Path(args.log_dir) if args.log_dir else root_dir / "runs" / "throughput_benchmark" / "sb3_maskableppo"
    previous = _latest_log_path(bench_dir, prefix="sb3_maskableppo")
    if previous is not None:
        prev_report = previous.read_text(encoding="utf-8")
        if _normalize_report(prev_report) == _normalize_report(report):
            print(f"throughput_log=unchanged matches={previous}")
            return 0
    log_path = _next_log_path(bench_dir, prefix="sb3_maskableppo")
    log_path.write_text(report, encoding="utf-8")
    print(f"throughput_log={log_path}")
    return 0


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    raise SystemExit(main())
