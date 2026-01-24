from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np

try:
    import torch
    import sb3_contrib
    import stable_baselines3 as sb3
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv
except Exception as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "SB3 2.7.x, sb3-contrib 2.7.x, and torch are required for MaskablePPO training."
    ) from exc

from vectortd.ai.config_loader import apply_overrides, deep_merge, dump_effective_config, load_json_config
from vectortd.ai.env import VectorTDEventEnv
from vectortd.ai.rewards import RewardConfig
from vectortd.ai.run_metadata import write_run_metadata
from vectortd.ai.run_summary import write_run_summary


def _resolve_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parents[3]


def _next_run_dir(root_dir: Path) -> Path:
    runs_root = root_dir / "runs" / "sb3_maskableppo"
    runs_root.mkdir(parents=True, exist_ok=True)
    max_idx = 0
    for path in runs_root.iterdir():
        if not path.is_dir():
            continue
        name = path.name
        if not name.startswith("run_"):
            continue
        suffix = name[4:]
        if suffix.isdigit():
            max_idx = max(max_idx, int(suffix))
    run_dir = runs_root / f"run_{max_idx + 1:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _default_num_envs() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)


def _read_cpu_max_freq_khz(cpu_id: int) -> int | None:
    cpu_dir = Path("/sys/devices/system/cpu") / f"cpu{cpu_id}" / "cpufreq"
    for name in ("cpuinfo_max_freq", "scaling_max_freq"):
        path = cpu_dir / name
        if not path.exists():
            continue
        try:
            return int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None
    return None


def _detect_p_core_cpus() -> list[int]:
    if sys.platform != "linux":
        return []
    cpu_root = Path("/sys/devices/system/cpu")
    if not cpu_root.exists():
        return []
    cpu_ids: list[int] = []
    for entry in cpu_root.iterdir():
        name = entry.name
        if name.startswith("cpu") and name[3:].isdigit():
            cpu_ids.append(int(name[3:]))
    cpu_ids.sort()
    freq_by_cpu: dict[int, int] = {}
    for cpu_id in cpu_ids:
        freq = _read_cpu_max_freq_khz(cpu_id)
        if freq is not None:
            freq_by_cpu[cpu_id] = freq
    if not freq_by_cpu:
        return []
    max_freq = max(freq_by_cpu.values())
    if max_freq <= 0:
        return []
    threshold = int(max_freq * 0.95)
    return sorted([cpu_id for cpu_id, freq in freq_by_cpu.items() if freq >= threshold])


def _maybe_set_affinity(cpu_ids: list[int]) -> bool:
    if sys.platform != "linux":
        return False
    if not cpu_ids:
        return False
    if not hasattr(os, "sched_setaffinity"):
        return False
    try:
        os.sched_setaffinity(0, set(cpu_ids))
    except OSError:
        return False
    return True


def _resolve_activation_fn(value):
    if value is None:
        return None
    if isinstance(value, str):
        if hasattr(torch.nn, value):
            return getattr(torch.nn, value)
        raise ValueError(f"Unknown activation_fn: {value}")
    return value


def _pause_if_requested(run_dir: Path, *, poll_sec: float = 2.0) -> None:
    pause_path = run_dir / "PAUSE"
    while pause_path.exists():
        time.sleep(poll_sec)


def _log_vecenv_pids(vec_env, label: str) -> None:
    processes = None
    for attr in ("processes", "_processes"):
        if hasattr(vec_env, attr):
            processes = getattr(vec_env, attr)
            break
    pids: list[int] = []
    if processes:
        for proc in processes:
            pid = getattr(proc, "pid", None)
            if pid:
                pids.append(int(pid))
    if not pids:
        for attr in ("workers", "_workers"):
            if hasattr(vec_env, attr):
                workers = getattr(vec_env, attr)
                for worker in workers:
                    pid = getattr(worker, "pid", None)
                    if pid:
                        pids.append(int(pid))
                break
    if pids:
        print(f"{label}_worker_pids={','.join(str(pid) for pid in pids)}")
    else:
        print(f"{label}_worker_pids=unknown")


def _verify_mini_outputs(run_dir: Path) -> list[str]:
    failures: list[str] = []
    tb_dir = run_dir / "tb"
    tb_events = list(tb_dir.rglob("events.out.tfevents*")) if tb_dir.exists() else []
    if not tb_events:
        failures.append("tensorboard output missing")
    best_dir = run_dir / "best_model"
    best_models = list(best_dir.glob("*.zip")) if best_dir.exists() else []
    if not best_models:
        failures.append("best_model missing")
    replay_dir = run_dir / "replays"
    replays = list(replay_dir.glob("best_*.json")) if replay_dir.exists() else []
    if not replays:
        failures.append("best replay missing")
    return failures


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return False


def make_train_env(
    rank: int,
    *,
    map_path: str,
    base_seed: int,
    run_dir: str,
    max_build_actions: int,
    max_wave_ticks: int,
    reward_config,
    place_cell_top_k: int | None,
    cpu_id: int | None,
    log_interval_sec: float,
    debug_actions: bool,
):
    def _init():
        if cpu_id is not None:
            _maybe_set_affinity([cpu_id])
        log_dir = Path(run_dir) / "console"
        env = VectorTDEventEnv(
            default_map=map_path,
            max_build_actions=max_build_actions,
            max_wave_ticks=max_wave_ticks,
            reward_config=reward_config,
            place_cell_top_k=place_cell_top_k,
            log_dir=log_dir,
            log_prefix=f"train_{rank}",
            log_interval_sec=log_interval_sec,
            debug_actions=debug_actions,
        )
        env._initial_seed = base_seed + rank
        monitor_dir = Path(run_dir) / "monitor" / "train"
        monitor_dir.mkdir(parents=True, exist_ok=True)
        env = Monitor(env, filename=str(monitor_dir / f"monitor_{rank}.csv"))
        return env

    return _init


def make_eval_env(
    rank: int,
    *,
    map_path: str,
    base_seed: int,
    run_dir: str,
    max_build_actions: int,
    max_wave_ticks: int,
    reward_config,
    place_cell_top_k: int | None,
    cpu_id: int | None,
    debug_actions: bool,
):
    def _init():
        if cpu_id is not None:
            _maybe_set_affinity([cpu_id])
        log_dir = Path(run_dir) / "console"
        env = VectorTDEventEnv(
            default_map=map_path,
            max_build_actions=max_build_actions,
            max_wave_ticks=max_wave_ticks,
            reward_config=reward_config,
            place_cell_top_k=place_cell_top_k,
            log_dir=log_dir,
            log_prefix=f"eval_{rank}",
            log_interval_sec=0.0,
            debug_actions=debug_actions,
        )
        env._initial_seed = base_seed + rank
        monitor_dir = Path(run_dir) / "monitor" / "eval"
        monitor_dir.mkdir(parents=True, exist_ok=True)
        env = Monitor(env, filename=str(monitor_dir / f"monitor_{rank}.csv"))
        return env

    return _init


class BestReplayCallback(BaseCallback):
    def __init__(
        self,
        *,
        map_path: str,
        replay_dir: Path,
        replay_seed: int,
        max_build_actions: int,
        max_wave_ticks: int,
        reward_config,
        place_cell_top_k: int | None,
    ) -> None:
        super().__init__(verbose=0)
        self.map_path = map_path
        self.replay_dir = replay_dir
        self.replay_seed = replay_seed
        self.max_build_actions = max_build_actions
        self.max_wave_ticks = max_wave_ticks
        self.reward_config = reward_config
        self.place_cell_top_k = place_cell_top_k

    def _on_step(self) -> bool:
        self._save_replay()
        return True

    def _save_replay(self) -> None:
        self.replay_dir.mkdir(parents=True, exist_ok=True)
        env = VectorTDEventEnv(
            default_map=self.map_path,
            max_build_actions=self.max_build_actions,
            max_wave_ticks=self.max_wave_ticks,
            reward_config=self.reward_config,
            place_cell_top_k=self.place_cell_top_k,
        )
        obs, info = env.reset(seed=self.replay_seed, options={"map_path": self.map_path})
        engine_seed = int(info.get("engine_seed", 0) or 0)
        actions: list[int] = []
        done = False
        while not done:
            mask = env.action_masks()
            action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
            try:
                action_id = int(action.item())
            except AttributeError:
                action_id = int(action)
            actions.append(action_id)
            obs, _, terminated, truncated, _ = env.step(action_id)
            done = terminated or truncated

        payload = {
            "map_path": self.map_path,
            "engine_seed": engine_seed,
            "actions": actions,
            "timesteps": int(self.num_timesteps),
        }
        replay_path = self.replay_dir / f"best_{self.num_timesteps}.json"
        replay_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


class DebugInvalidActionCallback(BaseCallback):
    def __init__(self, *, log_interval_steps: int = 2048) -> None:
        super().__init__(verbose=0)
        self.log_interval_steps = int(log_interval_steps)
        self._next_log_step = int(log_interval_steps)
        self._window_total = 0
        self._window_invalid = 0
        self._total_total = 0
        self._total_invalid = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") if self.locals is not None else None
        if infos:
            for info in infos:
                self._total_total += 1
                self._window_total += 1
                if info.get("invalid_action"):
                    self._total_invalid += 1
                    self._window_invalid += 1
        if self.num_timesteps >= self._next_log_step:
            window_rate = self._window_invalid / self._window_total if self._window_total else 0.0
            total_rate = self._total_invalid / self._total_total if self._total_total else 0.0
            print(
                "debug_invalid_action_rate={:.4f} window_invalid={} window_total={} total_invalid={} total_total={} steps={}".format(
                    window_rate,
                    self._window_invalid,
                    self._window_total,
                    self._total_invalid,
                    self._total_total,
                    self.num_timesteps,
                )
            )
            self._window_total = 0
            self._window_invalid = 0
            while self._next_log_step <= self.num_timesteps:
                self._next_log_step += self.log_interval_steps
        return True


class DebugRolloutStatsCallback(BaseCallback):
    def __init__(self, *, log_interval_rollouts: int = 1) -> None:
        super().__init__(verbose=0)
        self.log_interval_rollouts = int(log_interval_rollouts)
        self._rollouts = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self._rollouts += 1
        if self.log_interval_rollouts <= 0:
            return
        if self._rollouts % self.log_interval_rollouts != 0:
            return
        rollout_buffer = getattr(self.model, "rollout_buffer", None)
        if rollout_buffer is None:
            return
        advantages = np.asarray(getattr(rollout_buffer, "advantages", []), dtype=float).ravel()
        returns = np.asarray(getattr(rollout_buffer, "returns", []), dtype=float).ravel()
        values = np.asarray(getattr(rollout_buffer, "values", []), dtype=float).ravel()
        if advantages.size == 0 or returns.size == 0 or values.size == 0:
            return
        value_error = returns - values

        def _stats(arr: np.ndarray) -> tuple[float, float, float, float]:
            return float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())

        adv_mean, adv_std, adv_min, adv_max = _stats(advantages)
        ret_mean, ret_std, ret_min, ret_max = _stats(returns)
        val_mean, val_std, val_min, val_max = _stats(values)
        err_mean, err_std, err_min, err_max = _stats(value_error)
        print(
            "debug_rollout_stats rollouts={} n={} adv_mean={:.4f} adv_std={:.4f} adv_min={:.4f} adv_max={:.4f} "
            "return_mean={:.4f} return_std={:.4f} return_min={:.4f} return_max={:.4f} "
            "value_mean={:.4f} value_std={:.4f} value_min={:.4f} value_max={:.4f} "
            "value_error_mean={:.4f} value_error_std={:.4f} value_error_min={:.4f} value_error_max={:.4f}".format(
                self._rollouts,
                advantages.size,
                adv_mean,
                adv_std,
                adv_min,
                adv_max,
                ret_mean,
                ret_std,
                ret_min,
                ret_max,
                val_mean,
                val_std,
                val_min,
                val_max,
                err_mean,
                err_std,
                err_min,
                err_max,
            )
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--map", default="switchback")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--mini", action=argparse.BooleanOptionalAction, default=False)
    args = ap.parse_args()

    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)

    default_cfg = {
        "schema_version": 1,
        "run": {
            "total_timesteps": 50_000,
            "num_envs": _default_num_envs(),
            "num_eval_envs": 2,
            "log_interval_sec": 0.0,
        },
        "env": {
            "max_build_actions": 100,
            "max_wave_ticks": 20_000,
            "deterministic_eval": True,
        },
        "masking": {
            "place_cell_top_k": None,
        },
        "reward": {
            "score_weight": 1.0,
            "score_delta_clip": None,
            "life_loss_penalty": 100.0,
            "no_life_loss_bonus": 50.0,
            "terminal_loss_penalty": 10_000.0,
            "terminal_win_bonus": 10_000.0,
            "build_step_penalty": 0.0,
            "noop_penalty": 0.0,
            "set_mode_penalty": 0.0,
            "set_mode_noop_penalty": 0.0,
        },
        "sb3": {
            "algo": "MaskablePPO",
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "n_steps": 128,
            "batch_size": 256,
            "n_epochs": 4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": {
                "net_arch": [256, 256],
                "activation_fn": "Tanh",
                "ortho_init": False,
            },
        },
        "callbacks": {
            "eval_freq_steps": 10_000,
            "n_eval_episodes": 1,
            "save_freq_steps": 10_000,
        },
    }

    cfg_json = load_json_config(args.cfg)
    cfg = deep_merge(default_cfg, cfg_json)
    cfg["run"]["map"] = args.map
    cfg["run"]["seed"] = int(args.seed)
    cfg = apply_overrides(cfg, args.override)

    if args.mini:
        cfg["run"]["num_envs"] = 2
        cfg["run"]["num_eval_envs"] = 2
        cfg["run"]["total_timesteps"] = 10_000
        cfg["callbacks"]["eval_freq_steps"] = 2_000
        cfg["callbacks"]["save_freq_steps"] = 2_000
        cfg["callbacks"]["n_eval_episodes"] = 4

    root_dir = _resolve_root()
    run_dir = Path(args.run_dir) if args.run_dir else _next_run_dir(root_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_effective_config(run_dir, cfg)

    num_envs = int(cfg["run"]["num_envs"])
    eval_envs = int(cfg["run"]["num_eval_envs"])
    total_timesteps = int(cfg["run"]["total_timesteps"])
    log_interval_sec = float(cfg["run"]["log_interval_sec"])
    map_path = str(cfg["run"].get("map", args.map))
    seed = int(cfg["run"].get("seed", args.seed))
    max_build_actions = int(cfg["env"]["max_build_actions"])
    max_wave_ticks = int(cfg["env"]["max_wave_ticks"])
    deterministic_eval = bool(cfg["env"].get("deterministic_eval", True))
    place_cell_top_k = cfg["masking"].get("place_cell_top_k")
    if place_cell_top_k is not None:
        place_cell_top_k = int(place_cell_top_k)
    reward_cfg = RewardConfig(**cfg["reward"])

    callbacks_cfg = cfg["callbacks"]
    eval_freq_steps = int(callbacks_cfg["eval_freq_steps"])
    save_freq_steps = int(callbacks_cfg["save_freq_steps"])
    n_eval_episodes = int(callbacks_cfg["n_eval_episodes"])
    eval_freq_calls = max(eval_freq_steps // num_envs, 1)
    save_freq_calls = max(save_freq_steps // num_envs, 1)
    callbacks_cfg["eval_freq_calls"] = eval_freq_calls
    callbacks_cfg["save_freq_calls"] = save_freq_calls
    dump_effective_config(run_dir, cfg)

    chunk_steps = 2_000 if args.mini else 10_000
    if chunk_steps <= 0 or chunk_steps > total_timesteps:
        chunk_steps = total_timesteps

    p_core_cpus = _detect_p_core_cpus()
    main_cpu = None
    if p_core_cpus and num_envs < len(p_core_cpus):
        candidate = p_core_cpus[num_envs]
        if _maybe_set_affinity([candidate]):
            main_cpu = candidate

    tb_log = str(run_dir / "tb")
    console_log_path = run_dir / "console.log"
    log_handle = console_log_path.open("a", encoding="utf-8")
    stdout_orig = sys.stdout
    stderr_orig = sys.stderr
    sys.stdout = _Tee(stdout_orig, log_handle)
    sys.stderr = _Tee(stderr_orig, log_handle)
    print(f"run_dir={run_dir}")
    print(f"console_log={console_log_path}")
    if p_core_cpus:
        print(f"p_core_cpus={','.join(str(cpu) for cpu in p_core_cpus)}")
    else:
        print("p_core_cpus=unknown")
    print(f"train_envs={num_envs} eval_envs={eval_envs}")
    print(f"debug={args.debug} train_log_interval_sec={log_interval_sec}")
    if args.debug:
        print("debug_action_input_logging=enabled")

    policy_kwargs_cfg = dict(cfg["sb3"].get("policy_kwargs") or {})
    activation_fn = _resolve_activation_fn(policy_kwargs_cfg.get("activation_fn"))
    if activation_fn is not None:
        policy_kwargs_cfg["activation_fn"] = activation_fn
    policy_kwargs = policy_kwargs_cfg
    policy = cfg["sb3"].get("policy", "MlpPolicy")
    sb3_cfg = dict(cfg["sb3"])
    sb3_cfg.pop("algo", None)
    sb3_cfg.pop("policy", None)
    sb3_cfg.pop("policy_kwargs", None)
    n_steps = int(cfg["sb3"].get("n_steps", 1))
    training_meta = {
        "map": map_path,
        "seed": seed,
        "cfg_path": str(Path(args.cfg)),
        "overrides": list(args.override or []),
        "num_envs": num_envs,
        "eval_envs": eval_envs,
        "chunk_steps": chunk_steps,
        "eval_freq_steps": eval_freq_steps,
        "eval_freq_calls": eval_freq_calls,
        "save_freq_steps": save_freq_steps,
        "save_freq_calls": save_freq_calls,
        "n_eval_episodes": n_eval_episodes,
        "max_build_actions": max_build_actions,
        "max_wave_ticks": max_wave_ticks,
        "deterministic_eval": deterministic_eval,
        "place_cell_top_k": place_cell_top_k,
        "tensorboard_log": tb_log,
        "debug": args.debug,
        "mini": args.mini,
        "p_core_cpus": p_core_cpus,
        "main_process_cpu": main_cpu,
    }
    if args.debug:
        training_meta["debug_options"] = {
            "action_summary": True,
            "mask_summary": True,
            "reward_breakdown": True,
            "invariant_checks": True,
            "rollout_stats": True,
        }
    algorithm_meta = {
        "name": "sb3_maskableppo",
        "library": "sb3_contrib",
        "sb3_contrib_version": getattr(sb3_contrib, "__version__", "unknown"),
        "stable_baselines3_version": getattr(sb3, "__version__", "unknown"),
        "torch_version": getattr(torch, "__version__", "unknown"),
        "policy": policy,
        "policy_kwargs": {
            "net_arch": list((cfg["sb3"].get("policy_kwargs") or {}).get("net_arch") or []),
            "activation_fn": (cfg["sb3"].get("policy_kwargs") or {}).get("activation_fn", "unknown"),
            "ortho_init": (cfg["sb3"].get("policy_kwargs") or {}).get("ortho_init"),
        },
        "hyperparams": {
            "learning_rate": cfg["sb3"].get("learning_rate"),
            "gamma": cfg["sb3"].get("gamma"),
            "gae_lambda": cfg["sb3"].get("gae_lambda"),
            "n_steps": cfg["sb3"].get("n_steps"),
            "batch_size": cfg["sb3"].get("batch_size"),
            "n_epochs": cfg["sb3"].get("n_epochs"),
            "clip_range": cfg["sb3"].get("clip_range"),
            "ent_coef": cfg["sb3"].get("ent_coef"),
            "vf_coef": cfg["sb3"].get("vf_coef"),
            "max_grad_norm": cfg["sb3"].get("max_grad_norm"),
        },
        "device": "auto",
    }
    run_metadata_path = write_run_metadata(
        run_dir,
        total_timesteps=total_timesteps,
        training=training_meta,
        algorithm=algorithm_meta,
    )
    print(f"run_metadata={run_metadata_path} total_timesteps={total_timesteps} step_unit=timesteps")

    train_env = SubprocVecEnv(
        [
            make_train_env(
                i,
                map_path=map_path,
                base_seed=seed,
                run_dir=str(run_dir),
                max_build_actions=max_build_actions,
                max_wave_ticks=max_wave_ticks,
                reward_config=reward_cfg,
                place_cell_top_k=place_cell_top_k,
                cpu_id=p_core_cpus[i % len(p_core_cpus)] if p_core_cpus else None,
                log_interval_sec=log_interval_sec,
                debug_actions=args.debug,
            )
            for i in range(num_envs)
        ]
    )
    eval_env = SubprocVecEnv(
        [
            make_eval_env(
                i,
                map_path=map_path,
                base_seed=seed + 10_000,
                run_dir=str(run_dir),
                max_build_actions=max_build_actions,
                max_wave_ticks=max_wave_ticks,
                reward_config=reward_cfg,
                place_cell_top_k=place_cell_top_k,
                cpu_id=p_core_cpus[i] if p_core_cpus and i < len(p_core_cpus) else None,
                debug_actions=args.debug,
            )
            for i in range(eval_envs)
        ]
    )
    if args.debug:
        _log_vecenv_pids(train_env, "train_env")
        _log_vecenv_pids(eval_env, "eval_env")

    model = MaskablePPO(
        policy,
        train_env,
        tensorboard_log=tb_log,
        policy_kwargs=policy_kwargs,
        device="auto",
        seed=seed,
        verbose=1,
        **sb3_cfg,
    )
    write_run_metadata(
        run_dir,
        total_timesteps=total_timesteps,
        algorithm={"device_resolved": str(model.device)},
    )

    best_model_dir = run_dir / "best_model"
    eval_log_dir = run_dir / "eval_logs"
    replay_dir = run_dir / "replays"
    best_replay = BestReplayCallback(
        map_path=map_path,
        replay_dir=replay_dir,
        replay_seed=seed + 42,
        max_build_actions=max_build_actions,
        max_wave_ticks=max_wave_ticks,
        reward_config=reward_cfg,
        place_cell_top_k=place_cell_top_k,
    )
    eval_callback = MaskableEvalCallback(
        eval_env=eval_env,
        eval_freq=eval_freq_calls,
        n_eval_episodes=n_eval_episodes,
        best_model_save_path=str(best_model_dir),
        log_path=str(eval_log_dir),
        deterministic=deterministic_eval,
        render=False,
        callback_on_new_best=best_replay,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq_calls,
        save_path=str(run_dir / "checkpoints"),
        name_prefix="maskableppo",
    )
    callbacks = [eval_callback, checkpoint_callback]
    if args.debug:
        callbacks.append(DebugInvalidActionCallback(log_interval_steps=n_steps * num_envs))
        callbacks.append(DebugRolloutStatsCallback(log_interval_rollouts=1))
    callback = CallbackList(callbacks)

    remaining = total_timesteps
    error_info = None
    try:
        while remaining > 0:
            _pause_if_requested(run_dir)
            steps = min(chunk_steps, remaining)
            model.learn(total_timesteps=steps, reset_num_timesteps=False, callback=callback)
            remaining -= steps
            _pause_if_requested(run_dir)
    except BaseException as exc:
        error_info = {"type": type(exc).__name__, "message": str(exc)}
        print("training_failed=1", file=sys.stderr)
        traceback.print_exc()
        raise
    finally:
        actual_timesteps = int(getattr(model, "num_timesteps", 0) or 0)
        write_run_metadata(
            run_dir,
            total_timesteps=total_timesteps,
            training={"actual_timesteps": actual_timesteps},
        )
        try:
            log_handle.flush()
        except Exception:
            pass
        try:
            write_run_summary(
                run_dir,
                algorithm="sb3_maskableppo",
                console_log=console_log_path,
                monitor_train_dir=run_dir / "monitor" / "train",
                monitor_eval_dir=run_dir / "monitor" / "eval",
                eval_npz_path=eval_log_dir / "evaluations.npz",
                error=error_info,
            )
        except Exception:
            pass
        try:
            train_env.close()
        except Exception:
            pass
        try:
            eval_env.close()
        except Exception:
            pass
        sys.stdout = stdout_orig
        sys.stderr = stderr_orig
        log_handle.close()

    if args.mini:
        failures = _verify_mini_outputs(run_dir)
        if failures:
            for failure in failures:
                print(f"mini_check_failed={failure}")
            return 1

    return 0


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    raise SystemExit(main())
