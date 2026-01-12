from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np

from vectortd.ai.env import VectorTDEventEnv
from vectortd.ai.vectorized_env import VectorizedEnv
from vectortd.core.engine import Engine
from vectortd.core.model.map import load_map_json
from vectortd.core.rules.placement import buildable_cells, place_tower
from vectortd.core.rules.wave_spawner import start_next_wave


def _ensure_torch():
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise SystemExit("PyTorch is required for policy/train benchmarks. Install torch and retry.") from exc
    return torch


def _mask_list_to_tensor(mask_list, *, device, torch) -> "torch.Tensor":
    mask_array = np.asarray(mask_list, dtype=bool)
    return torch.as_tensor(mask_array, dtype=torch.bool, device=device)


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


def _choose_action(action_mode: str, rng: random.Random) -> int:
    if action_mode == "start_wave":
        return 1
    if action_mode == "noop":
        return 0
    if action_mode == "random":
        return rng.choice([0, 1])
    raise ValueError(f"Unknown action_mode={action_mode!r}")


def _resolve_map_path(map_path: str) -> Path:
    root = Path(__file__).resolve().parents[3]
    p = Path(map_path)
    if p.suffix:
        return p if p.is_absolute() else root / p
    if p.parent == Path("."):
        return root / "data/maps" / f"{p.name}.json"
    return root / p.with_suffix(".json")


def _parse_tower_kinds(raw: str) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _populate_towers(state, map_data, tower_kinds: list[str], tower_count: int, bank: int) -> int:
    if tower_count <= 0 or not tower_kinds:
        return 0
    state.bank = max(int(getattr(state, "bank", 0)), int(bank))
    cells = buildable_cells(map_data)
    cells.sort(key=lambda cell: (cell[1], cell[0]))
    placed = 0
    for cell_x, cell_y in cells:
        if placed >= tower_count:
            break
        kind = tower_kinds[placed % len(tower_kinds)]
        if place_tower(state, map_data, int(cell_x), int(cell_y), tower_kind=kind) is not None:
            placed += 1
    return placed


def _run_single(args) -> tuple[dict[str, Any], dict[str, Any]]:
    env = VectorTDEventEnv(max_build_actions=args.max_build_actions, timing_enabled=True)
    obs = env.reset(map_path=args.map, seed=args.seed)
    rng = random.Random(args.seed)
    start_time = time.perf_counter()
    steps = 0
    waves = 0
    wave_ticks_total = 0
    resets = 0

    for _ in range(args.steps):
        action = _choose_action(args.action, rng)
        obs, _, done, info = env.step(action)
        steps += 1
        if "wave_ticks" in info:
            waves += 1
            wave_ticks_total += int(info.get("wave_ticks") or 0)
        if done and args.reset_on_done:
            resets += 1
            obs = env.reset(map_path=args.map, seed=args.seed + resets)
    elapsed = time.perf_counter() - start_time
    timing = env.get_timing_snapshot()
    stats = {
        "benchmark": "env_step",
        "mode": "single",
        "num_envs": 1,
        "steps": steps,
        "waves": waves,
        "wave_ticks_total": wave_ticks_total,
        "elapsed": elapsed,
        "resets": resets,
    }
    return stats, timing


def _run_vectorized(args) -> tuple[dict[str, Any], dict[str, Any]]:
    num_envs = args.num_envs or max(1, (os.cpu_count() or 1) - 1)
    vec = VectorizedEnv(
        num_envs=num_envs,
        env_kwargs={"max_build_actions": args.max_build_actions, "timing_enabled": True},
    )
    base_seed = args.seed
    seeds = [base_seed + idx for idx in range(num_envs)]
    vec.reset(map_paths=args.map, seeds=seeds)

    rng = random.Random(args.seed)
    reset_counts = [0] * num_envs
    steps = 0
    waves = 0
    wave_ticks_total = 0
    resets = 0

    start_time = time.perf_counter()
    for _ in range(args.steps):
        actions = [_choose_action(args.action, rng) for _ in range(num_envs)]
        results = vec.step(actions)
        steps += num_envs
        reset_indices: list[int] = []
        reset_seeds: list[int] = []
        for idx, result in enumerate(results):
            if "wave_ticks" in result.info:
                waves += 1
                wave_ticks_total += int(result.info.get("wave_ticks") or 0)
            if result.done and args.reset_on_done:
                reset_counts[idx] += 1
                resets += 1
                reset_indices.append(idx)
                reset_seeds.append(base_seed + idx + reset_counts[idx] * num_envs)
        if reset_indices:
            vec.reset_at(reset_indices, map_paths=args.map, seeds=reset_seeds)
    elapsed = time.perf_counter() - start_time
    timings = vec.get_timings()
    vec.close()
    stats = {
        "benchmark": "env_step",
        "mode": "vectorized",
        "num_envs": num_envs,
        "steps": steps,
        "waves": waves,
        "wave_ticks_total": wave_ticks_total,
        "elapsed": elapsed,
        "resets": resets,
    }
    return stats, _aggregate_timings(timings)


def _run_engine_tick(args) -> tuple[dict[str, Any], dict[str, Any]]:
    map_path = _resolve_map_path(args.map)
    map_data = load_map_json(map_path)
    tower_kinds = _parse_tower_kinds(args.tower_kinds)

    def _run_pass(*, timing_enabled: bool, tower_timing_enabled: bool) -> tuple[dict[str, Any], Engine]:
        engine = Engine(map_data)
        engine.timing_enabled = timing_enabled
        engine.tower_timing_enabled = tower_timing_enabled
        engine.timing = {}
        engine.tower_timing = {}
        engine.reset()
        engine.state.auto_level = True
        tower_count = _populate_towers(
            engine.state,
            map_data,
            tower_kinds,
            args.tower_count,
            args.tower_bank,
        )
        start_next_wave(engine.state, map_data)

        ticks_target = args.ticks
        ticks = 0
        waves = 0
        resets = 0
        prev_level = int(getattr(engine.state, "level", 0))

        start_time = time.perf_counter()
        while ticks < ticks_target:
            engine.step(engine.FRAME_DT)
            ticks += 1
            current_level = int(getattr(engine.state, "level", 0))
            if current_level != prev_level:
                if current_level > prev_level:
                    waves += current_level - prev_level
                prev_level = current_level
            if getattr(engine.state, "game_over", False):
                if not args.reset_on_done:
                    break
                resets += 1
                engine.reset()
                engine.state.auto_level = True
                _populate_towers(
                    engine.state,
                    map_data,
                    tower_kinds,
                    args.tower_count,
                    args.tower_bank,
                )
                start_next_wave(engine.state, map_data)
                prev_level = int(getattr(engine.state, "level", 0))

        elapsed = time.perf_counter() - start_time
        stats = {
            "steps": ticks,
            "waves": waves,
            "wave_ticks_total": ticks,
            "elapsed": elapsed,
            "resets": resets,
            "ticks_target": ticks_target,
            "tower_count": tower_count,
        }
        return stats, engine

    base_stats, base_engine = _run_pass(timing_enabled=True, tower_timing_enabled=False)
    kind_stats, kind_engine = _run_pass(timing_enabled=False, tower_timing_enabled=True)

    stats = {
        "benchmark": "engine_tick",
        "mode": "single",
        "num_envs": 1,
        "steps": base_stats["steps"],
        "waves": base_stats["waves"],
        "wave_ticks_total": base_stats["wave_ticks_total"],
        "elapsed": base_stats["elapsed"],
        "resets": base_stats["resets"],
        "ticks_target": base_stats["ticks_target"],
        "tower_count": base_stats["tower_count"],
        "tower_kinds": tower_kinds,
        "tower_kind_run": kind_stats,
    }
    return stats, {"engine": dict(base_engine.timing), "tower_kind": dict(kind_engine.tower_timing)}


def _init_policy(args, *, map_name: str):
    torch = _ensure_torch()
    from vectortd.ai.training.ppo import PPOAgent, PPOConfig, SCALAR_KEYS

    probe_env = VectorTDEventEnv(max_build_actions=args.max_build_actions)
    obs = probe_env.reset(map_path=map_name, seed=args.seed)
    if probe_env.action_spec is None:
        raise RuntimeError("Missing action spec after reset")
    action_dim = probe_env.action_spec.num_actions
    slot_size = len(obs.get("tower_slot_features", []) or [])
    max_towers = len(obs.get("tower_slots", []) or [])
    obs_dim = len(SCALAR_KEYS) + max_towers * slot_size
    device = torch.device(args.device)
    config = PPOConfig(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
    )
    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=config, device=device)
    return agent, obs_dim, action_dim, slot_size, max_towers, device


def _run_policy_single(args) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = _ensure_torch()
    from vectortd.ai.training.ppo import batch_to_tensor

    agent, obs_dim, action_dim, slot_size, max_towers, device = _init_policy(args, map_name=args.map)
    env = VectorTDEventEnv(max_build_actions=args.max_build_actions, timing_enabled=True)
    obs = env.reset(map_path=args.map, seed=args.seed)
    rng = random.Random(args.seed)
    start_time = time.perf_counter()
    policy_time = 0.0
    policy_calls = 0
    steps = 0
    waves = 0
    wave_ticks_total = 0
    resets = 0

    for _ in range(args.steps):
        mask = env.get_action_mask()
        obs_tensor = batch_to_tensor([obs], max_towers=max_towers, slot_size=slot_size, device=device)
        mask_tensor = _mask_list_to_tensor([mask], device=device, torch=torch)
        t0 = time.perf_counter()
        actions, _, _ = agent.act(obs_tensor, mask_tensor)
        policy_time += time.perf_counter() - t0
        policy_calls += 1
        action = int(actions.item())
        obs, _, done, info = env.step(action)
        steps += 1
        if "wave_ticks" in info:
            waves += 1
            wave_ticks_total += int(info.get("wave_ticks") or 0)
        if done and args.reset_on_done:
            resets += 1
            obs = env.reset(map_path=args.map, seed=args.seed + resets)
    elapsed = time.perf_counter() - start_time
    timing = env.get_timing_snapshot()
    stats = {
        "benchmark": "policy_inference",
        "mode": "single",
        "num_envs": 1,
        "steps": steps,
        "waves": waves,
        "wave_ticks_total": wave_ticks_total,
        "elapsed": elapsed,
        "resets": resets,
        "policy_time_total": policy_time,
        "policy_calls": policy_calls,
    }
    return stats, timing


def _run_policy_vectorized(args) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = _ensure_torch()
    from vectortd.ai.training.ppo import batch_to_tensor

    agent, obs_dim, action_dim, slot_size, max_towers, device = _init_policy(args, map_name=args.map)
    num_envs = args.num_envs or max(1, (os.cpu_count() or 1) - 1)
    vec = VectorizedEnv(
        num_envs=num_envs,
        env_kwargs={"max_build_actions": args.max_build_actions, "timing_enabled": True},
    )
    base_seed = args.seed
    seeds = [base_seed + idx for idx in range(num_envs)]
    obs_list = vec.reset(map_paths=args.map, seeds=seeds)
    mask_list = vec.get_action_masks()

    start_time = time.perf_counter()
    policy_time = 0.0
    policy_calls = 0
    steps = 0
    waves = 0
    wave_ticks_total = 0
    resets = 0
    for _ in range(args.steps):
        obs_tensor = batch_to_tensor(obs_list, max_towers=max_towers, slot_size=slot_size, device=device)
        mask_tensor = _mask_list_to_tensor(mask_list, device=device, torch=torch)
        t0 = time.perf_counter()
        actions, _, _ = agent.act(obs_tensor, mask_tensor)
        policy_time += time.perf_counter() - t0
        policy_calls += 1
        actions_list = actions.detach().cpu().tolist()
        results = vec.step(actions_list)
        steps += num_envs
        reset_indices: list[int] = []
        reset_seeds: list[int] = []
        for idx, result in enumerate(results):
            if "wave_ticks" in result.info:
                waves += 1
                wave_ticks_total += int(result.info.get("wave_ticks") or 0)
            if result.done and args.reset_on_done:
                resets += 1
                reset_indices.append(idx)
                reset_seeds.append(base_seed + idx + resets * num_envs)
        obs_list = [res.obs for res in results]
        mask_list = [
            res.info.get("action_mask") if res.info.get("action_mask") is not None else mask_list[idx]
            for idx, res in enumerate(results)
        ]
        if reset_indices:
            reset_obs = vec.reset_at(reset_indices, map_paths=args.map, seeds=reset_seeds)
            for sub_idx, env_idx in enumerate(reset_indices):
                obs_list[env_idx] = reset_obs[sub_idx]
            mask_list = vec.get_action_masks()

    elapsed = time.perf_counter() - start_time
    timings = vec.get_timings()
    vec.close()
    stats = {
        "benchmark": "policy_inference",
        "mode": "vectorized",
        "num_envs": num_envs,
        "steps": steps,
        "waves": waves,
        "wave_ticks_total": wave_ticks_total,
        "elapsed": elapsed,
        "resets": resets,
        "policy_time_total": policy_time,
        "policy_calls": policy_calls,
    }
    return stats, _aggregate_timings(timings)


def _run_ppo_train(args) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = _ensure_torch()
    from vectortd.ai.training.ppo import batch_to_tensor, compute_gae

    agent, obs_dim, action_dim, slot_size, max_towers, device = _init_policy(args, map_name=args.map)
    num_envs = args.num_envs or max(1, (os.cpu_count() or 1) - 1)
    vec = VectorizedEnv(
        num_envs=num_envs,
        env_kwargs={"max_build_actions": args.max_build_actions, "timing_enabled": True},
    )
    base_seed = args.seed
    seeds = [base_seed + idx for idx in range(num_envs)]
    obs_list = vec.reset(map_paths=args.map, seeds=seeds)
    mask_list = vec.get_action_masks()

    start_time = time.perf_counter()
    policy_time = 0.0
    update_time = 0.0
    rollout_time_total = 0.0
    gae_time_total = 0.0
    batch_prep_time_total = 0.0
    total_env_steps = 0
    total_waves = 0
    total_wave_ticks = 0
    update_details = [] if args.detailed else None

    for update_idx in range(args.updates):
        obs_buf = []
        actions_buf = []
        logp_buf = []
        values_buf = []
        rewards_buf = []
        dones_buf = []
        masks_buf = []
        update_env_steps = 0
        update_waves = 0
        update_wave_ticks = 0
        policy_time_start = policy_time
        rollout_start = time.perf_counter()

        for _ in range(args.rollout_steps):
            obs_tensor = batch_to_tensor(obs_list, max_towers=max_towers, slot_size=slot_size, device=device)
            mask_tensor = _mask_list_to_tensor(mask_list, device=device, torch=torch)
            t0 = time.perf_counter()
            actions, log_probs, values = agent.act(obs_tensor, mask_tensor)
            policy_time += time.perf_counter() - t0

            actions_list = actions.detach().cpu().tolist()
            results = vec.step(actions_list)
            rewards = torch.tensor([res.reward for res in results], dtype=torch.float32, device=device)
            dones = torch.tensor([res.done for res in results], dtype=torch.float32, device=device)

            obs_buf.append(obs_tensor)
            actions_buf.append(actions)
            logp_buf.append(log_probs)
            values_buf.append(values)
            rewards_buf.append(rewards)
            dones_buf.append(dones)
            masks_buf.append(mask_tensor)

            total_env_steps += num_envs
            update_env_steps += num_envs
            obs_list = [res.obs for res in results]
            mask_list = [
                res.info.get("action_mask") if res.info.get("action_mask") is not None else mask_list[idx]
                for idx, res in enumerate(results)
            ]
            reset_indices = [idx for idx, res in enumerate(results) if res.done]
            if reset_indices:
                reset_seeds = [base_seed + idx + total_env_steps for idx in reset_indices]
                reset_obs = vec.reset_at(reset_indices, map_paths=args.map, seeds=reset_seeds)
                for sub_idx, env_idx in enumerate(reset_indices):
                    obs_list[env_idx] = reset_obs[sub_idx]
                mask_list = vec.get_action_masks()
            for result in results:
                if "wave_ticks" in result.info:
                    total_waves += 1
                    total_wave_ticks += int(result.info.get("wave_ticks") or 0)
                    update_waves += 1
                    update_wave_ticks += int(result.info.get("wave_ticks") or 0)

        rollout_time = time.perf_counter() - rollout_start
        rollout_time_total += rollout_time

        last_obs_tensor = batch_to_tensor(obs_list, max_towers=max_towers, slot_size=slot_size, device=device)
        last_mask_tensor = _mask_list_to_tensor(mask_list, device=device, torch=torch)
        with torch.no_grad():
            _, _, last_values = agent.act(last_obs_tensor, last_mask_tensor, deterministic=True)

        gae_start = time.perf_counter()
        rewards_tensor = torch.stack(rewards_buf)
        values_tensor = torch.stack(values_buf)
        dones_tensor = torch.stack(dones_buf)
        advantages, returns = compute_gae(
            rewards_tensor,
            values_tensor,
            dones_tensor,
            last_values.detach(),
            gamma=agent.config.gamma,
            gae_lambda=agent.config.gae_lambda,
        )
        gae_time = time.perf_counter() - gae_start
        gae_time_total += gae_time

        batch_prep_start = time.perf_counter()
        obs_tensor = torch.stack(obs_buf).reshape(-1, obs_dim)
        actions_tensor = torch.stack(actions_buf).reshape(-1)
        logp_tensor = torch.stack(logp_buf).reshape(-1)
        masks_tensor = torch.stack(masks_buf).reshape(-1, action_dim)
        advantages_tensor = advantages.reshape(-1).detach()
        returns_tensor = returns.reshape(-1).detach()
        batch_prep_time = time.perf_counter() - batch_prep_start
        batch_prep_time_total += batch_prep_time

        update_start = time.perf_counter()
        agent.update(
            obs=obs_tensor,
            actions=actions_tensor,
            masks=masks_tensor,
            old_log_probs=logp_tensor.detach(),
            returns=returns_tensor,
            advantages=advantages_tensor,
        )
        update_duration = time.perf_counter() - update_start
        update_time += update_duration
        if update_details is not None:
            update_details.append(
                {
                    "update": update_idx,
                    "rollout_sec": rollout_time,
                    "gae_sec": gae_time,
                    "batch_prep_sec": batch_prep_time,
                    "update_sec": update_duration,
                    "policy_sec": policy_time - policy_time_start,
                    "env_steps": update_env_steps,
                    "waves": update_waves,
                    "wave_ticks": update_wave_ticks,
                }
            )

    elapsed = time.perf_counter() - start_time
    timings = vec.get_timings()
    vec.close()
    stats = {
        "benchmark": "ppo_train",
        "mode": "vectorized",
        "num_envs": num_envs,
        "steps": total_env_steps,
        "waves": total_waves,
        "wave_ticks_total": total_wave_ticks,
        "elapsed": elapsed,
        "resets": 0,
        "policy_time_total": policy_time,
        "update_time_total": update_time,
        "updates": args.updates,
        "rollout_steps": args.rollout_steps,
        "rollout_time_total": rollout_time_total,
        "gae_time_total": gae_time_total,
        "batch_prep_time_total": batch_prep_time_total,
        "update_details": update_details,
    }
    return stats, _aggregate_timings(timings)


def _build_report(stats: dict[str, Any], timing: dict[str, Any], args) -> str:
    elapsed = float(stats["elapsed"])
    steps = int(stats["steps"])
    waves = int(stats["waves"])
    wave_ticks = int(stats["wave_ticks_total"])
    steps_per_sec = steps / elapsed if elapsed > 0 else 0.0
    ticks_per_sec = wave_ticks / elapsed if elapsed > 0 else 0.0

    action_label = args.action if stats.get("benchmark") == "env_step" else stats.get("benchmark")
    lines = [
        "VectorTD throughput benchmark",
        f"timestamp={time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"benchmark={stats.get('benchmark','env_step')}",
        f"mode={stats['mode']} num_envs={stats['num_envs']}",
        f"map={args.map} seed={args.seed} action={action_label}",
        f"steps={steps} waves={waves} wave_ticks={wave_ticks} resets={stats['resets']}",
        f"elapsed_sec={elapsed:.6f} steps_per_sec={steps_per_sec:.2f} ticks_per_sec={ticks_per_sec:.2f}",
    ]
    if stats.get("benchmark") == "engine_tick":
        ticks_target = stats.get("ticks_target")
        if ticks_target is not None:
            lines.append(f"ticks_target={int(ticks_target)}")
        tower_count = stats.get("tower_count")
        if tower_count is not None:
            lines.append(f"towers={int(tower_count)}")
        tower_kinds = stats.get("tower_kinds") or []
        if tower_kinds:
            lines.append(f"tower_kinds={','.join(tower_kinds)}")
        kind_run = stats.get("tower_kind_run") or {}
        if kind_run:
            kind_elapsed = float(kind_run.get("elapsed", 0.0))
            kind_steps = int(kind_run.get("steps", 0))
            kind_waves = int(kind_run.get("waves", 0))
            kind_ticks = int(kind_run.get("wave_ticks_total", 0))
            kind_resets = int(kind_run.get("resets", 0))
            kind_steps_per_sec = kind_steps / kind_elapsed if kind_elapsed > 0 else 0.0
            kind_ticks_per_sec = kind_ticks / kind_elapsed if kind_elapsed > 0 else 0.0
            lines.append("tower_kind_run:")
            lines.append(
                f"  steps={kind_steps} waves={kind_waves} wave_ticks={kind_ticks} resets={kind_resets}"
            )
            lines.append(
                f"  elapsed_sec={kind_elapsed:.6f} steps_per_sec={kind_steps_per_sec:.2f}"
                f" ticks_per_sec={kind_ticks_per_sec:.2f}"
            )

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

    tower_kind_timing = timing.get("tower_kind", {}) if isinstance(timing, dict) else {}
    if tower_kind_timing:
        lines.extend(_format_timing_block("tower_kind", tower_kind_timing))

    if stats.get("policy_time_total") is not None:
        policy_time = float(stats.get("policy_time_total", 0.0))
        policy_calls = float(stats.get("policy_calls", 0.0))
        if policy_calls > 0:
            lines.append(f"policy_time_total={policy_time:.6f} policy_calls={int(policy_calls)}")
            lines.append(f"policy_avg_sec={(policy_time / policy_calls):.6f}")
    if stats.get("update_time_total") is not None:
        update_time = float(stats.get("update_time_total", 0.0))
        updates = float(stats.get("updates", 0.0))
        if updates > 0:
            lines.append(f"update_time_total={update_time:.6f} updates={int(updates)}")
            lines.append(f"update_avg_sec={(update_time / updates):.6f}")

    if args.detailed and stats.get("benchmark") == "ppo_train":
        updates = float(stats.get("updates", 0.0))
        rollout_time = float(stats.get("rollout_time_total", 0.0))
        gae_time = float(stats.get("gae_time_total", 0.0))
        batch_prep_time = float(stats.get("batch_prep_time_total", 0.0))
        if updates > 0:
            lines.append(f"rollout_time_total={rollout_time:.6f} rollout_avg_sec={(rollout_time / updates):.6f}")
            lines.append(f"gae_time_total={gae_time:.6f} gae_avg_sec={(gae_time / updates):.6f}")
            lines.append(f"batch_prep_time_total={batch_prep_time:.6f} batch_prep_avg_sec={(batch_prep_time / updates):.6f}")
        detail_rows = stats.get("update_details") or []
        if detail_rows:
            lines.append("update_details:")
            for entry in detail_rows:
                lines.append(
                    "  "
                    + " ".join(
                        [
                            f"update={entry.get('update')}",
                            f"rollout_sec={entry.get('rollout_sec'):.6f}",
                            f"gae_sec={entry.get('gae_sec'):.6f}",
                            f"batch_prep_sec={entry.get('batch_prep_sec'):.6f}",
                            f"update_sec={entry.get('update_sec'):.6f}",
                            f"policy_sec={entry.get('policy_sec'):.6f}",
                            f"env_steps={entry.get('env_steps')}",
                            f"waves={entry.get('waves')}",
                            f"wave_ticks={entry.get('wave_ticks')}",
                        ]
                    )
                )

    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--benchmark",
        choices=["env_step", "policy_inference", "ppo_train", "engine_tick"],
        default="env_step",
    )
    ap.add_argument("--mode", choices=["single", "vectorized"], default="vectorized")
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--map", default="switchback")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--ticks", type=int, default=10000)
    ap.add_argument("--updates", type=int, default=2)
    ap.add_argument("--rollout-steps", type=int, default=128)
    ap.add_argument("--max-build-actions", type=int, default=100)
    ap.add_argument("--action", choices=["start_wave", "noop", "random"], default="start_wave")
    ap.add_argument("--reset-on-done", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--tower-kinds",
        default="green,red_refractor,red_spammer,red_rockets,purple1,purple2,purple3,blue1,blue2",
    )
    ap.add_argument("--tower-count", type=int, default=9)
    ap.add_argument("--tower-bank", type=int, default=100000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-range", type=float, default=0.2)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--update-epochs", type=int, default=4)
    ap.add_argument("--minibatch-size", type=int, default=256)
    ap.add_argument("--detailed", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--log-dir", default=None)
    args = ap.parse_args()

    if args.benchmark == "env_step":
        if args.mode == "single":
            stats, timing = _run_single(args)
        else:
            stats, timing = _run_vectorized(args)
    elif args.benchmark == "policy_inference":
        if args.mode == "single":
            stats, timing = _run_policy_single(args)
        else:
            stats, timing = _run_policy_vectorized(args)
    elif args.benchmark == "engine_tick":
        stats, timing = _run_engine_tick(args)
    else:
        stats, timing = _run_ppo_train(args)

    root_dir = Path(__file__).resolve().parents[3]
    bench_dir = Path(args.log_dir) if args.log_dir else root_dir / "runs" / "throughput_benchmark" / args.benchmark
    report = _build_report(stats, timing, args)
    previous = _latest_log_path(bench_dir, prefix=args.benchmark)
    if previous is not None:
        prev_report = previous.read_text(encoding="utf-8")
        if _normalize_report(prev_report) == _normalize_report(report):
            print(f"throughput_log=unchanged matches={previous}")
            return 0
    log_path = _next_log_path(bench_dir, prefix=args.benchmark)
    log_path.write_text(report, encoding="utf-8")
    print(f"throughput_log={log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
