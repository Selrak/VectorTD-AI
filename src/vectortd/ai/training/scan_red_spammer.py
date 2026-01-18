from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any

try:
    import torch
except Exception as exc:  # pragma: no cover - import guard
    raise SystemExit("PyTorch is required for PPO checkpoint evaluation.") from exc

from vectortd.ai.actions import ActionSpaceSpec, action_to_dict
from vectortd.ai.env import VectorTDEventEnv
from vectortd.ai.training.ppo import PPOAgent, PPOConfig, SCALAR_KEYS, batch_to_tensor
from vectortd.ai.vectorized_env import VectorizedEnv
from vectortd.io.replay import Replay, build_state_check, save_replay


@dataclass(frozen=True, slots=True)
class CheckpointInfo:
    step: int
    path: Path


def _resolve_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parents[3]


def _list_checkpoints(root: Path, *, min_step: int) -> list[CheckpointInfo]:
    entries: list[CheckpointInfo] = []
    for path in root.rglob("ppo_checkpoint_*.pt"):
        match = re.search(r"ppo_checkpoint_(\d+)\.pt$", path.name)
        if not match:
            continue
        step = int(match.group(1))
        if step < min_step:
            continue
        entries.append(CheckpointInfo(step=step, path=path))
    entries.sort(key=lambda item: item.step, reverse=True)
    return entries


def _config_from_checkpoint(payload: dict[str, Any]) -> PPOConfig:
    defaults = PPOConfig().__dict__
    overrides = payload.get("config") or {}
    merged = dict(defaults)
    merged.update({k: v for k, v in overrides.items() if k in defaults})
    return PPOConfig(**merged)


def _load_agent(checkpoint_path: Path, device: torch.device) -> tuple[PPOAgent, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device)
    if "obs_dim" not in payload or "action_dim" not in payload or "state_dict" not in payload:
        raise ValueError(f"Checkpoint missing required keys: {checkpoint_path}")
    config = _config_from_checkpoint(payload)
    agent = PPOAgent(
        obs_dim=int(payload["obs_dim"]),
        action_dim=int(payload["action_dim"]),
        config=config,
        device=device,
    )
    agent.model.load_state_dict(payload["state_dict"])
    agent.model.eval()
    return agent, payload


def _probe_env(
    map_name: str,
    *,
    max_build_actions: int,
    max_wave_ticks: int,
) -> tuple[VectorTDEventEnv, ActionSpaceSpec, int, int, int, int]:
    env = VectorTDEventEnv(max_build_actions=max_build_actions, max_wave_ticks=max_wave_ticks)
    env.reset(seed=1, options={"map_path": map_name})
    if env.action_spec is None:
        raise RuntimeError("Missing action spec after reset")
    obs_dict = env.last_obs or {}
    max_towers = len(obs_dict.get("tower_slots", []) or [])
    slot_size = len(obs_dict.get("tower_slot_features", []) or [])
    obs_dim = len(SCALAR_KEYS) + max_towers * slot_size
    action_dim = env.action_spec.num_actions
    return env, env.action_spec, obs_dim, action_dim, max_towers, slot_size


def _to_mask_tensor(masks: list[Any], device: torch.device) -> torch.Tensor:
    try:
        import numpy as np  # type: ignore

        mask_array = np.asarray(masks, dtype=bool)
        return torch.as_tensor(mask_array, dtype=torch.bool, device=device)
    except Exception:
        return torch.tensor(masks, dtype=torch.bool, device=device)


def _is_red_spammer_action(action_id: int, spec: ActionSpaceSpec, red_index: int) -> bool:
    if action_id < spec.offsets.place:
        return False
    if action_id >= spec.offsets.place + spec.place_count:
        return False
    cell_count = len(spec.cell_positions)
    if cell_count <= 0:
        return False
    idx = action_id - spec.offsets.place
    tower_type = idx // cell_count
    return tower_type == red_index


def _scan_policy_for_seed(
    agent: PPOAgent,
    map_name: str,
    spec: ActionSpaceSpec,
    *,
    max_towers: int,
    slot_size: int,
    device: torch.device,
    num_envs: int,
    seed_cursor: int,
    max_batches: int,
    max_build_actions: int,
    max_wave_ticks: int,
) -> tuple[int | None, int]:
    red_index = -1
    for idx, kind in enumerate(spec.tower_kinds):
        if kind == "red_spammer":
            red_index = idx
            break
    if red_index < 0:
        return None, seed_cursor

    vec = VectorizedEnv(
        num_envs=num_envs,
        env_kwargs={"max_build_actions": max_build_actions, "max_wave_ticks": max_wave_ticks},
    )
    try:
        for _ in range(max_batches):
            seeds = list(range(seed_cursor, seed_cursor + num_envs))
            seed_cursor += num_envs
            obs_list = vec.reset(map_paths=map_name, seeds=seeds)
            mask_list = vec.get_action_masks()
            done_flags = [False] * num_envs
            with torch.no_grad():
                while not all(done_flags):
                    obs_tensor = batch_to_tensor(
                        obs_list,
                        max_towers=max_towers,
                        slot_size=slot_size,
                        device=device,
                    )
                    mask_tensor = _to_mask_tensor(mask_list, device=device)
                    actions, _, _ = agent.act(obs_tensor, mask_tensor, deterministic=True)
                    action_ids = [int(value) for value in actions.detach().cpu().tolist()]
                    for env_idx, action_id in enumerate(action_ids):
                        if done_flags[env_idx]:
                            continue
                        if _is_red_spammer_action(action_id, spec, red_index):
                            return seeds[env_idx], seed_cursor
                    results = vec.step(action_ids)
                    for env_idx, result in enumerate(results):
                        obs_list[env_idx] = result.obs
                        mask_list[env_idx] = result.info.get("action_mask", mask_list[env_idx])
                        if result.done:
                            done_flags[env_idx] = True
    finally:
        vec.close()
    return None, seed_cursor


def _run_replay_episode(
    agent: PPOAgent,
    map_name: str,
    seed: int,
    *,
    device: torch.device,
    max_towers: int,
    slot_size: int,
    max_build_actions: int,
    max_wave_ticks: int,
) -> Replay:
    env = VectorTDEventEnv(max_build_actions=max_build_actions, max_wave_ticks=max_wave_ticks)
    obs, _ = env.reset(seed=seed, options={"map_path": map_name})
    state_checks: list[dict] = []
    done = False
    with torch.no_grad():
        while not done:
            mask = env.action_masks()
            obs_tensor = batch_to_tensor([obs], max_towers=max_towers, slot_size=slot_size, device=device)
            mask_tensor = _to_mask_tensor([mask], device=device)
            action, _, _ = agent.act(obs_tensor, mask_tensor, deterministic=True)
            action_id = int(action.item())
            pre_check = None
            if (
                env.action_spec is not None
                and env.engine is not None
                and action_id == env.action_spec.offsets.start_wave
            ):
                pre_check = build_state_check(
                    env.engine.state,
                    env.map_data,
                    env.action_spec,
                    wave_ticks=0,
                )
            obs, _, terminated, truncated, info = env.step(action_id)
            done = terminated or truncated
            if pre_check is not None and "wave_ticks" in info:
                wave_idx = max(0, len(env.episode_actions) - 1)
                post_check = build_state_check(
                    env.engine.state,
                    env.map_data,
                    env.action_spec,
                    wave_ticks=int(info.get("wave_ticks", 0) or 0),
                )
                state_checks.append(
                    {
                        "wave_index": wave_idx,
                        "pre": pre_check,
                        "post": post_check,
                    }
                )
    state = env.engine.state if env.engine is not None else None
    summary = {
        "bank": int(getattr(state, "bank", 0)) if state is not None else 0,
        "lives": int(getattr(state, "lives", 0)) if state is not None else 0,
        "score": int(getattr(state, "score", 0)) if state is not None else 0,
        "wave": int(getattr(state, "level", 0)) if state is not None else 0,
        "game_over": bool(getattr(state, "game_over", False)) if state is not None else False,
        "game_won": bool(getattr(state, "game_won", False)) if state is not None else False,
    }
    return Replay(
        map_path=env.map_path or map_name,
        seed=int(seed),
        waves=list(env.episode_actions),
        state_hashes=state_checks,
        final_summary=summary,
    )


def _compare_replays(base: Replay, other: Replay) -> list[str]:
    diffs: list[str] = []
    if base.final_summary != other.final_summary:
        diffs.append(f"summary_mismatch base={base.final_summary} other={other.final_summary}")
    if len(base.waves) != len(other.waves):
        diffs.append(f"wave_count_mismatch base={len(base.waves)} other={len(other.waves)}")
    else:
        for wave_idx, (wave_a, wave_b) in enumerate(zip(base.waves, other.waves)):
            if len(wave_a) != len(wave_b):
                diffs.append(
                    f"wave_len_mismatch wave={wave_idx} base={len(wave_a)} other={len(wave_b)}"
                )
                break
            for action_idx, (act_a, act_b) in enumerate(zip(wave_a, wave_b)):
                if act_a != act_b:
                    diffs.append(
                        "action_mismatch wave={} action={} base={} other={}".format(
                            wave_idx,
                            action_idx,
                            action_to_dict(act_a),
                            action_to_dict(act_b),
                        )
                    )
                    break
            if diffs:
                break
    hashes_a = base.state_hashes or []
    hashes_b = other.state_hashes or []
    if len(hashes_a) != len(hashes_b):
        diffs.append(f"state_hash_len_mismatch base={len(hashes_a)} other={len(hashes_b)}")
    else:
        for idx, (entry_a, entry_b) in enumerate(zip(hashes_a, hashes_b)):
            hash_a = entry_a.get("post", {}).get("hash") if isinstance(entry_a, dict) else None
            hash_b = entry_b.get("post", {}).get("hash") if isinstance(entry_b, dict) else None
            if hash_a != hash_b:
                diffs.append(f"state_hash_mismatch index={idx} base={hash_a} other={hash_b}")
                break
    return diffs


def _policy_id(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    return rel.as_posix().replace("/", "__").replace(".pt", "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-root", default=None)
    ap.add_argument("--min-step", type=int, default=1_000_000)
    ap.add_argument("--target-count", type=int, default=16)
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--max-batches-per-policy", type=int, default=12)
    ap.add_argument("--seed-start", type=int, default=1000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-build-actions", type=int, default=100)
    ap.add_argument("--max-wave-ticks", type=int, default=20000)
    ap.add_argument("--replay-count", type=int, default=16)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    root_dir = _resolve_root()
    checkpoint_root = Path(args.checkpoint_root) if args.checkpoint_root else root_dir / "runs" / "ppo"
    output_dir = Path(args.output_dir) if args.output_dir else root_dir / "runs" / "ppo" / "red_spammer_scan"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    torch.set_num_threads(max(1, (os.cpu_count() or 1) - 1))

    checkpoints = _list_checkpoints(checkpoint_root, min_step=args.min_step)
    print(f"checkpoint_root={checkpoint_root} candidates={len(checkpoints)} min_step={args.min_step}")
    if not checkpoints:
        return 1

    selections: list[dict[str, Any]] = []
    seed_cursor = int(args.seed_start)

    for entry in checkpoints:
        if len(selections) >= args.target_count:
            break
        print(f"policy_scan start path={entry.path} step={entry.step}")
        agent, payload = _load_agent(entry.path, device=device)
        map_name = str(payload.get("map", "switchback"))
        probe_env, spec, obs_dim, action_dim, max_towers, slot_size = _probe_env(
            map_name,
            max_build_actions=args.max_build_actions,
            max_wave_ticks=args.max_wave_ticks,
        )
        probe_env.close()
        if int(payload.get("obs_dim", -1)) != obs_dim or int(payload.get("action_dim", -1)) != action_dim:
            print(
                "policy_scan skip obs/action mismatch path={} obs_dim={} action_dim={} expected_obs={} expected_action={}".format(
                    entry.path,
                    payload.get("obs_dim"),
                    payload.get("action_dim"),
                    obs_dim,
                    action_dim,
                )
            )
            continue
        seed, seed_cursor = _scan_policy_for_seed(
            agent,
            map_name,
            spec,
            max_towers=max_towers,
            slot_size=slot_size,
            device=device,
            num_envs=args.num_envs,
            seed_cursor=seed_cursor,
            max_batches=args.max_batches_per_policy,
            max_build_actions=args.max_build_actions,
            max_wave_ticks=args.max_wave_ticks,
        )
        if seed is None:
            print(f"policy_scan miss path={entry.path} step={entry.step}")
            continue
        selection = {
            "policy_path": str(entry.path),
            "policy_step": entry.step,
            "map": map_name,
            "seed": int(seed),
        }
        selections.append(selection)
        print(f"policy_scan hit path={entry.path} step={entry.step} seed={seed}")

    selections_path = output_dir / "selections.json"
    selections_path.write_text(json.dumps(selections, indent=2, sort_keys=True), encoding="utf-8")
    print(f"selection_count={len(selections)} selections_path={selections_path}")

    if not selections:
        return 2

    non_deterministic: list[dict[str, Any]] = []
    for selection in selections:
        policy_path = Path(selection["policy_path"])
        policy_step = selection["policy_step"]
        map_name = selection["map"]
        seed = int(selection["seed"])
        agent, payload = _load_agent(policy_path, device=device)
        probe_env, spec, obs_dim, action_dim, max_towers, slot_size = _probe_env(
            map_name,
            max_build_actions=args.max_build_actions,
            max_wave_ticks=args.max_wave_ticks,
        )
        probe_env.close()
        if int(payload.get("obs_dim", -1)) != obs_dim or int(payload.get("action_dim", -1)) != action_dim:
            print(
                "policy_replay skip obs/action mismatch path={} obs_dim={} action_dim={} expected_obs={} expected_action={}".format(
                    policy_path,
                    payload.get("obs_dim"),
                    payload.get("action_dim"),
                    obs_dim,
                    action_dim,
                )
            )
            continue

        policy_id = _policy_id(policy_path, root_dir)
        replay_dir = output_dir / policy_id / f"seed_{seed}"
        replay_dir.mkdir(parents=True, exist_ok=True)
        replays: list[Replay] = []

        for idx in range(args.replay_count):
            replay = _run_replay_episode(
                agent,
                map_name,
                seed,
                device=device,
                max_towers=max_towers,
                slot_size=slot_size,
                max_build_actions=args.max_build_actions,
                max_wave_ticks=args.max_wave_ticks,
            )
            replay_path = replay_dir / f"replay_{idx:02d}.json"
            save_replay(replay_path, replay)
            replays.append(replay)
        base = replays[0]
        diffs: list[str] = []
        for idx, replay in enumerate(replays[1:], start=1):
            diff = _compare_replays(base, replay)
            if diff:
                diffs.append(f"run={idx} " + " | ".join(diff))
        if diffs:
            non_deterministic.append(
                {
                    "policy_path": str(policy_path),
                    "policy_step": policy_step,
                    "map": map_name,
                    "seed": seed,
                    "diffs": diffs,
                }
            )
            print(f"determinism_mismatch policy={policy_path} seed={seed}")
        else:
            print(f"determinism_ok policy={policy_path} seed={seed}")

    report = {
        "selection_count": len(selections),
        "selected": selections,
        "non_deterministic": non_deterministic,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"report_path={report_path}")
    if non_deterministic:
        print(f"non_deterministic_count={len(non_deterministic)}")
    else:
        print("non_deterministic_count=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
