from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

try:
    import torch
except Exception as exc:  # pragma: no cover - import guard
    raise SystemExit("PyTorch is required for PPO replay evaluation.") from exc

from vectortd.ai.env import VectorTDEventEnv
from vectortd.ai.training.ppo import PPOAgent, PPOConfig, SCALAR_KEYS, batch_to_tensor
from vectortd.io.replay import Replay, build_state_check, load_replay, save_replay


@dataclass(frozen=True, slots=True)
class Selection:
    policy_path: str
    policy_step: int
    map: str
    seed: int


def _resolve_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parents[3]


def _policy_id(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    return rel.as_posix().replace("/", "__").replace(".pt", "")


def _config_from_checkpoint(payload: dict[str, Any]) -> PPOConfig:
    defaults = PPOConfig().__dict__
    overrides = payload.get("config") or {}
    merged = dict(defaults)
    merged.update({k: v for k, v in overrides.items() if k in defaults})
    return PPOConfig(**merged)


def _probe_dims(
    map_name: str,
    *,
    max_build_actions: int,
    max_wave_ticks: int,
) -> tuple[int, int, int, int]:
    env = VectorTDEventEnv(max_build_actions=max_build_actions, max_wave_ticks=max_wave_ticks)
    env.reset(seed=1, options={"map_path": map_name})
    if env.action_spec is None:
        raise RuntimeError("Missing action spec after reset")
    obs_dict = env.last_obs or {}
    max_towers = len(obs_dict.get("tower_slots", []) or [])
    slot_size = len(obs_dict.get("tower_slot_features", []) or [])
    obs_dim = len(SCALAR_KEYS) + max_towers * slot_size
    action_dim = env.action_spec.num_actions
    env.close()
    return obs_dim, action_dim, max_towers, slot_size


def _to_mask_tensor(masks: list[Any], device: torch.device) -> torch.Tensor:
    try:
        import numpy as np  # type: ignore

        mask_array = np.asarray(masks, dtype=bool)
        return torch.as_tensor(mask_array, dtype=torch.bool, device=device)
    except Exception:
        return torch.tensor(masks, dtype=torch.bool, device=device)


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


def _replay_task(payload: dict[str, Any]) -> str:
    checkpoint_path = Path(payload["checkpoint_path"])
    map_name = payload["map"]
    seed = int(payload["seed"])
    replay_idx = int(payload["replay_idx"])
    output_dir = Path(payload["output_dir"])
    device = torch.device(payload["device"])
    max_build_actions = int(payload["max_build_actions"])
    max_wave_ticks = int(payload["max_wave_ticks"])

    torch.set_num_threads(1)
    torch.manual_seed(0)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    ckpt = torch.load(checkpoint_path, map_location=device)
    config = _config_from_checkpoint(ckpt)
    agent = PPOAgent(
        obs_dim=int(ckpt["obs_dim"]),
        action_dim=int(ckpt["action_dim"]),
        config=config,
        device=device,
    )
    agent.model.load_state_dict(ckpt["state_dict"])
    agent.model.eval()

    obs_dim, action_dim, max_towers, slot_size = _probe_dims(
        map_name,
        max_build_actions=max_build_actions,
        max_wave_ticks=max_wave_ticks,
    )
    if int(ckpt.get("obs_dim", -1)) != obs_dim or int(ckpt.get("action_dim", -1)) != action_dim:
        raise RuntimeError("Checkpoint dims mismatch for replay task")

    replay = _run_replay_episode(
        agent,
        map_name,
        seed,
        device=device,
        max_towers=max_towers,
        slot_size=slot_size,
        max_build_actions=max_build_actions,
        max_wave_ticks=max_wave_ticks,
    )
    replay_path = output_dir / f"replay_{replay_idx:02d}.json"
    save_replay(replay_path, replay)
    return str(replay_path)


def _diff_action_sequences(base: Replay, other: Replay) -> dict[str, Any] | None:
    if len(base.waves) != len(other.waves):
        return {
            "wave_count_base": len(base.waves),
            "wave_count_other": len(other.waves),
        }
    for wave_idx, (wave_a, wave_b) in enumerate(zip(base.waves, other.waves)):
        if len(wave_a) != len(wave_b):
            return {
                "wave": wave_idx,
                "wave_len_base": len(wave_a),
                "wave_len_other": len(wave_b),
            }
        for action_idx, (act_a, act_b) in enumerate(zip(wave_a, wave_b)):
            if act_a != act_b:
                return {
                    "wave": wave_idx,
                    "action": action_idx,
                    "action_base": act_a,
                    "action_other": act_b,
                }
    return None


def _diff_state_checks(base: Replay, other: Replay) -> dict[str, Any] | None:
    hashes_a = base.state_hashes or []
    hashes_b = other.state_hashes or []
    if len(hashes_a) != len(hashes_b):
        return {
            "state_hash_len_base": len(hashes_a),
            "state_hash_len_other": len(hashes_b),
        }
    for idx, (entry_a, entry_b) in enumerate(zip(hashes_a, hashes_b)):
        post_a = entry_a.get("post", {}) if isinstance(entry_a, dict) else {}
        post_b = entry_b.get("post", {}) if isinstance(entry_b, dict) else {}
        if post_a.get("hash") == post_b.get("hash"):
            continue
        section_a = post_a.get("section_hashes", {}) or {}
        section_b = post_b.get("section_hashes", {}) or {}
        ordering_a = post_a.get("ordering_hashes", {}) or {}
        ordering_b = post_b.get("ordering_hashes", {}) or {}
        return {
            "wave_index": entry_a.get("wave_index", idx),
            "rng_state_base": post_a.get("rng_state"),
            "rng_state_other": post_b.get("rng_state"),
            "rng_calls_base": post_a.get("rng_calls"),
            "rng_calls_other": post_b.get("rng_calls"),
            "wave_ticks_base": post_a.get("wave_ticks"),
            "wave_ticks_other": post_b.get("wave_ticks"),
            "summary_base": post_a.get("summary"),
            "summary_other": post_b.get("summary"),
            "section_diff": [key for key in section_a if section_a.get(key) != section_b.get(key)],
            "ordering_diff": [key for key in ordering_a if ordering_a.get(key) != ordering_b.get(key)],
        }
    return None


def _analyze_replays(replay_paths: list[Path]) -> dict[str, Any]:
    if not replay_paths:
        return {"error": "no_replays"}
    replays = [load_replay(path) for path in replay_paths]
    base = replays[0]
    diffs = []
    for idx, replay in enumerate(replays[1:], start=1):
        diff: dict[str, Any] = {"run": idx}
        if base.final_summary != replay.final_summary:
            diff["summary_diff"] = {
                "base": base.final_summary,
                "other": replay.final_summary,
            }
        action_diff = _diff_action_sequences(base, replay)
        if action_diff:
            diff["action_diff"] = action_diff
        state_diff = _diff_state_checks(base, replay)
        if state_diff:
            diff["state_diff"] = state_diff
        if len(diff) > 1:
            diffs.append(diff)
    return {"deterministic": not diffs, "diffs": diffs}


def _load_selections(path: Path) -> list[Selection]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    selections = []
    for entry in raw:
        selections.append(
            Selection(
                policy_path=str(entry["policy_path"]),
                policy_step=int(entry["policy_step"]),
                map=str(entry["map"]),
                seed=int(entry["seed"]),
            )
        )
    return selections


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selections", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--replay-count", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-build-actions", type=int, default=100)
    ap.add_argument("--max-wave-ticks", type=int, default=20000)
    args = ap.parse_args()

    root_dir = _resolve_root()
    selections_path = Path(args.selections) if args.selections else root_dir / "runs" / "ppo" / "red_spammer_scan" / "selections.json"
    output_dir = Path(args.output_dir) if args.output_dir else root_dir / "runs" / "ppo" / "red_spammer_scan_rerun"
    output_dir.mkdir(parents=True, exist_ok=True)

    selections = _load_selections(selections_path)
    if not selections:
        print("No selections found")
        return 1

    report: dict[str, Any] = {
        "selection_count": len(selections),
        "selections": [
            {
                "policy_path": selection.policy_path,
                "policy_step": selection.policy_step,
                "map": selection.map,
                "seed": selection.seed,
            }
            for selection in selections
        ],
        "policies": [],
    }

    for selection in selections:
        policy_path = Path(selection.policy_path)
        policy_id = _policy_id(policy_path, root_dir)
        replay_dir = output_dir / policy_id / f"seed_{selection.seed}"
        replay_dir.mkdir(parents=True, exist_ok=True)
        print(f"policy_start path={policy_path} seed={selection.seed}")

        tasks = []
        for idx in range(args.replay_count):
            tasks.append(
                {
                    "checkpoint_path": str(policy_path),
                    "map": selection.map,
                    "seed": selection.seed,
                    "replay_idx": idx,
                    "output_dir": str(replay_dir),
                    "device": args.device,
                    "max_build_actions": args.max_build_actions,
                    "max_wave_ticks": args.max_wave_ticks,
                }
            )

        ctx = torch.multiprocessing.get_context("spawn")
        with ctx.Pool(processes=max(1, args.workers)) as pool:
            for replay_path in pool.imap_unordered(_replay_task, tasks):
                print(f"replay_saved path={replay_path}")

        replay_paths = [replay_dir / f"replay_{idx:02d}.json" for idx in range(args.replay_count)]
        analysis = _analyze_replays(replay_paths)
        report_entry = {
            "policy_path": str(policy_path),
            "policy_step": selection.policy_step,
            "map": selection.map,
            "seed": selection.seed,
            "analysis": analysis,
        }
        report["policies"].append(report_entry)
        status = "deterministic" if analysis.get("deterministic") else "nondeterministic"
        print(f"policy_done path={policy_path} status={status}")

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"report_path={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
