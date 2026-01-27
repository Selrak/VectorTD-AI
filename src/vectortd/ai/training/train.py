from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Iterable

from vectortd.ai.actions import Noop, StartWave
from vectortd.ai.env import VectorTDEventEnv
from vectortd.ai.policies.baseline import make_policy
from vectortd.ai.rewards import RewardConfig
from vectortd.ai.training.pause import PauseController, normalize_pause_key
from vectortd.io.replay import Replay, build_state_check, save_replay


logger = logging.getLogger(__name__)


@dataclass
class Transition:
    obs: Any
    action: object
    reward: float
    done: bool
    info: dict


@dataclass
class EpisodeSummary:
    episode: int
    map_name: str
    seed: int
    steps: int
    waves: int
    total_reward: float
    score: int
    lives: int
    bank: int
    game_over: bool
    game_won: bool
    stop_reason: str


def _expand_list(values: Iterable[str] | None) -> list[str]:
    if not values:
        return []
    expanded: list[str] = []
    for value in values:
        parts = [part.strip() for part in value.split(",") if part.strip()]
        expanded.extend(parts)
    return expanded


def _is_start_wave(action, env: VectorTDEventEnv) -> bool:
    if isinstance(action, StartWave):
        return True
    if isinstance(action, int) and env.action_spec is not None:
        return action == env.action_spec.offsets.start_wave
    return False


def _resolve_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _checkpoint_path(checkpoint_dir: Path, episode: int) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"checkpoint_{episode:06d}.json"


def _replay_path(replay_dir: Path, episode: int) -> Path:
    replay_dir.mkdir(parents=True, exist_ok=True)
    return replay_dir / f"replay_{episode:06d}.json"


def _save_replay(
    env: VectorTDEventEnv,
    replay_dir: Path,
    episode: int,
    *,
    state_checks: list[dict] | None = None,
) -> Path | None:
    if env.engine is None:
        return None
    if not env.episode_actions:
        logger.warning("episode=%s replay skipped (no completed waves)", episode)
        return None
    if env.map_path is None:
        logger.warning("episode=%s replay skipped (missing map path)", episode)
        return None
    seed = env.episode_seed if env.episode_seed is not None else 1
    state = env.engine.state
    summary = {
        "bank": int(getattr(state, "bank", 0)),
        "lives": int(getattr(state, "lives", 0)),
        "score": int(getattr(state, "score", 0)),
        "wave": int(getattr(state, "level", 0)),
        "game_over": bool(getattr(state, "game_over", False)),
        "game_won": bool(getattr(state, "game_won", False)),
    }
    replay = Replay(
        map_path=env.map_path,
        seed=int(seed),
        waves=list(env.episode_actions),
        state_hashes=state_checks,
        final_summary=summary,
    )
    path = _replay_path(replay_dir, episode)
    save_replay(path, replay)
    logger.info("episode=%s replay_path=%s", episode, path)
    return path


def _save_checkpoint(
    checkpoint_dir: Path,
    episode: int,
    summary: EpisodeSummary,
    *,
    policy_name: str,
    config: dict,
) -> Path:
    payload = {
        "episode": episode,
        "policy": policy_name,
        "config": config,
        "summary": {
            "map": summary.map_name,
            "seed": summary.seed,
            "steps": summary.steps,
            "waves": summary.waves,
            "total_reward": summary.total_reward,
            "score": summary.score,
            "lives": summary.lives,
            "bank": summary.bank,
            "game_over": summary.game_over,
            "game_won": summary.game_won,
            "stop_reason": summary.stop_reason,
        },
    }
    path = _checkpoint_path(checkpoint_dir, episode)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _run_episode(
    *,
    episode: int,
    env: VectorTDEventEnv,
    policy,
    map_name: str,
    seed: int,
    max_steps: int,
    max_waves: int | None,
    collect_rollout: bool,
    collect_state_checks: bool = False,
    pause: PauseController | None = None,
) -> tuple[EpisodeSummary, list[Transition], list[dict] | None]:
    env.reset(seed=seed, options={"map_path": map_name})
    policy.reset(env)

    steps = 0
    waves = 0
    done = False
    total_reward = 0.0
    obs_dict: dict | None = None
    stop_reason = "max_steps"
    rollout: list[Transition] = []
    state_checks: list[dict] | None = [] if collect_state_checks else None

    while not done and steps < max_steps:
        if pause is not None:
            pause.wait_if_paused()
        if max_waves is not None and waves >= max_waves:
            stop_reason = "max_waves"
            break
        action = policy.next_action(env)
        if action is None:
            action = Noop()
        is_start_wave = _is_start_wave(action, env)
        pre_check = None
        if (
            is_start_wave
            and state_checks is not None
            and env.engine is not None
            and env.action_spec is not None
        ):
            pre_check = build_state_check(
                env.engine.state,
                env.map_data,
                env.action_spec,
                wave_ticks=0,
            )
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        obs_dict = env.last_obs
        steps += 1
        total_reward += float(reward)
        did_wave = "wave_ticks" in info
        if did_wave:
            waves += 1
            if pre_check is not None and env.engine is not None and env.action_spec is not None:
                wave_idx = max(0, len(env.episode_actions) - 1)
                wave_ticks = int(info.get("wave_ticks", 0) or 0)
                post_check = build_state_check(
                    env.engine.state,
                    env.map_data,
                    env.action_spec,
                    wave_ticks=wave_ticks,
                )
                state_checks.append(
                    {
                        "wave_index": wave_idx,
                        "pre": pre_check,
                        "post": post_check,
                    }
                )
        if collect_rollout:
            rollout.append(
                Transition(
                    obs=obs,
                    action=action,
                    reward=float(reward),
                    done=bool(done),
                    info=dict(info),
                )
            )

    if done:
        stop_reason = "done"
    if obs_dict is None:
        obs_dict = {}
    if env.engine is None:
        game_over = False
        game_won = False
    else:
        state = env.engine.state
        game_over = bool(getattr(state, "game_over", False))
        game_won = bool(getattr(state, "game_won", False))
    summary = EpisodeSummary(
        episode=episode,
        map_name=map_name,
        seed=seed,
        steps=steps,
        waves=waves,
        total_reward=total_reward,
        score=int(obs_dict.get("score", 0) or 0),
        lives=int(obs_dict.get("lives", 0) or 0),
        bank=int(obs_dict.get("bank", 0) or 0),
        game_over=game_over,
        game_won=game_won,
        stop_reason=stop_reason,
    )
    return summary, rollout, state_checks


def _run_eval(
    *,
    policy_name: str,
    eval_maps: list[str],
    eval_seeds: list[int],
    max_steps: int,
    max_waves: int | None,
    max_build_actions: int,
    reward_config: RewardConfig,
    pause: PauseController | None = None,
) -> dict:
    results: list[dict] = []
    total_score = 0.0
    total_waves = 0.0
    total_lives = 0.0
    for map_name in eval_maps:
        for seed in eval_seeds:
            env = VectorTDEventEnv(
                max_build_actions=max_build_actions,
                reward_config=reward_config,
            )
            policy = make_policy(policy_name, seed=seed, verbose=False)
            summary, _, _ = _run_episode(
                episode=0,
                env=env,
                policy=policy,
                map_name=map_name,
                seed=seed,
                max_steps=max_steps,
                max_waves=max_waves,
                collect_rollout=False,
                collect_state_checks=False,
                pause=pause,
            )
            results.append(
                {
                    "map": map_name,
                    "seed": seed,
                    "score": summary.score,
                    "waves": summary.waves,
                    "lives": summary.lives,
                    "game_over": summary.game_over,
                    "game_won": summary.game_won,
                }
            )
            total_score += summary.score
            total_waves += summary.waves
            total_lives += summary.lives
    count = max(1, len(results))
    return {
        "count": len(results),
        "mean_score": total_score / count,
        "mean_waves": total_waves / count,
        "mean_lives": total_lives / count,
        "cases": results,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["heuristic", "random"], default="heuristic")
    ap.add_argument("--map", action="append", default=None)
    ap.add_argument("--seed", action="append", default=None)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--max-waves", type=int, default=None)
    ap.add_argument("--max-build-actions", type=int, default=100)
    ap.add_argument("--checkpoint-every", type=int, default=5)
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--save-replay-every", type=int, default=0)
    ap.add_argument("--replay-dir", default=None)
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--eval-map", action="append", default=None)
    ap.add_argument("--eval-seed", action="append", default=None)
    ap.add_argument("--life-loss-penalty", type=float, default=0.5)
    ap.add_argument("--wave-total-reward", type=float, default=8.0)
    ap.add_argument("--terminal-win-reward", type=float, default=2.0)
    ap.add_argument("--terminal-loss-reward", type=float, default=-2.0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--pause", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--pause-key", default="space")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(message)s",
    )

    maps = _expand_list(args.map) or ["switchback"]
    seeds = [int(seed) for seed in _expand_list(args.seed)] if args.seed else [123]
    eval_maps = _expand_list(args.eval_map) or maps
    eval_seeds = [int(seed) for seed in _expand_list(args.eval_seed)] if args.eval_seed else seeds

    root_dir = _resolve_root()
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else root_dir / "runs" / "training"
    replay_dir = Path(args.replay_dir) if args.replay_dir else root_dir / "runs" / "training" / "replays"

    reward_config = RewardConfig(
        life_loss_penalty=args.life_loss_penalty,
        wave_total_reward=args.wave_total_reward,
        terminal_win_reward=args.terminal_win_reward,
        terminal_loss_reward=args.terminal_loss_reward,
    )

    env = VectorTDEventEnv(
        max_build_actions=args.max_build_actions,
        reward_config=reward_config,
    )
    policy = make_policy(args.policy, seed=seeds[0], verbose=args.verbose)
    pause = PauseController(enabled=args.pause, key=normalize_pause_key(args.pause_key))
    pause.start()

    config_payload = {
        "policy": args.policy,
        "maps": maps,
        "seeds": seeds,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "max_waves": args.max_waves,
        "max_build_actions": args.max_build_actions,
        "reward_config": {
            "life_loss_penalty": reward_config.life_loss_penalty,
            "wave_total_reward": reward_config.wave_total_reward,
            "terminal_win_reward": reward_config.terminal_win_reward,
            "terminal_loss_reward": reward_config.terminal_loss_reward,
            "max_waves": reward_config.max_waves,
        },
    }

    try:
        for episode_idx in range(1, args.episodes + 1):
            pause.wait_if_paused()
            map_name = maps[(episode_idx - 1) % len(maps)]
            seed = seeds[(episode_idx - 1) % len(seeds)]
            save_replay_now = args.save_replay_every > 0 and (episode_idx % args.save_replay_every) == 0
            summary, rollout, state_checks = _run_episode(
                episode=episode_idx,
                env=env,
                policy=policy,
                map_name=map_name,
                seed=seed,
                max_steps=args.max_steps,
                max_waves=args.max_waves,
                collect_rollout=True,
                collect_state_checks=save_replay_now,
                pause=pause,
            )
            if hasattr(policy, "update"):
                try:
                    policy.update(rollout)
                except TypeError:
                    pass
            logger.info(
                "episode=%s map=%s seed=%s steps=%s waves=%s reward=%.2f score=%s lives=%s bank=%s stop=%s",
                summary.episode,
                summary.map_name,
                summary.seed,
                summary.steps,
                summary.waves,
                summary.total_reward,
                summary.score,
                summary.lives,
                summary.bank,
                summary.stop_reason,
            )

            if args.checkpoint_every > 0 and (episode_idx % args.checkpoint_every) == 0:
                path = _save_checkpoint(
                    checkpoint_dir,
                    episode_idx,
                    summary,
                    policy_name=args.policy,
                    config=config_payload,
                )
                logger.info("checkpoint=%s", path)

            if save_replay_now:
                _save_replay(env, replay_dir, episode_idx, state_checks=state_checks)

            if args.eval_every > 0 and (episode_idx % args.eval_every) == 0:
                eval_summary = _run_eval(
                    policy_name=args.policy,
                    eval_maps=eval_maps,
                    eval_seeds=eval_seeds,
                    max_steps=args.max_steps,
                    max_waves=args.max_waves,
                    max_build_actions=args.max_build_actions,
                    reward_config=reward_config,
                    pause=pause,
                )
                logger.info(
                    "eval episode=%s count=%s mean_score=%.2f mean_waves=%.2f mean_lives=%.2f",
                    episode_idx,
                    eval_summary["count"],
                    eval_summary["mean_score"],
                    eval_summary["mean_waves"],
                    eval_summary["mean_lives"],
                )
    finally:
        pause.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
