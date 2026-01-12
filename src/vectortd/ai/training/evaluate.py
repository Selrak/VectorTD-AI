from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import statistics
from typing import Iterable

from vectortd.ai.actions import Noop, StartWave
from vectortd.ai.env import VectorTDEventEnv
from vectortd.ai.policies.baseline import make_policy


logger = logging.getLogger(__name__)


POLICY_CHOICES = ["heuristic", "random"]


@dataclass
class EpisodeResult:
    policy: str
    map_name: str
    seed: int
    steps: int
    waves: int
    score: int
    lives: int
    bank: int
    final_wave: int
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


def run_episode(
    *,
    policy_name: str,
    map_name: str,
    seed: int,
    max_steps: int,
    max_waves: int | None,
    max_build_actions: int,
    verbose: bool,
) -> EpisodeResult:
    env = VectorTDEventEnv(max_build_actions=max_build_actions)
    env.reset(map_path=map_name, seed=seed)
    policy = make_policy(policy_name, seed=seed, verbose=verbose)
    policy.reset(env)

    steps = 0
    waves = 0
    done = False
    obs: dict | None = None
    stop_reason = "max_steps"

    while not done and steps < max_steps:
        if max_waves is not None and waves >= max_waves:
            stop_reason = "max_waves"
            break
        action = policy.next_action(env)
        if action is None:
            action = Noop()
        is_start_wave = _is_start_wave(action, env)
        obs, _, done, info = env.step(action)
        steps += 1
        if is_start_wave:
            waves += 1
            if verbose:
                logger.info(
                    "policy=%s map=%s seed=%s wave_done=%s ticks=%s lives=%s score=%s bank=%s",
                    policy_name,
                    map_name,
                    seed,
                    waves,
                    info.get("wave_ticks"),
                    obs.get("lives"),
                    obs.get("score"),
                    obs.get("bank"),
                )

    if done:
        stop_reason = "done"
    if obs is None:
        obs = {}
    if env.engine is None:
        game_over = False
        game_won = False
        final_wave = int(obs.get("wave", 0) or 0)
    else:
        state = env.engine.state
        game_over = bool(getattr(state, "game_over", False))
        game_won = bool(getattr(state, "game_won", False))
        final_wave = int(getattr(state, "level", 0))
    return EpisodeResult(
        policy=policy_name,
        map_name=map_name,
        seed=seed,
        steps=steps,
        waves=waves,
        score=int(obs.get("score", 0) or 0),
        lives=int(obs.get("lives", 0) or 0),
        bank=int(obs.get("bank", 0) or 0),
        final_wave=final_wave,
        game_over=game_over,
        game_won=game_won,
        stop_reason=stop_reason,
    )


def _summary_stats(results: list[EpisodeResult]) -> dict[str, float]:
    scores = [res.score for res in results]
    waves = [res.waves for res in results]
    lives = [res.lives for res in results]
    wins = sum(1 for res in results if res.game_won)
    losses = sum(1 for res in results if res.game_over and not res.game_won)
    total = len(results)
    return {
        "mean_score": statistics.mean(scores) if scores else 0.0,
        "mean_waves": statistics.mean(waves) if waves else 0.0,
        "mean_lives": statistics.mean(lives) if lives else 0.0,
        "min_score": min(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
        "wins": float(wins),
        "losses": float(losses),
        "win_rate": (wins / total) if total else 0.0,
        "loss_rate": (losses / total) if total else 0.0,
    }


def _print_summary(
    *,
    policy_a: str,
    policy_b: str,
    results_a: list[EpisodeResult],
    results_b: list[EpisodeResult],
    show_runs: bool,
) -> None:
    stats_a = _summary_stats(results_a)
    stats_b = _summary_stats(results_b)
    print(f"policy_a={policy_a} policy_b={policy_b}")
    cases = len(results_a)
    if cases == 1:
        print("cases=1 (mean == value)")
    else:
        print(f"cases={cases}")
    print(
        "A stats: mean_score={mean_score:.1f} mean_waves={mean_waves:.1f} "
        "mean_lives={mean_lives:.1f} min_score={min_score:.1f} max_score={max_score:.1f} "
        "wins={wins:.0f} losses={losses:.0f} win_rate={win_rate:.2f} loss_rate={loss_rate:.2f}".format(**stats_a)
    )
    print(
        "B stats: mean_score={mean_score:.1f} mean_waves={mean_waves:.1f} "
        "mean_lives={mean_lives:.1f} min_score={min_score:.1f} max_score={max_score:.1f} "
        "wins={wins:.0f} losses={losses:.0f} win_rate={win_rate:.2f} loss_rate={loss_rate:.2f}".format(**stats_b)
    )

    better_a = 0
    better_b = 0
    ties = 0
    diffs: list[int] = []
    for res_a, res_b in zip(results_a, results_b):
        diff = res_a.score - res_b.score
        diffs.append(diff)
        if diff > 0:
            better_a += 1
        elif diff < 0:
            better_b += 1
        else:
            ties += 1
    mean_diff = statistics.mean(diffs) if diffs else 0.0
    print(f"head_to_head_score: a_better={better_a} b_better={better_b} ties={ties} mean_diff={mean_diff:.1f}")

    if show_runs:
        for res_a, res_b in zip(results_a, results_b):
            a_outcome = "won" if res_a.game_won else "lost" if res_a.game_over else "unfinished"
            b_outcome = "won" if res_b.game_won else "lost" if res_b.game_over else "unfinished"
            print(
                "case map={map} seed={seed} "
                "A score={a_score} waves={a_waves} lives={a_lives} stop={a_stop} outcome={a_outcome} "
                "B score={b_score} waves={b_waves} lives={b_lives} stop={b_stop} outcome={b_outcome}".format(
                    map=res_a.map_name,
                    seed=res_a.seed,
                    a_score=res_a.score,
                    a_waves=res_a.waves,
                    a_lives=res_a.lives,
                    a_stop=res_a.stop_reason,
                    a_outcome=a_outcome,
                    b_score=res_b.score,
                    b_waves=res_b.waves,
                    b_lives=res_b.lives,
                    b_stop=res_b.stop_reason,
                    b_outcome=b_outcome,
                )
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-a", choices=POLICY_CHOICES, default="heuristic")
    ap.add_argument("--policy-b", choices=POLICY_CHOICES, default="random")
    ap.add_argument("--map", action="append", default=None)
    ap.add_argument("--seed", action="append", default=None)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--max-waves", type=int, default=None)
    ap.add_argument("--max-build-actions", type=int, default=100)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--show-runs", action="store_true")
    args = ap.parse_args()

    maps = _expand_list(args.map) or ["switchback"]
    seeds = [int(seed) for seed in _expand_list(args.seed)] if args.seed else [123]

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(message)s",
    )

    results_a: list[EpisodeResult] = []
    results_b: list[EpisodeResult] = []
    for map_name in maps:
        for seed in seeds:
            results_a.append(
                run_episode(
                    policy_name=args.policy_a,
                    map_name=map_name,
                    seed=seed,
                    max_steps=args.max_steps,
                    max_waves=args.max_waves,
                    max_build_actions=args.max_build_actions,
                    verbose=args.verbose,
                )
            )
            results_b.append(
                run_episode(
                    policy_name=args.policy_b,
                    map_name=map_name,
                    seed=seed,
                    max_steps=args.max_steps,
                    max_waves=args.max_waves,
                    max_build_actions=args.max_build_actions,
                    verbose=args.verbose,
                )
            )

    _print_summary(
        policy_a=args.policy_a,
        policy_b=args.policy_b,
        results_a=results_a,
        results_b=results_b,
        show_runs=args.show_runs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
