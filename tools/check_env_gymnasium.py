import argparse
import random

try:
    import gymnasium as gym
except Exception as exc:  # pragma: no cover - import guard
    raise SystemExit("gymnasium is required for env checks. Install gymnasium and retry.") from exc

from vectortd.ai.env import VectorTDEventEnv


def _choose_action(mask, rng: random.Random) -> int:
    valid = [idx for idx, allowed in enumerate(mask) if bool(allowed)]
    if not valid:
        raise RuntimeError("No valid actions available")
    return rng.choice(valid)


def _resolve_check_env():
    try:
        from gymnasium.utils.env_checker import check_env

        return check_env
    except Exception:
        pass
    try:
        from gymnasium.utils import env_checker

        return env_checker.check_env
    except Exception:
        pass
    try:
        import gym
        from gym.utils.env_checker import check_env

        return check_env
    except Exception as exc:
        raise SystemExit("gymnasium env_checker is unavailable; update gymnasium or install gym.") from exc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="switchback")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--max-build-actions", type=int, default=100)
    ap.add_argument("--max-wave-ticks", type=int, default=20_000)
    ap.add_argument("--action-space-kind", choices=["legacy", "discrete_k"], default="legacy")
    args = ap.parse_args()

    env = VectorTDEventEnv(
        default_map=args.map,
        action_space_kind=args.action_space_kind,
        max_build_actions=args.max_build_actions,
        max_wave_ticks=args.max_wave_ticks,
    )
    check_env = _resolve_check_env()
    check_env(env, skip_render_check=True)

    rng = random.Random(args.seed)
    obs, _ = env.reset(seed=args.seed, options={"map_path": args.map})
    for step_idx in range(args.steps):
        mask = env.action_masks()
        action = _choose_action(mask, rng)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if done:
            obs, _ = env.reset(seed=args.seed + step_idx + 1, options={"map_path": args.map})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
