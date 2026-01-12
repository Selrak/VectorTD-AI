from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

try:
    import torch
except Exception as exc:  # pragma: no cover - import guard
    raise SystemExit("PyTorch is required for PPO training. Install torch and retry.") from exc

from vectortd.ai.env import VectorTDEventEnv
from vectortd.ai.vectorized_env import VectorizedEnv
from vectortd.ai.training.ppo import (
    PPOAgent,
    PPOConfig,
    SCALAR_KEYS,
    batch_to_tensor,
    compute_gae,
)
from vectortd.io.replay import Replay, save_replay


def _resolve_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parents[3]


def _next_run_dir(root_dir: Path) -> Path:
    runs_root = root_dir / "runs" / "ppo"
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


def _checkpoint_path(checkpoint_dir: Path, step: int) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"ppo_checkpoint_{step:08d}.pt"


def _replay_path(replay_dir: Path, step: int) -> Path:
    replay_dir.mkdir(parents=True, exist_ok=True)
    return replay_dir / f"ppo_replay_{step:08d}.json"


def _train_log_path(root_dir: Path, map_name: str, run_id: str) -> Path:
    log_dir = root_dir / "runs" / "ppo"
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_map = map_name.replace(" ", "_")
    return log_dir / f"train_log_{safe_map}_{run_id}.csv"


def _eval_log_path(root_dir: Path, map_name: str, run_id: str) -> Path:
    eval_dir = root_dir / "runs" / "ppo"
    eval_dir.mkdir(parents=True, exist_ok=True)
    safe_map = map_name.replace(" ", "_")
    return eval_dir / f"eval_log_{safe_map}_{run_id}.csv"


def _append_train_log(path: Path, row: dict[str, float | int | str]) -> None:
    header = [
        "step",
        "map",
        "run_id",
        "update",
        "num_envs",
        "elapsed_sec",
        "policy_loss",
        "value_loss",
        "entropy",
        "mean_reward",
        "mean_return",
        "mean_value",
        "mean_advantage",
    ]
    write_header = not path.exists()
    with path.open("a", encoding="utf-8") as handle:
        if write_header:
            handle.write(",".join(header) + "\n")
        values = []
        for key in header:
            value = row.get(key, "")
            values.append(str(value))
        handle.write(",".join(values) + "\n")


def _append_eval_log(path: Path, row: dict[str, float | int | str]) -> None:
    header = [
        "step",
        "map",
        "run_id",
        "eval_episodes",
        "mean_score",
        "max_score",
        "mean_wave",
        "max_wave",
        "mean_return",
        "mean_lives",
        "win_rate",
        "loss_rate",
        "mean_actions",
        "mean_actions_per_wave",
        "mean_steps",
        "elapsed_sec",
    ]
    write_header = not path.exists()
    with path.open("a", encoding="utf-8") as handle:
        if write_header:
            handle.write(",".join(header) + "\n")
        values = []
        for key in header:
            value = row.get(key, "")
            values.append(str(value))
        handle.write(",".join(values) + "\n")


def _save_replay(env: VectorTDEventEnv, replay_dir: Path, step: int) -> Path | None:
    if env.engine is None:
        return None
    if not env.episode_actions:
        return None
    if env.map_path is None:
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
        state_hashes=None,
        final_summary=summary,
    )
    path = _replay_path(replay_dir, step)
    save_replay(path, replay)
    return path


def _action_masks_from_results(results: list[Any], fallback: list[Any]) -> list[Any]:
    masks = []
    for idx, result in enumerate(results):
        mask = result.info.get("action_mask")
        if mask is None:
            mask = fallback[idx]
        masks.append(mask)
    return masks


def _to_mask_tensor(masks: list[Any], device: torch.device) -> torch.Tensor:
    try:
        import numpy as np  # type: ignore

        mask_array = np.asarray(masks, dtype=bool)
        return torch.as_tensor(mask_array, dtype=torch.bool, device=device)
    except Exception:
        return torch.tensor(masks, dtype=torch.bool, device=device)


def _run_dashboard(train_log: Path, eval_log: Path, *, map_name: str, out_dir: Path) -> None:
    env = os.environ.copy()
    if "PYTHONPATH" not in env:
        env["PYTHONPATH"] = "src"
    cmd = [
        sys.executable,
        "-m",
        "vectortd.ai.training.plot_training_dashboard",
        "--map",
        map_name,
        "--train-log",
        str(train_log),
        "--eval-log",
        str(eval_log),
        "--out-dir",
        str(out_dir),
    ]
    result = subprocess.run(cmd, env=env, check=False)
    if result.returncode != 0:
        print(f"dashboard=skipped status={result.returncode}")


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="switchback")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--total-steps", type=int, default=50_000)
    ap.add_argument("--rollout-steps", type=int, default=128)
    ap.add_argument("--max-build-actions", type=int, default=100)
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
    ap.add_argument("--checkpoint-every", type=int, default=10_000)
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--eval-every", type=int, default=10_000)
    ap.add_argument("--eval-episodes", type=int, default=1)
    ap.add_argument("--save-replay-every", type=int, default=0)
    ap.add_argument("--save-replay-count", type=int, default=10)
    ap.add_argument("--replay-dir", default=None)
    ap.add_argument("--progress-interval-sec", type=float, default=60.0)
    ap.add_argument("--plot-dashboard", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.set_num_threads(max(1, (os.cpu_count() or 1) - 1))
    root_dir = _resolve_root()
    run_dir = _next_run_dir(root_dir)
    run_id = run_dir.name

    probe_env = VectorTDEventEnv(max_build_actions=args.max_build_actions)
    first_obs = probe_env.reset(map_path=args.map, seed=args.seed)
    if probe_env.action_spec is None:
        raise RuntimeError("Missing action spec after reset")
    action_dim = probe_env.action_spec.num_actions
    slot_size = len(first_obs.get("tower_slot_features", []) or [])
    max_towers = len(first_obs.get("tower_slots", []) or [])
    obs_dim = len(SCALAR_KEYS) + max_towers * slot_size

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

    vec = VectorizedEnv(
        num_envs=args.num_envs,
        env_kwargs={"max_build_actions": args.max_build_actions},
    )
    seeds = [args.seed + idx for idx in range(vec.num_envs)]
    obs_list = vec.reset(map_paths=args.map, seeds=seeds)
    mask_list = vec.get_action_masks()

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else run_dir / "checkpoints"
    replay_dir = Path(args.replay_dir) if args.replay_dir else run_dir / "replays"
    train_log_path = run_dir / "train_log.csv"
    eval_log_path = run_dir / "eval_log.csv"
    console_log_path = run_dir / "console.log"

    stdout_orig = sys.stdout
    stderr_orig = sys.stderr
    log_handle = console_log_path.open("a", encoding="utf-8")
    sys.stdout = _Tee(stdout_orig, log_handle)
    sys.stderr = _Tee(stderr_orig, log_handle)
    print(f"run_dir={run_dir}")

    total_steps = 0
    last_log_time = time.perf_counter()
    train_start = time.perf_counter()
    update_idx = 0
    episodes_completed = 0
    replay_interval = 0
    next_replay_step = 0
    pending_replay = False
    replays_saved = 0
    next_checkpoint_step = args.checkpoint_every if args.checkpoint_every > 0 else None
    next_eval_step = args.eval_every if args.eval_every > 0 else None
    if args.save_replay_every > 0:
        replay_interval = args.save_replay_every
    elif args.save_replay_count > 0 and args.total_steps > 0:
        replay_interval = max(1, args.total_steps // args.save_replay_count)
    if replay_interval > 0:
        next_replay_step = replay_interval

    try:
        while total_steps < args.total_steps:
            #print(f"update_start step={total_steps} update={update_idx + 1}")
            obs_buf = []
            actions_buf = []
            logp_buf = []
            values_buf = []
            rewards_buf = []
            dones_buf = []
            masks_buf = []

            for _ in range(args.rollout_steps):
                obs_tensor = batch_to_tensor(obs_list, max_towers=max_towers, slot_size=slot_size, device=device)
                mask_tensor = _to_mask_tensor(mask_list, device=device)
                actions, log_probs, values = agent.act(obs_tensor, mask_tensor)
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

                obs_list = [res.obs for res in results]
                mask_list = _action_masks_from_results(results, mask_list)

                reset_indices = [idx for idx, res in enumerate(results) if res.done]
                if reset_indices:
                    episodes_completed += len(reset_indices)
                    reset_seeds = [args.seed + idx + total_steps for idx in reset_indices]
                    reset_obs = vec.reset_at(reset_indices, map_paths=args.map, seeds=reset_seeds)
                    for sub_idx, env_idx in enumerate(reset_indices):
                        obs_list[env_idx] = reset_obs[sub_idx]
                    mask_list = vec.get_action_masks()

                total_steps += vec.num_envs
                if total_steps >= args.total_steps:
                    break

            last_obs_tensor = batch_to_tensor(obs_list, max_towers=max_towers, slot_size=slot_size, device=device)
            last_mask_tensor = _to_mask_tensor(mask_list, device=device)
            with torch.no_grad():
                _, _, last_values = agent.act(last_obs_tensor, last_mask_tensor, deterministic=True)

            rewards_tensor = torch.stack(rewards_buf)
            values_tensor = torch.stack(values_buf)
            dones_tensor = torch.stack(dones_buf)
            advantages, returns = compute_gae(
                rewards_tensor,
                values_tensor,
                dones_tensor,
                last_values.detach(),
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
            )

            obs_tensor = torch.stack(obs_buf).reshape(-1, obs_dim)
            actions_tensor = torch.stack(actions_buf).reshape(-1)
            logp_tensor = torch.stack(logp_buf).reshape(-1)
            masks_tensor = torch.stack(masks_buf).reshape(-1, action_dim)
            advantages_tensor = advantages.reshape(-1).detach()
            returns_tensor = returns.reshape(-1).detach()

            update_stats = agent.update(
                obs=obs_tensor,
                actions=actions_tensor,
                masks=masks_tensor,
                old_log_probs=logp_tensor.detach(),
                returns=returns_tensor,
                advantages=advantages_tensor,
            )
            update_idx += 1
            mean_reward = float(rewards_tensor.mean().item())
            mean_return = float(returns.mean().item())
            mean_value = float(values_tensor.mean().item())
            mean_advantage = float(advantages.mean().item())
            elapsed_sec = time.perf_counter() - train_start
            _append_train_log(
                train_log_path,
                {
                    "step": total_steps,
                    "map": args.map,
                    "run_id": run_id,
                    "update": update_idx,
                    "num_envs": vec.num_envs,
                    "elapsed_sec": f"{elapsed_sec:.3f}",
                    "policy_loss": f"{update_stats['policy_loss']:.6f}",
                    "value_loss": f"{update_stats['value_loss']:.6f}",
                    "entropy": f"{update_stats['entropy']:.6f}",
                    "mean_reward": f"{mean_reward:.6f}",
                    "mean_return": f"{mean_return:.6f}",
                    "mean_value": f"{mean_value:.6f}",
                    "mean_advantage": f"{mean_advantage:.6f}",
                },
            )

            now = time.perf_counter()
            if args.progress_interval_sec > 0 and now - last_log_time > args.progress_interval_sec:
                last_log_time = now
                elapsed = now - train_start
                steps_per_sec = total_steps / elapsed if elapsed > 0 else 0.0
                print(
                    "steps={} updates={} episodes={} steps_per_sec={:.2f} policy_loss={:.4f} value_loss={:.4f} entropy={:.4f}".format(
                        total_steps,
                        update_idx,
                        episodes_completed,
                        steps_per_sec,
                        update_stats["policy_loss"],
                        update_stats["value_loss"],
                        update_stats["entropy"],
                    )
                )

            if next_checkpoint_step is not None and total_steps >= next_checkpoint_step:
                ckpt_path = _checkpoint_path(checkpoint_dir, total_steps)
                torch.save(
                    {
                        "step": total_steps,
                        "checkpoint_step": int(next_checkpoint_step),
                        "map": args.map,
                        "seed": args.seed,
                        "obs_dim": obs_dim,
                        "action_dim": action_dim,
                        "config": config.__dict__,
                        "state_dict": agent.model.state_dict(),
                        "optimizer": agent.optimizer.state_dict(),
                    },
                    ckpt_path,
                )
                print(f"checkpoint={ckpt_path}")
                while next_checkpoint_step is not None and next_checkpoint_step <= total_steps:
                    next_checkpoint_step += args.checkpoint_every

            if replay_interval > 0 and total_steps >= next_replay_step:
                pending_replay = True
                while next_replay_step <= total_steps:
                    next_replay_step += replay_interval

            did_eval = False
            if next_eval_step is not None and total_steps >= next_eval_step:
                print(f"eval_start step={total_steps} episodes={args.eval_episodes}")
                eval_env = VectorTDEventEnv(max_build_actions=args.max_build_actions)
                episode_scores = []
                episode_waves = []
                episode_returns = []
                episode_lives = []
                episode_actions = []
                episode_steps = []
                episode_wave_counts = []
                wins = 0
                losses = 0
                eval_start = time.perf_counter()
                eval_log_interval = args.progress_interval_sec
                for eval_idx in range(args.eval_episodes):
                    obs = eval_env.reset(map_path=args.map, seed=args.seed + 1000 + eval_idx)
                    done = False
                    episode_return = 0.0
                    actions = 0
                    steps = 0
                    wave_count = 0
                    last_eval_log = time.perf_counter()
                    print(f"eval_episode_start episode={eval_idx + 1} seed={args.seed + 1000 + eval_idx}")
                    while not done:
                        obs_tensor = batch_to_tensor([obs], max_towers=max_towers, slot_size=slot_size, device=device)
                        mask_tensor = _to_mask_tensor([eval_env.get_action_mask()], device=device)
                        action, _, _ = agent.act(obs_tensor, mask_tensor, deterministic=True)
                        obs, reward, done, info = eval_env.step(int(action.item()))
                        episode_return += float(reward)
                        actions += 1
                        steps += 1
                        if "wave_ticks" in info:
                            wave_count += 1
                        if eval_log_interval > 0 and time.perf_counter() - last_eval_log >= eval_log_interval:
                            last_eval_log = time.perf_counter()
                            state = eval_env.engine.state if eval_env.engine is not None else None
                            wave_idx = int(getattr(state, "level", 0)) if state is not None else 0
                            lives = int(getattr(state, "lives", 0)) if state is not None else 0
                            bank = int(getattr(state, "bank", 0)) if state is not None else 0
                            score = int(getattr(state, "score", 0)) if state is not None else 0
                            print(
                                "eval_progress episode={} actions={} waves_seen={} wave_idx={} lives={} bank={} score={}".format(
                                    eval_idx + 1,
                                    actions,
                                    wave_count,
                                    wave_idx,
                                    lives,
                                    bank,
                                    score,
                                )
                            )
                    state = eval_env.engine.state if eval_env.engine is not None else None
                    episode_scores.append(int(getattr(state, "score", 0)))
                    episode_waves.append(int(getattr(state, "level", 0)))
                    episode_returns.append(episode_return)
                    episode_lives.append(int(getattr(state, "lives", 0)))
                    episode_actions.append(actions)
                    episode_steps.append(steps)
                    episode_wave_counts.append(wave_count)
                    if state is not None:
                        if getattr(state, "game_won", False):
                            wins += 1
                        if getattr(state, "game_over", False):
                            losses += 1
                elapsed_eval = time.perf_counter() - eval_start
                eval_episodes = max(1, args.eval_episodes)
                mean_score = sum(episode_scores) / eval_episodes
                max_score = max(episode_scores) if episode_scores else 0
                mean_wave = sum(episode_waves) / eval_episodes
                max_wave = max(episode_waves) if episode_waves else 0
                mean_return = sum(episode_returns) / eval_episodes
                mean_lives = sum(episode_lives) / eval_episodes
                mean_actions = sum(episode_actions) / eval_episodes
                mean_steps = sum(episode_steps) / eval_episodes
                mean_actions_per_wave = 0.0
                if episode_wave_counts:
                    total_waves = sum(episode_wave_counts)
                    if total_waves > 0:
                        mean_actions_per_wave = sum(episode_actions) / total_waves
                eval_row = {
                    "step": total_steps,
                    "map": args.map,
                    "run_id": run_id,
                    "eval_episodes": eval_episodes,
                    "mean_score": f"{mean_score:.2f}",
                    "max_score": max_score,
                    "mean_wave": f"{mean_wave:.2f}",
                    "max_wave": max_wave,
                    "mean_return": f"{mean_return:.4f}",
                    "mean_lives": f"{mean_lives:.2f}",
                    "win_rate": f"{wins / eval_episodes:.3f}",
                    "loss_rate": f"{losses / eval_episodes:.3f}",
                    "mean_actions": f"{mean_actions:.2f}",
                    "mean_actions_per_wave": f"{mean_actions_per_wave:.2f}",
                    "mean_steps": f"{mean_steps:.2f}",
                    "elapsed_sec": f"{elapsed_eval:.3f}",
                }
                _append_eval_log(eval_log_path, eval_row)
                print(
                    "eval step={} mean_score={:.2f} mean_wave={:.2f} win_rate={:.2f} mean_return={:.2f}".format(
                        total_steps,
                        mean_score,
                        mean_wave,
                        wins / eval_episodes,
                        mean_return,
                    )
                )
                print(
                    "eval_done step={} episodes={} elapsed_sec={:.3f}".format(
                        total_steps,
                        eval_episodes,
                        elapsed_eval,
                    )
                )
                did_eval = True
                if pending_replay:
                    _save_replay(eval_env, replay_dir, total_steps)
                    pending_replay = False
                    replays_saved += 1
                while next_eval_step is not None and next_eval_step <= total_steps:
                    next_eval_step += args.eval_every

            if pending_replay and not did_eval:
                replay_env = VectorTDEventEnv(max_build_actions=args.max_build_actions)
                replay_seed = args.seed + 2000 + replays_saved
                obs = replay_env.reset(map_path=args.map, seed=replay_seed)
                done = False
                while not done:
                    obs_tensor = batch_to_tensor([obs], max_towers=max_towers, slot_size=slot_size, device=device)
                    mask_tensor = _to_mask_tensor([replay_env.get_action_mask()], device=device)
                    action, _, _ = agent.act(obs_tensor, mask_tensor, deterministic=True)
                    obs, _, done, _ = replay_env.step(int(action.item()))
                _save_replay(replay_env, replay_dir, total_steps)
                pending_replay = False
                replays_saved += 1
            if did_eval:
                print(f"training_resumed step={total_steps}")

        vec.close()
        if args.plot_dashboard:
            _run_dashboard(train_log_path, eval_log_path, map_name=args.map, out_dir=run_dir / "dashboard")
        return 0
    finally:
        sys.stdout = stdout_orig
        sys.stderr = stderr_orig
        log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
