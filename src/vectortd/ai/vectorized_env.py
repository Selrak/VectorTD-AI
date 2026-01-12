from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
import os
from typing import Any

from vectortd.ai.env import VectorTDEventEnv


def _default_num_envs() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)


def _expand(values, count: int, default):
    if values is None:
        return [default] * count
    if not isinstance(values, (list, tuple)):
        return [values] * count
    if not values:
        return [default] * count
    return [values[i % len(values)] for i in range(count)]


@dataclass
class StepResult:
    obs: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]


def _worker(conn, env_kwargs: dict[str, Any]) -> None:
    env = VectorTDEventEnv(**env_kwargs)
    while True:
        cmd, payload = conn.recv()
        if cmd == "reset":
            map_path = payload.get("map_path")
            seed = payload.get("seed")
            obs = env.reset(map_path=map_path, seed=seed)
            conn.send(("ok", obs))
        elif cmd == "step":
            action = payload.get("action")
            obs, reward, done, info = env.step(action)
            conn.send(("ok", (obs, reward, done, info)))
        elif cmd == "mask":
            mask = env.get_action_mask()
            conn.send(("ok", mask))
        elif cmd == "close":
            conn.send(("ok", None))
            break
        elif cmd == "timing":
            conn.send(("ok", env.get_timing_snapshot()))
        else:
            conn.send(("error", f"Unknown command {cmd!r}"))


class VectorizedEnv:
    def __init__(
        self,
        *,
        num_envs: int | None = None,
        env_kwargs: dict[str, Any] | None = None,
        start_method: str | None = None,
    ) -> None:
        if num_envs is None:
            num_envs = _default_num_envs()
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.num_envs = num_envs
        self.env_kwargs = env_kwargs or {}
        method = start_method or ("spawn" if os.name == "nt" else None)
        self._ctx = mp.get_context(method)
        self._pipes: list[Any] = []
        self._procs: list[mp.Process] = []
        for _ in range(self.num_envs):
            parent, child = self._ctx.Pipe()
            proc = self._ctx.Process(target=_worker, args=(child, dict(self.env_kwargs)))
            proc.daemon = True
            proc.start()
            child.close()
            self._pipes.append(parent)
            self._procs.append(proc)

    def reset(self, *, map_paths=None, seeds=None) -> list[dict[str, Any]]:
        map_list = _expand(map_paths, self.num_envs, None)
        seed_list = _expand(seeds, self.num_envs, None)
        for pipe, map_path, seed in zip(self._pipes, map_list, seed_list):
            pipe.send(("reset", {"map_path": map_path, "seed": seed}))
        return [self._recv(pipe) for pipe in self._pipes]

    def reset_at(self, indices: list[int], *, map_paths=None, seeds=None) -> list[dict[str, Any]]:
        if not indices:
            return []
        for idx in indices:
            if idx < 0 or idx >= self.num_envs:
                raise ValueError(f"Invalid env index {idx}")
        map_list = _expand(map_paths, len(indices), None)
        seed_list = _expand(seeds, len(indices), None)
        for idx, map_path, seed in zip(indices, map_list, seed_list):
            self._pipes[idx].send(("reset", {"map_path": map_path, "seed": seed}))
        return [self._recv(self._pipes[idx]) for idx in indices]

    def step(self, actions: list[Any]) -> list[StepResult]:
        if len(actions) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} actions, got {len(actions)}")
        for pipe, action in zip(self._pipes, actions):
            pipe.send(("step", {"action": action}))
        results: list[StepResult] = []
        for pipe in self._pipes:
            obs, reward, done, info = self._recv(pipe)
            results.append(StepResult(obs=obs, reward=float(reward), done=bool(done), info=dict(info)))
        return results

    def get_action_masks(self) -> list[Any]:
        for pipe in self._pipes:
            pipe.send(("mask", {}))
        return [self._recv(pipe) for pipe in self._pipes]

    def get_timings(self) -> list[dict[str, Any]]:
        for pipe in self._pipes:
            pipe.send(("timing", {}))
        return [self._recv(pipe) for pipe in self._pipes]

    def close(self) -> None:
        for pipe in self._pipes:
            pipe.send(("close", {}))
        for pipe in self._pipes:
            self._recv(pipe)
        for proc in self._procs:
            proc.join(timeout=1.0)

    def _recv(self, pipe):
        status, payload = pipe.recv()
        if status == "ok":
            return payload
        raise RuntimeError(payload)
