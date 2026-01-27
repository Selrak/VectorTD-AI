from __future__ import annotations

from dataclasses import dataclass

from vectortd.core.rules.wave_spawner import LEVELS


REWARD_LIFE_LOSS = 0.5
REWARD_WAVE_TOTAL = 8.0
REWARD_TERMINAL_WIN = 2.0
REWARD_TERMINAL_LOSS = -2.0
DEFAULT_MAX_WAVES = len(LEVELS)


@dataclass(frozen=True, slots=True)
class RewardState:
    bank: int
    lives: int
    score: int
    level: int


@dataclass(frozen=True, slots=True)
class RewardConfig:
    life_loss_penalty: float = REWARD_LIFE_LOSS
    wave_total_reward: float = REWARD_WAVE_TOTAL
    terminal_win_reward: float = REWARD_TERMINAL_WIN
    terminal_loss_reward: float = REWARD_TERMINAL_LOSS
    max_waves: int | None = DEFAULT_MAX_WAVES


def reward_state_from(state) -> RewardState:
    return RewardState(
        bank=int(getattr(state, "bank", 0)),
        lives=int(getattr(state, "lives", 0)),
        score=int(getattr(state, "score", 0)),
        level=int(getattr(state, "level", 0)),
    )


def compute_reward(
    prev_state: RewardState,
    new_state: RewardState,
    *,
    config: RewardConfig,
    episode_done: bool = False,
    game_won: bool = False,
) -> float:
    breakdown = compute_reward_breakdown(
        prev_state,
        new_state,
        config=config,
        episode_done=episode_done,
        game_won=game_won,
    )
    return float(breakdown["total"])


def _wave_reward_scale(config: RewardConfig) -> float:
    max_waves = config.max_waves
    if max_waves is None:
        return 0.2
    max_waves_int = int(max_waves)
    if max_waves_int > 0:
        return float(config.wave_total_reward) / float(max_waves_int)
    return 0.2


def compute_reward_breakdown(
    prev_state: RewardState,
    new_state: RewardState,
    *,
    config: RewardConfig,
    episode_done: bool = False,
    game_won: bool = False,
) -> dict[str, float]:
    delta_lives = max(0, prev_state.lives - new_state.lives)
    delta_waves = max(0, new_state.level - prev_state.level)
    if episode_done and not game_won and delta_waves > 0:
        delta_waves -= 1

    r_life = -float(config.life_loss_penalty) * float(delta_lives)
    r_wave = _wave_reward_scale(config) * float(delta_waves)
    if episode_done:
        r_terminal = float(config.terminal_win_reward) if game_won else float(config.terminal_loss_reward)
    else:
        r_terminal = 0.0

    total = float(r_life + r_wave + r_terminal)
    return {
        "total": total,
        "r_life": float(r_life),
        "r_wave": float(r_wave),
        "r_terminal": float(r_terminal),
        "delta_lives": float(delta_lives),
        "delta_waves": float(delta_waves),
    }
