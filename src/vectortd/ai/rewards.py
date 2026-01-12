from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RewardState:
    bank: int
    lives: int
    score: int
    level: int


@dataclass(frozen=True, slots=True)
class RewardConfig:
    score_weight: float = 1.0
    bank_weight: float = 0.0
    life_loss_penalty: float = 100.0
    no_life_loss_bonus: float = 50.0
    terminal_win_bonus: float = 10_000.0
    terminal_loss_penalty: float = 10_000.0
    build_step_penalty: float = 0.0
    build_action_limit_penalty: float = 0.0
    invalid_action_penalty: float = 0.0


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
    phase_transition: str,
    config: RewardConfig,
    invalid_action: bool = False,
    build_action_limit_violation: bool = False,
    episode_done: bool = False,
    game_won: bool = False,
) -> float:
    reward = 0.0
    if phase_transition == "BUILD_ACTION":
        reward += config.build_step_penalty
    elif phase_transition == "WAVE_COMPLETE":
        bank_delta = new_state.bank - prev_state.bank
        score_delta = new_state.score - prev_state.score
        lives_delta = new_state.lives - prev_state.lives
        lives_lost = max(0, -lives_delta)
        reward += score_delta * config.score_weight
        reward += bank_delta * config.bank_weight
        if lives_lost == 0:
            reward += config.no_life_loss_bonus
        reward -= lives_lost * config.life_loss_penalty
    if invalid_action:
        reward += config.invalid_action_penalty
    if build_action_limit_violation:
        reward += config.build_action_limit_penalty
    if episode_done:
        if game_won:
            reward += config.terminal_win_bonus
        else:
            reward -= config.terminal_loss_penalty
    return float(reward)
