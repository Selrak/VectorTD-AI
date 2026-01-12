from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
except Exception as exc:  # pragma: no cover - import guard
    raise ImportError("PyTorch is required for PPO training. Install torch and retry.") from exc


SCALAR_KEYS = (
    "bank_norm",
    "lives_norm",
    "score_norm",
    "wave_norm",
    "interest_norm",
    "base_hp_norm",
    "base_worth_norm",
    "tower_count_norm",
    "wave_current_present",
    "wave_current_type_norm",
    "wave_current_hp_norm",
    "wave_next_present",
    "wave_next_type_norm",
    "wave_next_hp_norm",
)


def flatten_observation(obs: dict, *, max_towers: int, slot_size: int) -> list[float]:
    values: list[float] = [float(obs.get(key, 0.0) or 0.0) for key in SCALAR_KEYS]
    tower_slots = obs.get("tower_slots", []) or []
    empty_slot = [0.0] * slot_size
    for idx in range(max_towers):
        slot = tower_slots[idx] if idx < len(tower_slots) else empty_slot
        values.extend(float(value) for value in slot)
    return values


def batch_to_tensor(
    obs_batch: Iterable[dict],
    *,
    max_towers: int,
    slot_size: int,
    device: torch.device,
) -> torch.Tensor:
    data = [flatten_observation(obs, max_towers=max_towers, slot_size=slot_size) for obs in obs_batch]
    return torch.tensor(data, dtype=torch.float32, device=device)


def mask_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype != torch.bool:
        mask = mask.to(dtype=torch.bool)
    return logits.masked_fill(~mask, -1e9)


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 256


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: tuple[int, int] = (256, 256)) -> None:
        super().__init__()
        layers = []
        last_dim = obs_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden))
            layers.append(nn.Tanh())
            last_dim = hidden
        self.shared = nn.Sequential(*layers)
        self.policy = nn.Linear(last_dim, action_dim)
        self.value = nn.Linear(last_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.shared(obs)
        logits = self.policy(hidden)
        value = self.value(hidden).squeeze(-1)
        return logits, value


class PPOAgent:
    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        config: PPOConfig,
        device: torch.device,
    ) -> None:
        self.config = config
        self.device = device
        self.model = ActorCritic(obs_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

    def act(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.model(obs)
        logits = mask_logits(logits, mask)
        dist = Categorical(logits=logits)
        if deterministic:
            actions = torch.argmax(logits, dim=-1)
        else:
            actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs, values

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.model(obs)
        logits = mask_logits(logits, mask)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values

    def update(
        self,
        *,
        obs: torch.Tensor,
        actions: torch.Tensor,
        masks: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ) -> dict[str, float]:
        config = self.config
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        batch_size = obs.shape[0]
        indices = torch.randperm(batch_size, device=obs.device)

        policy_losses = []
        value_losses = []
        entropy_losses = []

        for _ in range(config.update_epochs):
            for start in range(0, batch_size, config.minibatch_size):
                end = start + config.minibatch_size
                batch_idx = indices[start:end]

                log_probs, entropy, values = self.evaluate_actions(
                    obs[batch_idx],
                    actions[batch_idx],
                    masks[batch_idx],
                )
                ratio = torch.exp(log_probs - old_log_probs[batch_idx])
                surr1 = ratio * advantages[batch_idx]
                surr2 = torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range) * advantages[
                    batch_idx
                ]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, returns[batch_idx])
                entropy_loss = -entropy.mean()

                loss = policy_loss + config.value_coef * value_loss + config.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), config.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(float(policy_loss.detach().cpu()))
                value_losses.append(float(value_loss.detach().cpu()))
                entropy_losses.append(float(entropy.mean().detach().cpu()))

        return {
            "policy_loss": float(sum(policy_losses) / max(1, len(policy_losses))),
            "value_loss": float(sum(value_losses) / max(1, len(value_losses))),
            "entropy": float(sum(entropy_losses) / max(1, len(entropy_losses))),
        }


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_values: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    steps, num_envs = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(num_envs, device=rewards.device)
    for t in reversed(range(steps)):
        next_values = last_values if t == steps - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
        gae = delta + gamma * gae_lambda * next_nonterminal * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns
