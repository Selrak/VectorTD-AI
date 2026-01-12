from __future__ import annotations


def normalize_seed(seed: int | None) -> int:
    if seed is None:
        return 1
    seed_val = int(seed) & 0x7FFFFFFF
    return seed_val if seed_val != 0 else 1


def seed_state(state, seed: int | None) -> int:
    seed_val = normalize_seed(seed)
    setattr(state, "rng_state", seed_val)
    if hasattr(state, "rng_calls"):
        setattr(state, "rng_calls", 0)
    return seed_val


def get_state_seed(state) -> int:
    return int(getattr(state, "rng_state", 1))
