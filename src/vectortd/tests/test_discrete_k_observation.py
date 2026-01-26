from vectortd.ai.env import VectorTDEventEnv


def test_discrete_k_place_candidates_align_with_mask() -> None:
    env = VectorTDEventEnv(action_space_kind="discrete_k", default_map="switchback")
    _, info = env.reset(seed=123)
    obs_dict = env.last_obs
    assert obs_dict is not None

    candidates = obs_dict.get("place_candidates") or []
    features = obs_dict.get("place_candidate_features") or []
    assert candidates
    assert len(features) == 12

    for row in candidates:
        assert len(row) == len(features)
        for value in row:
            assert 0.0 <= float(value) <= 1.0

    spec = env.action_spec
    assert spec is not None
    assert len(candidates) == spec.place_count

    mask = info.get("action_mask")
    assert mask is not None
    mask = list(mask)
    base = spec.offsets.place
    for idx, row in enumerate(candidates):
        valid = row[0] >= 0.5
        affordable = row[1] >= 0.5
        expected = valid and affordable
        assert bool(mask[base + idx]) == expected
