from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("regime", ["nominal", "family", "zero"])
def test_delay_stream_is_reproducible_and_independent_of_protocol(regime):
    from latency_meta_mdp.runtime.evaluate_conveyor import conveyor_delay_sampler

    a, identity = conveyor_delay_sampler(Path.cwd(), regime=regime, scene_seed=2000, policy_seed=0)
    b, other = conveyor_delay_sampler(Path.cwd(), regime=regime, scene_seed=2000, policy_seed=0)
    assert identity == other
    values = [a() for _ in range(100)]
    assert values == [b() for _ in range(100)]
    if regime == "zero":
        assert set(values) == {0}
    else:
        assert min(values) >= 1 and max(values) <= 20
        assert np.isclose(sum(identity["probabilities"]), 1)


def test_nominal_law_is_shared_but_family_is_assigned_per_scene():
    from latency_meta_mdp.runtime.evaluate_conveyor import conveyor_delay_sampler

    def identity(regime, seed):
        return conveyor_delay_sampler(Path.cwd(), regime=regime, scene_seed=seed, policy_seed=0)[1]

    assert identity("nominal", 2000)["probabilities"] == identity("nominal", 2001)["probabilities"]
    assert identity("family", 2000)["probabilities"] != identity("family", 2001)["probabilities"]
