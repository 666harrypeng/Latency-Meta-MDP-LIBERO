from __future__ import annotations

from pathlib import Path

from latency_meta_mdp.belief.flow.multilaw_evaluation_config import (
    load_multilaw_flow_evaluation_config,
)


def test_multilaw_evaluation_config_locks_three_shifted_laws() -> None:
    config = load_multilaw_flow_evaluation_config(
        Path("configs/analysis/dinov3_flow_belief_multilaw_evaluation_v1.yaml")
    )

    assert tuple(config.shifted_laws) == ("fast", "slow", "wide")
    assert config.shifted_laws["fast"].mean_logit_offset == -0.20
    assert config.shifted_laws["slow"].mean_logit_offset == 0.20
    assert config.shifted_laws["wide"].log_concentration_offset == -0.25
