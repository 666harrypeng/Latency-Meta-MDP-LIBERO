from __future__ import annotations

from pathlib import Path

from latency_meta_mdp.openpi_sft import register_sft_configs
from latency_meta_mdp.sft_profile import load_sft_profile

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_register_sft_configs_builds_three_full_parameter_h16_profiles(
    tmp_path: Path,
) -> None:
    from openpi.training.config import _CONFIGS_DICT

    profile = load_sft_profile(
        _PROJECT_ROOT / "configs/policy/pi05_panda_ball_full_sft_v1.yaml"
    )
    names = register_sft_configs(profile)

    assert names == tuple(
        profile.levels[level].config_name for level in (1, 2, 3)
    )
    for level, name in zip((1, 2, 3), names, strict=True):
        config = _CONFIGS_DICT[name]
        assert config.model.pi05 is True
        assert config.model.action_horizon == 16
        assert config.model.discrete_state_input is False
        assert config.model.paligemma_variant == "gemma_2b"
        assert config.model.action_expert_variant == "gemma_300m"
        assert config.freeze_filter.__class__.__name__ == "Nothing"
        assert config.batch_size == 64
        assert config.num_train_steps == 12_000
        assert config.save_interval == 4_000
        assert config.policy_metadata["level"] == level

        data = config.data.create(tmp_path, config.model)
        assert data.repo_id == profile.levels[level].repo_id
        assert data.drop_n_last_frames == 15
        assert data.action_sequence_keys == ("actions",)
