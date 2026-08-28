from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from safetensors.torch import load_file as load_safetensors

from latency_meta_mdp.belief.flow.config import load_flow_belief_config
from latency_meta_mdp.belief.flow.fresh_decoder_config import (
    load_fresh_decoder_probe_config,
)
from latency_meta_mdp.belief.flow.fresh_decoder_evaluation import (
    evaluate_cached_vector_field,
)
from latency_meta_mdp.belief.flow.fresh_decoder_probe import (
    CachedBeliefSplit,
    FrozenEncoderFreshDecoder,
    cache_frozen_belief_splits,
)
from latency_meta_mdp.belief.flow.fresh_decoder_run import (
    summarize_fresh_decoder_metrics,
)
from latency_meta_mdp.belief.flow.fresh_decoder_training import train_fresh_vector_field
from latency_meta_mdp.belief.flow.model import FlowBeliefModel
from latency_meta_mdp.belief.flow.training_data import FlowBeliefNormalization
from latency_meta_mdp.belief.flow.vector_field import ConditionalStateVectorField
from latency_meta_mdp.vision_probe_data import ProbeSplit


def _config():
    return load_flow_belief_config(Path("configs/belief/dinov3_flow_belief_v1.yaml"))


def _cached_split() -> CachedBeliefSplit:
    return CachedBeliefSplit(
        belief_tokens=np.zeros((3, 8, 192), dtype=np.float32),
        latency_probabilities=np.full((3, 20), 0.05, dtype=np.float64),
        target_states=np.zeros((3, 20, 22), dtype=np.float32),
        interaction_mode=np.zeros((3, 20), dtype=np.int8),
        absorbing=np.zeros((3, 20), dtype=np.bool_),
        episode_ids=("episode-a", "episode-b", "episode-c"),
        source_ticks=np.asarray([5, 6, 7], dtype=np.int64),
    )


def test_cached_belief_split_is_readonly_and_shape_checked() -> None:
    cached = _cached_split()

    assert cached.belief_tokens.flags.writeable is False
    assert cached.latency_probabilities.flags.writeable is False
    assert cached.target_states.shape == (3, 20, 22)
    assert cached.episode_ids == ("episode-a", "episode-b", "episode-c")

    with pytest.raises(ValueError, match="shape"):
        CachedBeliefSplit(
            belief_tokens=np.zeros((3, 7, 192), dtype=np.float32),
            latency_probabilities=np.full((3, 20), 0.05, dtype=np.float64),
            target_states=np.zeros((3, 20, 22), dtype=np.float32),
            interaction_mode=np.zeros((3, 20), dtype=np.int8),
            absorbing=np.zeros((3, 20), dtype=np.bool_),
            episode_ids=("episode-a", "episode-b", "episode-c"),
            source_ticks=np.asarray([5, 6, 7], dtype=np.int64),
        )


def test_fresh_decoder_probe_freezes_an_exact_encoder_copy() -> None:
    config = _config()
    torch.manual_seed(17)
    source = FlowBeliefModel(config)
    source_encoder_state = {
        name: value.detach().clone() for name, value in source.encoder.state_dict().items()
    }
    source_decoder_state = {
        name: value.detach().clone() for name, value in source.vector_field.state_dict().items()
    }

    probe = FrozenEncoderFreshDecoder.from_source_model(
        source_model=source,
        config=config,
        decoder_seed=29,
    )

    assert probe.encoder.training is False
    assert all(not parameter.requires_grad for parameter in probe.encoder.parameters())
    assert all(parameter.requires_grad for parameter in probe.fresh_decoder.parameters())
    for name, value in probe.encoder.state_dict().items():
        torch.testing.assert_close(value, source_encoder_state[name])
    assert any(
        not torch.equal(value, source_decoder_state[name])
        for name, value in probe.fresh_decoder.state_dict().items()
    )
    for name, value in source.vector_field.state_dict().items():
        torch.testing.assert_close(value, source_decoder_state[name])


def test_fresh_decoder_has_no_raw_context_input() -> None:
    parameters = inspect.signature(FrozenEncoderFreshDecoder.decode).parameters

    assert set(parameters) == {
        "self",
        "noisy_state",
        "flow_time",
        "belief_tokens",
        "delay_ticks",
    }


def test_fresh_decoder_rejects_training_mode_for_frozen_encoder() -> None:
    probe = FrozenEncoderFreshDecoder.from_source_model(
        source_model=FlowBeliefModel(_config()),
        config=_config(),
        decoder_seed=31,
    )

    probe.train()

    assert probe.encoder.training is False
    assert probe.fresh_decoder.training is True


def test_cache_frozen_belief_splits_normalizes_targets_and_preserves_identity() -> None:
    sample = SimpleNamespace(
        episode_id="episode-a",
        source_tick=7,
        vision_history=np.zeros((6, 2, 196, 384), dtype=np.float16),
        robot_proprio_history=np.full((6, 16), 3.0, dtype=np.float32),
        remaining_actions=np.full((25, 7), 5.0, dtype=np.float32),
        latency_probabilities=np.full(20, 0.05, dtype=np.float64),
        target_states=np.full((20, 22), 9.0, dtype=np.float32),
        target_interaction_mode=np.arange(20, dtype=np.int8) % 4,
        target_absorbing=np.zeros(20, dtype=np.bool_),
    )

    class Corpus:
        sample_references = {
            ProbeSplit.TRAIN: ((0, 0),),
            ProbeSplit.VALIDATION: ((0, 0),),
            ProbeSplit.HOLDOUT: (),
        }

        @staticmethod
        def materialize(split, offset):
            assert split in (ProbeSplit.TRAIN, ProbeSplit.VALIDATION)
            assert offset == 0
            return sample

    class Probe:
        calls = []

        def encode(self, **values):
            self.calls.append(values)
            return torch.full((len(values["vision_history"]), 8, 192), 2.0)

    normalization = FlowBeliefNormalization(
        proprio_mean=np.ones(16, dtype=np.float32),
        proprio_std=np.full(16, 2.0, dtype=np.float32),
        action_mean=np.ones(7, dtype=np.float32),
        action_std=np.full(7, 4.0, dtype=np.float32),
        target_mean=np.ones(22, dtype=np.float32),
        target_std=np.full(22, 2.0, dtype=np.float32),
    )
    probe = Probe()

    cached = cache_frozen_belief_splits(
        corpus=Corpus(),
        probe=probe,
        normalization=normalization,
        splits=(ProbeSplit.TRAIN, ProbeSplit.VALIDATION),
        batch_size=2,
        device="cpu",
    )

    assert set(cached) == {ProbeSplit.TRAIN, ProbeSplit.VALIDATION}
    assert len(probe.calls) == 2
    np.testing.assert_array_equal(cached[ProbeSplit.TRAIN].belief_tokens, 2.0)
    np.testing.assert_array_equal(cached[ProbeSplit.TRAIN].target_states, 4.0)
    np.testing.assert_array_equal(
        probe.calls[0]["proprio_history"].numpy(),
        1.0,
    )
    np.testing.assert_array_equal(
        probe.calls[0]["remaining_actions"].numpy(),
        1.0,
    )
    assert cached[ProbeSplit.TRAIN].episode_ids == ("episode-a",)
    assert cached[ProbeSplit.TRAIN].source_ticks.tolist() == [7]


def _random_cached_split(*, count: int, seed: int) -> CachedBeliefSplit:
    rng = np.random.default_rng(seed)
    return CachedBeliefSplit(
        belief_tokens=rng.normal(size=(count, 8, 192)).astype(np.float32),
        latency_probabilities=np.full((count, 20), 0.05, dtype=np.float64),
        target_states=rng.normal(size=(count, 20, 22)).astype(np.float32),
        interaction_mode=np.zeros((count, 20), dtype=np.int8),
        absorbing=np.zeros((count, 20), dtype=np.bool_),
        episode_ids=tuple(f"episode-{index}" for index in range(count)),
        source_ticks=np.arange(count, dtype=np.int64),
    )


def test_train_fresh_vector_field_writes_decoder_only_checkpoint(tmp_path: Path) -> None:
    config = replace(
        _config(),
        flow_hidden_dim=32,
        flow_residual_block_count=1,
        sampled_delay_query_count=2,
        validation_flow_draw_count=2,
        batch_size=2,
        max_epochs=2,
        early_stopping_patience=2,
        solver_step_count=2,
        evaluation_sample_count=2,
    )
    cached = {
        ProbeSplit.TRAIN: _random_cached_split(count=4, seed=7),
        ProbeSplit.VALIDATION: _random_cached_split(count=2, seed=11),
    }
    original_tokens = np.array(cached[ProbeSplit.TRAIN].belief_tokens, copy=True)

    manifest_path = train_fresh_vector_field(
        cached=cached,
        config=config,
        level=1,
        decoder_seed=41,
        output_dir=tmp_path / "seed_000041",
        device="cpu",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state = load_safetensors(manifest_path.parent / "vector_field.safetensors")
    assert manifest["format_id"] == "flow_belief_fresh_decoder_seed_v1"
    assert manifest["level"] == 1
    assert manifest["decoder_seed"] == 41
    assert manifest["epochs_completed"] == 2
    assert set(manifest["artifacts"]) == {
        "metrics.json",
        "training_history.json",
        "vector_field.safetensors",
    }
    assert state
    assert all(not name.startswith("encoder.") for name in state)
    np.testing.assert_array_equal(cached[ProbeSplit.TRAIN].belief_tokens, original_tokens)


def test_train_fresh_vector_field_rejects_missing_validation_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="training and validation"):
        train_fresh_vector_field(
            cached={ProbeSplit.TRAIN: _random_cached_split(count=2, seed=5)},
            config=replace(
                _config(),
                flow_hidden_dim=32,
                flow_residual_block_count=1,
                batch_size=2,
                max_epochs=1,
            ),
            level=1,
            decoder_seed=43,
            output_dir=tmp_path / "missing-validation",
            device="cpu",
        )


def test_evaluate_cached_vector_field_reports_physical_and_delay_metrics() -> None:
    config = replace(
        _config(),
        flow_hidden_dim=32,
        flow_residual_block_count=1,
        batch_size=2,
        solver_step_count=2,
        evaluation_sample_count=3,
    )
    normalization = FlowBeliefNormalization(
        proprio_mean=np.zeros(16, dtype=np.float32),
        proprio_std=np.ones(16, dtype=np.float32),
        action_mean=np.zeros(7, dtype=np.float32),
        action_std=np.ones(7, dtype=np.float32),
        target_mean=np.zeros(22, dtype=np.float32),
        target_std=np.ones(22, dtype=np.float32),
    )

    metrics = evaluate_cached_vector_field(
        vector_field=ConditionalStateVectorField(config),
        cached=_random_cached_split(count=2, seed=17),
        normalization=normalization,
        config=config,
        level=1,
        device="cpu",
    )

    assert set(metrics["per_delay"]) == {str(delay) for delay in range(1, 21)}
    assert metrics["overall"]["distribution"]["sample_count"] == 3
    assert metrics["overall"]["physical"]["object_position"]["rmse_display"] > 0.0
    assert "pre_handoff" in metrics
    assert metrics["sampling"]["solver_step_count"] == 2


def test_fresh_decoder_probe_config_locks_three_unique_seeds(tmp_path: Path) -> None:
    path = tmp_path / "probe.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "probe_id": "dinov3_flow_belief_fresh_decoder_probe_v1",
                "decoder_seeds": [20260831, 20260832, 20260833],
                "max_epochs": 100,
                "early_stopping_patience": 12,
                "rmse_ratio_max": 1.2,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config = load_fresh_decoder_probe_config(path)

    assert config.decoder_seeds == (20260831, 20260832, 20260833)
    assert config.rmse_ratio_max == 1.2


def test_fresh_decoder_probe_config_rejects_duplicate_seeds(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "probe_id": "dinov3_flow_belief_fresh_decoder_probe_v1",
                "decoder_seeds": [5, 5, 7],
                "max_epochs": 2,
                "early_stopping_patience": 1,
                "rmse_ratio_max": 1.2,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sorted unique"):
        load_fresh_decoder_probe_config(path)


def _physical_metrics(*, object_position: float, robot_qpos: float) -> dict:
    return {
        "overall": {
            "physical": {
                "object_position": {"rmse_display": object_position},
                "robot_joint_position": {"rmse_display": robot_qpos},
            }
        }
    }


def test_fresh_decoder_summary_uses_three_seed_median_and_gate() -> None:
    summary = summarize_fresh_decoder_metrics(
        joint_metrics=_physical_metrics(object_position=2.0, robot_qpos=0.01),
        fresh_metrics={
            11: _physical_metrics(object_position=2.2, robot_qpos=0.011),
            13: _physical_metrics(object_position=2.6, robot_qpos=0.013),
            17: _physical_metrics(object_position=2.3, robot_qpos=0.0115),
        },
        rmse_ratio_max=1.2,
    )

    assert summary["median_rmse_ratios"]["object_position"] == pytest.approx(1.15)
    assert summary["median_rmse_ratios"]["robot_joint_position"] == pytest.approx(1.15)
    assert summary["admitted"] is True


def test_fresh_decoder_summary_rejects_a_failed_state_group() -> None:
    summary = summarize_fresh_decoder_metrics(
        joint_metrics=_physical_metrics(object_position=2.0, robot_qpos=0.01),
        fresh_metrics={
            11: _physical_metrics(object_position=2.5, robot_qpos=0.011),
            13: _physical_metrics(object_position=2.6, robot_qpos=0.011),
            17: _physical_metrics(object_position=2.7, robot_qpos=0.011),
        },
        rmse_ratio_max=1.2,
    )

    assert summary["admitted"] is False
    assert summary["failed_groups"] == ["object_position"]
