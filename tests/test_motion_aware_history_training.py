from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from latency_meta_mdp.belief.causal_return.motion_aware_contracts import (
    MotionAwareHistoryEstimate,
    MotionAwareHistorySample,
    load_motion_aware_history_config,
)
from latency_meta_mdp.belief.causal_return.motion_aware_training import (
    MotionAwareDataset,
    MotionAwareTrainingProvenance,
    _rename_directory_no_replace,
    build_motion_aware_normalization,
    collate_motion_aware_items,
    masked_motion_aware_huber,
    motion_aware_huber_terms,
    train_level_motion_aware_history,
)
from latency_meta_mdp.vision_probe_data import ProbeSplit


class _StateValueCorpus:
    def __init__(self) -> None:
        self.sample_references = {
            ProbeSplit.TRAIN: ((0, 0), (0, 1)),
            ProbeSplit.VALIDATION: ((1, 0),),
            ProbeSplit.HOLDOUT: (),
        }
        self.values = {
            (ProbeSplit.TRAIN, 0): (
                np.zeros((6, 16), dtype=np.float32),
                np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
                np.asarray([3.0, 4.0, 5.0], dtype=np.float32),
            ),
            (ProbeSplit.TRAIN, 1): (
                np.full((6, 16), 2.0, dtype=np.float32),
                np.asarray([2.0, 3.0, 4.0], dtype=np.float32),
                np.asarray([5.0, 6.0, 7.0], dtype=np.float32),
            ),
            (ProbeSplit.VALIDATION, 0): (
                np.full((6, 16), 1000.0, dtype=np.float32),
                np.full(3, 1000.0, dtype=np.float32),
                np.full(3, 1000.0, dtype=np.float32),
            ),
        }

    def state_values(self, split: ProbeSplit, offset: int):
        return self.values[(split, offset)]


def _sample(
    *, episode_id: str, seed: int, exact: bool, level: int = 1
) -> MotionAwareHistorySample:
    return MotionAwareHistorySample(
        episode_id=episode_id,
        level=level,
        scene_seed=seed,
        source_tick=25,
        source_phase="pregrasp",
        transition_offset_ticks=0,
        transition_offset_valid=exact,
        vision_history=np.zeros((6, 2, 196, 384), dtype=np.float16),
        robot_history=np.zeros((6, 16), dtype=np.float32),
        history_valid_mask=np.ones(6, dtype=np.bool_),
        object_position_target=np.ones(3, dtype=np.float32),
        object_velocity_target=np.ones(3, dtype=np.float32),
    )


class _TrainingCorpus:
    level = 1

    def __init__(self) -> None:
        self.samples = {
            ProbeSplit.TRAIN: tuple(
                _sample(episode_id=f"train-{index}", seed=1000 + index, exact=False)
                for index in range(4)
            ),
            ProbeSplit.VALIDATION: tuple(
                _sample(episode_id=f"validation-{index}", seed=1180 + index, exact=False)
                for index in range(2)
            ),
            ProbeSplit.HOLDOUT: (),
        }
        self.sample_references = {
            split: tuple((0, index) for index in range(len(samples)))
            for split, samples in self.samples.items()
        }

    def materialize(self, split: ProbeSplit, offset: int) -> MotionAwareHistorySample:
        return self.samples[split][offset]

    @property
    def episode_counts(self) -> dict[ProbeSplit, int]:
        return {split: len(samples) for split, samples in self.samples.items()}

    def state_values(self, split: ProbeSplit, offset: int):
        sample = self.materialize(split, offset)
        return (
            sample.robot_history,
            sample.object_position_target,
            sample.object_velocity_target,
        )


def _provenance() -> MotionAwareTrainingProvenance:
    return MotionAwareTrainingProvenance(
        implementation_revision="1" * 40,
        implementation_source_sha256="2" * 64,
        implementation_dirty=True,
        bounded_review=False,
        input_sha256={
            "source_bulk_manifest": "3" * 64,
            "cache_run_manifest": "4" * 64,
            "vision_config": "5" * 64,
            "temporal_config": "6" * 64,
            "split_config": "7" * 64,
            "motion_aware_config": "8" * 64,
        },
    )


def test_training_provenance_requires_clean_and_unbounded_for_eligibility() -> None:
    dirty = _provenance()
    assert dirty.artifact_eligible is False
    bounded = replace(dirty, implementation_dirty=False, bounded_review=True)
    assert bounded.artifact_eligible is False
    canonical = replace(dirty, implementation_dirty=False, bounded_review=False)
    assert canonical.artifact_eligible is True


class _TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.readout = nn.Parameter(torch.ones(6))

    def forward(
        self,
        *,
        vision_history: torch.Tensor,
        robot_history_normalized: torch.Tensor,
        history_valid_mask: torch.Tensor,
    ) -> MotionAwareHistoryEstimate:
        batch = vision_history.shape[0]
        context = self.readout[None, :1, None].expand(batch, 2, 192)
        return MotionAwareHistoryEstimate(
            history_context_tokens=context,
            object_position_normalized=self.readout[None, :3].expand(batch, -1),
            object_velocity_normalized=self.readout[None, 3:].expand(batch, -1),
        )


def test_normalization_uses_training_samples_only() -> None:
    normalization = build_motion_aware_normalization(_StateValueCorpus())
    np.testing.assert_allclose(normalization.robot_mean, np.ones(16))
    np.testing.assert_allclose(normalization.robot_std, np.ones(16))
    np.testing.assert_allclose(normalization.position_mean, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(normalization.position_std, np.ones(3))
    np.testing.assert_allclose(normalization.velocity_mean, [4.0, 5.0, 6.0])
    np.testing.assert_allclose(normalization.velocity_std, np.ones(3))
    assert not normalization.robot_mean.flags.writeable


def test_masked_huber_ignores_only_exact_transition_velocity() -> None:
    predicted_position = torch.zeros(2, 3, requires_grad=True)
    predicted_velocity = torch.zeros(2, 3, requires_grad=True)
    target_position = torch.ones(2, 3)
    target_velocity = torch.tensor([[1.0, 1.0, 1.0], [100.0, 100.0, 100.0]])
    exact = torch.tensor([False, True])
    loss = masked_motion_aware_huber(
        predicted_position=predicted_position,
        predicted_velocity=predicted_velocity,
        target_position=target_position,
        target_velocity=target_velocity,
        exact_transition_mask=exact,
        delta=1.0,
    )
    changed = masked_motion_aware_huber(
        predicted_position=predicted_position,
        predicted_velocity=predicted_velocity,
        target_position=target_position,
        target_velocity=torch.tensor(
            [[1.0, 1.0, 1.0], [10_000.0, 10_000.0, 10_000.0]]
        ),
        exact_transition_mask=exact,
        delta=1.0,
    )
    torch.testing.assert_close(loss, changed)
    loss.backward()
    torch.testing.assert_close(predicted_velocity.grad[1], torch.zeros(3))
    assert torch.count_nonzero(predicted_velocity.grad[0]) == 3


def test_all_exact_batch_has_graph_safe_zero_velocity_loss() -> None:
    predicted_position = torch.zeros(2, 3, requires_grad=True)
    predicted_velocity = torch.zeros(2, 3, requires_grad=True)
    loss = masked_motion_aware_huber(
        predicted_position=predicted_position,
        predicted_velocity=predicted_velocity,
        target_position=torch.ones(2, 3),
        target_velocity=torch.full((2, 3), 1000.0),
        exact_transition_mask=torch.ones(2, dtype=torch.bool),
        delta=1.0,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert predicted_velocity.grad is not None
    torch.testing.assert_close(predicted_velocity.grad, torch.zeros(2, 3))


def test_masked_huber_rejects_incompatible_shapes() -> None:
    values = torch.zeros(2, 3)
    with np.testing.assert_raises_regex(ValueError, "shape"):
        masked_motion_aware_huber(
            predicted_position=values,
            predicted_velocity=values,
            target_position=values,
            target_velocity=values,
            exact_transition_mask=torch.zeros(3, dtype=torch.bool),
            delta=1.0,
        )


def test_loss_terms_aggregate_independently_of_exact_batch_composition() -> None:
    predicted_position = torch.zeros(4, 3)
    predicted_velocity = torch.zeros(4, 3)
    target_position = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 10.0
    target_velocity = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 5.0
    exact = torch.tensor([False, True, False, True])
    full = motion_aware_huber_terms(
        predicted_position=predicted_position,
        predicted_velocity=predicted_velocity,
        target_position=target_position,
        target_velocity=target_velocity,
        exact_transition_mask=exact,
        delta=1.0,
    )
    first = motion_aware_huber_terms(
        predicted_position=predicted_position[:2],
        predicted_velocity=predicted_velocity[:2],
        target_position=target_position[:2],
        target_velocity=target_velocity[:2],
        exact_transition_mask=exact[:2],
        delta=1.0,
    )
    second = motion_aware_huber_terms(
        predicted_position=predicted_position[2:],
        predicted_velocity=predicted_velocity[2:],
        target_position=target_position[2:],
        target_velocity=target_velocity[2:],
        exact_transition_mask=exact[2:],
        delta=1.0,
    )
    combined = first + second
    torch.testing.assert_close(full.position_sum, combined.position_sum)
    torch.testing.assert_close(full.velocity_sum, combined.velocity_sum)
    assert full.position_count == combined.position_count
    assert full.velocity_count == combined.velocity_count
    torch.testing.assert_close(full.mean(), combined.mean())


def test_dataset_and_collate_preserve_normalized_inputs_and_exact_mask() -> None:
    corpus = _TrainingCorpus()
    normalization = build_motion_aware_normalization(corpus)
    dataset = MotionAwareDataset(
        corpus=corpus,
        split=ProbeSplit.TRAIN,
        normalization=normalization,
    )
    ordinary = dataset[0]
    exact_sample = _sample(episode_id="exact", seed=1100, exact=True, level=3)
    corpus.samples = {
        **corpus.samples,
        ProbeSplit.TRAIN: (corpus.samples[ProbeSplit.TRAIN][0], exact_sample),
    }
    corpus.sample_references[ProbeSplit.TRAIN] = ((0, 0), (0, 1))
    dataset = MotionAwareDataset(
        corpus=corpus,
        split=ProbeSplit.TRAIN,
        normalization=normalization,
    )
    batch = collate_motion_aware_items([ordinary, dataset[1]])
    assert batch.vision_history.shape == (2, 6, 2, 196, 384)
    assert batch.robot_history_normalized.shape == (2, 6, 16)
    assert batch.object_position_target.shape == (2, 3)
    torch.testing.assert_close(batch.exact_transition_mask, torch.tensor([False, True]))
    assert dataset.valid_velocity_fraction == 0.5


def test_training_publishes_atomic_checkpoint_and_refuses_overwrite(tmp_path: Path) -> None:
    config = load_motion_aware_history_config(
        Path("configs/belief/causal_return/motion_aware_history.yaml")
    )
    config = replace(
        config,
        batch_size=2,
        learning_rate=0.1,
        max_epochs=5,
        early_stopping_patience=5,
        dropout=0.0,
    )
    output = tmp_path / "L1"
    manifest = train_level_motion_aware_history(
        corpus=_TrainingCorpus(),
        config=config,
        output_dir=output,
        device="cpu",
        provenance=_provenance(),
        model_factory=lambda _config: _TinyEncoder(),
    )
    assert manifest == output / "manifest.json"
    assert (output / "model.safetensors").is_file()
    assert (output / "normalization.npz").is_file()
    assert (output / "training_history.json").is_file()
    value = json.loads(manifest.read_text(encoding="utf-8"))
    assert value["implementation_revision"] == "1" * 40
    assert value["implementation_source_sha256"] == "2" * 64
    assert value["input_sha256"]["source_bulk_manifest"] == "3" * 64
    assert value["training_episode_count"] == 4
    assert value["validation_episode_count"] == 2
    assert value["artifact_eligible"] is False
    with pytest.raises(FileExistsError, match="exists"):
        train_level_motion_aware_history(
            corpus=_TrainingCorpus(),
            config=config,
            output_dir=output,
            device="cpu",
            provenance=_provenance(),
            model_factory=lambda _config: _TinyEncoder(),
        )


def test_directory_publication_never_replaces_existing_target(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "new.txt").write_text("new", encoding="utf-8")
    target = tmp_path / "target"
    target.mkdir()
    marker = target / "existing.txt"
    marker.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _rename_directory_no_replace(source, target)
    assert marker.read_text(encoding="utf-8") == "existing"
    assert source.is_dir()
