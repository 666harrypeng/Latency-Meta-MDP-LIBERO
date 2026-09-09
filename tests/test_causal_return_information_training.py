import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from latency_meta_mdp.belief.causal_return.contracts import (
    InformationStateSample,
    load_information_state_config,
)
from latency_meta_mdp.belief.causal_return.information_training import (
    InformationStateDataset,
    build_information_state_normalization,
    collate_information_state_items,
    heteroscedastic_gaussian_nll,
    train_level_information_state,
)
from latency_meta_mdp.vision_probe_data import ProbeSplit


def _sample(*, value: float, seed: int, source_tick: int = 25) -> InformationStateSample:
    return InformationStateSample(
        episode_id=f"l1-seed-{seed:06d}-attempt-000",
        level=1,
        scene_seed=seed,
        source_tick=source_tick,
        source_phase="pregrasp",
        motion_curvature=0.0,
        motion_transition_distance_ticks=-1,
        vision_history=np.full((6, 2, 196, 384), value, dtype=np.float16),
        robot_history=np.full((6, 16), value, dtype=np.float32),
        history_time_ms=np.asarray([-100, -80, -60, -40, -20, 0], dtype=np.int64),
        history_valid_mask=np.ones(6, dtype=np.bool_),
        object_state_target=np.full(6, value, dtype=np.float32),
    )


@dataclass
class _Corpus:
    level: int
    samples: dict[ProbeSplit, tuple[InformationStateSample, ...]]

    @property
    def sample_references(self):
        return {
            split: tuple((0, index) for index in range(len(values)))
            for split, values in self.samples.items()
        }

    def materialize(self, split, offset):
        return self.samples[split][offset]


def _corpus() -> _Corpus:
    return _Corpus(
        level=1,
        samples={
            ProbeSplit.TRAIN: (_sample(value=0.0, seed=1000), _sample(value=2.0, seed=1001)),
            ProbeSplit.VALIDATION: (_sample(value=100.0, seed=1180),),
            ProbeSplit.HOLDOUT: (),
        }
    )


def test_normalization_uses_training_split_only() -> None:
    normalization = build_information_state_normalization(_corpus())

    np.testing.assert_allclose(normalization.robot_mean, 1.0)
    np.testing.assert_allclose(normalization.robot_std, 1.0)
    np.testing.assert_allclose(normalization.object_mean, 1.0)
    np.testing.assert_allclose(normalization.object_std, 1.0)


def test_dataset_and_collate_preserve_audit_metadata_outside_model_inputs() -> None:
    corpus = _corpus()
    normalization = build_information_state_normalization(corpus)
    dataset = InformationStateDataset(
        corpus=corpus,
        split=ProbeSplit.TRAIN,
        normalization=normalization,
    )

    item = dataset[0]
    batch = collate_information_state_items([item, dataset[1]])

    assert item.vision_history.shape == (6, 2, 196, 384)
    assert item.robot_history.shape == (6, 16)
    assert item.object_state_target.shape == (6,)
    assert batch.vision_history.shape == (2, 6, 2, 196, 384)
    assert batch.robot_history.shape == (2, 6, 16)
    assert batch.object_state_target.shape == (2, 6)
    assert batch.source_phase == ("pregrasp", "pregrasp")
    assert not hasattr(batch, "remaining_actions")
    assert not hasattr(batch, "latency_probabilities")


def test_heteroscedastic_loss_has_expected_closed_form() -> None:
    target = torch.zeros(2, 6)
    mean = torch.stack((torch.zeros(6), torch.ones(6)))
    log_scale = torch.zeros(2, 6)

    loss = heteroscedastic_gaussian_nll(
        mean=mean,
        log_scale=log_scale,
        target=target,
    )

    torch.testing.assert_close(loss, torch.tensor(0.25))


def test_level_training_writes_complete_no_overwrite_checkpoint(tmp_path) -> None:
    config = replace(
        load_information_state_config(
            Path("configs/belief/causal_return/information_state.yaml")
        ),
        temporal_fusion_layer_count=1,
        dropout=0.0,
        batch_size=1,
        max_epochs=1,
        early_stopping_patience=1,
    )
    output = tmp_path / "L1"

    manifest_path = train_level_information_state(
        corpus=_corpus(),
        config=config,
        output_dir=output,
        device="cpu",
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["format_id"] == "causal_return_information_state_checkpoint"
    assert manifest["level"] == 1
    assert manifest["training_sample_count"] == 2
    assert manifest["validation_sample_count"] == 1
    assert set(manifest["artifacts"]) == {
        "model.safetensors",
        "normalization.npz",
        "training_history.json",
    }
    assert (output / "model.safetensors").is_file()
    try:
        train_level_information_state(
            corpus=_corpus(),
            config=config,
            output_dir=output,
            device="cpu",
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("information-state training must not overwrite checkpoints")
