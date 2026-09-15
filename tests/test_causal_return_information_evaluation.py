import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from latency_meta_mdp.legacy.belief.causal_return.contracts import (
    InformationStateSample,
    load_information_state_config,
)
from latency_meta_mdp.legacy.belief.causal_return.information_evaluation import (
    compute_information_state_metrics,
    evaluate_level_information_state,
)
from latency_meta_mdp.legacy.belief.causal_return.information_training import (
    train_level_information_state,
)
from latency_meta_mdp.legacy.vision_probe_data import ProbeSplit


def _sample(value: float, seed: int) -> InformationStateSample:
    return InformationStateSample(
        episode_id=f"l1-seed-{seed:06d}-attempt-000",
        level=1,
        scene_seed=seed,
        source_tick=25,
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
    level: int = 1

    def __post_init__(self):
        self.samples = {
            ProbeSplit.TRAIN: (_sample(0.0, 1000), _sample(1.0, 1001)),
            ProbeSplit.VALIDATION: (_sample(0.5, 1180),),
            ProbeSplit.HOLDOUT: (),
        }
        self.sample_references = {
            split: tuple((0, index) for index in range(len(rows)))
            for split, rows in self.samples.items()
        }

    def materialize(self, split, offset):
        return self.samples[split][offset]


def test_information_metrics_report_physical_errors_coverage_and_subsets() -> None:
    target = np.zeros((4, 6), dtype=np.float64)
    mean = np.asarray(
        [
            [0.001, 0.0, 0.0, 0.01, 0.0, 0.0],
            [0.002, 0.0, 0.0, 0.02, 0.0, 0.0],
            [0.003, 0.0, 0.0, 0.03, 0.0, 0.0],
            [0.004, 0.0, 0.0, 0.04, 0.0, 0.0],
        ]
    )
    scale = np.full((4, 6), 0.1)
    phases = np.asarray(["pregrasp", "approach", "close", "lift"])
    curvature = np.asarray([0.0, 0.1, 0.2, 0.3])
    transition_distance = np.asarray([-1, 9, 5, 0])
    normalized_nll = np.asarray([0.1, 0.2, 0.3, 0.4])

    metrics = compute_information_state_metrics(
        object_state_mean=mean,
        object_state_scale=scale,
        object_state_target=target,
        source_phase=phases,
        motion_curvature=curvature,
        motion_transition_distance_ticks=transition_distance,
        normalized_nll=normalized_nll,
    )

    assert np.isclose(metrics["overall"]["object_position_rmse_m"], np.sqrt(7.5e-6))
    assert np.isclose(metrics["overall"]["object_velocity_rmse_m_s"], np.sqrt(7.5e-4))
    np.testing.assert_allclose(metrics["overall"]["signed_position_bias_m"], [0.0025, 0.0, 0.0])
    assert metrics["overall"]["coverage_68"] == 1.0
    assert metrics["overall"]["coverage_95"] == 1.0
    assert metrics["overall"]["normalized_nll"] == 0.25
    assert metrics["overall"]["invalid_value_count"] == 0
    assert set(metrics["phase"]) == {"pregrasp", "approach", "close", "lift"}
    assert metrics["high_curvature"]["context_count"] == 1
    assert metrics["transition_near"]["context_count"] == 2


def test_information_metrics_reject_nonfinite_predictions() -> None:
    mean = np.zeros((1, 6))
    mean[0, 0] = np.nan

    try:
        compute_information_state_metrics(
            object_state_mean=mean,
            object_state_scale=np.ones((1, 6)),
            object_state_target=np.zeros((1, 6)),
            source_phase=np.asarray(["pregrasp"]),
            motion_curvature=np.asarray([0.0]),
            motion_transition_distance_ticks=np.asarray([-1]),
            normalized_nll=np.asarray([0.0]),
        )
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("nonfinite information-state predictions must fail")


def test_level_evaluation_reloads_checkpoint_and_writes_physical_artifact(tmp_path) -> None:
    config = replace(
        load_information_state_config(
            Path("configs/legacy/belief/causal_return/information_state.yaml")
        ),
        temporal_fusion_layer_count=1,
        dropout=0.0,
        batch_size=1,
        max_epochs=1,
        early_stopping_patience=1,
    )
    corpus = _Corpus()
    checkpoint = tmp_path / "checkpoint"
    train_level_information_state(
        corpus=corpus,
        config=config,
        output_dir=checkpoint,
        device="cpu",
    )

    output = tmp_path / "evaluation"
    manifest_path = evaluate_level_information_state(
        corpus=corpus,
        checkpoint_dir=checkpoint,
        output_dir=output,
        device="cpu",
    )

    manifest = json.loads(manifest_path.read_text())
    metrics = json.loads((output / "metrics.json").read_text())
    assert manifest["format_id"] == "causal_return_information_state_evaluation"
    assert manifest["level"] == 1
    assert manifest["context_count"] == 1
    assert metrics["overall"]["context_count"] == 1
    with np.load(output / "predictions.npz", allow_pickle=False) as arrays:
        assert arrays["object_state_mean"].shape == (1, 6)
        assert arrays["object_state_scale"].shape == (1, 6)
        assert arrays["object_state_target"].shape == (1, 6)
