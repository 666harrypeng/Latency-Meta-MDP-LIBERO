from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file as save_safetensors
from torch import nn

from latency_meta_mdp.artifacts import sha256_file
from latency_meta_mdp.belief.causal_return.motion_aware_contracts import (
    MotionAwareHistoryEstimate,
    MotionAwareHistorySample,
    load_motion_aware_history_config,
)
from latency_meta_mdp.belief.causal_return.motion_aware_evaluation import (
    benchmark_motion_aware_history_encoder,
    compute_motion_aware_metrics,
    evaluate_level_motion_aware_history,
    evaluate_motion_aware_admission,
    match_baseline_contexts,
    paired_episode_bootstrap,
)
from latency_meta_mdp.vision_probe_data import ProbeSplit


class _TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.readout = nn.Parameter(torch.zeros(6))

    def forward(
        self,
        *,
        vision_history: torch.Tensor,
        robot_history_normalized: torch.Tensor,
        history_valid_mask: torch.Tensor,
    ) -> MotionAwareHistoryEstimate:
        batch = vision_history.shape[0]
        return MotionAwareHistoryEstimate(
            history_context_tokens=self.readout[None, :1, None].expand(batch, 2, 192),
            object_position_normalized=self.readout[None, :3].expand(batch, -1),
            object_velocity_normalized=self.readout[None, 3:].expand(batch, -1),
        )


class _EvaluationCorpus:
    level = 1

    def __init__(self) -> None:
        self.samples = tuple(
            MotionAwareHistorySample(
                episode_id=f"episode-{index}",
                level=1,
                scene_seed=1180 + index,
                source_tick=25,
                source_phase="pregrasp",
                transition_offset_ticks=0,
                transition_offset_valid=False,
                vision_history=np.zeros((6, 2, 196, 384), dtype=np.float16),
                robot_history=np.zeros((6, 16), dtype=np.float32),
                history_valid_mask=np.ones(6, dtype=np.bool_),
                object_position_target=np.zeros(3, dtype=np.float32),
                object_velocity_target=np.zeros(3, dtype=np.float32),
            )
            for index in range(2)
        )
        self.sample_references = {
            ProbeSplit.TRAIN: (),
            ProbeSplit.VALIDATION: ((0, 0), (0, 1)),
            ProbeSplit.HOLDOUT: (),
        }

    def materialize(self, split: ProbeSplit, offset: int) -> MotionAwareHistorySample:
        assert split is ProbeSplit.VALIDATION
        return self.samples[offset]

    def transition_metadata(self, split: ProbeSplit, offset: int):
        sample = self.materialize(split, offset)
        return sample.transition_offset_ticks, sample.transition_offset_valid


def _checkpoint(tmp_path: Path) -> tuple[Path, object]:
    config = load_motion_aware_history_config(
        Path("configs/belief/causal_return/motion_aware_history.yaml")
    )
    root = tmp_path / "checkpoint"
    root.mkdir()
    save_safetensors(_TinyEncoder().state_dict(), root / "model.safetensors")
    np.savez(
        root / "normalization.npz",
        robot_mean=np.zeros(16, dtype=np.float32),
        robot_std=np.ones(16, dtype=np.float32),
        position_mean=np.zeros(3, dtype=np.float32),
        position_std=np.ones(3, dtype=np.float32),
        velocity_mean=np.zeros(3, dtype=np.float32),
        velocity_std=np.ones(3, dtype=np.float32),
    )
    (root / "training_history.json").write_text("[]\n", encoding="utf-8")
    artifacts = {
        name: sha256_file(root / name)
        for name in ("model.safetensors", "normalization.npz", "training_history.json")
    }
    manifest = {
        "schema_version": 1,
        "format_id": "causal_return_motion_aware_history_checkpoint",
        "scientific_gate_pass": False,
        "artifact_eligible": False,
        "level": 1,
        "config": asdict(config),
        "implementation_revision": "1" * 40,
        "implementation_source_sha256": "2" * 64,
        "implementation_dirty": True,
        "bounded_review": True,
        "input_sha256": {
            "source_bulk_manifest": "a" * 64,
            "cache_run_manifest": "b" * 64,
            "vision_config": "c" * 64,
            "temporal_config": "d" * 64,
            "split_config": "e" * 64,
            "motion_aware_config": "f" * 64,
        },
        "training_sample_count": 2,
        "validation_sample_count": 2,
        "training_episode_count": 1,
        "validation_episode_count": 2,
        "best_epoch": 0,
        "best_validation_loss": 0.0,
        "epochs_completed": 1,
        "wall_seconds": 0.1,
        "parameter_count": 6,
        "artifacts": artifacts,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root, config


def _metric_arrays():
    offsets = np.asarray([-6, -2, 0, 1, 3, 8], dtype=np.int64)
    valid = np.ones(6, dtype=np.bool_)
    target = np.zeros((6, 6), dtype=np.float32)
    prediction = np.zeros((6, 6), dtype=np.float32)
    prediction[:, 0] = np.arange(6, dtype=np.float32) / 1000.0
    prediction[:, 3] = np.arange(6, dtype=np.float32) / 100.0
    return prediction, target, offsets, valid


def test_signed_transition_metrics_partition_every_l3_context_once() -> None:
    prediction, target, offsets, valid = _metric_arrays()
    metrics = compute_motion_aware_metrics(
        object_state_prediction=prediction,
        object_state_target=target,
        source_phase=np.asarray(
            ["pregrasp", "pregrasp", "approach", "approach", "close", "lift"]
        ),
        transition_offset_ticks=offsets,
        transition_offset_valid=valid,
    )
    assert metrics["overall"]["context_count"] == 6
    regimes = metrics["transition"]
    assert regimes["before"]["context_count"] == 1
    assert regimes["exact"]["context_count"] == 1
    assert regimes["after_1"]["context_count"] == 1
    assert regimes["after_2_to_5"]["context_count"] == 1
    assert regimes["far"]["context_count"] == 2
    assert sum(value["context_count"] for value in regimes.values()) == 6
    assert regimes["exact"]["object_velocity_rmse_m_s"] == pytest.approx(0.02)


def test_non_l3_transition_metadata_routes_all_contexts_to_far() -> None:
    prediction, target, offsets, _ = _metric_arrays()
    metrics = compute_motion_aware_metrics(
        object_state_prediction=prediction,
        object_state_target=target,
        source_phase=np.asarray(["pregrasp"] * 6),
        transition_offset_ticks=np.zeros(6, dtype=np.int64),
        transition_offset_valid=np.zeros(6, dtype=np.bool_),
    )
    assert metrics["transition"]["far"]["context_count"] == 6
    assert all(
        metrics["transition"][name]["context_count"] == 0
        for name in ("before", "exact", "after_1", "after_2_to_5")
    )


def _context_arrays(order=(0, 1, 2)):
    episode_id = np.asarray(["episode-b", "episode-a", "episode-a"])[list(order)]
    source_tick = np.asarray([30, 25, 26], dtype=np.int64)[list(order)]
    phase = np.asarray(["approach", "pregrasp", "pregrasp"])[list(order)]
    target = np.arange(18, dtype=np.float32).reshape(3, 6)[list(order)] / 100.0
    prediction = target + 0.01
    return {
        "episode_id": episode_id,
        "source_tick": source_tick,
        "source_phase": phase,
        "object_state_target": target,
        "object_state_prediction": prediction,
    }


def test_baseline_matching_reorders_by_identity_and_uses_canonical_tolerance() -> None:
    candidate = _context_arrays(order=(0, 1, 2))
    baseline = _context_arrays(order=(2, 0, 1))
    baseline["object_state_target"] = baseline["object_state_target"].astype(np.float64)
    baseline["object_state_target"] += 1e-8
    matched = match_baseline_contexts(
        candidate=candidate,
        baseline=baseline,
        candidate_source_sha256="a" * 64,
        baseline_source_sha256="a" * 64,
    )
    assert matched.episode_id.tolist() == ["episode-a", "episode-a", "episode-b"]
    assert matched.source_tick.tolist() == [25, 26, 30]
    np.testing.assert_allclose(
        matched.candidate_target,
        matched.baseline_target,
        rtol=1e-6,
        atol=1e-7,
    )


def test_baseline_matching_rejects_identity_source_and_target_drift() -> None:
    candidate = _context_arrays()
    baseline = _context_arrays()
    with pytest.raises(ValueError, match="source"):
        match_baseline_contexts(
            candidate=candidate,
            baseline=baseline,
            candidate_source_sha256="a" * 64,
            baseline_source_sha256="b" * 64,
        )
    drifted = {**baseline, "source_tick": np.asarray([30, 25, 99])}
    with pytest.raises(ValueError, match="identity"):
        match_baseline_contexts(
            candidate=candidate,
            baseline=drifted,
            candidate_source_sha256="a" * 64,
            baseline_source_sha256="a" * 64,
        )
    drifted = {
        **baseline,
        "object_state_target": baseline["object_state_target"] + 0.1,
    }
    with pytest.raises(ValueError, match="target"):
        match_baseline_contexts(
            candidate=candidate,
            baseline=drifted,
            candidate_source_sha256="a" * 64,
            baseline_source_sha256="a" * 64,
        )


def test_episode_clustered_bootstrap_is_deterministic_and_detects_improvement() -> None:
    episodes = np.asarray(["a", "a", "b", "b", "c", "c"])
    baseline_error = np.asarray([3.0, 2.0, 4.0, 3.0, 2.0, 2.0])
    candidate_error = baseline_error - 1.0
    first = paired_episode_bootstrap(
        episode_id=episodes,
        baseline_error=baseline_error,
        candidate_error=candidate_error,
        seed=20260830,
        resample_count=10_000,
    )
    second = paired_episode_bootstrap(
        episode_id=episodes,
        baseline_error=baseline_error,
        candidate_error=candidate_error,
        seed=20260830,
        resample_count=10_000,
    )
    assert first == second
    assert first["lower_95"] > 0.0
    assert first["scientific_improvement_pass"] is True


def test_l3_admission_requires_position_before_and_after_gates() -> None:
    episode_id = np.asarray(["a", "a", "b", "b", "c", "c"])
    offsets = np.asarray([-6, -2, 0, 2, 3, 8])
    valid = np.ones(6, dtype=np.bool_)
    target = np.zeros((6, 6), dtype=np.float32)
    baseline = np.full((6, 6), 0.02, dtype=np.float32)
    candidate = np.full((6, 6), 0.01, dtype=np.float32)
    phase = np.asarray(["pregrasp"] * 6)
    candidate_metrics = compute_motion_aware_metrics(
        object_state_prediction=candidate,
        object_state_target=target,
        source_phase=phase,
        transition_offset_ticks=offsets,
        transition_offset_valid=valid,
    )
    baseline_metrics = compute_motion_aware_metrics(
        object_state_prediction=baseline,
        object_state_target=target,
        source_phase=phase,
        transition_offset_ticks=offsets,
        transition_offset_valid=valid,
    )
    result = evaluate_motion_aware_admission(
        level=3,
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
        episode_id=episode_id,
        transition_offset_ticks=offsets,
        transition_offset_valid=valid,
        candidate_prediction=candidate,
        baseline_prediction=baseline,
        target=target,
    )
    assert result["scientific_gate_pass"] is True
    assert result["blockers"] == []

    formal = evaluate_motion_aware_admission(
        level=3,
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
        episode_id=episode_id,
        transition_offset_ticks=offsets,
        transition_offset_valid=valid,
        candidate_prediction=candidate,
        baseline_prediction=baseline,
        target=target,
        require_formal_episode_count=True,
    )
    assert formal["scientific_gate_pass"] is False
    assert "formal_validation_episode_count" in formal["blockers"]

    degraded = candidate.copy()
    degraded[1, :3] = 0.1
    degraded_metrics = compute_motion_aware_metrics(
        object_state_prediction=degraded,
        object_state_target=target,
        source_phase=phase,
        transition_offset_ticks=offsets,
        transition_offset_valid=valid,
    )
    result = evaluate_motion_aware_admission(
        level=3,
        candidate_metrics=degraded_metrics,
        baseline_metrics=baseline_metrics,
        episode_id=episode_id,
        transition_offset_ticks=offsets,
        transition_offset_valid=valid,
        candidate_prediction=degraded,
        baseline_prediction=baseline,
        target=target,
    )
    assert result["scientific_gate_pass"] is False
    assert "before_position_regression" in result["blockers"]


def test_runtime_benchmark_reports_requested_batches(tmp_path: Path) -> None:
    _root, config = _checkpoint(tmp_path)
    metrics = benchmark_motion_aware_history_encoder(
        model=_TinyEncoder().eval(),
        config=config,
        device="cpu",
        batch_sizes=(1, 2),
        warmup_count=2,
        trial_count=5,
    )
    assert set(metrics["batches"]) == {"1", "2"}
    assert metrics["batches"]["1"]["p50_ms"] >= 0.0
    assert metrics["parameter_count"] == 6


def test_level_evaluation_publishes_matched_atomic_artifact(tmp_path: Path) -> None:
    checkpoint, _config = _checkpoint(tmp_path)
    baseline = {
        "episode_id": np.asarray(["episode-1", "episode-0"]),
        "source_tick": np.asarray([25, 25], dtype=np.int64),
        "source_phase": np.asarray(["pregrasp", "pregrasp"]),
        "object_state_target": np.zeros((2, 6), dtype=np.float32),
        "object_state_prediction": np.full((2, 6), 0.1, dtype=np.float32),
    }
    output = tmp_path / "evaluation"
    manifest = evaluate_level_motion_aware_history(
        corpus=_EvaluationCorpus(),
        checkpoint_dir=checkpoint,
        baseline=baseline,
        baseline_source_sha256="a" * 64,
        output_dir=output,
        device="cpu",
        model_factory=lambda _config: _TinyEncoder(),
        runtime_warmup_count=2,
        runtime_trial_count=5,
        runtime_batch_sizes=(1, 2),
    )
    value = json.loads(manifest.read_text(encoding="utf-8"))
    assert value["scientific_gate_pass"] is True
    assert value["artifact_eligible"] is False
    assert (output / "predictions.npz").is_file()
    assert (output / "metrics.json").is_file()
    assert (output / "comparison.json").is_file()
    assert (output / "runtime.json").is_file()
