import numpy as np

from latency_meta_mdp.io.artifacts import sha256_file
from latency_meta_mdp.legacy.belief.flow.buffer_causality_evaluation import (
    compute_buffer_causality_metrics,
    latency_weighted_context_mean,
    verify_flow_run_eligibility,
)


def test_paired_metrics_measure_effect_direction_gain_and_false_coupling() -> None:
    targets = np.zeros((1, 2, 2, 22), dtype=np.float32)
    targets[:, 1, :, :7] = 1.0
    predictions = np.zeros((1, 2, 2, 3, 22), dtype=np.float32)
    predictions[:, 1, :, :, :7] = 0.5
    predictions[:, 1, :, :, 16:19] = 2.0
    tokens = np.zeros((1, 2, 8, 192), dtype=np.float32)
    tokens[:, 1] = 0.25

    metrics = compute_buffer_causality_metrics(
        predictions=predictions,
        targets=targets,
        belief_tokens=tokens,
        branch_ids=("expert", "hold"),
    )

    np.testing.assert_allclose(metrics.qpos_error[0, 1], 0.5)
    np.testing.assert_allclose(metrics.qpos_expert_control_error[0, 1], 1.0)
    np.testing.assert_allclose(metrics.qpos_conditioning_gain[0, 1], 0.5)
    np.testing.assert_allclose(metrics.qpos_effect_cosine[0, 1], 1.0)
    np.testing.assert_allclose(metrics.qpos_effect_magnitude_ratio[0, 1], 0.5)
    np.testing.assert_allclose(metrics.gt_object_position_effect[0, 1], 0.0)
    np.testing.assert_allclose(metrics.predicted_object_position_effect[0, 1], np.sqrt(12.0))
    np.testing.assert_array_equal(metrics.object_effect_is_invariant[0, 1], True)
    np.testing.assert_allclose(metrics.token_delta_rms[0, 1], 0.25)


def test_paired_metrics_keep_zero_effect_cosine_explicitly_invalid() -> None:
    predictions = np.zeros((1, 2, 1, 2, 22), dtype=np.float32)
    targets = np.zeros((1, 2, 1, 22), dtype=np.float32)
    tokens = np.zeros((1, 2, 8, 192), dtype=np.float32)

    metrics = compute_buffer_causality_metrics(
        predictions=predictions,
        targets=targets,
        belief_tokens=tokens,
        branch_ids=("expert", "hold"),
    )

    np.testing.assert_array_equal(metrics.qpos_effect_cosine_valid[0, 1], False)
    np.testing.assert_allclose(metrics.qpos_effect_cosine[0, 1], 0.0)
    np.testing.assert_allclose(metrics.qpos_effect_magnitude_ratio[0, 1], 0.0)


def test_checkpoint_eligibility_is_bound_by_parent_run_manifest(tmp_path) -> None:
    import json

    checkpoint = tmp_path / "run" / "L2"
    checkpoint.mkdir(parents=True)
    level_manifest = checkpoint / "manifest.json"
    level_manifest.write_text('{"format_id":"level_flow_belief_v1","level":2}\n')
    parent = {
        "format_id": "flow_belief_multilaw_run_v2",
        "eligible": True,
        "blockers": [],
        "levels": [2],
        "level_manifests": {"L2": "L2/manifest.json"},
        "artifacts": {"L2/manifest.json": sha256_file(level_manifest)},
    }
    (checkpoint.parent / "manifest.json").write_text(json.dumps(parent))

    assert verify_flow_run_eligibility(checkpoint_dir=checkpoint, level=2) is True

    parent["eligible"] = False
    (checkpoint.parent / "manifest.json").write_text(json.dumps(parent))
    assert verify_flow_run_eligibility(checkpoint_dir=checkpoint, level=2) is False


def test_latency_weighted_mean_uses_each_context_law_before_context_average() -> None:
    values = np.asarray(
        [
            [[1.0, 3.0], [10.0, 20.0]],
            [[5.0, 9.0], [30.0, 50.0]],
        ]
    )
    probabilities = np.asarray([[0.75, 0.25], [0.25, 0.75]])

    result = latency_weighted_context_mean(values, probabilities)

    np.testing.assert_allclose(result, [4.75, 28.75])
