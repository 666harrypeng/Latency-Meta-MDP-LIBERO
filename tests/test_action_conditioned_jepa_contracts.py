from __future__ import annotations

from dataclasses import fields

import pytest
import torch


def _launch_context(*, batch_size: int = 2):
    from latency_meta_mdp.belief.jepa.contracts import LaunchContextBatch

    return LaunchContextBatch(
        vision_history=torch.zeros(batch_size, 6, 2, 196, 384, dtype=torch.float16),
        proprio_history=torch.zeros(batch_size, 6, 16, dtype=torch.float32),
        executed_controls=torch.zeros(batch_size, 5, 1, 7, dtype=torch.float32),
        executable_controls=torch.zeros(batch_size, 20, 1, 7, dtype=torch.float32),
    )


def _rollout(*, batch_size: int = 1):
    from latency_meta_mdp.belief.jepa.contracts import FutureLatentRollout

    return FutureLatentRollout(
        native_delay_ticks=torch.arange(1, 21, dtype=torch.int64),
        future_visual_latents=torch.zeros(batch_size, 20, 2, 196, 384, dtype=torch.float16),
        future_proprio=torch.zeros(batch_size, 20, 16, dtype=torch.float32),
    )


def test_launch_context_requires_exact_shapes_dtypes_and_semantics() -> None:
    batch = _launch_context(batch_size=2)

    assert batch.batch_size == 2
    assert batch.device == torch.device("cpu")
    assert batch.proprio_space == "normalized"
    assert batch.control_space == "controller_native"
    assert {field.name for field in fields(batch)} == {
        "vision_history",
        "proprio_history",
        "executed_controls",
        "executable_controls",
    }
    batch.validate_finite()


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    (
        (
            "vision_history",
            torch.zeros(2, 6, 2, 195, 384, dtype=torch.float16),
            "vision_history",
        ),
        (
            "proprio_history",
            torch.zeros(2, 6, 16, dtype=torch.float16),
            "proprio_history",
        ),
        (
            "executed_controls",
            torch.zeros(2, 6, 1, 7, dtype=torch.float32),
            "executed_controls",
        ),
        (
            "executable_controls",
            torch.zeros(2, 20, 1, 6, dtype=torch.float32),
            "executable_controls",
        ),
    ),
)
def test_launch_context_rejects_shape_or_dtype_drift(field, replacement, match) -> None:
    from latency_meta_mdp.belief.jepa.contracts import LaunchContextBatch

    values = {
        "vision_history": torch.zeros(2, 6, 2, 196, 384, dtype=torch.float16),
        "proprio_history": torch.zeros(2, 6, 16, dtype=torch.float32),
        "executed_controls": torch.zeros(2, 5, 1, 7, dtype=torch.float32),
        "executable_controls": torch.zeros(2, 20, 1, 7, dtype=torch.float32),
    }
    values[field] = replacement

    with pytest.raises(ValueError, match=match):
        LaunchContextBatch(**values)


def test_launch_context_rejects_device_mismatch_before_compute() -> None:
    from latency_meta_mdp.belief.jepa.contracts import LaunchContextBatch

    with pytest.raises(ValueError, match="device"):
        LaunchContextBatch(
            vision_history=torch.zeros(1, 6, 2, 196, 384, dtype=torch.float16),
            proprio_history=torch.empty(1, 6, 16, dtype=torch.float32, device="meta"),
            executed_controls=torch.zeros(1, 5, 1, 7, dtype=torch.float32),
            executable_controls=torch.zeros(1, 20, 1, 7, dtype=torch.float32),
        )


def test_finite_validation_is_explicit_for_large_model_tensors() -> None:
    context = _launch_context(batch_size=1)
    context.vision_history[0, 0, 0, 0, 0] = torch.nan

    with pytest.raises(ValueError, match="finite"):
        context.validate_finite()


def test_future_rollout_contract_exposes_physical_proprio() -> None:
    rollout = _rollout(batch_size=2)

    assert rollout.batch_size == 2
    assert rollout.future_proprio_units == "physical_si"
    rollout.validate_finite()


def test_future_rollout_rejects_shape_dtype_or_device_drift() -> None:
    from latency_meta_mdp.belief.jepa.contracts import FutureLatentRollout

    with pytest.raises(ValueError, match="future_visual_latents"):
        FutureLatentRollout(
            native_delay_ticks=torch.arange(1, 21, dtype=torch.int64),
            future_visual_latents=torch.zeros(1, 20, 2, 196, 384, dtype=torch.float32),
            future_proprio=torch.zeros(1, 20, 16, dtype=torch.float32),
        )
    with pytest.raises(ValueError, match="future_proprio"):
        FutureLatentRollout(
            native_delay_ticks=torch.arange(1, 21, dtype=torch.int64),
            future_visual_latents=torch.zeros(1, 20, 2, 196, 384, dtype=torch.float16),
            future_proprio=torch.zeros(1, 19, 16, dtype=torch.float32),
        )
    with pytest.raises(ValueError, match="device"):
        FutureLatentRollout(
            native_delay_ticks=torch.arange(1, 21, dtype=torch.int64),
            future_visual_latents=torch.empty(
                1, 20, 2, 196, 384, dtype=torch.float16, device="meta"
            ),
            future_proprio=torch.zeros(1, 20, 16, dtype=torch.float32),
        )


def test_return_belief_accepts_exact_d20_probability_mixture() -> None:
    from latency_meta_mdp.belief.jepa.contracts import (
        ReturnLatentBeliefBatch,
    )

    rollout = _rollout(batch_size=2)
    belief = ReturnLatentBeliefBatch(
        delay_ticks=torch.arange(1, 21, dtype=torch.int64),
        delay_probabilities=torch.full((2, 20), 1.0 / 20, dtype=torch.float32),
        future_visual_latents=rollout.future_visual_latents,
        future_proprio=rollout.future_proprio,
    )

    assert belief.batch_size == 2
    assert belief.maximum_delay_ticks == 20
    assert belief.future_proprio_units == "physical_si"
    assert belief.future_visual_latents is rollout.future_visual_latents
    assert belief.future_proprio is rollout.future_proprio
    belief.validate_finite()


def test_return_belief_rejects_invalid_delay_mass() -> None:
    from latency_meta_mdp.belief.jepa.contracts import (
        ReturnLatentBeliefBatch,
    )

    rollout = _rollout(batch_size=1)
    probabilities = torch.full((1, 20), 1.0 / 20, dtype=torch.float32)
    probabilities[0, 0] = -0.1

    with pytest.raises(ValueError, match="probabilities"):
        ReturnLatentBeliefBatch(
            delay_ticks=torch.arange(1, 21, dtype=torch.int64),
            delay_probabilities=probabilities,
            future_visual_latents=rollout.future_visual_latents,
            future_proprio=rollout.future_proprio,
        )


@pytest.mark.parametrize(
    "delay_ticks",
    (
        torch.arange(0, 20, dtype=torch.int64),
        torch.arange(1, 21, dtype=torch.int32),
        torch.arange(1, 20, dtype=torch.int64),
    ),
)
def test_return_belief_rejects_noncanonical_delay_ticks(delay_ticks) -> None:
    from latency_meta_mdp.belief.jepa.contracts import (
        ReturnLatentBeliefBatch,
    )

    rollout = _rollout(batch_size=1)
    with pytest.raises(ValueError, match="delay"):
        ReturnLatentBeliefBatch(
            delay_ticks=delay_ticks,
            delay_probabilities=torch.full((1, 20), 1.0 / 20),
            future_visual_latents=rollout.future_visual_latents,
            future_proprio=rollout.future_proprio,
        )


def test_return_belief_rejects_probability_sum_or_nonfinite_values() -> None:
    from latency_meta_mdp.belief.jepa.contracts import (
        ReturnLatentBeliefBatch,
    )

    rollout = _rollout(batch_size=1)
    for probabilities in (
        torch.full((1, 20), 0.04),
        torch.full((1, 20), float("nan")),
    ):
        with pytest.raises(ValueError, match="probabilities"):
            ReturnLatentBeliefBatch(
                delay_ticks=torch.arange(1, 21, dtype=torch.int64),
                delay_probabilities=probabilities,
                future_visual_latents=rollout.future_visual_latents,
                future_proprio=rollout.future_proprio,
            )


def test_return_belief_rejects_empty_or_mismatched_batches() -> None:
    from latency_meta_mdp.belief.jepa.contracts import (
        ReturnLatentBeliefBatch,
    )

    with pytest.raises(ValueError, match="non-empty"):
        ReturnLatentBeliefBatch(
            delay_ticks=torch.arange(1, 21, dtype=torch.int64),
            delay_probabilities=torch.empty(0, 20, dtype=torch.float32),
            future_visual_latents=torch.empty(0, 20, 2, 196, 384, dtype=torch.float16),
            future_proprio=torch.empty(0, 20, 16, dtype=torch.float32),
        )
    rollout = _rollout(batch_size=2)
    with pytest.raises(ValueError, match="future_visual_latents"):
        ReturnLatentBeliefBatch(
            delay_ticks=torch.arange(1, 21, dtype=torch.int64),
            delay_probabilities=torch.full((1, 20), 1.0 / 20, dtype=torch.float32),
            future_visual_latents=rollout.future_visual_latents,
            future_proprio=rollout.future_proprio,
        )


def test_stride4_contract_preserves_macro_controls_and_native_anchors() -> None:
    """Catches flattening away macro-control order or retaining fixed D20 tensor shapes."""

    from latency_meta_mdp.belief.jepa.contracts import (
        FutureLatentRollout,
        LaunchContextBatch,
        ReturnLatentBeliefBatch,
    )

    context = LaunchContextBatch(
        vision_history=torch.zeros(2, 3, 2, 196, 384, dtype=torch.float16),
        proprio_history=torch.zeros(2, 3, 16, dtype=torch.float32),
        executed_controls=torch.zeros(2, 2, 4, 7, dtype=torch.float32),
        executable_controls=torch.zeros(2, 5, 4, 7, dtype=torch.float32),
    )
    anchors = torch.tensor([4, 8, 12, 16, 20], dtype=torch.int64)
    rollout = FutureLatentRollout(
        native_delay_ticks=anchors,
        future_visual_latents=torch.zeros(2, 5, 2, 196, 384, dtype=torch.float16),
        future_proprio=torch.zeros(2, 5, 16, dtype=torch.float32),
    )
    belief = ReturnLatentBeliefBatch(
        delay_ticks=anchors,
        delay_probabilities=torch.full((2, 5), 0.2, dtype=torch.float32),
        future_visual_latents=rollout.future_visual_latents,
        future_proprio=rollout.future_proprio,
    )

    assert context.history_observation_count == 3
    assert context.model_stride_ticks == 4
    assert context.native_rollout_steps == 5
    assert rollout.native_delay_count == 5
    assert belief.native_delay_count == 5
    assert belief.maximum_delay_ticks == 20
