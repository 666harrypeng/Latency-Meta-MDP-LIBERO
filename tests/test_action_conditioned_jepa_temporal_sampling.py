from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

CONFIG_ROOT = Path("configs/belief/action_conditioned_jepa")


@pytest.mark.parametrize(
    ("config_name", "config_id", "stride", "history", "rollout", "history_offsets"),
    (
        (
            "dense_20ms_history_100ms.yaml",
            "dense_20ms_history_100ms",
            1,
            6,
            20,
            (-5, -4, -3, -2, -1, 0),
        ),
        (
            "stride2_40ms_history_120ms.yaml",
            "stride2_40ms_history_120ms",
            2,
            4,
            10,
            (-6, -4, -2, 0),
        ),
        (
            "stride4_80ms_history_160ms.yaml",
            "stride4_80ms_history_160ms",
            4,
            3,
            5,
            (-8, -4, 0),
        ),
        (
            "stride5_100ms_history_200ms.yaml",
            "stride5_100ms_history_200ms",
            5,
            3,
            4,
            (-10, -5, 0),
        ),
    ),
)
def test_temporal_configs_resolve_literal_source_offsets(
    config_name: str,
    config_id: str,
    stride: int,
    history: int,
    rollout: int,
    history_offsets: tuple[int, ...],
) -> None:
    """Catches a stride applied to counts but not to physical source indices."""

    from latency_meta_mdp.belief.action_conditioned_jepa.config import (
        load_jepa_temporal_sampling,
    )

    sampling = load_jepa_temporal_sampling(CONFIG_ROOT / config_name)

    assert sampling.config_id == config_id
    assert sampling.model_stride_ticks == stride
    assert sampling.history_observation_count == history
    assert sampling.native_rollout_steps == rollout
    assert sampling.training_rollout_steps == 2
    assert sampling.history_source_offsets == history_offsets
    assert sampling.native_future_offsets == tuple(range(stride, 21, stride))
    assert sampling.history_span_ticks == -history_offsets[0]
    assert sampling.history_span_us(formal_tick_us=20_000) == -history_offsets[0] * 20_000


@pytest.mark.parametrize(
    ("config_name", "expected_past", "expected_future"),
    (
        (
            "dense_20ms_history_100ms.yaml",
            ((-5,), (-4,), (-3,), (-2,), (-1,)),
            tuple((tick,) for tick in range(20)),
        ),
        (
            "stride2_40ms_history_120ms.yaml",
            ((-6, -5), (-4, -3), (-2, -1)),
            tuple((tick, tick + 1) for tick in range(0, 20, 2)),
        ),
        (
            "stride4_80ms_history_160ms.yaml",
            ((-8, -7, -6, -5), (-4, -3, -2, -1)),
            tuple(tuple(range(tick, tick + 4)) for tick in range(0, 20, 4)),
        ),
        (
            "stride5_100ms_history_200ms.yaml",
            ((-10, -9, -8, -7, -6), (-5, -4, -3, -2, -1)),
            tuple(tuple(range(tick, tick + 5)) for tick in range(0, 20, 5)),
        ),
    ),
)
def test_temporal_configs_preserve_every_ordered_micro_control(
    config_name: str,
    expected_past: tuple[tuple[int, ...], ...],
    expected_future: tuple[tuple[int, ...], ...],
) -> None:
    """Catches averaging, endpoint-only selection, or gaps inside macro controls."""

    from latency_meta_mdp.belief.action_conditioned_jepa.config import (
        load_jepa_temporal_sampling,
    )

    sampling = load_jepa_temporal_sampling(CONFIG_ROOT / config_name)

    assert sampling.past_macro_control_offsets == expected_past
    assert sampling.future_macro_control_offsets == expected_future
    assert tuple(offset for block in expected_future for offset in block) == tuple(range(20))


def test_history_readiness_uses_available_source_ticks_not_buffer_cursor() -> None:
    """Catches reintroducing the old cursor>=5 shortcut for all model strides."""

    from latency_meta_mdp.belief.action_conditioned_jepa.config import (
        load_jepa_temporal_sampling,
    )

    sampling = load_jepa_temporal_sampling(
        CONFIG_ROOT / "stride5_100ms_history_200ms.yaml"
    )

    assert not sampling.is_history_ready(9)
    assert sampling.is_history_ready(10)
    assert sampling.is_history_ready(30)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("model_stride_ticks", True, "positive integer"),
        ("history_observation_count", 1, "at least two"),
        ("native_rollout_steps", 3, "400 ms horizon"),
        ("training_rollout_steps", 3, "exactly two"),
        ("latency_quantizer", "drop_missing_bins", "quantizer"),
    ),
)
def test_temporal_sampling_rejects_semantic_drift(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    """Catches malformed configs that change time or latency semantics silently."""

    from latency_meta_mdp.belief.action_conditioned_jepa.config import (
        load_jepa_temporal_sampling,
    )

    payload = {
        "schema_version": 1,
        "config_id": "dense_20ms_history_100ms",
        "model_stride_ticks": 1,
        "history_observation_count": 6,
        "native_rollout_steps": 20,
        "training_rollout_steps": 2,
        "latency_quantizer": "nearest_native_anchor_upper_tie",
    }
    payload[field] = value
    path = tmp_path / "temporal.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=message):
        load_jepa_temporal_sampling(path)


def test_model_loader_accepts_an_explicit_temporal_candidate() -> None:
    """Catches loading candidate metadata without binding it to the resolved model config."""

    from latency_meta_mdp.belief.action_conditioned_jepa.config import (
        load_action_conditioned_jepa_config,
    )

    config = load_action_conditioned_jepa_config(
        model_path=CONFIG_ROOT / "model.yaml",
        level_path=CONFIG_ROOT / "l3.yaml",
        temporal_sampling_path=CONFIG_ROOT / "stride4_80ms_history_160ms.yaml",
    )

    assert config.temporal_sampling.config_id == "stride4_80ms_history_160ms"
    assert config.history_ticks == 3
    assert config.history_span_ticks == 8
    assert config.model_stride_ticks == 4
    assert config.native_rollout_steps == 5
    assert config.macro_action_dim == 28


def test_macro_latency_quantizer_conserves_every_d20_bin_with_upper_ties() -> None:
    """Catches dropping or independently renormalizing latency bins on a coarse grid."""

    from latency_meta_mdp.belief.action_conditioned_jepa.config import (
        load_jepa_temporal_sampling,
    )
    from latency_meta_mdp.belief.action_conditioned_jepa.latent_return import (
        quantize_d20_probabilities,
    )

    sampling = load_jepa_temporal_sampling(
        CONFIG_ROOT / "stride2_40ms_history_120ms.yaml"
    )
    probabilities = torch.arange(1, 21, dtype=torch.float32)
    probabilities = (probabilities / probabilities.sum()).unsqueeze(0)

    anchors, macro = quantize_d20_probabilities(
        probabilities=probabilities,
        sampling=sampling,
    )

    assert torch.equal(anchors, torch.arange(2, 21, 2, dtype=torch.int64))
    expected = probabilities.reshape(1, 10, 2).sum(dim=2)
    torch.testing.assert_close(macro, expected)
    torch.testing.assert_close(macro.sum(dim=1), torch.ones(1))


def test_dense_latency_quantizer_is_exact_identity() -> None:
    """Catches changing D20 probabilities in the full-rate reference."""

    from latency_meta_mdp.belief.action_conditioned_jepa.config import (
        load_jepa_temporal_sampling,
    )
    from latency_meta_mdp.belief.action_conditioned_jepa.latent_return import (
        quantize_d20_probabilities,
    )

    sampling = load_jepa_temporal_sampling(CONFIG_ROOT / "dense_20ms_history_100ms.yaml")
    probabilities = torch.full((2, 20), 0.05, dtype=torch.float32)

    anchors, macro = quantize_d20_probabilities(
        probabilities=probabilities,
        sampling=sampling,
    )

    assert torch.equal(anchors, torch.arange(1, 21, dtype=torch.int64))
    assert macro is probabilities


def test_macro_assembler_attaches_quantized_weights_without_copying_rollout() -> None:
    """Catches a macro Belief that keeps D20 weights or recomputes fixed-delay futures."""

    from latency_meta_mdp.belief.action_conditioned_jepa.config import (
        load_jepa_temporal_sampling,
    )
    from latency_meta_mdp.belief.action_conditioned_jepa.contracts import FutureLatentRollout
    from latency_meta_mdp.belief.action_conditioned_jepa.latent_return import (
        assemble_return_latent_belief,
        quantize_d20_probabilities,
    )

    sampling = load_jepa_temporal_sampling(
        CONFIG_ROOT / "stride4_80ms_history_160ms.yaml"
    )
    anchors = torch.tensor([4, 8, 12, 16, 20], dtype=torch.int64)
    rollout = FutureLatentRollout(
        native_delay_ticks=anchors,
        future_visual_latents=torch.zeros(1, 5, 2, 196, 384, dtype=torch.float16),
        future_proprio=torch.zeros(1, 5, 16, dtype=torch.float32),
    )
    d20 = torch.arange(1, 21, dtype=torch.float32).unsqueeze(0)
    d20 /= d20.sum(dim=1, keepdim=True)
    _, expected = quantize_d20_probabilities(probabilities=d20, sampling=sampling)

    belief = assemble_return_latent_belief(
        rollout,
        d20,
        sampling=sampling,
    )

    assert belief.future_visual_latents is rollout.future_visual_latents
    assert belief.future_proprio is rollout.future_proprio
    assert torch.equal(belief.delay_ticks, anchors)
    torch.testing.assert_close(belief.delay_probabilities, expected)
