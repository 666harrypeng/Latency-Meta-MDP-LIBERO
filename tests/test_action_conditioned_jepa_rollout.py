from __future__ import annotations

import copy
from pathlib import Path
from types import MethodType

import numpy as np
import pytest
import torch

from latency_meta_mdp.belief.action_conditioned_jepa.config import (
    load_action_conditioned_jepa_config,
)
from latency_meta_mdp.belief.action_conditioned_jepa.contracts import LaunchContextBatch
from latency_meta_mdp.belief.action_conditioned_jepa.data_adapter import (
    JepaProprioNormalization,
)
from latency_meta_mdp.belief.action_conditioned_jepa.upstream_adapter import (
    load_upstream_primitives,
)


def _config():
    return load_action_conditioned_jepa_config(
        model_path=Path("configs/belief/action_conditioned_jepa/model.yaml"),
        level_path=Path("configs/belief/action_conditioned_jepa/l3.yaml"),
        temporal_sampling_path=Path(
            "configs/belief/action_conditioned_jepa/dense_20ms_history_100ms.yaml"
        ),
    )


def _normalization() -> JepaProprioNormalization:
    return JepaProprioNormalization(
        level=3,
        mean=np.zeros(16, dtype=np.float32),
        scale=np.ones(16, dtype=np.float32),
        constant_dimension_mask=np.zeros(16, dtype=np.bool_),
        episode_ids=("train-episode",),
        boundary_count=10,
        source_manifest_sha256="a" * 64,
        split_manifest_sha256="b" * 64,
        sample_index_sha256="c" * 64,
    )


def _context(*, device: str = "cpu") -> LaunchContextBatch:
    controls = (
        torch.linspace(-0.5, 0.5, 20, device=device)
        .view(1, 20, 1, 1)
        .repeat(1, 1, 1, 7)
    )
    return LaunchContextBatch(
        vision_history=torch.zeros(1, 6, 2, 196, 384, dtype=torch.float16, device=device),
        proprio_history=torch.zeros(1, 6, 16, dtype=torch.float32, device=device),
        executed_controls=torch.zeros(1, 5, 1, 7, dtype=torch.float32, device=device),
        executable_controls=controls.to(torch.float32),
    )


def _primitives():
    return load_upstream_primitives(
        reference=_config().upstream_reference,
        project_root=Path.cwd(),
    )


def test_dual_view_coordinates_keep_all_patches_and_one_proprio_token() -> None:
    from latency_meta_mdp.belief.action_conditioned_jepa.rollout import (
        build_spatiotemporal_coordinates,
    )

    coordinates = build_spatiotemporal_coordinates(
        history_ticks=6,
        patch_grid=(14, 14),
        view_count=2,
        include_proprio=True,
    )

    assert coordinates.shape == (6, 393, 3)
    torch.testing.assert_close(coordinates[0, :196, 1:], coordinates[0, 196:392, 1:])
    assert torch.equal(coordinates[:, -1, 1:], torch.zeros(6, 2))
    assert torch.equal(coordinates[:, 0, 0], torch.arange(6, dtype=torch.float32))


def test_block_causal_mask_allows_same_and_past_times_only() -> None:
    from latency_meta_mdp.belief.action_conditioned_jepa.rollout import (
        build_temporal_block_causal_mask,
    )

    mask = build_temporal_block_causal_mask(
        history_ticks=3,
        tokens_per_time=2,
        temporal_window=3,
    )

    assert mask.shape == (6, 6)
    assert torch.all(mask[0:2, 0:2])
    assert not torch.any(mask[0:2, 2:])
    assert torch.all(mask[4:6, :])


def test_dual_view_block_matches_upstream_for_single_regular_grid() -> None:
    from latency_meta_mdp.belief.action_conditioned_jepa.rollout import (
        build_spatiotemporal_coordinates,
        build_temporal_block_causal_mask,
    )
    from latency_meta_mdp.belief.action_conditioned_jepa.upstream_adapter import (
        adapt_upstream_block_for_coordinates,
        build_dual_view_upstream_types,
    )

    primitives = _primitives()
    upstream = primitives.fw_adaln_block_type(
        dim=72,
        num_heads=3,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=lambda dim: torch.nn.LayerNorm(dim, eps=1e-6),
        use_rope=True,
        grid_size=2,
    ).eval()
    adapted = copy.deepcopy(upstream)
    adapt_upstream_block_for_coordinates(
        adapted,
        types=build_dual_view_upstream_types(primitives),
    )
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(2, 12, 72, generator=generator)
    actions = torch.randn(2, 3, 72, generator=generator)
    mask = build_temporal_block_causal_mask(
        history_ticks=3,
        tokens_per_time=4,
        temporal_window=3,
    )
    coordinates = build_spatiotemporal_coordinates(
        history_ticks=3,
        patch_grid=(2, 2),
        view_count=1,
        include_proprio=False,
    )

    expected = upstream(
        x,
        actions,
        attn_mask=mask,
        T=3,
        H_patches=2,
        W_patches=2,
        cond_tokens=0,
    )
    actual = adapted(
        x,
        actions,
        coordinates=coordinates,
        attn_mask=mask,
        tokens_per_time=4,
    )

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)

    changed = x.clone()
    changed[:, -4:].add_(3.0)
    changed_actual = adapted(
        changed,
        actions,
        coordinates=coordinates,
        attn_mask=mask,
        tokens_per_time=4,
    )
    torch.testing.assert_close(actual[:, :4], changed_actual[:, :4], atol=0.0, rtol=0.0)


def test_full_model_token_topology_has_no_additive_time_or_space() -> None:
    from latency_meta_mdp.belief.action_conditioned_jepa.rollout import (
        ActionConditionedJepaPredictor,
    )

    model = ActionConditionedJepaPredictor(
        config=_config(),
        proprio_normalization=_normalization(),
        project_root=Path.cwd(),
        primitives=_primitives(),
    )
    tokens, coordinates = model.build_context_tokens(_context())

    assert tokens.shape == (1, 6, 393, 384)
    assert coordinates.shape == (6, 393, 3)
    assert model.view_embedding.weight.shape == (2, 384)
    assert torch.count_nonzero(model.view_embedding.weight[0]) == 0
    assert not torch.equal(model.view_embedding.weight[0], model.view_embedding.weight[1])
    assert model.additive_time_embedding is None
    assert model.additive_spatial_embedding is None
    assert model.rope_coordinate_names == ("time", "y", "x")
    assert model.parameter_count == 16_282_000


def test_ar20_rollout_cannot_read_unexecuted_control_suffix(monkeypatch) -> None:
    from latency_meta_mdp.belief.action_conditioned_jepa.rollout import (
        ActionConditionedJepaPredictor,
    )

    model = ActionConditionedJepaPredictor(
        config=_config(),
        proprio_normalization=_normalization(),
        project_root=Path.cwd(),
        primitives=_primitives(),
    )

    def scripted_predict_next(
        self,
        *,
        vision_history,
        proprio_history,
        executed_controls,
        outgoing_control,
    ):
        effect = outgoing_control[:, 0, :1]
        visual = vision_history[:, -1].to(torch.float32) + effect[:, None, None, :]
        proprio = proprio_history[:, -1] + effect
        return visual, proprio

    monkeypatch.setattr(
        model,
        "predict_next",
        MethodType(scripted_predict_next, model),
    )
    context = _context()
    changed_controls = context.executable_controls.clone()
    changed_controls[:, 7:].mul_(-1.0)
    changed = LaunchContextBatch(
        vision_history=context.vision_history,
        proprio_history=context.proprio_history,
        executed_controls=context.executed_controls,
        executable_controls=changed_controls,
    )

    base = model.rollout_native(context)
    other = model.rollout_native(changed)

    torch.testing.assert_close(
        base.future_visual_latents[:, :7],
        other.future_visual_latents[:, :7],
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        base.future_proprio[:, :7],
        other.future_proprio[:, :7],
        atol=0.0,
        rtol=0.0,
    )
    assert not torch.equal(
        base.future_proprio[:, 7:],
        other.future_proprio[:, 7:],
    )


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for exact K6 model smoke")
def test_exact_predictor_cuda_forward_gradient_and_ar20() -> None:
    from latency_meta_mdp.belief.action_conditioned_jepa.rollout import (
        ActionConditionedJepaPredictor,
    )

    model = ActionConditionedJepaPredictor(
        config=_config(),
        proprio_normalization=_normalization(),
        project_root=Path.cwd(),
        primitives=_primitives(),
    ).cuda()
    context = _context(device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        visual, proprio = model.predict_next(
            vision_history=context.vision_history,
            proprio_history=context.proprio_history,
            executed_controls=context.executed_controls,
            outgoing_control=context.executable_controls[:, 0],
        )
        loss = visual.float().square().mean() + proprio.float().square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        any(parameter.grad is not None for parameter in block.parameters())
        for block in model.backbone.predictor_blocks
    )
    assert not context.vision_history.requires_grad

    with torch.autocast("cuda", dtype=torch.bfloat16):
        endpoint = model.rollout_endpoint_for_loss(context, horizon=2)
    assert all(torch.isfinite(value).all() for value in endpoint)
    rollout = model.rollout_native(context)
    assert rollout.future_visual_latents.dtype == torch.float16
    assert rollout.future_proprio.dtype == torch.float32
    rollout.validate_finite()


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for full predictor parity")
def test_complete_single_view_predictor_matches_upstream() -> None:
    from functools import partial

    from latency_meta_mdp.belief.action_conditioned_jepa.rollout import (
        ActionConditionedJepaPredictor,
    )

    config = _config()
    primitives = _primitives()
    project = (
        ActionConditionedJepaPredictor(
            config=config,
            proprio_normalization=_normalization(),
            project_root=Path.cwd(),
            primitives=primitives,
        )
        .cuda()
        .eval()
    )
    upstream = (
        primitives.vision_transformer_adaln_type(
            img_size=(224, 224),
            patch_size=16,
            num_frames=6,
            tubelet_size=1,
            embed_dim=384,
            predictor_embed_dim=384,
            depth=6,
            num_heads=16,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            init_std=0.02,
            use_silu=False,
            is_causal=False,
            use_activation_checkpointing=False,
            local_window=(6, -1, -1),
            use_rope=True,
            action_dim=7,
            proprio_dim=16,
            use_proprio=False,
            init_scale_factor_adaln=10,
            proprio_encoding="token",
            proprio_tokens=0,
            proprio_encoder_inpred=False,
            action_encoder_inpred=True,
        )
        .cuda()
        .eval()
    )
    project_state = project.backbone.state_dict()
    upstream_state = upstream.state_dict()
    for name, value in tuple(upstream_state.items()):
        if name in project_state and project_state[name].shape == value.shape:
            upstream_state[name] = project_state[name]
    upstream.load_state_dict(upstream_state, strict=True)
    generator = torch.Generator(device="cuda").manual_seed(17)
    vision = torch.randn(1, 6, 1, 14, 14, 384, device="cuda", generator=generator)
    controls = torch.randn(1, 6, 7, device="cuda", generator=generator)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = upstream(vision, controls)[0]
        actual = project.forward_single_view_compat(vision, controls)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_stride4_predictor_uses_one_shared_model_path_and_macro_action_width() -> None:
    """Catches a separate coarse predictor or an action encoder that sees only one micro-control."""

    from latency_meta_mdp.belief.action_conditioned_jepa.rollout import (
        ActionConditionedJepaPredictor,
    )

    config = load_action_conditioned_jepa_config(
        model_path=Path("configs/belief/action_conditioned_jepa/model.yaml"),
        level_path=Path("configs/belief/action_conditioned_jepa/l3.yaml"),
        temporal_sampling_path=Path(
            "configs/belief/action_conditioned_jepa/stride4_80ms_history_160ms.yaml"
        ),
    )
    model = ActionConditionedJepaPredictor(
        config=config,
        proprio_normalization=_normalization(),
        project_root=Path.cwd(),
        primitives=_primitives(),
    )

    assert model.config.history_ticks == 3
    assert model.backbone.action_encoder.in_features == 28
    assert model.coordinates.shape == (3, 393, 3)
    assert model.attention_mask.shape == (3 * 393, 3 * 393)


def test_macro_rollout_cannot_read_later_control_blocks(monkeypatch) -> None:
    """Catches a native anchor depending on controls after its physical-time endpoint."""

    from latency_meta_mdp.belief.action_conditioned_jepa.config import (
        load_action_conditioned_jepa_config,
    )
    from latency_meta_mdp.belief.action_conditioned_jepa.contracts import LaunchContextBatch
    from latency_meta_mdp.belief.action_conditioned_jepa.rollout import (
        ActionConditionedJepaPredictor,
    )

    config = load_action_conditioned_jepa_config(
        model_path=Path("configs/belief/action_conditioned_jepa/model.yaml"),
        level_path=Path("configs/belief/action_conditioned_jepa/l3.yaml"),
        temporal_sampling_path=Path(
            "configs/belief/action_conditioned_jepa/stride2_40ms_history_120ms.yaml"
        ),
    )
    model = ActionConditionedJepaPredictor(
        config=config,
        proprio_normalization=_normalization(),
        project_root=Path.cwd(),
        primitives=_primitives(),
    )

    def scripted_predict_next(
        self,
        *,
        vision_history,
        proprio_history,
        executed_controls,
        outgoing_control,
    ):
        effect = outgoing_control.sum(dim=(1, 2), keepdim=False)[:, None]
        visual = vision_history[:, -1].to(torch.float32) + effect[:, None, None, :]
        proprio = proprio_history[:, -1] + effect
        return visual, proprio

    monkeypatch.setattr(model, "predict_next", MethodType(scripted_predict_next, model))
    future = torch.arange(20, dtype=torch.float32).reshape(1, 10, 2, 1).repeat(1, 1, 1, 7)
    context = LaunchContextBatch(
        vision_history=torch.zeros(1, 4, 2, 196, 384, dtype=torch.float16),
        proprio_history=torch.zeros(1, 4, 16, dtype=torch.float32),
        executed_controls=torch.zeros(1, 3, 2, 7, dtype=torch.float32),
        executable_controls=future,
    )
    changed_future = future.clone()
    changed_future[:, 4:].mul_(-1)
    changed = LaunchContextBatch(
        vision_history=context.vision_history,
        proprio_history=context.proprio_history,
        executed_controls=context.executed_controls,
        executable_controls=changed_future,
    )

    base = model.rollout_native(context)
    other = model.rollout_native(changed)

    assert torch.equal(base.native_delay_ticks, torch.arange(2, 21, 2, dtype=torch.int64))
    torch.testing.assert_close(base.future_proprio[:, :4], other.future_proprio[:, :4])
    assert not torch.equal(base.future_proprio[:, 4:], other.future_proprio[:, 4:])
