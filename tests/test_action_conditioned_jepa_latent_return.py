from __future__ import annotations

from pathlib import Path

import pytest
import torch
from test_action_conditioned_jepa_data import _normalization, _record

from latency_meta_mdp.belief.action_conditioned_jepa.contracts import FutureLatentRollout


def _sampling():
    from latency_meta_mdp.belief.action_conditioned_jepa.config import (
        load_jepa_temporal_sampling,
    )

    return load_jepa_temporal_sampling(
        Path("configs/belief/action_conditioned_jepa/dense_20ms_history_100ms.yaml")
    )


def _rollout(*, batch_size: int = 2) -> FutureLatentRollout:
    values = torch.arange(batch_size * 20, dtype=torch.float32).reshape(batch_size, 20)
    visual = values[:, :, None, None, None].expand(batch_size, 20, 2, 196, 384)
    proprio = values[:, :, None].expand(batch_size, 20, 16)
    return FutureLatentRollout(
        native_delay_ticks=torch.arange(1, 21, dtype=torch.int64),
        future_visual_latents=visual.to(torch.float16).contiguous(),
        future_proprio=proprio.to(torch.float32).contiguous(),
    )


def _probabilities(*, batch_size: int = 2) -> torch.Tensor:
    weights = torch.arange(1, 21, dtype=torch.float32)
    weights /= weights.sum()
    return weights.unsqueeze(0).repeat(batch_size, 1)


def test_upper_tie_assignment_has_one_public_authoritative_mapping() -> None:
    """Catches inconsistent nearest-anchor ties between PMF assembly and evaluation."""

    from latency_meta_mdp.belief.action_conditioned_jepa.latent_return import (
        upper_tie_nearest_anchor_indices,
    )

    assignment = upper_tie_nearest_anchor_indices(
        dense_delay_ticks=torch.arange(1, 21, dtype=torch.int64),
        native_delay_ticks=torch.tensor([4, 8, 12, 16, 20], dtype=torch.int64),
    )

    assert assignment.tolist() == [
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        4,
        4,
        4,
    ]


def test_changing_only_pmf_never_changes_fixed_delay_futures() -> None:
    from latency_meta_mdp.belief.action_conditioned_jepa.latent_return import (
        assemble_return_latent_belief,
    )

    rollout = _rollout()
    beta = _probabilities()
    uniform = torch.full((2, 20), 0.05, dtype=torch.float32)
    first = assemble_return_latent_belief(rollout, beta, sampling=_sampling())
    second = assemble_return_latent_belief(rollout, uniform, sampling=_sampling())

    assert first.future_visual_latents is rollout.future_visual_latents
    assert first.future_proprio is rollout.future_proprio
    assert second.future_visual_latents is rollout.future_visual_latents
    assert second.future_proprio is rollout.future_proprio
    assert torch.equal(first.future_visual_latents, second.future_visual_latents)
    assert torch.equal(first.future_proprio, second.future_proprio)
    assert not torch.equal(first.delay_probabilities, second.delay_probabilities)


def test_assembler_rejects_instead_of_renormalizing_invalid_pmf() -> None:
    from latency_meta_mdp.belief.action_conditioned_jepa.latent_return import (
        assemble_return_latent_belief,
    )

    probabilities = _probabilities(batch_size=1) * 0.9
    with pytest.raises(ValueError, match="probabilities"):
        assemble_return_latent_belief(
            _rollout(batch_size=1),
            probabilities,
            sampling=_sampling(),
        )


def test_weighted_summaries_match_direct_aggregation() -> None:
    from latency_meta_mdp.belief.action_conditioned_jepa.latent_return import (
        assemble_return_latent_belief,
        weighted_future_proprio,
        weighted_future_visual_latents,
    )

    belief = assemble_return_latent_belief(
        _rollout(),
        _probabilities(),
        sampling=_sampling(),
    )
    expected_proprio = torch.sum(
        belief.delay_probabilities[..., None] * belief.future_proprio,
        dim=1,
    )
    expected_visual = torch.sum(
        belief.delay_probabilities[..., None, None, None] * belief.future_visual_latents,
        dim=1,
    )

    torch.testing.assert_close(weighted_future_proprio(belief), expected_proprio)
    torch.testing.assert_close(weighted_future_visual_latents(belief), expected_visual)


def test_return_belief_serialization_is_exact_and_no_overwrite(tmp_path: Path) -> None:
    from latency_meta_mdp.belief.action_conditioned_jepa.latent_return import (
        assemble_return_latent_belief,
        load_return_latent_belief,
        write_return_latent_belief,
    )

    belief = assemble_return_latent_belief(
        _rollout(batch_size=1),
        _probabilities(batch_size=1),
        sampling=_sampling(),
    )
    output = tmp_path / "belief"
    manifest = write_return_latent_belief(
        output,
        belief,
        latency_law_sha256="a" * 64,
    )
    loaded = load_return_latent_belief(output)

    assert manifest == output / "manifest.json"
    assert loaded.latency_law_sha256 == "a" * 64
    assert torch.equal(loaded.belief.delay_ticks, belief.delay_ticks)
    assert torch.equal(loaded.belief.delay_probabilities, belief.delay_probabilities)
    assert torch.equal(loaded.belief.future_visual_latents, belief.future_visual_latents)
    assert torch.equal(loaded.belief.future_proprio, belief.future_proprio)
    with pytest.raises(FileExistsError):
        write_return_latent_belief(
            output,
            belief,
            latency_law_sha256="a" * 64,
        )


def test_runtime_history_is_unavailable_before_real_k6(tmp_path: Path) -> None:
    from latency_meta_mdp.belief.action_conditioned_jepa.runtime import JepaRuntimeHistory

    record = _record(tmp_path, terminal_tick=10)
    runtime = JepaRuntimeHistory(_normalization(record), temporal_sampling=_sampling())
    for tick in range(5):
        runtime.append_boundary(
            vision_features=torch.from_numpy(record.cache.features[tick].copy()),
            proprio=torch.from_numpy(record.proprio_physical[tick].copy()),
            executed_control_from_previous=(
                None if tick == 0 else torch.from_numpy(record.controls[tick - 1].copy())
            ),
        )
        assert not runtime.ready
    with pytest.raises(RuntimeError, match="complete real history"):
        runtime.build_launch_context(torch.zeros(20, 7, dtype=torch.float32))


def test_streamed_runtime_context_matches_every_offline_ready_context(tmp_path: Path) -> None:
    from latency_meta_mdp.belief.action_conditioned_jepa.data_adapter import (
        materialize_jepa_sample,
    )
    from latency_meta_mdp.belief.action_conditioned_jepa.runtime import JepaRuntimeHistory

    record = _record(tmp_path, terminal_tick=12)
    normalization = _normalization(record)
    runtime = JepaRuntimeHistory(normalization, temporal_sampling=_sampling())
    for tick in range(record.terminal_tick):
        runtime.append_boundary(
            vision_features=torch.from_numpy(record.cache.features[tick].copy()),
            proprio=torch.from_numpy(record.proprio_physical[tick].copy()),
            executed_control_from_previous=(
                None if tick == 0 else torch.from_numpy(record.controls[tick - 1].copy())
            ),
        )
        if tick < 5:
            assert not runtime.ready
            continue
        expected_sample = materialize_jepa_sample(
            record,
            source_tick=tick,
            normalization=normalization,
        )
        actual = runtime.build_launch_context(expected_sample.executable_controls)
        expected = expected_sample.launch_context
        assert torch.equal(actual.vision_history, expected.vision_history)
        assert torch.equal(actual.proprio_history, expected.proprio_history)
        assert torch.equal(actual.executed_controls, expected.executed_controls)
        assert torch.equal(actual.executable_controls, expected.executable_controls)


def test_runtime_history_keeps_only_latest_aligned_k6(tmp_path: Path) -> None:
    from latency_meta_mdp.belief.action_conditioned_jepa.runtime import JepaRuntimeHistory

    record = _record(tmp_path, terminal_tick=12)
    runtime = JepaRuntimeHistory(_normalization(record), temporal_sampling=_sampling())
    for tick in range(9):
        runtime.append_boundary(
            vision_features=torch.from_numpy(record.cache.features[tick].copy()),
            proprio=torch.from_numpy(record.proprio_physical[tick].copy()),
            executed_control_from_previous=(
                None if tick == 0 else torch.from_numpy(record.controls[tick - 1].copy())
            ),
        )
    context = runtime.build_launch_context(
        torch.from_numpy(record.controls[9:12].copy()).new_zeros(20, 7)
    )

    assert torch.equal(
        context.vision_history[0],
        torch.from_numpy(record.cache.features[3:9].copy()),
    )
    assert torch.equal(
        context.executed_controls[0, :, 0],
        torch.from_numpy(record.controls[3:8].copy()),
    )


def test_stride5_runtime_samples_spaced_history_and_groups_real_controls(tmp_path: Path) -> None:
    """Catches lowering model frequency by dropping controls or refreshing only every fifth tick."""

    from latency_meta_mdp.belief.action_conditioned_jepa.config import (
        load_jepa_temporal_sampling,
    )
    from latency_meta_mdp.belief.action_conditioned_jepa.runtime import JepaRuntimeHistory

    record = _record(tmp_path, terminal_tick=30)
    sampling = load_jepa_temporal_sampling(
        Path("configs/belief/action_conditioned_jepa/stride5_100ms_history_200ms.yaml")
    )
    runtime = JepaRuntimeHistory(_normalization(record), temporal_sampling=sampling)
    for tick in range(11):
        runtime.append_boundary(
            vision_features=torch.from_numpy(record.cache.features[tick].copy()),
            proprio=torch.from_numpy(record.proprio_physical[tick].copy()),
            executed_control_from_previous=(
                None if tick == 0 else torch.from_numpy(record.controls[tick - 1].copy())
            ),
        )
    context = runtime.build_launch_context(
        torch.from_numpy(record.controls[10:30].copy())
    )

    assert runtime.ready
    assert context.vision_history.shape == (1, 3, 2, 196, 384)
    assert context.executed_controls.shape == (1, 2, 5, 7)
    assert context.executable_controls.shape == (1, 4, 5, 7)
    assert torch.equal(
        context.vision_history[0],
        torch.from_numpy(record.cache.features[[0, 5, 10]].copy()),
    )
    assert torch.equal(
        context.executed_controls[0],
        torch.from_numpy(record.controls[0:10].reshape(2, 5, 7).copy()),
    )


@pytest.mark.integration
def test_formal_episode_offline_online_context_parity() -> None:
    from latency_meta_mdp.belief.action_conditioned_jepa.config import (
        load_action_conditioned_jepa_config,
    )
    from latency_meta_mdp.belief.action_conditioned_jepa.data_adapter import (
        load_jepa_proprio_normalization,
        load_verified_jepa_inputs,
        load_verified_jepa_record,
        materialize_jepa_sample,
    )
    from latency_meta_mdp.belief.action_conditioned_jepa.runtime import JepaRuntimeHistory

    root = Path(__file__).resolve().parents[1]
    source_root = root / "outputs/source_corpus/panda-ball-structured-source-quota-formal-100x4-v1"
    cache_manifest = (
        root
        / "outputs/derived/vision_features/dinov3-vits16-structured-source-100x4-v1/manifest.json"
    )
    split_manifest = (
        root
        / "outputs/derived/source_splits/panda-ball-structured-source-quota-formal-100x4-v1"
        / "train80-validation20-seed20260903-v1.json"
    )
    normalization_path = (
        root
        / "outputs/derived/action_conditioned_jepa/proprio_normalization"
        / "panda-ball-structured-source-quota-formal-100x4-v1"
        / "train80-validation20-seed20260903-v1/L3.json"
    )
    if not all(
        path.exists() for path in (source_root, cache_manifest, split_manifest, normalization_path)
    ):
        pytest.skip("formal source/cache/split/normalization artifacts are absent")
    config = load_action_conditioned_jepa_config(
        model_path=root / "configs/belief/action_conditioned_jepa/model.yaml",
        level_path=root / "configs/belief/action_conditioned_jepa/l3.yaml",
        temporal_sampling_path=(
            root
            / "configs/belief/action_conditioned_jepa/dense_20ms_history_100ms.yaml"
        ),
    )
    inputs = load_verified_jepa_inputs(
        source_root=source_root,
        cache_run_manifest=cache_manifest,
        split_manifest_path=split_manifest,
        config=config,
    )
    episode_id = next(value for value in inputs.split.train_episode_ids if "source-L3-" in value)
    record = load_verified_jepa_record(
        inputs,
        episode_id=episode_id,
        level=3,
        split="train",
    )
    normalization = load_jepa_proprio_normalization(normalization_path)
    runtime = JepaRuntimeHistory(normalization, temporal_sampling=_sampling())

    for tick in range(record.terminal_tick):
        runtime.append_boundary(
            vision_features=torch.from_numpy(record.cache.features[tick].copy()),
            proprio=torch.from_numpy(record.proprio_physical[tick].copy()),
            executed_control_from_previous=(
                None if tick == 0 else torch.from_numpy(record.controls[tick - 1].copy())
            ),
        )
        if tick < 5:
            assert not runtime.ready
            continue
        expected_sample = materialize_jepa_sample(
            record,
            source_tick=tick,
            normalization=normalization,
        )
        actual = runtime.build_launch_context(expected_sample.executable_controls)
        expected = expected_sample.launch_context
        assert torch.equal(actual.vision_history, expected.vision_history)
        assert torch.equal(actual.proprio_history, expected.proprio_history)
        assert torch.equal(actual.executed_controls, expected.executed_controls)
        assert torch.equal(actual.executable_controls, expected.executable_controls)
