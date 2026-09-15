from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from test_action_conditioned_jepa_data import _record

from latency_meta_mdp.io.paths import repository_root


def _signal_episode(tmp_path: Path):
    from latency_meta_mdp.belief.jepa.diagnostics.history_signal import (
        TemporalSignalEpisode,
    )

    record = _record(tmp_path, terminal_tick=30)
    ticks = np.arange(31, dtype=np.float32)
    object_position = np.zeros((31, 3), dtype=np.float32)
    object_position[:, 0] = ticks * 0.001
    object_velocity = np.zeros((31, 3), dtype=np.float32)
    object_velocity[:, 0] = 0.05
    eef_position = np.zeros((31, 3), dtype=np.float32)
    eef_position[:, 1] = ticks * 0.002
    left = ticks >= 25
    right = ticks >= 25
    handoff = tuple("robot" if tick >= 27 else "driver" for tick in range(31))
    segments = np.where(ticks >= 20, 1, 0).astype(np.int64)
    return TemporalSignalEpisode(
        record=record,
        object_position=object_position,
        object_linear_velocity=object_velocity,
        eef_position=eef_position,
        left_pad_contact=left,
        right_pad_contact=right,
        handoff_state=handoff,
        motion_segment_index=segments,
    )


def _samplings():
    from latency_meta_mdp.belief.jepa.config import (
        load_jepa_temporal_sampling,
    )

    root = Path("configs/models/jepa")
    return tuple(
        load_jepa_temporal_sampling(root / name)
        for name in (
            "dense_20ms_history_100ms.yaml",
            "stride2_40ms_history_120ms.yaml",
            "stride4_80ms_history_160ms.yaml",
            "stride5_100ms_history_200ms.yaml",
        )
    )


def test_temporal_signal_audit_uses_shared_contexts_and_native_physical_time(
    tmp_path: Path,
) -> None:
    """Catches comparing candidates on different launch contexts or wrong stride endpoints."""

    from latency_meta_mdp.belief.jepa.diagnostics.history_signal import (
        summarize_temporal_signals,
    )

    report = summarize_temporal_signals(
        episodes=(_signal_episode(tmp_path),),
        samplings=_samplings(),
        latent_samples_per_episode=1,
    )

    assert report["shared_context_count"] == 20
    assert report["shared_context_start_tick"] == 10
    for config_id, stride, rollout_steps in (
        ("dense_20ms_history_100ms", 1, 20),
        ("stride2_40ms_history_120ms", 2, 10),
        ("stride4_80ms_history_160ms", 4, 5),
        ("stride5_100ms_history_200ms", 5, 4),
    ):
        candidate = report["candidates"][config_id]
        assert candidate["model_stride_ticks"] == stride
        assert candidate["native_rollout_steps"] == rollout_steps
        assert candidate["shared_context_count"] == 20
        assert candidate["one_step_latent_sample_count"] == 1
        assert candidate["one_step_latent_rms"]["mean"] == float(stride)
        assert np.isclose(
            candidate["one_step_object_displacement_m"]["median"],
            stride * 0.001,
        )
        assert np.isclose(
            candidate["one_step_eef_displacement_m"]["median"],
            stride * 0.002,
        )


def test_temporal_signal_audit_reports_absorption_and_event_aliasing(tmp_path: Path) -> None:
    """Catches accepting a coarse stride without exposing terminal/contact crossings."""

    from latency_meta_mdp.belief.jepa.diagnostics.history_signal import (
        summarize_temporal_signals,
    )

    report = summarize_temporal_signals(
        episodes=(_signal_episode(tmp_path),),
        samplings=_samplings(),
        latent_samples_per_episode=1,
    )
    candidate = report["candidates"]["stride5_100ms_history_200ms"]

    assert candidate["absorbing_context_count"] == 19
    assert np.isclose(candidate["one_step_contact_crossing_fraction"], 0.25)
    assert np.isclose(candidate["one_step_handoff_crossing_fraction"], 0.25)
    assert candidate["native_anchor_ticks"] == [5, 10, 15, 20]


def test_temporal_signal_report_is_provenance_bound_and_no_overwrite(tmp_path: Path) -> None:
    """Catches publishing an analysis report detached from source/cache/split identity."""

    from latency_meta_mdp.belief.jepa.diagnostics.history_signal import (
        summarize_temporal_signals,
        write_temporal_signal_report,
    )

    report = summarize_temporal_signals(
        episodes=(_signal_episode(tmp_path),),
        samplings=_samplings(),
        latent_samples_per_episode=1,
    )
    output = tmp_path / "report"
    report_path = write_temporal_signal_report(
        output_dir=output,
        report=report,
        source_manifest_sha256="a" * 64,
        cache_manifest_sha256="b" * 64,
        split_manifest_sha256="c" * 64,
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_path == output / "report.json"
    assert payload["provenance"] == {
        "cache_manifest_sha256": "b" * 64,
        "source_manifest_sha256": "a" * 64,
        "split_manifest_sha256": "c" * 64,
    }
    with pytest.raises(FileExistsError):
        write_temporal_signal_report(
            output_dir=output,
            report=report,
            source_manifest_sha256="a" * 64,
            cache_manifest_sha256="b" * 64,
            split_manifest_sha256="c" * 64,
        )


@pytest.mark.integration
def test_formal_temporal_signal_episode_loads_from_verified_source_and_cache() -> None:
    """Catches auditing physical rows that are not joined to the verified DINO episode."""

    from latency_meta_mdp.belief.jepa.config import (
        load_action_conditioned_jepa_config,
    )
    from latency_meta_mdp.belief.jepa.corpus import (
        load_verified_jepa_inputs,
    )
    from latency_meta_mdp.belief.jepa.diagnostics.history_signal import (
        load_temporal_signal_episodes,
        summarize_temporal_signals,
    )

    root = repository_root()
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
    if not all(path.exists() for path in (source_root, cache_manifest, split_manifest)):
        pytest.skip("formal source/cache/split artifacts are absent")
    config = load_action_conditioned_jepa_config(
        model_path=root / "configs/models/jepa/model.yaml",
        level_path=root / "configs/models/jepa/l3.yaml",
        temporal_sampling_path=(root / "configs/models/jepa/dense_20ms_history_100ms.yaml"),
    )
    inputs = load_verified_jepa_inputs(
        source_root=source_root,
        cache_run_manifest=cache_manifest,
        split_manifest_path=split_manifest,
        config=config,
    )
    episode_id = next(value for value in inputs.split.train_episode_ids if "source-L3-" in value)

    episodes = load_temporal_signal_episodes(
        inputs=inputs,
        episode_ids=(episode_id,),
        level=3,
        split="train",
    )
    report = summarize_temporal_signals(
        episodes=episodes,
        samplings=_samplings(),
        latent_samples_per_episode=1,
    )

    assert len(episodes) == 1
    assert episodes[0].record.episode_id == episode_id
    assert episodes[0].object_position.shape[1] == 3
    assert episodes[0].object_linear_velocity.shape[1] == 3
    assert episodes[0].eef_position.shape[1] == 3
    assert report["episode_count"] == 1
    assert report["shared_context_count"] == episodes[0].record.terminal_tick - 10
