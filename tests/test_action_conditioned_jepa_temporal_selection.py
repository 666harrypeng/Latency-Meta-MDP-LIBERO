from __future__ import annotations

from pathlib import Path

import pytest

from latency_meta_mdp.belief.action_conditioned_jepa.temporal_view import SharedJepaSampleIndex


def _indices() -> tuple[SharedJepaSampleIndex, ...]:
    return tuple(
        SharedJepaSampleIndex(
            level=3,
            split="train",
            episode_id=f"episode-{master}-{realization}",
            source_tick=10 + realization,
            boundary_disposition=(
                "recorded_complete" if realization == 0 else "certified_absorbing_extension"
            ),
        )
        for master in range(8)
        for realization in range(2)
    )


def _episode_masters() -> dict[str, int]:
    return {
        f"episode-{master}-{realization}": master
        for master in range(8)
        for realization in range(2)
    }


def test_grouped_four_folds_are_deterministic_complete_and_disjoint() -> None:
    """Catches fold assignment that leaks one master across fit and development."""

    from latency_meta_mdp.belief.action_conditioned_jepa.temporal_selection import (
        build_grouped_master_folds,
    )

    first = build_grouped_master_folds(
        master_indices=tuple(range(8)),
        fold_count=4,
        fold_seed=20260904,
    )
    second = build_grouped_master_folds(
        master_indices=tuple(range(8)),
        fold_count=4,
        fold_seed=20260904,
    )

    assert first == second
    assert tuple(fold.fold_index for fold in first) == (0, 1, 2, 3)
    assert all(len(fold.development_master_indices) == 2 for fold in first)
    assert all(len(fold.fit_master_indices) == 6 for fold in first)
    assert all(
        set(fold.fit_master_indices).isdisjoint(fold.development_master_indices)
        for fold in first
    )
    assert sorted(
        master for fold in first for master in fold.development_master_indices
    ) == list(range(8))


def test_temporal_selection_artifact_round_trips_one_shared_index(tmp_path: Path) -> None:
    """Catches writing one duplicated launch-index inventory per temporal candidate."""

    from latency_meta_mdp.belief.action_conditioned_jepa.temporal_selection import (
        load_temporal_selection_artifact,
        write_temporal_selection_artifact,
    )

    output = tmp_path / "selection"
    path = write_temporal_selection_artifact(
        output_dir=output,
        selection_id="l3-trainpool-temporal-selection",
        indices=_indices(),
        episode_master_indices=_episode_masters(),
        train_pool_master_indices=tuple(range(8)),
        candidate_config_ids=(
            "dense_20ms_history_100ms",
            "stride2_40ms_history_120ms",
            "stride4_80ms_history_160ms",
            "stride5_100ms_history_200ms",
        ),
        fold_count=4,
        fold_seed=20260904,
        source_manifest_sha256="a" * 64,
        cache_manifest_sha256="b" * 64,
        split_manifest_sha256="c" * 64,
    )
    loaded = load_temporal_selection_artifact(output)

    assert path == output / "manifest.json"
    assert {value.name for value in output.iterdir()} == {"indices.jsonl", "manifest.json"}
    assert loaded.indices == _indices()
    assert loaded.manifest["shared_context_count"] == 16
    assert loaded.manifest["recorded_complete_count"] == 8
    assert loaded.manifest["certified_absorbing_extension_count"] == 8
    assert len(loaded.folds) == 4
    assert all(len(fold.development_episode_ids) == 4 for fold in loaded.folds)
    assert all(len(fold.fit_episode_ids) == 12 for fold in loaded.folds)
    with pytest.raises(FileExistsError):
        write_temporal_selection_artifact(
            output_dir=output,
            selection_id="l3-trainpool-temporal-selection",
            indices=_indices(),
            episode_master_indices=_episode_masters(),
            train_pool_master_indices=tuple(range(8)),
            candidate_config_ids=(
                "dense_20ms_history_100ms",
                "stride2_40ms_history_120ms",
                "stride4_80ms_history_160ms",
                "stride5_100ms_history_200ms",
            ),
            fold_count=4,
            fold_seed=20260904,
            source_manifest_sha256="a" * 64,
            cache_manifest_sha256="b" * 64,
            split_manifest_sha256="c" * 64,
        )


def test_temporal_selection_loader_rejects_tampered_indices(tmp_path: Path) -> None:
    """Catches trusting a manifest after its shared launch index was modified."""

    from latency_meta_mdp.belief.action_conditioned_jepa.temporal_selection import (
        load_temporal_selection_artifact,
        write_temporal_selection_artifact,
    )

    output = tmp_path / "selection"
    write_temporal_selection_artifact(
        output_dir=output,
        selection_id="l3-trainpool-temporal-selection",
        indices=_indices(),
        episode_master_indices=_episode_masters(),
        train_pool_master_indices=tuple(range(8)),
        candidate_config_ids=("dense_20ms_history_100ms",),
        fold_count=4,
        fold_seed=20260904,
        source_manifest_sha256="a" * 64,
        cache_manifest_sha256="b" * 64,
        split_manifest_sha256="c" * 64,
    )
    with (output / "indices.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{}\n")

    with pytest.raises(ValueError, match="verification"):
        load_temporal_selection_artifact(output)
