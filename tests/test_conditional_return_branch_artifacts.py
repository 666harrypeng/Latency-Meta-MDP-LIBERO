from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from latency_meta_mdp.artifacts import sha256_file
from latency_meta_mdp.belief.conditional_return_flow.branch_artifacts import (
    ControlBranchCorpus,
    load_verified_control_branch_corpus,
    write_control_branch_corpus,
)
from latency_meta_mdp.belief.conditional_return_flow.branch_contracts import (
    BranchRollout,
    ControlContinuationSpec,
    ExecutablePrefix,
    SourceContextIdentity,
    SourceSelectionExclusion,
)


def _source(index: int) -> SourceContextIdentity:
    phase = ("pregrasp", "approach", "close", "lift")[index]
    source_tick = 40 + index
    return SourceContextIdentity(
        source_context_id=f"L1-seed001000-tick{source_tick:06d}",
        episode_id="l1-seed-001000-attempt-000",
        level=1,
        scene_seed=1000,
        split="train",
        source_tick=source_tick,
        source_phase=phase,
        history_start_tick=source_tick - 5,
        source_episode_manifest_sha256="a" * 64,
        source_arrays_sha256="b" * 64,
        source_metadata_sha256="c" * 64,
        source_observation_sha256=hex(index + 1)[2:] * 64,
        motion_profile_sha256="e" * 64,
    )


def _rollout(source: SourceContextIdentity, kind: str) -> BranchRollout:
    controls = np.zeros((20, 7), dtype=np.float32)
    controls[:, 6] = -1.0
    nominal = kind == "nominal"
    return BranchRollout(
        source=source,
        branch=ControlContinuationSpec(kind=kind, arm_scale=None, prefix_real_ticks=None),
        prefix=ExecutablePrefix(
            controls=controls,
            from_active_buffer_mask=np.ones(20, dtype=bool),
            last_executed_gripper_command=-1.0,
        ),
        target_states=np.zeros((20, 22), dtype=np.float32),
        target_absorbing=np.zeros(20, dtype=bool),
        target_handoff_state=np.full(20, "driven"),
        target_outcome_status=np.full(20, "running"),
        source_replay_max_abs=0.0,
        source_fingerprint_match=True,
        nominal_future_valid=nominal,
        nominal_future_max_abs=0.0,
    )


def _selection_exclusion() -> SourceSelectionExclusion:
    return SourceSelectionExclusion(
        episode_id="l1-seed-001000-attempt-000",
        level=1,
        scene_seed=1000,
        split="train",
        source_phase="lift",
        reason="no_phase_tick_in_source_interval",
        source_interval_minimum=25,
        source_interval_maximum=80,
        phase_tick_count=12,
        interval_phase_tick_count=0,
        running_interval_phase_tick_count=0,
    )


def _corpus() -> ControlBranchCorpus:
    sources = (_source(0), _source(1), _source(2))
    rollouts = (
        _rollout(sources[0], "nominal"),
        _rollout(sources[0], "hold"),
        _rollout(sources[1], "nominal"),
        _rollout(sources[1], "hold"),
        _rollout(sources[1], "hold"),
        _rollout(sources[2], "nominal"),
        _rollout(sources[2], "hold"),
    )
    return ControlBranchCorpus(
        contexts=sources,
        canonical_replay_fingerprints=("1" * 64, "2" * 64, "3" * 64),
        rollouts=rollouts,
        selection_exclusions=(_selection_exclusion(),),
        expected_selection_slot_count=4,
        required_branch_kinds=("nominal", "hold"),
        requested_levels=(1,),
        selection_truncations=(),
    )


def _manifest_fields(*, implementation_dirty: bool = False) -> dict[str, object]:
    input_paths = {
        "source_bulk_manifest": "/source/manifest.json",
        "split_config": "/configs/split.yaml",
        "temporal_config": "/configs/temporal.yaml",
        "branch_config": "/configs/branch.yaml",
        "control_config": "/configs/control.yaml",
    }
    return {
        "implementation_revision": "4" * 40,
        "implementation_source_sha256": "5" * 64,
        "implementation_dirty": implementation_dirty,
        "input_paths": input_paths,
        "input_sha256": {name: "6" * 64 for name in input_paths},
        "collection_wall_time_seconds": 1.0,
        "peak_rss_bytes": 1024,
    }


def _rewrite_branch_arrays(manifest: Path, mutate) -> None:
    arrays_path = manifest.parent / "branches.npz"
    with np.load(arrays_path, allow_pickle=False) as source:
        arrays = {name: np.array(source[name], copy=True) for name in source.files}
    mutate(arrays)
    with arrays_path.open("wb") as handle:
        np.savez(handle, **arrays)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["artifacts"]["branches.npz"] = sha256_file(arrays_path)
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def test_branch_artifact_supports_variable_rows_per_source(tmp_path: Path) -> None:
    manifest = write_control_branch_corpus(
        corpus=_corpus(),
        output_dir=tmp_path / "run",
        manifest_fields=_manifest_fields(),
        bounded=False,
    )

    loaded = load_verified_control_branch_corpus(manifest)

    assert loaded.source_context_index.tolist() == [0, 0, 1, 1, 1, 2, 2]
    assert loaded.target_states.shape == (7, 20, 22)
    assert loaded.scientific_gate_pass is True
    assert loaded.artifact_eligible is True


def test_branch_artifact_writes_strict_v3_schema_and_selection_scope(tmp_path: Path) -> None:
    manifest = write_control_branch_corpus(
        corpus=_corpus(),
        output_dir=tmp_path / "run",
        manifest_fields=_manifest_fields(),
        bounded=False,
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 3
    assert payload["format_id"] == "conditional_return_control_branch_corpus_v3"
    assert payload["allowed_scene_seed_ranges"] is None
    assert payload["selected_episode_count"] == 1
    assert len(payload["selected_episode_identities_sha256"]) == 64


def test_corpus_rejects_conflicting_rows_under_one_episode_id() -> None:
    base = _corpus()
    conflicting = replace(base.contexts[1], scene_seed=1001)

    with pytest.raises(ValueError, match="episode identity"):
        replace(base, contexts=(base.contexts[0], conflicting, base.contexts[2]))


def test_branch_loader_rejects_legacy_v1_with_explicit_error(tmp_path: Path) -> None:
    manifest = write_control_branch_corpus(
        corpus=_corpus(),
        output_dir=tmp_path / "run",
        manifest_fields=_manifest_fields(),
        bounded=False,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["schema_version"] = 1
    payload["format_id"] = "conditional_return_control_branch_corpus_v1"
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="legacy v1"):
        load_verified_control_branch_corpus(manifest)


def test_branch_artifact_never_overwrites(tmp_path: Path) -> None:
    target = tmp_path / "run"
    write_control_branch_corpus(
        corpus=_corpus(),
        output_dir=target,
        manifest_fields=_manifest_fields(),
        bounded=False,
    )

    with pytest.raises(FileExistsError):
        write_control_branch_corpus(
            corpus=_corpus(),
            output_dir=target,
            manifest_fields=_manifest_fields(),
            bounded=False,
        )


def test_branch_artifact_rejects_nested_tamper(tmp_path: Path) -> None:
    manifest = write_control_branch_corpus(
        corpus=_corpus(),
        output_dir=tmp_path / "run",
        manifest_fields=_manifest_fields(),
        bounded=False,
    )
    parameters = manifest.parent / "branch_parameters.jsonl"
    parameters.write_text(parameters.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_verified_control_branch_corpus(manifest)


def test_branch_loader_rejects_missing_provenance_field(tmp_path: Path) -> None:
    manifest = write_control_branch_corpus(
        corpus=_corpus(),
        output_dir=tmp_path / "run",
        manifest_fields=_manifest_fields(),
        bounded=False,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    del payload["implementation_revision"]
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest fields"):
        load_verified_control_branch_corpus(manifest)


def test_branch_loader_rejects_incomplete_input_provenance(tmp_path: Path) -> None:
    manifest = write_control_branch_corpus(
        corpus=_corpus(),
        output_dir=tmp_path / "run",
        manifest_fields=_manifest_fields(),
        bounded=False,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    del payload["input_paths"]["control_config"]
    del payload["input_sha256"]["control_config"]
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="provenance values"):
        load_verified_control_branch_corpus(manifest)


def test_branch_loader_rejects_action_outside_contract_after_rehash(tmp_path: Path) -> None:
    manifest = write_control_branch_corpus(
        corpus=_corpus(),
        output_dir=tmp_path / "run",
        manifest_fields=_manifest_fields(),
        bounded=False,
    )

    def mutate(arrays: dict[str, np.ndarray]) -> None:
        arrays["executable_controls"][0, 0, 0] = 1000.0

    _rewrite_branch_arrays(manifest, mutate)

    with pytest.raises(ValueError, match="ActionContract"):
        load_verified_control_branch_corpus(manifest)


def test_missing_nominal_parity_is_a_loadable_scientific_failure(tmp_path: Path) -> None:
    base = _corpus()
    rollouts = tuple(
        replace(row, nominal_future_valid=False, nominal_future_max_abs=0.0)
        if row.branch.kind == "nominal"
        else row
        for row in base.rollouts
    )
    corpus = replace(base, rollouts=rollouts)
    manifest = write_control_branch_corpus(
        corpus=corpus,
        output_dir=tmp_path / "run",
        manifest_fields=_manifest_fields(),
        bounded=False,
    )

    loaded = load_verified_control_branch_corpus(manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert loaded.scientific_gate_pass is False
    assert payload["scientific_blockers"] == ["missing_nominal_future"]


def test_incomplete_inventory_is_a_loadable_scientific_failure(tmp_path: Path) -> None:
    base = _corpus()
    rollouts = tuple(
        row
        for row in base.rollouts
        if not (row.source == base.contexts[2] and row.branch.kind == "hold")
    )
    corpus = replace(base, rollouts=rollouts)
    manifest = write_control_branch_corpus(
        corpus=corpus,
        output_dir=tmp_path / "run",
        manifest_fields=_manifest_fields(),
        bounded=False,
    )

    loaded = load_verified_control_branch_corpus(manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert loaded.scientific_gate_pass is False
    assert payload["scientific_blockers"] == ["incomplete_branch_inventory"]


def test_scientific_and_artifact_status_are_independent(tmp_path: Path) -> None:
    manifest = write_control_branch_corpus(
        corpus=_corpus(),
        output_dir=tmp_path / "run",
        manifest_fields=_manifest_fields(implementation_dirty=True),
        bounded=True,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload["scientific_gate_pass"] is True
    assert payload["artifact_eligible"] is False
    assert set(payload["artifact_blockers"]) == {"bounded_review", "implementation_dirty"}


def test_branch_artifact_accounts_for_legitimate_source_selection_exclusion(
    tmp_path: Path,
) -> None:
    corpus = _corpus()
    exclusion = _selection_exclusion()

    manifest = write_control_branch_corpus(
        corpus=corpus,
        output_dir=tmp_path / "run",
        manifest_fields=_manifest_fields(),
        bounded=False,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    loaded = load_verified_control_branch_corpus(manifest)

    assert payload["selection_slot_count"] == 4
    assert payload["selection_exclusion_count"] == 1
    assert loaded.selection_exclusions == (exclusion,)


def test_branch_loader_rejects_manifest_counts_that_disagree_with_children(
    tmp_path: Path,
) -> None:
    manifest = write_control_branch_corpus(
        corpus=_corpus(),
        output_dir=tmp_path / "run",
        manifest_fields=_manifest_fields(),
        bounded=False,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["context_count"] = 1
    payload["branch_count"] = 1
    payload["selection_slot_count"] = 1
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest counts do not match"):
        load_verified_control_branch_corpus(manifest)


def test_branch_loader_rejects_requested_level_that_disagrees_with_contexts(
    tmp_path: Path,
) -> None:
    manifest = write_control_branch_corpus(
        corpus=_corpus(),
        output_dir=tmp_path / "run",
        manifest_fields=_manifest_fields(),
        bounded=False,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["requested_levels"] = [3]
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="requested levels do not match"):
        load_verified_control_branch_corpus(manifest)


def test_branch_loader_rejects_eligibility_status_that_disagrees_with_bounded(
    tmp_path: Path,
) -> None:
    manifest = write_control_branch_corpus(
        corpus=_corpus(),
        output_dir=tmp_path / "run",
        manifest_fields=_manifest_fields(),
        bounded=False,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["bounded"] = True
    payload["artifact_eligible"] = True
    payload["artifact_blockers"] = []
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact eligibility status is inconsistent"):
        load_verified_control_branch_corpus(manifest)


def test_branch_loader_rejects_duplicate_selection_exclusion_after_rehash(
    tmp_path: Path,
) -> None:
    corpus = _corpus()
    manifest = write_control_branch_corpus(
        corpus=corpus,
        output_dir=tmp_path / "run",
        manifest_fields=_manifest_fields(),
        bounded=False,
    )
    exclusions_path = manifest.parent / "selection_exclusions.json"
    exclusions = json.loads(exclusions_path.read_text(encoding="utf-8"))
    exclusions.append(dict(exclusions[0]))
    exclusions_path.write_text(json.dumps(exclusions, sort_keys=True) + "\n", encoding="utf-8")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["artifacts"]["selection_exclusions.json"] = sha256_file(exclusions_path)
    payload["selection_exclusion_count"] = 2
    payload["selection_slot_count"] = 5
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="selection slots must be unique"):
        load_verified_control_branch_corpus(manifest)
