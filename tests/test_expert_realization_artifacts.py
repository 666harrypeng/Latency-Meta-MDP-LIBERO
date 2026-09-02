from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from test_expert_realization_recording import SHA_A, SHA_B, SHA_C, SHA_D, SHA_E, make_episode


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _implementation():
    from latency_meta_mdp.expert_realization.artifacts import ImplementationIdentity

    return ImplementationIdentity("1" * 40, SHA_E, False)


def _execution_implementation():
    from latency_meta_mdp.expert_realization.artifacts import ImplementationIdentity

    return ImplementationIdentity("1" * 40, SHA_E, True)


def _task_payloads():
    task = _json_bytes({"format_id": "structured_expert_task_instance_payload_v1", "level": 1})
    motion = _json_bytes({"format_id": "structured_expert_motion_profile_v1", "kind": "linear"})
    initial = b"NUMPY-INITIAL-STATE-V1"
    return {
        "task_instance.json": task,
        "motion_profile.json": motion,
        "initial_state.npz": initial,
    }


def _task_manifest():
    from latency_meta_mdp.expert_realization.artifacts import TaskInstanceManifest
    from latency_meta_mdp.expert_realization.contracts import TaskInstanceId

    payloads = _task_payloads()
    identity = TaskInstanceId(
        1, 4000, _sha(payloads["motion_profile.json"]), _sha(payloads["initial_state.npz"])
    )
    return TaskInstanceManifest(
        schema_version=1,
        format_id="structured_expert_task_instance_v1",
        task_instance_id=identity,
        task_id="dynamic_grasp_lift",
        instruction="grasp and lift the moving ball",
        decision_source_tick=5,
        shared_prefix_policy="settle_open_hold_v1",
        task_config_sha256=SHA_A,
        motion_config_sha256=SHA_B,
        runtime_config_sha256=SHA_C,
        controller_config_sha256=SHA_D,
        implementation=_implementation(),
        artifacts={name: _sha(payload) for name, payload in payloads.items()},
    )


def _plan_payloads():
    payloads = {
        "strategies/realization-000.json": _json_bytes(
            {"format_id": "structured_expert_strategy_v1", "family": "canonical_direct"}
        ),
        "planner_candidates/realization-000.jsonl": (
            b'{"format_id":"structured_expert_planner_candidate_v1","fingerprint":"fp0"}\n'
        ),
        "selected_references/realization-000.npz": b"NUMPY-SELECTED-REFERENCE-V1",
        "strategies/realization-001.json": _json_bytes(
            {"format_id": "structured_expert_strategy_v1", "family": "early_high_arc"}
        ),
        "planner_candidates/realization-001.jsonl": (
            b'{"format_id":"structured_expert_planner_candidate_v1","failure":"timeout"}\n'
        ),
    }
    for index in range(2, 8):
        payloads[f"strategies/realization-{index:03d}.json"] = _json_bytes(
            {"format_id": "structured_expert_strategy_v1", "family": "planner_failure"}
        )
        payloads[f"planner_candidates/realization-{index:03d}.jsonl"] = (
            b'{"format_id":"structured_expert_planner_candidate_v1","failure":"timeout"}\n'
        )
    return payloads


def _plan_manifest(*, task_manifest_sha: str, task_manifest_path: str):
    from latency_meta_mdp.expert_realization.artifacts import (
        ArtifactRef,
        FrozenPlanRow,
        FrozenPlanSetManifest,
    )
    from latency_meta_mdp.expert_realization.contracts import (
        ExpertRealizationKey,
        build_realization_universe_identity,
    )

    task = _task_manifest().task_instance_id
    payloads = _plan_payloads()
    rows = [
        FrozenPlanRow(
            expert_realization_key=ExpertRealizationKey(task, 0, SHA_C),
            strategy=ArtifactRef(
                "strategies/realization-000.json", _sha(payloads["strategies/realization-000.json"])
            ),
            planner_candidates=ArtifactRef(
                "planner_candidates/realization-000.jsonl",
                _sha(payloads["planner_candidates/realization-000.jsonl"]),
            ),
            selection_status="selected",
            selected_candidate_fingerprint="fp0",
            selected_reference=ArtifactRef(
                "selected_references/realization-000.npz",
                _sha(payloads["selected_references/realization-000.npz"]),
            ),
        ),
        FrozenPlanRow(
            expert_realization_key=ExpertRealizationKey(task, 1, SHA_C),
            strategy=ArtifactRef(
                "strategies/realization-001.json", _sha(payloads["strategies/realization-001.json"])
            ),
            planner_candidates=ArtifactRef(
                "planner_candidates/realization-001.jsonl",
                _sha(payloads["planner_candidates/realization-001.jsonl"]),
            ),
            selection_status="planner_failure",
            selected_candidate_fingerprint=None,
            selected_reference=None,
        ),
    ]
    for index in range(2, 8):
        rows.append(
            FrozenPlanRow(
                expert_realization_key=ExpertRealizationKey(task, index, SHA_C),
                strategy=ArtifactRef(
                    f"strategies/realization-{index:03d}.json",
                    _sha(payloads[f"strategies/realization-{index:03d}.json"]),
                ),
                planner_candidates=ArtifactRef(
                    f"planner_candidates/realization-{index:03d}.jsonl",
                    _sha(payloads[f"planner_candidates/realization-{index:03d}.jsonl"]),
                ),
                selection_status="planner_failure",
                selected_candidate_fingerprint=None,
                selected_reference=None,
            )
        )
    slots = tuple(range(8))
    universe = build_realization_universe_identity(
        request_sha256=SHA_E,
        task_instance_id=task,
        realization_slots=slots,
    )
    return FrozenPlanSetManifest(
        schema_version=1,
        format_id="structured_expert_plan_set_v1",
        task_instance_id=task,
        task_instance_manifest=ArtifactRef(task_manifest_path, task_manifest_sha),
        structured_expert_config_sha256=SHA_C,
        curobo_planner_config_sha256=SHA_D,
        realization_universe_sha256=universe.universe_sha256,
        realization_slots=slots,
        implementation=_implementation(),
        realizations=tuple(rows),
    )


def _publish_ancestors(pilot_root: Path):
    from latency_meta_mdp.expert_realization.artifacts import (
        publish_frozen_plan_set,
        publish_task_instance,
    )

    base = pilot_root / "task_instances/L1/seed-4000"
    task_root = base / "task_instance"
    publish_task_instance(_task_manifest(), task_root, artifacts=_task_payloads())
    task_manifest_path = "task_instances/L1/seed-4000/task_instance/manifest.json"
    task_manifest_sha = _sha((task_root / "manifest.json").read_bytes())
    plan = _plan_manifest(
        task_manifest_sha=task_manifest_sha,
        task_manifest_path=task_manifest_path,
    )
    plan_root = base / "plan_set"
    publish_frozen_plan_set(plan, plan_root, artifacts=_plan_payloads(), artifact_root=pilot_root)
    return task_root, plan_root


def _attempt_manifest(
    pilot_root: Path,
    *,
    realization_index: int,
    status_class: str,
    episode: bool,
):
    from latency_meta_mdp.expert_realization.artifacts import ArtifactRef, AttemptManifest
    from latency_meta_mdp.expert_realization.contracts import AttemptId, ExpertRealizationId

    base = pilot_root / "task_instances/L1/seed-4000"
    task_ref_path = "task_instances/L1/seed-4000/task_instance/manifest.json"
    plan_ref_path = "task_instances/L1/seed-4000/plan_set/manifest.json"
    plan_sha = _sha((base / "plan_set/manifest.json").read_bytes())
    plan, verified_sha = __import__(
        "latency_meta_mdp.expert_realization.artifacts", fromlist=["load_verified_frozen_plan"]
    ).load_verified_frozen_plan(base / "plan_set", artifact_root=pilot_root)
    assert verified_sha == plan_sha
    key = plan.realizations[realization_index].expert_realization_key
    attempt_id = AttemptId(ExpertRealizationId(key, plan_sha), 0)
    status_payload = _json_bytes(
        {"format_id": "structured_expert_attempt_status_v1", "status_class": status_class}
    )
    qualification_payload = _json_bytes(
        {"format_id": "structured_expert_qualification_v1", "eligible": False}
    )
    manifest = AttemptManifest(
        schema_version=1,
        format_id="structured_expert_attempt_v1",
        attempt_id=attempt_id,
        task_instance_manifest=ArtifactRef(
            task_ref_path, _sha((pilot_root / task_ref_path).read_bytes())
        ),
        frozen_plan_set_manifest=ArtifactRef(plan_ref_path, plan_sha),
        status_class=status_class,
        status=ArtifactRef("status.json", _sha(status_payload)),
        episode_manifest=(
            ArtifactRef("synchronized_episode/manifest.json", "0" * 64) if episode else None
        ),
        qualification=(
            ArtifactRef("qualification.json", _sha(qualification_payload)) if episode else None
        ),
        implementation=_execution_implementation(),
        pilot_config_sha256="f" * 64,
        pilot_gate_config_sha256="0" * 64,
    )
    return manifest, status_payload, qualification_payload


def _episode_for_attempt(pilot: Path, plan_root: Path):
    from dataclasses import replace

    from latency_meta_mdp.expert_realization.contracts import AttemptId, ExpertRealizationId

    episode = make_episode()
    task = _task_manifest()
    plan_sha = _sha((plan_root / "manifest.json").read_bytes())
    plan = _plan_manifest(
        task_manifest_sha=_sha(
            (pilot / "task_instances/L1/seed-4000/task_instance/manifest.json").read_bytes()
        ),
        task_manifest_path="task_instances/L1/seed-4000/task_instance/manifest.json",
    )
    row = plan.realizations[0]
    realization = ExpertRealizationId(row.expert_realization_key, plan_sha)
    attempt = AttemptId(realization, 0)
    metadata = replace(
        episode.metadata,
        task_instance_id=task.task_instance_id,
        expert_realization_id=realization,
        attempt_id=attempt,
        task_config_sha256=task.task_config_sha256,
        motion_config_sha256=task.motion_config_sha256,
        runtime_config_sha256=task.runtime_config_sha256,
        controller_config_sha256=task.controller_config_sha256,
        structured_expert_config_sha256=plan.structured_expert_config_sha256,
        curobo_planner_config_sha256=plan.curobo_planner_config_sha256,
        task_instance_manifest_sha256=plan.task_instance_manifest.sha256,
        frozen_plan_set_manifest_sha256=plan_sha,
        realization_universe_sha256=plan.realization_universe_sha256,
        strategy_sha256=row.strategy.sha256,
        planner_candidates_sha256=row.planner_candidates.sha256,
        selected_reference_sha256=row.selected_reference.sha256,
        implementation=_execution_implementation(),
        pilot_config_sha256="f" * 64,
        pilot_gate_config_sha256="0" * 64,
    )
    return replace(
        episode,
        metadata=metadata,
        transitions=tuple(
            replace(
                transition,
                expert_audit=replace(
                    transition.expert_audit,
                    expert_realization_id=realization,
                ),
            )
            for transition in episode.transitions
        ),
    )


def test_task_manifest_is_strictly_preplan_and_binds_raw_motion_and_state_hashes() -> None:
    """Break caught: pre-plan identity embeds later IDs or hashes the wrong raw inputs."""
    from latency_meta_mdp.expert_realization.artifacts import TaskInstanceManifest

    manifest = _task_manifest()
    mapping = manifest.to_mapping()
    assert "realizations" not in mapping
    assert "task_instance_plan_set_sha256" not in json.dumps(mapping)
    assert (
        manifest.task_instance_id.motion_profile_sha256 == manifest.artifacts["motion_profile.json"]
    )
    assert manifest.task_instance_id.initial_state_sha256 == manifest.artifacts["initial_state.npz"]
    with pytest.raises(ValueError):
        TaskInstanceManifest.from_mapping({**mapping, "attempts": []})


def test_frozen_plan_contains_keys_only_and_raw_manifest_digest_finalizes_ids(
    tmp_path: Path,
) -> None:
    """Break caught: a circular finalized ID is serialized before the plan manifest hash exists."""
    from latency_meta_mdp.expert_realization.artifacts import load_verified_frozen_plan
    from latency_meta_mdp.expert_realization.contracts import ExpertRealizationId

    pilot = tmp_path / "pilot"
    _, plan_root = _publish_ancestors(pilot)
    loaded, raw_sha = load_verified_frozen_plan(plan_root, artifact_root=pilot)
    mapping = loaded.to_mapping()

    assert "expert_realization_id" not in json.dumps(mapping)
    assert raw_sha == _sha((plan_root / "manifest.json").read_bytes())
    assert (
        ExpertRealizationId(
            loaded.realizations[0].expert_realization_key, raw_sha
        ).task_instance_plan_set_sha256
        == raw_sha
    )
    circular = mapping.copy()
    circular["realizations"] = [
        {**circular["realizations"][0], "expert_realization_id": {"bad": True}},
        circular["realizations"][1],
    ]
    (plan_root / "manifest.json").write_bytes(_json_bytes(circular))
    with pytest.raises(ValueError):
        load_verified_frozen_plan(plan_root, artifact_root=pilot)


def test_frozen_plan_rejects_any_subset_of_the_eight_requested_keys() -> None:
    """Break caught: a partial planner result is published as the complete frozen key set."""
    from dataclasses import replace

    plan = _plan_manifest(
        task_manifest_sha=SHA_A,
        task_manifest_path="task_instances/L1/seed-4000/task_instance/manifest.json",
    )
    with pytest.raises(ValueError, match="universe"):
        replace(plan, realizations=plan.realizations[:1])


@pytest.mark.parametrize(
    ("status_class", "realization_index", "episode", "valid"),
    [
        ("planner_failure", 1, False, True),
        ("infrastructure_failure", 0, False, True),
        ("infrastructure_failure", 0, True, True),
        ("success", 0, False, False),
        ("task_failure", 0, False, False),
        ("safety_failure", 0, False, False),
        ("diversity_rejection", 0, False, False),
    ],
)
def test_attempt_status_contract_controls_episode_and_qualification_presence(
    tmp_path: Path, status_class: str, realization_index: int, episode: bool, valid: bool
) -> None:
    """Break caught: status classes publish an impossible partial or absent episode contract."""
    pilot = tmp_path / status_class
    _publish_ancestors(pilot)
    if valid:
        manifest, _, _ = _attempt_manifest(
            pilot,
            realization_index=realization_index,
            status_class=status_class,
            episode=episode,
        )
        assert manifest.status_class.value == status_class
    else:
        with pytest.raises(ValueError):
            _attempt_manifest(
                pilot,
                realization_index=realization_index,
                status_class=status_class,
                episode=episode,
            )


def test_attempt_publication_and_loader_verify_episode_and_ancestor_identity_joins(
    tmp_path: Path,
) -> None:
    """Break caught: a valid child is attached to the wrong task, plan digest, or realization."""
    from dataclasses import replace

    from latency_meta_mdp.expert_realization.artifacts import (
        load_verified_attempt,
        publish_attempt,
    )
    from latency_meta_mdp.expert_realization.recording_artifacts import write_structured_episode

    pilot = tmp_path / "pilot"
    _, plan_root = _publish_ancestors(pilot)
    episode = _episode_for_attempt(pilot, plan_root)
    episode_source = tmp_path / "episode-source"
    write_structured_episode(episode, episode_source)
    manifest, status, qualification = _attempt_manifest(
        pilot, realization_index=0, status_class="success", episode=True
    )
    manifest = replace(
        manifest,
        episode_manifest=replace(
            manifest.episode_manifest,
            sha256=_sha((episode_source / "manifest.json").read_bytes()),
        ),
    )
    target = pilot / "task_instances/L1/seed-4000/attempts/realization-000/attempt-000"
    publish_attempt(
        manifest,
        target,
        status=status,
        episode=episode_source,
        qualification=qualification,
        artifact_root=pilot,
    )
    loaded = load_verified_attempt(target, artifact_root=pilot)
    assert loaded.attempt_id == manifest.attempt_id
    assert loaded.status_class.value == "success"

    ancestor = pilot / manifest.frozen_plan_set_manifest.path
    original_ancestor = ancestor.read_bytes()
    ancestor.write_bytes(ancestor.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_verified_attempt(target, artifact_root=pilot)
    ancestor.write_bytes(original_ancestor)

    metadata_path = target / "synchronized_episode/metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["metadata"]["strategy_sha256"] = "9" * 64
    metadata_path.write_bytes(_json_bytes(metadata))
    episode_manifest_path = target / "synchronized_episode/manifest.json"
    episode_manifest = json.loads(episode_manifest_path.read_text(encoding="utf-8"))
    episode_manifest["artifacts"]["metadata.json"] = _sha(metadata_path.read_bytes())
    episode_manifest_path.write_bytes(_json_bytes(episode_manifest))
    attempt_manifest_path = target / "manifest.json"
    attempt_manifest = json.loads(attempt_manifest_path.read_text(encoding="utf-8"))
    attempt_manifest["episode_manifest"]["sha256"] = _sha(episode_manifest_path.read_bytes())
    attempt_manifest_path.write_bytes(_json_bytes(attempt_manifest))
    with pytest.raises(ValueError, match="provenance"):
        load_verified_attempt(target, artifact_root=pilot)


@pytest.mark.parametrize(
    "field",
    [
        "task_instance_manifest_sha256",
        "task_config_sha256",
        "motion_config_sha256",
        "runtime_config_sha256",
        "controller_config_sha256",
        "structured_expert_config_sha256",
        "curobo_planner_config_sha256",
        "strategy_sha256",
        "planner_candidates_sha256",
        "selected_reference_sha256",
        "pilot_config_sha256",
        "pilot_gate_config_sha256",
        "implementation",
    ],
)
def test_attempt_publication_rejects_each_false_episode_provenance_fact(
    tmp_path: Path, field: str
) -> None:
    """Break caught: one syntactically valid false provenance field crosses publication."""
    from dataclasses import replace

    from latency_meta_mdp.expert_realization.artifacts import publish_attempt
    from latency_meta_mdp.expert_realization.recording_artifacts import write_structured_episode

    pilot = tmp_path / field
    _, plan_root = _publish_ancestors(pilot)
    episode = _episode_for_attempt(pilot, plan_root)
    wrong_value = _implementation() if field == "implementation" else "9" * 64
    episode = replace(episode, metadata=replace(episode.metadata, **{field: wrong_value}))
    source = tmp_path / f"episode-{field}"
    write_structured_episode(episode, source)
    manifest, status, qualification = _attempt_manifest(
        pilot, realization_index=0, status_class="success", episode=True
    )
    manifest = replace(
        manifest,
        episode_manifest=replace(
            manifest.episode_manifest,
            sha256=_sha((source / "manifest.json").read_bytes()),
        ),
    )
    target = pilot / "task_instances/L1/seed-4000/attempts/realization-000/attempt-000"
    with pytest.raises(
        ValueError,
        match="provenance|metadata|hash|realization_seed|namespace",
    ):
        publish_attempt(
            manifest,
            target,
            status=status,
            episode=source,
            qualification=qualification,
            artifact_root=pilot,
        )


def test_attempt_publication_rejects_undeclared_legacy_child_and_external_symlink(
    tmp_path: Path,
) -> None:
    """Break caught: attempt ingestion recursively copies undeclared or external bytes."""
    from dataclasses import replace

    from latency_meta_mdp.expert_realization.artifacts import publish_attempt
    from latency_meta_mdp.expert_realization.recording_artifacts import write_structured_episode

    pilot = tmp_path / "pilot"
    _, plan_root = _publish_ancestors(pilot)
    episode = _episode_for_attempt(pilot, plan_root)
    source = tmp_path / "episode-source"
    write_structured_episode(episode, source)
    manifest, status, qualification = _attempt_manifest(
        pilot, realization_index=0, status_class="success", episode=True
    )
    manifest = replace(
        manifest,
        episode_manifest=replace(
            manifest.episode_manifest,
            sha256=_sha((source / "manifest.json").read_bytes()),
        ),
    )
    target = pilot / "task_instances/L1/seed-4000/attempts/realization-000/attempt-000"
    legacy = source / "legacy.json"
    legacy.write_bytes(_json_bytes({"format_id": "vision_feature_cache_v1"}))
    with pytest.raises(ValueError, match="inventory|legacy"):
        publish_attempt(
            manifest,
            target,
            status=status,
            episode=source,
            qualification=qualification,
            artifact_root=pilot,
        )
    legacy.unlink()
    outside = tmp_path / "outside.txt"
    outside.write_text("external", encoding="utf-8")
    (source / "external.txt").symlink_to(outside)
    with pytest.raises(ValueError, match="inventory|symlink"):
        publish_attempt(
            manifest,
            target,
            status=status,
            episode=source,
            qualification=qualification,
            artifact_root=pilot,
        )


def test_explicit_episode_snapshot_survives_source_mutation_and_copied_attempt_revalidates(
    tmp_path: Path,
) -> None:
    """Break caught: publication reopens mutable episode paths after verification."""
    from dataclasses import replace

    from latency_meta_mdp.expert_realization.artifacts import (
        load_verified_attempt,
        load_verified_attempt_bundle,
        publish_attempt,
    )
    from latency_meta_mdp.expert_realization.recording_artifacts import (
        load_verified_structured_episode_snapshot,
        write_structured_episode,
    )

    pilot = tmp_path / "pilot"
    _, plan_root = _publish_ancestors(pilot)
    source = tmp_path / "episode-source"
    write_structured_episode(_episode_for_attempt(pilot, plan_root), source)
    snapshot = load_verified_structured_episode_snapshot(source)
    manifest, status, qualification = _attempt_manifest(
        pilot, realization_index=0, status_class="success", episode=True
    )
    manifest = replace(
        manifest,
        episode_manifest=replace(manifest.episode_manifest, sha256=snapshot.manifest_sha256),
    )
    (source / "metadata.json").write_text("mutated after snapshot", encoding="utf-8")
    target = pilot / "task_instances/L1/seed-4000/attempts/realization-000/attempt-000"
    publish_attempt(
        manifest,
        target,
        status=status,
        episode=snapshot,
        qualification=qualification,
        artifact_root=pilot,
    )
    bundle = load_verified_attempt_bundle(target, artifact_root=pilot)
    assert bundle.manifest.attempt_id == manifest.attempt_id
    assert bundle.status_bytes == status
    assert bundle.qualification_bytes == qualification
    assert bundle.episode.files["manifest.json"] == snapshot.files["manifest.json"]
    assert load_verified_attempt(target, artifact_root=pilot).attempt_id == manifest.attempt_id
    copied = target / "synchronized_episode"
    (copied / "undeclared.txt").write_text("not in manifest", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        load_verified_attempt(target, artifact_root=pilot)


def test_verified_plan_bundle_retains_immutable_child_bytes_after_filesystem_mutation(
    tmp_path: Path,
) -> None:
    """Break caught: a verified consumer must reopen mutable plan paths in its hot loop."""
    from latency_meta_mdp.expert_realization.artifacts import (
        load_verified_frozen_plan_bundle,
    )

    pilot = tmp_path / "pilot"
    _, plan_root = _publish_ancestors(pilot)
    bundle = load_verified_frozen_plan_bundle(plan_root, artifact_root=pilot)
    path = "strategies/realization-000.json"
    verified = bundle.files[path]
    (plan_root / path).write_text("mutated", encoding="utf-8")

    assert bundle.manifest_sha256 == _sha(bundle.manifest_bytes)
    assert bundle.files[path] == verified
    assert bundle.task.manifest.task_instance_id == bundle.manifest.task_instance_id


def test_publish_failure_before_rename_leaves_no_final_or_loadable_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    """Break caught: an interrupted staging build becomes visible as a complete task instance."""
    from latency_meta_mdp.expert_realization import artifacts as module

    target = tmp_path / "task_instances/L1/seed-4000/task_instance"
    original = module._write_file_fsynced
    calls = 0

    def fail_before_manifest(path: Path, payload: bytes) -> None:
        nonlocal calls
        calls += 1
        if path.name == "manifest.json":
            raise OSError("injected before manifest")
        original(path, payload)

    monkeypatch.setattr(module, "_write_file_fsynced", fail_before_manifest)
    with pytest.raises(OSError, match="injected"):
        module.publish_task_instance(_task_manifest(), target, artifacts=_task_payloads())
    assert not target.exists()
    assert list(tmp_path.glob(".*.building-*")) == []


def test_atomic_noreplace_winner_never_mutates_existing_targets_even_in_race(
    tmp_path: Path,
) -> None:
    """Break caught: a precheck race lets the second worker replace the first publication."""
    from latency_meta_mdp.expert_realization.artifacts import publish_task_instance

    target = tmp_path / "task_instances/L1/seed-4000/task_instance"

    def publish() -> str:
        try:
            publish_task_instance(_task_manifest(), target, artifacts=_task_payloads())
            return "published"
        except FileExistsError:
            return "exists"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = sorted(pool.map(lambda _: publish(), range(2)))
    assert results == ["exists", "published"]
    before = (target / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        publish_task_instance(_task_manifest(), target, artifacts=_task_payloads())
    assert (target / "manifest.json").read_bytes() == before


def test_publication_fsyncs_every_file_and_bottom_up_directories_around_rename(
    tmp_path: Path, monkeypatch
) -> None:
    """Break caught: rename exposes bytes that were never made durable in filesystem order."""
    from latency_meta_mdp.expert_realization import artifacts as module

    pilot = tmp_path / "pilot"
    base = pilot / "task_instances/L1/seed-4000"
    task_root = base / "task_instance"
    module.publish_task_instance(_task_manifest(), task_root, artifacts=_task_payloads())
    plan = _plan_manifest(
        task_manifest_sha=_sha((task_root / "manifest.json").read_bytes()),
        task_manifest_path="task_instances/L1/seed-4000/task_instance/manifest.json",
    )
    events: list[tuple[str, str]] = []
    real_fsync = os.fsync
    real_rename = module._rename_noreplace

    def fsync(fd: int) -> None:
        path = os.readlink(f"/proc/self/fd/{fd}")
        events.append(("fsync", path))
        real_fsync(fd)

    def rename(source: Path, target: Path) -> None:
        events.append(("rename", str(target)))
        real_rename(source, target)

    monkeypatch.setattr(module.os, "fsync", fsync)
    monkeypatch.setattr(module, "_rename_noreplace", rename)
    target = base / "plan_set"
    module.publish_frozen_plan_set(plan, target, artifacts=_plan_payloads(), artifact_root=pilot)

    rename_index = next(index for index, event in enumerate(events) if event[0] == "rename")
    before = events[:rename_index]
    after = events[rename_index + 1 :]
    for name in (*_plan_payloads(), "manifest.json"):
        assert any(event[0] == "fsync" and event[1].endswith("/" + name) for event in before)
    for directory in ("strategies", "planner_candidates", "selected_references"):
        assert any(event[0] == "fsync" and event[1].endswith("/" + directory) for event in before)
    assert any(event[0] == "fsync" and ".building-" in event[1] for event in before)
    assert any(event[0] == "fsync" and event[1] == str(base) for event in after)


def test_publication_hashes_fsynced_children_before_writing_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    """Break caught: a manifest records input hashes before durable staged files exist."""
    from latency_meta_mdp.expert_realization import artifacts as module

    events: list[tuple[str, str]] = []
    real_write = module._write_file_fsynced
    real_hash = module._hash_file

    def write(path: Path, payload: bytes) -> None:
        real_write(path, payload)
        events.append(("write_fsynced", path.name))

    def hash_file(path: Path) -> str:
        events.append(("hash", path.name))
        return real_hash(path)

    monkeypatch.setattr(module, "_write_file_fsynced", write)
    monkeypatch.setattr(module, "_hash_file", hash_file)
    target = tmp_path / "task_instances/L1/seed-4000/task_instance"
    module.publish_task_instance(_task_manifest(), target, artifacts=_task_payloads())

    manifest_write = events.index(("write_fsynced", "manifest.json"))
    for name in _task_payloads():
        assert events.index(("write_fsynced", name)) < events.index(("hash", name))
        assert events.index(("hash", name)) < manifest_write


def test_verified_loaders_reject_path_traversal_mixed_legacy_and_raw_corruption(
    tmp_path: Path,
) -> None:
    """Break caught: a new parent trusts unsafe paths, historical children, or changed raw bytes."""
    from latency_meta_mdp.expert_realization.artifacts import load_verified_frozen_plan

    pilot = tmp_path / "pilot"
    _, plan_root = _publish_ancestors(pilot)
    strategy = plan_root / "strategies/realization-000.json"
    strategy.write_bytes(strategy.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_verified_frozen_plan(plan_root, artifact_root=pilot)

    pilot2 = tmp_path / "pilot2"
    _, plan_root2 = _publish_ancestors(pilot2)
    manifest_path = plan_root2 / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["realizations"][0]["strategy"]["path"] = "../../outside.json"
    manifest_path.write_bytes(_json_bytes(raw))
    with pytest.raises(ValueError, match="path"):
        load_verified_frozen_plan(plan_root2, artifact_root=pilot2)

    pilot3 = tmp_path / "pilot3"
    payloads = _task_payloads()
    payloads["task_instance.json"] = _json_bytes({"format_id": "panda_ball_bulk_range_v1"})
    with pytest.raises(ValueError, match="legacy"):
        __import__(
            "latency_meta_mdp.expert_realization.artifacts", fromlist=["publish_task_instance"]
        ).publish_task_instance(
            _task_manifest(),
            pilot3 / "task_instances/L1/seed-4000/task_instance",
            artifacts=payloads,
        )


@pytest.mark.parametrize(
    "legacy_id",
    [
        "panda_ball_bulk_first_tranche_v1",
        "panda_ball_bulk_range_v1",
        "expert_pilot_run_v1",
        "panda_ball_formal_corpus_v1",
        "vision_feature_cache_v1",
    ],
)
def test_new_plan_loader_rejects_every_locked_legacy_top_level_format(
    tmp_path: Path, legacy_id: str
) -> None:
    """Break caught: a flat bulk, pilot, corpus, or DINO source enters the new hierarchy."""
    from latency_meta_mdp.expert_realization.artifacts import load_verified_frozen_plan

    root = tmp_path / legacy_id
    root.mkdir()
    (root / "manifest.json").write_bytes(_json_bytes({"format_id": legacy_id}))
    with pytest.raises(ValueError):
        load_verified_frozen_plan(root, artifact_root=tmp_path)


def test_manifest_child_references_must_use_the_locked_hierarchical_layout() -> None:
    """Break caught: a valid hash is attached under an ambiguous or flat legacy-style path."""
    from latency_meta_mdp.expert_realization.artifacts import (
        AttemptManifest,
        FrozenPlanSetManifest,
    )

    plan = _plan_manifest(
        task_manifest_sha=SHA_A,
        task_manifest_path="task_instances/L1/seed-4000/task_instance/manifest.json",
    )
    raw_plan = plan.to_mapping()
    raw_plan["realizations"][0]["strategy"]["path"] = "strategies/wrong.json"
    with pytest.raises(ValueError, match="path"):
        FrozenPlanSetManifest.from_mapping(raw_plan)

    attempt = {
        "schema_version": 1,
        "format_id": "structured_expert_attempt_v1",
        "attempt_id": {
            "expert_realization_id": {
                "expert_realization_key": plan.realizations[0].expert_realization_key.to_mapping(),
                "task_instance_plan_set_sha256": SHA_A,
            },
            "attempt_index": 0,
        },
        "task_instance_manifest": {
            "path": "task_instances/L1/seed-4000/task_instance/manifest.json",
            "sha256": SHA_B,
        },
        "frozen_plan_set_manifest": {
            "path": "task_instances/L1/seed-4000/plan_set/manifest.json",
            "sha256": SHA_A,
        },
        "status_class": "infrastructure_failure",
        "status": {"path": "flat-status.json", "sha256": SHA_C},
        "episode_manifest": None,
        "qualification": None,
        "implementation": _execution_implementation().to_mapping(),
        "pilot_config_sha256": "f" * 64,
        "pilot_gate_config_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="status"):
        AttemptManifest.from_mapping(attempt, structured_expert_config_sha256=SHA_C)

    with pytest.raises(ValueError, match="path"):
        _plan_manifest(
            task_manifest_sha=SHA_A,
            task_manifest_path="alias/task_instance/manifest.json",
        )


def test_manifest_schema_versions_reject_boolean_true() -> None:
    """Break caught: manifest constructors treat boolean true as schema integer one."""
    from dataclasses import replace

    from latency_meta_mdp.expert_realization.artifacts import (
        ArtifactRef,
        AttemptManifest,
        StructuredPilotManifest,
    )

    task = _task_manifest()
    with pytest.raises(ValueError):
        replace(task, schema_version=True)
    plan = _plan_manifest(
        task_manifest_sha=SHA_A,
        task_manifest_path="task_instances/L1/seed-4000/task_instance/manifest.json",
    )
    with pytest.raises(ValueError):
        replace(plan, schema_version=True)
    attempt = {
        "schema_version": True,
        "format_id": "structured_expert_attempt_v1",
        "attempt_id": {
            "expert_realization_id": {
                "expert_realization_key": plan.realizations[0].expert_realization_key.to_mapping(),
                "task_instance_plan_set_sha256": SHA_A,
            },
            "attempt_index": 0,
        },
        "task_instance_manifest": {
            "path": "task_instances/L1/seed-4000/task_instance/manifest.json",
            "sha256": SHA_B,
        },
        "frozen_plan_set_manifest": {
            "path": "task_instances/L1/seed-4000/plan_set/manifest.json",
            "sha256": SHA_A,
        },
        "status_class": "infrastructure_failure",
        "status": {"path": "status.json", "sha256": SHA_C},
        "episode_manifest": None,
        "qualification": None,
        "implementation": _execution_implementation().to_mapping(),
        "pilot_config_sha256": "f" * 64,
        "pilot_gate_config_sha256": "0" * 64,
    }
    with pytest.raises(ValueError):
        AttemptManifest.from_mapping(attempt, structured_expert_config_sha256=SHA_C)
    with pytest.raises(ValueError):
        StructuredPilotManifest(
            schema_version=True,
            format_id="structured_expert_pilot_v1",
            pilot_id="pilot-v1",
            pilot_config_sha256=SHA_A,
            pilot_gate_config_sha256=SHA_B,
            structured_expert_config_sha256=SHA_C,
            curobo_planner_config_sha256=SHA_D,
            implementation=_implementation(),
            task_instances=(
                ArtifactRef("task_instances/L1/seed-4000/task_instance/manifest.json", SHA_A),
            ),
            frozen_plan_sets=(),
            attempts=(),
            bounded_review_only=True,
            training_eligibility={
                "formal_training_authorized": False,
                "formal_dino_cache_authorized": False,
                "jepa_training_authorized": False,
                "policy_training_authorized": False,
                "meta_policy_training_authorized": False,
            },
        )


def test_publishers_reject_noncanonical_target_aliases(tmp_path: Path) -> None:
    """Break caught: hashes are published under paths not derived from their bound identity."""
    from latency_meta_mdp.expert_realization.artifacts import (
        load_verified_task_instance,
        publish_task_instance,
    )

    with pytest.raises(ValueError, match="target"):
        publish_task_instance(
            _task_manifest(), tmp_path / "alias/task_instance", artifacts=_task_payloads()
        )
    canonical = tmp_path / "task_instances/L1/seed-4000/task_instance"
    publish_task_instance(_task_manifest(), canonical, artifacts=_task_payloads())
    alias = tmp_path / "copied/task_instance"
    alias.parent.mkdir()
    canonical.rename(alias)
    with pytest.raises(ValueError, match="canonical"):
        load_verified_task_instance(alias)


def test_canonical_publish_and_load_reject_intermediate_symlink_ancestry(
    tmp_path: Path,
) -> None:
    """Break caught: a lexical canonical path escapes through task_instances symlink."""
    from latency_meta_mdp.expert_realization.artifacts import (
        load_verified_task_instance,
        publish_task_instance,
    )

    pilot = tmp_path / "pilot"
    pilot.mkdir()
    outside = tmp_path / "outside/task_instances"
    outside.mkdir(parents=True)
    (pilot / "task_instances").symlink_to(outside, target_is_directory=True)
    target = pilot / "task_instances/L1/seed-4000/task_instance"
    with pytest.raises(ValueError, match="symlink"):
        publish_task_instance(_task_manifest(), target, artifacts=_task_payloads())

    real = tmp_path / "real/task_instances/L1/seed-4000/task_instance"
    publish_task_instance(_task_manifest(), real, artifacts=_task_payloads())
    load_alias = tmp_path / "load-alias"
    load_alias.symlink_to(tmp_path / "real", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        load_verified_task_instance(load_alias / "task_instances/L1/seed-4000/task_instance")


def test_plan_loader_rejects_symlinked_artifact_root_and_intermediate_child_dir(
    tmp_path: Path,
) -> None:
    """Break caught: verified plan refs cross a symlinked root or child directory."""
    from latency_meta_mdp.expert_realization.artifacts import load_verified_frozen_plan

    pilot = tmp_path / "pilot"
    _, plan_root = _publish_ancestors(pilot)
    alias = tmp_path / "pilot-alias"
    alias.symlink_to(pilot, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        load_verified_frozen_plan(
            alias / "task_instances/L1/seed-4000/plan_set", artifact_root=alias
        )

    strategies = plan_root / "strategies"
    real_strategies = plan_root / "real_strategies"
    strategies.rename(real_strategies)
    strategies.symlink_to(real_strategies.name, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        load_verified_frozen_plan(plan_root, artifact_root=pilot)


def test_publication_rejects_group_or_world_writable_parent(tmp_path: Path) -> None:
    """Break caught: cleanup runs beneath a publication parent outside the cooperative model."""
    from latency_meta_mdp.expert_realization.artifacts import publish_task_instance

    parent = tmp_path / "task_instances/L1/seed-4000"
    parent.mkdir(parents=True)
    parent.chmod(0o777)
    target = parent / "task_instance"
    try:
        with pytest.raises(ValueError, match="owner-only|writable"):
            publish_task_instance(_task_manifest(), target, artifacts=_task_payloads())
    finally:
        parent.chmod(0o755)


def test_fault_at_final_rename_cleans_only_owned_staging_and_leaves_no_final(
    tmp_path: Path, monkeypatch
) -> None:
    """Break caught: a rename failure leaves a visible final or a stale owned build."""
    from latency_meta_mdp.expert_realization import artifacts as module

    target = tmp_path / "task_instances/L1/seed-4000/task_instance"

    def fail_rename(source: Path, destination: Path) -> None:
        raise OSError("injected before final rename")

    monkeypatch.setattr(module, "_rename_noreplace", fail_rename)
    with pytest.raises(OSError, match="final rename"):
        module.publish_task_instance(_task_manifest(), target, artifacts=_task_payloads())
    assert not target.exists()
    assert list(tmp_path.glob(".*.building-*")) == []


def test_cleanup_refuses_to_remove_replaced_or_symlinked_staging_path(
    tmp_path: Path, monkeypatch
) -> None:
    """Break caught: failure cleanup recursively deletes a path that replaced its owned inode."""
    from latency_meta_mdp.expert_realization import artifacts as module

    victim = tmp_path / "victim"
    victim.mkdir()
    marker = victim / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    displaced: list[Path] = []
    original_write = module._write_file_fsynced

    def replace_staging_before_failure(path: Path, payload: bytes) -> None:
        if path.name != "manifest.json":
            original_write(path, payload)
            return
        staging = path.parent
        backup = tmp_path / "displaced-owned-staging"
        staging.rename(backup)
        staging.symlink_to(victim, target_is_directory=True)
        displaced.append(backup)
        raise OSError("injected after staging replacement")

    monkeypatch.setattr(module, "_write_file_fsynced", replace_staging_before_failure)
    target = tmp_path / "task_instances/L1/seed-4000/task_instance"
    with pytest.raises(OSError, match="injected after staging replacement"):
        module.publish_task_instance(_task_manifest(), target, artifacts=_task_payloads())

    assert marker.read_text(encoding="utf-8") == "keep"
    assert displaced[0].is_dir()


def test_pilot_manifest_is_strict_inventory_with_failures_and_all_training_flags_false() -> None:
    """Break caught: inventory loading drops failed attempts or silently authorizes training."""
    from latency_meta_mdp.expert_realization.artifacts import (
        ArtifactRef,
        StructuredPilotManifest,
    )

    ref = ArtifactRef("task_instances/L1/seed-4000/task_instance/manifest.json", SHA_A)
    pilot = StructuredPilotManifest(
        schema_version=1,
        format_id="structured_expert_pilot_v1",
        pilot_id="pilot-v1",
        pilot_config_sha256=SHA_A,
        pilot_gate_config_sha256=SHA_B,
        structured_expert_config_sha256=SHA_C,
        curobo_planner_config_sha256=SHA_D,
        implementation=_implementation(),
        task_instances=(ref,),
        frozen_plan_sets=(
            ArtifactRef("task_instances/L1/seed-4000/plan_set/manifest.json", SHA_B),
        ),
        attempts=(
            ArtifactRef(
                "task_instances/L1/seed-4000/attempts/realization-001/attempt-000/manifest.json",
                SHA_C,
            ),
        ),
        bounded_review_only=True,
        training_eligibility={
            "formal_training_authorized": False,
            "formal_dino_cache_authorized": False,
            "jepa_training_authorized": False,
            "policy_training_authorized": False,
            "meta_policy_training_authorized": False,
        },
    )
    assert StructuredPilotManifest.from_mapping(pilot.to_mapping()) == pilot
    second = ArtifactRef("task_instances/L1/seed-4001/task_instance/manifest.json", SHA_D)
    with pytest.raises(ValueError, match="order"):
        StructuredPilotManifest(
            **{
                **pilot.to_mapping(),
                "implementation": pilot.implementation,
                "task_instances": (second, ref),
                "frozen_plan_sets": pilot.frozen_plan_sets,
                "attempts": pilot.attempts,
            }
        )
    with pytest.raises(ValueError):
        StructuredPilotManifest.from_mapping(
            {**pilot.to_mapping(), "training_eligibility": {"formal_training_authorized": True}}
        )
