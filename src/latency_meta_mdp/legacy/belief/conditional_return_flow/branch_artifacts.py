"""Atomic artifacts for same-source-information control branches."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from latency_meta_mdp.io.artifacts import sha256_file
from latency_meta_mdp.legacy.belief.conditional_return_flow.branch_contracts import (
    BranchRollout,
    ControlContinuationSpec,
    ExecutablePrefix,
    SourceContextIdentity,
    SourceSelectionExclusion,
    normalize_scene_seed_ranges,
)

_ARTIFACT_NAMES = (
    "contexts.jsonl",
    "branch_parameters.jsonl",
    "branches.npz",
    "selection_exclusions.json",
    "selection_truncations.json",
)
_FORMAT_ID = "conditional_return_control_branch_corpus_v3"
_LEGACY_FORMAT_IDS = {
    "conditional_return_control_branch_corpus_v1",
    "conditional_return_control_branch_corpus_v2",
}
_PROVENANCE_FIELDS = {
    "implementation_revision",
    "implementation_source_sha256",
    "implementation_dirty",
    "input_paths",
    "input_sha256",
    "collection_wall_time_seconds",
    "peak_rss_bytes",
}
_INPUT_NAMES = {
    "source_bulk_manifest",
    "split_config",
    "temporal_config",
    "branch_config",
    "control_config",
}
_RESERVED_FIELDS = {
    "schema_version",
    "format_id",
    "scientific_gate_pass",
    "scientific_blockers",
    "artifact_eligible",
    "artifact_blockers",
    "bounded",
    "context_count",
    "branch_count",
    "selection_slot_count",
    "selection_exclusion_count",
    "selection_truncation_count",
    "allowed_scene_seed_ranges",
    "selected_episode_count",
    "selected_episode_identities_sha256",
    "requested_levels",
    "required_branch_kinds",
    "artifacts",
}
_MANIFEST_FIELDS = _PROVENANCE_FIELDS | _RESERVED_FIELDS


def _is_sha256(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_revision(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _selected_episode_identities(
    corpus: ControlBranchCorpus,
) -> tuple[tuple[str, int, int, str], ...]:
    identities: dict[str, tuple[str, int, int, str]] = {}
    source_hashes: dict[str, tuple[str, str, str, str]] = {}
    for row in (*corpus.contexts, *corpus.selection_truncations):
        identity = (row.episode_id, row.level, row.scene_seed, row.split)
        hashes = (
            row.source_episode_manifest_sha256,
            row.source_arrays_sha256,
            row.source_metadata_sha256,
            row.motion_profile_sha256,
        )
        if (
            identities.setdefault(row.episode_id, identity) != identity
            or source_hashes.setdefault(row.episode_id, hashes) != hashes
        ):
            raise ValueError("source selection episode identity is inconsistent")
    for row in corpus.selection_exclusions:
        identity = (row.episode_id, row.level, row.scene_seed, row.split)
        if identities.setdefault(row.episode_id, identity) != identity:
            raise ValueError("source selection episode identity is inconsistent")
    return tuple(sorted(identities.values()))


def _episode_identities_sha256(values: tuple[tuple[str, int, int, str], ...]) -> str:
    payload = json.dumps(list(values), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_provenance(value: dict[str, Any]) -> None:
    if set(value) != _PROVENANCE_FIELDS:
        raise ValueError("control branch provenance fields are invalid")
    paths = value["input_paths"]
    digests = value["input_sha256"]
    wall_time = value["collection_wall_time_seconds"]
    peak_rss = value["peak_rss_bytes"]
    if (
        not _is_git_revision(value["implementation_revision"])
        or not _is_sha256(value["implementation_source_sha256"])
        or type(value["implementation_dirty"]) is not bool
        or not isinstance(paths, dict)
        or not isinstance(digests, dict)
        or set(paths) != _INPUT_NAMES
        or set(digests) != _INPUT_NAMES
        or any(not isinstance(name, str) or not name for name in paths)
        or any(not isinstance(path, str) or not path for path in paths.values())
        or any(not _is_sha256(digest) for digest in digests.values())
        or isinstance(wall_time, bool)
        or not isinstance(wall_time, (int, float))
        or not np.isfinite(wall_time)
        or wall_time < 0.0
        or isinstance(peak_rss, bool)
        or not isinstance(peak_rss, int)
        or peak_rss <= 0
    ):
        raise ValueError("control branch provenance values are invalid")


def _validate_manifest_contract(manifest: dict[str, Any]) -> None:
    if manifest.get("format_id") in _LEGACY_FORMAT_IDS or manifest.get("schema_version") in {1, 2}:
        raise ValueError("legacy v1/v2 control branch artifacts require explicit recertification")
    if manifest.get("schema_version") != 3 or manifest.get("format_id") != _FORMAT_ID:
        raise ValueError("unsupported control branch artifact schema")
    if set(manifest) != _MANIFEST_FIELDS:
        raise ValueError("control branch manifest fields are invalid")
    _validate_provenance({name: manifest[name] for name in _PROVENANCE_FIELDS})


@dataclass(frozen=True)
class ControlBranchCorpus:
    contexts: tuple[SourceContextIdentity, ...]
    canonical_replay_fingerprints: tuple[str, ...]
    rollouts: tuple[BranchRollout, ...]
    selection_exclusions: tuple[SourceSelectionExclusion, ...]
    expected_selection_slot_count: int
    required_branch_kinds: tuple[str, ...]
    requested_levels: tuple[int, ...]
    selection_truncations: tuple[SourceContextIdentity, ...] = ()
    allowed_scene_seed_ranges: tuple[tuple[int, int], ...] | None = None

    def __post_init__(self) -> None:
        contexts = tuple(self.contexts)
        fingerprints = tuple(self.canonical_replay_fingerprints)
        rollouts = tuple(self.rollouts)
        exclusions = tuple(self.selection_exclusions)
        truncations = tuple(self.selection_truncations)
        seed_ranges = normalize_scene_seed_ranges(self.allowed_scene_seed_ranges)
        if not contexts or not rollouts or len(fingerprints) != len(contexts):
            raise ValueError("control branch corpus cannot be empty or misaligned")
        identifiers = tuple(row.source_context_id for row in contexts)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("control branch context IDs must be unique")
        if any(not _is_sha256(value) for value in fingerprints):
            raise ValueError("canonical replay fingerprints must be SHA256 digests")
        if any(not isinstance(row, SourceSelectionExclusion) for row in exclusions):
            raise TypeError("source selection exclusions must use the typed contract")
        if any(not isinstance(row, SourceContextIdentity) for row in truncations):
            raise TypeError("source selection truncations must use source identities")
        if (
            isinstance(self.expected_selection_slot_count, bool)
            or not isinstance(self.expected_selection_slot_count, int)
            or self.expected_selection_slot_count <= 0
            or len(contexts) + len(exclusions) + len(truncations)
            != self.expected_selection_slot_count
        ):
            raise ValueError("source selection slots are not completely accounted for")
        context_slots = {(row.episode_id, row.source_phase) for row in contexts}
        exclusion_slots = {(row.episode_id, row.source_phase) for row in exclusions}
        truncation_slots = {(row.episode_id, row.source_phase) for row in truncations}
        if (
            len(context_slots) != len(contexts)
            or len(exclusion_slots) != len(exclusions)
            or len(truncation_slots) != len(truncations)
            or context_slots & exclusion_slots
            or context_slots & truncation_slots
            or exclusion_slots & truncation_slots
        ):
            raise ValueError("source selection slots must be unique and disjoint")
        episode_identities = _selected_episode_identities(self)
        if len(episode_identities) * 4 != self.expected_selection_slot_count:
            raise ValueError("source selection slots do not form complete episode inventories")
        selection_rows = (*contexts, *truncations, *exclusions)
        if seed_ranges is not None and any(
            not any(start <= row.scene_seed < stop for start, stop in seed_ranges)
            for row in selection_rows
        ):
            raise ValueError("source selection contains a seed outside its admitted ranges")
        if not self.required_branch_kinds or len(set(self.required_branch_kinds)) != len(
            self.required_branch_kinds
        ):
            raise ValueError("required branch kinds must be non-empty and unique")
        if (
            not self.requested_levels
            or self.requested_levels != tuple(sorted(set(self.requested_levels)))
            or any(level not in (1, 2, 3) for level in self.requested_levels)
        ):
            raise ValueError("requested levels must be sorted unique values from 1, 2, 3")
        known = set(identifiers)
        if any(row.source.source_context_id not in known for row in rollouts):
            raise ValueError("branch rollout references an unknown source context")
        object.__setattr__(self, "contexts", contexts)
        object.__setattr__(self, "canonical_replay_fingerprints", fingerprints)
        object.__setattr__(self, "rollouts", rollouts)
        object.__setattr__(self, "selection_exclusions", exclusions)
        object.__setattr__(self, "selection_truncations", truncations)
        object.__setattr__(self, "allowed_scene_seed_ranges", seed_ranges)


@dataclass(frozen=True)
class VerifiedControlBranchCorpus:
    manifest_path: Path
    scientific_gate_pass: bool
    artifact_eligible: bool
    selection_exclusions: tuple[SourceSelectionExclusion, ...]
    selection_truncations: tuple[SourceContextIdentity, ...]
    source_context_index: np.ndarray
    branch_kind: np.ndarray
    executable_controls: np.ndarray
    from_active_buffer_mask: np.ndarray
    target_states: np.ndarray
    target_absorbing: np.ndarray
    target_handoff_state: np.ndarray
    target_outcome_status: np.ndarray
    source_replay_max_abs: np.ndarray
    source_fingerprint_match: np.ndarray
    nominal_future_valid: np.ndarray
    nominal_future_max_abs: np.ndarray


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_no_replace(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"control branch output exists: {target}")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace publication requires renameat2")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1)
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(f"control branch output exists: {target}")
    raise OSError(error, os.strerror(error), str(target))


def _absorbing_suffix_is_valid(rollout: BranchRollout) -> bool:
    indices = np.flatnonzero(rollout.target_absorbing)
    if len(indices) == 0:
        return True
    first = int(indices[0])
    if not np.all(rollout.target_absorbing[first:]):
        return False
    return bool(
        np.array_equal(
            rollout.target_states[first:],
            np.repeat(rollout.target_states[first : first + 1], 20 - first, axis=0),
        )
    )


def _scientific_blockers(corpus: ControlBranchCorpus) -> list[str]:
    blockers = []
    if set(corpus.requested_levels) - {row.level for row in corpus.contexts}:
        blockers.append("missing_requested_level")
    by_source = {row.source_context_id: set() for row in corpus.contexts}
    for rollout in corpus.rollouts:
        by_source[rollout.source.source_context_id].add(rollout.branch.kind)
        if rollout.source_replay_max_abs != 0.0:
            blockers.append("source_replay_mismatch")
        if not rollout.source_fingerprint_match:
            blockers.append("source_fingerprint_mismatch")
        if rollout.branch.kind == "nominal":
            if not rollout.nominal_future_valid:
                blockers.append("missing_nominal_future")
            elif rollout.nominal_future_max_abs > 1e-6:
                blockers.append("nominal_future_mismatch")
        elif rollout.nominal_future_valid:
            blockers.append("unexpected_non_nominal_future")
        if not _absorbing_suffix_is_valid(rollout):
            blockers.append("invalid_absorbing_suffix")
    required = set(corpus.required_branch_kinds)
    if any(not required <= observed for observed in by_source.values()):
        blockers.append("incomplete_branch_inventory")
    return sorted(set(blockers))


def _flatten(corpus: ControlBranchCorpus) -> dict[str, np.ndarray]:
    context_index = {row.source_context_id: index for index, row in enumerate(corpus.contexts)}
    return {
        "source_context_index": np.asarray(
            [context_index[row.source.source_context_id] for row in corpus.rollouts],
            dtype=np.int64,
        ),
        "branch_kind": np.asarray([row.branch.kind for row in corpus.rollouts]),
        "executable_controls": np.stack([row.prefix.controls for row in corpus.rollouts]),
        "from_active_buffer_mask": np.stack(
            [row.prefix.from_active_buffer_mask for row in corpus.rollouts]
        ),
        "target_states": np.stack([row.target_states for row in corpus.rollouts]),
        "target_absorbing": np.stack([row.target_absorbing for row in corpus.rollouts]),
        "target_handoff_state": np.stack([row.target_handoff_state for row in corpus.rollouts]),
        "target_outcome_status": np.stack([row.target_outcome_status for row in corpus.rollouts]),
        "source_replay_max_abs": np.asarray(
            [row.source_replay_max_abs for row in corpus.rollouts], dtype=np.float64
        ),
        "source_fingerprint_match": np.asarray(
            [row.source_fingerprint_match for row in corpus.rollouts], dtype=bool
        ),
        "nominal_future_valid": np.asarray(
            [row.nominal_future_valid for row in corpus.rollouts], dtype=bool
        ),
        "nominal_future_max_abs": np.asarray(
            [row.nominal_future_max_abs for row in corpus.rollouts], dtype=np.float64
        ),
    }


def write_control_branch_corpus(
    *,
    corpus: ControlBranchCorpus,
    output_dir: Path,
    manifest_fields: dict[str, Any],
    bounded: bool,
) -> Path:
    target = output_dir.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"control branch output exists: {target}")
    if _RESERVED_FIELDS & set(manifest_fields):
        raise ValueError("control branch manifest fields overwrite reserved fields")
    _validate_provenance(manifest_fields)
    if type(bounded) is not bool or type(manifest_fields.get("implementation_dirty")) is not bool:
        raise TypeError("bounded and implementation_dirty must be booleans")
    if not bounded and corpus.selection_truncations:
        raise ValueError("unbounded control branch corpus cannot truncate source selection")
    scientific_blockers = _scientific_blockers(corpus)
    scientific_pass = not scientific_blockers
    artifact_blockers = []
    if bounded:
        artifact_blockers.append("bounded_review")
    if manifest_fields["implementation_dirty"]:
        artifact_blockers.append("implementation_dirty")
    artifact_eligible = scientific_pass and not artifact_blockers
    arrays = _flatten(corpus)
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f".{target.name}.building-{uuid.uuid4().hex}"
    building.mkdir()
    try:
        context_rows = []
        for identity, fingerprint in zip(
            corpus.contexts, corpus.canonical_replay_fingerprints, strict=True
        ):
            context_rows.append(
                {**asdict(identity), "canonical_replay_fingerprint_sha256": fingerprint}
            )
        _write_jsonl(building / "contexts.jsonl", context_rows)
        parameter_rows = [
            {
                "branch_index": index,
                "source_context_id": row.source.source_context_id,
                "kind": row.branch.kind,
                "arm_scale": row.branch.arm_scale,
                "prefix_real_ticks": row.branch.prefix_real_ticks,
            }
            for index, row in enumerate(corpus.rollouts)
        ]
        _write_jsonl(building / "branch_parameters.jsonl", parameter_rows)
        with (building / "branches.npz").open("xb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        _write_json(
            building / "selection_exclusions.json",
            [asdict(row) for row in corpus.selection_exclusions],
        )
        _write_json(
            building / "selection_truncations.json",
            [asdict(row) for row in corpus.selection_truncations],
        )
        artifacts = {name: sha256_file(building / name) for name in _ARTIFACT_NAMES}
        episode_identities = _selected_episode_identities(corpus)
        manifest = {
            **manifest_fields,
            "schema_version": 3,
            "format_id": _FORMAT_ID,
            "scientific_gate_pass": scientific_pass,
            "scientific_blockers": scientific_blockers,
            "artifact_eligible": artifact_eligible,
            "artifact_blockers": artifact_blockers,
            "bounded": bounded,
            "context_count": len(corpus.contexts),
            "branch_count": len(corpus.rollouts),
            "selection_slot_count": corpus.expected_selection_slot_count,
            "selection_exclusion_count": len(corpus.selection_exclusions),
            "selection_truncation_count": len(corpus.selection_truncations),
            "allowed_scene_seed_ranges": (
                None
                if corpus.allowed_scene_seed_ranges is None
                else [list(row) for row in corpus.allowed_scene_seed_ranges]
            ),
            "selected_episode_count": len(episode_identities),
            "selected_episode_identities_sha256": _episode_identities_sha256(episode_identities),
            "requested_levels": list(corpus.requested_levels),
            "required_branch_kinds": list(corpus.required_branch_kinds),
            "artifacts": artifacts,
        }
        _write_json(building / "manifest.json", manifest)
        _fsync_directory(building)
        _rename_directory_no_replace(building, target)
        _fsync_directory(target.parent)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return target / "manifest.json"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            raise ValueError(f"control branch JSONL contains a blank row: {path}:{line_number}")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"control branch JSONL row must be an object: {path}:{line_number}")
        rows.append(row)
    return rows


def _load_context_contract(path: Path) -> tuple[tuple[SourceContextIdentity, ...], tuple[str, ...]]:
    identity_fields = set(SourceContextIdentity.__dataclass_fields__)
    expected_fields = identity_fields | {"canonical_replay_fingerprint_sha256"}
    identities = []
    fingerprints = []
    for row in _read_jsonl(path):
        if set(row) != expected_fields:
            raise ValueError("control branch context fields are invalid")
        values = {name: row[name] for name in identity_fields}
        identity = SourceContextIdentity(**values)
        fingerprint = row["canonical_replay_fingerprint_sha256"]
        if not isinstance(fingerprint, str) or not _is_sha256(fingerprint):
            raise ValueError("canonical replay fingerprint is invalid")
        identities.append(identity)
        fingerprints.append(fingerprint)
    return tuple(identities), tuple(fingerprints)


def _load_exclusion_contract(path: Path) -> tuple[SourceSelectionExclusion, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("source selection exclusions must be a JSON array")
    expected_fields = set(SourceSelectionExclusion.__dataclass_fields__)
    exclusions = []
    for row in payload:
        if not isinstance(row, dict) or set(row) != expected_fields:
            raise ValueError("source selection exclusion fields are invalid")
        exclusions.append(SourceSelectionExclusion(**row))
    return tuple(exclusions)


def _load_truncation_contract(path: Path) -> tuple[SourceContextIdentity, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("source selection truncations must be a JSON array")
    expected_fields = set(SourceContextIdentity.__dataclass_fields__)
    truncations = []
    for row in payload:
        if not isinstance(row, dict) or set(row) != expected_fields:
            raise ValueError("source selection truncation fields are invalid")
        truncations.append(SourceContextIdentity(**row))
    return tuple(truncations)


def _load_parameter_contract(path: Path) -> list[dict[str, Any]]:
    expected_fields = {
        "branch_index",
        "source_context_id",
        "kind",
        "arm_scale",
        "prefix_real_ticks",
    }
    rows = _read_jsonl(path)
    for index, row in enumerate(rows):
        if set(row) != expected_fields or row["branch_index"] != index:
            raise ValueError("control branch parameter fields or indices are invalid")
        if not isinstance(row["source_context_id"], str):
            raise ValueError("control branch parameter source ID must be a string")
        ControlContinuationSpec(
            kind=row["kind"],
            arm_scale=row["arm_scale"],
            prefix_real_ticks=row["prefix_real_ticks"],
        )
    return rows


def _require_manifest_integer(manifest: dict[str, Any], name: str) -> int:
    value = manifest.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"control branch manifest {name} must be a non-negative integer")
    return value


def _reconstruct_loaded_corpus(
    *,
    manifest: dict[str, Any],
    contexts: tuple[SourceContextIdentity, ...],
    fingerprints: tuple[str, ...],
    exclusions: tuple[SourceSelectionExclusion, ...],
    truncations: tuple[SourceContextIdentity, ...],
    allowed_scene_seed_ranges: tuple[tuple[int, int], ...] | None,
    parameters: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
) -> ControlBranchCorpus:
    branch_count = len(parameters)
    expected_shapes_and_dtypes = {
        "source_context_index": ((branch_count,), np.dtype(np.int64)),
        "executable_controls": ((branch_count, 20, 7), np.dtype(np.float32)),
        "from_active_buffer_mask": ((branch_count, 20), np.dtype(np.bool_)),
        "target_states": ((branch_count, 20, 22), np.dtype(np.float32)),
        "target_absorbing": ((branch_count, 20), np.dtype(np.bool_)),
        "source_replay_max_abs": ((branch_count,), np.dtype(np.float64)),
        "source_fingerprint_match": ((branch_count,), np.dtype(np.bool_)),
        "nominal_future_valid": ((branch_count,), np.dtype(np.bool_)),
        "nominal_future_max_abs": ((branch_count,), np.dtype(np.float64)),
    }
    for name, (shape, dtype) in expected_shapes_and_dtypes.items():
        if arrays[name].shape != shape or arrays[name].dtype != dtype:
            raise ValueError(f"control branch array contract is invalid: {name}")
    for name in ("branch_kind", "target_handoff_state", "target_outcome_status"):
        expected_shape = (branch_count,) if name == "branch_kind" else (branch_count, 20)
        if arrays[name].shape != expected_shape or arrays[name].dtype.kind not in {"U", "S"}:
            raise ValueError(f"control branch string array contract is invalid: {name}")
    numeric_names = {
        "executable_controls",
        "target_states",
        "source_replay_max_abs",
        "nominal_future_max_abs",
    }
    if any(not np.all(np.isfinite(arrays[name])) for name in numeric_names):
        raise ValueError("control branch numeric arrays must be finite")
    controls = arrays["executable_controls"]
    if np.any(controls < -1.0) or np.any(controls > 1.0):
        raise ValueError("control branch executable controls violate ActionContract")

    requested_levels_raw = manifest.get("requested_levels")
    required_kinds_raw = manifest.get("required_branch_kinds")
    if (
        not isinstance(requested_levels_raw, list)
        or not requested_levels_raw
        or any(
            isinstance(value, bool) or not isinstance(value, int) for value in requested_levels_raw
        )
        or not isinstance(required_kinds_raw, list)
        or not required_kinds_raw
        or any(not isinstance(value, str) for value in required_kinds_raw)
    ):
        raise ValueError("control branch requested levels or branch kinds are invalid")
    requested_levels = tuple(requested_levels_raw)
    required_kinds = tuple(required_kinds_raw)

    rollouts = []
    for index, parameter in enumerate(parameters):
        context_index = int(arrays["source_context_index"][index])
        if not 0 <= context_index < len(contexts):
            raise ValueError("control branch source context index is out of range")
        source = contexts[context_index]
        kind = str(arrays["branch_kind"][index])
        if parameter["source_context_id"] != source.source_context_id or parameter["kind"] != kind:
            raise ValueError("control branch parameter rows disagree with numeric arrays")
        branch = ControlContinuationSpec(
            kind=kind,
            arm_scale=parameter["arm_scale"],
            prefix_real_ticks=parameter["prefix_real_ticks"],
        )
        controls = arrays["executable_controls"][index]
        rollout = BranchRollout(
            source=source,
            branch=branch,
            prefix=ExecutablePrefix(
                controls=controls,
                from_active_buffer_mask=arrays["from_active_buffer_mask"][index],
                last_executed_gripper_command=float(controls[0, 6]),
            ),
            target_states=arrays["target_states"][index],
            target_absorbing=arrays["target_absorbing"][index],
            target_handoff_state=arrays["target_handoff_state"][index],
            target_outcome_status=arrays["target_outcome_status"][index],
            source_replay_max_abs=float(arrays["source_replay_max_abs"][index]),
            source_fingerprint_match=bool(arrays["source_fingerprint_match"][index]),
            nominal_future_valid=bool(arrays["nominal_future_valid"][index]),
            nominal_future_max_abs=float(arrays["nominal_future_max_abs"][index]),
        )
        rollouts.append(rollout)
    corpus = ControlBranchCorpus(
        contexts=contexts,
        canonical_replay_fingerprints=fingerprints,
        rollouts=tuple(rollouts),
        selection_exclusions=exclusions,
        selection_truncations=truncations,
        expected_selection_slot_count=_require_manifest_integer(manifest, "selection_slot_count"),
        required_branch_kinds=required_kinds,
        requested_levels=requested_levels,
        allowed_scene_seed_ranges=allowed_scene_seed_ranges,
    )
    observed_levels = (
        {row.level for row in contexts}
        | {row.level for row in exclusions}
        | {row.level for row in truncations}
    )
    if observed_levels != set(requested_levels):
        raise ValueError("control branch requested levels do not match child records")
    return corpus


def load_verified_control_branch_corpus(manifest_path: Path) -> VerifiedControlBranchCorpus:
    path = manifest_path.resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("control branch manifest must be a JSON object")
    _validate_manifest_contract(manifest)
    if not isinstance(manifest["artifacts"], dict) or set(manifest["artifacts"]) != set(
        _ARTIFACT_NAMES
    ):
        raise ValueError("control branch artifact inventory is invalid")
    for name in _ARTIFACT_NAMES:
        artifact_path = path.parent / name
        if not artifact_path.is_file() or sha256_file(artifact_path) != manifest["artifacts"].get(
            name
        ):
            raise ValueError(f"control branch artifact hash mismatch: {name}")
    contexts, fingerprints = _load_context_contract(path.parent / "contexts.jsonl")
    parameters = _load_parameter_contract(path.parent / "branch_parameters.jsonl")
    exclusions = _load_exclusion_contract(path.parent / "selection_exclusions.json")
    truncations = _load_truncation_contract(path.parent / "selection_truncations.json")
    context_count = _require_manifest_integer(manifest, "context_count")
    branch_count = _require_manifest_integer(manifest, "branch_count")
    selection_slot_count = _require_manifest_integer(manifest, "selection_slot_count")
    selection_exclusion_count = _require_manifest_integer(manifest, "selection_exclusion_count")
    selection_truncation_count = _require_manifest_integer(manifest, "selection_truncation_count")
    allowed_scene_seed_ranges = normalize_scene_seed_ranges(manifest["allowed_scene_seed_ranges"])
    selected_episode_count = _require_manifest_integer(manifest, "selected_episode_count")
    selected_episode_identities_sha256 = manifest["selected_episode_identities_sha256"]
    if not _is_sha256(selected_episode_identities_sha256):
        raise ValueError("selected episode identity digest is invalid")
    if (
        context_count != len(contexts)
        or branch_count != len(parameters)
        or selection_slot_count != len(contexts) + len(exclusions) + len(truncations)
        or selection_exclusion_count != len(exclusions)
        or selection_truncation_count != len(truncations)
    ):
        raise ValueError("control branch manifest counts do not match child artifacts")
    with np.load(path.parent / "branches.npz", allow_pickle=False) as source:
        arrays = {name: np.array(source[name], copy=True) for name in source.files}
    required = {
        "source_context_index",
        "branch_kind",
        "executable_controls",
        "from_active_buffer_mask",
        "target_states",
        "target_absorbing",
        "target_handoff_state",
        "target_outcome_status",
        "source_replay_max_abs",
        "source_fingerprint_match",
        "nominal_future_valid",
        "nominal_future_max_abs",
    }
    if set(arrays) != required:
        raise ValueError("control branch numeric artifact fields are invalid")
    corpus = _reconstruct_loaded_corpus(
        manifest=manifest,
        contexts=contexts,
        fingerprints=fingerprints,
        exclusions=exclusions,
        truncations=truncations,
        allowed_scene_seed_ranges=allowed_scene_seed_ranges,
        parameters=parameters,
        arrays=arrays,
    )
    selected_episode_identities = _selected_episode_identities(corpus)
    if selected_episode_count != len(
        selected_episode_identities
    ) or selected_episode_identities_sha256 != _episode_identities_sha256(
        selected_episode_identities
    ):
        raise ValueError("selected episode identity provenance is inconsistent")
    scientific_blockers = _scientific_blockers(corpus)
    scientific_gate_pass = not scientific_blockers
    if (
        type(manifest.get("scientific_gate_pass")) is not bool
        or manifest.get("scientific_gate_pass") is not scientific_gate_pass
        or manifest.get("scientific_blockers") != scientific_blockers
    ):
        raise ValueError("control branch scientific status is inconsistent with child artifacts")
    bounded = manifest.get("bounded")
    implementation_dirty = manifest.get("implementation_dirty")
    if type(bounded) is not bool or type(implementation_dirty) is not bool:
        raise ValueError("control branch bounded and implementation-dirty status must be boolean")
    if not bounded and truncations:
        raise ValueError("unbounded control branch artifact cannot contain selection truncations")
    expected_artifact_blockers = []
    if bounded:
        expected_artifact_blockers.append("bounded_review")
    if implementation_dirty:
        expected_artifact_blockers.append("implementation_dirty")
    expected_artifact_eligible = scientific_gate_pass and not expected_artifact_blockers
    if (
        type(manifest.get("artifact_eligible")) is not bool
        or manifest.get("artifact_eligible") is not expected_artifact_eligible
        or manifest.get("artifact_blockers") != expected_artifact_blockers
    ):
        raise ValueError("control branch artifact eligibility status is inconsistent")
    for value in arrays.values():
        value.setflags(write=False)
    return VerifiedControlBranchCorpus(
        manifest_path=path,
        scientific_gate_pass=scientific_gate_pass,
        artifact_eligible=expected_artifact_eligible,
        selection_exclusions=exclusions,
        selection_truncations=truncations,
        **arrays,
    )
