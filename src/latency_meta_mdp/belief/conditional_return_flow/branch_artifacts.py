"""Atomic artifacts for same-source-information control branches."""

from __future__ import annotations

import ctypes
import errno
import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from latency_meta_mdp.artifacts import sha256_file
from latency_meta_mdp.belief.conditional_return_flow.branch_contracts import (
    BranchRollout,
    SourceContextIdentity,
)

_ARTIFACT_NAMES = (
    "contexts.jsonl",
    "branch_parameters.jsonl",
    "branches.npz",
    "selection_exclusions.json",
)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True)
class ControlBranchCorpus:
    contexts: tuple[SourceContextIdentity, ...]
    canonical_replay_fingerprints: tuple[str, ...]
    rollouts: tuple[BranchRollout, ...]
    selection_exclusions: tuple[dict[str, Any], ...]
    required_branch_kinds: tuple[str, ...]
    requested_levels: tuple[int, ...]

    def __post_init__(self) -> None:
        contexts = tuple(self.contexts)
        fingerprints = tuple(self.canonical_replay_fingerprints)
        rollouts = tuple(self.rollouts)
        if not contexts or not rollouts or len(fingerprints) != len(contexts):
            raise ValueError("control branch corpus cannot be empty or misaligned")
        identifiers = tuple(row.source_context_id for row in contexts)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("control branch context IDs must be unique")
        if any(not _is_sha256(value) for value in fingerprints):
            raise ValueError("canonical replay fingerprints must be SHA256 digests")
        if (
            not self.required_branch_kinds
            or len(set(self.required_branch_kinds)) != len(self.required_branch_kinds)
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
        object.__setattr__(self, "selection_exclusions", tuple(self.selection_exclusions))


@dataclass(frozen=True)
class VerifiedControlBranchCorpus:
    manifest_path: Path
    scientific_gate_pass: bool
    artifact_eligible: bool
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
        if rollout.nominal_future_valid and rollout.nominal_future_max_abs > 1e-6:
            blockers.append("nominal_future_mismatch")
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
        "target_handoff_state": np.stack(
            [row.target_handoff_state for row in corpus.rollouts]
        ),
        "target_outcome_status": np.stack(
            [row.target_outcome_status for row in corpus.rollouts]
        ),
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
    reserved = {
        "schema_version",
        "format_id",
        "scientific_gate_pass",
        "scientific_blockers",
        "artifact_eligible",
        "artifact_blockers",
        "bounded",
        "context_count",
        "branch_count",
        "requested_levels",
        "required_branch_kinds",
        "artifacts",
    }
    if reserved & set(manifest_fields):
        raise ValueError("control branch manifest fields overwrite reserved fields")
    if type(bounded) is not bool or type(manifest_fields.get("implementation_dirty")) is not bool:
        raise TypeError("bounded and implementation_dirty must be booleans")
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
        _write_json(building / "selection_exclusions.json", list(corpus.selection_exclusions))
        artifacts = {name: sha256_file(building / name) for name in _ARTIFACT_NAMES}
        manifest = {
            **manifest_fields,
            "schema_version": 1,
            "format_id": "conditional_return_control_branch_corpus_v1",
            "scientific_gate_pass": scientific_pass,
            "scientific_blockers": scientific_blockers,
            "artifact_eligible": artifact_eligible,
            "artifact_blockers": artifact_blockers,
            "bounded": bounded,
            "context_count": len(corpus.contexts),
            "branch_count": len(corpus.rollouts),
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


def load_verified_control_branch_corpus(manifest_path: Path) -> VerifiedControlBranchCorpus:
    path = manifest_path.resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("format_id") != "conditional_return_control_branch_corpus_v1"
        or not isinstance(manifest.get("artifacts"), dict)
    ):
        raise ValueError("control branch manifest contract is invalid")
    for name in _ARTIFACT_NAMES:
        artifact_path = path.parent / name
        if not artifact_path.is_file() or sha256_file(artifact_path) != manifest["artifacts"].get(
            name
        ):
            raise ValueError(f"control branch artifact hash mismatch: {name}")
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
    for value in arrays.values():
        value.setflags(write=False)
    return VerifiedControlBranchCorpus(
        manifest_path=path,
        scientific_gate_pass=bool(manifest["scientific_gate_pass"]),
        artifact_eligible=bool(manifest["artifact_eligible"]),
        **arrays,
    )
