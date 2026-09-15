"""Cross-level summary for Flow Belief action-buffer causality evidence."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from latency_meta_mdp.io.artifacts import collect_implementation_provenance, sha256_file


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def summarize_buffer_causality_levels(
    *,
    project_root: Path,
    evaluation_manifests: tuple[Path, ...],
    output_dir: Path,
    collect_provenance: bool = True,
) -> Path:
    """Preserve every level summary and expose weak phase/branch regimes."""

    if len(evaluation_manifests) != 3:
        raise ValueError("buffer-causality summary requires exactly three level manifests")
    level_summaries = {}
    manifests = {}
    branch_inventory = None
    input_hashes = {}
    for path_value in evaluation_manifests:
        path = path_value.resolve()
        manifest = _load_json(path)
        level = manifest.get("level")
        if (
            manifest.get("format_id") != "level_flow_belief_buffer_causality_evaluation_v1"
            or level not in (1, 2, 3)
            or level in manifests
        ):
            raise ValueError("buffer-causality level manifest identity is invalid")
        branches = tuple(manifest.get("branch_ids", ()))
        if branch_inventory is None:
            branch_inventory = branches
        elif branch_inventory != branches:
            raise ValueError("buffer-causality level branch inventories disagree")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict) or "summary.json" not in artifacts:
            raise ValueError("buffer-causality level artifact inventory is invalid")
        for name, digest in artifacts.items():
            artifact = path.parent / name
            if not artifact.is_file() or sha256_file(artifact) != digest:
                raise ValueError(f"buffer-causality level artifact hash mismatch: {name}")
        manifests[level] = manifest
        level_summaries[level] = _load_json(path.parent / "summary.json")
        input_hashes[f"L{level}"] = sha256_file(path)
    if set(manifests) != {1, 2, 3}:
        raise ValueError("buffer-causality summary requires levels 1, 2, and 3")

    negative = []
    negative_weighted = []
    false_coupling = []
    false_coupling_weighted = []
    for level in (1, 2, 3):
        summary = level_summaries[level]
        for branch in summary["branches"].values():
            false_coupling.append(float(branch["invariant_object_false_coupling_m"]))
            false_coupling_weighted.append(
                float(branch["invariant_object_false_coupling_latency_weighted_m"])
            )
        for phase, phase_rows in sorted(summary["phases"].items()):
            for branch_id, branch in sorted(phase_rows.items()):
                gain = float(branch["qpos_conditioning_gain_rad"])
                if branch_id != "expert" and gain < 0.0:
                    negative.append(
                        {
                            "branch": branch_id,
                            "gain_rad": gain,
                            "level": level,
                            "phase": phase,
                        }
                    )
                if branch_id != "expert":
                    weighted_gain = float(branch["qpos_conditioning_gain_latency_weighted_rad"])
                    if weighted_gain < 0.0:
                        negative_weighted.append(
                            {
                                "branch": branch_id,
                                "gain_rad": weighted_gain,
                                "level": level,
                                "phase": phase,
                            }
                        )
    cross_level = {}
    for branch_id in branch_inventory or ():
        rows = [level_summaries[level]["branches"][branch_id] for level in (1, 2, 3)]
        numeric_sets = [
            {key for key, value in row.items() if isinstance(value, (int, float))} for row in rows
        ]
        numeric_keys = sorted(set.intersection(*numeric_sets))
        cross_level[branch_id] = {
            key: float(np.mean([float(row[key]) for row in rows])) for key in numeric_keys
        }
    report = {
        "levels": {f"L{level}": level_summaries[level] for level in (1, 2, 3)},
        "cross_level_equal_weight_branch_means": cross_level,
        "negative_qpos_conditioning_gain_regimes": negative,
        "negative_qpos_conditioning_gain_latency_weighted_regimes": negative_weighted,
        "maximum_invariant_object_false_coupling_m": max(false_coupling),
        "maximum_invariant_object_false_coupling_latency_weighted_m": max(false_coupling_weighted),
        "decision_status": "coauthor_review_required",
    }
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"buffer-causality summary output exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f".{target.name}.building-{uuid.uuid4().hex}"
    building.mkdir()
    try:
        _write_json(building / "summary.json", report)
        provenance = (
            collect_implementation_provenance(project_root.resolve())
            if collect_provenance
            else None
        )
        blockers = []
        if provenance is not None and provenance.dirty:
            blockers.append("implementation_dirty")
        if any(manifest.get("eligible") is not True for manifest in manifests.values()):
            blockers.append("one_or_more_level_evaluations_ineligible")
        manifest = {
            "schema_version": 1,
            "format_id": "flow_belief_buffer_causality_summary_v1",
            "eligible": not blockers,
            "blockers": blockers,
            "levels": [1, 2, 3],
            "branch_ids": list(branch_inventory or ()),
            "implementation_revision": provenance.revision if provenance is not None else None,
            "implementation_source_sha256": (
                provenance.source_sha256 if provenance is not None else None
            ),
            "implementation_dirty": provenance.dirty if provenance is not None else False,
            "evaluation_manifest_sha256": input_hashes,
            "artifacts": {"summary.json": sha256_file(building / "summary.json")},
        }
        _write_json(building / "manifest.json", manifest)
        os.rename(building, target)
    except BaseException:
        import shutil

        shutil.rmtree(building, ignore_errors=True)
        raise
    return target / "manifest.json"
