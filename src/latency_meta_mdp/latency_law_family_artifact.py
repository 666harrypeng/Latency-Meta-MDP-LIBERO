"""Atomic certification for episode-level smooth latency-law assignments."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from latency_meta_mdp.artifacts import collect_implementation_provenance, sha256_file
from latency_meta_mdp.latency_law import load_latency_law
from latency_meta_mdp.latency_law_family import load_episode_latency_law_family


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


def _numeric_summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("latency-law family summary values are invalid")
    quantiles = np.quantile(array, (0.1, 0.5, 0.9))
    return {
        "minimum": float(np.min(array)),
        "p10": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p90": float(quantiles[2]),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def certify_episode_latency_law_family(
    *,
    project_root: Path,
    source_bulk_manifest: Path,
    nominal_law_path: Path,
    family_config_path: Path,
    output_dir: Path,
) -> Path:
    source_path = source_bulk_manifest.resolve()
    nominal_path = nominal_law_path.resolve()
    family_path = family_config_path.resolve()
    for path in (source_path, nominal_path, family_path):
        if not path.is_file():
            raise FileNotFoundError(f"latency-law family input does not exist: {path}")
    source = _load_json(source_path)
    admitted = source.get("admitted_episode_manifests")
    if (
        source.get("format_id") != "panda_ball_formal_corpus_v1"
        or source.get("eligible") is not True
        or source.get("levels") != [1, 2, 3]
        or not isinstance(admitted, list)
        or len(admitted) != 600
    ):
        raise ValueError("latency-law family requires the eligible formal corpus")
    nominal = load_latency_law(nominal_path)
    family = load_episode_latency_law_family(family_path)
    if (
        family.base_law_id != nominal.law_id
        or family.latency_deadline_seconds != nominal.latency_deadline_seconds
        or family.control_tick_seconds != nominal.control_tick_seconds
    ):
        raise ValueError("latency-law family and nominal law disagree")
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"latency-law family output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f"{target.name}.building-{uuid.uuid4().hex}"
    building.mkdir()
    provenance = collect_implementation_provenance(project_root)
    assignments = []
    probabilities = []
    alphas = []
    betas = []
    floors = []
    continuous_means = []
    continuous_modes = []
    effective_means = []
    region_masses = []
    source_root = source_path.parent
    try:
        for relative in sorted(admitted):
            episode_dir = (source_root / relative).parent
            metadata = _load_json(episode_dir / "metadata.json")
            law = family.sample_for_episode(
                level=int(metadata["level"]),
                episode_id=str(metadata["episode_id"]),
                scene_seed=int(metadata["scene_seed"]),
            )
            row = {
                "episode_id": metadata["episode_id"],
                "level": metadata["level"],
                "scene_seed": metadata["scene_seed"],
                "generation_seed": law.generation_seed,
                "shape_alpha": law.shape_alpha,
                "shape_beta": law.shape_beta,
                "uniform_floor": law.uniform_floor,
                "continuous_mean_seconds": law.continuous_mean_seconds,
                "continuous_mode_seconds": law.continuous_mode_seconds,
                "effective_mean_seconds": law.effective_mean_seconds,
                "region_probabilities": list(law.region_probabilities),
                "probability_sha256": law.probability_sha256,
                "probabilities": law.probabilities.tolist(),
            }
            assignments.append(row)
            probabilities.append(law.probabilities)
            alphas.append(law.shape_alpha)
            betas.append(law.shape_beta)
            floors.append(law.uniform_floor)
            continuous_means.append(law.continuous_mean_seconds)
            continuous_modes.append(law.continuous_mode_seconds)
            effective_means.append(law.effective_mean_seconds)
            region_masses.append(law.region_probabilities)
        by_scene = defaultdict(list)
        for row in assignments:
            by_scene[int(row["scene_seed"])].append(str(row["probability_sha256"]))
        mismatch_count = sum(len(rows) != 3 or len(set(rows)) != 1 for rows in by_scene.values())
        probability_array = np.stack(probabilities)
        unique_by_scene = np.stack(
            [
                probability_array[
                    next(
                        index
                        for index, row in enumerate(assignments)
                        if int(row["scene_seed"]) == scene_seed
                    )
                ]
                for scene_seed in sorted(by_scene)
            ]
        )
        nominal_l1 = np.sum(
            np.abs(unique_by_scene - nominal.probabilities[None]),
            axis=1,
        )
        region_array = np.asarray(region_masses, dtype=np.float64)
        summary = {
            "family_id": family.family_id,
            "assignment_count": len(assignments),
            "unique_scene_seed_count": len(by_scene),
            "unique_probability_count": len(
                {str(row["probability_sha256"]) for row in assignments}
            ),
            "cross_level_probability_mismatch_count": mismatch_count,
            "shape_alpha": _numeric_summary(np.asarray(alphas)),
            "shape_beta": _numeric_summary(np.asarray(betas)),
            "uniform_floor": _numeric_summary(np.asarray(floors)),
            "continuous_mean_seconds": _numeric_summary(np.asarray(continuous_means)),
            "continuous_mode_seconds": _numeric_summary(np.asarray(continuous_modes)),
            "effective_mean_seconds": _numeric_summary(np.asarray(effective_means)),
            "nominal_probability_l1_distance": _numeric_summary(nominal_l1),
            "region_probability_summaries": [
                _numeric_summary(region_array[:, index]) for index in range(4)
            ],
            "bin_probability_p10": np.quantile(unique_by_scene, 0.1, axis=0).tolist(),
            "bin_probability_median": np.quantile(unique_by_scene, 0.5, axis=0).tolist(),
            "bin_probability_p90": np.quantile(unique_by_scene, 0.9, axis=0).tolist(),
        }
        if mismatch_count or len(by_scene) != 200:
            raise RuntimeError("latency-law family cross-level assignment parity failed")
        _write_json(building / "assignments.json", assignments)
        _write_json(building / "summary.json", summary)
        np.savez(
            building / "family_arrays.npz",
            probabilities=probability_array,
            unique_scene_probabilities=unique_by_scene,
            shape_alpha=np.asarray(alphas, dtype=np.float64),
            shape_beta=np.asarray(betas, dtype=np.float64),
            uniform_floor=np.asarray(floors, dtype=np.float64),
            effective_mean_seconds=np.asarray(effective_means, dtype=np.float64),
        )
        artifacts = {
            name: sha256_file(building / name)
            for name in ("assignments.json", "summary.json", "family_arrays.npz")
        }
        _write_json(
            building / "manifest.json",
            {
                "schema_version": 1,
                "format_id": "episode_latency_law_family_artifact_v1",
                "eligible": not provenance.dirty,
                "blockers": [] if not provenance.dirty else ["implementation_worktree_dirty"],
                "implementation_revision": provenance.revision,
                "implementation_source_sha256": provenance.source_sha256,
                "implementation_dirty": provenance.dirty,
                "family_id": family.family_id,
                "assignment_count": len(assignments),
                "input_sha256": {
                    "source_bulk_manifest": sha256_file(source_path),
                    "nominal_law": sha256_file(nominal_path),
                    "family_config": sha256_file(family_path),
                },
                "artifacts": artifacts,
            },
        )
        os.rename(building, target)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return target / "manifest.json"
