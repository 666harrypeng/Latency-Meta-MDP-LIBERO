"""Atomic certification for the derived multi-law Flow Belief data view."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from latency_meta_mdp.artifacts import collect_implementation_provenance, sha256_file
from latency_meta_mdp.belief.common.feature_corpus import (
    load_level_feature_belief_corpus,
)
from latency_meta_mdp.belief.flow.multilaw_config import (
    load_multilaw_flow_training_config,
)
from latency_meta_mdp.belief.flow.training_data import build_flow_belief_normalization
from latency_meta_mdp.vision_encoder import load_vision_encoder_spec
from latency_meta_mdp.vision_probe_data import ProbeSplit


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _probability_sha256(probability: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(probability, dtype="<f8").tobytes(order="C")).hexdigest()


def certify_multilaw_belief_data(
    *,
    project_root: Path,
    source_bulk_manifest: Path,
    cache_run_manifest: Path,
    vision_config_path: Path,
    temporal_config_path: Path,
    nominal_law_path: Path,
    family_config_path: Path,
    split_config_path: Path,
    multilaw_config_path: Path,
    output_dir: Path,
) -> Path:
    inputs = {
        "source_bulk_manifest": source_bulk_manifest.resolve(),
        "cache_run_manifest": cache_run_manifest.resolve(),
        "vision_config": vision_config_path.resolve(),
        "temporal_config": temporal_config_path.resolve(),
        "nominal_law": nominal_law_path.resolve(),
        "family_config": family_config_path.resolve(),
        "split_config": split_config_path.resolve(),
        "multilaw_config": multilaw_config_path.resolve(),
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(f"multi-law Belief data input does not exist: {path}")
    spec = load_vision_encoder_spec(inputs["vision_config"])
    multilaw = load_multilaw_flow_training_config(inputs["multilaw_config"])
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"multi-law Belief data output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f"{target.name}.building-{uuid.uuid4().hex}"
    building.mkdir()
    provenance = collect_implementation_provenance(project_root)
    level_manifests = {}
    law_hashes_by_scene: dict[int, dict[int, str]] = defaultdict(dict)
    try:
        for level in (1, 2, 3):
            corpus = load_level_feature_belief_corpus(
                project_root=project_root,
                source_bulk_manifest=inputs["source_bulk_manifest"],
                cache_run_manifest=inputs["cache_run_manifest"],
                expected_spec=spec,
                temporal_config_path=inputs["temporal_config"],
                latency_law_path=inputs["nominal_law"],
                latency_law_family_path=inputs["family_config"],
                split_plan_path=inputs["split_config"],
                level=level,
            )
            if corpus.latency_law_family_id != multilaw.latency_law_family_id:
                raise ValueError("multi-law Belief corpus and training config disagree")
            normalization = build_flow_belief_normalization(corpus)
            episode_rows = []
            minimum_query_probability = 1.0
            for record in corpus.records:
                probability = np.asarray(record.latency_probabilities, dtype=np.float64)
                query_probability = (
                    1.0 - multilaw.tail_query_uniform_mix
                ) * probability + multilaw.tail_query_uniform_mix / len(probability)
                law_hash = _probability_sha256(probability)
                law_hashes_by_scene[record.scene_seed][level] = law_hash
                minimum_query_probability = min(
                    minimum_query_probability,
                    float(np.min(query_probability)),
                )
                episode_rows.append(
                    {
                        "episode_id": record.episode_id,
                        "level": level,
                        "scene_seed": record.scene_seed,
                        "split": record.split.value,
                        "latency_probability_sha256": law_hash,
                        "latency_probabilities": probability.tolist(),
                        "training_query_probabilities": query_probability.tolist(),
                    }
                )
            materialized = corpus.materialize(ProbeSplit.TRAIN, 0)
            if hasattr(materialized, "realized_delay_tick"):
                raise RuntimeError("multi-law Belief data exposes a realized delay")
            level_dir = building / f"L{level}"
            level_dir.mkdir()
            _write_json(level_dir / "episode_laws.json", episode_rows)
            np.savez(
                level_dir / "normalization.npz",
                proprio_mean=normalization.proprio_mean,
                proprio_std=normalization.proprio_std,
                action_mean=normalization.action_mean,
                action_std=normalization.action_std,
                target_mean=normalization.target_mean,
                target_std=normalization.target_std,
            )
            summary = {
                "level": level,
                "latency_law_family_id": corpus.latency_law_family_id,
                "tail_query_uniform_mix": multilaw.tail_query_uniform_mix,
                "episode_counts": {
                    split.value: corpus.episode_counts[split] for split in ProbeSplit
                },
                "sample_counts": {split.value: corpus.sample_counts[split] for split in ProbeSplit},
                "unique_episode_law_count": len(
                    {row["latency_probability_sha256"] for row in episode_rows}
                ),
                "minimum_training_query_probability": minimum_query_probability,
                "realized_delay_exposed": False,
                "belief_token_input_contract": {
                    "vision_history": [6, 2, 196, 384],
                    "proprio_history": [6, 16],
                    "remaining_actions": [25, 7],
                    "latency_probabilities": [20],
                    "future_state_targets": [20, 22],
                },
            }
            _write_json(level_dir / "summary.json", summary)
            artifacts = {
                name: sha256_file(level_dir / name)
                for name in ("episode_laws.json", "normalization.npz", "summary.json")
            }
            _write_json(
                level_dir / "manifest.json",
                {
                    "schema_version": 1,
                    "format_id": "level_multilaw_flow_belief_data_v1",
                    "level": level,
                    "artifacts": artifacts,
                },
            )
            level_manifests[f"L{level}"] = f"L{level}/manifest.json"
        mismatch_count = sum(
            set(level_hashes) != {1, 2, 3} or len(set(level_hashes.values())) != 1
            for level_hashes in law_hashes_by_scene.values()
        )
        if len(law_hashes_by_scene) != 200 or mismatch_count:
            raise RuntimeError("multi-law Belief data cross-level law parity failed")
        artifacts = {
            path.relative_to(building).as_posix(): sha256_file(path)
            for path in sorted(building.rglob("*"))
            if path.is_file()
        }
        _write_json(
            building / "manifest.json",
            {
                "schema_version": 1,
                "format_id": "multilaw_flow_belief_data_artifact_v1",
                "eligible": not provenance.dirty,
                "blockers": [] if not provenance.dirty else ["implementation_worktree_dirty"],
                "implementation_revision": provenance.revision,
                "implementation_source_sha256": provenance.source_sha256,
                "implementation_dirty": provenance.dirty,
                "levels": [1, 2, 3],
                "latency_law_family_id": multilaw.latency_law_family_id,
                "cross_level_probability_mismatch_count": mismatch_count,
                "level_manifests": level_manifests,
                "input_sha256": {name: sha256_file(path) for name, path in inputs.items()},
                "artifacts": artifacts,
            },
        )
        os.rename(building, target)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return target / "manifest.json"
