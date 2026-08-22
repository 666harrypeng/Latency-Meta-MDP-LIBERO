"""Immutable certification for the nominal categorical latency law."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

from latency_meta_mdp.artifacts import collect_implementation_provenance, sha256_file
from latency_meta_mdp.latency_harness import LaunchContext
from latency_meta_mdp.latency_law import CategoricalDelaySampler, load_latency_law


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_latency_law_certification(
    *,
    project_root: Path,
    config_path: Path,
    output_dir: Path,
    sample_count: int,
    sampling_seed: int,
    maximum_probability_error: float,
) -> Path:
    """Compare seeded samples against the exact grid law and publish evidence."""

    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count <= 0
    ):
        raise ValueError("sample_count must be a positive integer")
    if (
        isinstance(sampling_seed, bool)
        or not isinstance(sampling_seed, int)
        or sampling_seed < 0
    ):
        raise ValueError("sampling_seed must be a non-negative integer")
    if (
        not np.isfinite(maximum_probability_error)
        or not 0.0 < maximum_probability_error < 1.0
    ):
        raise ValueError("maximum_probability_error must be in (0, 1)")

    project = project_root.resolve()
    config = config_path.resolve()
    law = load_latency_law(config)
    sampler = CategoricalDelaySampler(law=law, seed=sampling_seed)
    samples = np.fromiter(
        (sampler() for _ in range(sample_count)),
        dtype=np.int64,
        count=sample_count,
    )
    counts = np.bincount(samples, minlength=law.bin_count + 1)[1 : law.bin_count + 1]
    empirical = counts.astype(np.float64) / sample_count
    probability_error = np.abs(empirical - law.probabilities)
    launch_fields = [field.name for field in fields(LaunchContext)]
    realized_visible = "realized_delay_ticks" in launch_fields
    passed = bool(
        int(samples.min()) >= 1
        and int(samples.max()) <= law.bin_count
        and float(probability_error.max()) <= maximum_probability_error
        and not realized_visible
    )
    provenance = collect_implementation_provenance(project)

    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"latency-law certification output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.building-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"latency-law certification staging output already exists: {staging}")
    try:
        staging.mkdir()
        _write_json(
            staging / "manifest.json",
            {
                "schema_version": 1,
                "format_id": "categorical_latency_law_certification_v1",
                "implementation_revision": provenance.revision,
                "implementation_source_sha256": provenance.source_sha256,
                "implementation_dirty": provenance.dirty,
                "eligible": passed and not provenance.dirty,
                "passed": passed,
                "law_id": law.law_id,
                "law_config_sha256": sha256_file(config),
                "shape_alpha": law.shape_alpha,
                "shape_beta": law.shape_beta,
                "latency_deadline_seconds": law.latency_deadline_seconds,
                "control_tick_seconds": law.control_tick_seconds,
                "delay_ticks": list(law.delay_ticks),
                "theoretical_probabilities": law.probabilities.tolist(),
                "empirical_probabilities": empirical.tolist(),
                "sample_count": sample_count,
                "sampling_seed": sampling_seed,
                "sample_min_ticks": int(samples.min()),
                "sample_max_ticks": int(samples.max()),
                "maximum_probability_error": float(probability_error.max()),
                "maximum_probability_error_tolerance": maximum_probability_error,
                "continuous_mode_seconds": law.continuous_mode_seconds,
                "continuous_mean_seconds": law.continuous_mean_seconds,
                "effective_mean_seconds": law.effective_mean_seconds,
                "region_probabilities": list(law.region_probabilities),
                "law_visibility": law.law_visibility,
                "launch_context_fields": launch_fields,
                "realized_delay_visible_at_launch": realized_visible,
            },
        )
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target / "manifest.json"
