"""Render one level of selected Flow Belief contexts in a real RoboSuite scene."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from latency_meta_mdp.envs.control import load_action_contract
from latency_meta_mdp.envs.task import load_task_spec, make_dynamic_grasp_lift_environment
from latency_meta_mdp.io.artifacts import sha256_file
from latency_meta_mdp.legacy.belief.flow.ghost_config import load_flow_belief_ghost_config
from latency_meta_mdp.legacy.belief.flow.ghost_context_renderer import (
    GhostContextRenderInput,
    render_flow_belief_ghost_context,
)
from latency_meta_mdp.legacy.belief.flow.ghost_environment import GhostEnvironmentAdapter
from latency_meta_mdp.legacy.belief.flow.ghost_visuals import write_rgb_video
from latency_meta_mdp.legacy.belief_data import load_belief_episode
from latency_meta_mdp.legacy.terminal_absorbing_tail import build_terminal_absorbing_tail
from latency_meta_mdp.runtime.temporal_contract import load_temporal_contract

_SAMPLE_FIELDS = frozenset(
    {
        "validation_offsets",
        "display_delay_ticks",
        "latency_probabilities",
        "normalized_samples",
        "normalized_targets",
        "physical_samples",
        "physical_targets",
        "interaction_mode",
        "absorbing",
    }
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _verify_artifacts(manifest_path: Path, manifest: dict[str, Any]) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("ghost quality level artifact inventory is invalid")
    root = manifest_path.parent.resolve()
    for relative, digest in artifacts.items():
        artifact = (root / relative).resolve()
        try:
            artifact.relative_to(root)
        except ValueError as exc:
            raise ValueError("ghost quality artifact escapes its level root") from exc
        if not artifact.is_file() or sha256_file(artifact) != digest:
            raise ValueError(f"ghost quality artifact hash mismatch: {relative}")


def _context_name(index: int, identity: dict[str, Any]) -> str:
    return (
        f"context_{index:02d}_seed_{int(identity['scene_seed']):06d}_"
        f"tick_{int(identity['source_tick']):04d}"
    )


def _load_level_inputs(
    *,
    level: int,
    quality_sample_manifest: Path,
    quality_level_manifest: Path,
    display_delay_ticks: tuple[int, ...],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    run = _load_json(quality_sample_manifest)
    manifest = _load_json(quality_level_manifest)
    if (
        not isinstance(run, dict)
        or run.get("format_id") != "flow_belief_quality_sample_run_v1"
        or run.get("eligible") is not True
        or run.get("level_manifests", {}).get(f"L{level}")
        != quality_level_manifest.relative_to(quality_sample_manifest.parent).as_posix()
    ):
        raise ValueError("ghost rendering quality run and level are not aligned")
    relative_manifest = quality_level_manifest.relative_to(
        quality_sample_manifest.parent
    ).as_posix()
    if run.get("artifacts", {}).get(relative_manifest) != sha256_file(quality_level_manifest):
        raise ValueError("ghost rendering quality level manifest hash is invalid")
    if (
        not isinstance(manifest, dict)
        or manifest.get("format_id") != "level_flow_belief_quality_samples_v1"
        or manifest.get("eligible") is not True
        or manifest.get("level") != level
        or tuple(manifest.get("display_delay_ticks", [])) != display_delay_ticks
        or manifest.get("sample_count") != 32
    ):
        raise ValueError("ghost rendering quality level manifest is invalid")
    _verify_artifacts(quality_level_manifest, manifest)
    level_root = quality_level_manifest.parent
    selections = _load_json(level_root / "selection.json")
    if not isinstance(selections, list) or len(selections) != manifest.get("context_count"):
        raise ValueError("ghost rendering quality selections are invalid")
    with np.load(level_root / "samples.npz", allow_pickle=False) as source:
        if set(source.files) != _SAMPLE_FIELDS:
            raise ValueError("ghost rendering quality sample fields are invalid")
        arrays = {name: np.array(source[name], copy=True) for name in source.files}
    context_count = len(selections)
    expected_shapes = {
        "validation_offsets": (context_count,),
        "display_delay_ticks": (5,),
        "normalized_samples": (context_count, 5, 32, 22),
        "physical_samples": (context_count, 5, 32, 22),
        "physical_targets": (context_count, 5, 22),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"ghost rendering quality {name} shape is invalid")
    if not np.array_equal(arrays["display_delay_ticks"], display_delay_ticks):
        raise ValueError("ghost rendering delays do not match the ghost config")
    for index, selection in enumerate(selections):
        identity = selection.get("identity") if isinstance(selection, dict) else None
        if (
            not isinstance(identity, dict)
            or identity.get("level") != level
            or identity.get("validation_offset") != int(arrays["validation_offsets"][index])
        ):
            raise ValueError("ghost selection identity and sample rows are misaligned")
    return selections, arrays, manifest


def render_flow_belief_ghost_level(
    *,
    level: int,
    output_dir: Path,
    project_root: Path,
    source_bulk_manifest: Path,
    quality_sample_manifest: Path,
    quality_level_manifest: Path,
    task_config_path: Path,
    control_config_path: Path,
    temporal_config_path: Path,
    ghost_config_path: Path,
) -> Path:
    started = time.perf_counter()
    if level not in (1, 2, 3) or not project_root.is_dir():
        raise ValueError("ghost level or project root is invalid")
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"Flow ghost level already exists: {target}")
    config = load_flow_belief_ghost_config(ghost_config_path)
    selections, samples, sample_manifest = _load_level_inputs(
        level=level,
        quality_sample_manifest=quality_sample_manifest,
        quality_level_manifest=quality_level_manifest,
        display_delay_ticks=config.display_delay_ticks,
    )
    source = _load_json(source_bulk_manifest)
    if (
        not isinstance(source, dict)
        or source.get("format_id") != "panda_ball_formal_corpus_v1"
        or source.get("eligible") is not True
        or level not in source.get("levels", [])
    ):
        raise ValueError("ghost rendering source corpus is invalid")
    sample_run = _load_json(quality_sample_manifest)
    if sample_run.get("input_sha256", {}).get("source_bulk_manifest") != sha256_file(
        source_bulk_manifest
    ):
        raise ValueError("ghost rendering source corpus does not match quality samples")
    task_spec = load_task_spec(task_config_path)
    action_contract = load_action_contract(control_config_path)
    temporal = load_temporal_contract(temporal_config_path)
    target.mkdir(parents=True)
    contexts_root = target / "contexts"
    contexts_root.mkdir()
    context_records = []
    video_panels = []
    invalid_total = 0
    source_root = source_bulk_manifest.parent
    try:
        for context_index, selection in enumerate(selections):
            identity = selection["identity"]
            scene_seed = int(identity["scene_seed"])
            episode = load_belief_episode(source_root / f"episodes/L{level}/seed_{scene_seed:06d}")
            tail = build_terminal_absorbing_tail(
                episode=episode,
                temporal_contract=temporal,
                action_contract=action_contract,
            )
            env = make_dynamic_grasp_lift_environment(
                spec=task_spec,
                seed=scene_seed,
                offscreen=True,
                controller_config=action_contract.to_robosuite_config(),
            )
            try:
                result = render_flow_belief_ghost_context(
                    context=GhostContextRenderInput(
                        level=level,
                        episode_id=identity["episode_id"],
                        scene_seed=scene_seed,
                        validation_offset=identity["validation_offset"],
                        source_tick=identity["source_tick"],
                        source_phase=selection["source_phase"],
                        roles=tuple(selection["roles"]),
                        display_tag=f"role {', '.join(selection['roles'])}",
                        delay_ticks=samples["display_delay_ticks"],
                        normalized_samples=samples["normalized_samples"][context_index],
                        physical_samples=samples["physical_samples"][context_index],
                        physical_targets=samples["physical_targets"][context_index],
                        absorbing=samples["absorbing"][context_index],
                    ),
                    output_dir=contexts_root / _context_name(context_index, identity),
                    episode=episode,
                    tail=tail,
                    adapter=GhostEnvironmentAdapter(
                        env=env,
                        camera_name=config.camera_name,
                        width=config.width,
                        height=config.height,
                    ),
                    config=config,
                )
            finally:
                env.close()
            invalid_total += result.invalid_sample_count
            context_records.append(
                {
                    "identity": identity,
                    "roles": selection["roles"],
                    "directory": (contexts_root / _context_name(context_index, identity))
                    .relative_to(target)
                    .as_posix(),
                }
            )
            video_panels.append((scene_seed, result.source_tick, result.panel))
        ordered_frames = np.stack(
            [row[2] for row in sorted(video_panels, key=lambda value: value[:2])]
        )
        write_rgb_video(
            frames=ordered_frames,
            output_path=target / "selected_cases.mp4",
            fps=1,
        )
        artifacts = {
            path.relative_to(target).as_posix(): sha256_file(path)
            for path in sorted(target.rglob("*"))
            if path.is_file()
        }
        sample_count = len(selections) * len(config.display_delay_ticks) * 32
        manifest = {
            "schema_version": 1,
            "format_id": "level_flow_belief_ghost_v1",
            "eligible": True,
            "level": level,
            "context_count": len(selections),
            "sample_count": sample_count,
            "invalid_sample_count": invalid_total,
            "invalid_sample_fraction": invalid_total / sample_count,
            "display_delay_ticks": list(config.display_delay_ticks),
            "review_video_kind": "selected_case_sequence",
            "contexts": context_records,
            "wall_seconds": time.perf_counter() - started,
            "input_sha256": {
                "source_bulk_manifest": sha256_file(source_bulk_manifest),
                "quality_sample_manifest": sha256_file(quality_sample_manifest),
                "quality_level_manifest": sha256_file(quality_level_manifest),
                "quality_samples": sample_manifest["artifacts"]["samples.npz"],
                "task_config": sha256_file(task_config_path),
                "control_config": sha256_file(control_config_path),
                "temporal_config": sha256_file(temporal_config_path),
                "ghost_config": sha256_file(ghost_config_path),
            },
            "artifacts": artifacts,
        }
        _write_json(target / "manifest.json", manifest)
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return target / "manifest.json"
