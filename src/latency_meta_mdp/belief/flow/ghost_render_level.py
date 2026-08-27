"""Render one level of selected Flow Belief contexts in a real RoboSuite scene."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from latency_meta_mdp.artifacts import sha256_file
from latency_meta_mdp.belief.flow.ghost_config import (
    FlowBeliefGhostConfig,
    load_flow_belief_ghost_config,
)
from latency_meta_mdp.belief.flow.ghost_environment import GhostEnvironmentAdapter
from latency_meta_mdp.belief.flow.ghost_state import (
    reconstruct_return_state,
    select_sample_medoid,
    valid_sample_mask,
)
from latency_meta_mdp.belief.flow.ghost_visuals import (
    assemble_context_panel,
    compose_agentview_ghost,
    render_joint_band_plot,
    render_trajectory_plot,
    write_rgb_video,
)
from latency_meta_mdp.belief_data import load_belief_episode
from latency_meta_mdp.control import load_action_contract
from latency_meta_mdp.return_belief_geometry import build_absorbing_return_state_stream
from latency_meta_mdp.task import load_task_spec, make_dynamic_grasp_lift_environment
from latency_meta_mdp.temporal_contract import load_temporal_contract
from latency_meta_mdp.terminal_absorbing_tail import build_terminal_absorbing_tail

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


def _save_rgb(path: Path, value: np.ndarray) -> None:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise ValueError("ghost PNG must be a uint8 RGB image")
    Image.fromarray(array).save(path)


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


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.count_nonzero(np.asarray(left) | np.asarray(right))
    if union == 0:
        raise ValueError("ghost mask IoU requires a non-empty union")
    return float(np.count_nonzero(np.asarray(left) & np.asarray(right)) / union)


def _mask_centroid(mask: np.ndarray) -> np.ndarray:
    rows, columns = np.nonzero(np.asarray(mask))
    if len(rows) == 0:
        raise ValueError("ghost ball mask must be non-empty")
    return np.asarray([columns.mean(), rows.mean()], dtype=np.float64)


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
    config: FlowBeliefGhostConfig,
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
        or tuple(manifest.get("display_delay_ticks", [])) != config.display_delay_ticks
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
    if not np.array_equal(arrays["display_delay_ticks"], config.display_delay_ticks):
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
        config=config,
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
    context_records: list[dict[str, Any]] = []
    video_panels: list[tuple[int, int, np.ndarray]] = []
    invalid_total = 0
    source_root = source_bulk_manifest.parent
    delays = np.asarray(config.display_delay_ticks, dtype=np.int64)
    try:
        for context_index, selection in enumerate(selections):
            identity = selection["identity"]
            scene_seed = int(identity["scene_seed"])
            source_tick = int(identity["source_tick"])
            episode_dir = source_root / f"episodes/L{level}/seed_{scene_seed:06d}"
            episode = load_belief_episode(episode_dir)
            if (
                episode.level != level
                or episode.scene_seed != scene_seed
                or episode.episode_id != identity["episode_id"]
            ):
                raise ValueError("ghost selection does not match its source episode")
            tail = build_terminal_absorbing_tail(
                episode=episode,
                temporal_contract=temporal,
                action_contract=action_contract,
            )
            return_states = build_absorbing_return_state_stream(tail)
            gripper_qpos = tail.extend_boundary_array(episode.deployment.gripper_qpos)
            gripper_qvel = tail.extend_boundary_array(episode.deployment.gripper_qvel)
            object_pose = tail.extend_boundary_array(episode.supervision.object_pose)
            object_velocity = tail.extend_boundary_array(episode.supervision.object_velocity)
            agentview = tail.extend_boundary_array(episode.deployment.agentview_rgb)
            target_ticks = source_tick + delays
            if source_tick >= episode.boundary_count or np.any(
                target_ticks >= tail.extended_boundary_count
            ):
                raise ValueError("ghost source or target tick lies outside the episode view")
            expected_targets = return_states[target_ticks]
            if not np.allclose(
                samples["physical_targets"][context_index],
                expected_targets,
                atol=1e-6,
                rtol=1e-6,
            ):
                raise ValueError("ghost quality target does not match the source future state")
            env = make_dynamic_grasp_lift_environment(
                spec=task_spec,
                seed=scene_seed,
                offscreen=True,
                controller_config=action_contract.to_robosuite_config(),
            )
            try:
                adapter = GhostEnvironmentAdapter(
                    env=env,
                    camera_name=config.camera_name,
                    width=config.width,
                    height=config.height,
                )
                physical = samples["physical_samples"][context_index]
                normalized = samples["normalized_samples"][context_index]
                validity = np.stack(
                    [
                        valid_sample_mask(
                            physical_samples=row,
                            joint_ranges=adapter.joint_ranges,
                            gripper_width_range=adapter.gripper_width_range,
                            object_position_bounds=adapter.object_position_bounds,
                        )
                        for row in physical
                    ]
                )
                invalid_total += int(np.count_nonzero(~validity))
                if np.any(np.count_nonzero(validity, axis=1) == 0):
                    raise ValueError("ghost rendering found a delay without a valid sample")
                medoid_indices = np.asarray(
                    [
                        select_sample_medoid(normalized[row], validity[row])
                        for row in range(len(delays))
                    ],
                    dtype=np.int64,
                )
                ground_truth_images = []
                overlays = []
                eef_samples = np.empty((5, 32, 3), dtype=np.float64)
                eef_targets = np.empty((5, 3), dtype=np.float64)
                delay_records = []
                context_dir = contexts_root / _context_name(context_index, identity)
                overlay_dir = context_dir / "overlays"
                ground_truth_dir = context_dir / "ground_truth"
                overlay_dir.mkdir(parents=True)
                ground_truth_dir.mkdir()
                for delay_index, (delay, target_tick) in enumerate(
                    zip(delays, target_ticks, strict=True)
                ):
                    nuisance = {
                        "target_gripper_qpos": gripper_qpos[target_tick],
                        "target_gripper_qvel": gripper_qvel[target_tick],
                        "target_object_pose": object_pose[target_tick],
                        "target_object_velocity": object_velocity[target_tick],
                    }
                    gt_state = reconstruct_return_state(
                        predicted_state=samples["physical_targets"][context_index, delay_index],
                        **nuisance,
                    )
                    sim_time = float(tail.boundary_time_us[target_tick]) / 1_000_000.0
                    gt_render = adapter.render_state(
                        state=gt_state,
                        sim_time_seconds=sim_time,
                    )
                    repeated = adapter.render_state(
                        state=gt_state,
                        sim_time_seconds=sim_time,
                    )
                    rgb_mae = float(
                        np.mean(
                            np.abs(
                                gt_render.rgb.astype(np.float32)
                                - agentview[target_tick].astype(np.float32)
                            )
                        )
                    )
                    robot_iou = _mask_iou(gt_render.robot_mask, repeated.robot_mask)
                    ball_centroid_error = float(
                        np.linalg.norm(
                            _mask_centroid(gt_render.ball_mask) - _mask_centroid(repeated.ball_mask)
                        )
                    )
                    if (
                        rgb_mae > config.ground_truth_rgb_mae_max
                        or robot_iou < config.robot_mask_iou_min
                        or ball_centroid_error > config.ball_centroid_error_px_max
                    ):
                        raise ValueError("ghost reconstructed ground truth failed render parity")
                    medoid_index = int(medoid_indices[delay_index])
                    predicted_state = reconstruct_return_state(
                        predicted_state=physical[delay_index, medoid_index],
                        **nuisance,
                    )
                    prediction_render = adapter.render_state(
                        state=predicted_state,
                        sim_time_seconds=sim_time,
                    )
                    overlay = compose_agentview_ghost(
                        background_rgb=gt_render.rgb,
                        ground_truth_robot_mask=gt_render.robot_mask,
                        ground_truth_ball_mask=gt_render.ball_mask,
                        prediction_robot_mask=prediction_render.robot_mask,
                        prediction_ball_mask=prediction_render.ball_mask,
                        config=config,
                    )
                    _save_rgb(
                        ground_truth_dir / f"delay_{int(delay):02d}.png",
                        gt_render.rgb,
                    )
                    _save_rgb(
                        overlay_dir / f"delay_{int(delay):02d}.png",
                        overlay,
                    )
                    ground_truth_images.append(gt_render.rgb)
                    overlays.append(overlay)
                    eef_targets[delay_index] = gt_render.eef_position
                    for sample_index in range(32):
                        sample_state = reconstruct_return_state(
                            predicted_state=physical[delay_index, sample_index],
                            **nuisance,
                        )
                        eef_samples[delay_index, sample_index] = adapter.forward_state(
                            state=sample_state,
                            sim_time_seconds=sim_time,
                        )
                    delay_records.append(
                        {
                            "delay_tick": int(delay),
                            "target_tick": int(target_tick),
                            "absorbing": bool(samples["absorbing"][context_index, delay_index]),
                            "valid_sample_count": int(np.count_nonzero(validity[delay_index])),
                            "invalid_sample_count": int(np.count_nonzero(~validity[delay_index])),
                            "medoid_sample_index": medoid_index,
                            "ground_truth_rgb_mae": rgb_mae,
                            "deterministic_robot_mask_iou": robot_iou,
                            "deterministic_ball_centroid_error_px": ball_centroid_error,
                        }
                    )
            finally:
                env.close()
            trajectory_plot = render_trajectory_plot(
                delay_ticks=delays,
                object_samples=physical[:, :, 16:19],
                object_targets=samples["physical_targets"][context_index, :, 16:19],
                eef_samples=eef_samples,
                eef_targets=eef_targets,
                config=config,
            )
            joint_plot = render_joint_band_plot(
                delay_ticks=delays,
                joint_samples=physical[:, :, :7],
                joint_targets=samples["physical_targets"][context_index, :, :7],
                config=config,
            )
            title = (
                f"L{level} seed {scene_seed} tick {source_tick} | "
                f"roles: {', '.join(selection['roles'])}"
            )
            panel = assemble_context_panel(
                current_rgb=agentview[source_tick],
                ground_truth_rgb=np.stack(ground_truth_images),
                ghost_overlays=np.stack(overlays),
                trajectory_plot=trajectory_plot,
                joint_plot=joint_plot,
                delay_ticks=delays,
                title=title,
            )
            _save_rgb(context_dir / "panel.png", panel)
            np.savez(
                context_dir / "geometry.npz",
                delay_ticks=delays,
                target_ticks=target_ticks,
                valid_sample_mask=validity,
                medoid_sample_indices=medoid_indices,
                object_samples=physical[:, :, 16:19],
                object_targets=samples["physical_targets"][context_index, :, 16:19],
                eef_samples=eef_samples,
                eef_targets=eef_targets,
                joint_samples=physical[:, :, :7],
                joint_targets=samples["physical_targets"][context_index, :, :7],
            )
            context_manifest = {
                "schema_version": 1,
                "format_id": "flow_belief_ghost_context_v1",
                "identity": identity,
                "roles": selection["roles"],
                "source_phase": selection["source_phase"],
                "delays": delay_records,
            }
            _write_json(context_dir / "metrics.json", context_manifest)
            context_records.append(
                {
                    "identity": identity,
                    "roles": selection["roles"],
                    "directory": context_dir.relative_to(target).as_posix(),
                }
            )
            video_panels.append((scene_seed, source_tick, panel))
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
        sample_count = len(selections) * len(delays) * 32
        manifest = {
            "schema_version": 1,
            "format_id": "level_flow_belief_ghost_v1",
            "eligible": True,
            "level": level,
            "context_count": len(selections),
            "sample_count": sample_count,
            "invalid_sample_count": invalid_total,
            "invalid_sample_fraction": invalid_total / sample_count,
            "display_delay_ticks": delays.tolist(),
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
