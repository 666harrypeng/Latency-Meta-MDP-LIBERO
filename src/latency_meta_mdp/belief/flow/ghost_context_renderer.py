"""Reusable rendering of one Flow Belief launch context."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from latency_meta_mdp.belief.flow.ghost_config import FlowBeliefGhostConfig
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
    render_state_cloud_plot,
)
from latency_meta_mdp.belief.flow.rolling_visuals import (
    RollingJointLimits,
    RollingPlotLimits,
)
from latency_meta_mdp.belief_data import BeliefEpisodeView
from latency_meta_mdp.return_belief_geometry import build_absorbing_return_state_stream
from latency_meta_mdp.terminal_absorbing_tail import TerminalAbsorbingTailView


def _readonly(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class GhostContextRenderInput:
    level: int
    episode_id: str
    scene_seed: int
    validation_offset: int
    source_tick: int
    source_phase: str
    roles: tuple[str, ...]
    display_tag: str
    delay_ticks: np.ndarray
    normalized_samples: np.ndarray
    physical_samples: np.ndarray
    physical_targets: np.ndarray
    absorbing: np.ndarray

    def __post_init__(self) -> None:
        if (
            self.level not in (1, 2, 3)
            or not self.episode_id
            or isinstance(self.scene_seed, bool)
            or not isinstance(self.scene_seed, int)
            or self.scene_seed < 0
            or isinstance(self.validation_offset, bool)
            or not isinstance(self.validation_offset, int)
            or self.validation_offset < 0
            or isinstance(self.source_tick, bool)
            or not isinstance(self.source_tick, int)
            or self.source_tick < 0
            or self.source_phase not in ("pregrasp", "approach", "close", "lift")
            or not self.roles
            or any(not role.strip() for role in self.roles)
            or not self.display_tag.strip()
        ):
            raise ValueError("ghost context identity is invalid")
        expected = {
            "delay_ticks": (5,),
            "normalized_samples": (5, 32, 22),
            "physical_samples": (5, 32, 22),
            "physical_targets": (5, 22),
            "absorbing": (5,),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"ghost context {name} has invalid shape")
        if not np.array_equal(self.delay_ticks, np.asarray([1, 5, 10, 15, 20])):
            raise ValueError("ghost context delays are invalid")
        if any(
            not np.all(np.isfinite(getattr(self, name)))
            for name in ("normalized_samples", "physical_samples", "physical_targets")
        ):
            raise ValueError("ghost context arrays must be finite")
        for name, dtype in (
            ("delay_ticks", np.int64),
            ("normalized_samples", np.float32),
            ("physical_samples", np.float32),
            ("physical_targets", np.float32),
            ("absorbing", np.bool_),
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype=dtype))
        object.__setattr__(self, "roles", tuple(self.roles))

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "level": self.level,
            "scene_seed": self.scene_seed,
            "source_tick": self.source_tick,
            "validation_offset": self.validation_offset,
        }


@dataclass(frozen=True)
class GhostContextGeometry:
    target_ticks: np.ndarray
    valid_sample_mask: np.ndarray
    medoid_sample_indices: np.ndarray
    eef_samples: np.ndarray
    eef_targets: np.ndarray

    def __post_init__(self) -> None:
        expected = {
            "target_ticks": (5,),
            "valid_sample_mask": (5, 32),
            "medoid_sample_indices": (5,),
            "eef_samples": (5, 32, 3),
            "eef_targets": (5, 3),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"ghost context geometry {name} has invalid shape")
        if not np.all(np.isfinite(self.eef_samples)) or not np.all(np.isfinite(self.eef_targets)):
            raise ValueError("ghost context EEF geometry must be finite")
        if np.any(np.count_nonzero(self.valid_sample_mask, axis=1) == 0):
            raise ValueError("ghost context geometry requires a valid sample per delay")
        for name, dtype in (
            ("target_ticks", np.int64),
            ("valid_sample_mask", np.bool_),
            ("medoid_sample_indices", np.int64),
            ("eef_samples", np.float64),
            ("eef_targets", np.float64),
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype=dtype))


@dataclass(frozen=True)
class GhostDelayMetrics:
    delay_tick: int
    target_tick: int
    absorbing: bool
    valid_sample_count: int
    invalid_sample_count: int
    medoid_sample_index: int
    medoid_object_position_error_mm: float
    medoid_eef_position_error_mm: float
    medoid_joint_rmse_mrad: float
    ground_truth_rgb_mae: float
    deterministic_robot_mask_iou: float
    deterministic_ball_centroid_error_px: float


@dataclass(frozen=True)
class GhostContextRenderResult:
    source_tick: int
    panel: np.ndarray
    delay_metrics: tuple[GhostDelayMetrics, ...]
    geometry: GhostContextGeometry
    invalid_sample_count: int

    def __post_init__(self) -> None:
        panel = np.asarray(self.panel)
        if panel.shape != (1200, 1920, 3) or panel.dtype != np.uint8:
            raise ValueError("ghost context result panel is invalid")
        if len(self.delay_metrics) != 5:
            raise ValueError("ghost context result delay metrics are invalid")
        object.__setattr__(self, "panel", _readonly(panel, dtype=np.uint8))
        object.__setattr__(self, "delay_metrics", tuple(self.delay_metrics))


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


def _nuisance_streams(
    *,
    episode: BeliefEpisodeView,
    tail: TerminalAbsorbingTailView,
) -> dict[str, np.ndarray]:
    return {
        "gripper_qpos": tail.extend_boundary_array(episode.deployment.gripper_qpos),
        "gripper_qvel": tail.extend_boundary_array(episode.deployment.gripper_qvel),
        "object_pose": tail.extend_boundary_array(episode.supervision.object_pose),
        "object_velocity": tail.extend_boundary_array(episode.supervision.object_velocity),
        "agentview": tail.extend_boundary_array(episode.deployment.agentview_rgb),
        "eef_position": tail.extend_boundary_array(episode.deployment.eef_position_world),
    }


def _nuisance(values: dict[str, np.ndarray], target_tick: int) -> dict[str, np.ndarray]:
    return {
        "target_gripper_qpos": values["gripper_qpos"][target_tick],
        "target_gripper_qvel": values["gripper_qvel"][target_tick],
        "target_object_pose": values["object_pose"][target_tick],
        "target_object_velocity": values["object_velocity"][target_tick],
    }


def prepare_flow_belief_ghost_context(
    *,
    context: GhostContextRenderInput,
    episode: BeliefEpisodeView,
    tail: TerminalAbsorbingTailView,
    adapter: GhostEnvironmentAdapter,
) -> GhostContextGeometry:
    if (
        episode.level != context.level
        or episode.scene_seed != context.scene_seed
        or episode.episode_id != context.episode_id
    ):
        raise ValueError("ghost context does not match its source episode")
    target_ticks = context.source_tick + context.delay_ticks
    if context.source_tick >= episode.boundary_count or np.any(
        target_ticks >= tail.extended_boundary_count
    ):
        raise ValueError("ghost context source or target lies outside the episode")
    expected_targets = build_absorbing_return_state_stream(tail)[target_ticks]
    if not np.allclose(context.physical_targets, expected_targets, atol=1e-6, rtol=1e-6):
        raise ValueError("ghost context target does not match the source future state")
    validity = np.stack(
        [
            valid_sample_mask(
                physical_samples=row,
                joint_ranges=adapter.joint_ranges,
                gripper_width_range=adapter.gripper_width_range,
                object_position_bounds=adapter.object_position_bounds,
            )
            for row in context.physical_samples
        ]
    )
    if np.any(np.count_nonzero(validity, axis=1) == 0):
        raise ValueError("ghost context has a delay without a valid sample")
    medoids = np.asarray(
        [
            select_sample_medoid(context.normalized_samples[row], validity[row])
            for row in range(len(context.delay_ticks))
        ],
        dtype=np.int64,
    )
    values = _nuisance_streams(episode=episode, tail=tail)
    eef_samples = np.empty((5, 32, 3), dtype=np.float64)
    for delay_index, target_tick in enumerate(target_ticks):
        nuisance = _nuisance(values, int(target_tick))
        sim_time = float(tail.boundary_time_us[target_tick]) / 1_000_000.0
        for sample_index in range(32):
            state = reconstruct_return_state(
                predicted_state=context.physical_samples[delay_index, sample_index],
                **nuisance,
            )
            eef_samples[delay_index, sample_index] = adapter.forward_state(
                state=state,
                sim_time_seconds=sim_time,
            )
    return GhostContextGeometry(
        target_ticks=target_ticks,
        valid_sample_mask=validity,
        medoid_sample_indices=medoids,
        eef_samples=eef_samples,
        eef_targets=values["eef_position"][target_ticks],
    )


def render_flow_belief_ghost_context(
    *,
    context: GhostContextRenderInput,
    output_dir: Path,
    episode: BeliefEpisodeView,
    tail: TerminalAbsorbingTailView,
    adapter: GhostEnvironmentAdapter,
    config: FlowBeliefGhostConfig,
    geometry: GhostContextGeometry | None = None,
    object_limits: RollingPlotLimits | None = None,
    eef_limits: RollingPlotLimits | None = None,
    joint_limits: RollingJointLimits | None = None,
) -> GhostContextRenderResult:
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"ghost context output already exists: {target}")
    if not isinstance(adapter, GhostEnvironmentAdapter) or not isinstance(
        config, FlowBeliefGhostConfig
    ):
        raise TypeError("ghost context rendering requires typed adapter and config")
    prepared = geometry or prepare_flow_belief_ghost_context(
        context=context,
        episode=episode,
        tail=tail,
        adapter=adapter,
    )
    values = _nuisance_streams(episode=episode, tail=tail)
    target.mkdir(parents=True)
    overlay_dir = target / "overlays"
    ground_truth_dir = target / "ground_truth"
    overlay_dir.mkdir()
    ground_truth_dir.mkdir()
    overlays = []
    delay_metrics = []
    try:
        for delay_index, (delay, target_tick) in enumerate(
            zip(context.delay_ticks, prepared.target_ticks, strict=True)
        ):
            nuisance = _nuisance(values, int(target_tick))
            gt_state = reconstruct_return_state(
                predicted_state=context.physical_targets[delay_index],
                **nuisance,
            )
            sim_time = float(tail.boundary_time_us[target_tick]) / 1_000_000.0
            gt_render = adapter.render_state(state=gt_state, sim_time_seconds=sim_time)
            repeated = adapter.render_state(state=gt_state, sim_time_seconds=sim_time)
            rgb_mae = float(
                np.mean(
                    np.abs(
                        gt_render.rgb.astype(np.float32)
                        - values["agentview"][target_tick].astype(np.float32)
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
            medoid_index = int(prepared.medoid_sample_indices[delay_index])
            medoid = context.physical_samples[delay_index, medoid_index]
            prediction_state = reconstruct_return_state(
                predicted_state=medoid,
                **nuisance,
            )
            prediction_render = adapter.render_state(
                state=prediction_state,
                sim_time_seconds=sim_time,
            )
            physical_target = context.physical_targets[delay_index]
            overlay = compose_agentview_ghost(
                background_rgb=gt_render.rgb,
                ground_truth_robot_mask=gt_render.robot_mask,
                ground_truth_ball_mask=gt_render.ball_mask,
                prediction_robot_mask=prediction_render.robot_mask,
                prediction_ball_mask=prediction_render.ball_mask,
                config=config,
            )
            _save_rgb(ground_truth_dir / f"delay_{int(delay):02d}.png", gt_render.rgb)
            _save_rgb(overlay_dir / f"delay_{int(delay):02d}.png", overlay)
            overlays.append(overlay)
            delay_metrics.append(
                GhostDelayMetrics(
                    delay_tick=int(delay),
                    target_tick=int(target_tick),
                    absorbing=bool(context.absorbing[delay_index]),
                    valid_sample_count=int(
                        np.count_nonzero(prepared.valid_sample_mask[delay_index])
                    ),
                    invalid_sample_count=int(
                        np.count_nonzero(~prepared.valid_sample_mask[delay_index])
                    ),
                    medoid_sample_index=medoid_index,
                    medoid_object_position_error_mm=float(
                        np.linalg.norm(medoid[16:19] - physical_target[16:19]) * 1_000.0
                    ),
                    medoid_eef_position_error_mm=float(
                        np.linalg.norm(prediction_render.eef_position - gt_render.eef_position)
                        * 1_000.0
                    ),
                    medoid_joint_rmse_mrad=float(
                        np.sqrt(np.mean(np.square(medoid[:7] - physical_target[:7]))) * 1_000.0
                    ),
                    ground_truth_rgb_mae=rgb_mae,
                    deterministic_robot_mask_iou=robot_iou,
                    deterministic_ball_centroid_error_px=ball_centroid_error,
                )
            )
        object_plot = render_state_cloud_plot(
            delay_ticks=context.delay_ticks,
            state_samples=context.physical_samples[:, :, 16:19],
            state_targets=context.physical_targets[:, 16:19],
            title="Object future distribution",
            state_label="Ball center",
            config=config,
            xy_limits_mm=object_limits,
        )
        eef_plot = render_state_cloud_plot(
            delay_ticks=context.delay_ticks,
            state_samples=prepared.eef_samples,
            state_targets=prepared.eef_targets,
            title="EEF future distribution",
            state_label="End effector",
            config=config,
            xy_limits_mm=eef_limits,
        )
        joint_plot = render_joint_band_plot(
            delay_ticks=context.delay_ticks,
            joint_samples=context.physical_samples[:, :, :7],
            joint_targets=context.physical_targets[:, :7],
            config=config,
            joint_limits=joint_limits,
        )
        summaries = tuple(
            (
                f"ball {row.medoid_object_position_error_mm:.1f} mm | "
                f"EEF {row.medoid_eef_position_error_mm:.1f} mm | "
                f"q {row.medoid_joint_rmse_mrad:.1f} mrad | "
                f"{row.valid_sample_count}/32 valid"
            )
            for row in delay_metrics
        )
        panel = assemble_context_panel(
            current_rgb=values["agentview"][context.source_tick],
            ghost_overlays=np.stack(overlays),
            object_plot=object_plot,
            eef_plot=eef_plot,
            joint_plot=joint_plot,
            delay_ticks=context.delay_ticks,
            delay_summaries=summaries,
            title="Flow Belief future-state quality",
            subtitle=(
                f"L{context.level} | seed {context.scene_seed} | "
                f"source tick {context.source_tick} | phase {context.source_phase} | "
                f"{context.display_tag}"
            ),
        )
        _save_rgb(target / "panel.png", panel)
        np.savez(
            target / "geometry.npz",
            delay_ticks=context.delay_ticks,
            target_ticks=prepared.target_ticks,
            valid_sample_mask=prepared.valid_sample_mask,
            medoid_sample_indices=prepared.medoid_sample_indices,
            object_samples=context.physical_samples[:, :, 16:19],
            object_targets=context.physical_targets[:, 16:19],
            eef_samples=prepared.eef_samples,
            eef_targets=prepared.eef_targets,
            joint_samples=context.physical_samples[:, :, :7],
            joint_targets=context.physical_targets[:, :7],
        )
        _write_json(
            target / "metrics.json",
            {
                "schema_version": 1,
                "format_id": "flow_belief_ghost_context_v1",
                "identity": context.identity,
                "roles": list(context.roles),
                "display_tag": context.display_tag,
                "source_phase": context.source_phase,
                "delays": [dataclasses.asdict(row) for row in delay_metrics],
            },
        )
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return GhostContextRenderResult(
        source_tick=context.source_tick,
        panel=panel,
        delay_metrics=tuple(delay_metrics),
        geometry=prepared,
        invalid_sample_count=int(np.count_nonzero(~prepared.valid_sample_mask)),
    )
