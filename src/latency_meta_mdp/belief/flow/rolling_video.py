"""Nested launch-window and discrete-delay video composition."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import numpy as np

from latency_meta_mdp.belief.flow.ghost_config import FlowBeliefGhostConfig
from latency_meta_mdp.belief.flow.ghost_visuals import (
    assemble_context_panel,
    render_joint_band_plot,
    render_state_cloud_plot,
)
from latency_meta_mdp.belief.flow.rolling_visuals import (
    RollingJointLimits,
    RollingPlotLimits,
)


@dataclass(frozen=True)
class ProgressiveRevealStage:
    delay_index: int
    delay_tick: int
    alpha: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.delay_index, bool)
            or not isinstance(self.delay_index, int)
            or self.delay_index < 0
            or isinstance(self.delay_tick, bool)
            or not isinstance(self.delay_tick, int)
            or self.delay_tick <= 0
            or not np.isfinite(self.alpha)
            or not 0.0 < self.alpha <= 1.0
        ):
            raise ValueError("progressive reveal stage is invalid")


def build_progressive_reveal_schedule(
    *,
    delay_ticks: tuple[int, ...],
    fade_alphas: tuple[float, ...],
    final_hold_frames: int,
) -> tuple[ProgressiveRevealStage, ...]:
    if (
        not delay_ticks
        or delay_ticks != tuple(sorted(set(delay_ticks)))
        or any(
            isinstance(delay, bool) or not isinstance(delay, int) or delay <= 0
            for delay in delay_ticks
        )
    ):
        raise ValueError("progressive delay ticks must be sorted unique positive integers")
    if (
        not fade_alphas
        or any(not np.isfinite(alpha) or not 0.0 < alpha <= 1.0 for alpha in fade_alphas)
        or tuple(fade_alphas) != tuple(sorted(fade_alphas))
        or fade_alphas[-1] != 1.0
    ):
        raise ValueError("progressive fade alphas must increase and end at one")
    if (
        isinstance(final_hold_frames, bool)
        or not isinstance(final_hold_frames, int)
        or final_hold_frames < 0
    ):
        raise ValueError("progressive final hold frame count is invalid")
    stages = [
        ProgressiveRevealStage(delay_index=index, delay_tick=delay, alpha=float(alpha))
        for index, delay in enumerate(delay_ticks)
        for alpha in fade_alphas
    ]
    stages.extend(
        ProgressiveRevealStage(
            delay_index=len(delay_ticks) - 1,
            delay_tick=delay_ticks[-1],
            alpha=1.0,
        )
        for _ in range(final_hold_frames)
    )
    return tuple(stages)


def iter_progressive_context_frames(
    *,
    current_rgb: np.ndarray,
    ghost_overlays: np.ndarray,
    object_samples: np.ndarray,
    object_targets: np.ndarray,
    eef_samples: np.ndarray,
    eef_targets: np.ndarray,
    joint_samples: np.ndarray,
    joint_targets: np.ndarray,
    delay_ticks: np.ndarray,
    delay_summaries: tuple[str, ...],
    config: FlowBeliefGhostConfig,
    object_limits: RollingPlotLimits,
    eef_limits: RollingPlotLimits,
    joint_limits: RollingJointLimits,
    title: str,
    subtitle_prefix: str,
    fade_alphas: tuple[float, ...] = (0.35, 0.7, 1.0),
    final_hold_frames: int = 4,
) -> Iterator[np.ndarray]:
    delays = tuple(int(value) for value in np.asarray(delay_ticks))
    schedule = build_progressive_reveal_schedule(
        delay_ticks=delays,
        fade_alphas=fade_alphas,
        final_hold_frames=final_hold_frames,
    )
    last_key = None
    last_frame = None
    for stage in schedule:
        key = (stage.delay_index, stage.alpha)
        if key == last_key and last_frame is not None:
            yield np.array(last_frame, copy=True)
            continue
        object_plot = render_state_cloud_plot(
            delay_ticks=np.asarray(delays),
            state_samples=object_samples,
            state_targets=object_targets,
            title="Object future distribution",
            state_label="Ball center",
            config=config,
            xy_limits_mm=object_limits,
            progressive_delay_index=stage.delay_index,
            progressive_alpha=stage.alpha,
        )
        eef_plot = render_state_cloud_plot(
            delay_ticks=np.asarray(delays),
            state_samples=eef_samples,
            state_targets=eef_targets,
            title="EEF future distribution",
            state_label="End effector",
            config=config,
            xy_limits_mm=eef_limits,
            progressive_delay_index=stage.delay_index,
            progressive_alpha=stage.alpha,
        )
        joint_plot = render_joint_band_plot(
            delay_ticks=np.asarray(delays),
            joint_samples=joint_samples,
            joint_targets=joint_targets,
            config=config,
            joint_limits=joint_limits,
            progressive_delay_index=stage.delay_index,
        )
        frame = assemble_context_panel(
            current_rgb=current_rgb,
            ghost_overlays=ghost_overlays,
            object_plot=object_plot,
            eef_plot=eef_plot,
            joint_plot=joint_plot,
            delay_ticks=np.asarray(delays),
            delay_summaries=delay_summaries,
            title=title,
            subtitle=(
                f"{subtitle_prefix} | revealing +{stage.delay_tick * 20} ms (d={stage.delay_tick})"
            ),
            progressive_delay_index=stage.delay_index,
            progressive_alpha=stage.alpha,
        )
        last_key = key
        last_frame = frame
        yield np.array(frame, copy=True)


def _flatten(frame_groups: Iterable[Iterable[np.ndarray]]) -> Iterator[np.ndarray]:
    for group in frame_groups:
        yield from group


def write_progressive_review_video(
    *,
    frame_groups: Iterable[Iterable[np.ndarray]],
    output_path: Path,
    fps: int,
) -> int:
    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise ValueError("progressive video fps must be positive")
    if output_path.exists():
        raise FileExistsError(f"progressive video already exists: {output_path}")
    frames = _flatten(frame_groups)
    try:
        first = np.asarray(next(frames))
    except StopIteration as exc:
        raise ValueError("progressive video requires at least one frame") from exc
    if first.ndim != 3 or first.shape[-1] != 3 or first.dtype != np.uint8:
        raise ValueError("progressive video frame is invalid")
    height, width = first.shape[:2]
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-n",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ],
        stdin=subprocess.PIPE,
    )
    if process.stdin is None:
        raise RuntimeError("ffmpeg did not create a progressive-video input pipe")
    frame_count = 0
    try:
        for frame in chain((first,), frames):
            value = np.asarray(frame)
            if value.shape != first.shape or value.dtype != np.uint8:
                raise ValueError("progressive video frames must share shape and dtype")
            process.stdin.write(value.tobytes())
            frame_count += 1
    except BaseException:
        process.stdin.close()
        process.wait()
        output_path.unlink(missing_ok=True)
        raise
    process.stdin.close()
    return_code = process.wait()
    if return_code:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"progressive video encoding failed with exit code {return_code}")
    return frame_count
