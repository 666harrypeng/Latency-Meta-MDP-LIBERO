"""Pillow-based Flow Belief ghost overlays, plots, panels, and videos."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from latency_meta_mdp.belief.flow.ghost_config import FlowBeliefGhostConfig


def _validate_rgb(value: np.ndarray, *, shape: tuple[int, int, int]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype != np.uint8:
        raise ValueError("ghost visualization RGB array is invalid")
    return array


def compose_agentview_ghost(
    *,
    background_rgb: np.ndarray,
    ground_truth_robot_mask: np.ndarray,
    ground_truth_ball_mask: np.ndarray,
    prediction_robot_mask: np.ndarray,
    prediction_ball_mask: np.ndarray,
    config: FlowBeliefGhostConfig,
) -> np.ndarray:
    background = np.asarray(background_rgb)
    if background.ndim != 3 or background.shape[-1] != 3 or background.dtype != np.uint8:
        raise ValueError("ghost composition requires uint8 RGB background")
    height, width = background.shape[:2]
    masks = [
        np.asarray(mask, dtype=np.bool_)
        for mask in (
            ground_truth_robot_mask,
            ground_truth_ball_mask,
            prediction_robot_mask,
            prediction_ball_mask,
        )
    ]
    if any(mask.shape != (height, width) for mask in masks):
        raise ValueError("ghost composition masks must match the RGB image")
    ground_truth = masks[0] | masks[1]
    prediction = masks[2] | masks[3]
    only_ground_truth = ground_truth & ~prediction
    only_prediction = prediction & ~ground_truth
    overlap = ground_truth & prediction
    output = background.astype(np.float32)
    alpha = config.overlay_alpha
    output[only_ground_truth] = (1.0 - alpha) * output[only_ground_truth] + alpha * np.asarray(
        config.ground_truth_rgb, dtype=np.float32
    )
    output[only_prediction] = (1.0 - alpha) * output[only_prediction] + alpha * np.asarray(
        config.prediction_rgb, dtype=np.float32
    )
    output[overlap] = np.asarray(config.overlap_rgb, dtype=np.float32)
    return np.clip(np.rint(output), 0, 255).astype(np.uint8)


def _xy_projection(values: np.ndarray, *, size: int, margin: int) -> tuple[np.ndarray, tuple]:
    points = np.asarray(values, dtype=np.float64)
    flat = points.reshape(-1, 2)
    if flat.size == 0 or not np.all(np.isfinite(flat)):
        raise ValueError("trajectory plot points must be finite and non-empty")
    low = flat.min(axis=0)
    high = flat.max(axis=0)
    span = np.maximum(high - low, 1e-6)
    low -= 0.1 * span
    high += 0.1 * span
    span = high - low
    projected = np.empty_like(points)
    projected[..., 0] = margin + (points[..., 0] - low[0]) / span[0] * (size - 2 * margin)
    projected[..., 1] = size - margin - (points[..., 1] - low[1]) / span[1] * (size - 2 * margin)
    return projected, (low, high)


def render_trajectory_plot(
    *,
    delay_ticks: np.ndarray,
    object_samples: np.ndarray,
    object_targets: np.ndarray,
    eef_samples: np.ndarray,
    eef_targets: np.ndarray,
    config: FlowBeliefGhostConfig,
) -> np.ndarray:
    delays = np.asarray(delay_ticks)
    arrays = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (object_samples, object_targets, eef_samples, eef_targets)
    )
    if (
        delays.shape != (5,)
        or arrays[0].shape != (5, 32, 3)
        or arrays[1].shape != (5, 3)
        or arrays[2].shape != (5, 32, 3)
        or arrays[3].shape != (5, 3)
        or any(not np.all(np.isfinite(value)) for value in arrays)
    ):
        raise ValueError("trajectory plot inputs have invalid shapes or values")
    size = 512
    margin = 42
    all_xy = np.concatenate(
        (
            arrays[0][..., :2].reshape(-1, 2),
            arrays[1][..., :2],
            arrays[2][..., :2].reshape(-1, 2),
            arrays[3][..., :2],
        )
    )
    _, (low, high) = _xy_projection(all_xy, size=size, margin=margin)

    def project(points: np.ndarray) -> np.ndarray:
        values = np.asarray(points)[..., :2]
        span = high - low
        result = np.empty_like(values)
        result[..., 0] = margin + (values[..., 0] - low[0]) / span[0] * (size - 2 * margin)
        result[..., 1] = size - margin - (values[..., 1] - low[1]) / span[1] * (size - 2 * margin)
        return result

    image = Image.new("RGB", (size, size), (248, 248, 248))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((margin, margin, size - margin, size - margin), outline=(80, 80, 80, 255))
    draw.text((12, 10), "Object / EEF future clouds (top-down)", fill=(20, 20, 20, 255))
    gt_color = (*config.ground_truth_rgb, 255)
    pred_color = (*config.prediction_rgb, 80)
    object_cloud = project(arrays[0])
    eef_cloud = project(arrays[2])
    object_gt = project(arrays[1])
    eef_gt = project(arrays[3])
    for delay_index, delay in enumerate(delays):
        for point in object_cloud[delay_index]:
            x, y = point
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=pred_color)
        for point in eef_cloud[delay_index]:
            x, y = point
            draw.rectangle((x - 1, y - 1, x + 1, y + 1), fill=(213, 94, 0, 50))
        x, y = object_gt[delay_index]
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=gt_color, width=2)
        ex, ey = eef_gt[delay_index]
        draw.line((ex - 5, ey, ex + 5, ey), fill=gt_color, width=2)
        draw.line((ex, ey - 5, ex, ey + 5), fill=gt_color, width=2)
        draw.text((x + 6, y - 6), f"d{int(delay)}", fill=(40, 40, 40, 255))
    return np.asarray(image, dtype=np.uint8)


def render_joint_band_plot(
    *,
    delay_ticks: np.ndarray,
    joint_samples: np.ndarray,
    joint_targets: np.ndarray,
    config: FlowBeliefGhostConfig,
) -> np.ndarray:
    delays = np.asarray(delay_ticks, dtype=np.float64)
    samples = np.asarray(joint_samples, dtype=np.float64)
    targets = np.asarray(joint_targets, dtype=np.float64)
    if (
        delays.shape != (5,)
        or samples.shape != (5, 32, 7)
        or targets.shape != (5, 7)
        or not np.all(np.isfinite(samples))
        or not np.all(np.isfinite(targets))
    ):
        raise ValueError("joint-band plot inputs have invalid shapes or values")
    image = Image.new("RGB", (512, 512), (248, 248, 248))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.text((12, 8), "Joint future bands", fill=(20, 20, 20, 255))
    median = np.median(samples, axis=1)
    lower, upper = np.quantile(samples, (0.16, 0.84), axis=1)
    for joint in range(7):
        column = joint % 2
        row = joint // 2
        left = 18 + column * 250
        top = 32 + row * 118
        right = left + 230
        bottom = top + 96
        all_values = np.concatenate((lower[:, joint], upper[:, joint], targets[:, joint]))
        low = float(all_values.min())
        high = float(all_values.max())
        if high - low < 1e-6:
            high = low + 1e-6

        def point(delay: float, value: float) -> tuple[float, float]:
            x = left + (delay - delays[0]) / (delays[-1] - delays[0]) * (right - left)
            y = bottom - (value - low) / (high - low) * (bottom - top)
            return x, y

        draw.rectangle((left, top, right, bottom), outline=(130, 130, 130, 255))
        polygon = [point(d, v) for d, v in zip(delays, lower[:, joint], strict=True)]
        polygon += [
            point(d, v) for d, v in reversed(list(zip(delays, upper[:, joint], strict=True)))
        ]
        draw.polygon(polygon, fill=(*config.prediction_rgb, 45))
        draw.line(
            [point(d, v) for d, v in zip(delays, median[:, joint], strict=True)],
            fill=(*config.prediction_rgb, 255),
            width=2,
        )
        draw.line(
            [point(d, v) for d, v in zip(delays, targets[:, joint], strict=True)],
            fill=(*config.ground_truth_rgb, 255),
            width=2,
        )
        draw.text((left + 3, top + 2), f"q{joint + 1}", fill=(30, 30, 30, 255))
    return np.asarray(image, dtype=np.uint8)


def assemble_context_panel(
    *,
    current_rgb: np.ndarray,
    ground_truth_rgb: np.ndarray,
    ghost_overlays: np.ndarray,
    trajectory_plot: np.ndarray,
    joint_plot: np.ndarray,
    delay_ticks: np.ndarray,
    title: str,
) -> np.ndarray:
    current = _validate_rgb(current_rgb, shape=(256, 256, 3))
    ground_truth = np.asarray(ground_truth_rgb)
    overlays = np.asarray(ghost_overlays)
    if ground_truth.shape != (5, 256, 256, 3) or ground_truth.dtype != np.uint8:
        raise ValueError("context panel ground-truth images are invalid")
    if overlays.shape != (5, 256, 256, 3) or overlays.dtype != np.uint8:
        raise ValueError("context panel overlays are invalid")
    trajectory = _validate_rgb(trajectory_plot, shape=(512, 512, 3))
    joints = _validate_rgb(joint_plot, shape=(512, 512, 3))
    delays = np.asarray(delay_ticks)
    if delays.shape != (5,):
        raise ValueError("context panel delays are invalid")
    canvas = Image.new("RGB", (1280, 1280), (238, 238, 238))
    draw = ImageDraw.Draw(canvas)
    canvas.paste(Image.fromarray(current), (0, 0))
    draw.text((8, 8), "Current input", fill=(255, 255, 255))
    draw.text((8, 264), title, fill=(20, 20, 20))
    for row, delay in enumerate(delays):
        y = row * 256
        canvas.paste(Image.fromarray(ground_truth[row]), (256, y))
        canvas.paste(Image.fromarray(overlays[row]), (512, y))
        draw.text((264, y + 8), f"GT d={int(delay)}", fill=(255, 255, 255))
        draw.text((520, y + 8), f"Overlay d={int(delay)}", fill=(255, 255, 255))
    canvas.paste(Image.fromarray(trajectory), (768, 0))
    canvas.paste(Image.fromarray(joints), (768, 512))
    return np.asarray(canvas, dtype=np.uint8)


def write_rgb_video(
    *,
    frames: np.ndarray,
    output_path: Path,
    fps: int,
) -> None:
    values = np.asarray(frames)
    if (
        values.ndim != 4
        or values.shape[-1] != 3
        or values.dtype != np.uint8
        or len(values) == 0
        or isinstance(fps, bool)
        or not isinstance(fps, int)
        or fps <= 0
    ):
        raise ValueError("ghost video frames or fps are invalid")
    if output_path.exists():
        raise FileExistsError(f"ghost video already exists: {output_path}")
    height, width = values.shape[1:3]
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
        raise RuntimeError("ffmpeg did not create a ghost-video input pipe")
    try:
        for frame in values:
            process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close()
    return_code = process.wait()
    if return_code:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"ghost video encoding failed with exit code {return_code}")
