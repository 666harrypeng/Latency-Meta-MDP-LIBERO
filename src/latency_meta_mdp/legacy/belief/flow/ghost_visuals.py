"""Pillow-based Flow Belief ghost overlays, plots, panels, and videos."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from latency_meta_mdp.legacy.belief.flow.ghost_config import FlowBeliefGhostConfig
from latency_meta_mdp.legacy.belief.flow.rolling_visuals import (
    RollingJointLimits,
    RollingPlotLimits,
)


def _validate_rgb(value: np.ndarray, *, shape: tuple[int, int, int]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype != np.uint8:
        raise ValueError("ghost visualization RGB array is invalid")
    return array


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size=size)
    except OSError:
        return ImageFont.load_default()


def _paste_vertical_text(
    image: Image.Image,
    *,
    text: str,
    center: tuple[int, int],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
) -> None:
    bounds = font.getbbox(text)
    width = bounds[2] - bounds[0] + 4
    height = bounds[3] - bounds[1] + 4
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    ImageDraw.Draw(layer).text((2 - bounds[0], 2 - bounds[1]), text, font=font, fill=fill)
    rotated = layer.rotate(90, expand=True)
    image.paste(
        rotated,
        (center[0] - rotated.width // 2, center[1] - rotated.height // 2),
        rotated,
    )


def _blend_color(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    image[mask] = (1.0 - alpha) * image[mask] + alpha * np.asarray(color, dtype=np.float32)


def _erode_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    result = np.asarray(mask, dtype=np.bool_).copy()
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
    return result


def _inner_contour(mask: np.ndarray, width: int) -> np.ndarray:
    return np.asarray(mask, dtype=np.bool_) & ~_erode_mask(mask, width)


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
    overlap = ground_truth & prediction
    output = background.astype(np.float32)
    _blend_color(
        output,
        overlap,
        config.overlap_rgb,
        config.overlap_alpha,
    )
    rows, columns = np.indices(overlap.shape)
    overlap_hatch = overlap & ((rows + columns) % config.overlap_hatch_spacing_px < 2)
    _blend_color(
        output,
        overlap_hatch,
        config.overlap_rgb,
        config.overlap_hatch_alpha,
    )
    _blend_color(
        output,
        _inner_contour(ground_truth, config.ground_truth_outline_width_px),
        config.ground_truth_rgb,
        config.overlay_alpha,
    )
    _blend_color(
        output,
        _inner_contour(prediction, config.prediction_outline_width_px),
        config.prediction_rgb,
        config.overlay_alpha,
    )
    return np.clip(np.rint(output), 0, 255).astype(np.uint8)


def render_state_cloud_plot(
    *,
    delay_ticks: np.ndarray,
    state_samples: np.ndarray,
    state_targets: np.ndarray,
    title: str,
    state_label: str,
    config: FlowBeliefGhostConfig,
    xy_limits_mm: RollingPlotLimits | None = None,
    progressive_delay_index: int | None = None,
    progressive_alpha: float = 1.0,
) -> np.ndarray:
    delays = np.asarray(delay_ticks, dtype=np.int64)
    samples = np.asarray(state_samples, dtype=np.float64)
    targets = np.asarray(state_targets, dtype=np.float64)
    if (
        delays.shape != (5,)
        or samples.shape != (5, 32, 3)
        or targets.shape != (5, 3)
        or not np.all(np.isfinite(samples))
        or not np.all(np.isfinite(targets))
        or not title
        or not state_label
    ):
        raise ValueError("state-cloud plot inputs have invalid shapes or values")
    if progressive_delay_index is not None and (
        isinstance(progressive_delay_index, bool)
        or not isinstance(progressive_delay_index, int)
        or not 0 <= progressive_delay_index < len(delays)
    ):
        raise ValueError("state-cloud progressive delay index is invalid")
    if not np.isfinite(progressive_alpha) or not 0.0 < progressive_alpha <= 1.0:
        raise ValueError("state-cloud progressive alpha must lie in (0, 1]")
    size = 500
    plot_left, plot_top, plot_size = 82, 92, 340
    origin = targets[0, :2]
    sample_xy = (samples[..., :2] - origin) * 1_000.0
    target_xy = (targets[..., :2] - origin) * 1_000.0
    flat = np.concatenate((sample_xy.reshape(-1, 2), target_xy), axis=0)
    if xy_limits_mm is None:
        center = 0.5 * (flat.min(axis=0) + flat.max(axis=0))
        half_span = max(float(np.max(np.abs(flat - center))), 1.0) * 1.16
        low = center - half_span
        high = center + half_span
    else:
        if not isinstance(xy_limits_mm, RollingPlotLimits):
            raise TypeError("state-cloud shared limits must be RollingPlotLimits")
        low = np.asarray([xy_limits_mm.x_min_mm, xy_limits_mm.y_min_mm])
        high = np.asarray([xy_limits_mm.x_max_mm, xy_limits_mm.y_max_mm])
        if np.any(flat < low - 1e-9) or np.any(flat > high + 1e-9):
            raise ValueError("state-cloud points exceed the shared rolling limits")

    def project(points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        span = high - low
        result = np.empty_like(values)
        result[..., 0] = plot_left + (values[..., 0] - low[0]) / span[0] * plot_size
        result[..., 1] = plot_top + plot_size - (values[..., 1] - low[1]) / span[1] * plot_size
        return result

    image = Image.new("RGB", (size, size), (250, 250, 250))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.text((18, 14), title, fill=(24, 24, 24, 255), font=_font(21, bold=True))
    draw.text(
        (18, 44),
        f"Top-down displacement from d=1 target | {state_label}",
        fill=(70, 70, 70, 255),
        font=_font(13),
    )
    draw.rectangle(
        (plot_left, plot_top, plot_left + plot_size, plot_top + plot_size),
        outline=(90, 90, 90, 255),
        width=1,
    )
    for fraction in (0.0, 0.5, 1.0):
        x = plot_left + fraction * plot_size
        y = plot_top + fraction * plot_size
        draw.line((x, plot_top, x, plot_top + plot_size), fill=(215, 215, 215, 255))
        draw.line((plot_left, y, plot_left + plot_size, y), fill=(215, 215, 215, 255))
        x_value = low[0] + fraction * (high[0] - low[0])
        y_value = high[1] - fraction * (high[1] - low[1])
        draw.text(
            (x - 17, plot_top + plot_size + 7),
            f"{x_value:.0f}",
            fill=(55, 55, 55, 255),
            font=_font(12),
        )
        draw.text(
            (plot_left - 50, y - 7),
            f"{y_value:.0f}",
            fill=(55, 55, 55, 255),
            font=_font(12),
        )
    _paste_vertical_text(
        image,
        text="relative Y (mm)",
        center=(20, 262),
        font=_font(13),
        fill=(40, 40, 40, 255),
    )
    cloud = project(sample_xy)
    target_points = project(target_xy)
    visible_count = len(delays) if progressive_delay_index is None else progressive_delay_index + 1
    draw.line(
        [tuple(point) for point in target_points[:visible_count]],
        fill=(*config.ground_truth_rgb, 180),
        width=2,
    )
    for delay_index, delay in enumerate(delays[:visible_count]):
        active = progressive_delay_index is not None and delay_index == progressive_delay_index
        sample_alpha = 58
        target_alpha = 255
        if progressive_delay_index is not None:
            sample_alpha = int(
                round((110 if active else 34) * (progressive_alpha if active else 1.0))
            )
            target_alpha = int(
                round((255 if active else 150) * (progressive_alpha if active else 1.0))
            )
        for point in cloud[delay_index]:
            x, y = point
            draw.ellipse(
                (x - 2.5, y - 2.5, x + 2.5, y + 2.5),
                fill=(*config.prediction_rgb, sample_alpha),
            )
        x, y = target_points[delay_index]
        draw.ellipse(
            (x - 6, y - 6, x + 6, y + 6),
            fill=(250, 250, 250, 255),
            outline=(*config.ground_truth_rgb, target_alpha),
            width=4 if active else 3,
        )
        draw.text(
            (x + 8, y - 18),
            f"{int(delay) * 20} ms",
            fill=(*config.ground_truth_rgb, target_alpha),
            font=_font(12, bold=True),
        )
    draw.ellipse((82, 459, 91, 468), fill=(*config.prediction_rgb, 90))
    draw.text((97, 455), "Flow samples", fill=(55, 55, 55, 255), font=_font(12))
    draw.ellipse(
        (190, 458, 201, 469),
        fill=(250, 250, 250, 255),
        outline=(*config.ground_truth_rgb, 255),
        width=2,
    )
    draw.text((207, 455), "GT future state", fill=(55, 55, 55, 255), font=_font(12))
    draw.text((352, 455), "relative X (mm)", fill=(40, 40, 40, 255), font=_font(12))
    return np.asarray(image, dtype=np.uint8)


def joint_quantile_bands(
    joint_samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    samples = np.asarray(joint_samples, dtype=np.float64)
    if samples.ndim != 3 or samples.shape[0] != 5 or samples.shape[2] != 7:
        raise ValueError("joint samples must have shape [5, samples, 7]")
    if samples.shape[1] < 2 or not np.all(np.isfinite(samples)):
        raise ValueError("joint samples must contain finite distribution samples")
    values = np.quantile(samples, (0.025, 0.16, 0.5, 0.84, 0.975), axis=1)
    return tuple(values[index] for index in range(5))  # type: ignore[return-value]


def render_joint_band_plot(
    *,
    delay_ticks: np.ndarray,
    joint_samples: np.ndarray,
    joint_targets: np.ndarray,
    config: FlowBeliefGhostConfig,
    joint_limits: RollingJointLimits | None = None,
    progressive_delay_index: int | None = None,
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
    if progressive_delay_index is not None and (
        isinstance(progressive_delay_index, bool)
        or not isinstance(progressive_delay_index, int)
        or not 0 <= progressive_delay_index < len(delays)
    ):
        raise ValueError("joint-band progressive delay index is invalid")
    image = Image.new("RGB", (820, 500), (250, 250, 250))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.text(
        (18, 14),
        "Robot joint future distributions",
        fill=(24, 24, 24, 255),
        font=_font(21, bold=True),
    )
    draw.text(
        (18, 43),
        "GT, Flow median, 68% band, and 95% band",
        fill=(70, 70, 70, 255),
        font=_font(13),
    )
    draw.line((535, 52, 557, 52), fill=(*config.ground_truth_rgb, 255), width=3)
    draw.text((563, 44), "GT", fill=(55, 55, 55, 255), font=_font(12))
    draw.line((603, 52, 625, 52), fill=(*config.prediction_rgb, 255), width=3)
    draw.text((631, 44), "Flow median", fill=(55, 55, 55, 255), font=_font(12))
    lower95, lower68, median, upper68, upper95 = joint_quantile_bands(samples)
    visible_count = len(delays) if progressive_delay_index is None else progressive_delay_index + 1
    for joint in range(7):
        column = joint % 2
        row = joint // 2
        left = 55 + column * 400
        top = 76 + row * 103
        right = left + 350
        bottom = top + 72
        if joint_limits is None:
            all_values = np.concatenate((lower95[:, joint], upper95[:, joint], targets[:, joint]))
            low = float(all_values.min())
            high = float(all_values.max())
            if high - low < 1e-6:
                high = low + 1e-6
            pad = 0.08 * (high - low)
            low -= pad
            high += pad
        else:
            if not isinstance(joint_limits, RollingJointLimits):
                raise TypeError("joint shared limits must be RollingJointLimits")
            low, high = (float(value) for value in joint_limits.radians[joint])
            if (
                np.any(samples[:, :, joint] < low - 1e-9)
                or np.any(samples[:, :, joint] > high + 1e-9)
                or np.any(targets[:, joint] < low - 1e-9)
                or np.any(targets[:, joint] > high + 1e-9)
            ):
                raise ValueError("joint values exceed the shared rolling limits")

        def point(delay: float, value: float) -> tuple[float, float]:
            x = left + (delay - delays[0]) / (delays[-1] - delays[0]) * (right - left)
            y = bottom - (value - low) / (high - low) * (bottom - top)
            return x, y

        draw.rectangle((left, top, right, bottom), outline=(120, 120, 120, 255))
        for fraction in (0.0, 0.5, 1.0):
            y = top + fraction * (bottom - top)
            draw.line((left, y, right, y), fill=(220, 220, 220, 255))
        visible_delays = delays[:visible_count]
        if visible_count >= 2:
            polygon95 = [
                point(d, v)
                for d, v in zip(visible_delays, lower95[:visible_count, joint], strict=True)
            ]
            polygon95 += [
                point(d, v)
                for d, v in reversed(
                    list(
                        zip(
                            visible_delays,
                            upper95[:visible_count, joint],
                            strict=True,
                        )
                    )
                )
            ]
            polygon68 = [
                point(d, v)
                for d, v in zip(visible_delays, lower68[:visible_count, joint], strict=True)
            ]
            polygon68 += [
                point(d, v)
                for d, v in reversed(
                    list(
                        zip(
                            visible_delays,
                            upper68[:visible_count, joint],
                            strict=True,
                        )
                    )
                )
            ]
            draw.polygon(polygon95, fill=(*config.prediction_rgb, 25))
            draw.polygon(polygon68, fill=(*config.prediction_rgb, 58))
            draw.line(
                [
                    point(d, v)
                    for d, v in zip(visible_delays, median[:visible_count, joint], strict=True)
                ],
                fill=(*config.prediction_rgb, 255),
                width=2,
            )
            draw.line(
                [
                    point(d, v)
                    for d, v in zip(visible_delays, targets[:visible_count, joint], strict=True)
                ],
                fill=(*config.ground_truth_rgb, 255),
                width=2,
            )
        else:
            x_median, y_median = point(delays[0], median[0, joint])
            x_target, y_target = point(delays[0], targets[0, joint])
            draw.line(
                (
                    x_median,
                    point(delays[0], lower95[0, joint])[1],
                    x_median,
                    point(delays[0], upper95[0, joint])[1],
                ),
                fill=(*config.prediction_rgb, 100),
                width=4,
            )
            draw.ellipse(
                (x_median - 3, y_median - 3, x_median + 3, y_median + 3),
                fill=(*config.prediction_rgb, 255),
            )
            draw.ellipse(
                (x_target - 3, y_target - 3, x_target + 3, y_target + 3),
                fill=(*config.ground_truth_rgb, 255),
            )
        draw.text(
            (left + 4, top + 3),
            f"q{joint + 1}",
            fill=(30, 30, 30, 255),
            font=_font(12, bold=True),
        )
        draw.text(
            (left - 43, top - 4),
            f"{high:.2f}",
            fill=(70, 70, 70, 255),
            font=_font(10),
        )
        draw.text(
            (left - 43, bottom - 8),
            f"{low:.2f}",
            fill=(70, 70, 70, 255),
            font=_font(10),
        )
        if joint in (5, 6):
            for delay in delays:
                x, _ = point(float(delay), low)
                draw.text(
                    (x - 12, bottom + 4),
                    f"{int(delay) * 20}",
                    fill=(60, 60, 60, 255),
                    font=_font(10),
                )
    _paste_vertical_text(
        image,
        text="joint angle (rad)",
        center=(13, 270),
        font=_font(12),
        fill=(45, 45, 45, 255),
    )
    draw.text((690, 475), "latency (ms)", fill=(45, 45, 45, 255), font=_font(12))
    return np.asarray(image, dtype=np.uint8)


def assemble_context_panel(
    *,
    current_rgb: np.ndarray,
    ghost_overlays: np.ndarray,
    object_plot: np.ndarray,
    eef_plot: np.ndarray,
    joint_plot: np.ndarray,
    delay_ticks: np.ndarray,
    delay_summaries: tuple[str, ...],
    title: str,
    subtitle: str,
    progressive_delay_index: int | None = None,
    progressive_alpha: float = 1.0,
) -> np.ndarray:
    current = _validate_rgb(current_rgb, shape=(256, 256, 3))
    overlays = np.asarray(ghost_overlays)
    if overlays.shape != (5, 256, 256, 3) or overlays.dtype != np.uint8:
        raise ValueError("context panel overlays are invalid")
    objects = _validate_rgb(object_plot, shape=(500, 500, 3))
    eef = _validate_rgb(eef_plot, shape=(500, 500, 3))
    joints = _validate_rgb(joint_plot, shape=(500, 820, 3))
    delays = np.asarray(delay_ticks)
    if (
        delays.shape != (5,)
        or len(delay_summaries) != 5
        or any(not value for value in delay_summaries)
        or not title
        or not subtitle
    ):
        raise ValueError("context panel labels or delays are invalid")
    if progressive_delay_index is not None and (
        isinstance(progressive_delay_index, bool)
        or not isinstance(progressive_delay_index, int)
        or not 0 <= progressive_delay_index < len(delays)
    ):
        raise ValueError("context panel progressive delay index is invalid")
    if not np.isfinite(progressive_alpha) or not 0.0 < progressive_alpha <= 1.0:
        raise ValueError("context panel progressive alpha must lie in (0, 1]")
    canvas = Image.new("RGB", (1920, 1200), (242, 244, 247))
    draw = ImageDraw.Draw(canvas)
    draw.text((32, 18), title, fill=(22, 28, 36), font=_font(30, bold=True))
    draw.text((34, 58), subtitle, fill=(75, 82, 92), font=_font(17))
    legend_x = 1110
    draw.line((legend_x, 35, legend_x + 36, 35), fill=(0, 114, 178), width=5)
    draw.text((legend_x + 45, 25), "GT contour", fill=(45, 45, 45), font=_font(14))
    draw.line((legend_x + 180, 35, legend_x + 216, 35), fill=(213, 94, 0), width=3)
    draw.text(
        (legend_x + 225, 25),
        "Flow medoid contour",
        fill=(45, 45, 45),
        font=_font(14),
    )
    overlap_box = (legend_x + 420, 25, legend_x + 456, 45)
    draw.rectangle(overlap_box, fill=(246, 242, 184), outline=(190, 174, 25), width=1)
    for offset in range(-12, 48, 8):
        draw.line(
            (
                max(overlap_box[0], overlap_box[0] + offset),
                max(overlap_box[1], overlap_box[3] - offset),
                min(overlap_box[2], overlap_box[0] + offset + 20),
                min(overlap_box[3], overlap_box[3] - offset + 20),
            ),
            fill=(190, 174, 25),
            width=1,
        )
    draw.text(
        (legend_x + 465, 25),
        "overlap tint + hatch",
        fill=(45, 45, 45),
        font=_font(14),
    )
    draw.text(
        (32, 98),
        "Representative medoid state (one valid Flow sample per delay)",
        fill=(45, 52, 62),
        font=_font(18, bold=True),
    )
    tile_width = 300
    tile_gap = 15
    image_size = 276
    for column in range(6):
        x = 30 + column * (tile_width + tile_gap)
        draw.rounded_rectangle(
            (x, 130, x + tile_width, 500),
            radius=8,
            fill=(255, 255, 255),
            outline=(207, 211, 217),
            width=1,
        )
        if column == 0:
            label = "Launch context | t = h"
            image = current
            summary_lines = ("Shared input for all", "five delay queries")
        else:
            delay_index = column - 1
            delay = int(delays[delay_index])
            label = f"Return +{delay * 20} ms | d = {delay}"
            image = overlays[delay_index]
            pieces = [piece.strip() for piece in delay_summaries[delay_index].split("|")]
            summary_lines = (" | ".join(pieces[:2]), " | ".join(pieces[2:]))
            if progressive_delay_index is not None:
                if delay_index > progressive_delay_index:
                    image = np.full_like(image, 235)
                    summary_lines = ("Future query", "not revealed yet")
                elif delay_index < progressive_delay_index:
                    image = np.clip(
                        0.58 * image.astype(np.float32) + 0.42 * 235.0,
                        0,
                        255,
                    ).astype(np.uint8)
                else:
                    image = np.clip(
                        progressive_alpha * image.astype(np.float32)
                        + (1.0 - progressive_alpha) * 235.0,
                        0,
                        255,
                    ).astype(np.uint8)
                    label += " | revealing"
        draw.text((x + 12, 142), label, fill=(30, 35, 42), font=_font(15, bold=True))
        resized = Image.fromarray(image).resize((image_size, image_size), Image.Resampling.LANCZOS)
        canvas.paste(resized, (x + 12, 171))
        draw.text((x + 12, 454), summary_lines[0], fill=(62, 67, 75), font=_font(12))
        draw.text((x + 12, 474), summary_lines[1], fill=(62, 67, 75), font=_font(12))
    draw.text(
        (32, 540),
        (
            "Full predictive distribution (32 Flow samples per delay)"
            if progressive_delay_index is None
            else "Progressive predictive distribution (near future to far future)"
        ),
        fill=(45, 52, 62),
        font=_font(18, bold=True),
    )
    canvas.paste(Image.fromarray(objects), (30, 580))
    canvas.paste(Image.fromarray(eef), (545, 580))
    canvas.paste(Image.fromarray(joints), (1070, 580))
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
