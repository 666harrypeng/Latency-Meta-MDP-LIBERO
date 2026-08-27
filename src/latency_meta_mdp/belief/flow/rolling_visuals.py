"""Shared-scale summary visuals for rolling Flow Belief inspection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _readonly(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size=size)
    except OSError:
        return ImageFont.load_default()


@dataclass(frozen=True)
class RollingPlotLimits:
    x_min_mm: float
    x_max_mm: float
    y_min_mm: float
    y_max_mm: float

    def __post_init__(self) -> None:
        values = (self.x_min_mm, self.x_max_mm, self.y_min_mm, self.y_max_mm)
        if any(not np.isfinite(value) for value in values):
            raise ValueError("rolling plot limits must be finite")
        x_span = self.x_max_mm - self.x_min_mm
        y_span = self.y_max_mm - self.y_min_mm
        if x_span <= 0.0 or y_span <= 0.0 or not np.isclose(x_span, y_span):
            raise ValueError("rolling plot limits must define one positive square domain")


@dataclass(frozen=True)
class RollingJointLimits:
    radians: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.radians, dtype=np.float64)
        if (
            values.shape != (7, 2)
            or not np.all(np.isfinite(values))
            or np.any(values[:, 0] >= values[:, 1])
        ):
            raise ValueError("rolling joint limits are invalid")
        object.__setattr__(self, "radians", _readonly(values, dtype=np.float64))


def derive_rolling_plot_limits(
    state_samples: np.ndarray,
    state_targets: np.ndarray,
) -> RollingPlotLimits:
    samples = np.asarray(state_samples, dtype=np.float64)
    targets = np.asarray(state_targets, dtype=np.float64)
    if (
        samples.ndim != 4
        or samples.shape[1:] != (5, 32, 3)
        or targets.shape != (samples.shape[0], 5, 3)
        or not np.all(np.isfinite(samples))
        or not np.all(np.isfinite(targets))
    ):
        raise ValueError("rolling state samples or targets have invalid shapes")
    origins = targets[:, :1, :2]
    relative_samples = (samples[..., :2] - origins[:, :, None, :]) * 1_000.0
    relative_targets = (targets[..., :2] - origins) * 1_000.0
    values = np.concatenate(
        (relative_samples.reshape(-1, 2), relative_targets.reshape(-1, 2)),
        axis=0,
    )
    center = 0.5 * (values.min(axis=0) + values.max(axis=0))
    half_span = max(float(np.max(np.abs(values - center))), 1.0) * 1.10
    return RollingPlotLimits(
        x_min_mm=float(center[0] - half_span),
        x_max_mm=float(center[0] + half_span),
        y_min_mm=float(center[1] - half_span),
        y_max_mm=float(center[1] + half_span),
    )


def derive_rolling_joint_limits(
    joint_samples: np.ndarray,
    joint_targets: np.ndarray,
) -> RollingJointLimits:
    samples = np.asarray(joint_samples, dtype=np.float64)
    targets = np.asarray(joint_targets, dtype=np.float64)
    if (
        samples.ndim != 4
        or samples.shape[1:] != (5, 32, 7)
        or targets.shape != (samples.shape[0], 5, 7)
        or not np.all(np.isfinite(samples))
        or not np.all(np.isfinite(targets))
    ):
        raise ValueError("rolling joint samples or targets have invalid shapes")
    values = np.concatenate((samples.reshape(-1, 7), targets.reshape(-1, 7)), axis=0)
    low = values.min(axis=0)
    high = values.max(axis=0)
    span = np.maximum(high - low, 1e-6)
    padding = 0.08 * span
    return RollingJointLimits(np.stack((low - padding, high + padding), axis=1))


def _square_project(
    values: np.ndarray,
    *,
    low: np.ndarray,
    high: np.ndarray,
    left: float,
    top: float,
    size: float,
) -> np.ndarray:
    projected = np.empty_like(values, dtype=np.float64)
    projected[..., 0] = left + (values[..., 0] - low[0]) / (high[0] - low[0]) * size
    projected[..., 1] = top + size - (values[..., 1] - low[1]) / (high[1] - low[1]) * size
    return projected


def render_rolling_trajectory_overview(
    *,
    boundary_ticks: np.ndarray,
    object_position: np.ndarray,
    boundary_phase: np.ndarray,
    selected_source_ticks: np.ndarray,
    critical_start_tick: int,
    critical_end_tick: int,
    maximum_delay_ticks: int,
    level: int,
    scene_seed: int,
) -> np.ndarray:
    ticks = np.asarray(boundary_ticks, dtype=np.int64)
    positions = np.asarray(object_position, dtype=np.float64)
    phases = np.asarray(boundary_phase)
    selected = np.asarray(selected_source_ticks, dtype=np.int64)
    if (
        ticks.ndim != 1
        or len(ticks) < 2
        or positions.shape != (len(ticks), 3)
        or phases.shape != (len(ticks),)
        or selected.ndim != 1
        or len(selected) == 0
        or not np.all(np.isfinite(positions))
        or not np.all(np.isin(phases, ("pregrasp", "approach")))
        or not np.all(np.isin(selected, ticks))
        or critical_start_tick != int(ticks[0])
        or critical_end_tick != int(ticks[-1])
        or maximum_delay_ticks <= 0
        or level not in (1, 2, 3)
        or scene_seed < 0
    ):
        raise ValueError("rolling trajectory overview inputs are invalid")
    image = Image.new("RGB", (1440, 900), (242, 244, 247))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.text(
        (36, 24),
        "Flow Belief rolling inspection coverage",
        fill=(22, 28, 36, 255),
        font=_font(30, bold=True),
    )
    draw.text(
        (38, 66),
        f"L{level} | seed {scene_seed} | critical ticks {critical_start_tick}-{critical_end_tick}",
        fill=(75, 82, 92, 255),
        font=_font(17),
    )
    origin = positions[0, :2]
    xy_mm = (positions[:, :2] - origin) * 1_000.0
    low = xy_mm.min(axis=0)
    high = xy_mm.max(axis=0)
    center = 0.5 * (low + high)
    half_span = max(float(np.max(np.abs(xy_mm - center))), 1.0) * 1.12
    low = center - half_span
    high = center + half_span
    left, top, size = 90.0, 150.0, 610.0
    draw.rectangle((left, top, left + size, top + size), fill=(255, 255, 255, 255))
    for fraction in (0.0, 0.5, 1.0):
        x = left + fraction * size
        y = top + fraction * size
        draw.line((x, top, x, top + size), fill=(220, 223, 228, 255), width=1)
        draw.line((left, y, left + size, y), fill=(220, 223, 228, 255), width=1)
    projected = _square_project(xy_mm, low=low, high=high, left=left, top=top, size=size)
    phase_colors = {"pregrasp": (0, 114, 178, 255), "approach": (213, 94, 0, 255)}
    for index in range(len(projected) - 1):
        draw.line(
            (*projected[index], *projected[index + 1]),
            fill=phase_colors[str(phases[index])],
            width=5,
        )
    tick_to_index = {int(tick): index for index, tick in enumerate(ticks)}
    for ordinal, tick in enumerate(selected):
        x, y = projected[tick_to_index[int(tick)]]
        draw.ellipse(
            (x - 8, y - 8, x + 8, y + 8), fill=(255, 255, 255), outline=(25, 25, 25), width=3
        )
        label_x = x + 10 if ordinal % 2 == 0 else x - 45
        draw.text(
            (label_x, y - 12),
            f"h{int(tick)}",
            fill=(25, 25, 25),
            font=_font(13, bold=True),
        )
    draw.text(
        (90, 118),
        "Ground-truth object path (relative XY, mm)",
        fill=(40, 45, 52),
        font=_font(18, bold=True),
    )
    timeline_left, timeline_right = 805.0, 1365.0
    timeline_top = 180.0
    timeline_height = 520.0

    def time_x(tick: int) -> float:
        return timeline_left + (tick - critical_start_tick) / (
            critical_end_tick - critical_start_tick
        ) * (timeline_right - timeline_left)

    draw.text(
        (805, 118),
        "Launch windows on the critical-phase timeline",
        fill=(40, 45, 52),
        font=_font(18, bold=True),
    )
    phase_runs = []
    run_start = 0
    for index in range(1, len(phases) + 1):
        if index == len(phases) or phases[index] != phases[run_start]:
            phase_runs.append((run_start, index - 1, str(phases[run_start])))
            run_start = index
    for start_index, end_index, phase in phase_runs:
        x0 = time_x(int(ticks[start_index]))
        x1 = time_x(int(ticks[end_index]))
        draw.rectangle((x0, 145, x1, 170), fill=phase_colors[phase])
        draw.text((x0 + 4, 147), phase, fill=(255, 255, 255), font=_font(12, bold=True))
    lane_height = min(54.0, timeline_height / len(selected))
    for row, source_tick in enumerate(selected):
        y = timeline_top + row * lane_height
        x0 = time_x(int(source_tick))
        x1 = time_x(int(source_tick) + maximum_delay_ticks)
        draw.text(
            (735, y - 8), f"h={int(source_tick)}", fill=(45, 45, 45), font=_font(13, bold=True)
        )
        draw.line((timeline_left, y, timeline_right, y), fill=(215, 218, 223), width=1)
        draw.rounded_rectangle(
            (x0, y - 9, x1, y + 9), radius=4, fill=(68, 68, 68, 90), outline=(30, 30, 30), width=2
        )
    draw.line(
        (
            timeline_left,
            timeline_top + len(selected) * lane_height + 18,
            timeline_right,
            timeline_top + len(selected) * lane_height + 18,
        ),
        fill=(70, 70, 70),
        width=2,
    )
    for tick in (
        critical_start_tick,
        (critical_start_tick + critical_end_tick) // 2,
        critical_end_tick,
    ):
        x = time_x(tick)
        draw.text(
            (x - 18, timeline_top + len(selected) * lane_height + 26),
            f"{tick * 20} ms",
            fill=(60, 60, 60),
            font=_font(12),
        )
    return np.asarray(image, dtype=np.uint8)


def _heat_color(value: float, maximum: float) -> tuple[int, int, int, int]:
    fraction = 0.0 if maximum <= 0.0 else min(max(value / maximum, 0.0), 1.0)
    low = np.asarray([255.0, 250.0, 245.0])
    high = np.asarray([180.0, 35.0, 20.0])
    rgb = np.rint((1.0 - fraction) * low + fraction * high).astype(np.uint8)
    return int(rgb[0]), int(rgb[1]), int(rgb[2]), 255


def render_rolling_metric_heatmaps(
    *,
    source_ticks: np.ndarray,
    delay_ticks: np.ndarray,
    object_error_mm: np.ndarray,
    eef_error_mm: np.ndarray,
    joint_rmse_mrad: np.ndarray,
    invalid_sample_fraction: np.ndarray,
    level: int,
    scene_seed: int,
) -> np.ndarray:
    sources = np.asarray(source_ticks, dtype=np.int64)
    delays = np.asarray(delay_ticks, dtype=np.int64)
    arrays = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (
            object_error_mm,
            eef_error_mm,
            joint_rmse_mrad,
            invalid_sample_fraction,
        )
    )
    expected = (len(sources), len(delays))
    if (
        sources.ndim != 1
        or delays.ndim != 1
        or len(sources) == 0
        or len(delays) == 0
        or any(value.shape != expected or not np.all(np.isfinite(value)) for value in arrays)
        or any(np.any(value < 0.0) for value in arrays)
        or np.any(arrays[-1] > 1.0)
        or level not in (1, 2, 3)
        or scene_seed < 0
    ):
        raise ValueError("rolling heatmap inputs are invalid")
    image = Image.new("RGB", (1440, 900), (242, 244, 247))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.text(
        (36, 24), "Flow Belief rolling error map", fill=(22, 28, 36), font=_font(30, bold=True)
    )
    draw.text(
        (38, 66),
        f"L{level} | seed {scene_seed} | source tick x delay",
        fill=(75, 82, 92),
        font=_font(17),
    )
    facets = (
        ("Object medoid error (mm)", arrays[0]),
        ("EEF medoid error (mm)", arrays[1]),
        ("Robot joint RMSE (mrad)", arrays[2]),
        ("Invalid sample fraction", arrays[3]),
    )
    cell_width = min(72, 560 // len(sources))
    cell_height = min(52, 260 // len(delays))
    for facet_index, (title, values) in enumerate(facets):
        column = facet_index % 2
        row = facet_index // 2
        left = 70 + column * 690
        top = 125 + row * 380
        plot_left = left + 90
        plot_top = top + 55
        maximum = float(values.max())
        draw.rounded_rectangle(
            (left, top, left + 630, top + 350),
            radius=8,
            fill=(255, 255, 255),
            outline=(205, 209, 216),
        )
        draw.text((left + 18, top + 14), title, fill=(35, 40, 48), font=_font(19, bold=True))
        for source_index, source_tick in enumerate(sources):
            for delay_index, delay_tick in enumerate(delays):
                x0 = plot_left + source_index * cell_width
                y0 = plot_top + (len(delays) - delay_index - 1) * cell_height
                value = float(values[source_index, delay_index])
                draw.rectangle(
                    (x0, y0, x0 + cell_width, y0 + cell_height),
                    fill=_heat_color(value, maximum),
                    outline=(245, 245, 245),
                )
                if cell_width >= 48:
                    label = f"{value:.1f}" if facet_index < 3 else f"{value:.2f}"
                    draw.text((x0 + 5, y0 + 8), label, fill=(25, 25, 25), font=_font(11))
            x = plot_left + source_index * cell_width
            draw.text(
                (x, plot_top + len(delays) * cell_height + 7),
                str(int(source_tick)),
                fill=(55, 55, 55),
                font=_font(11),
            )
        for delay_index, delay_tick in enumerate(delays):
            y = plot_top + (len(delays) - delay_index - 1) * cell_height
            draw.text(
                (plot_left - 58, y + 8),
                f"{int(delay_tick) * 20} ms",
                fill=(55, 55, 55),
                font=_font(11),
            )
        draw.text(
            (
                plot_left + len(sources) * cell_width / 2 - 34,
                plot_top + len(delays) * cell_height + 29,
            ),
            "source tick",
            fill=(55, 55, 55),
            font=_font(12),
        )
        draw.text((left + 18, plot_top + 90), "delay", fill=(55, 55, 55), font=_font(12))
        draw.rectangle(
            (left + 470, top + 20, left + 580, top + 35), fill=_heat_color(maximum, maximum)
        )
        draw.text((left + 585, top + 15), f"max {maximum:.2f}", fill=(55, 55, 55), font=_font(11))
    return np.asarray(image, dtype=np.uint8)
