"""Review-video rendering shared by pilot and bulk collection."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from latency_meta_mdp.data.recording import SynchronizedEpisode


def write_review_video(
    *,
    episode: SynchronizedEpisode,
    output_path: Path,
    fps: int,
) -> None:
    """Encode synchronized agent/wrist RGB boundaries side by side."""

    height, width, channels = episode.boundaries[0].deployment.images["agentview"].rgb.shape
    if channels != 3:
        raise ValueError("review video requires RGB policy cameras")
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
            f"{width * 2}x{height}",
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
        raise RuntimeError("ffmpeg did not create its review-video input pipe")
    try:
        for boundary in episode.boundaries:
            agent = boundary.deployment.images["agentview"].rgb
            wrist = boundary.deployment.images["robot0_eye_in_hand"].rgb
            if agent.shape != (height, width, 3) or wrist.shape != (height, width, 3):
                raise ValueError("review video cameras changed shape within one episode")
            process.stdin.write(np.concatenate([agent, wrist], axis=1).tobytes())
    finally:
        process.stdin.close()
    return_code = process.wait()
    if return_code:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg review-video encoding failed with exit code {return_code}")
