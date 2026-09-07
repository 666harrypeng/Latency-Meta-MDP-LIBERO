"""Synchronized main/wrist videos recorded independently of the actor timing ledger."""

from __future__ import annotations

import os
import subprocess
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from latency_meta_mdp.policy_execution import PolicyObservation


class DualCameraVideoWriter:
    """Write every formal boundary to a single main-left/wrist-right 50 Hz MP4."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = self.path.with_name(f".{self.path.stem}-{uuid.uuid4().hex}.mp4")
        self.process = None
        self.shape = None
        self.frame_count = 0

    def __enter__(self):
        return self

    def write(self, observation: PolicyObservation):
        if observation.formal_tick != self.frame_count:
            raise ValueError("video must record consecutive formal boundaries starting at zero")
        main, wrist = observation.image, observation.wrist_image
        height = max(main.shape[0], wrist.shape[0]) + 24
        width = main.shape[1] + wrist.shape[1]
        height += height % 2
        width += width % 2
        shape = (height, width, 3)
        if self.shape is not None and self.shape != shape:
            raise ValueError("camera dimensions changed during a video")
        canvas = np.zeros(shape, np.uint8)
        canvas[24 : 24 + main.shape[0], : main.shape[1]] = main
        canvas[24 : 24 + wrist.shape[0], main.shape[1] : main.shape[1] + wrist.shape[1]] = wrist
        image = Image.fromarray(canvas)
        draw = ImageDraw.Draw(image)
        draw.text((4, 5), f"main  t={observation.formal_tick * 0.02:.2f}s", fill="white")
        draw.text((main.shape[1] + 4, 5), f"wrist  tick={observation.formal_tick}", fill="white")
        if self.process is None:
            self.shape = shape
            self.process = subprocess.Popen(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-n",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{width}x{height}",
                    "-r",
                    "50",
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-threads",
                    "1",
                    "-movflags",
                    "+faststart",
                    str(self.temporary),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        self.process.stdin.write(np.asarray(image).tobytes())
        self.frame_count += 1

    def __exit__(self, exc_type, exc, traceback):
        if self.process is None:
            return False
        self.process.stdin.close()
        code = self.process.wait(timeout=30)
        errors = self.process.stderr.read().decode(errors="replace")
        self.process.stderr.close()
        if code:
            if exc is None:
                raise RuntimeError(f"video encoding failed: {errors[-1000:]}")
            return False
        # Atomic publication without overwriting an existing recording. A video
        # from an interrupted episode remains useful for diagnosing that error.
        os.link(self.temporary, self.path)
        self.temporary.unlink()
        return False
