"""Lossless audit-only source frames for actual observation/prediction calls."""

import hashlib
import json
from pathlib import Path

import numpy as np


class SourceArchive:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "frames").mkdir()
        (self.root / "queries").mkdir()
        self.frames = {}
        self.count = 0

    def record(self, history, controls, context, *, kind):
        ticks = []
        for obs in history:
            values = (obs.image, obs.wrist_image, obs.state)
            digest = hashlib.sha256()
            for value in values:
                digest.update(np.asarray(value).tobytes())
            identity = digest.hexdigest()
            tick = obs.formal_tick
            if tick in self.frames and self.frames[tick] != identity:
                raise ValueError("same source frame identity has different content")
            if tick not in self.frames:
                np.savez_compressed(
                    self.root / f"frames/{tick}.npz",
                    image=obs.image,
                    wrist_image=obs.wrist_image,
                    state=obs.state,
                )
                self.frames[tick] = identity
            ticks.append(tick)
        np.savez_compressed(
            self.root / f"queries/{self.count}.npz",
            source_ticks=ticks,
            executed_controls=np.asarray(controls),
            previous_actions=context.previous_actions,
            previous_action_mask=context.previous_action_mask,
            origin_tick=context.origin_tick,
            buffer_version=context.buffer_version,
            query_ticks=context.estimated_delay_ticks,
            kind=kind,
            prompt=context.observation.prompt,
        )
        self.count += 1
        (self.root / "index.json").write_text(
            json.dumps(dict(frames=self.frames, queries=self.count))
        )
