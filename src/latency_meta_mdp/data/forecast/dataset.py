"""Virtual balanced-q conditioning over the unchanged clean LeRobot training source."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from latency_meta_mdp.data.sampling import UniformDelaySampler
from latency_meta_mdp.io.artifacts import sha256_file


class ForecastPolicyDataset:
    def __init__(self, native_dataset, *, episode_rows, cache):
        self.native_dataset, self.cache = native_dataset, cache
        self.episodes = tuple(episode_rows)
        self.ends = np.cumsum([e["frame_count"] for e in self.episodes])
        if not len(self.ends) or int(self.ends[-1]) != len(native_dataset):
            raise ValueError("forecast view/native dataset length mismatch")
        if len({e["episode_id"] for e in self.episodes}) != len(self.episodes) or {
            e["episode_id"] for e in self.episodes
        } != set(cache.episodes):
            raise ValueError("forecast cache/native episode inventory mismatch")
        for e in self.episodes:
            stored = cache.episodes[e["episode_id"]]
            if any(
                stored[k] != e[k] for k in ("frame_count", "level", "logical_master_task_index")
            ):
                raise ValueError("forecast/native episode identities mismatch")
        self._episode_arrays = {}

    def __getstate__(self):
        return {**self.__dict__, "_episode_arrays": {}}

    def __len__(self):
        return len(self.native_dataset) * 20

    def training_sampler(self, *, seed=0, batch_size=20, start_batch=0):
        indices = np.arange(len(self.native_dataset), dtype=np.int64) * 20
        return UniformDelaySampler(
            [indices + q for q in range(20)],
            seed=seed,
            batch_size=batch_size,
            start_batch=start_batch,
        )

    def __getitem__(self, index):
        if (
            not isinstance(index, (int, np.integer))
            or isinstance(index, bool)
            or not 0 <= index < len(self)
        ):
            raise IndexError("forecast policy index out of range")
        native_index, offset = divmod(int(index), 20)
        episode_index = int(np.searchsorted(self.ends, native_index, side="right"))
        h = native_index - (int(self.ends[episode_index - 1]) if episode_index else 0)
        episode = self.episodes[episode_index]
        episode_id = episode["episode_id"]
        if episode_id not in self._episode_arrays:
            # Keep only one small episode state/control array per worker.
            self._episode_arrays.clear()
            self._episode_arrays[episode_id] = self.cache.episode_arrays(episode_id)
        state, actions = self._episode_arrays[episode_id]
        raw = self.native_dataset[native_index]
        if not np.allclose(np.asarray(raw["state"]), state[h], atol=1e-6, rtol=0):
            raise ValueError("native/forecast current state is misaligned")
        if ("frame_index" in raw and int(raw["frame_index"]) != h) or (
            "episode_index" in raw and int(raw["episode_index"]) != episode_index
        ):
            raise ValueError("native/forecast frame identity is misaligned")
        n = min(50, len(actions) - h)
        target = np.zeros((50, 7), np.float32)
        target[:n] = actions[h : h + n]
        q = offset + 1
        forecast = self.cache.read(episode_id, h, q)
        return {
            **{k: raw[k] for k in ("image", "wrist_image", "state", "prompt")},
            "actions": target,
            "actions_is_pad": np.arange(50) >= n,
            "forecast": {
                "rgb": None if forecast is None else forecast[0],
                "proprio": None if forecast is None else forecast[1],
                "query_ticks": q,
            },
        }

    def coverage(self):
        sources = len(self.native_dataset)
        ready = [sum(max(0, e["frame_count"] - q - 9) for e in self.episodes) for q in range(1, 21)]
        return {
            "real_action_sources": sources,
            "virtual_pairs": sources * 20,
            "forecast_available_by_query": ready,
            "forecast_missing_by_query": [sources - n for n in ready],
        }


def load_forecast_policy_dataset(native_dataset, spec):
    from latency_meta_mdp.data.forecast.cache import ForecastCache

    if not isinstance(spec, dict) or set(spec) != {
        "cache_root",
        "policy_export_manifest",
        "bindings",
    }:
        raise ValueError("forecast dataset specification is invalid")
    cache = ForecastCache(Path(spec["cache_root"]), expected_bindings=spec["bindings"])
    path = Path(spec["policy_export_manifest"]).resolve()
    export = json.loads(path.read_text())
    if (
        export.get("format_id") != "metamdp_lerobot_structured_train_v1"
        or export.get("split") != "train"
    ):
        raise ValueError("forecast dataset requires the structured train export")
    if any(
        export.get(k) != spec["bindings"][k]
        for k in ("source_manifest_sha256", "split_manifest_sha256")
    ):
        raise ValueError("forecast/export source split binding mismatch")
    levels = {e["level"] for e in cache.episodes.values()}
    rows = [r for r in export["datasets"] if r["level"] in levels]
    if len(levels) != 1 or len(rows) != 1:
        raise ValueError("forecast export level mismatch")
    nested_path = (path.parent / rows[0]["dataset_manifest"]).resolve()
    if (
        not nested_path.is_relative_to(path.parent)
        or sha256_file(nested_path) != rows[0]["dataset_manifest_sha256"]
    ):
        raise ValueError("forecast export manifest hash mismatch")
    nested = json.loads(nested_path.read_text())
    return ForecastPolicyDataset(native_dataset, episode_rows=nested["episodes"], cache=cache)
