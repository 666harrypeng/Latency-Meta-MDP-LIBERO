"""Native observation replacement and future-indexed expert labels from one cache."""

import numpy as np

from latency_meta_mdp.data.forecast.dataset import ForecastPolicyDataset
from latency_meta_mdp.data.sampling import UniformDelaySampler


def eligible_counts(episodes):
    return [sum(max(0, e["frame_count"] - 10 - q) for e in episodes) for q in range(1, 21)]


class ForecastOnlyPolicyDataset(ForecastPolicyDataset):
    def __init__(self, native_dataset, *, episode_rows, cache):
        super().__init__(native_dataset, episode_rows=episode_rows, cache=cache)
        source, groups, offset = [], [], 0
        for q in range(1, 21):
            base = 0
            per_query = []
            for episode in self.episodes:
                n = episode["frame_count"]
                per_query.extend((base + np.arange(10, max(10, n - q))) * 20 + q - 1)
                base += n
            if not per_query:
                raise ValueError("forecast-only needs real targets at every query")
            source.extend(per_query)
            groups.append(np.arange(offset, offset + len(per_query), dtype=np.int64))
            offset += len(per_query)
        self.source_indices = np.asarray(source, dtype=np.int64)
        self.indices_by_query = tuple(groups)

    def __len__(self):
        return len(self.source_indices)

    def training_sampler(self, *, seed=0, batch_size=20, start_batch=0):
        return UniformDelaySampler(
            self.indices_by_query, seed=seed, batch_size=batch_size, start_batch=start_batch
        )

    def __getitem__(self, index):
        if (
            not isinstance(index, (int, np.integer))
            or isinstance(index, bool)
            or not 0 <= index < len(self)
        ):
            raise IndexError("forecast-only policy index out of range")
        source = int(self.source_indices[index])
        # Parent validation uses the full virtual view, not this filtered view's length.
        sample = self.source_sample(source)
        native_index, q0 = divmod(source, 20)
        episode_index = int(np.searchsorted(self.ends, native_index, side="right"))
        h = native_index - (int(self.ends[episode_index - 1]) if episode_index else 0)
        _, actions = self._episode_arrays[self.episodes[episode_index]["episode_id"]]
        start = h + q0 + 1
        n = min(50, len(actions) - start)
        forecast = sample["forecast"]
        if n <= 0 or forecast["rgb"] is None:
            raise ValueError("forecast-only source lacks forecast or real future actions")
        target = np.zeros((50, 7), np.float32)
        target[:n] = actions[start : start + n]
        return {
            "image": forecast["rgb"][0],
            "wrist_image": forecast["rgb"][1],
            "state": forecast["proprio"],
            "prompt": sample["prompt"],
            "actions": target,
            "actions_is_pad": np.arange(50) >= n,
        }

    def coverage(self):
        return {
            "real_action_sources": len(self.native_dataset),
            "eligible_pairs": len(self),
            "eligible_pairs_by_query": eligible_counts(self.episodes),
            "target_alignment": "forecast_time",
        }
