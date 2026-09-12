"""Public frozen-forecast features and episode-indexed decision-stage replay."""

import json
from pathlib import Path

import numpy as np

FEATURE_FORMAT = "rtc_meta_spatial_forecast_v1"


def meta_features(state):
    observation, buffer, packet = state["observation"], state["buffer"], state["belief"]
    if packet is None:
        raise ValueError("Meta replay requires an explicit prepared forecast packet")
    visual = np.zeros((2, 2, 196, 384), np.float16)
    if packet.current_visual_latents is not None:
        visual[0] = packet.current_visual_latents
    future = np.zeros(16, np.float32)
    if packet.available:
        visual[1], future = packet.future_visual_latents, packet.future_proprio
    history, history_mask = np.zeros(32, np.float32), np.zeros(32, np.float32)
    delays = buffer.delay_history_ticks
    if not 1 <= len(delays) <= 32:
        raise ValueError("Meta features require the declared completed-delay window")
    history[: len(delays)] = np.asarray(delays) / 20
    history_mask[: len(delays)] = 1
    vector = np.concatenate(
        (
            observation.state,
            future,
            buffer.unread_action_buffer.ravel(),
            buffer.unread_action_mask.astype(np.float32),
            history,
            history_mask,
            [
                observation.formal_tick / 1000,
                buffer.plan_age / 50,
                buffer.remaining_actions / 50,
                buffer.estimated_delay_ticks / 20,
                float(packet.available),
            ],
        )
    ).astype(np.float32)
    if vector.shape != (501,) or not np.isfinite(vector).all() or not np.isfinite(visual).all():
        raise ValueError("invalid public Meta feature shapes or values")
    # The last legal launch boundary has only Launch as an admissible action.
    legal = np.asarray([buffer.remaining_actions > 20, True], dtype=bool)
    return visual, vector, legal


class MetaEpisodeReplay:
    """Store each decision state once; transitions refer to integer state indices."""

    def __init__(self):
        self._indices, self._states, self._transitions = {}, [], []

    def _state_index(self, state):
        if state is None:
            return -1
        key = (state["observation"].formal_tick, state["buffer"].buffer_version)
        if key not in self._indices:
            self._indices[key] = len(self._states)
            self._states.append(meta_features(state))
        return self._indices[key]

    def __call__(self, transition):
        self._transitions.append(
            {
                "state_index": self._state_index(transition.state),
                "next_state_index": self._state_index(transition.next_state),
                "action": int(transition.action == "launch"),
                "reward": transition.reward,
                "bootstrap_discount": transition.bootstrap_discount,
                "duration_ticks": transition.duration_ticks,
                "terminated": transition.terminated,
                "truncated": transition.truncated,
                "shielded": transition.shielded,
            }
        )

    def save(self, path: Path, *, metadata):
        if not self._states or not self._transitions:
            raise ValueError("cannot export empty Meta replay")
        arrays = dict(
            zip(
                ("visual", "vector", "legal_actions"),
                (np.stack(values) for values in zip(*self._states)),
                strict=True,
            )
        )
        arrays.update(
            {
                key: np.asarray([row[key] for row in self._transitions])
                for key in self._transitions[0]
            }
        )
        arrays["metadata_utf8"] = np.frombuffer(
            json.dumps({"feature_format": FEATURE_FORMAT, **metadata}, allow_nan=False).encode(),
            dtype=np.uint8,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as f:
            np.savez_compressed(f, **arrays)
        return {
            "states": len(self._states),
            "transitions": len(self._transitions),
            "feature_format": FEATURE_FORMAT,
            "bytes": path.stat().st_size,
        }


class ExploratoryCursorScheduler:
    """Ordinary rollouts with varied launch intervals and reproducible exploration.

    Only request timing changes; robot actions still come from the frozen VLA.
    Feasibility and forced latest launches are owned by the existing runtime shield.
    """

    def __init__(self, *, seed, epsilon=0.25):
        if not 0 <= epsilon <= 1:
            raise ValueError("exploration probability must lie in [0,1]")
        self.rng = np.random.default_rng(seed)
        self.cursor = int(self.rng.choice([10, 20, 28]))
        self.epsilon = epsilon

    def __call__(self, state, observation, belief):
        if self.rng.random() < self.epsilon:
            return bool(self.rng.integers(2))
        return state.plan_age >= self.cursor
