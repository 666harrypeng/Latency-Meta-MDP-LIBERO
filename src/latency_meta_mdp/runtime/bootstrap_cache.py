"""Reuse native bootstrap actions without keeping a second VLA in device memory."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np


def _observation_digest(observation):
    digest = hashlib.sha256()
    digest.update(str(observation.formal_tick).encode())
    for key, value in sorted(observation.to_policy_inputs().items()):
        digest.update(key.encode())
        if isinstance(value, str):
            digest.update(value.encode())
        else:
            array = np.asarray(value)
            digest.update(str((array.shape, array.dtype)).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


class CachedBootstrapPolicy:
    """Cache only tick-zero H50 actions; preserve the native policy's noise draw."""

    def __init__(
        self, path, *, identity, noise_rng, native_policy=None, native_on_observation_miss=False
    ):
        self.path = Path(path)
        self.identity = identity
        self.noise_rng = noise_rng
        self.native_policy = native_policy
        self.native_on_observation_miss = native_on_observation_miss
        self.last_record = None

    def __call__(self, observation):
        if observation.formal_tick != 0:
            raise ValueError("bootstrap cache is only valid at tick zero")
        observation_sha = _observation_digest(observation)
        hit = self.path.is_file()
        persist = not hit
        if hit:
            with np.load(self.path, allow_pickle=False) as stored:
                record = json.loads(str(stored["metadata"]))
                actions = stored["actions"].copy()
            if record["identity"] != self.identity:
                raise ValueError("bootstrap cache identity differs")
            if record["actions_sha256"] != hashlib.sha256(actions.tobytes()).hexdigest():
                raise ValueError("bootstrap cache actions differ")
            if record["observation_sha256"] != observation_sha:
                if not self.native_on_observation_miss:
                    raise ValueError("bootstrap cache observation differs")
                hit = False
        if not hit:
            if self.native_policy is None:
                raise FileNotFoundError(f"Generate native bootstrap cache first: {self.path}")
            start = time.perf_counter_ns()
            actions = np.asarray(self.native_policy(observation)["actions"])
            record = {
                "identity": self.identity,
                "observation_sha256": observation_sha,
                "native_wall_ns": time.perf_counter_ns() - start,
                "actions_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
            }
        if actions.shape != (50, 7) or not np.isfinite(actions).all():
            raise ValueError("bootstrap cache requires finite H50 controller actions")
        if hit:
            # InProcessOpenpiPolicy consumes this draw even for the shared bootstrap.
            self.noise_rng.standard_normal((50, 32), dtype=np.float32)
        elif persist:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("xb") as stream:
                np.savez(stream, actions=actions, metadata=json.dumps(record, sort_keys=True))
        self.last_record = {
            **record,
            "cache_hit": hit,
            "path": str(self.path) if hit or persist else None,
        }
        return {"actions": actions}
