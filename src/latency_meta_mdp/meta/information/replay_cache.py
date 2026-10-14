"""Bounded host replay snapshots, retaining first-load hash and change checks."""

import copy
import sys
from collections import OrderedDict
from dataclasses import fields, is_dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from latency_meta_mdp.meta.information.config import validate_binding
from latency_meta_mdp.meta.information.dataset import load_episode


def _size(value, seen):
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, np.ndarray):
        return value.nbytes + 128
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        return size + sum(_size(k, seen) + _size(v, seen) for k, v in value.items())
    if isinstance(value, tuple | list):
        return size + sum(_size(v, seen) for v in value)
    if is_dataclass(value):
        return size + sum(_size(getattr(value, f.name), seen) for f in fields(value))
    return size


def _signature(path):
    result = []
    for name in ("complete.json", "records.json", "visual.npz", "audit.json"):
        stat = (path / name).stat()
        result.append((stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    return tuple(result)


class EpisodeCache:
    def __init__(self, max_bytes=32 * 2**30):
        if max_bytes < 0:
            raise ValueError("negative replay cache budget")
        self.max_bytes = max_bytes
        self.bytes_used = self.hits = self.misses = 0
        self.load_seconds = 0.0
        self.entries = OrderedDict()

    def load(self, path, *, binding):
        started = perf_counter()
        path = Path(path).resolve()
        signature = _signature(path)
        entry = self.entries.pop(path, None)
        if entry is not None:
            previous, previous_binding, rows, size = entry
            if previous == signature:
                self.entries[path] = entry
                validate_binding(binding, previous_binding)
                self.hits += 1
                self.load_seconds += perf_counter() - started
                return rows
            self.bytes_used -= size
        # Rehash and decode after eviction or any file metadata change.
        rows = load_episode(path, binding=binding)
        if _signature(path) != signature:
            raise ValueError("replay files changed while loading")
        self.misses += 1
        size = _size(rows, set())
        if size <= self.max_bytes:
            while self.bytes_used + size > self.max_bytes:
                _, old = self.entries.popitem(last=False)
                self.bytes_used -= old[3]
            self.entries[path] = (signature, copy.deepcopy(binding), rows, size)
            self.bytes_used += size
        self.load_seconds += perf_counter() - started
        return rows

    def metrics(self):
        return dict(
            bytes_used=self.bytes_used,
            max_bytes=self.max_bytes,
            hits=self.hits,
            misses=self.misses,
            load_seconds=self.load_seconds,
        )
