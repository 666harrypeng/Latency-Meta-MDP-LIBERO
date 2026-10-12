"""Verified read-only mmap inputs shared by disposable learner processes.

Only raw ActorPacket inputs are packed. Trainable features, recurrent states and
Bellman targets are always recomputed by the learner.
"""

import errno
import json
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
from pathlib import Path
from time import perf_counter

import numpy as np

from latency_meta_mdp.meta.information.config import sha256, validate_binding
from latency_meta_mdp.meta.information.dataset import _json, load_episode
from latency_meta_mdp.meta.information.schema import ACTOR_FIELDS, ActorPacket

_PACKED_FILES = ("records.json", "visual.npy")
_SOURCE_FILES = ("records.json", "audit.json", "visual.npz")
_ARRAY_FIELDS = ("current_proprio", "observed_proprio", "forecast_proprio", "buffer", "buffer_mask")


def _signature(path, names):
    result = []
    for name in (*names, "complete.json"):
        stat = (path / name).stat()
        result.append((stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    return tuple(result)


@dataclass(frozen=True)
class PackedEntry:
    source: Path
    path: Path
    source_signature: tuple
    packed_signature: tuple

    def verify_unchanged(self):
        if self.source_signature != _signature(self.source, _SOURCE_FILES):
            raise ValueError("source replay changed after admission")
        if self.packed_signature != _signature(self.path, _PACKED_FILES):
            raise ValueError("packed replay changed after admission")


def prepare_packed_episode(source, cache_root, *, binding, expected_receipt):
    source, cache_root = Path(source).resolve(), Path(cache_root).resolve()
    before = _signature(source, _SOURCE_FILES)
    validate_binding(expected_receipt, sha256(source / "complete.json"))
    original = json.loads((source / "complete.json").read_text())
    validate_binding(binding, original["binding"])
    if set(original["files"]) != set(_SOURCE_FILES):
        raise ValueError("source file manifest differs")
    for name, expected in original["files"].items():
        if sha256(source / name) != expected:
            raise ValueError("source replay hash differs")
    format_root = cache_root / "mmap-v1"
    format_root.mkdir(parents=True, exist_ok=True)
    target = format_root / expected_receipt
    expected = dict(format_version=1, source_receipt_sha256=expected_receipt, binding=binding)
    if target.exists():
        packed_before = _signature(target, _PACKED_FILES)
        receipt = json.loads((target / "complete.json").read_text())
        validate_binding(expected, {key: receipt[key] for key in expected})
        if set(receipt["files"]) != set(_PACKED_FILES):
            raise ValueError("packed file manifest differs")
        for name, digest in receipt["files"].items():
            if sha256(target / name) != digest:
                raise ValueError("packed replay hash differs")
    else:
        # The canonical decoder validates every packet and folds forced stages.
        rows = load_episode(source, binding=binding)
        pending = Path(tempfile.mkdtemp(prefix=".pending-", dir=format_root))
        try:
            visuals, indices = [], {}

            def packet_values(packet):
                values = {}
                for field in fields(ActorPacket):
                    value = getattr(packet, field.name)
                    if field.name.endswith("_tokens") and value is not None:
                        key = id(value)
                        if key not in indices:
                            indices[key] = len(visuals)
                            visuals.append(value)
                        value = indices[key]
                    values[field.name] = value
                return values

            packed_rows = [
                {
                    **{k: v for k, v in row.items() if k not in ("packet", "next_packet")},
                    "packet": packet_values(row["packet"]),
                }
                for row in rows
            ]
            pool = np.lib.format.open_memmap(
                pending / "visual.npy",
                mode="w+",
                dtype=np.float32,
                shape=(len(visuals), 2, 196, 384),
            )
            for i, value in enumerate(visuals):
                pool[i] = value
            pool.flush()
            del pool
            (pending / "records.json").write_text(
                json.dumps(packed_rows, default=_json, allow_nan=False)
            )
            receipt = dict(expected, files={name: sha256(pending / name) for name in _PACKED_FILES})
            (pending / "complete.json").write_text(json.dumps(receipt, indent=2) + "\n")
            if before != _signature(source, _SOURCE_FILES):
                raise ValueError("source replay changed during packing")
            try:
                pending.rename(target)
            except OSError as exc:
                if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise
                return prepare_packed_episode(
                    source, cache_root, binding=binding, expected_receipt=expected_receipt
                )
            packed_before = _signature(target, _PACKED_FILES)
        finally:
            if pending.exists():
                shutil.rmtree(pending)
    entry = PackedEntry(source, target, before, packed_before)
    entry.verify_unchanged()
    return entry


def _load(entry):
    """Reattach inputs already validated at preparation and pinned by signatures."""
    entry.verify_unchanged()
    pool = np.load(entry.path / "visual.npy", mmap_mode="r", allow_pickle=False)
    rows = json.loads((entry.path / "records.json").read_text())
    for row in rows:
        values = row["packet"]
        if set(values) != ACTOR_FIELDS:
            raise ValueError("packed actor schema differs")
        # Calling __post_init__ would copy and rescan every repeated visual input.
        # This bypass is private to the admitted immutable derived representation.
        packet = object.__new__(ActorPacket)
        for key, value in values.items():
            if key.endswith("_tokens") and value is not None:
                value = pool[value]
            elif key in _ARRAY_FIELDS:
                value = np.asarray(value, dtype=bool if key == "buffer_mask" else np.float32)
                value.setflags(write=False)
            elif key == "completed_delays":
                value = tuple(value)
            object.__setattr__(packet, key, value)
        row["packet"] = packet
    for i, row in enumerate(rows):
        row["next_packet"] = rows[i + 1]["packet"] if i + 1 < len(rows) else None
    entry.verify_unchanged()
    return rows


class PackedReplay:
    def __init__(self, entries, *, sources=None):
        self.entries, self.sources = list(entries), sources
        if sources is not None and len(sources) != len(entries):
            raise ValueError("packed replay source count differs")
        self._loaded = {}
        self.hits = self.misses = 0
        self.load_seconds = 0.0

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        started = perf_counter()
        entry = self.entries[index]
        entry.verify_unchanged()
        if index in self._loaded:
            self.hits += 1
        else:
            self._loaded[index] = _load(entry)
            self.misses += 1
        self.load_seconds += perf_counter() - started
        return self._loaded[index]

    def __getstate__(self):
        # Spawn transfers only admission receipts, never arrays or open mmap state.
        return dict(entries=self.entries, sources=self.sources)

    def __setstate__(self, state):
        self.__init__(**state)

    def metrics(self):
        return dict(
            storage="mmap",
            hits=self.hits,
            misses=self.misses,
            load_seconds=self.load_seconds,
            mapped_episodes=len(self._loaded),
        )


def prepare_packed_replay(replay, cache_root, *, workers=4, progress=None):
    streams = (replay.offline, replay.online) if hasattr(replay, "offline") else (replay,)
    entries = [entry for stream in streams for entry in stream.entries]

    def prepare(entry):
        return prepare_packed_episode(
            entry[0], cache_root, binding=entry[1], expected_receipt=entry[2]
        )

    prepared = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for entry in pool.map(prepare, entries):
            prepared.append(entry)
            if progress is not None:
                progress(
                    dict(phase="prepare_replay", episodes=len(prepared), expected=len(entries))
                )
    return PackedReplay(prepared, sources=getattr(replay, "sources", None))
