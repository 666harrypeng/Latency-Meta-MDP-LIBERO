"""Verified read-only mmap inputs shared by disposable learner processes.

Only raw ActorPacket inputs are packed. Trainable features, recurrent states and
Bellman targets are always recomputed by the learner.
"""

import errno
import hashlib
import json
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from time import perf_counter

import numpy as np

from latency_meta_mdp.meta.information.config import sha256, validate_binding
from latency_meta_mdp.meta.information.dataset import _json, load_episode
from latency_meta_mdp.meta.information.schema import ACTOR_FIELDS, ActorPacket

_PACKED_FILES = ("records.json", "visual.npy")
_SOURCE_FILES = ("records.json", "audit.json", "visual.npz")
_ARRAY_FIELDS = ("current_proprio", "observed_proprio", "forecast_proprio", "buffer", "buffer_mask")


class CacheBudget:
    """Reserve conservative sizes before parallel workers allocate derived arrays."""

    def __init__(self, directory, maximum):
        from latency_meta_mdp.meta.information.replay_retention import tree_bytes

        if type(maximum) is not int or maximum < 1:
            raise ValueError("positive replay cache byte ceiling required")
        self.maximum = maximum
        self.used = tree_bytes(directory)
        self.lock = threading.Lock()
        self.reserve(0)

    def reserve(self, size, *, required=True):
        with self.lock:
            if self.used + size > self.maximum:
                if not required:
                    return False
                raise RuntimeError("replay cache storage ceiling reached before array allocation")
            self.used += size
            return True


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
    raw: bool = False

    def verify_unchanged(self):
        if self.source_signature != _signature(self.source, _SOURCE_FILES):
            raise ValueError("source replay changed after admission")
        if self.packed_signature != _signature(self.path, _PACKED_FILES):
            raise ValueError("packed replay changed after admission")


def prepare_packed_episode(
    source, cache_root, *, binding, expected_receipt, raw=False, storage_budget=None
):
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
    format_root = cache_root / ("mmap-raw-v2" if raw else "mmap-v1")
    format_root.mkdir(parents=True, exist_ok=True)
    target = format_root / expected_receipt
    expected = dict(
        format_version=2 if raw else 1, source_receipt_sha256=expected_receipt, binding=binding
    )
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
        decoded = load_episode(source, binding=binding, raw=raw)
        rows = decoded["rows"] if raw else decoded
        pending = Path(tempfile.mkdtemp(prefix=".pending-", dir=format_root))
        try:
            visuals, indices, hashes = [], {}, {}

            def visual_index(value):
                key = id(value)
                if key not in indices:
                    digest = hashlib.sha256(value.tobytes()).digest() if raw else key
                    if digest not in hashes:
                        hashes[digest] = len(visuals)
                        visuals.append(value)
                    # Keep the object alive so Python cannot reuse this identity.
                    indices[key] = (value, hashes[digest])
                return indices[key][1]

            events = []
            if raw:
                # Keep observation frame groups contiguous for zero-copy mmap reads.
                # Packet and forecast references reuse these exact feature bytes.
                for event in decoded["events"]:
                    payload = dict(event.payload)
                    if event.kind == "observe":
                        frames = payload["visual"]
                        start = len(visuals)
                        for value in frames:
                            digest = hashlib.sha256(value.tobytes()).digest()
                            hashes.setdefault(digest, len(visuals))
                            visuals.append(value)
                        payload["visual"] = [start, len(frames)]
                    elif event.kind == "predict":
                        payload["visual"] = [visual_index(payload["visual"]), 1]
                    events.append(
                        dict(
                            event_id=event.event_id,
                            kind=event.kind,
                            available_at=event.available_at,
                            represented_ticks=event.represented_ticks,
                            payload=payload,
                        )
                    )

            def packet_values(packet):
                values = {}
                for field in fields(packet):
                    value = getattr(packet, field.name)
                    if field.name.endswith("_tokens") and value is not None:
                        value = visual_index(value)
                    elif field.name == "action_spec":
                        value = asdict(value)
                    values[field.name] = value
                return values

            packed_rows = [
                {
                    **{k: v for k, v in row.items() if k not in ("packet", "next_packet")},
                    "packet": packet_values(row["packet"]),
                }
                for row in rows
            ]
            packed = packed_rows
            if raw:
                packed = dict(episode_id=decoded["episode_id"], rows=packed_rows, events=events)
            records_json = json.dumps(packed, default=_json, allow_nan=False)
            if storage_budget is not None:
                storage_budget.reserve(
                    len(visuals) * 2 * 196 * 384 * 4 + len(records_json.encode()) + 65536
                )
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
            (pending / "records.json").write_text(records_json)
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
                    source, cache_root, binding=binding, expected_receipt=expected_receipt, raw=raw
                )
            packed_before = _signature(target, _PACKED_FILES)
        finally:
            if pending.exists():
                shutil.rmtree(pending)
    entry = PackedEntry(source, target, before, packed_before, raw=raw)
    entry.verify_unchanged()
    return entry


def _load(entry):
    """Reattach inputs already validated at preparation and pinned by signatures."""
    entry.verify_unchanged()
    pool = np.load(entry.path / "visual.npy", mmap_mode="r", allow_pickle=False)
    if entry.raw:
        raise ValueError("raw timeline must use load_packed_raw")
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


def compact_packed_episode(entry, cache_root, *, storage_budget=None):
    """Store admitted raw features in half only when every FP32 bit roundtrips.

    Frozen vision services already emit half features, but source packets expose
    FP32. This is lossless storage, not a change to model or training precision.
    Keep the original cache and fall back to it for arbitrary FP32 inputs.
    """
    if not entry.raw:
        raise ValueError("compact storage requires raw replay")
    entry.verify_unchanged()
    root = Path(cache_root) / "mmap-raw-f16-v1"
    target = root / entry.path.name
    original = json.loads((entry.path / "complete.json").read_text())
    expected = dict(
        format_version=3,
        source_receipt_sha256=original["source_receipt_sha256"],
        binding=original["binding"],
        parent_receipt_sha256=sha256(entry.path / "complete.json"),
    )
    if target.exists():
        before = _signature(target, _PACKED_FILES)
        receipt = json.loads((target / "complete.json").read_text())
        validate_binding(expected, {k: receipt[k] for k in expected})
        if set(receipt["files"]) != set(_PACKED_FILES):
            raise ValueError("compact replay file manifest differs")
        for name, digest in receipt["files"].items():
            validate_binding(digest, sha256(target / name))
        if before != _signature(target, _PACKED_FILES):
            raise ValueError("compact replay changed during verification")
    else:
        source = np.load(entry.path / "visual.npy", mmap_mode="r", allow_pickle=False)
        if source.dtype != np.float32:
            raise ValueError("compact source must be admitted FP32")
        for start in range(0, len(source), 32):
            values = source[start : start + 32]
            with np.errstate(over="ignore", under="ignore"):
                restored = values.astype(np.float16).astype(np.float32)
            if not np.array_equal(values.view(np.uint32), restored.view(np.uint32)):
                entry.verify_unchanged()
                return entry
        if storage_budget is not None and not storage_budget.reserve(
            source.size * 2 + (entry.path / "records.json").stat().st_size + 65536, required=False
        ):
            return entry
        root.mkdir(parents=True, exist_ok=True)
        pending = Path(tempfile.mkdtemp(prefix=".pending-", dir=root))
        try:
            pool = np.lib.format.open_memmap(
                pending / "visual.npy", mode="w+", dtype=np.float16, shape=source.shape
            )
            for start in range(0, len(source), 32):
                pool[start : start + 32] = source[start : start + 32]
            pool.flush()
            del pool
            shutil.copyfile(entry.path / "records.json", pending / "records.json")
            receipt = dict(expected, files={name: sha256(pending / name) for name in _PACKED_FILES})
            (pending / "complete.json").write_text(json.dumps(receipt, indent=2) + "\n")
            entry.verify_unchanged()
            try:
                pending.rename(target)
            except OSError as exc:
                if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise
                return compact_packed_episode(entry, cache_root, storage_budget=storage_budget)
        finally:
            if pending.exists():
                shutil.rmtree(pending)
        before = _signature(target, _PACKED_FILES)
    entry.verify_unchanged()
    result = PackedEntry(entry.source, target, entry.source_signature, before, raw=True)
    result.verify_unchanged()
    return result


def load_packed_raw(entry):
    """Restore already validated immutable raw inputs without copying mmap features."""
    from types import MappingProxyType

    from latency_meta_mdp.meta.information.memory_schema import (
        ActionSpec,
        AttentionPacket,
        PublicEvent,
    )

    if not entry.raw:
        raise ValueError("raw timeline cache required")
    entry.verify_unchanged()
    pool = np.load(entry.path / "visual.npy", mmap_mode="r", allow_pickle=False)
    data = json.loads((entry.path / "records.json").read_text())
    for row in data["rows"]:
        values = row["packet"]
        packet_class = AttentionPacket if "action_spec" in values else ActorPacket
        if set(values) != {f.name for f in fields(packet_class)}:
            raise ValueError("packed raw actor fields differ")
        packet = object.__new__(packet_class)
        for key, value in values.items():
            if key.endswith("_tokens") and value is not None:
                value = pool[value]
            elif key in _ARRAY_FIELDS:
                value = np.asarray(value, dtype=bool if key == "buffer_mask" else np.float32)
                value.setflags(write=False)
            elif key in ("completed_delays", "decision_key"):
                value = tuple(value)
            elif key == "action_spec":
                value = ActionSpec(**value)
            object.__setattr__(packet, key, value)
        row["packet"] = packet
    for i, row in enumerate(data["rows"]):
        row["next_packet"] = data["rows"][i + 1]["packet"] if i + 1 < len(data["rows"]) else None
    events = []
    for value in data["events"]:
        payload = value["payload"]
        if value["kind"] in ("observe", "predict"):
            start, count = payload["visual"]
            payload["visual"] = (
                pool[start : start + count] if value["kind"] == "observe" else pool[start]
            )
        for name in ("proprio", "action"):
            if isinstance(payload.get(name), list):
                payload[name] = np.asarray(payload[name], dtype=np.float32)
                payload[name].setflags(write=False)
        event = object.__new__(PublicEvent)
        for key, item in value.items():
            if key in ("available_at", "represented_ticks"):
                item = tuple(item)
            object.__setattr__(event, key, MappingProxyType(payload) if key == "payload" else item)
        events.append(event)
    data["events"] = events
    entry.verify_unchanged()
    return data


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
