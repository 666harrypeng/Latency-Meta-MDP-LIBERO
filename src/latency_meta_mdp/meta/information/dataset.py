"""Hash-verified episode storage; privileged reports never enter public replay."""

import hashlib
import json
import zipfile
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

from latency_meta_mdp.meta.information.config import sha256, validate_binding
from latency_meta_mdp.meta.information.records import decision_view
from latency_meta_mdp.meta.information.schema import ACTOR_FIELDS, ActorPacket


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save_visual_archive(path, arrays):
    """Lossless NumPy archive with fast DEFLATE; existing readers remain compatible."""
    with Path(path).open("xb") as stream:
        with zipfile.ZipFile(
            stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
        ) as archive:
            for name, value in arrays.items():
                with archive.open(name + ".npy", "w", force_zip64=True) as entry:
                    np.lib.format.write_array(entry, value, allow_pickle=False)


def save_episode(path, recorder, *, report, binding):
    if not recorder.closed:
        raise ValueError("cannot publish an open episode")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    arrays, indices = {}, {}

    def visual(value):
        key = hashlib.sha256(value.tobytes()).hexdigest()
        if key not in indices:
            indices[key] = f"visual_{len(indices)}"
            arrays[indices[key]] = value
        return indices[key]

    def pack(packet):
        if packet is None:
            return None
        values = {}
        for field in fields(packet):
            value = getattr(packet, field.name)
            if field.name.endswith("_tokens") and value is not None:
                value = visual(value)
            elif field.name == "action_spec":
                value = asdict(value)
            values[field.name] = value
        return values

    rows = [
        {**row, "packet": pack(row["packet"]), "next_packet": pack(row["next_packet"])}
        for row in recorder.rows
    ]
    events = []
    for event in recorder.events:
        payload = dict(event.payload)
        if event.kind == "observe":
            payload["visual"] = [visual(v) for v in payload["visual"]]
        elif event.kind == "predict":
            payload["visual"] = visual(payload["visual"])
        events.append(
            dict(
                event_id=event.event_id,
                kind=event.kind,
                available_at=event.available_at,
                represented_ticks=event.represented_ticks,
                payload=payload,
            )
        )
    save_visual_archive(path / "visual.npz", arrays)
    metadata = dict(
        episode_id=recorder.episode_id,
        binding=binding,
        rows=rows,
        visual_arrays=len(arrays),
        operations=recorder.operations,
        summary=recorder.summary(),
    )
    if recorder.action_spec is not None:
        metadata.update(schema_version=2, events=events)
    (path / "records.json").write_text(json.dumps(metadata, default=_json, allow_nan=False))
    (path / "audit.json").write_text(json.dumps(report, default=_json, allow_nan=False))
    receipt = dict(
        binding=binding,
        files={name: sha256(path / name) for name in ("visual.npz", "records.json", "audit.json")},
        terminated=bool(rows and rows[-1]["terminated"]),
    )
    temporary = path / "receipt.tmp"
    temporary.write_text(json.dumps(receipt, indent=2))
    temporary.replace(path / "complete.json")


def load_episode(path, *, binding, raw=False):
    path = Path(path)
    receipt = json.loads((path / "complete.json").read_text())
    validate_binding(binding, receipt["binding"])
    if set(receipt["files"]) != {"visual.npz", "records.json", "audit.json"}:
        raise ValueError("episode file manifest differs")
    for name, expected in receipt["files"].items():
        if sha256(path / name) != expected:
            raise ValueError("episode content hash differs")
    data = json.loads((path / "records.json").read_text())
    validate_binding(binding, data["binding"])
    version = data.get("schema_version", 1)
    if version not in (1, 2):
        raise ValueError("unknown episode schema")
    with np.load(path / "visual.npz", allow_pickle=False) as store:
        visual_cache = {}

        def visual(key):
            if key not in visual_cache:
                visual_cache[key] = np.asarray(store[key], dtype=np.float32)
                visual_cache[key].setflags(write=False)
            return visual_cache[key]

        def unpack(values):
            if values is None:
                return None
            packet_class = ActorPacket
            expected_fields = ACTOR_FIELDS
            if version == 2:
                from latency_meta_mdp.meta.information.memory_schema import (
                    ActionSpec,
                    AttentionPacket,
                )

                packet_class = AttentionPacket
                expected_fields = {f.name for f in fields(packet_class)}
            if set(values) != expected_fields:
                raise ValueError("actor schema has missing or privileged fields")
            values = dict(values)
            for name in ("current_tokens", "forecast_tokens"):
                if values[name] is not None:
                    values[name] = visual(values[name])
            if version == 2:
                values["action_spec"] = ActionSpec(**values["action_spec"])
                values["decision_key"] = tuple(values["decision_key"])
            values["buffer_mask"] = np.asarray(values["buffer_mask"], bool)
            packet = packet_class(**values)
            # Validation above owns independent scalar/buffer copies. Visual sources
            # came from this verified archive and can share their read-only storage.
            for name in ("current_tokens", "forecast_tokens"):
                if values[name] is not None:
                    object.__setattr__(packet, name, values[name])
            return packet

        rows = [
            {**row, "packet": unpack(row["packet"]), "next_packet": unpack(row["next_packet"])}
            for row in data["rows"]
        ]
        events = []
        if version == 2:
            from latency_meta_mdp.meta.information.memory_schema import PublicEvent

            for value in data["events"]:
                value = dict(value)
                payload = dict(value["payload"])
                if value["kind"] == "observe":
                    payload["visual"] = np.stack([visual(k) for k in payload["visual"]])
                elif value["kind"] == "predict":
                    payload["visual"] = visual(payload["visual"])
                value["payload"] = payload
                events.append(PublicEvent(**value))
    if raw:
        return dict(episode_id=data["episode_id"], rows=rows, events=events)
    return decision_view(rows)
