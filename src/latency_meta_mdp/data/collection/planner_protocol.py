"""Strict JSON/NPZ boundary for isolated CuRobo worker operations."""

from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from latency_meta_mdp.data.collection.robot_bridge import PandaPlanningBridge

FK_REQUEST_FORMAT = "structured_expert_fk_request_v1"
FK_RESULT_FORMAT = "structured_expert_fk_result_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _npz_bytes(**arrays: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.savez(output, **arrays)
    return output.getvalue()


def _strict_json(path: Path, fields: set[str], *, name: str) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} must be valid JSON") from error
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{name} fields are invalid")
    return value, payload


@dataclass(frozen=True)
class FkBatch:
    positions_world: np.ndarray
    rotations_world: np.ndarray

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions_world)
        rotations = np.asarray(self.rotations_world)
        if (
            positions.dtype != np.float64
            or positions.ndim != 2
            or positions.shape[0] == 0
            or positions.shape[1:] != (3,)
            or rotations.dtype != np.float64
            or rotations.shape != (positions.shape[0], 3, 3)
            or not np.all(np.isfinite(positions))
            or not np.all(np.isfinite(rotations))
        ):
            raise ValueError("FK batch must contain finite float64 [N,3]/[N,3,3] arrays")
        identities = np.repeat(np.eye(3, dtype=np.float64)[None], len(rotations), axis=0)
        if not np.allclose(
            np.transpose(rotations, (0, 2, 1)) @ rotations,
            identities,
            atol=1.0e-5,
            rtol=0.0,
        ) or not np.allclose(np.linalg.det(rotations), 1.0, atol=1.0e-5, rtol=0.0):
            raise ValueError("FK rotations must be proper rotation matrices")
        positions = np.array(positions, copy=True)
        rotations = np.array(rotations, copy=True)
        positions.setflags(write=False)
        rotations.setflags(write=False)
        object.__setattr__(self, "positions_world", positions)
        object.__setattr__(self, "rotations_world", rotations)


def write_fk_request(root: Path, *, bridge: PandaPlanningBridge, qpos: np.ndarray) -> str:
    if not isinstance(bridge, PandaPlanningBridge):
        raise TypeError("bridge must be a PandaPlanningBridge")
    values = np.asarray(qpos)
    if values.dtype != np.float64 or values.ndim != 2 or values.shape[1] != 7:
        raise ValueError("FK request qpos must be float64[N,7]")
    if not np.all(np.isfinite(values)):
        raise ValueError("FK request qpos must be finite")
    if len(values) == 0:
        raise ValueError("FK request qpos must be non-empty")
    if np.any(values <= bridge.joint_lower) or np.any(values >= bridge.joint_upper):
        raise ValueError("FK request qpos must lie strictly inside joint limits")
    root = Path(root)
    root.mkdir(parents=False, exist_ok=False)
    qpos_payload = _npz_bytes(qpos=values)
    (root / "qpos.npz").write_bytes(qpos_payload)
    request_payload = _json_bytes(
        {
            "schema_version": 1,
            "format_id": FK_REQUEST_FORMAT,
            "bridge": bridge.to_mapping(),
            "bridge_sha256": bridge.sha256,
            "joint_names": list(bridge.joint_names),
            "batch_count": len(values),
            "qpos_sha256": _hash(qpos_payload),
        }
    )
    (root / "request.json").write_bytes(request_payload)
    return _hash(request_payload)


def load_fk_request(root: Path) -> tuple[PandaPlanningBridge, np.ndarray, str]:
    root = Path(root)
    if {item.name for item in root.iterdir()} != {"request.json", "qpos.npz"}:
        raise ValueError("FK request inventory is invalid")
    raw, request_payload = _strict_json(
        root / "request.json",
        {
            "schema_version",
            "format_id",
            "bridge",
            "bridge_sha256",
            "joint_names",
            "batch_count",
            "qpos_sha256",
        },
        name="FK request",
    )
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or (raw["format_id"] != FK_REQUEST_FORMAT)
    ):
        raise ValueError("FK request format is invalid")
    if type(raw["batch_count"]) is not int or raw["batch_count"] <= 0:
        raise ValueError("FK request batch_count must be a positive integer")
    bridge = PandaPlanningBridge.from_mapping(raw["bridge"])
    if raw["bridge_sha256"] != bridge.sha256 or raw["joint_names"] != list(bridge.joint_names):
        raise ValueError("FK request bridge identity mismatch")
    qpos_payload = (root / "qpos.npz").read_bytes()
    if _hash(qpos_payload) != raw["qpos_sha256"]:
        raise ValueError("FK request qpos hash mismatch")
    try:
        with np.load(io.BytesIO(qpos_payload), allow_pickle=False) as source:
            if source.files != ["qpos"]:
                raise ValueError("FK request NPZ fields are invalid")
            qpos = np.array(source["qpos"], copy=True)
    except (OSError, ValueError) as error:
        raise ValueError("FK request qpos NPZ is invalid") from error
    if (
        qpos.dtype != np.float64
        or qpos.ndim != 2
        or qpos.shape != (raw["batch_count"], 7)
        or not np.all(np.isfinite(qpos))
    ):
        raise ValueError("FK request qpos shape or dtype is invalid")
    if np.any(qpos <= bridge.joint_lower) or np.any(qpos >= bridge.joint_upper):
        raise ValueError("FK request qpos lies outside joint limits")
    qpos.setflags(write=False)
    return bridge, qpos, _hash(request_payload)


def write_fk_result(root: Path, *, request_sha256: str, batch: FkBatch) -> str:
    if not isinstance(batch, FkBatch):
        raise TypeError("batch must be an FkBatch")
    if type(request_sha256) is not str or _SHA256.fullmatch(request_sha256) is None:
        raise ValueError("request_sha256 must be a SHA-256 digest")
    root = Path(root)
    root.mkdir(parents=False, exist_ok=False)
    fk_payload = _npz_bytes(
        positions_world=batch.positions_world,
        rotations_world=batch.rotations_world,
    )
    (root / "fk.npz").write_bytes(fk_payload)
    result_payload = _json_bytes(
        {
            "schema_version": 1,
            "format_id": FK_RESULT_FORMAT,
            "request_sha256": request_sha256,
            "batch_count": len(batch.positions_world),
            "fk_sha256": _hash(fk_payload),
        }
    )
    (root / "result.json").write_bytes(result_payload)
    return _hash(result_payload)


def load_fk_result(root: Path, *, expected_request_sha256: str) -> FkBatch:
    root = Path(root)
    if {item.name for item in root.iterdir()} != {"result.json", "fk.npz"}:
        raise ValueError("FK result inventory is invalid")
    raw, _ = _strict_json(
        root / "result.json",
        {"schema_version", "format_id", "request_sha256", "batch_count", "fk_sha256"},
        name="FK result",
    )
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or (raw["format_id"] != FK_RESULT_FORMAT)
    ):
        raise ValueError("FK result format is invalid")
    if type(raw["batch_count"]) is not int or raw["batch_count"] <= 0:
        raise ValueError("FK result batch_count must be a positive integer")
    if raw["request_sha256"] != expected_request_sha256:
        raise ValueError("FK result request identity mismatch")
    fk_payload = (root / "fk.npz").read_bytes()
    if _hash(fk_payload) != raw["fk_sha256"]:
        raise ValueError("FK result hash mismatch")
    try:
        with np.load(io.BytesIO(fk_payload), allow_pickle=False) as source:
            if set(source.files) != {"positions_world", "rotations_world"}:
                raise ValueError("FK result NPZ fields are invalid")
            batch = FkBatch(
                positions_world=np.array(source["positions_world"], copy=True),
                rotations_world=np.array(source["rotations_world"], copy=True),
            )
    except (OSError, ValueError) as error:
        raise ValueError("FK result NPZ is invalid") from error
    if len(batch.positions_world) != raw["batch_count"]:
        raise ValueError("FK result batch count mismatch")
    return batch


def request_curobo_fk_subprocess(
    bridge: PandaPlanningBridge,
    qpos: np.ndarray,
    *,
    worker_python: Path = Path(".venv-expert-realization/bin/python"),
) -> FkBatch:
    with tempfile.TemporaryDirectory(prefix="structured-fk-") as directory:
        root = Path(directory)
        request_root = root / "request"
        result_root = root / "result"
        request_sha = write_fk_request(request_root, bridge=bridge, qpos=qpos)
        completed = subprocess.run(
            [
                str(worker_python),
                "-m",
                "latency_meta_mdp.data.collection.curobo_worker",
                "--request",
                str(request_root),
                "--result",
                str(result_root),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            diagnostic = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"CuRobo FK worker failed: {diagnostic[-2000:]}")
        return load_fk_result(result_root, expected_request_sha256=request_sha)
