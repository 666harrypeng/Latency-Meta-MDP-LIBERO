from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from test_expert_realization_robot_bridge import _bridge


def test_fk_request_and_result_round_trip_exact_schemas(tmp_path: Path) -> None:
    """Break caught: worker request/result shapes or hashes can drift silently."""
    from latency_meta_mdp.data.collection.planner_protocol import (
        FkBatch,
        load_fk_request,
        load_fk_result,
        write_fk_request,
        write_fk_result,
    )

    bridge = _bridge()
    qpos = np.stack([bridge.home_joint_positions, bridge.home_joint_positions + 0.01])
    request_dir = tmp_path / "request"
    request_sha = write_fk_request(request_dir, bridge=bridge, qpos=qpos)
    loaded_bridge, loaded_qpos, loaded_sha = load_fk_request(request_dir)
    assert loaded_bridge == bridge
    assert np.array_equal(loaded_qpos, qpos)
    assert loaded_sha == request_sha

    positions = np.zeros((2, 3), dtype=np.float64)
    rotations = np.repeat(np.eye(3, dtype=np.float64)[None], 2, axis=0)
    result_dir = tmp_path / "result"
    write_fk_result(
        result_dir,
        request_sha256=request_sha,
        batch=FkBatch(positions_world=positions, rotations_world=rotations),
    )
    loaded = load_fk_result(result_dir, expected_request_sha256=request_sha)
    assert np.array_equal(loaded.positions_world, positions)
    assert np.array_equal(loaded.rotations_world, rotations)


@pytest.mark.parametrize(
    "mutation",
    (
        "qpos_shape",
        "qpos_dtype",
        "qpos_empty",
        "qpos_limit",
        "result_hash",
        "extra_field",
        "bool_count",
    ),
)
def test_fk_protocol_rejects_malformed_or_corrupted_payloads(tmp_path: Path, mutation: str) -> None:
    """Break caught: malformed worker artifacts cross the process trust boundary."""
    import json

    from latency_meta_mdp.data.collection.planner_protocol import (
        FkBatch,
        load_fk_request,
        load_fk_result,
        write_fk_request,
        write_fk_result,
    )

    bridge = _bridge()
    request_dir = tmp_path / "request"
    request_sha = write_fk_request(
        request_dir,
        bridge=bridge,
        qpos=bridge.home_joint_positions[None],
    )
    if mutation == "qpos_shape":
        np.savez(request_dir / "qpos.npz", qpos=np.zeros((1, 8), dtype=np.float64))
        with pytest.raises(ValueError):
            load_fk_request(request_dir)
        return
    if mutation == "qpos_dtype":
        np.savez(request_dir / "qpos.npz", qpos=np.zeros((1, 7), dtype=np.float32))
        with pytest.raises(ValueError):
            load_fk_request(request_dir)
        return
    if mutation == "qpos_empty":
        invalid_root = tmp_path / "invalid-empty"
        with pytest.raises(ValueError, match="non-empty"):
            write_fk_request(
                invalid_root,
                bridge=bridge,
                qpos=np.empty((0, 7), dtype=np.float64),
            )
        assert not invalid_root.exists()
        return
    if mutation == "qpos_limit":
        invalid_root = tmp_path / "invalid-limit"
        qpos = np.array(bridge.home_joint_positions[None], copy=True)
        qpos[0, 0] = bridge.joint_upper[0]
        with pytest.raises(ValueError, match="joint limits"):
            write_fk_request(invalid_root, bridge=bridge, qpos=qpos)
        assert not invalid_root.exists()
        return
    if mutation == "extra_field":
        path = request_dir / "request.json"
        payload = json.loads(path.read_text())
        payload["unknown"] = True
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError):
            load_fk_request(request_dir)
        return
    if mutation == "bool_count":
        path = request_dir / "request.json"
        payload = json.loads(path.read_text())
        payload["batch_count"] = True
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="batch_count"):
            load_fk_request(request_dir)
        return
    result_dir = tmp_path / "result"
    write_fk_result(
        result_dir,
        request_sha256=request_sha,
        batch=FkBatch(
            positions_world=np.zeros((1, 3), dtype=np.float64),
            rotations_world=np.eye(3, dtype=np.float64)[None],
        ),
    )
    with pytest.raises(ValueError):
        load_fk_result(result_dir, expected_request_sha256="0" * 64)


def test_fk_batch_rejects_non_rotation_matrices() -> None:
    """Break caught: finite 3x3 arrays are accepted even when they are not SO(3)."""
    from latency_meta_mdp.data.collection.planner_protocol import FkBatch

    with pytest.raises(ValueError, match="rotation"):
        FkBatch(
            positions_world=np.zeros((1, 3), dtype=np.float64),
            rotations_world=np.zeros((1, 3, 3), dtype=np.float64),
        )
