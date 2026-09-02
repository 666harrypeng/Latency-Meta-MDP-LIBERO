"""Exact RoboSuite Panda to CuRobo Franka kinematic bridge."""

from __future__ import annotations

import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, fields
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _readonly(value: Any, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float64 or array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite float64{shape}")
    result = np.array(array, copy=True, order="C")
    result.setflags(write=False)
    return result


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _asset_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _canonical_json(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass(frozen=True, eq=False)
class PandaPlanningBridge:
    schema_version: int
    bridge_id: str
    joint_names: tuple[str, ...]
    joint_lower: np.ndarray
    joint_upper: np.ndarray
    joint_velocity: np.ndarray
    joint_acceleration: np.ndarray
    home_joint_positions: np.ndarray
    base_position_world: np.ndarray
    base_rotation_world: np.ndarray
    hand_to_tcp_position: np.ndarray
    hand_to_tcp_rotation: np.ndarray
    open_gripper_positions: np.ndarray
    robosuite_version: str
    curobo_version: str
    robosuite_urdf_sha256: str
    curobo_urdf_sha256: str
    curobo_config_sha256: str
    robosuite_asset_tree_sha256: str
    curobo_asset_tree_sha256: str
    curobo_tool_frame: str
    robosuite_eef_site: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("bridge schema_version must be integer 1")
        if self.bridge_id != "robosuite_panda_to_curobo_franka_v1":
            raise ValueError("unsupported Panda planning bridge")
        if type(self.joint_names) is not tuple or self.joint_names != _JOINT_NAMES:
            raise ValueError("Panda joint order is invalid")
        for name in (
            "robosuite_version",
            "curobo_version",
            "curobo_tool_frame",
            "robosuite_eef_site",
        ):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        if self.robosuite_version != "1.5.2" or self.curobo_version != "0.8.0":
            raise ValueError("robot runtime versions do not match the bridge")
        if self.curobo_tool_frame != "panda_hand" or (
            self.robosuite_eef_site != "gripper0_right_grip_site"
        ):
            raise ValueError("tool/TCP names do not match the bridge")
        for name in (
            "robosuite_urdf_sha256",
            "curobo_urdf_sha256",
            "curobo_config_sha256",
            "robosuite_asset_tree_sha256",
            "curobo_asset_tree_sha256",
        ):
            value = getattr(self, name)
            if type(value) is not str or _SHA256.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        contracts = {
            "joint_lower": (7,),
            "joint_upper": (7,),
            "joint_velocity": (7,),
            "joint_acceleration": (7,),
            "home_joint_positions": (7,),
            "base_position_world": (3,),
            "base_rotation_world": (3, 3),
            "hand_to_tcp_position": (3,),
            "hand_to_tcp_rotation": (3, 3),
            "open_gripper_positions": (2,),
        }
        for name, shape in contracts.items():
            object.__setattr__(self, name, _readonly(getattr(self, name), shape=shape, name=name))
        if not np.all(self.joint_lower < self.joint_upper):
            raise ValueError("joint position limits are invalid")
        if not np.all(self.joint_velocity > 0) or not np.all(self.joint_acceleration > 0):
            raise ValueError("joint dynamic limits must be positive")
        if not np.all(
            (self.home_joint_positions > self.joint_lower)
            & (self.home_joint_positions < self.joint_upper)
        ):
            raise ValueError("home joint position must lie strictly inside limits")
        for name in ("base_rotation_world", "hand_to_tcp_rotation"):
            rotation = getattr(self, name)
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9, rtol=0.0) or not (
                np.isclose(np.linalg.det(rotation), 1.0, atol=1e-9, rtol=0.0)
            ):
                raise ValueError(f"{name} must be a proper rotation")
        if not np.array_equal(self.open_gripper_positions, [0.04, 0.04]):
            raise ValueError("CuRobo open-gripper locks must be [0.04, 0.04]")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PandaPlanningBridge):
            return False
        return all(
            np.array_equal(getattr(self, item.name), getattr(other, item.name))
            if isinstance(getattr(self, item.name), np.ndarray)
            else getattr(self, item.name) == getattr(other, item.name)
            for item in fields(self)
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            item.name: (
                getattr(self, item.name).tolist()
                if isinstance(getattr(self, item.name), np.ndarray)
                else list(getattr(self, item.name))
                if item.name == "joint_names"
                else getattr(self, item.name)
            )
            for item in fields(self)
        }

    @classmethod
    def from_mapping(cls, mapping: Any) -> PandaPlanningBridge:
        if type(mapping) is not dict or set(mapping) != {item.name for item in fields(cls)}:
            raise ValueError("PandaPlanningBridge fields are invalid")
        raw = dict(mapping)
        raw["joint_names"] = tuple(raw["joint_names"])
        for name in (
            "joint_lower",
            "joint_upper",
            "joint_velocity",
            "joint_acceleration",
            "home_joint_positions",
            "base_position_world",
            "base_rotation_world",
            "hand_to_tcp_position",
            "hand_to_tcp_rotation",
            "open_gripper_positions",
        ):
            raw[name] = np.asarray(raw[name], dtype=np.float64)
        return cls(**raw)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_mapping())).hexdigest()


def _parse_urdf_limits(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = ET.fromstring(path.read_bytes())
    joints = {item.attrib.get("name"): item for item in root.findall("joint")}
    lower, upper, velocity = [], [], []
    for name in _JOINT_NAMES:
        if name not in joints:
            raise ValueError(f"URDF is missing {name}")
        limit = joints[name].find("limit")
        if limit is None:
            raise ValueError(f"URDF joint {name} is missing limits")
        lower.append(float(limit.attrib["lower"]))
        upper.append(float(limit.attrib["upper"]))
        velocity.append(float(limit.attrib["velocity"]))
    return tuple(np.asarray(row, dtype=np.float64) for row in (lower, upper, velocity))


def _robot_asset_paths(project_root: Path) -> dict[str, Path]:
    import robosuite

    robosuite_root = Path(robosuite.__file__).resolve().parent
    robosuite_assets = robosuite_root / "models/assets/bullet_data/panda_description"
    curobo_assets = (
        project_root / "third_party/curobo/curobo/content/assets/robot/franka_description"
    )
    return {
        "robosuite_assets": robosuite_assets,
        "robosuite_urdf": robosuite_assets / "urdf/panda_arm_hand.urdf",
        "curobo_assets": curobo_assets,
        "curobo_urdf": curobo_assets / "franka_panda.urdf",
        "curobo_config": project_root
        / "third_party/curobo/curobo/content/configs/robot/franka.yml",
    }


def verify_panda_planning_bridge_assets(
    bridge: PandaPlanningBridge, *, project_root: Path
) -> None:
    if not isinstance(bridge, PandaPlanningBridge):
        raise TypeError("bridge must be a PandaPlanningBridge")
    paths = _robot_asset_paths(Path(project_root).resolve())
    expected = {
        "robosuite_urdf_sha256": _sha256_file(paths["robosuite_urdf"]),
        "curobo_urdf_sha256": _sha256_file(paths["curobo_urdf"]),
        "curobo_config_sha256": _sha256_file(paths["curobo_config"]),
        "robosuite_asset_tree_sha256": _asset_tree_sha256(paths["robosuite_assets"]),
        "curobo_asset_tree_sha256": _asset_tree_sha256(paths["curobo_assets"]),
    }
    mismatches = [name for name, digest in expected.items() if getattr(bridge, name) != digest]
    if mismatches:
        raise ValueError(f"Panda planning bridge asset mismatch: {mismatches}")


def build_panda_planning_bridge(project_root: Path) -> PandaPlanningBridge:
    """Derive the base/TCP bridge from one fresh certified RoboSuite runtime."""
    from latency_meta_mdp.expert_realization.task_instance import (
        _build_task_instance_runtime,
        materialize_task_instance,
    )

    root = Path(project_root).resolve()
    paths = _robot_asset_paths(root)
    lower, upper, velocity = _parse_urdf_limits(paths["robosuite_urdf"])
    curobo_mapping = yaml.safe_load(paths["curobo_config"].read_text(encoding="utf-8"))
    cspace = curobo_mapping["robot_cfg"]["kinematics"]["cspace"]
    if tuple(cspace["joint_names"][:7]) != _JOINT_NAMES:
        raise ValueError("CuRobo joint order does not match RoboSuite Panda")
    acceleration = np.full(7, float(cspace["max_acceleration"]), dtype=np.float64)
    locks = curobo_mapping["robot_cfg"]["kinematics"]["lock_joints"]
    open_gripper = np.array(
        [locks["panda_finger_joint1"], locks["panda_finger_joint2"]], dtype=np.float64
    )
    task = materialize_task_instance(project_root=root, level=1, task_instance_seed=4000)
    runtime = _build_task_instance_runtime(task)
    try:
        snapshot = runtime.executor.initialize()
        robot = runtime.env.robots[0]
        model, data = runtime.env.sim.model, runtime.env.sim.data
        base_id = model.body_name2id(robot.robot_model.root_body)
        hand_id = model.body_name2id("robot0_right_hand")
        site_id = robot.eef_site_id[robot.arms[0]]
        site_name = model.site_id2name(site_id)
        base_position = np.asarray(data.body_xpos[base_id], dtype=np.float64)
        base_rotation = np.asarray(data.body_xmat[base_id].reshape(3, 3), dtype=np.float64)
        hand_position = np.asarray(data.body_xpos[hand_id], dtype=np.float64)
        hand_rotation = np.asarray(data.body_xmat[hand_id].reshape(3, 3), dtype=np.float64)
        tcp_position = np.asarray(data.site_xpos[site_id], dtype=np.float64)
        tcp_rotation = np.asarray(data.site_xmat[site_id].reshape(3, 3), dtype=np.float64)
        hand_to_tcp_position = hand_rotation.T @ (tcp_position - hand_position)
        hand_to_tcp_rotation = hand_rotation.T @ tcp_rotation
        if not np.allclose(base_position, [-0.56, 0.0, 0.912], atol=1e-9, rtol=0.0):
            raise ValueError("RoboSuite Panda base translation drifted")
        if not np.allclose(base_rotation, np.eye(3), atol=1e-9, rtol=0.0):
            raise ValueError("RoboSuite Panda base rotation drifted")
        if not np.allclose(hand_to_tcp_position, [0.0, 0.0, 0.097], atol=1e-9, rtol=0.0):
            raise ValueError("RoboSuite hand-to-TCP translation drifted")
        expected_tcp_rotation = np.array(
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        if not np.allclose(
            hand_to_tcp_rotation, expected_tcp_rotation, atol=1e-9, rtol=0.0
        ):
            raise ValueError("RoboSuite hand-to-TCP rotation drifted")
        home = np.asarray(snapshot.robot_qpos, dtype=np.float64)
    finally:
        runtime.close()
    return PandaPlanningBridge(
        schema_version=1,
        bridge_id="robosuite_panda_to_curobo_franka_v1",
        joint_names=_JOINT_NAMES,
        joint_lower=lower,
        joint_upper=upper,
        joint_velocity=velocity,
        joint_acceleration=acceleration,
        home_joint_positions=home,
        base_position_world=base_position,
        base_rotation_world=base_rotation,
        hand_to_tcp_position=hand_to_tcp_position,
        hand_to_tcp_rotation=hand_to_tcp_rotation,
        open_gripper_positions=open_gripper,
        robosuite_version=version("robosuite"),
        curobo_version="0.8.0",
        robosuite_urdf_sha256=_sha256_file(paths["robosuite_urdf"]),
        curobo_urdf_sha256=_sha256_file(paths["curobo_urdf"]),
        curobo_config_sha256=_sha256_file(paths["curobo_config"]),
        robosuite_asset_tree_sha256=_asset_tree_sha256(paths["robosuite_assets"]),
        curobo_asset_tree_sha256=_asset_tree_sha256(paths["curobo_assets"]),
        curobo_tool_frame="panda_hand",
        robosuite_eef_site=site_name,
    )


def sample_fk_parity_set(
    bridge: PandaPlanningBridge, *, interior_count: int = 16
) -> np.ndarray:
    if not isinstance(bridge, PandaPlanningBridge):
        raise TypeError("bridge must be a PandaPlanningBridge")
    if type(interior_count) is not int or interior_count <= 0:
        raise ValueError("interior_count must be positive")
    rows = [np.array(bridge.home_joint_positions, copy=True)]
    rng = np.random.default_rng(np.random.SeedSequence([2026, 9, 1]))
    margin = 0.05 * (bridge.joint_upper - bridge.joint_lower)
    interior_lower = bridge.joint_lower + margin
    interior_upper = bridge.joint_upper - margin
    rows.extend(rng.uniform(interior_lower, interior_upper, size=(interior_count, 7)))
    for joint_index in range(7):
        lower = np.array(bridge.home_joint_positions, copy=True)
        upper = np.array(bridge.home_joint_positions, copy=True)
        lower[joint_index] = interior_lower[joint_index]
        upper[joint_index] = interior_upper[joint_index]
        rows.extend((lower, upper))
    samples = np.asarray(rows, dtype=np.float64)
    if len(np.unique(samples, axis=0)) != len(samples):
        raise RuntimeError("FK parity set contains duplicate rows")
    samples.setflags(write=False)
    return samples


def compute_robosuite_fk(
    bridge: PandaPlanningBridge, qpos: np.ndarray, *, project_root: Path
):
    """Evaluate policy-EEF world FK through one fresh MuJoCo model."""
    from latency_meta_mdp.expert_realization.planner_protocol import FkBatch
    from latency_meta_mdp.expert_realization.task_instance import (
        _build_task_instance_runtime,
        materialize_task_instance,
    )

    values = np.asarray(qpos)
    if values.dtype != np.float64 or values.ndim != 2 or values.shape[1] != 7:
        raise ValueError("qpos must be float64[N,7]")
    if np.any(values <= bridge.joint_lower) or np.any(values >= bridge.joint_upper):
        raise ValueError("qpos lies outside the bridge joint limits")
    verify_panda_planning_bridge_assets(bridge, project_root=project_root)
    task = materialize_task_instance(project_root=project_root, level=1, task_instance_seed=4000)
    runtime = _build_task_instance_runtime(task)
    positions, rotations = [], []
    try:
        runtime.executor.initialize()
        robot = runtime.env.robots[0]
        indexes = np.asarray(robot._ref_joint_pos_indexes, dtype=int)
        site_id = robot.eef_site_id[robot.arms[0]]
        for row in values:
            runtime.env.sim.data.qpos[indexes] = row
            runtime.env.sim.forward()
            positions.append(np.array(runtime.env.sim.data.site_xpos[site_id], copy=True))
            rotations.append(
                np.array(runtime.env.sim.data.site_xmat[site_id].reshape(3, 3), copy=True)
            )
    finally:
        runtime.close()
    return FkBatch(
        positions_world=np.asarray(positions, dtype=np.float64),
        rotations_world=np.asarray(rotations, dtype=np.float64),
    )


@dataclass(frozen=True)
class FkParityReport:
    sample_count: int
    max_translation_error_m: float
    max_rotation_error_degrees: float
    joint_order_matches: bool
    base_transform_matches: bool
    tcp_transform_matches: bool
    eligible: bool


def qualify_fk_parity(
    bridge: PandaPlanningBridge, robosuite_fk: Any, curobo_fk: Any
) -> FkParityReport:
    if robosuite_fk.positions_world.shape != curobo_fk.positions_world.shape or (
        robosuite_fk.rotations_world.shape != curobo_fk.rotations_world.shape
    ):
        raise ValueError("FK batch shapes do not match")
    translation = np.linalg.norm(
        robosuite_fk.positions_world - curobo_fk.positions_world, axis=1
    )
    angles = []
    for left, right in zip(
        robosuite_fk.rotations_world, curobo_fk.rotations_world, strict=True
    ):
        relative = left.T @ right
        cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
        angles.append(math.degrees(math.acos(float(cosine))))
    max_translation = float(np.max(translation, initial=0.0))
    max_rotation = float(np.max(angles, initial=0.0))
    joint_order_matches = bridge.joint_names == _JOINT_NAMES
    base_transform_matches = np.allclose(
        bridge.base_position_world, [-0.56, 0.0, 0.912], atol=1e-9, rtol=0.0
    ) and np.allclose(bridge.base_rotation_world, np.eye(3), atol=1e-9, rtol=0.0)
    tcp_transform_matches = np.allclose(
        bridge.hand_to_tcp_position, [0.0, 0.0, 0.097], atol=1e-9, rtol=0.0
    ) and np.allclose(
        bridge.hand_to_tcp_rotation,
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        atol=1e-9,
        rtol=0.0,
    )
    eligible = (
        max_translation <= 0.001
        and max_rotation <= 0.5
        and joint_order_matches
        and base_transform_matches
        and tcp_transform_matches
    )
    return FkParityReport(
        sample_count=len(translation),
        max_translation_error_m=max_translation,
        max_rotation_error_degrees=max_rotation,
        joint_order_matches=joint_order_matches,
        base_transform_matches=base_transform_matches,
        tcp_transform_matches=tcp_transform_matches,
        eligible=eligible,
    )
