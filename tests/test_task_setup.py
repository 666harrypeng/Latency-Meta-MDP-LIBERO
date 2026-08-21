from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from latency_meta_mdp.snapshots import BoundarySnapshotter
from latency_meta_mdp.task import load_task_spec, make_dynamic_grasp_lift_environment
from latency_meta_mdp.timing import ClockLedger

_TASK_CONFIG = Path("configs/task/dynamic_grasp_lift_l0.yaml")
_CAMERA_RESOURCE = Path("assets/camera/libero_tabletop_agentview_v1.json")
_LIBERO_AGENTVIEW_POS = np.array([0.6586131746834771, 0.0, 1.6103500240372423])
_LIBERO_AGENTVIEW_QUAT = np.array(
    [0.6380177212293417, 0.30484971880648903, 0.30484983801576876, 0.6380177212293417]
)


def _object_pixel_count(env, segmentation: np.ndarray) -> int:
    geom_ids = {
        env.sim.model.geom_name2id(name)
        for name in (*env.ball.visual_geoms, *env.ball.contact_geoms)
    }
    mask = (segmentation[:, :, 0] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(
        segmentation[:, :, 1], list(geom_ids)
    )
    return int(mask.sum())


def test_task_spec_is_explicit_and_uses_two_policy_cameras() -> None:
    spec = load_task_spec(_TASK_CONFIG)
    camera_resource = json.loads(_CAMERA_RESOURCE.read_text())

    assert spec.task_id == "dynamic_grasp_lift"
    assert spec.level == 0
    assert spec.instruction == "Grasp the ball and lift it."
    assert spec.ball_radius_m == 0.03
    assert spec.ball_diameter_m == 0.06
    assert spec.agentview_resource_id == "libero_tabletop_agentview_v1"
    assert spec.policy_camera_names == ("agentview", "robot0_eye_in_hand")
    np.testing.assert_allclose(spec.agentview_position, _LIBERO_AGENTVIEW_POS, atol=0, rtol=0)
    np.testing.assert_allclose(
        spec.agentview_quaternion_wxyz, _LIBERO_AGENTVIEW_QUAT, atol=0, rtol=0
    )
    assert spec.agentview_fovy_degrees == 45.0
    np.testing.assert_allclose(spec.agentview_position, camera_resource["position"], atol=0, rtol=0)
    np.testing.assert_allclose(
        spec.agentview_quaternion_wxyz,
        camera_resource["normalized_quaternion_wxyz"],
        atol=0,
        rtol=0,
    )


def test_ball_setup_is_deterministic_and_uses_native_robosuite_geometry() -> None:
    spec = load_task_spec(_TASK_CONFIG)
    env_a = make_dynamic_grasp_lift_environment(spec=spec, seed=7, offscreen=False)
    env_b = make_dynamic_grasp_lift_environment(spec=spec, seed=7, offscreen=False)
    try:
        assert len(env_a.robots) == 1
        assert type(env_a.robots[0].robot_model).__name__ == "Panda"
        assert env_a.robots[0].arms == ["right"]
        assert env_a.task_object is env_a.ball
        assert env_a.ball.horizontal_radius == spec.ball_radius_m
        np.testing.assert_allclose(
            env_a.ball.get_bounding_box_half_size(),
            [spec.ball_radius_m] * 3,
            atol=0,
            rtol=0,
        )
        np.testing.assert_array_equal(
            env_a.sim.data.get_joint_qpos(env_a.ball.joints[0]),
            env_b.sim.data.get_joint_qpos(env_b.ball.joints[0]),
        )
        np.testing.assert_allclose(
            env_a.sim.data.body_xpos[env_a.ball_body_id],
            spec.ball_initial_position,
            atol=1e-12,
        )
        assert env_a.ball.important_sites == {"obj": "ball_default_site"}
        assert env_a.current_grasp() is False
        assert env_a.backend_goal_reached() is False
    finally:
        env_a.close()
        env_b.close()


def test_official_agentview_and_wrist_see_ball_at_same_boundary() -> None:
    spec = load_task_spec(_TASK_CONFIG)
    env = make_dynamic_grasp_lift_environment(spec=spec, seed=7, offscreen=True)
    try:
        agent_camera_id = env.sim.model.camera_name2id("agentview")
        np.testing.assert_allclose(
            env.sim.model.cam_pos[agent_camera_id], spec.agentview_position, atol=1e-12
        )
        np.testing.assert_allclose(
            env.sim.model.cam_quat[agent_camera_id],
            spec.agentview_quaternion_wxyz,
            atol=1e-12,
        )
        snapshotter = BoundarySnapshotter(
            camera_names=spec.policy_camera_names,
            segmentation_camera_names=spec.policy_camera_names,
            width=256,
            height=256,
        )
        snapshot = snapshotter.capture(
            env=env,
            ledger=ClockLedger(physics_dt_us=2_000, formal_tick_us=20_000),
            commanded_world={},
        )
        object_pixel_counts = {
            camera_name: _object_pixel_count(env, snapshot.cameras[camera_name].segmentation)
            for camera_name in spec.policy_camera_names
        }
    finally:
        env.close()

    for camera_name in spec.policy_camera_names:
        sample = snapshot.cameras[camera_name]
        assert sample.rgb.shape == (256, 256, 3)
        assert sample.segmentation is not None
        assert object_pixel_counts[camera_name] >= 25


def test_task_spec_rejects_unknown_fields(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text(_TASK_CONFIG.read_text() + "unknown_field: true\n")

    with pytest.raises(ValueError, match="unknown task config fields"):
        load_task_spec(config)


def test_backend_goal_requires_grasp_and_lift_threshold() -> None:
    spec = load_task_spec(_TASK_CONFIG)
    env = make_dynamic_grasp_lift_environment(spec=spec, seed=7, offscreen=False)
    try:
        resting_center_z = spec.table_offset[2] + spec.ball_radius_m

        def set_ball_height(lift_height_m: float) -> None:
            qpos = np.array(env.sim.data.get_joint_qpos(env.ball.joints[0]), copy=True)
            qpos[2] = resting_center_z + lift_height_m
            env.sim.data.set_joint_qpos(env.ball.joints[0], qpos)
            env.sim.data.set_joint_qvel(env.ball.joints[0], np.zeros(6))
            env.sim.forward()

        env.current_grasp = lambda: False
        set_ball_height(spec.lift_success_height_m + 0.01)
        assert env.backend_goal_reached() is False

        env.current_grasp = lambda: True
        set_ball_height(spec.lift_success_height_m - 1e-6)
        assert env.backend_goal_reached() is False

        set_ball_height(spec.lift_success_height_m)
        assert env.backend_goal_reached() is True
    finally:
        env.close()


def test_motion_deadline_does_not_fail_an_already_grasped_ball() -> None:
    spec = load_task_spec(_TASK_CONFIG)
    env = make_dynamic_grasp_lift_environment(spec=spec, seed=7, offscreen=False)
    try:
        env.current_grasp = lambda: True
        env.mark_motion_deadline(3_000_000)
        env.evaluate_boundary_terminal(3_000_000)
        assert env.done is False
        assert env.task_failure is False
        assert env.terminal_reason is None
    finally:
        env.close()
