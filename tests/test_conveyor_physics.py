from dataclasses import replace
from pathlib import Path

import numpy as np


def make_runtime():
    from latency_meta_mdp.envs.conveyor.config import load_conveyor_spec
    from latency_meta_mdp.envs.conveyor.task import make_conveyor_runtime

    spec = load_conveyor_spec(Path("configs/tasks/conveyor_sort/surface.yaml"))
    spec = replace(spec, spawn_count=None, supply_ticks=100, drain_ticks=500)
    return make_conveyor_runtime(spec, seed=7, offscreen=False)


def test_contact_transport_is_physical_and_stops_when_airborne():
    runtime = make_runtime()
    try:
        runtime.executor.initialize()
        hold = np.array([0, 0, 0, 0, 0, 0, -1.0])
        for _ in range(100):
            runtime.executor.step_formal(hold)
        world, env = runtime.world, runtime.env
        event = world.arrivals[0]
        pos = world.position(0).copy()
        assert pos[1] > event.xy[1] + 0.05
        assert pos[2] > runtime.spec.belt_top_z
        assert world.drive_invocations > 0
        # No obstacle-free projection to a scripted position: an airborne ball
        # obeys gravity and keeps its own perturbed horizontal velocity.
        joint = env.parcels[0].joints[0]
        elevated = np.array(env.sim.data.get_joint_qpos(joint), copy=True)
        elevated[2] += 0.15
        env.sim.data.set_joint_qpos(joint, elevated)
        env.sim.data.set_joint_qvel(joint, [0.03, -0.02, 0, 0, 0, 0])
        env.sim.forward()
        runtime.executor.step_formal(hold)
        after = world.position(0)
        assert after[2] < elevated[2] and after[1] < elevated[1]
        assert np.linalg.norm(env.sim.data.xfrc_applied[env.parcel_body_ids[0]]) == 0
        assert np.isclose(env.sim.data.time, runtime.executor.ledger.time_us / 1e6)
    finally:
        runtime.env.close()


def test_retirement_disables_contact_and_no_gravity_drift_hidden_objects():
    runtime = make_runtime()
    try:
        runtime.executor.initialize()
        env = runtime.env
        hold = np.array([0, 0, 0, 0, 0, 0, -1.0])
        for _ in range(30):
            runtime.executor.step_formal(hold)
        runtime.world.ledger.finish(0, "miss", 30)
        env.park(0)
        geom_ids = env.parcel_geom_ids[0]
        assert not env.sim.model.geom_contype[geom_ids].any()
        assert not env.sim.model.geom_conaffinity[geom_ids].any()
        assert not env.sim.model.geom_rgba[geom_ids, 3].any()
        env.sim.forward()
        parked = runtime.world.position(0).copy()
        for _ in range(5):
            runtime.executor.step_formal(hold)
        np.testing.assert_allclose(runtime.world.position(0), parked, atol=1e-9)
        assert len(runtime.world.ledger.statuses) == 1
    finally:
        runtime.env.close()


def test_cameras_and_ball_properties_are_inherited_and_replay_is_deterministic():
    traces = []
    for _ in range(2):
        runtime = make_runtime()
        try:
            env, scene = runtime.env, runtime.spec.scene
            i = env.sim.model.camera_name2id("agentview")
            np.testing.assert_allclose(
                env.sim.model.cam_pos[i], runtime.spec.agentview_position, atol=1e-15
            )
            np.testing.assert_allclose(
                env.sim.model.cam_quat[i], scene.agentview_quaternion_wxyz, atol=1e-15
            )
            assert env.sim.model.cam_fovy[i] == scene.agentview_fovy_degrees
            assert env.parcels[0].horizontal_radius == scene.ball_radius_m
            np.testing.assert_allclose(
                env.sim.model.body_mass[env.parcel_body_ids[0]],
                4 / 3 * np.pi * scene.ball_radius_m**3 * scene.ball_density_kg_m3,
            )
            runtime.executor.initialize()
            rows = []
            for t in range(100):
                runtime.executor.step_formal(np.array([0, 0, 0, 0, 0, 0, -1.0]))
                if t >= 25:
                    rows.append(runtime.world.position(0))
            traces.append(np.array(rows))
        finally:
            runtime.env.close()
    np.testing.assert_array_equal(*traces)


def test_upstream_extension_preserves_robot_downstream_and_wrist():
    from latency_meta_mdp.envs.conveyor.config import load_conveyor_spec
    from latency_meta_mdp.envs.conveyor.task import make_conveyor_runtime

    spec = load_conveyor_spec(Path("configs/tasks/conveyor_sort/surface.yaml"))
    reference = replace(
        spec,
        upstream_extension_m=0.0,
        agentview_retreat_m=0.0,
        agentview_upstream_shift_m=0.0,
        spawn_y=(-0.34, -0.30),
    )
    records = []
    for config in (reference, spec):
        runtime = make_conveyor_runtime(
            replace(config, spawn_count=None, supply_ticks=100), seed=7, offscreen=False
        )
        env = runtime.env
        try:
            records.append(
                {
                    "robot": env.sim.model.body_pos[
                        env.sim.model.body_name2id("robot0_base")
                    ].copy(),
                    "wrist_pos": env.sim.model.cam_pos[
                        env.sim.model.camera_name2id("robot0_eye_in_hand")
                    ].copy(),
                    "wrist_quat": env.sim.model.cam_quat[
                        env.sim.model.camera_name2id("robot0_eye_in_hand")
                    ].copy(),
                    "edges": env.sim.data.geom_xpos[env.belt_geom_id, 1]
                    + np.array([-1, 1]) * env.sim.model.geom_size[env.belt_geom_id, 1],
                }
            )
            table = env.sim.model.geom_name2id("table_collision")
            np.testing.assert_allclose(
                env.sim.data.geom_xpos[table, 1] + env.sim.model.geom_size[table, 1], 0.4
            )
            assert "conveyor_spawn" not in env.sim.model.site_names
            goal = env.sim.model.site_name2id("conveyor_goal")
            assert env.sim.model.site_size[goal, 0] < 0.05
            np.testing.assert_allclose(env.sim.model.site_rgba[goal], spec.goal_rgba, atol=1e-7)
        finally:
            env.close()
    for key in ("robot", "wrist_pos", "wrist_quat"):
        np.testing.assert_array_equal(records[0][key], records[1][key])
    assert np.isclose(records[0]["edges"][1], records[1]["edges"][1])
    assert np.isclose(records[0]["edges"][0] - records[1]["edges"][0], spec.upstream_extension_m)


def test_measured_gripper_width_reaches_the_release_threshold():
    runtime = make_runtime()
    try:
        runtime.executor.initialize()
        env, robot = runtime.env, runtime.env.robots[0]
        indices = robot._ref_gripper_joint_pos_indexes[robot.arms[0]]
        for _ in range(30):
            runtime.executor.step_formal(np.array([0, 0, 0, 0, 0, 0, 1.0]))
        fingers = env.sim.data.qpos[indices]
        closed_width = float(fingers[0] - fingers[1])
        for _ in range(20):
            runtime.executor.step_formal(np.array([0, 0, 0, 0, 0, 0, -1.0]))
        fingers = env.sim.data.qpos[indices]
        opened_width = float(fingers[0] - fingers[1])
        assert opened_width >= runtime.spec.release_width_m
        assert opened_width - closed_width >= runtime.spec.release_opening_delta_m
        assert runtime.world.ledger.summary()["successes"] == 0
    finally:
        runtime.env.close()


def test_conveyor_control_profile_is_explicit_and_does_not_change_moving_ball():
    from latency_meta_mdp.envs.control import load_action_contract

    runtime = make_runtime()
    try:
        old = load_action_contract(Path("configs/runtime/control/panda_osc_pose_delta_v1.yaml"))
        assert old.kp == 150.0
        assert runtime.action_contract.contract_id == "panda_osc_pose_delta_conveyor_v2"
        assert runtime.action_contract.kp == 300.0
        assert not runtime.action_contract.uncouple_position_orientation
        previous = load_action_contract(
            Path("configs/legacy/conveyor_sort/panda_osc_pose_delta_conveyor_v1.yaml")
        )
        assert previous.kp == 300.0 and previous.uncouple_position_orientation
        np.testing.assert_array_equal(runtime.action_contract.arm_output_high, old.arm_output_high)
        runtime.action_contract.verify_runtime(runtime.env)
    finally:
        runtime.env.close()
