from __future__ import annotations

import numpy as np
import pytest

from latency_meta_mdp.envs.backend import (
    FormalStepExecutor,
    RoboSuitePlant,
    instrument_stock_step,
    make_g1_environment,
)
from latency_meta_mdp.envs.snapshots import BoundarySnapshotter
from latency_meta_mdp.runtime.timing import ClockLedger


def test_stock_step_has_ten_substeps_one_goal_and_measured_observation_age() -> None:
    env = make_g1_environment(seed=7, offscreen=False)
    try:
        report = instrument_stock_step(
            env, np.zeros(env.action_dim), observable_name="robot0_joint_pos"
        )
    finally:
        env.close()

    assert report.step1_count == 10
    assert report.step2_count == 10
    assert report.control_refresh_count == 10
    assert report.goal_refresh_count == 1
    assert report.observable_update_count == 10
    assert report.policy_step_flags == (True,) + (False,) * 9
    assert report.start_time_seconds == 0.0
    assert report.end_time_seconds == 0.02
    assert report.latest_sample_time_seconds == 0.02
    assert report.returned_observation_age_seconds == 0.0


def test_split_step_snapshot_uses_direct_fresh_boundary_state_and_images() -> None:
    env = make_g1_environment(seed=7, offscreen=True)
    try:
        snapshotter = BoundarySnapshotter(
            camera_names=("agentview", "robot0_eye_in_hand"),
            segmentation_camera_names=("agentview",),
            width=64,
            height=64,
        )
        plant = RoboSuitePlant(env=env, snapshotter=snapshotter)
        ledger = ClockLedger(physics_dt_us=2_000, formal_tick_us=20_000)
        executor = FormalStepExecutor(plant=plant, ledger=ledger)

        initial = executor.initialize()
        initial_qpos = initial.qpos.copy()
        env._get_observations = lambda *args, **kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
            AssertionError("stock observation cache must not be used")
        )
        final = executor.step_formal(np.zeros(env.action_dim))
    finally:
        env.close()

    assert initial.physics_step_index == 0
    assert initial.formal_tick_index == 0
    assert initial.time_us == 0
    assert final.physics_step_index == 10
    assert final.formal_tick_index == 1
    assert final.time_us == 20_000
    assert final.sim_time_seconds == pytest.approx(0.02, abs=1e-12)
    assert set(final.cameras) == {"agentview", "robot0_eye_in_hand"}
    assert final.cameras["agentview"].rgb.shape == (64, 64, 3)
    assert final.cameras["robot0_eye_in_hand"].rgb.shape == (64, 64, 3)
    assert final.cameras["agentview"].segmentation.shape == (64, 64, 2)
    assert final.cameras["robot0_eye_in_hand"].segmentation is None
    assert all(sample.source_time_us == 20_000 for sample in final.cameras.values())
    assert all(sample.source_physics_step == 10 for sample in final.cameras.values())
    assert not final.qpos.flags.writeable
    assert final.robot_gripper_qpos.shape == (2,)
    assert final.robot_gripper_qvel.shape == (2,)
    assert final.object_qvel.shape == (6,)
    assert not final.robot_gripper_qpos.flags.writeable
    assert not final.robot_gripper_qvel.flags.writeable
    assert not final.object_qvel.flags.writeable
    assert not final.cameras["agentview"].rgb.flags.writeable
    np.testing.assert_array_equal(initial.qpos, initial_qpos)


def test_formal_backend_rejects_non_euler_integrator() -> None:
    import mujoco

    env = make_g1_environment(seed=7, offscreen=False)
    try:
        env.sim.model.opt.integrator = int(mujoco.mjtIntegrator.mjINT_RK4)
        snapshotter = BoundarySnapshotter(camera_names=(), width=16, height=16)
        with pytest.raises(ValueError, match="Euler integrator"):
            RoboSuitePlant(env=env, snapshotter=snapshotter)
    finally:
        env.close()
