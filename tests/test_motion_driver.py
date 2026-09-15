from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from latency_meta_mdp.envs.backend import FormalStepExecutor, RoboSuitePlant
from latency_meta_mdp.envs.motion import DrivenBallWorld, build_motion_profile, load_motion_config
from latency_meta_mdp.envs.snapshots import BoundarySnapshotter
from latency_meta_mdp.envs.task import load_task_spec, make_dynamic_grasp_lift_environment
from latency_meta_mdp.runtime.timing import ClockLedger

_TASK_CONFIG = Path("configs/tasks/moving_ball/task/dynamic_grasp_lift_l0.yaml")
_MOTION_ROOT = Path("configs/tasks/moving_ball/motion")


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_motion_driver_matches_profile_without_changing_g1_clock(level: int) -> None:
    task_spec = load_task_spec(_TASK_CONFIG)
    motion_config = load_motion_config(_MOTION_ROOT / f"dynamic_grasp_lift_l{level}.yaml")
    profile = build_motion_profile(
        config=motion_config,
        seed=7,
        workspace_z=task_spec.ball_initial_position[2],
    )
    env = make_dynamic_grasp_lift_environment(spec=task_spec, seed=7, offscreen=False)
    try:
        snapshotter = BoundarySnapshotter(camera_names=(), width=16, height=16)
        world = DrivenBallWorld(profile=profile, motion_level=level)
        plant = RoboSuitePlant(env=env, snapshotter=snapshotter, world_writer=world)
        ledger = ClockLedger(physics_dt_us=2_000, formal_tick_us=20_000)
        executor = FormalStepExecutor(plant=plant, ledger=ledger)
        snapshots = [executor.initialize()]
        for _ in range(50):
            snapshots.append(executor.step_formal(np.zeros(env.action_dim)))
    finally:
        env.close()

    assert ledger.physics_step_index == 500
    assert ledger.formal_tick_index == 50
    assert plant.world_write_count == 501
    assert plant.step1_count == 501
    assert plant.step2_count == 500
    for snapshot in snapshots:
        expected = profile.sample(snapshot.time_us)
        np.testing.assert_allclose(snapshot.object_qpos[:3], expected.position, atol=1e-12)
        np.testing.assert_allclose(snapshot.commanded_world["target_position"], expected.position)
        np.testing.assert_allclose(snapshot.commanded_world["target_velocity"], expected.velocity)
        assert snapshot.commanded_world["motion_level"].item() == level
        assert snapshot.commanded_world["segment_index"].item() == expected.segment_index


def test_ungrasped_dynamic_motion_deadline_terminates_rollout() -> None:
    task_spec = load_task_spec(_TASK_CONFIG)
    motion_config = load_motion_config(_MOTION_ROOT / "dynamic_grasp_lift_l1.yaml")
    profile = build_motion_profile(
        config=motion_config,
        seed=7,
        workspace_z=task_spec.ball_initial_position[2],
    )
    env = make_dynamic_grasp_lift_environment(spec=task_spec, seed=7, offscreen=False)
    try:
        snapshotter = BoundarySnapshotter(camera_names=(), width=16, height=16)
        plant = RoboSuitePlant(
            env=env,
            snapshotter=snapshotter,
            world_writer=DrivenBallWorld(profile=profile, motion_level=1),
        )
        ledger = ClockLedger(physics_dt_us=2_000, formal_tick_us=20_000)
        executor = FormalStepExecutor(plant=plant, ledger=ledger)
        final_snapshot = executor.initialize()
        for _ in range(motion_config.anchor_time_us // 20_000):
            final_snapshot = executor.step_formal(np.zeros(env.action_dim))

        assert ledger.time_us == motion_config.anchor_time_us
        assert final_snapshot.commanded_world["motion_terminal"].item() is True
        assert env.done is True
        assert env.task_failure is True
        assert env.terminal_reason == "motion_deadline_ungrasped"
        assert env.motion_deadline_us == motion_config.anchor_time_us
        with pytest.raises(RuntimeError, match="formal plant is terminal"):
            executor.step_formal(np.zeros(env.action_dim))
        assert ledger.time_us == motion_config.anchor_time_us
    finally:
        env.close()
