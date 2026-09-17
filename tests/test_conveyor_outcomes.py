import pytest


def test_last_born_finishing_does_not_end_older_active_parcel():
    from latency_meta_mdp.envs.conveyor.outcomes import ParcelLedger

    ledger = ParcelLedger(supply_ticks=1500, drain_ticks=500)
    ledger.spawn(0, 0)
    ledger.spawn(1, 1499)
    ledger.finish(1, "success", 1500)
    assert not ledger.advance(1500)
    assert not ledger.advance(1999)
    assert ledger.advance(2000)
    assert ledger.statuses == {0: "timeout", 1: "success"}
    assert ledger.summary()["spawned"] == 2
    assert ledger.summary()["timeouts"] == 1
    with pytest.raises(ValueError):
        ledger.finish(1, "miss", 2000)


def test_no_early_episode_reset_and_no_birth_at_cutoff():
    from latency_meta_mdp.envs.conveyor.outcomes import ParcelLedger

    ledger = ParcelLedger(supply_ticks=1500, drain_ticks=500)
    ledger.spawn(0, 25)
    ledger.finish(0, "miss", 300)
    assert not ledger.advance(300)
    assert ledger.advance(1500)
    with pytest.raises(ValueError):
        ledger.spawn(1, 1500)
    assert ledger.summary()["success_rate"] == 0


def tracker():
    from latency_meta_mdp.envs.conveyor.outcomes import GoalDeliveryTracker

    return GoalDeliveryTracker(
        center=(0, 0, 1.0),
        radius=0.06,
        ball_radius=0.03,
        resting_height=0.84,
        min_lift=0.04,
        grasp_ticks=3,
        release_width_m=0.07,
        release_opening_delta_m=0.002,
    )


def hold_in_goal(goal):
    for tick in range(3):
        assert not goal.observe(
            0, tick, (0, 0, 0.97), grasped=True, opening=False, gripper_width=0.06
        )


def test_delivery_requires_actual_opening_and_loss_of_grasp_after_held_overlap():
    goal = tracker()
    hold_in_goal(goal)
    assert not goal.observe(0, 3, (0, 0, 0.97), grasped=True, opening=True, gripper_width=0.061)
    assert not goal.observe(0, 4, (0, 0, 0.97), grasped=False, opening=True, gripper_width=0.065)
    assert goal.observe(0, 5, (0, 0, 0.96), grasped=False, opening=True, gripper_width=0.072)
    assert not goal.observe(0, 6, (0, 0, 0.95), grasped=False, opening=True, gripper_width=0.078)


def test_slip_without_open_command_cannot_be_relabelled_as_delivery():
    goal = tracker()
    hold_in_goal(goal)
    assert not goal.observe(0, 3, (0, 0, 0.97), grasped=False, opening=False, gripper_width=0.06)
    assert not goal.observe(0, 4, (0, 0, 0.96), grasped=False, opening=True, gripper_width=0.08)


def test_opening_outside_goal_or_ungrasped_overlap_cannot_deliver():
    goal = tracker()
    hold_in_goal(goal)
    assert not goal.observe(0, 3, (0.2, 0, 0.97), grasped=True, opening=True, gripper_width=0.065)
    assert not goal.observe(0, 4, (0, 0, 0.96), grasped=False, opening=True, gripper_width=0.08)
    other = tracker()
    for tick in range(5):
        assert not other.observe(
            1, tick, (0, 0, 0.97), grasped=False, opening=True, gripper_width=0.08
        )


def test_sustained_grasp_below_lift_height_is_not_eligible():
    goal = tracker()
    for tick in range(3):
        assert not goal.observe(
            0, tick, (0, 0, 0.85), grasped=True, opening=False, gripper_width=0.06
        )
    assert not goal.observe(0, 3, (0, 0, 0.97), grasped=False, opening=True, gripper_width=0.08)
