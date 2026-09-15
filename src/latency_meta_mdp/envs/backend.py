"""Project-owned split-step execution independent of stock ``env.step``."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from latency_meta_mdp.envs.snapshots import BoundarySnapshotter
from latency_meta_mdp.runtime.timing import ClockLedger


class SplitStepPlant(Protocol):
    """Minimal simulator surface consumed by the formal executor."""

    sim_time_seconds: float

    def write_world_at(self, current_time_us: int) -> None: ...

    def step1(self) -> None: ...

    def materialize_boundary(self, ledger: ClockLedger) -> Any: ...

    def apply_control(self, action: object, *, policy_step: bool) -> None: ...

    def step2(self) -> None: ...


class FormalStepExecutor:
    """Advance exactly one formal interval while preserving split-step order."""

    def __init__(self, *, plant: SplitStepPlant, ledger: ClockLedger) -> None:
        self.plant = plant
        self.ledger = ledger
        self.initialized = False
        self.faulted = False
        self._boundary_prepared = False

    def initialize(self) -> Any:
        if self.initialized:
            raise RuntimeError("formal executor is already initialized")
        if self.faulted:
            raise RuntimeError("formal executor is faulted")
        try:
            self.ledger.validate_sim_time(self.plant.sim_time_seconds)
            snapshot = self._prepare_boundary()
        except BaseException:
            self.faulted = True
            raise
        self.initialized = True
        return snapshot

    def step_formal(self, action: object) -> Any:
        if self.faulted:
            raise RuntimeError("formal executor is faulted")
        if not self.initialized:
            raise RuntimeError("formal executor is not initialized")
        if getattr(self.plant, "formal_terminal", False):
            raise RuntimeError("formal plant is terminal")
        if not self._boundary_prepared:
            raise RuntimeError("formal boundary is not prepared")

        try:
            for substep in range(self.ledger.physics_steps_per_tick):
                if substep:
                    self._prepare_physics_point()
                self.plant.apply_control(action, policy_step=substep == 0)
                self.plant.step2()
                self._boundary_prepared = False
                self.ledger.record_successful_step(sim_time_seconds=self.plant.sim_time_seconds)
            if not self.ledger.at_formal_boundary:
                raise RuntimeError("formal step ended away from a formal boundary")
            return self._prepare_boundary()
        except BaseException:
            self.faulted = True
            self._boundary_prepared = False
            raise

    def _prepare_physics_point(self) -> None:
        self.plant.write_world_at(self.ledger.time_us)
        self.plant.step1()

    def _prepare_boundary(self) -> Any:
        self._prepare_physics_point()
        snapshot = self.plant.materialize_boundary(self.ledger)
        self._boundary_prepared = True
        return snapshot


WorldWriter = Callable[[Any, int], Mapping[str, Any] | None]


@dataclass(frozen=True)
class AppliedControlSample:
    physics_step_index: int
    formal_tick_index: int
    time_us: int
    policy_step: bool
    action: np.ndarray
    actuator_ctrl: np.ndarray
    robot_qpos: np.ndarray

    def __post_init__(self) -> None:
        for name in ("action", "actuator_ctrl", "robot_qpos"):
            value = np.array(getattr(self, name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


ControlObserver = Callable[[AppliedControlSample], None]


@dataclass(frozen=True)
class PreparedPhysicsPoint:
    physics_step_index: int
    formal_tick_index: int
    time_us: int
    at_formal_boundary: bool


PhysicsPointObserver = Callable[[PreparedPhysicsPoint], Mapping[str, Any] | None]


@dataclass(frozen=True)
class CompletedPhysicsStep:
    physics_step_index: int
    formal_tick_index: int
    time_us: int
    at_formal_boundary: bool


CompletedPhysicsStepObserver = Callable[[CompletedPhysicsStep], None]


class RoboSuitePlant:
    """Narrow adapter around a mounted-Panda RoboSuite environment."""

    def __init__(
        self,
        *,
        env: Any,
        snapshotter: BoundarySnapshotter,
        world_writer: WorldWriter | None = None,
        control_observer: ControlObserver | None = None,
        physics_point_observer: PhysicsPointObserver | None = None,
        completed_physics_step_observer: CompletedPhysicsStepObserver | None = None,
    ) -> None:
        import mujoco

        if not env.lite_physics:
            raise ValueError("RoboSuitePlant requires lite_physics split stepping")
        if abs(float(env.sim.model.opt.timestep) - 0.002) > 1e-12:
            raise ValueError("compiled MuJoCo timestep must be 0.002 seconds")
        if int(env.sim.model.opt.integrator) != int(mujoco.mjtIntegrator.mjINT_EULER):
            raise ValueError("RoboSuitePlant requires the Euler integrator")
        if mujoco.get_mjcb_control() is not None:
            raise ValueError("external MuJoCo control callback is not allowed")
        self._env = env
        self._snapshotter = snapshotter
        self._world_writer = world_writer
        self._control_observer = control_observer
        self._physics_point_observer = physics_point_observer
        self._completed_physics_step_observer = completed_physics_step_observer
        self._commanded_world: Mapping[str, Any] = {}
        self.world_write_count = 0
        self.step1_count = 0
        self.step2_count = 0
        self.boundary_count = 0
        self.control_refresh_count = 0
        self.goal_refresh_count = 0

    @property
    def sim_time_seconds(self) -> float:
        return float(self._env.sim.data.time)

    @property
    def formal_terminal(self) -> bool:
        return bool(self._env.done)

    def write_world_at(self, current_time_us: int) -> None:
        if self._world_writer is None:
            self._commanded_world = {}
        else:
            values = self._world_writer(self._env, current_time_us)
            self._commanded_world = {} if values is None else values
        self.world_write_count += 1

    def step1(self) -> None:
        self._env.sim.step1()
        if self._physics_point_observer is not None:
            time_us = round(float(self._env.sim.data.time) * 1_000_000)
            physics_step_index = time_us // 2_000
            updates = self._physics_point_observer(
                PreparedPhysicsPoint(
                    physics_step_index=physics_step_index,
                    formal_tick_index=physics_step_index // 10,
                    time_us=time_us,
                    at_formal_boundary=physics_step_index % 10 == 0,
                )
            )
            if updates is not None:
                overlap = set(self._commanded_world) & set(updates)
                if overlap:
                    raise ValueError(f"physics observer overwrote world fields: {sorted(overlap)}")
                self._commanded_world = {**self._commanded_world, **updates}
        self.step1_count += 1

    def materialize_boundary(self, ledger: ClockLedger) -> Any:
        snapshot = self._snapshotter.capture(
            env=self._env,
            ledger=ledger,
            commanded_world=self._commanded_world,
        )
        evaluate_terminal = getattr(self._env, "evaluate_boundary_terminal", None)
        if callable(evaluate_terminal):
            evaluate_terminal(ledger.time_us)
        self.boundary_count += 1
        return snapshot

    def apply_control(self, action: object, *, policy_step: bool) -> None:
        action_vector = np.asarray(action, dtype=float)
        self._env._pre_action(action_vector, policy_step=policy_step)
        if self._control_observer is not None:
            time_us = round(float(self._env.sim.data.time) * 1_000_000)
            robot = self._env.robots[0]
            robot_qpos_indexes = np.asarray(robot._ref_joint_pos_indexes, dtype=int)
            self._control_observer(
                AppliedControlSample(
                    physics_step_index=time_us // 2_000,
                    formal_tick_index=time_us // 20_000,
                    time_us=time_us,
                    policy_step=policy_step,
                    action=action_vector,
                    actuator_ctrl=self._env.sim.data.ctrl,
                    robot_qpos=self._env.sim.data.qpos[robot_qpos_indexes],
                )
            )
        self.control_refresh_count += 1
        self.goal_refresh_count += int(policy_step)

    def step2(self) -> None:
        self._env.sim.step2()
        if self._completed_physics_step_observer is not None:
            time_us = round(float(self._env.sim.data.time) * 1_000_000)
            physics_step_index = time_us // 2_000
            self._completed_physics_step_observer(
                CompletedPhysicsStep(
                    physics_step_index=physics_step_index,
                    formal_tick_index=physics_step_index // 10,
                    time_us=time_us,
                    at_formal_boundary=physics_step_index % 10 == 0,
                )
            )
        self.step2_count += 1


@dataclass(frozen=True)
class StockStepReport:
    step1_count: int
    step2_count: int
    control_refresh_count: int
    goal_refresh_count: int
    observable_update_count: int
    policy_step_flags: tuple[bool, ...]
    start_time_seconds: float
    end_time_seconds: float
    latest_sample_time_seconds: float
    returned_observation_age_seconds: float


def make_g1_environment(*, seed: int, offscreen: bool) -> Any:
    """Create the pinned stock Lift/Panda calibration environment."""
    import mujoco
    import robosuite as suite
    from robosuite.controllers import load_composite_controller_config

    env = suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=load_composite_controller_config(robot="Panda"),
        has_renderer=False,
        has_offscreen_renderer=offscreen,
        use_camera_obs=False,
        control_freq=50,
        lite_physics=True,
        horizon=10_000,
        ignore_done=True,
        hard_reset=False,
        seed=seed,
    )
    env.sim.model.opt.integrator = int(mujoco.mjtIntegrator.mjINT_EULER)
    env.reset()
    env.sim.forward()
    env.task_object = env.cube
    env.task_object_body_id = env.cube_body_id
    return env


def instrument_stock_step(env: Any, action: np.ndarray, *, observable_name: str) -> StockStepReport:
    """Count native calls and observe cache sampling without additional simulator reads."""
    observable = env._observables[observable_name]
    controller = env.robots[0].composite_controller
    original_step1 = env.sim.step1
    original_step2 = env.sim.step2
    original_pre_action = env._pre_action
    original_update_observables = env._update_observables
    original_filter = observable._filter
    original_set_goal = controller.set_goal
    original_run_controller = controller.run_controller
    step1_count = 0
    step2_count = 0
    update_count = 0
    goal_count = 0
    control_count = 0
    policy_flags: list[bool] = []
    sample_times: list[float] = []

    def wrapped_step1() -> None:
        nonlocal step1_count
        step1_count += 1
        original_step1()

    def wrapped_step2() -> None:
        nonlocal step2_count
        step2_count += 1
        original_step2()

    def wrapped_pre_action(inner_action: np.ndarray, policy_step: bool = False) -> None:
        policy_flags.append(bool(policy_step))
        original_pre_action(inner_action, policy_step)

    def wrapped_update_observables(force: bool = False) -> None:
        nonlocal update_count
        update_count += 1
        original_update_observables(force=force)

    def wrapped_filter(value: Any) -> Any:
        sample_times.append(float(env.sim.data.time))
        return original_filter(value)

    def wrapped_set_goal(goal: Any) -> None:
        nonlocal goal_count
        goal_count += 1
        original_set_goal(goal)

    def wrapped_run_controller(enabled_parts: Any) -> Any:
        nonlocal control_count
        control_count += 1
        return original_run_controller(enabled_parts)

    env.sim.step1 = wrapped_step1
    env.sim.step2 = wrapped_step2
    env._pre_action = wrapped_pre_action
    env._update_observables = wrapped_update_observables
    observable._filter = wrapped_filter
    controller.set_goal = wrapped_set_goal
    controller.run_controller = wrapped_run_controller
    start_time = float(env.sim.data.time)
    try:
        observations, _, _, _ = env.step(action)
        if observable_name not in observations:
            raise RuntimeError(
                f"instrumented observable missing from returned data: {observable_name}"
            )
    finally:
        env.sim.step1 = original_step1
        env.sim.step2 = original_step2
        env._pre_action = original_pre_action
        env._update_observables = original_update_observables
        observable._filter = original_filter
        controller.set_goal = original_set_goal
        controller.run_controller = original_run_controller
    end_time = float(env.sim.data.time)
    if not sample_times:
        raise RuntimeError(f"instrumented observable was never sampled: {observable_name}")
    latest_sample_time = sample_times[-1]
    return StockStepReport(
        step1_count=step1_count,
        step2_count=step2_count,
        control_refresh_count=control_count,
        goal_refresh_count=goal_count,
        observable_update_count=update_count,
        policy_step_flags=tuple(policy_flags),
        start_time_seconds=round(start_time, 12),
        end_time_seconds=round(end_time, 12),
        latest_sample_time_seconds=round(latest_sample_time, 12),
        returned_observation_age_seconds=round(end_time - latest_sample_time, 12),
    )
