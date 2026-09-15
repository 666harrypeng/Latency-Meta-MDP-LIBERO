"""Real-simulator direct versus logical-zero-delay parity calibration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from latency_meta_mdp.data.expert_collection import (
    ExpertEpisodeSpec,
    build_expert_episode_runtime,
)
from latency_meta_mdp.data.recording import RecordProfile
from latency_meta_mdp.envs.expert import ExpertObservation, is_qualified_expert_episode
from latency_meta_mdp.envs.outcomes import OutcomeStatus, TerminalReason
from latency_meta_mdp.envs.snapshots import BoundarySnapshot
from latency_meta_mdp.runtime.latency_client import OneStepLatencyClient
from latency_meta_mdp.runtime.latency_harness import (
    FixedDelaySampler,
    HarnessEvent,
    LogicalLatencyHarness,
)


def _readonly_array(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ParityLaneTrace:
    boundary_time_us: np.ndarray
    executed_action: np.ndarray
    robot_qpos: np.ndarray
    robot_qvel: np.ndarray
    gripper_qpos: np.ndarray
    gripper_qvel: np.ndarray
    object_qpos: np.ndarray
    object_qvel: np.ndarray
    actuator_ctrl: np.ndarray
    agentview_rgb: np.ndarray
    robot0_eye_in_hand_rgb: np.ndarray
    physical_events: tuple[tuple[str, int, str | None], ...]
    terminal_status: OutcomeStatus
    terminal_reason: TerminalReason
    terminal_time_us: int

    def __post_init__(self) -> None:
        for name in self.array_field_names():
            object.__setattr__(self, name, _readonly_array(getattr(self, name)))
        object.__setattr__(self, "physical_events", tuple(self.physical_events))
        boundary_count = self.boundary_time_us.shape[0]
        if self.boundary_time_us.ndim != 1 or boundary_count < 2:
            raise ValueError("parity trace requires at least two boundary times")
        if self.executed_action.ndim != 2 or self.executed_action.shape[0] != boundary_count - 1:
            raise ValueError("parity trace requires T actions for T+1 boundaries")
        for name in self.array_field_names():
            if name == "executed_action":
                continue
            value = getattr(self, name)
            if value.shape[0] != boundary_count:
                raise ValueError(f"{name} does not align with parity boundaries")
        if not isinstance(self.terminal_status, OutcomeStatus):
            raise TypeError("terminal_status must be an OutcomeStatus")
        if not isinstance(self.terminal_reason, TerminalReason):
            raise TypeError("terminal_reason must be a TerminalReason")
        if self.terminal_time_us != int(self.boundary_time_us[-1]):
            raise ValueError("terminal time does not match the final parity boundary")

    @classmethod
    def array_field_names(cls) -> tuple[str, ...]:
        return (
            "boundary_time_us",
            "executed_action",
            "robot_qpos",
            "robot_qvel",
            "gripper_qpos",
            "gripper_qvel",
            "object_qpos",
            "object_qvel",
            "actuator_ctrl",
            "agentview_rgb",
            "robot0_eye_in_hand_rgb",
        )


@dataclass(frozen=True)
class LatencyParityResult:
    direct: ParityLaneTrace
    harness: ParityLaneTrace
    harness_events: tuple[HarnessEvent, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "harness_events", tuple(self.harness_events))

    @staticmethod
    def _first_row_mismatch(left: np.ndarray, right: np.ndarray) -> int:
        if left.ndim == 1:
            mismatch = left != right
        else:
            mismatch = np.any(left != right, axis=tuple(range(1, left.ndim)))
        indices = np.flatnonzero(mismatch)
        return 0 if indices.size == 0 else int(indices[0])

    def validate_exact(self) -> None:
        for name in ParityLaneTrace.array_field_names():
            direct = getattr(self.direct, name)
            harness = getattr(self.harness, name)
            if direct.shape != harness.shape:
                raise ValueError(f"zero-delay parity mismatch: {name} shape")
            if not np.array_equal(direct, harness):
                index = self._first_row_mismatch(direct, harness)
                coordinate = "transition" if name == "executed_action" else "boundary"
                raise ValueError(f"zero-delay parity mismatch: {name} at {coordinate} {index}")
        for name in (
            "physical_events",
            "terminal_status",
            "terminal_reason",
            "terminal_time_us",
        ):
            if getattr(self.direct, name) != getattr(self.harness, name):
                raise ValueError(f"zero-delay parity mismatch: {name}")


def _snapshot_arrays(snapshot: BoundarySnapshot) -> dict[str, np.ndarray | int]:
    return {
        "boundary_time_us": snapshot.time_us,
        "robot_qpos": snapshot.robot_qpos,
        "robot_qvel": snapshot.robot_qvel,
        "gripper_qpos": snapshot.robot_gripper_qpos,
        "gripper_qvel": snapshot.robot_gripper_qvel,
        "object_qpos": snapshot.object_qpos,
        "object_qvel": snapshot.object_qvel,
        "actuator_ctrl": snapshot.actuator_ctrl,
        "agentview_rgb": snapshot.cameras["agentview"].rgb,
        "robot0_eye_in_hand_rgb": snapshot.cameras["robot0_eye_in_hand"].rgb,
    }


def _run_lane(
    *,
    project_root: Path,
    seed: int,
    camera_width: int,
    camera_height: int,
    use_harness: bool,
) -> tuple[ParityLaneTrace, tuple[HarnessEvent, ...]]:
    spec = ExpertEpisodeSpec(
        episode_id=f"parity-l1-seed-{seed:06d}-{'harness' if use_harness else 'direct'}",
        level=1,
        scene_seed=seed,
        motion_seed=seed,
        expert_seed=seed,
        record_profile=RecordProfile.PILOT_DEBUG,
        camera_width=camera_width,
        camera_height=camera_height,
    )
    runtime = build_expert_episode_runtime(project_root=project_root, spec=spec)
    snapshots: list[dict[str, np.ndarray | int]] = []
    actions: list[np.ndarray] = []
    harness: LogicalLatencyHarness[ExpertObservation, np.ndarray] | None = None
    client: OneStepLatencyClient[ExpertObservation] | None = None
    if use_harness:
        harness = LogicalLatencyHarness(
            formal_tick_us=runtime.runtime_config.formal_tick_us,
            delay_sampler=FixedDelaySampler(0),
            simulation_time_reader=lambda: runtime.executor.ledger.time_us,
        )
        client = OneStepLatencyClient(
            action_contract=runtime.contract,
            harness=harness,
        )
    try:
        snapshot = runtime.executor.initialize()
        snapshots.append(_snapshot_arrays(snapshot))
        max_steps = (
            runtime.expert_config.collection_max_duration_us
            // runtime.runtime_config.formal_tick_us
        )
        for _ in range(max_steps):
            arm = runtime.env.robots[0].part_controllers["right"]
            observation = ExpertObservation.from_snapshot(
                snapshot,
                world_to_base_rotation=arm.origin_ori.T,
            )
            if client is None:
                decision = runtime.expert.next_action(
                    observation=observation,
                    handoff_state=runtime.handoff.state,
                )
                action = decision.action
                snapshot = runtime.executor.step_formal(action)
            else:
                next_snapshot: BoundarySnapshot | None = None

                def infer(context) -> np.ndarray:
                    return runtime.expert.next_action(
                        observation=context.observation,
                        handoff_state=runtime.handoff.state,
                    ).action

                def execute(selected_action: np.ndarray) -> None:
                    nonlocal next_snapshot
                    next_snapshot = runtime.executor.step_formal(selected_action)

                action = client.run_boundary(
                    formal_tick=observation.formal_tick_index,
                    observation=observation,
                    launch=True,
                    infer=infer,
                    execute=execute,
                )
                if next_snapshot is None:
                    raise RuntimeError("zero-delay client did not advance the simulator")
                snapshot = next_snapshot
            actions.append(np.array(action, copy=True))
            snapshots.append(_snapshot_arrays(snapshot))
            if runtime.tracker.status is not OutcomeStatus.RUNNING:
                break
        if not is_qualified_expert_episode(
            status=runtime.tracker.status,
            terminal_time_us=runtime.tracker.terminal_time_us,
            max_duration_us=runtime.expert_config.collection_max_duration_us,
        ):
            raise RuntimeError("parity lane did not satisfy expert collection qualification")
        if runtime.tracker.terminal_reason is None or runtime.tracker.terminal_time_us is None:
            raise RuntimeError("parity lane is missing terminal outcome provenance")
        trace = ParityLaneTrace(
            boundary_time_us=np.asarray(
                [record["boundary_time_us"] for record in snapshots], dtype=np.int64
            ),
            executed_action=np.stack(actions),
            robot_qpos=np.stack([record["robot_qpos"] for record in snapshots]),
            robot_qvel=np.stack([record["robot_qvel"] for record in snapshots]),
            gripper_qpos=np.stack([record["gripper_qpos"] for record in snapshots]),
            gripper_qvel=np.stack([record["gripper_qvel"] for record in snapshots]),
            object_qpos=np.stack([record["object_qpos"] for record in snapshots]),
            object_qvel=np.stack([record["object_qvel"] for record in snapshots]),
            actuator_ctrl=np.stack([record["actuator_ctrl"] for record in snapshots]),
            agentview_rgb=np.stack([record["agentview_rgb"] for record in snapshots]),
            robot0_eye_in_hand_rgb=np.stack(
                [record["robot0_eye_in_hand_rgb"] for record in snapshots]
            ),
            physical_events=tuple(
                (
                    event.kind,
                    event.time_us,
                    None if event.terminal_reason is None else event.terminal_reason.value,
                )
                for event in runtime.tracker.events
            ),
            terminal_status=runtime.tracker.status,
            terminal_reason=runtime.tracker.terminal_reason,
            terminal_time_us=runtime.tracker.terminal_time_us,
        )
        return trace, () if harness is None else harness.events
    finally:
        runtime.close()


def run_direct_zero_latency_parity(
    *,
    project_root: Path,
    seed: int,
    camera_width: int,
    camera_height: int,
) -> LatencyParityResult:
    """Run independent direct and zero-delay L1 lanes and require exact parity."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("parity seed must be a non-negative integer")
    for name, value in (("camera_width", camera_width), ("camera_height", camera_height)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    direct, direct_events = _run_lane(
        project_root=project_root,
        seed=seed,
        camera_width=camera_width,
        camera_height=camera_height,
        use_harness=False,
    )
    if direct_events:
        raise RuntimeError("direct parity lane unexpectedly produced harness events")
    harness, harness_events = _run_lane(
        project_root=project_root,
        seed=seed,
        camera_width=camera_width,
        camera_height=camera_height,
        use_harness=True,
    )
    result = LatencyParityResult(
        direct=direct,
        harness=harness,
        harness_events=harness_events,
    )
    result.validate_exact()
    return result


def run_direct_expert_trace(
    *,
    project_root: Path,
    seed: int,
    camera_width: int,
    camera_height: int,
) -> ParityLaneTrace:
    """Run one direct L1 expert lane for downstream infrastructure calibration."""

    trace, events = _run_lane(
        project_root=project_root,
        seed=seed,
        camera_width=camera_width,
        camera_height=camera_height,
        use_harness=False,
    )
    if events:
        raise RuntimeError("direct expert trace unexpectedly produced harness events")
    return trace
