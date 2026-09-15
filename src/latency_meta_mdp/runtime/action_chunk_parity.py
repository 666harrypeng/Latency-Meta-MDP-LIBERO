"""Real L1 parity calibration for the warm-start sharp chunk client."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from latency_meta_mdp.data.expert_collection import (
    ExpertEpisodeSpec,
    build_expert_episode_runtime,
)
from latency_meta_mdp.data.recording import RecordProfile
from latency_meta_mdp.envs.outcomes import OutcomeStatus
from latency_meta_mdp.envs.snapshots import BoundarySnapshot
from latency_meta_mdp.runtime.action_chunk_client import (
    BootstrapRecord,
    ChunkClientEvent,
    SharpActionChunkClient,
    load_action_chunk_client_config,
)
from latency_meta_mdp.runtime.latency_harness import (
    FixedDelaySampler,
    HarnessEvent,
    HarnessEventKind,
    LogicalLatencyHarness,
)
from latency_meta_mdp.runtime.latency_parity import (
    LatencyParityResult,
    ParityLaneTrace,
    run_direct_expert_trace,
)


@dataclass(frozen=True)
class ActionChunkParityResult:
    direct: ParityLaneTrace
    chunk: ParityLaneTrace
    bootstrap: BootstrapRecord
    harness_events: tuple[HarnessEvent, ...]
    chunk_events: tuple[ChunkClientEvent, ...]
    generator_kind: str = "reference_action_replay_oracle"

    def __post_init__(self) -> None:
        object.__setattr__(self, "harness_events", tuple(self.harness_events))
        object.__setattr__(self, "chunk_events", tuple(self.chunk_events))
        if self.generator_kind != "reference_action_replay_oracle":
            raise ValueError("action-chunk parity generator identity is invalid")

    @property
    def chunk_launch_ticks(self) -> tuple[int, ...]:
        return tuple(
            event.formal_tick
            for event in self.harness_events
            if event.kind is HarnessEventKind.LAUNCH
        )

    @property
    def starvation_ticks(self) -> tuple[int, ...]:
        return tuple(
            event.formal_tick
            for event in self.harness_events
            if event.kind is HarnessEventKind.STARVATION
        )

    def validate_exact(self) -> None:
        LatencyParityResult(
            direct=self.direct,
            harness=self.chunk,
            harness_events=(),
        ).validate_exact()


def _chunk_at(actions: np.ndarray, *, start_tick: int, horizon: int) -> np.ndarray:
    if (
        isinstance(start_tick, bool)
        or not isinstance(start_tick, int)
        or start_tick < 0
        or horizon <= 0
    ):
        raise ValueError("replay chunk coordinates are invalid")
    if actions.ndim != 2 or actions.shape[0] == 0:
        raise ValueError("reference action trace must be a non-empty matrix")
    if start_tick < actions.shape[0]:
        selected = np.array(actions[start_tick : start_tick + horizon], copy=True)
        pad_action = selected[-1]
    else:
        selected = np.empty((0, actions.shape[1]), dtype=actions.dtype)
        pad_action = actions[-1]
    if selected.shape[0] < horizon:
        selected = np.concatenate(
            [
                selected,
                np.repeat(
                    np.asarray(pad_action)[None, :],
                    horizon - selected.shape[0],
                    axis=0,
                ),
            ],
            axis=0,
        )
    selected.setflags(write=False)
    return selected


def _snapshot_values(snapshot: BoundarySnapshot) -> dict[str, np.ndarray | int]:
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


def _trace_from_runtime(
    *,
    snapshots: list[dict[str, np.ndarray | int]],
    actions: list[np.ndarray],
    runtime,
) -> ParityLaneTrace:
    if runtime.tracker.terminal_reason is None or runtime.tracker.terminal_time_us is None:
        raise RuntimeError("chunk parity lane is missing terminal outcome provenance")
    return ParityLaneTrace(
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
        robot0_eye_in_hand_rgb=np.stack([record["robot0_eye_in_hand_rgb"] for record in snapshots]),
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


def run_action_chunk_zero_delay_parity(
    *,
    project_root: Path,
    seed: int,
    camera_width: int,
    camera_height: int,
) -> ActionChunkParityResult:
    """Require exact L1 parity using reference actions repackaged into H50 chunks."""

    direct = run_direct_expert_trace(
        project_root=project_root,
        seed=seed,
        camera_width=camera_width,
        camera_height=camera_height,
    )
    spec = ExpertEpisodeSpec(
        episode_id=f"chunk-parity-l1-seed-{seed:06d}",
        level=1,
        scene_seed=seed,
        motion_seed=seed,
        expert_seed=seed,
        record_profile=RecordProfile.PILOT_DEBUG,
        camera_width=camera_width,
        camera_height=camera_height,
    )
    runtime = build_expert_episode_runtime(project_root=project_root, spec=spec)
    config = load_action_chunk_client_config(
        project_root / "configs/runtime/client/sharp_return_time_h50_e25_v1.yaml"
    )
    harness: LogicalLatencyHarness[BoundarySnapshot, np.ndarray] = LogicalLatencyHarness(
        formal_tick_us=runtime.runtime_config.formal_tick_us,
        delay_sampler=FixedDelaySampler(0),
        simulation_time_reader=lambda: runtime.executor.ledger.time_us,
    )
    client = SharpActionChunkClient(
        action_contract=runtime.contract,
        config=config,
        harness=harness,
        simulation_time_reader=lambda: runtime.executor.ledger.time_us,
        monotonic_ns=time.monotonic_ns,
    )
    snapshots: list[dict[str, np.ndarray | int]] = []
    actions: list[np.ndarray] = []
    try:
        snapshot = runtime.executor.initialize()
        snapshots.append(_snapshot_values(snapshot))
        bootstrap = client.bootstrap(
            observation=snapshot,
            infer=lambda observation: _chunk_at(
                direct.executed_action,
                start_tick=0,
                horizon=config.prediction_horizon,
            ),
        )
        max_steps = (
            runtime.expert_config.collection_max_duration_us
            // runtime.runtime_config.formal_tick_us
        )
        for _ in range(max_steps):
            next_snapshot: BoundarySnapshot | None = None

            def infer(context) -> np.ndarray:
                return _chunk_at(
                    direct.executed_action,
                    start_tick=context.launch_formal_tick,
                    horizon=config.prediction_horizon,
                )

            def execute(action: np.ndarray) -> None:
                nonlocal next_snapshot
                next_snapshot = runtime.executor.step_formal(action)

            action = client.run_boundary(
                formal_tick=snapshot.formal_tick_index,
                observation=snapshot,
                infer=infer,
                execute=execute,
            )
            if next_snapshot is None:
                raise RuntimeError("chunk client did not advance the simulator")
            snapshot = next_snapshot
            actions.append(np.array(action, copy=True))
            snapshots.append(_snapshot_values(snapshot))
            if runtime.tracker.status is not OutcomeStatus.RUNNING:
                break
        trace = _trace_from_runtime(
            snapshots=snapshots,
            actions=actions,
            runtime=runtime,
        )
        result = ActionChunkParityResult(
            direct=direct,
            chunk=trace,
            bootstrap=bootstrap,
            harness_events=harness.events,
            chunk_events=client.chunk_events,
        )
        result.validate_exact()
        return result
    finally:
        runtime.close()
