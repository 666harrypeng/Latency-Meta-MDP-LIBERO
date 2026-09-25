"""Continuous parcel evaluation using the shared logical-latency policy runtime."""

from __future__ import annotations

import time

import numpy as np

from latency_meta_mdp.runtime.policy_execution import (
    FixedCursorScheduler,
    LogicalPolicyRuntime,
    PhysicalStepResult,
)


def run_conveyor_policy_episode(
    *,
    runtime,
    policy,
    client_config,
    delay_sampler,
    maximum_steps: int,
    record_observation=None,
    scheduler=None,
    policy_alignment=None,
    minimum_decision_tick=10,
) -> dict:
    """Own and close a conveyor simulator; each delivery rewards but does not terminate.

    The actor receives camera/proprio observations only. Parcel truth is read for
    reporting and reward, and the shared harness controls all request timing.
    An external step limit truncates without inventing parcel failures.
    """
    try:
        if type(maximum_steps) is not int or maximum_steps <= 0:
            raise ValueError("maximum_steps must be a positive integer")
        started = time.perf_counter()
        latest = runtime.executor.initialize()
        engine = LogicalPolicyRuntime(
            action_contract=runtime.action_contract,
            client_config=client_config,
            simulation_time_reader=lambda: runtime.executor.ledger.time_us,
            delay_sampler=delay_sampler,
            policy=policy,
            bootstrap_policy=lambda obs: policy(obs, None),
            scheduler=FixedCursorScheduler(25) if scheduler is None else scheduler,
            policy_alignment=policy_alignment,
            minimum_decision_tick=minimum_decision_tick,
        )
        recording_seconds = 0.0
        recorded_frames = 0

        def observe():
            return latest

        def record():
            nonlocal recording_seconds, recorded_frames
            if record_observation is not None:
                began = time.perf_counter()
                record_observation(latest)
                recording_seconds += time.perf_counter() - began
                recorded_frames += 1

        def execute(action):
            nonlocal latest
            latest = runtime.executor.step_formal(action)
            return PhysicalStepResult(
                reward=float(runtime.env.boundary_reward), terminated=bool(runtime.env.done)
            )

        record()
        engine.bootstrap(observe)
        actions = []
        while not engine.terminated and len(actions) < maximum_steps:
            action, _ = engine.step(
                formal_tick=latest.formal_tick, observe=observe, execute=execute
            )
            actions.append(np.asarray(action).tolist())
            record()
        truncated = not engine.terminated
        if truncated:
            engine.truncate(formal_tick=latest.formal_tick)
        return {
            **engine.summary(),
            "task_id": "conveyor_sort",
            "parcels": runtime.world.ledger.summary(),
            "parcel_events": runtime.world.ledger.events,
            "truncated": truncated,
            "executed_steps": len(actions),
            "elapsed_simulation_seconds": len(actions) * 0.02,
            "wall_seconds": time.perf_counter() - started,
            "recording_wall_seconds": recording_seconds,
            "recorded_frames": recorded_frames,
            "executed_actions": actions,
            "stage_events": engine.events,
            "request_ledger": engine.request_ledger(),
            "learned_policy_result": True,
        }
    finally:
        runtime.env.close()
