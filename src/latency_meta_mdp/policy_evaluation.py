"""Closed-loop native-policy episodes on the formal logical-latency runtime."""

from __future__ import annotations

import time

import numpy as np

from latency_meta_mdp.policy_execution import (
    FixedCursorScheduler,
    LogicalPolicyRuntime,
    PhysicalStepResult,
    policy_observation_from_snapshot,
)


def run_native_policy_episode(
    *,
    runtime,
    policy,
    client_config,
    delay_sampler,
    maximum_steps: int = 1000,
    record_observation=None,
    bootstrap_policy=None,
    belief_provider_factory=None,
    allow_privileged_belief: bool = False,
    latency_probabilities=None,
    policy_alignment=None,
    forecast_provider=None,
) -> dict:
    """Execute an owned simulator to its task terminal condition or an explicit time limit.

    Hidden task state is used only for reward/status reporting. The actor receives
    the same current image/proprio interface as training. The simulator is closed
    even if policy inference or execution fails.
    """
    try:
        if type(maximum_steps) is not int or maximum_steps <= 0:
            raise ValueError("maximum_steps must be a positive integer")
        started = time.perf_counter()
        latest = runtime.executor.initialize()
        if belief_provider_factory is not None and bootstrap_policy is None:
            raise ValueError("conditioned evaluation requires an explicit native bootstrap")
        belief = (
            belief_provider_factory(runtime, lambda: latest)
            if belief_provider_factory is not None
            else None
        )
        engine = LogicalPolicyRuntime(
            action_contract=runtime.action_contract,
            client_config=client_config,
            simulation_time_reader=lambda: runtime.executor.ledger.time_us,
            delay_sampler=delay_sampler,
            policy=policy,
            bootstrap_policy=bootstrap_policy or (lambda obs: policy(obs, None)),
            scheduler=FixedCursorScheduler(25),
            belief_provider=belief,
            policy_uses_belief=belief is not None,
            allow_privileged_belief=allow_privileged_belief,
            latency_probabilities=latency_probabilities,
            policy_alignment=policy_alignment,
            forecast_provider=forecast_provider,
        )

        def observe():
            return policy_observation_from_snapshot(latest)

        recording_seconds = 0.0
        recorded_frames = 0

        def record():
            nonlocal recording_seconds, recorded_frames
            if record_observation is not None:
                began = time.perf_counter()
                record_observation(observe())
                recording_seconds += time.perf_counter() - began
                recorded_frames += 1

        def execute(action):
            nonlocal latest
            latest = runtime.executor.step_formal(action)
            status = runtime.tracker.status.value
            return PhysicalStepResult(
                reward=float(status == "success"), terminated=status != "running"
            )

        record()
        engine.bootstrap(observe)
        actions = []
        while not engine.terminated and len(actions) < maximum_steps:
            action, _ = engine.step(
                formal_tick=latest.formal_tick_index, observe=observe, execute=execute
            )
            actions.append(np.asarray(action).tolist())
            record()
        status = runtime.tracker.status.value
        success = status == "success"
        reason = getattr(runtime.tracker, "terminal_reason", None)
        truncated = not engine.terminated
        elapsed = len(actions) * 0.02
        differences = np.diff(np.asarray(actions), axis=0)
        return {
            **engine.summary(),
            "success": success,
            "truncated": truncated,
            "task_status": "time_limit" if truncated else status,
            "terminal_reason": getattr(reason, "value", reason),
            "executed_steps": len(actions),
            "elapsed_simulation_seconds": elapsed,
            "completion_time_seconds": elapsed if success else None,
            "wall_seconds": time.perf_counter() - started,
            "recording_wall_seconds": recording_seconds,
            "recorded_frames": recorded_frames,
            "mean_action_step_l2": float(np.linalg.norm(differences, axis=1).mean())
            if len(differences)
            else 0.0,
            "executed_actions": actions,
            "stage_events": engine.events,
            "request_ledger": engine.request_ledger(),
            "learned_policy_result": True,
        }
    finally:
        runtime.close()
