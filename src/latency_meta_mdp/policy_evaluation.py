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
    *, runtime, policy, client_config, delay_sampler, maximum_steps: int = 1000
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
        engine = LogicalPolicyRuntime(
            action_contract=runtime.action_contract,
            client_config=client_config,
            simulation_time_reader=lambda: runtime.executor.ledger.time_us,
            delay_sampler=delay_sampler,
            policy=policy,
            bootstrap_policy=lambda obs: policy(obs, None),
            scheduler=FixedCursorScheduler(25),
        )

        def observe():
            return policy_observation_from_snapshot(latest)

        def execute(action):
            nonlocal latest
            latest = runtime.executor.step_formal(action)
            status = runtime.tracker.status.value
            return PhysicalStepResult(
                reward=float(status == "success"), terminated=status != "running"
            )

        engine.bootstrap(observe)
        actions = []
        while not engine.terminated and len(actions) < maximum_steps:
            action, _ = engine.step(
                formal_tick=latest.formal_tick_index, observe=observe, execute=execute
            )
            actions.append(np.asarray(action).tolist())
        status = runtime.tracker.status.value
        success = status == "success"
        truncated = not engine.terminated
        elapsed = len(actions) * 0.02
        differences = np.diff(np.asarray(actions), axis=0)
        return {
            **engine.summary(),
            "success": success,
            "truncated": truncated,
            "task_status": "time_limit" if truncated else status,
            "executed_steps": len(actions),
            "elapsed_simulation_seconds": elapsed,
            "completion_time_seconds": elapsed if success else None,
            "wall_seconds": time.perf_counter() - started,
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
