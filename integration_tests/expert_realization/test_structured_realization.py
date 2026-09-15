from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_seed_3999_canonical_realization_reaches_physical_handoff_and_lift() -> None:
    """Gate: one planned smooth approach executes through the real OSC/contact stack."""
    from latency_meta_mdp.data.collection.contracts import (
        ExpertRealizationKey,
        StrategyFamily,
    )
    from latency_meta_mdp.data.collection.planner import generate_planner_candidates
    from latency_meta_mdp.data.collection.robot_bridge import build_panda_planning_bridge
    from latency_meta_mdp.data.collection.rollout import execute_structured_realization
    from latency_meta_mdp.data.collection.selector import select_task_instance_plan_set
    from latency_meta_mdp.data.collection.strategy import (
        StructuredStrategyConfig,
        sample_strategy,
    )
    from latency_meta_mdp.data.collection.task_instance import materialize_task_instance
    from latency_meta_mdp.data.collection.trajectory_intent import build_trajectory_intent

    root = Path.cwd()
    task = materialize_task_instance(project_root=root, level=1, task_instance_seed=3999)
    config = StructuredStrategyConfig.from_path(
        root / "configs/data/expert_realization/panda_ball_structured.yaml"
    )
    key = ExpertRealizationKey(task.task_instance_id, 0, config.source_sha256)
    strategy = sample_strategy(
        task,
        key,
        config,
        assigned_family=StrategyFamily.CANONICAL_DIRECT,
    )
    intent = build_trajectory_intent(task, task.expected_anchor, strategy)
    candidates = generate_planner_candidates(
        expert_realization_key=key,
        structured_expert_config_sha256=config.source_sha256,
        bridge=build_panda_planning_bridge(root),
        intent=intent,
        start_qpos=task.expected_anchor.anchor_robot_qpos,
    )
    plan_set = select_task_instance_plan_set(
        task_instance=task,
        candidates_by_key={key: candidates},
    )
    rollout = execute_structured_realization(
        task_instance=task,
        intent=intent,
        reference=plan_set.references[0],
        maximum_formal_ticks=220,
    )

    assert rollout.terminal_status == "success", (
        rollout.terminal_reason,
        rollout.terminal_tick,
        rollout.phase_sequence[-10:],
        np.linalg.norm(
            rollout.eef_positions_world[-10:] - rollout.object_positions_world[-10:],
            axis=1,
        ).tolist(),
        rollout.eef_positions_world[-1].tolist(),
        rollout.object_positions_world[-1].tolist(),
    )
    assert rollout.physical_handoff_tick is not None
    assert rollout.terminal_tick is not None
    assert rollout.terminal_tick < 220
    assert len(rollout.agentview_rgb) == rollout.terminal_tick + 1
    assert "smooth_approach" in rollout.phase_sequence
    assert "grasp_funnel" in rollout.phase_sequence
    assert "close_stabilize" in rollout.phase_sequence
    assert "lift" in rollout.phase_sequence
