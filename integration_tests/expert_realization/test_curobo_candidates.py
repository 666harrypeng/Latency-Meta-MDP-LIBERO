from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_real_curobo_candidate_set_is_complete_seeded_and_numerical(tmp_path: Path) -> None:
    """Gate: one realization produces eight frozen records plus a repeated-seed audit."""
    from latency_meta_mdp.expert_realization.contracts import (
        ExpertRealizationKey,
        StrategyFamily,
    )
    from latency_meta_mdp.expert_realization.planner import (
        PlannerCandidateStatus,
        generate_planner_candidates,
        load_planner_candidates,
        write_planner_candidates,
    )
    from latency_meta_mdp.expert_realization.robot_bridge import build_panda_planning_bridge
    from latency_meta_mdp.expert_realization.strategy import (
        StructuredStrategyConfig,
        sample_strategy,
    )
    from latency_meta_mdp.expert_realization.task_instance import materialize_task_instance
    from latency_meta_mdp.expert_realization.trajectory_intent import build_trajectory_intent

    root = Path.cwd()
    task = materialize_task_instance(project_root=root, level=1, task_instance_seed=4000)
    config = StructuredStrategyConfig.from_path(
        root / "configs/expert_realization/panda_ball_structured.yaml"
    )
    key = ExpertRealizationKey(task.task_instance_id, 0, config.source_sha256)
    strategy = sample_strategy(
        task,
        key,
        config,
        assigned_family=StrategyFamily.CANONICAL_DIRECT,
    )
    intent = build_trajectory_intent(task, task.expected_anchor, strategy)
    bridge = build_panda_planning_bridge(root)

    candidates = generate_planner_candidates(
        expert_realization_key=key,
        structured_expert_config_sha256=config.source_sha256,
        bridge=bridge,
        intent=intent,
        start_qpos=task.expected_anchor.anchor_robot_qpos,
    )

    assert len(candidates) == 8
    assert [item.candidate_index for item in candidates] == list(range(8))
    assert candidates[0].deterministic_replay_verified is True
    assert any(item.status is PlannerCandidateStatus.SUCCESS for item in candidates)
    arrival_duration_seconds = (
        intent.approach.funnel_entry_target_tick - task.decision_source_tick
    ) * 0.02
    for candidate in candidates:
        assert candidate.requested_seed == candidate.effective_seed
        assert candidate.planning_time_seconds >= 0.0
        if candidate.status is PlannerCandidateStatus.SUCCESS:
            assert len(candidate.geometric_seed_qpos_path) >= 2
            assert len(candidate.qpos_path) >= 2
            assert len(candidate.timestamps_seconds) == len(candidate.qpos_path)
            assert len(candidate.eef_positions_world) == len(candidate.qpos_path)
            assert candidate.timestamps_seconds[-1] == pytest.approx(
                arrival_duration_seconds,
                abs=1.0e-12,
            )
            assert all(
                value == pytest.approx(0.02, abs=1.0e-12)
                for value in candidate.timestamps_seconds[1:]
                - candidate.timestamps_seconds[:-1]
            )

    artifact = tmp_path / "candidate-set"
    write_planner_candidates(artifact, candidates)
    assert (
        load_planner_candidates(
            artifact,
            structured_expert_config_sha256=config.source_sha256,
        )
        == candidates
    )
