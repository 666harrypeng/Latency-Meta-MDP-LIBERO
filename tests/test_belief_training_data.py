from __future__ import annotations

from pathlib import Path

import numpy as np

from latency_meta_mdp.belief_data import load_belief_episode
from latency_meta_mdp.control import load_action_contract
from latency_meta_mdp.episode_artifacts import write_synchronized_episode_artifact
from latency_meta_mdp.expert_collection import ExpertEpisodeSpec, collect_expert_episode
from latency_meta_mdp.latency_law import load_latency_law
from latency_meta_mdp.recording import RecordProfile
from latency_meta_mdp.temporal_contract import load_temporal_contract
from latency_meta_mdp.terminal_absorbing_tail import build_terminal_absorbing_tail


def _tail(tmp_path: Path):
    complete = collect_expert_episode(
        project_root=Path.cwd(),
        spec=ExpertEpisodeSpec(
            episode_id="l3-seed-000010-belief-training",
            level=3,
            scene_seed=10,
            motion_seed=10,
            expert_seed=10,
            record_profile=RecordProfile.BELIEF,
            camera_width=8,
            camera_height=8,
        ),
    )
    output = tmp_path / "episode"
    write_synchronized_episode_artifact(episode=complete, output_dir=output)
    episode = load_belief_episode(output)
    contract = load_temporal_contract(
        Path("configs/temporal/h50_e25_d20_k6_v1.yaml")
    )
    return (
        build_terminal_absorbing_tail(
            episode=episode,
            temporal_contract=contract,
            action_contract=load_action_contract(
                Path("configs/control/panda_osc_pose_delta_v1.yaml")
            ),
        ),
        contract,
        load_latency_law(
            Path("configs/latency/truncated_beta_5_26_400ms_v1.yaml")
        ),
    )


def test_belief_training_context_materializes_one_context_with_target_table(
    tmp_path: Path,
) -> None:
    from latency_meta_mdp.belief_training_data import (
        InteractionMode,
        build_belief_training_indices,
        materialize_belief_training_context,
    )

    tail, contract, law = _tail(tmp_path)
    indices = build_belief_training_indices(
        tail_view=tail,
        temporal_contract=contract,
    )
    assert len(indices) == tail.real_transition_count - 25
    assert indices[0].source_tick == 25
    assert indices[0].history_start_tick == 20
    assert indices[-1].source_tick == tail.real_transition_count - 1

    first = materialize_belief_training_context(
        index=indices[0],
        tail_view=tail,
        temporal_contract=contract,
        latency_law=law,
    )
    assert first.agentview_history.shape == (6, 8, 8, 3)
    assert first.wrist_history.shape == (6, 8, 8, 3)
    assert first.robot_proprio_history.shape == (6, 16)
    assert first.remaining_actions.shape == (25, 7)
    assert first.latency_probabilities.shape == (20,)
    assert first.target_delay_ticks.tolist() == list(range(1, 21))
    assert first.target_states.shape == (20, 22)
    assert first.target_interaction_mode.shape == (20,)
    assert first.target_absorbing.shape == (20,)
    assert not hasattr(first, "realized_delay_tick")
    np.testing.assert_array_equal(
        first.remaining_actions,
        tail.expert_actions[25:50],
    )

    last = materialize_belief_training_context(
        index=indices[-1],
        tail_view=tail,
        temporal_contract=contract,
        latency_law=law,
    )
    assert np.all(last.target_absorbing)
    assert np.all(last.target_interaction_mode == InteractionMode.TERMINAL)
    np.testing.assert_array_equal(last.target_states[:, 7:14], 0.0)
    np.testing.assert_array_equal(last.target_states[:, 15], 0.0)
    np.testing.assert_array_equal(last.target_states[:, 19:22], 0.0)
    assert last.remaining_actions.shape == (25, 7)


def test_delay_query_sampling_is_seeded_weighted_and_target_aligned(
    tmp_path: Path,
) -> None:
    from latency_meta_mdp.belief_training_data import (
        build_belief_training_indices,
        materialize_belief_training_context,
        sample_delay_queries,
    )

    tail, contract, law = _tail(tmp_path)
    index = build_belief_training_indices(
        tail_view=tail,
        temporal_contract=contract,
    )[0]
    context = materialize_belief_training_context(
        index=index,
        tail_view=tail,
        temporal_contract=contract,
        latency_law=law,
    )
    left = sample_delay_queries(
        context=context,
        rng=np.random.default_rng(123),
        query_count=4,
    )
    right = sample_delay_queries(
        context=context,
        rng=np.random.default_rng(123),
        query_count=4,
    )

    np.testing.assert_array_equal(left.delay_ticks, right.delay_ticks)
    np.testing.assert_array_equal(left.target_states, right.target_states)
    assert left.delay_ticks.shape == (4,)
    assert left.target_states.shape == (4, 22)
    assert left.interaction_mode.shape == (4,)
    assert left.absorbing.shape == (4,)
    for row, delay_tick in enumerate(left.delay_ticks):
        target_index = int(delay_tick) - 1
        np.testing.assert_array_equal(
            left.target_states[row], context.target_states[target_index]
        )
