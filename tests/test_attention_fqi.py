import numpy as np
import pytest
import torch


def test_sparse_bellman_backup_ignores_illegal_actions_and_terminal_continuation():
    from latency_meta_mdp.meta.information.attention_fqi import bellman_targets

    reward = torch.tensor([0.0, 0.1, 0.1])
    terminal = torch.tensor([False, False, True])
    q = torch.tensor([[0.2, 100.0], [0.3, 0.4], [100.0, 100.0]])
    legal = torch.tensor([[True, False], [True, True], [False, False]])
    torch.testing.assert_close(
        bellman_targets(reward, terminal, q, legal), torch.tensor([0.2, 0.5, 0.1])
    )
    with pytest.raises(ValueError, match="nonterminal"):
        bellman_targets(reward, terminal, q, torch.zeros_like(legal))


def test_group_sampler_weights_transitions_instead_of_episodes_and_resumes():
    from latency_meta_mdp.meta.information.attention_replay import DecisionReplay, DecisionSampler

    replay = DecisionReplay(
        [[{}], [{}] * 9, [{}] * 5, [{}] * 5], groups=["startup", "startup", "old", "latest"]
    )
    sampler = DecisionSampler(replay, seed=4)
    ids = sampler.sample(100000)
    counts = np.bincount(ids, minlength=20) / len(ids)
    assert counts[:10].sum() == pytest.approx(0.5, abs=0.01)
    assert counts[10:15].sum() == pytest.approx(0.25, abs=0.01)
    assert counts[15:].sum() == pytest.approx(0.25, abs=0.01)
    assert counts[1:10].sum() / counts[0] == pytest.approx(9, rel=0.05)
    state = sampler.state_dict()
    expected = sampler.sample(100)
    restored = DecisionSampler(replay, seed=0)
    restored.load_state_dict(state)
    np.testing.assert_array_equal(expected, restored.sample(100))


def test_fixed_target_fitting_updates_model_without_updating_target(tmp_path):
    from attention_fixtures import make_batch

    from latency_meta_mdp.meta.information.attention_fqi import fit_attention_iteration
    from latency_meta_mdp.meta.information.attention_model import AttentionStageQ
    from latency_meta_mdp.meta.information.attention_replay import DecisionReplay, DecisionSampler
    from latency_meta_mdp.meta.information.fqi import model_hash
    from latency_meta_mdp.meta.information.memory import MemoryStore
    from latency_meta_mdp.meta.information.memory_schema import AttentionPacket, DecisionRef

    _, config = make_batch()
    store = MemoryStore(config=config["memory"])
    store.reset("tiny")
    context = store.context(DecisionRef("tiny", 0), AttentionPacket(10, "G", decision_key=(10, 1)))
    row = dict(
        context=context,
        next_context=None,
        action=8,
        task_reward=0.1,
        terminated=True,
        legal=np.array([True] * 9 + [False, False]),
        next_legal=np.zeros(11, bool),
    )
    replay = DecisionReplay([[row]])
    torch.manual_seed(0)  # Fix initialization for the short loss-decrease regression.
    model = AttentionStageQ(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    before = model_hash(model)
    progress = []
    metrics = fit_attention_iteration(
        model,
        replay,
        optimizer=optimizer,
        optimizer_updates=4,
        global_batch=2,
        microbatch=1,
        sampler=DecisionSampler(replay, seed=1),
        conservative_weight=0.0,
        progress=progress.append,
    )
    assert metrics["target_hash_before"] == metrics["target_hash_after"] == before
    assert model_hash(model) != before
    assert metrics["optimizer_updates"] == 4 and metrics["transition_visits"] == 8
    assert metrics["final_loss"] < metrics["initial_loss"]
    assert progress[0]["phase"] == "target_progress"
    assert progress[0]["completed_decisions"] == 0
    assert progress[1]["phase"] == "targets"
    assert progress[1]["completed_decisions"] == progress[1]["unique_decisions"] == 1
    assert sum(event["phase"] == "fit" for event in progress) == 4
