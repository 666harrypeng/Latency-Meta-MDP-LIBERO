"""Microbatch is an execution choice, not an optimizer budget."""

import pytest


def test_batch_modes_and_rank_divisibility():
    from latency_meta_mdp.meta.information.fqi_batch import batch_candidates

    assert batch_candidates("full", 256, 1) == [256]
    assert batch_candidates("full", 256, 2) == [128]
    assert batch_candidates("128", 256, 1) == [128]
    assert batch_candidates(128, 256, 1) == [128]
    assert batch_candidates("auto", 256, 1) == [256, 128, 64, 32, 16, 8, 4, 2, 1]
    for bad in ("wrong", 0, -1, 127):
        with pytest.raises(ValueError):
            batch_candidates(bad, 256, 1)
    with pytest.raises(ValueError):
        batch_candidates("full", 256, 3)


def test_auto_probe_preserves_weights_rng_and_gradients():
    import copy

    import torch
    from attention_fixtures import rich_replay, small_config

    from latency_meta_mdp.meta.information.attention_model import AttentionStageQ
    from latency_meta_mdp.meta.information.fqi_batch import select_microbatch

    if not torch.cuda.is_available():
        pytest.skip("CUDA memory probe required")
    model = AttentionStageQ(small_config()).cuda()
    state = copy.deepcopy(model.state_dict())
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    selected, report = select_microbatch(
        "auto",
        model=model,
        replay=rich_replay(model.config),
        global_batch=4,
        world_size=1,
        indices=[0, 1, 2, 0],
        bf16=True,
    )
    assert selected == 4 and report[0]["status"] == "fits"
    for key, value in state.items():
        torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)
    assert all(p.grad is None for p in model.parameters())
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
