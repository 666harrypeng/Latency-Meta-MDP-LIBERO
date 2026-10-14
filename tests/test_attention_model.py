import torch
from attention_fixtures import make_batch


def test_each_history_layer_rereads_full_inputs_and_all_branches_receive_gradient():
    from latency_meta_mdp.meta.information.attention_model import AttentionStageQ

    batch, config = make_batch()
    model = AttentionStageQ(config)
    seen = []
    hooks = [
        layer.register_forward_pre_hook(lambda m, args: seen.append(args[1]))
        for layer in model.history
    ]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(2):
        optimizer.zero_grad()
        loss = (model(batch) - 1).square().mean()
        loss.backward()
        optimizer.step()
    assert seen[0] is seen[1]
    assert seen[0].shape[1] > 1000
    for layer in (model.visual.channel, model.history, model.forecast, model.critical):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in layer.parameters())
    assert model.queries.shape == (8, 32)
    assert model(batch).shape == (2, 11)
    for hook in hooks:
        hook.remove()


def test_batch_and_single_inference_match_and_padding_cannot_change_values():
    from latency_meta_mdp.meta.information.attention_model import AttentionStageQ
    from latency_meta_mdp.meta.information.memory_schema import MemoryBatch

    batch, config = make_batch(missing_visual=True, missing_forecast=True)
    model = AttentionStageQ(config).eval()
    for head in model.heads:
        torch.nn.init.normal_(head.weight, std=0.02)
    with torch.no_grad():
        expected = model(batch)
        single = MemoryBatch({k: v[:1] for k, v in batch.tensors.items()}, batch.refs[:1])
        torch.testing.assert_close(expected[:1], model(single), atol=2e-6, rtol=2e-5)
        batch.tensors["visual"].fill_(1000)
        batch.tensors["forecast"].fill_(-1000)
        torch.testing.assert_close(expected, model(batch))
        assert torch.isfinite(expected).all()
