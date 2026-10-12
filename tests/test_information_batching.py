"""Batching must preserve online causal recurrence and all trainable gradients."""

import copy

import numpy as np
import pytest
import torch

from latency_meta_mdp.meta.information.model import StageQ
from latency_meta_mdp.meta.information.schema import ActorPacket


def packets():
    rng = np.random.default_rng(12)
    tokens = rng.normal(size=(2, 196, 384)).astype(np.float32)
    return [
        ActorPacket(10, "G"),
        ActorPacket(10, "O", current_tick=10, current_tokens=tokens),
        ActorPacket(
            10,
            "P",
            current_tick=10,
            current_tokens=tokens,
            forecast_tick=10,
            forecast_tokens=-tokens,
        ),
        ActorPacket(14, "G", current_tick=10, current_tokens=tokens),
        ActorPacket(18, "G"),
    ]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_block_matches_online_values_hidden_and_gradients_without_future_leakage(
    device, monkeypatch
):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(9)
    reference = StageQ().to(device)
    for head in reference.heads:
        torch.nn.init.normal_(head.weight, std=0.03)
    batched = copy.deepcopy(reference)
    rows = packets()
    h0 = torch.randn(128, device=device)
    h = h0.clone()
    expected = []
    for packet in rows:
        q, h = reference.step(packet, h)
        expected.append(q)
    expected = torch.stack(expected)
    actual, final = batched.block(rows, h0.clone())
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(final, h, atol=2e-6, rtol=2e-5)
    expected.square().sum().backward()
    actual.square().sum().backward()
    for (name, p), (_, other) in zip(reference.named_parameters(), batched.named_parameters()):
        torch.testing.assert_close(other.grad, p.grad, atol=3e-6, rtol=3e-4, msg=name)
    with torch.no_grad():
        prefix, _ = batched.block(rows[:3], h0)
    torch.testing.assert_close(prefix, actual[:3])


def test_absent_heads_stay_without_gradient_and_block_carries_hidden():
    model = StageQ()
    rows = packets()
    q, h = model.block(rows[:1])
    q.sum().backward()
    assert model.heads[0].weight.grad is not None
    assert model.heads[1].weight.grad is None
    assert model.heads[2].weight.grad is None
    with torch.no_grad():
        tail, final = model.block(rows[1:], h.detach())
        complete, last = model.block(rows)
    torch.testing.assert_close(torch.cat([q, tail]), complete)
    torch.testing.assert_close(final, last)
