import torch


def test_physical_time_rope_common_clock_shift_preserves_attention():
    from latency_meta_mdp.meta.information.attention import rotate_time

    torch.manual_seed(3)
    q, k = torch.randn(2, 64), torch.randn(2, 64)
    tq, tk = torch.tensor([0.0, 0.0]), torch.tensor([-1.0, 10.0])
    a = (rotate_time(q, tq) * rotate_time(k, tk)).sum(-1)
    b = (rotate_time(q, tq + 100) * rotate_time(k, tk + 100)).sum(-1)
    torch.testing.assert_close(a, b, atol=2e-4, rtol=2e-4)


def test_masked_history_does_not_change_query_and_empty_bank_is_finite():
    from latency_meta_mdp.meta.information.attention import HistoryBlock

    torch.manual_seed(1)
    block = HistoryBlock(32, 4, 64)
    query, bank = torch.randn(2, 8, 32), torch.randn(2, 5, 32)
    times, mask = torch.arange(5).expand(2, -1).float(), torch.ones(2, 5, dtype=torch.bool)
    baseline = block(query, bank, times, mask)
    padded = torch.cat((bank, torch.randn(2, 3, 32) * 100), dim=1)
    after = block(
        query,
        padded,
        torch.cat((times, torch.zeros(2, 3)), 1),
        torch.cat((mask, torch.zeros(2, 3, dtype=torch.bool)), 1),
    )
    torch.testing.assert_close(baseline, after)
    assert torch.isfinite(block(query, bank, times, torch.zeros_like(mask))).all()


def test_visual_tokenizer_keeps_each_frame_and_view_independent():
    from latency_meta_mdp.meta.information.attention import VisualTokenizer

    torch.manual_seed(0)
    tokenizer = VisualTokenizer(32, 4, 16)
    frames = torch.randn(2, 2, 196, 384)
    dense = tokenizer(frames, source_kind=0, dense=True)
    pooled = tokenizer(frames, source_kind=0, dense=False)
    assert dense.shape == (2, 2, 196, 32)
    assert pooled.shape == (2, 2, 16, 32)
    frames[1] *= 10
    torch.testing.assert_close(pooled[0], tokenizer(frames, source_kind=0, dense=False)[0])
