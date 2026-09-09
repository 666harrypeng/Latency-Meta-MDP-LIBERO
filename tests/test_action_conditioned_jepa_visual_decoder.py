import pytest
import torch

from latency_meta_mdp.belief.action_conditioned_jepa.visual_decoder import (
    DualViewVisualDecoder,
    VisualDecoderConfig,
    unpatchify_rgb,
    visual_reconstruction_loss,
)


def test_unpatchify_keeps_spatial_coordinates_and_rgb_channels():
    image = torch.arange(3 * 224 * 224).reshape(1, 3, 224, 224).float()
    patches = image.reshape(1, 3, 14, 16, 14, 16).permute(0, 2, 4, 3, 5, 1).reshape(1, 196, 768)
    torch.testing.assert_close(unpatchify_rgb(patches), image)


def test_decoder_preserves_camera_identity_and_detaches_belief_inputs():
    torch.manual_seed(4)
    decoder = DualViewVisualDecoder(VisualDecoderConfig(depth=1))
    z = torch.randn(1, 2, 196, 384, requires_grad=True)
    rgb = decoder(z)
    assert rgb.shape == (1, 2, 3, 224, 224)
    assert rgb.dtype == torch.float32 and torch.isfinite(rgb).all()
    changed = z.detach().clone()
    changed[:, 1, 97] += torch.linspace(-2, 2, 384)
    other = decoder(changed)
    torch.testing.assert_close(other[:, 0], rgb[:, 0], rtol=0, atol=0)
    assert not torch.equal(other[:, 1], rgb[:, 1])
    visual_reconstruction_loss(rgb, torch.zeros_like(rgb))["total"].backward()
    assert z.grad is None
    assert all(p.grad is not None for p in decoder.parameters())
    assert any(torch.count_nonzero(p.grad) for p in decoder.parameters())


def test_decoder_accepts_five_anchor_batch_without_merging_time_or_views():
    decoder = DualViewVisualDecoder(VisualDecoderConfig(depth=1)).eval()
    z = torch.randn(1, 5, 2, 196, 384)
    with torch.no_grad():
        together = decoder(z)
        separate = torch.stack([decoder(z[:, k]) for k in range(5)], dim=1)
    torch.testing.assert_close(together, separate, atol=1e-6, rtol=1e-5)
    with pytest.raises(ValueError):
        decoder(torch.randn(1, 2, 195, 384))


def test_loss_identity_and_spatial_edge_sensitivity():
    target = torch.rand(2, 2, 3, 224, 224)
    terms = visual_reconstruction_loss(target, target)
    assert all(x.item() == 0 for x in terms.values())
    changed = target.roll(1, dims=-1)
    assert visual_reconstruction_loss(changed, target)["edge"].item() > 0


def test_grouped_decoder_split_is_disjoint_and_reproducible():
    from latency_meta_mdp.belief.action_conditioned_jepa.visual_decoder_data import (
        partition_decoder_masters,
    )

    fit, heldout = partition_decoder_masters(tuple(range(10)), holdout_count=2, seed=17)
    assert len(fit) == 8 and len(heldout) == 2 and not set(fit) & set(heldout)
    assert set(fit) | set(heldout) == set(range(10))
    assert (fit, heldout) == partition_decoder_masters(tuple(range(10)), holdout_count=2, seed=17)


def test_decoder_dataset_pairs_exact_boundaries_without_copying_feature_cache(tmp_path):
    import json

    import numpy as np

    from latency_meta_mdp.belief.action_conditioned_jepa.visual_decoder_data import (
        VisualDecoderDataset,
    )

    z = np.lib.format.open_memmap(
        tmp_path / "features.npy", mode="w+", dtype="float16", shape=(3, 2, 196, 384)
    )
    rgb = np.lib.format.open_memmap(
        tmp_path / "rgb.npy", mode="w+", dtype="uint8", shape=(3, 2, 3, 224, 224)
    )
    for t in range(3):
        for v in range(2):
            z[t, v] = t * 10 + v
            rgb[t, v] = t * 10 + v
    z.flush()
    rgb.flush()
    m = {
        "complete": True,
        "episodes": [
            {
                "episode_id": "example",
                "master_index": 3,
                "partition": "fit",
                "boundary_count": 3,
                "feature_path": "features.npy",
                "rgb_path": "rgb.npy",
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    ds = VisualDecoderDataset(tmp_path / "manifest.json", project_root=tmp_path, partition="fit")
    assert len(ds) == 3
    features, pixels = ds[2]
    assert features[0, 0, 0].item() == pixels[0, 0, 0, 0].item() == 20
    assert features[1, 0, 0].item() == pixels[1, 0, 0, 0].item() == 21
    assert features.dtype == torch.float16 and pixels.dtype == torch.uint8
    assert ds.__getstate__()["_maps"] == {}
