from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from latency_meta_mdp.belief_data import load_belief_episode
from latency_meta_mdp.hf_dino_encoder import HfDinoPatchEncoder
from latency_meta_mdp.vision_encoder import load_vision_encoder_spec


def test_pinned_dinov3_extracts_full_frozen_spatial_grid() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("real DINOv3 integration requires CUDA")
    episode_path = Path(
        "outputs/bulk/expert/panda-ball-first-tranche-e58eee5/"
        "attempts/L1/seed_001000"
    )
    if not episode_path.is_dir():
        pytest.skip("real DINOv3 integration requires the local first tranche")
    spec = load_vision_encoder_spec(
        Path("configs/vision/dinov3_vits16_lvd1689m_224_v1.yaml")
    )
    encoder = HfDinoPatchEncoder.from_pretrained(
        spec=spec,
        device="cuda",
        local_files_only=True,
    )
    episode = load_belief_episode(episode_path)
    images = np.stack(
        (
            episode.deployment.agentview_rgb[0],
            episode.deployment.wrist_rgb[0],
        ),
        axis=0,
    )

    first = encoder.encode_numpy(images)
    second = encoder.encode_numpy(images)

    assert first.shape == (2, 196, 384)
    assert first.dtype == np.float16
    assert np.all(np.isfinite(first))
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first[0], first[1])
    assert encoder.runtime_info.device == "cuda"
    assert encoder.runtime_info.compute_dtype == "float16"
