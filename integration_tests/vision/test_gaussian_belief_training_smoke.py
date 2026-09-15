from __future__ import annotations

from pathlib import Path

import pytest

from latency_meta_mdp.data.vision.contracts import load_vision_encoder_spec
from latency_meta_mdp.legacy.vision_probe_data import ProbeSplit


def test_real_l1_gaussian_belief_batch_backpropagates() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Gaussian belief training smoke requires CUDA")
    source = Path("outputs/bulk/expert/panda-ball-first-tranche-e58eee5/manifest.json")
    cache = Path(
        "outputs/derived/vision_features/dinov3-vits16-first-tranche-54b9562/manifest.json"
    )
    if not source.is_file() or not cache.is_file():
        pytest.skip("Gaussian belief training smoke requires the local first tranche")
    from latency_meta_mdp.legacy.belief_feature_corpus import load_level_feature_belief_corpus
    from latency_meta_mdp.legacy.gaussian_belief_config import load_gaussian_belief_config
    from latency_meta_mdp.legacy.gaussian_belief_model import (
        GaussianBeliefModel,
        diagonal_gaussian_nll,
    )
    from latency_meta_mdp.legacy.gaussian_belief_training import collate_gaussian_belief_items
    from latency_meta_mdp.legacy.gaussian_belief_training_data import (
        GaussianBeliefDataset,
        build_gaussian_belief_normalization,
    )

    config = load_gaussian_belief_config(
        Path("configs/legacy/belief/dinov3_gaussian_belief_v1.yaml")
    )
    corpus = load_level_feature_belief_corpus(
        project_root=Path.cwd(),
        source_bulk_manifest=source,
        cache_run_manifest=cache,
        expected_spec=load_vision_encoder_spec(
            Path("configs/models/vision/dinov3_vits16_lvd1689m_224_v1.yaml")
        ),
        temporal_config_path=Path("configs/contracts/temporal/h50_e25_d20_k6_v1.yaml"),
        latency_law_path=Path("configs/runtime/latency/truncated_beta_5_26_400ms_v1.yaml"),
        level=1,
    )
    dataset = GaussianBeliefDataset(
        corpus=corpus,
        split=ProbeSplit.TRAIN,
        normalization=build_gaussian_belief_normalization(corpus),
        config=config,
        exhaustive_queries=False,
    )
    batch = collate_gaussian_belief_items([dataset[0], dataset[1]])
    model = GaussianBeliefModel(config).cuda()

    mean, log_std, belief = model(
        vision_history=batch.vision_history.cuda(),
        proprio_history=batch.proprio_history.cuda(),
        remaining_actions=batch.remaining_actions.cuda(),
        latency_probabilities=batch.latency_probabilities.cuda(),
        delay_ticks=batch.delay_ticks.cuda(),
    )
    loss = diagonal_gaussian_nll(
        mean=mean,
        log_std=log_std,
        target=batch.target_states.cuda(),
    )
    loss.backward()

    assert belief.shape == (2, 4, 128)
    assert mean.shape == (2, 4, 22)
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.decoder.parameters())
