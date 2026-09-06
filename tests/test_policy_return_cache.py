import pickle
from pathlib import Path

import numpy as np
import pytest
from test_action_conditioned_jepa_data import _normalization, _record


def test_prediction_cache_is_indexed_immutable_and_spawn_safe(tmp_path):
    import torch

    from latency_meta_mdp.belief.action_conditioned_jepa.config import load_jepa_temporal_sampling
    from latency_meta_mdp.belief.action_conditioned_jepa.contracts import FutureLatentRollout
    from latency_meta_mdp.policy_return_cache import (
        NominalReturnPredictionCache,
        write_nominal_return_predictions,
    )

    record = _record(tmp_path, terminal_tick=13)
    sampling = load_jepa_temporal_sampling(
        Path("configs/belief/action_conditioned_jepa/stride4_80ms_history_160ms.yaml")
    )

    class Predictor:
        def rollout_native(self, context):
            assert context.executable_controls.shape[1:] == (5, 4, 7)
            b = context.batch_size
            return FutureLatentRollout(
                native_delay_ticks=torch.tensor([4, 8, 12, 16, 20], dtype=torch.int64),
                future_visual_latents=context.vision_history[:, -1:, ...]
                .expand(b, 5, 2, 196, 384)
                .clone(),
                future_proprio=torch.zeros((b, 5, 16), dtype=torch.float32),
            )

    bindings = {
        "checkpoint_sha256": "a" * 64,
        "source_manifest_sha256": "b" * 64,
        "split_manifest_sha256": "c" * 64,
        "vision_cache_manifest_sha256": "d" * 64,
        "normalization_sha256": "e" * 64,
        "temporal_config_sha256": "f" * 64,
        "model_config_sha256": "1" * 64,
    }
    path = tmp_path / "predictions"
    write_nominal_return_predictions(
        records=(record,),
        normalization=_normalization(record),
        sampling=sampling,
        model=Predictor(),
        device=torch.device("cpu"),
        batch_size=2,
        output_dir=path,
        bindings=bindings,
    )
    cache = NominalReturnPredictionCache(path, expected_bindings=bindings)
    np.testing.assert_array_equal(cache.read(record.episode_id, 11)["visual"], 11)
    encoded = pickle.dumps(cache)
    assert len(encoded) < 10_000
    restored = pickle.loads(encoded)
    np.testing.assert_array_equal(restored.read(record.episode_id, 12)["visual"], 12)
    with pytest.raises(ValueError, match="binding"):
        NominalReturnPredictionCache(
            path, expected_bindings={**bindings, "checkpoint_sha256": "0" * 64}
        )
    with pytest.raises(FileExistsError):
        write_nominal_return_predictions(
            records=(record,),
            normalization=_normalization(record),
            sampling=sampling,
            model=Predictor(),
            device=torch.device("cpu"),
            batch_size=2,
            output_dir=path,
            bindings=bindings,
        )
