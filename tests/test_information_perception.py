import numpy as np
import pytest
import torch
from test_action_conditioned_jepa_rollout import _normalization
from test_rtc_forecast_provider import _context, _obs

from latency_meta_mdp.belief.jepa.contracts import FutureLatentPrediction
from latency_meta_mdp.belief.jepa.forecast_provider import FrozenForecastEngine
from latency_meta_mdp.meta.information.perception import PerceptionAccess
from latency_meta_mdp.meta.information.records import StageRecorder


class Encoder:
    def __init__(self):
        self.batches = []

    def encode(self, images):
        self.batches.append(len(images))
        return torch.tensor(images[:, 0, 0, 0]).half()[:, None, None].expand(-1, 196, 384)


class Predictor(torch.nn.Module):
    def predict_at(self, q):
        self.query = q
        return FutureLatentPrediction(
            q.source_ticks,
            q.source_ticks + q.query_ticks,
            q.vision_history[:, -1],
            q.proprio_history[:, -1],
        )


class Decoder(torch.nn.Module):
    def forward(self, x):
        return torch.zeros(len(x), 2, 3, 224, 224)


@pytest.mark.parametrize("bound,batches", [(False, [2, 4]), (True, [6])])
def test_observe_and_predict_share_exact_frames_without_free_encoding(bound, batches):
    encoder = Encoder()
    engine = FrozenForecastEngine(Predictor(), Decoder(), device="cpu")
    access = PerceptionAccess(engine, encoder, _normalization(), StageRecorder("test"))
    for tick in range(11):
        access.ingest(_obs(tick), None if tick == 0 else np.zeros(7))
    assert encoder.batches == []
    current = access.observe(_context(10), prepare_history=bound)
    np.testing.assert_array_equal(current[:, 0, 0], [10, 11])
    forecast = access.predict(_context(10))
    assert encoder.batches == batches
    np.testing.assert_array_equal(engine.predictor.query.vision_history[0, :, 0, 0, 0], [2, 6, 10])
    assert forecast.available
    with pytest.raises(ValueError, match="observ"):
        access.predict(_context(9))


def test_prediction_rejects_changed_buffer_or_query_after_observe():
    from dataclasses import replace

    encoder = Encoder()
    engine = FrozenForecastEngine(Predictor(), Decoder(), device="cpu")
    access = PerceptionAccess(engine, encoder, _normalization(), StageRecorder("identity"))
    for tick in range(11):
        access.ingest(_obs(tick), None if tick == 0 else np.zeros(7))
    context = _context(10)
    access.observe(context)
    for change in (
        {"buffer_version": 1},
        {"estimated_delay_ticks": 8},
        {"previous_actions": np.ones((50, 7))},
    ):
        with pytest.raises(ValueError, match="snapshot"):
            access.predict(replace(context, **change))


def test_cached_overlap_and_transfers_are_metered():
    encoder = Encoder()
    engine = FrozenForecastEngine(Predictor(), Decoder(), device="cpu")
    recorder = StageRecorder("cached-overlap")
    access = PerceptionAccess(engine, encoder, _normalization(), recorder)
    for tick in range(15):
        access.ingest(_obs(tick), None if tick == 0 else np.zeros(7))
        if tick in (10, 14):
            access.observe(_context(tick), prepare_history=True)
            access.predict(_context(tick))
    assert encoder.batches == [6, 2]
    components = [op["component"] for op in recorder.operations]
    assert components.count("feature_cache") == 4
    assert components.count("feature_transfer") == 2
    assert components.count("forecast_prepare_overhead") == 2
