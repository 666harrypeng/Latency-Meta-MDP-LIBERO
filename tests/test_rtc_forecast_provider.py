import dataclasses
import itertools
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from latency_meta_mdp.runtime.policy_execution import PolicyObservation
from latency_meta_mdp.runtime.rtc_protocol import RtcInferenceContext


def _obs(tick):
    return PolicyObservation(
        tick,
        np.full((8, 8, 3), tick, np.uint8),
        np.full((8, 8, 3), tick + 1, np.uint8),
        np.full(16, tick, np.float32),
    )


def _context(tick, q=7):
    return RtcInferenceContext(0, tick, _obs(tick), 0, np.zeros((50, 7)), np.arange(50) < 25, q)


def test_provider_encodes_only_requested_real_history_and_shares_batch_engine():
    from test_action_conditioned_jepa_rollout import _normalization

    from latency_meta_mdp.belief.jepa.contracts import FutureLatentPrediction
    from latency_meta_mdp.belief.jepa.forecast_provider import (
        DirectForecastProvider,
        FrozenForecastEngine,
    )

    class Encoder:
        calls = []

        def encode(self, rgb):
            self.calls.append(rgb.copy())
            return torch.tensor(rgb[:, 0, 0, 0]).half()[:, None, None].expand(-1, 196, 384)

    class Predictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.zeros(()))

        def predict_at(self, query):
            self.query = query
            return FutureLatentPrediction(
                query.source_ticks,
                query.source_ticks + query.query_ticks,
                query.vision_history[:, -1],
                query.proprio_history[:, -1],
            )

    class Decoder(torch.nn.Module):
        calls = 0

        def forward(self, visual):
            self.calls += 1
            return torch.full((visual.shape[0], 2, 3, 224, 224), 0.5)

    encoder, model = Encoder(), Predictor()
    engine = FrozenForecastEngine(model, Decoder(), device="cpu")
    provider = DirectForecastProvider(
        engine, encoder=encoder, normalization=_normalization(), episode_id="episode"
    )
    for tick in range(11):
        provider.observe(_obs(tick), None if tick == 0 else np.full(7, tick / 20, np.float32))
    assert not encoder.calls
    forecast = provider.predict(_context(10))
    assert len(encoder.calls) == 1
    np.testing.assert_array_equal(encoder.calls[0][:, 0, 0, 0], [2, 3, 6, 7, 10, 11])
    np.testing.assert_allclose(
        model.query.executed_controls[0, :, :, 0].flatten(), np.arange(3, 11) / 20
    )
    assert (
        forecast.source_tick == 10 and forecast.target_tick == 17 and forecast.buffer_version == 0
    )
    assert forecast.rgb.shape == (2, 224, 224, 3)
    assert forecast.rgb.dtype == np.uint8 and not forecast.rgb.flags.writeable
    assert not model.parameter.requires_grad
    from latency_meta_mdp.runtime.rtc_protocol import RtcForecastContext

    c = _context(10)
    before = engine.decoder.calls
    packet = provider.prepare(
        RtcForecastContext(
            c.origin_tick,
            c.observation,
            c.buffer_version,
            c.previous_actions,
            c.previous_action_mask,
            c.estimated_delay_ticks,
        )
    )
    assert engine.decoder.calls == before
    assert packet.current_visual_latents.shape == (2, 196, 384)
    assert packet.future_visual_latents.shape == (2, 196, 384)
    assert packet.future_proprio.shape == (16,)
    assert not hasattr(packet.context, "request_id")
    assert not hasattr(packet.context, "realized_delay_ticks")
    decoded = packet.decode(c)
    np.testing.assert_array_equal(decoded.rgb, forecast.rgb)
    np.testing.assert_array_equal(decoded.proprio, forecast.proprio)
    assert packet.decode(c) is decoded and engine.decoder.calls == before + 1
    for changed in (
        dataclasses.replace(c, buffer_version=1),
        dataclasses.replace(c, previous_actions=np.ones((50, 7))),
    ):
        with pytest.raises(ValueError, match="snapshot"):
            packet.decode(changed)
    with pytest.raises(ValueError, match="source"):
        provider.predict(_context(9))
    with pytest.raises(ValueError, match="source differs"):
        provider.predict(
            dataclasses.replace(
                _context(10), observation=dataclasses.replace(_obs(10), state=np.ones(16))
            )
        )
    with pytest.raises(ValueError, match="consecutive"):
        provider.observe(_obs(12), np.zeros(7))
    provider.reset(episode_id="next")
    provider.observe(_obs(0), None)
    assert not provider.predict(_context(0)).available


def test_forecast_packet_cannot_be_attached_to_another_request_context():
    from latency_meta_mdp.data.forecast.samples import DecodedForecast

    packet = DecodedForecast(10, 17, 7, 0, None, None)
    context = dataclasses.replace(_context(10), forecast=packet)
    assert context.forecast is packet
    for change in ({"source_tick": 9}, {"buffer_version": 2}, {"requested_query_ticks": 8}):
        with pytest.raises(ValueError, match="forecast"):
            dataclasses.replace(context, forecast=dataclasses.replace(packet, **change))


def test_logical_forecast_uses_completed_estimate_and_records_stage_order():
    from latency_meta_mdp.data.forecast.samples import DecodedForecast
    from latency_meta_mdp.envs.control import load_action_contract
    from latency_meta_mdp.runtime.policy_execution import (
        FixedCursorScheduler,
        LogicalPolicyRuntime,
        PhysicalStepResult,
    )
    from latency_meta_mdp.runtime.rtc_protocol import load_rtc_client_config

    seen, observations = [], []
    clock = SimpleNamespace(tick=0)

    class Provider:
        def observe(self, observation, previous):
            observations.append(observation.formal_tick)

        def predict(self, context):
            seen.append(context.estimated_delay_ticks)
            assert not hasattr(context, "realized_delay_ticks")
            return DecodedForecast(
                context.origin_tick,
                context.origin_tick + context.estimated_delay_ticks,
                context.estimated_delay_ticks,
                context.buffer_version,
                np.zeros((2, 224, 224, 3), np.uint8),
                np.zeros(16, np.float32),
            )

    def policy(obs, context):
        assert context.forecast is not None
        return np.zeros((50, 7))

    runtime = LogicalPolicyRuntime(
        action_contract=load_action_contract(
            Path("configs/runtime/control/panda_osc_pose_delta_v1.yaml")
        ),
        client_config=dataclasses.replace(
            load_rtc_client_config(Path("configs/runtime/client/rtc_observation_time_h50_v1.yaml")),
            initial_delay_ticks=(4,),
        ),
        simulation_time_reader=lambda: clock.tick * 20000,
        delay_sampler=lambda: 12,
        policy=policy,
        bootstrap_policy=lambda obs: np.zeros((50, 7)),
        scheduler=FixedCursorScheduler(2),
        forecast_provider=Provider(),
        policy_alignment="observation_time",
        monotonic_ns=itertools.count(0, 10).__next__,
    )
    runtime.bootstrap(lambda: _obs(0))

    def execute(action):
        clock.tick += 1
        return PhysicalStepResult(0, clock.tick == 27)

    while clock.tick < 27:
        runtime.step(formal_tick=clock.tick, observe=lambda: _obs(clock.tick), execute=execute)
    assert seen == [4, 12]
    assert observations == list(range(27))
    assert runtime.summary()["forecast_calls"] == 2
    for row in runtime.request_ledger():
        assert (
            row["request_launch_ns"]
            <= row["forecast_start_ns"]
            <= row["forecast_complete_ns"]
            <= row["policy_launch_ns"]
        )
        assert (
            row["forecast_target_tick"] == row["source_formal_tick"] + row["estimated_delay_ticks"]
        )
        assert np.shape(row["previous_action_buffer"]) == (50, 7)
