"""Private raw history with explicit, metered Observe and Predict access."""

from collections import deque
from contextlib import contextmanager

import numpy as np
import torch

from latency_meta_mdp.belief.jepa.forecast_provider import DirectForecastProvider


@contextmanager
def measured(recorder, component, device, **metadata):
    """Synchronize GPU service work; logical simulation time remains independent."""
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    with recorder.measure(component, **metadata):
        try:
            yield
        finally:
            if device.type == "cuda":
                torch.cuda.synchronize(device)


class _Engine:
    def __init__(self, engine, recorder):
        self.engine, self.recorder = engine, recorder
        self.predictor, self.device = engine.predictor, engine.device

    def predict_latents(self, query):
        with measured(self.recorder, "jepa", self.device):
            return self.engine.predict_latents(query)

    def decode(self, visual):
        with measured(self.recorder, "decoder", self.device):
            return self.engine.decode(visual)


class _FrameCache:
    def __init__(self, encoder, recorder, device):
        self.encoder, self.recorder, self.device = encoder, recorder, device
        self.cache = {}
        self.keys = ()

    def encode(self, images):
        if len(images) != len(self.keys):
            raise ValueError("private encoder frame identity mismatch")
        missing = [i for i, key in enumerate(self.keys) if key not in self.cache]
        if missing:
            with measured(self.recorder, "dino", self.device, batch_size=len(missing)):
                with torch.inference_mode():
                    tokens = self.encoder.encode(images[missing])
        with measured(
            self.recorder, "feature_cache", self.device, cache_hits=len(images) - len(missing)
        ):
            if missing:
                for i, token in zip(missing, tokens, strict=True):
                    self.cache[self.keys[i]] = token.detach().clone()
            return torch.stack([self.cache[k] for k in self.keys])


class PerceptionAccess:
    """One episode; never pass this object or its history to the scheduler."""

    def __init__(self, engine, encoder, normalization, recorder, *, source_archive=None):
        self._encoder = _FrameCache(encoder, recorder, engine.device)
        self._provider = DirectForecastProvider(
            _Engine(engine, recorder),
            encoder=self._encoder,
            normalization=normalization,
            episode_id=recorder.episode_id,
        )
        self._history = deque(maxlen=9)
        self._controls = deque(maxlen=8)
        self._recorder, self._archive = recorder, source_archive
        self._observed_tick = -1
        self._observed_context = None

    def ingest(self, observation, previous_action):
        self._provider.observe(observation, previous_action)
        self._history.append(observation)
        if previous_action is not None:
            self._controls.append(np.asarray(previous_action).copy())
        oldest = self._history[0].formal_tick
        self._encoder.cache = {k: v for k, v in self._encoder.cache.items() if k[0] >= oldest}

    def _validate_observation(self, context):
        if not self._history or self._history[-1].formal_tick != context.origin_tick:
            raise ValueError("observe requires the latest real observation")
        if any(
            not np.array_equal(getattr(self._history[-1], k), getattr(context.observation, k))
            for k in ("image", "wrist_image", "state", "prompt")
        ):
            raise ValueError("observe context differs from captured observation")

    def _frames(self, history):
        self._encoder.keys = tuple((obs.formal_tick, view) for obs in history for view in (0, 1))
        return np.stack([im for obs in history for im in (obs.image, obs.wrist_image)])

    def prediction_ready(self, context):
        """Input availability only; neither a prediction nor a privileged task signal."""
        return bool(
            context.origin_tick >= 10
            and len(self._history) >= 9
            and self._history[-1].formal_tick == context.origin_tick
            and self._observed_tick == context.origin_tick
            and context.previous_action_mask[: context.estimated_delay_ticks].all()
        )

    def observe(
        self, context, *, prepare_history=False, return_history=False, allow_partial_history=False
    ):
        self._validate_observation(context)
        if not allow_partial_history and (context.origin_tick < 10 or len(self._history) < 9):
            raise ValueError("observe requires real warmup history")
        history = list(self._history)[::-4][::-1] if prepare_history else [self._history[-1]]
        tokens = self._encoder.encode(self._frames(history))
        if not return_history:
            tokens = tokens[-2:]
        with measured(self._recorder, "feature_transfer", self._encoder.device):
            current = tokens.float().cpu().numpy().copy()
        current.setflags(write=False)
        self._observed_tick = context.origin_tick
        self._observed_context = context
        if self._archive is not None:
            with self._recorder.measure("source_archive"):
                self._archive.record(history, self._controls, context, kind="observe")
        if return_history:
            return dict(
                source_ticks=tuple(obs.formal_tick for obs in history),
                visual=current.reshape(len(history), 2, 196, 384),
                proprio=np.stack([obs.state for obs in history]),
            )
        return current

    def predict(self, context):
        self._validate_observation(context)
        if self._observed_tick != context.origin_tick:
            raise ValueError("predict requires an explicit current observe")
        source = self._observed_context
        if (
            source.buffer_version != context.buffer_version
            or source.estimated_delay_ticks != context.estimated_delay_ticks
            or not np.array_equal(source.previous_actions, context.previous_actions)
            or not np.array_equal(source.previous_action_mask, context.previous_action_mask)
        ):
            raise ValueError("prediction snapshot changed after observe")
        history = list(self._history)[::4]
        self._frames(history)
        if self._archive is not None:
            with self._recorder.measure("source_archive"):
                self._archive.record(history, self._controls, context, kind="predict")
        with measured(self._recorder, "forecast_prepare_overhead", self._encoder.device):
            return self._provider.prepare(context)
