"""Fixed-source numerical comparisons and warm service timing, outside training replay."""

import json
from collections import deque
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from latency_meta_mdp.belief.jepa.forecast_provider import DirectForecastProvider
from latency_meta_mdp.meta.information.perception import PerceptionAccess
from latency_meta_mdp.meta.information.records import StageRecorder


def _difference(a, b):
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def time_service(call, *, device, warmups=10, repeats=50):
    def sync():
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)

    durations = []
    for i in range(warmups + repeats):
        sync()
        start = perf_counter()
        call()
        sync()
        elapsed = perf_counter() - start
        if i >= warmups:
            durations.append(elapsed)
    return dict(
        warmups=warmups,
        repeats=repeats,
        seconds=durations,
        median_seconds=float(np.median(durations)),
        p95_seconds=float(np.percentile(durations, 95)),
    )


class _QueryCapture:
    def __init__(self, engine):
        self.engine, self.predictor, self.device = engine, engine.predictor, engine.device
        self.query = None

    def predict_latents(self, query):
        self.query = query
        return self.engine.predict_latents(query)

    def decode(self, visual):
        return self.engine.decode(visual)


class SnapshotProbe:
    """Diagnostic-only wrapper. Extra calls are reported separately, never training cost."""

    def __init__(
        self, access, *, engine, encoder, normalization, output_dir, infer, calibrate=False
    ):
        self.access, self.engine, self.encoder = access, engine, encoder
        self.normalization, self.output_dir = normalization, Path(output_dir)
        self.infer, self.calibrate = infer, calibrate
        self.history, self.controls = deque(maxlen=9), deque(maxlen=8)
        self.report = None

    def ingest(self, observation, previous_action):
        self.access.ingest(observation, previous_action)
        self.history.append(observation)
        if previous_action is not None:
            self.controls.append(np.asarray(previous_action).copy())

    def observe(self, context, *, prepare_history=False):
        return self.access.observe(context, prepare_history=prepare_history)

    def predict(self, context):
        result = self.access.predict(context)
        if self.report is None:
            self._check(context)
        return result

    def _populate(self, provider, *, access):
        for i, observation in enumerate(self.history):
            previous = None if i == 0 else self.controls[i - 1]
            if access:
                provider.ingest(observation, previous)
            else:
                provider.observe(observation, previous)

    def _check(self, context):
        self.output_dir.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(
            self.output_dir / "source.npz",
            ticks=np.array([o.formal_tick for o in self.history]),
            images=np.stack([[o.image, o.wrist_image] for o in self.history]),
            states=np.stack([o.state for o in self.history]),
            controls=np.stack(self.controls),
            previous_actions=context.previous_actions,
            previous_action_mask=context.previous_action_mask,
            buffer_version=context.buffer_version,
            prompt=str(context.observation.prompt),
        )
        capture = _QueryCapture(self.engine)
        reference = DirectForecastProvider(
            capture, encoder=self.encoder, normalization=self.normalization, episode_id="snapshot"
        )
        self._populate(reference, access=False)
        comparisons = []
        for q in (0, 9, 20):
            ctx = replace(context, estimated_delay_ticks=q)
            expected = reference.prepare(ctx)
            expected_rgb = expected.decode(ctx).rgb
            expected_actions = self.infer(ctx, expected.decode(ctx))
            paths = {}
            candidates = {"reference_repeat": reference.prepare(ctx)}
            for mode in ("bound", "split"):
                access = PerceptionAccess(
                    self.engine,
                    self.encoder,
                    self.normalization,
                    StageRecorder(f"diagnostic-{q}-{mode}"),
                )
                self._populate(access, access=True)
                access.observe(ctx, prepare_history=mode == "bound")
                candidates[mode] = access.predict(ctx)
            for name, actual in candidates.items():
                decoded = actual.decode(ctx)
                paths[name] = dict(
                    current_max_abs=_difference(
                        expected.current_visual_latents, actual.current_visual_latents
                    ),
                    forecast_max_abs=_difference(
                        expected.future_visual_latents, actual.future_visual_latents
                    ),
                    proprio_max_abs=_difference(expected.future_proprio, actual.future_proprio),
                    rgb_max_abs=_difference(expected_rgb, decoded.rgb),
                    rgb_changed_fraction=float(np.mean(expected_rgb != decoded.rgb)),
                    actions_max_abs=_difference(expected_actions, self.infer(ctx, decoded)),
                )
            comparisons.append(dict(q=q, paths=paths))
        self.report = dict(
            source_tick=context.origin_tick,
            comparisons=comparisons,
            scope="fixed-source diagnostics, excluded from rollout cost and training",
        )
        (self.output_dir / "comparison.json").write_text(json.dumps(self.report, indent=2) + "\n")
        print("SNAPSHOT_COMPARISON " + json.dumps(self.report), flush=True)
        if self.calibrate:
            self._calibrate(ctx, expected, capture.query)

    def _calibrate(self, context, forecast, query):
        from latency_meta_mdp.meta.information.model import StageQ
        from latency_meta_mdp.meta.information.schema import ActorPacket

        images = np.stack([im for o in list(self.history)[::4] for im in (o.image, o.wrist_image)])
        prediction = self.engine.predict_latents(query)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = StageQ().to(self.engine.device).eval().requires_grad_(False)
        packet = ActorPacket(
            context.origin_tick,
            "P",
            current_tokens=forecast.current_visual_latents,
            forecast_tokens=forecast.future_visual_latents,
            current_tick=context.origin_tick,
            forecast_tick=context.origin_tick,
            forecast_buffer_version=context.buffer_version,
            buffer_version=context.buffer_version,
            current_proprio=context.observation.state,
            observed_proprio=context.observation.state,
            forecast_proprio=forecast.future_proprio,
            buffer=context.previous_actions,
            buffer_mask=context.previous_action_mask,
            estimated_delay=context.estimated_delay_ticks,
        )
        with torch.inference_mode():
            calls = {
                "dino_batch6": lambda: self.encoder.encode(images),
                "dino_batch2": lambda: self.encoder.encode(images[-2:]),
                "dino_batch4": lambda: self.encoder.encode(images[:4]),
                "jepa": lambda: self.engine.predict_latents(query),
                "decoder": lambda: self.engine.decode(prediction.visual_latents),
                "vla": lambda: self.infer(context, forecast.decode(context)),
                "meta_q": lambda: model.step(packet),
            }
            profile = {}
            for name, call in calls.items():
                profile[name] = time_service(call, device=self.engine.device)
                print(f"CALIBRATED {name} {profile[name]['median_seconds']:.6f}s", flush=True)
        (self.output_dir / "service-profile.json").write_text(
            json.dumps(
                dict(
                    services=profile,
                    device=str(self.engine.device),
                    source_tick=context.origin_tick,
                    scope="warm model services only; excludes capture and system overhead",
                ),
                indent=2,
            )
            + "\n"
        )
