"""Fresh policy Belief from real observation history, plus isolated privileged GT replay."""

from __future__ import annotations

import contextlib

import numpy as np
import torch

from latency_meta_mdp.belief.jepa.ar.return_belief import (
    assemble_return_latent_belief,
    quantize_d20_probabilities,
)
from latency_meta_mdp.belief.jepa.contracts import LaunchContextBatch
from latency_meta_mdp.runtime.policy_execution import policy_observation_from_snapshot


class PrivilegedDelayCoupler:
    """Reserve one actual logical delay for an oracle; the harness consumes that draw."""

    def __init__(self, sample_delay):
        self.sample_delay = sample_delay
        self._reserved = None

    def peek(self):
        if self._reserved is None:
            value = self.sample_delay()
            if type(value) is not int or not 1 <= value <= 20:
                raise ValueError("oracle delay coupler requires the original D20 support")
            self._reserved = value
        return self._reserved

    def __call__(self):
        value = self.peek()
        self._reserved = None
        return value


class PrefixBeliefProvider:
    """Attach the public law after prediction; keep oracle delay a separate input."""

    def __init__(self, provider, *, probabilities, known_delay_reader=None):
        self.privileged = bool(getattr(provider, "privileged", False))
        if known_delay_reader is not None and not self.privileged:
            raise ValueError("known delay requires a privileged evaluation provider")
        if getattr(provider, "known_delay_reader", None) is not None:
            raise ValueError("prefix conditioning requires the complete five-anchor forecast")
        pmf = np.array(probabilities, dtype=np.float32, copy=True)
        if (
            pmf.shape != (20,)
            or not np.isfinite(pmf).all()
            or np.any(pmf < 0)
            or not np.isclose(pmf.sum(), 1, atol=1e-6, rtol=0)
        ):
            raise ValueError("prefix law must be a normalized D20 PMF")
        pmf.setflags(write=False)
        self.probabilities = pmf
        self.provider = provider
        self.known_delay_reader = known_delay_reader

    def observe(self, observation, previous_action):
        self.provider.observe(observation, previous_action)

    def __call__(self, observation, executable_controls):
        forecast = self.provider(observation, executable_controls)
        if set(forecast) != {"visual", "proprio", "delay_ticks", "probabilities"}:
            raise ValueError("prefix provider requires the unprivileged forecast packet fields")
        result = {"return_belief": {**forecast, "latency_probabilities": self.probabilities.copy()}}
        if self.known_delay_reader is not None:
            delay = self.known_delay_reader()
            if type(delay) is not int or not 1 <= delay <= 20:
                raise ValueError("known delay must be an integer on D20")
            result["known_delay_oracle"] = {"known_delay_ticks": delay}
        return result


class FrozenJepaBelief:
    """Store 50 Hz RGB/proprio/controls; encode only the real frames a query needs."""

    privileged = False

    def __init__(self, *, model, encoder, normalization, sampling, probabilities, device):
        if getattr(model, "training", False):
            raise ValueError("online Belief must use an eval-mode frozen JEPA")
        self.model = model
        self.encoder = encoder
        self.normalization = normalization
        self.sampling = sampling
        self.device = torch.device(device)
        self.probabilities = torch.tensor(
            np.asarray(probabilities)[None], dtype=torch.float32, device=self.device
        )
        quantize_d20_probabilities(probabilities=self.probabilities, sampling=sampling)
        self.observations = {}
        self.controls = {}
        self.features = {}
        self.last_tick = -1

    def observe(self, observation, previous_action):
        tick = observation.formal_tick
        if tick != self.last_tick + 1 or (tick == 0) != (previous_action is None):
            raise ValueError("online history must contain every real boundary and executed control")
        self.observations[tick] = observation
        if previous_action is not None:
            control = np.asarray(previous_action, dtype=np.float32)
            if (
                control.shape != (7,)
                or not np.isfinite(control).all()
                or np.any(np.abs(control) > 1)
            ):
                raise ValueError("online executed control must be controller-native 7D")
            self.controls[tick - 1] = control.copy()
        self.last_tick = tick
        earliest = tick - self.sampling.history_span_ticks
        for collection in (self.observations, self.controls, self.features):
            for key in tuple(collection):
                if key < earliest:
                    del collection[key]

    @torch.inference_mode()
    def __call__(self, observation, executable_controls):
        tick = observation.formal_tick
        ticks = [tick + offset for offset in self.sampling.history_source_offsets]
        if tick != self.last_tick or tick < 10 or any(t not in self.observations for t in ticks):
            raise ValueError("Belief query lacks the complete real, history-ready source")
        missing = [t for t in ticks if t not in self.features]
        if missing:
            images = np.stack(
                [
                    image
                    for t in missing
                    for image in (self.observations[t].image, self.observations[t].wrist_image)
                ]
            )
            encoded = self.encoder.encode(images).to(self.device)
            if encoded.shape != (len(missing) * 2, 196, 384) or encoded.dtype != torch.float16:
                raise ValueError("encoder does not provide the admitted two-view DINO patches")
            for index, t in enumerate(missing):
                self.features[t] = encoded[index * 2 : index * 2 + 2].detach()
        state = np.stack([self.observations[t].state for t in ticks])
        controls = np.stack(
            [self.controls[t] for t in range(tick - self.sampling.history_span_ticks, tick)]
        )
        future = np.asarray(executable_controls, dtype=np.float32)
        if future.shape != (20, 7):
            raise ValueError("Belief needs the actual executable D20 control prefix")
        context = LaunchContextBatch(
            vision_history=torch.stack([self.features[t] for t in ticks])[None],
            proprio_history=torch.tensor(
                self.normalization.normalize(state)[None], device=self.device, dtype=torch.float32
            ),
            executed_controls=torch.tensor(
                controls.reshape(1, 2, 4, 7), device=self.device, dtype=torch.float32
            ),
            executable_controls=torch.tensor(
                future.reshape(1, 5, 4, 7), device=self.device, dtype=torch.float32
            ),
        )
        scope = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else contextlib.nullcontext()
        )
        with scope:
            rollout = self.model.rollout_native(context)
        belief = assemble_return_latent_belief(rollout, self.probabilities, sampling=self.sampling)
        return {
            "visual": belief.future_visual_latents[0].cpu().numpy(),
            "proprio": belief.future_proprio[0].cpu().numpy(),
            "delay_ticks": belief.delay_ticks.cpu().numpy(),
            "probabilities": belief.delay_probabilities[0].cpu().numpy(),
        }


class ReplayGroundTruthBelief:
    """Evaluation-only future under the actual buffer, replaying full controller history.

    Replaying avoids pretending that MuJoCo qpos/qvel alone restore controller,
    motion and contact state. Terminal extension is explicit task absorption for
    evaluation; it never writes training/source artifacts. Known delay must be
    supplied by an explicitly privileged harness for the very same request.
    """

    privileged = True

    def __init__(
        self,
        *,
        runtime_factory,
        current_snapshot,
        encoder,
        sampling,
        probabilities,
        known_delay_reader=None,
    ):
        self.runtime_factory = runtime_factory
        self.current_snapshot = current_snapshot
        self.encoder = encoder
        self.sampling = sampling
        self.known_delay_reader = known_delay_reader
        _, mass = quantize_d20_probabilities(
            probabilities=torch.tensor(np.asarray(probabilities)[None], dtype=torch.float32),
            sampling=sampling,
        )
        self.macro = mass[0].numpy()
        self.actions = []
        self.last_tick = -1
        self.last_query = {}

    def observe(self, observation, previous_action):
        if observation.formal_tick != self.last_tick + 1 or (self.last_tick < 0) != (
            previous_action is None
        ):
            raise ValueError("GT replay history must be complete")
        if previous_action is not None:
            self.actions.append(np.array(previous_action, copy=True))
        self.last_tick = observation.formal_tick

    def __call__(self, observation, executable_controls):
        if observation.formal_tick != len(self.actions):
            raise ValueError("GT query is not at the observed source boundary")
        controls = np.asarray(executable_controls, dtype=np.float32)
        if (
            controls.shape != (20, 7)
            or not np.isfinite(controls).all()
            or np.any(np.abs(controls) > 1)
        ):
            raise ValueError("GT future requires the actual controller-native D20 prefix")
        known = None if self.known_delay_reader is None else self.known_delay_reader()
        if known is not None and (type(known) is not int or not 1 <= known <= 20):
            raise ValueError("privileged realized delay must be on the original D20 grid")
        offsets = list(self.sampling.native_future_offsets) if known is None else [known]
        runtime = self.runtime_factory()
        try:
            snapshot = runtime.executor.initialize()
            for action in self.actions:
                snapshot = runtime.executor.step_formal(action)
            source = self.current_snapshot()
            replay_error = max(
                float(
                    np.max(
                        np.abs(
                            np.asarray(getattr(snapshot, name)) - np.asarray(getattr(source, name))
                        )
                    )
                )
                for name in ("qpos", "qvel", "actuator_ctrl")
            )
            if replay_error > 5e-7:
                raise ValueError("GT replay failed to restore the actual source")
            images = []
            states = []
            absorbing = []
            terminal = None
            outcome = None
            for offset in range(1, max(offsets) + 1):
                if terminal is None:
                    snapshot = runtime.executor.step_formal(controls[offset - 1])
                    if runtime.tracker.status.value != "running":
                        terminal = offset
                        outcome = runtime.tracker.status.value
                if offset not in offsets:
                    continue
                state = policy_observation_from_snapshot(snapshot).state.copy()
                absorbed = terminal is not None and offset > terminal
                if absorbed:
                    state[7:14] = 0
                    state[15] = 0
                states.append(state)
                absorbing.append(absorbed)
                images.extend(
                    [snapshot.cameras["agentview"].rgb, snapshot.cameras["robot0_eye_in_hand"].rgb]
                )
            encoded = self.encoder.encode_numpy(np.stack(images)).reshape(len(offsets), 2, 196, 384)
            self.last_query = {
                "source_tick": observation.formal_tick,
                "replay_max_abs": replay_error,
                "terminal_offset": terminal,
                "terminal_outcome": outcome,
                "absorbing": absorbing,
                "privileged": True,
                "known_delay_ticks": known,
            }
            if known is not None:
                return {"visual": encoded[0], "proprio": states[0], "known_delay_ticks": known}
            return {
                "visual": encoded,
                "proprio": np.stack(states),
                "delay_ticks": np.array(offsets),
                "probabilities": self.macro.copy(),
            }
        finally:
            runtime.close()
