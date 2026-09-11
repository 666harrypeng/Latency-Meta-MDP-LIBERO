"""In-process policy integration on the existing controlled logical-latency client.

Wall stages are measured independently. They are not silently added to, or
subtracted from, configured logical delay. This is not a concurrency benchmark.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np

from latency_meta_mdp.action_chunk_client import ChunkDecisionState, SharpActionChunkClient
from latency_meta_mdp.latency_harness import HarnessEventKind, LogicalLatencyHarness
from latency_meta_mdp.meta_transitions import DecisionStageAccumulator
from latency_meta_mdp.rtc_client import RtcActionChunkClient
from latency_meta_mdp.rtc_protocol import (
    RtcActionChunkClientConfig,
    RtcDecisionState,
    RtcInferenceContext,
    TimedActionPlan,
)


@dataclass(frozen=True)
class PolicyObservation:
    formal_tick: int
    image: np.ndarray
    wrist_image: np.ndarray
    state: np.ndarray
    prompt: str = "Grasp the moving ball and lift it."

    def __post_init__(self):
        if type(self.formal_tick) is not int or self.formal_tick < 0 or not self.prompt:
            raise ValueError("policy observation identity is invalid")
        for name in ("image", "wrist_image", "state"):
            value = np.array(getattr(self, name), copy=True)
            if name == "state":
                value = value.astype(np.float32)
                valid = value.shape == (16,) and np.isfinite(value).all()
            else:
                valid = (
                    value.ndim == 3
                    and value.shape[-1] == 3
                    and value.dtype == np.uint8
                    and value.shape[0] > 0
                    and value.shape[1] > 0
                )
            if not valid:
                raise ValueError(
                    "policy observation must have two RGB images and finite current 16D proprio"
                )
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def to_policy_inputs(self):
        return {
            "observation/image": self.image,
            "observation/wrist_image": self.wrist_image,
            "observation/state": self.state,
            "prompt": self.prompt,
        }


def policy_observation_from_snapshot(snapshot) -> PolicyObservation:
    for name in ("agentview", "robot0_eye_in_hand"):
        camera = snapshot.cameras[name]
        if (
            getattr(camera, "source_formal_tick", snapshot.formal_tick_index)
            != snapshot.formal_tick_index
        ):
            raise ValueError("policy camera does not belong to the current formal boundary")
    state = np.concatenate(
        (
            snapshot.robot_qpos,
            snapshot.robot_qvel,
            [
                snapshot.robot_gripper_qpos[0] - snapshot.robot_gripper_qpos[1],
                snapshot.robot_gripper_qvel[0] - snapshot.robot_gripper_qvel[1],
            ],
        )
    ).astype(np.float32)
    return PolicyObservation(
        formal_tick=snapshot.formal_tick_index,
        image=snapshot.cameras["agentview"].rgb,
        wrist_image=snapshot.cameras["robot0_eye_in_hand"].rgb,
        state=state,
    )


class InProcessOpenpiPolicy:
    """Explicit-noise OpenPI bridge; raw return latents never enter a network client.

    Keep the patched OpenPI source context alive for this object's lifetime.
    A shared Generator can sequence bootstrap and subsequent policy noise.
    """

    def __init__(self, policy, *, noise_rng, belief_input_key="return_belief"):
        from openpi.policies.policy import Policy

        if not isinstance(policy, Policy):
            raise TypeError("this bridge supports only an in-process OpenPI Policy")
        if not isinstance(noise_rng, np.random.Generator) or belief_input_key not in {
            "return_belief",
            "known_delay_oracle",
            "prefix",
        }:
            raise ValueError("policy noise stream or Belief input route is invalid")
        self.policy = policy
        self.noise_rng = noise_rng
        self.belief_input_key = belief_input_key

    def __call__(self, observation: PolicyObservation, belief):
        inputs = observation.to_policy_inputs()
        if belief is not None:
            if self.belief_input_key == "prefix":
                if not isinstance(belief, dict) or set(belief) not in (
                    {"return_belief"},
                    {"return_belief", "known_delay_oracle"},
                ):
                    raise ValueError("prefix bridge requires its explicit nested input packet")
                inputs.update(belief)
            else:
                inputs[self.belief_input_key] = belief
        noise = self.noise_rng.standard_normal((50, 32), dtype=np.float32)
        return self.policy.infer(inputs, noise=noise)


class InProcessRtcOpenpiPolicy(InProcessOpenpiPolicy):
    """Use the native policy transforms for both observations and the old action buffer."""

    action_alignment = "observation_time"

    def __init__(self, policy, *, noise_rng):
        super().__init__(policy, noise_rng=noise_rng)
        model = policy._model
        self.uses_forecast = getattr(model, "image_keys", None) is not None
        if policy._is_pytorch_model or not getattr(model, "pi05", False):
            raise ValueError("RTC bridge requires the matched JAX pi0.5 model")
        if getattr(model, "active_action_dim", None) != 7 or model.action_horizon != 50:
            raise ValueError("RTC bridge requires the native H50 active-7D action contract")
        if any(
            getattr(model, name, None) is not None
            for name in ("return_belief_adapter", "return_belief_prefix")
        ):
            raise ValueError("return-indexed policy checkpoints cannot use RTC alignment")

    def __call__(self, observation: PolicyObservation, context=None):
        if context is None:
            if self.uses_forecast:
                raise ValueError("forecast policy requires a request context and native bootstrap")
            return super().__call__(observation, None)
        if (
            not isinstance(context, RtcInferenceContext)
            or context.origin_tick != observation.formal_tick
        ):
            raise ValueError("RTC policy requires a matching source-time request context")
        inputs = observation.to_policy_inputs()
        if self.uses_forecast != (context.forecast is not None):
            raise ValueError("RTC model and forecast input mode disagree")
        if context.forecast is not None:
            inputs["forecast"] = context.forecast.to_condition()
        inputs["actions"] = context.previous_actions
        inputs["actions_is_pad"] = ~context.previous_action_mask
        noise = self.noise_rng.standard_normal((50, 32), dtype=np.float32)
        return self.policy.infer(inputs, noise=noise, rtc_delay_ticks=context.estimated_delay_ticks)


@dataclass(frozen=True)
class PhysicalStepResult:
    reward: float
    terminated: bool = False

    def __post_init__(self):
        if not np.isfinite(self.reward) or type(self.terminated) is not bool:
            raise ValueError("physical reward/termination is invalid")


@dataclass(frozen=True)
class FixedCursorScheduler:
    interval: int = 25

    def __post_init__(self):
        if type(self.interval) is not int or self.interval <= 0:
            raise ValueError("fixed launch interval must be positive")

    def __call__(self, state, observation, belief):
        age = state.plan_age if isinstance(state, RtcDecisionState) else state.actions_consumed
        return age >= self.interval


class ImmediateLaunchScheduler:
    def __call__(self, state, observation, belief):
        return True


@dataclass(frozen=True)
class BufferCoverageScheduler:
    minimum_remaining: int = 20

    def __post_init__(self):
        if type(self.minimum_remaining) is not int or not 0 <= self.minimum_remaining <= 50:
            raise ValueError("buffer coverage threshold must lie in [0,50]")

    def __call__(self, state, observation, belief):
        return state.remaining_actions <= self.minimum_remaining


class LatentChangeScheduler:
    """Mean coordinate RMS change in current, deployment-visible latent features.

    The first eligible decision launches to establish a reference. Threshold
    calibration belongs to train-pool closed-loop development after policy SFT.
    """

    def __init__(self, *, feature_reader, threshold: float):
        if not callable(feature_reader) or not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("latent threshold requires a feature reader and positive threshold")
        self.feature_reader = feature_reader
        self.threshold = float(threshold)
        self.reference = None
        self.current = None
        self.last_error = None

    def __call__(self, state, observation, belief):
        current = np.asarray(self.feature_reader(observation), dtype=np.float32)
        if not current.size or not np.isfinite(current).all():
            raise ValueError("latent threshold input must be nonempty and finite")
        if self.reference is not None and current.shape != self.reference.shape:
            raise ValueError("latent threshold feature shape changed")
        self.current = current.copy()
        self.last_error = (
            None
            if self.reference is None
            else float(np.sqrt(np.mean((current - self.reference) ** 2)))
        )
        return self.reference is None or self.last_error >= self.threshold

    def on_launch(self, state, observation, belief):
        self.reference = self.current.copy()


class LogicalPolicyRuntime:
    """Connect public observations, optional Belief, scheduling and H50 replacement.

    A conditioned policy needs an explicit native bootstrap callable, so early
    history is never fabricated. Learned scheduler callbacks can use cadence >1;
    after a Launch, the next no-pending boundary is always a decision stage.
    """

    def __init__(
        self,
        *,
        action_contract,
        client_config,
        simulation_time_reader,
        delay_sampler,
        policy,
        scheduler,
        bootstrap_policy=None,
        belief_provider=None,
        forecast_provider=None,
        policy_uses_belief=False,
        scheduler_uses_belief=False,
        decision_interval_ticks=None,
        minimum_decision_tick=10,
        shield_latest_launch=True,
        gamma=1.0,
        monotonic_ns=time.perf_counter_ns,
        allow_privileged_belief=False,
        latency_probabilities=None,
        policy_alignment=None,
    ):
        self.is_rtc = isinstance(client_config, RtcActionChunkClientConfig)
        if forecast_provider is not None and (
            not self.is_rtc
            or belief_provider is not None
            or policy_uses_belief
            or scheduler_uses_belief
        ):
            raise ValueError("RTC forecast provider cannot be mixed with legacy Belief routes")
        if forecast_provider is not None and bootstrap_policy is None:
            raise ValueError("forecast runtime requires an explicit native bootstrap policy")
        if self.is_rtc:
            if policy_alignment != "observation_time":
                raise ValueError("RTC requires an explicitly verified observation_time policy")
            if latency_probabilities is not None:
                raise ValueError("RTC actor uses completed-delay history, not the episode PMF")
            if policy_uses_belief or scheduler_uses_belief:
                raise ValueError("RTC forecast integration requires its new query interface")
        if getattr(belief_provider, "privileged", False) and not allow_privileged_belief:
            raise ValueError(
                "a privileged Belief provider requires an explicit oracle/GT evaluation lane"
            )
        if (policy_uses_belief or scheduler_uses_belief) and belief_provider is None:
            raise ValueError("the selected policy/scheduler requires a Belief provider")
        if policy_uses_belief and bootstrap_policy is None:
            raise ValueError("conditioned runtime requires an explicit native bootstrap policy")
        interval = (
            (4 if scheduler_uses_belief else 1)
            if decision_interval_ticks is None
            else decision_interval_ticks
        )
        if (
            type(interval) is not int
            or interval <= 0
            or type(minimum_decision_tick) is not int
            or minimum_decision_tick < 0
        ):
            raise ValueError("decision cadence/history warmup is invalid")
        self.clock = monotonic_ns
        self.latency_probabilities = None
        if latency_probabilities is not None:
            probability = np.array(latency_probabilities, dtype=np.float32, copy=True)
            if (
                probability.shape != (20,)
                or not np.isfinite(probability).all()
                or np.any(probability < 0)
                or not np.isclose(probability.sum(), 1, atol=1e-6, rtol=0)
            ):
                raise ValueError("runtime episode law must be a normalized D20 PMF")
            probability.setflags(write=False)
            self.latency_probabilities = probability
        self.events = []
        self.transitions = []
        self.collector = DecisionStageAccumulator(gamma=gamma)
        self.harness = LogicalLatencyHarness(
            formal_tick_us=20_000,
            delay_sampler=delay_sampler,
            simulation_time_reader=simulation_time_reader,
            monotonic_ns=monotonic_ns,
        )
        client_type = RtcActionChunkClient if self.is_rtc else SharpActionChunkClient
        self.client = client_type(
            action_contract=action_contract,
            config=client_config,
            harness=self.harness,
            simulation_time_reader=simulation_time_reader,
            monotonic_ns=monotonic_ns,
            on_chunk_install=self._installed,
        )
        self.policy = policy
        self.scheduler = scheduler
        self.bootstrap_policy = bootstrap_policy or (lambda observation: policy(observation, None))
        self.belief_provider = belief_provider
        self.forecast_provider = forecast_provider
        self.forecast_calls = 0
        self.privileged_belief = bool(getattr(belief_provider, "privileged", False))
        self.policy_uses_belief = policy_uses_belief
        self.scheduler_uses_belief = scheduler_uses_belief
        self.interval = interval
        self.minimum_tick = minimum_decision_tick
        self.shield = shield_latest_launch
        self.last_decision_tick = None
        self.last_action = None
        self.previous_action = None
        self.prepared_belief = None
        self.terminated = False
        self.total_reward = 0.0
        self.policy_calls = 0
        self.bootstrap_calls = 0
        self.belief_calls = 0
        self.action_limit_projections = 0
        self.shield_interventions = 0

    def _event(self, stage, tick, start, end=None, **fields):
        end = start if end is None else end
        if type(start) is not int or type(end) is not int or start < 0 or end < start:
            raise ValueError("wall clock moved backward")
        self.events.append(
            {
                "stage": stage,
                "formal_tick": tick,
                "wall_start_ns": start,
                "wall_end_ns": end,
                "wall_duration_ns": end - start,
                **fields,
            }
        )

    def _installed(self, event):
        stage = "bootstrap_install" if event.formal_tick is None else "chunk_install"
        self._event(
            stage,
            event.formal_tick,
            self.clock(),
            request_id=event.source_request_id,
            chunk_id=event.chunk_id,
            installed_index=event.installed_chunk_index,
        )

    def _project(self, output):
        actions = np.asarray(output["actions"] if isinstance(output, dict) else output, dtype=float)
        if actions.shape != (50, 7) or not np.isfinite(actions).all():
            raise ValueError("policy must return finite H50 controller-native 7D actions")
        projected = np.clip(actions, -1, 1)
        self.action_limit_projections += int(np.count_nonzero(projected != actions))
        return projected

    def bootstrap(self, observe):
        observation = observe()
        if not isinstance(observation, PolicyObservation) or observation.formal_tick != 0:
            raise ValueError("bootstrap requires a public observation at tick zero")

        def infer(obs):
            start = self.clock()
            self.bootstrap_calls += 1
            self._event("bootstrap_launch", None, start)
            output = self.bootstrap_policy(obs)
            self._event("bootstrap_return", None, self.clock())
            return self._project(output)

        return self.client.bootstrap(observation=observation, infer=infer)

    def _belief(self, observation, state):
        start = self.clock()
        belief = self.belief_provider(observation, state.executable_controls)
        self.belief_calls += 1
        self._event("belief", observation.formal_tick, start, self.clock())
        return belief

    def _launch_deadline_reached(self, state):
        if self.is_rtc:
            return state.remaining_actions <= self.client.config.maximum_delay_ticks
        return state.actions_consumed >= self.client.config.launch_trigger_horizon

    def step(self, *, formal_tick: int, observe, execute):
        if self.terminated:
            raise RuntimeError("the policy episode already terminated")
        start = self.clock()
        observation = observe()
        if not isinstance(observation, PolicyObservation) or observation.formal_tick != formal_tick:
            raise ValueError("observation is not aligned to the current physical boundary")
        if self.belief_provider is not None and (
            self.policy_uses_belief or self.scheduler_uses_belief
        ):
            self.belief_provider.observe(observation, self.previous_action)
        self._event("observation", formal_tick, start, self.clock())
        if self.forecast_provider is not None:
            began = self.clock()
            self.forecast_provider.observe(observation, self.previous_action)
            self._event("forecast_history", formal_tick, began, self.clock())
        self.prepared_belief = None
        step_result = None

        def decide(state: ChunkDecisionState | RtcDecisionState):
            # The episode law is known information for both Meta ablations;
            # only future-state prediction is removed in the no-Belief lane.
            if not self.is_rtc:
                state = replace(state, latency_probabilities=self.latency_probabilities)
            deadline = self.shield and self._launch_deadline_reached(state)
            due = (
                self.last_decision_tick is None
                or self.last_action == "launch"
                or formal_tick - self.last_decision_tick >= self.interval
                or deadline
            )
            if (formal_tick < self.minimum_tick and not (self.is_rtc and deadline)) or not due:
                return False
            belief = self._belief(observation, state) if self.scheduler_uses_belief else None
            decision_state = {"observation": observation, "buffer": state, "belief": belief}
            if self.collector.active:
                self.transitions.append(
                    self.collector.finish(next_formal_tick=formal_tick, next_state=decision_state)
                )
            began = self.clock()
            proposed = self.scheduler(state, observation, belief)
            if type(proposed) is not bool:
                raise TypeError("scheduler must return a boolean Launch/Wait decision")
            forced = deadline and not proposed
            launch = proposed or forced
            if launch and callable(getattr(self.scheduler, "on_launch", None)):
                self.scheduler.on_launch(state, observation, belief)
            self.shield_interventions += int(forced)
            self._event(
                "meta_decision",
                formal_tick,
                began,
                self.clock(),
                proposed_launch=proposed,
                launch=launch,
                shielded=forced,
            )
            self.collector.begin(
                formal_tick=formal_tick,
                state=decision_state,
                action="launch" if launch else "wait",
                proposed_action="launch" if proposed else "wait",
                shielded=forced,
            )
            self.last_decision_tick = formal_tick
            self.last_action = "launch" if launch else "wait"
            if launch and self.policy_uses_belief:
                self.prepared_belief = (
                    belief if belief is not None else self._belief(observation, state)
                )
            return launch

        def infer(context):
            if self.forecast_provider is not None:
                self._event(
                    "request_launch", formal_tick, self.clock(), request_id=context.request_id
                )
                began = self.clock()
                forecast = self.forecast_provider.predict(context)
                context = replace(context, forecast=forecast)
                self.forecast_calls += 1
                self._event(
                    "forecast",
                    formal_tick,
                    began,
                    self.clock(),
                    request_id=context.request_id,
                    target_tick=forecast.target_tick,
                    available=forecast.available,
                )
            rtc_fields = (
                {
                    "origin_tick": context.origin_tick,
                    "estimated_delay_ticks": context.estimated_delay_ticks,
                    "buffer_version": context.buffer_version,
                    "available_prefix_actions": int(context.previous_action_mask.sum()),
                }
                if self.is_rtc
                else {}
            )
            if self.is_rtc and (
                context.forecast is not None or getattr(self.policy, "plan_construction", None)
            ):
                rtc_fields.update(
                    previous_action_buffer=context.previous_actions.tolist(),
                    previous_action_mask=context.previous_action_mask.tolist(),
                )
            self._event(
                "policy_launch",
                formal_tick,
                self.clock(),
                request_id=context.request_id,
                **rtc_fields,
            )
            self.policy_calls += 1
            condition = self.prepared_belief if self.policy_uses_belief else None
            if self.is_rtc:
                condition = context
            output = self.policy(context.observation, condition)
            self._event(
                "policy_return", formal_tick, self.clock(), request_id=context.request_id,
                handoff=output.get("handoff") if isinstance(output, dict) else None,
            )
            actions = self._project(output)
            if self.is_rtc:
                return TimedActionPlan(
                    origin_tick=context.origin_tick, request_id=context.request_id,
                    buffer_version=context.buffer_version, actions=actions,
                    valid_mask=np.ones(50, dtype=bool),
                )
            return actions

        def advance(action):
            nonlocal step_result
            began = self.clock()
            step_result = execute(action)
            self._event("control_and_next_observation", formal_tick, began, self.clock())
            if not isinstance(step_result, PhysicalStepResult):
                raise TypeError("execute must return PhysicalStepResult")
            self.previous_action = np.array(action, copy=True)
            self.total_reward += step_result.reward
            if self.collector.active:
                self.collector.add_reward(formal_tick=formal_tick, reward=step_result.reward)

        action = self.client.run_boundary(
            formal_tick=formal_tick,
            observation=observation,
            decide_launch=decide,
            infer=infer,
            execute=advance,
        )
        if step_result.terminated:
            self.terminated = True
            if self.collector.active:
                self.transitions.append(
                    self.collector.finish(
                        next_formal_tick=formal_tick + 1, next_state=None, terminated=True
                    )
                )
        return action, step_result

    def request_ledger(self):
        """Audit-only stage timestamps; unresolved pending delays remain undisclosed.

        Fixed schedulers may decide before computing policy-only Belief. The
        timestamps preserve that real order. Observation timing covers the
        supplied observe callback; capture done by execute is in the control
        stage. No wall interval is treated as configured physical delay.
        """
        arrivals = {
            e.request_id: e for e in self.harness.events if e.kind is HarnessEventKind.ARRIVAL
        }
        rows = []
        for launch in (e for e in self.events if e["stage"] == "policy_launch"):
            request_id, tick = launch["request_id"], launch["formal_tick"]
            stages = {
                e["stage"]: e
                for e in self.events
                if e["formal_tick"] == tick
                and e["stage"]
                in {"observation", "belief", "meta_decision", "request_launch", "forecast"}
            }
            returned = next(
                e
                for e in self.events
                if e["stage"] == "policy_return" and e["request_id"] == request_id
            )
            installed = next(
                (
                    e
                    for e in self.events
                    if e["stage"] == "chunk_install" and e["request_id"] == request_id
                ),
                None,
            )
            arrival = arrivals.get(request_id)
            rows.append(
                {
                    "request_id": request_id,
                    "launch_formal_tick": tick,
                    "source_formal_tick": launch.get("origin_tick", tick),
                    "estimated_delay_ticks": launch.get("estimated_delay_ticks"),
                    "installed_index": None if installed is None else installed["installed_index"],
                    "observation_start_ns": stages["observation"]["wall_start_ns"],
                    "observation_ready_ns": stages["observation"]["wall_end_ns"],
                    "belief_complete_ns": stages.get("belief", {}).get("wall_end_ns"),
                    "meta_decision_ns": stages["meta_decision"]["wall_end_ns"],
                    "policy_launch_ns": launch["wall_start_ns"],
                    "request_launch_ns": stages.get("request_launch", launch)["wall_start_ns"],
                    "forecast_start_ns": stages.get("forecast", {}).get("wall_start_ns"),
                    "forecast_complete_ns": stages.get("forecast", {}).get("wall_end_ns"),
                    "forecast_target_tick": stages.get("forecast", {}).get("target_tick"),
                    "forecast_available": stages.get("forecast", {}).get("available"),
                    "previous_action_buffer": launch.get("previous_action_buffer"),
                    "previous_action_mask": launch.get("previous_action_mask"),
                    "policy_return_ns": returned["wall_start_ns"],
                    "planned_handoff_tick": (returned.get("handoff") or {}).get(
                        "planned_handoff_tick"
                    ),
                    "policy_input_tick": (returned.get("handoff") or {}).get("policy_input_tick"),
                    "handoff_forecast_used": (returned.get("handoff") or {}).get("forecast_used"),
                    "chunk_install_ns": None if installed is None else installed["wall_start_ns"],
                    "arrival_formal_tick": None if arrival is None else arrival.arrival_formal_tick,
                    "realized_delay_ticks": None
                    if arrival is None
                    else arrival.realized_delay_ticks,
                }
            )
        return rows

    def summary(self):
        return {
            "protocol_id": self.client.config.protocol_id,
            "plan_construction": getattr(self.policy, "plan_construction", None),
            "rtc": {
                "initial_delay_ticks": list(self.client.config.initial_delay_ticks),
                "delay_history_capacity": self.client.config.delay_history_capacity,
                "delay_history_ticks": list(self.client.delay_history.delays),
                "timeout_count": self.client.timeout_count,
            }
            if self.is_rtc
            else None,
            "latency_mode": "controlled_logical_policy_delay",
            "concurrent_deployment_verified": False,
            "policy_calls": self.policy_calls,
            "bootstrap_calls": self.bootstrap_calls,
            "belief_calls": self.belief_calls,
            "forecast_calls": self.forecast_calls,
            "forecast_wall_ns": sum(
                e["wall_duration_ns"] for e in self.events if e["stage"] == "forecast"
            ),
            "forecast_history_wall_ns": sum(
                e["wall_duration_ns"] for e in self.events if e["stage"] == "forecast_history"
            ),
            "shield_interventions": self.shield_interventions,
            "privileged_belief": self.privileged_belief,
            "action_limit_projections": self.action_limit_projections,
            "total_reward": self.total_reward,
            "terminated": self.terminated,
            "pending_at_terminal": self.harness.pending,
            "starvation_ticks": sum(
                e.kind is HarnessEventKind.STARVATION for e in self.harness.events
            ),
            "decision_interval_ticks": self.interval,
            "minimum_decision_tick": self.minimum_tick,
            "belief_wall_ns": sum(
                e["wall_duration_ns"] for e in self.events if e["stage"] == "belief"
            ),
        }
