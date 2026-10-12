"""Information-stage scheduler composed with the unmodified RTC executor."""

import time
from dataclasses import replace

import numpy as np

from latency_meta_mdp.meta.information.schema import (
    LAUNCH,
    OBSERVE,
    PREDICT,
    SLEEP_TICKS,
    ActorPacket,
    legal_actions,
)
from latency_meta_mdp.runtime.latency_harness import LogicalLatencyHarness
from latency_meta_mdp.runtime.rtc_client import RtcActionChunkClient
from latency_meta_mdp.runtime.rtc_protocol import RtcForecastContext, TimedActionPlan


def run_information_episode(
    *,
    runtime,
    policy,
    bootstrap_policy,
    perception,
    scheduler,
    recorder,
    client_config,
    delay_sampler,
    maximum_steps,
    mode="bound",
    record_observation=None,
):
    """Collect one episode. Raw pixels and parcel truth never enter scheduler.decide."""
    capture_original = None
    plant = getattr(runtime.executor, "plant", None)
    if plant is not None:
        capture_original = plant.materialize_boundary

        def capture(ledger):
            with recorder.measure("capture"):
                return capture_original(ledger)

        plant.materialize_boundary = capture
    try:
        if type(maximum_steps) is not int or maximum_steps <= 0 or mode not in ("bound", "split"):
            raise ValueError("invalid information rollout configuration")

        def clock():
            return runtime.executor.ledger.time_us

        scheduler.reset(episode_id=recorder.episode_id)
        harness = LogicalLatencyHarness(
            formal_tick_us=20_000, delay_sampler=delay_sampler, simulation_time_reader=clock
        )
        client = RtcActionChunkClient(
            action_contract=runtime.action_contract,
            config=client_config,
            harness=harness,
            simulation_time_reader=clock,
            monotonic_ns=time.monotonic_ns,
        )
        with recorder.measure("simulation_initialize"):
            latest = runtime.executor.initialize()
        projection_count = 0

        def project(output):
            nonlocal projection_count
            value = output["actions"] if isinstance(output, dict) else output
            actions = np.asarray(value, dtype=float)
            if actions.shape != (50, 7) or not np.isfinite(actions).all():
                raise ValueError("policy must return finite H50 controller-native 7D actions")
            clipped = np.clip(actions, -1, 1)
            projection_count += int(np.count_nonzero(clipped != actions))
            return clipped

        with recorder.measure("bootstrap"):
            client.bootstrap(observation=latest, infer=lambda obs: project(bootstrap_policy(obs)))
        cached = dict(current_tokens=None, forecast_tokens=None)
        prepared = None
        wake, previous_action, last_decision = 10, -1, 0
        actions, requests = [], []

        def decide(state):
            nonlocal wake, previous_action, last_decision, prepared
            if latest.formal_tick < wake:
                return False
            context = RtcForecastContext.from_decision(latest, state)
            for stage in ("G", "O", "P"):
                packet = ActorPacket(
                    tick=state.formal_tick,
                    stage=stage,
                    **cached,
                    current_proprio=latest.state,
                    buffer_version=state.buffer_version,
                    buffer=state.unread_action_buffer,
                    buffer_mask=state.unread_action_mask,
                    estimated_delay=state.estimated_delay_ticks,
                    plan_age=state.plan_age,
                    completed_delays=state.delay_history_ticks,
                    previous_action=previous_action,
                    elapsed_ticks=state.formal_tick - last_decision,
                )
                mask = legal_actions(
                    stage, packet.remaining, mode=mode, forecast_fresh=packet.forecast_fresh
                )
                forced = int(mask.sum()) == 1
                start = time.perf_counter()
                decision = None if forced else scheduler.decide(packet, mask.copy())
                seconds = time.perf_counter() - start
                action = int(np.flatnonzero(mask)[0]) if forced else decision.action
                if type(action) is not int or not 0 <= action < 8 or not mask[action]:
                    raise ValueError("scheduler proposed an illegal action")
                recorder.start(
                    packet,
                    action,
                    forced=forced,
                    probability=1.0 if forced else decision.probability,
                )
                if not forced:
                    recorder.add_cost(f"meta:{len(recorder.rows)}", scheduler.component, seconds)
                previous_action, last_decision = action, state.formal_tick
                if action < 5:
                    wake = state.formal_tick + SLEEP_TICKS[action]
                    return False
                if action == OBSERVE:
                    current = perception.observe(context, prepare_history=mode == "bound")
                    cached.update(
                        current_tokens=current,
                        current_tick=state.formal_tick,
                        observed_proprio=latest.state,
                    )
                elif action == PREDICT:
                    prepared = perception.predict(context)
                    if not prepared.available:
                        raise ValueError("predict requires a valid real-history forecast")
                    cached.update(
                        forecast_tokens=prepared.future_visual_latents,
                        forecast_proprio=prepared.future_proprio,
                        forecast_tick=state.formal_tick,
                        forecast_buffer_version=state.buffer_version,
                    )
                elif action == LAUNCH:
                    wake = state.formal_tick + 1
                    return True
            raise RuntimeError("information stage did not reach an executable outcome")

        def infer(context):
            if prepared is None:
                raise RuntimeError("launch without prediction")
            enriched = replace(context, forecast=prepared.decode(context))
            with recorder.measure("vla"):
                controls = project(policy(latest, enriched))
            requests.append(
                dict(
                    request_id=context.request_id,
                    origin_tick=context.origin_tick,
                    buffer_version=context.buffer_version,
                    estimated_delay=context.estimated_delay_ticks,
                )
            )
            return TimedActionPlan(
                context.origin_tick,
                controls,
                np.ones(50, bool),
                context.request_id,
                context.buffer_version,
            )

        def execute(action):
            nonlocal latest
            tick = latest.formal_tick
            with recorder.measure(
                "physics" if plant is not None else "simulation_step_and_capture"
            ):
                latest = runtime.executor.step_formal(action)
            reward = float(runtime.env.boundary_reward)
            if not reward.is_integer() or reward < 0:
                raise ValueError("conveyor reward must be an integer delivery count")
            recorder.add_reward(tick, int(reward))

        def record():
            if record_observation is not None:
                with recorder.measure("video"):
                    record_observation(latest)

        record()
        while not runtime.env.done and len(actions) < maximum_steps:
            perception.ingest(latest, None if not actions else actions[-1])
            action = client.run_boundary(
                formal_tick=latest.formal_tick,
                observation=latest,
                infer=infer,
                execute=execute,
                decide_launch=decide,
            )
            actions.append(np.asarray(action).copy())
            record()
        recorder.finish(latest.formal_tick, terminated=bool(runtime.env.done))
        events = harness.events
        for request in requests:
            arrival = next(
                (
                    e
                    for e in events
                    if e.kind.value == "arrival" and e.request_id == request["request_id"]
                ),
                None,
            )
            request["realized_delay"] = None if arrival is None else arrival.realized_delay_ticks
        return dict(
            **recorder.summary(),
            action_limit_projections=projection_count,
            requests=requests,
            starvation_ticks=sum(e.kind.value == "starvation" for e in events),
            executed_actions=np.asarray(actions),
            executed_steps=len(actions),
            terminated=bool(runtime.env.done),
            truncated=not runtime.env.done,
            parcels=runtime.world.ledger.summary(),
            parcel_events=runtime.world.ledger.events,
        )
    finally:
        if capture_original is not None:
            plant.materialize_boundary = capture_original
        runtime.env.close()
