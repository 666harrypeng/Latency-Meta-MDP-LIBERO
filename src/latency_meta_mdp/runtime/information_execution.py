"""Information-stage scheduler composed with the unmodified RTC executor."""

import hashlib
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
from latency_meta_mdp.runtime.rtc_client import RtcActionChunkClient, executable_prefix
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
    action_spec=None,
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
            allow_hold=action_spec is not None and action_spec.starvation_mode == "allow_hold",
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
        ordinal, harness_cursor = 0, 0
        holding = False

        def key(tick):
            nonlocal ordinal
            ordinal += 1
            return (tick, ordinal)

        def emit(kind, tick, represented, payload):
            if action_spec is None:
                return
            from latency_meta_mdp.meta.information.memory_schema import PublicEvent

            available = key(tick)
            event = PublicEvent(
                f"{recorder.episode_id}:{available[1]}",
                kind,
                available,
                tuple(represented),
                payload,
            )
            recorder.events.append(event)
            if hasattr(scheduler, "ingest"):
                scheduler.ingest(event)

        def arrivals():
            nonlocal harness_cursor
            for event in harness.events[harness_cursor:]:
                if event.kind.value == "arrival":
                    emit(
                        "arrival",
                        event.formal_tick,
                        (event.formal_tick,),
                        dict(
                            request_id=event.request_id,
                            realized_delay=event.realized_delay_ticks,
                            buffer_version=client.buffer_version,
                        ),
                    )
            harness_cursor = len(harness.events)

        def decide(state):
            nonlocal wake, previous_action, last_decision, prepared
            arrivals()
            if latest.formal_tick < wake:
                return False
            context = RtcForecastContext.from_decision(latest, state)
            if client.allow_hold:
                controls, valid = executable_prefix(
                    state, allow_hold=True, horizon=client_config.maximum_delay_ticks
                )
                context = replace(context, previous_actions=controls, previous_action_mask=valid)
            for stage in ("G", "O", "P"):
                packet_class, extra = ActorPacket, {}
                if action_spec is not None:
                    from latency_meta_mdp.meta.information.memory_schema import AttentionPacket

                    packet_class = AttentionPacket
                    extra = dict(
                        action_spec=action_spec,
                        decision_key=key(state.formal_tick),
                        episode_remaining=max(0, maximum_steps - state.formal_tick),
                    )
                packet = packet_class(
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
                    **extra,
                )
                mask = (
                    legal_actions(
                        stage, packet.remaining, mode=mode, forecast_fresh=packet.forecast_fresh
                    )
                    if action_spec is None
                    else action_spec.legal(
                        stage, packet.remaining, forecast_fresh=packet.forecast_fresh
                    )
                )
                forced = int(mask.sum()) == 1
                start = time.perf_counter()
                decision = None if forced else scheduler.decide(packet, mask.copy())
                seconds = time.perf_counter() - start
                action = int(np.flatnonzero(mask)[0]) if forced else decision.action
                if type(action) is not int or not 0 <= action < len(mask) or not mask[action]:
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
                emit(
                    "decision",
                    state.formal_tick,
                    (state.formal_tick,),
                    dict(
                        action=action,
                        probability=1.0 if forced else decision.probability,
                        stage=stage,
                    ),
                )
                sleeps = SLEEP_TICKS if action_spec is None else action_spec.sleep_ticks
                observe, predict, launch = (
                    (OBSERVE, PREDICT, LAUNCH)
                    if action_spec is None
                    else (
                        action_spec.id_for("observe"),
                        action_spec.id_for("predict"),
                        action_spec.id_for("launch"),
                    )
                )
                if action < len(sleeps):
                    wake = state.formal_tick + sleeps[action]
                    return False
                if action == observe:
                    if action_spec is None:
                        current = perception.observe(context, prepare_history=mode == "bound")
                    else:
                        bundle = perception.observe(
                            context, prepare_history=True, return_history=True
                        )
                        emit(
                            "observe",
                            state.formal_tick,
                            bundle["source_ticks"],
                            dict(visual=bundle["visual"], proprio=bundle["proprio"]),
                        )
                        current = bundle["visual"][-1]
                    cached.update(
                        current_tokens=current,
                        current_tick=state.formal_tick,
                        observed_proprio=latest.state,
                    )
                elif action == predict:
                    prepared = perception.predict(context)
                    if not prepared.available:
                        raise ValueError("predict requires a valid real-history forecast")
                    cached.update(
                        forecast_tokens=prepared.future_visual_latents,
                        forecast_proprio=prepared.future_proprio,
                        forecast_tick=state.formal_tick,
                        forecast_buffer_version=state.buffer_version,
                    )
                    emit(
                        "predict",
                        state.formal_tick,
                        (state.formal_tick + state.estimated_delay_ticks,),
                        dict(
                            visual=prepared.future_visual_latents,
                            proprio=prepared.future_proprio,
                            source_tick=state.formal_tick,
                            query_ticks=state.estimated_delay_ticks,
                            buffer_version=state.buffer_version,
                            prefix_hash=hashlib.sha256(
                                context.previous_actions.tobytes()
                                + context.previous_action_mask.tobytes()
                            ).hexdigest(),
                        ),
                    )
                elif action == launch:
                    wake = state.formal_tick + 1
                    return True
            raise RuntimeError("information stage did not reach an executable outcome")

        def infer(context):
            if prepared is None:
                raise RuntimeError("launch without prediction")
            enriched = replace(context, forecast=prepared.decode(context))
            emit(
                "request",
                context.origin_tick,
                (context.origin_tick,),
                dict(
                    request_id=context.request_id,
                    estimated_delay=context.estimated_delay_ticks,
                    buffer_version=context.buffer_version,
                ),
            )
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
            nonlocal latest, holding
            arrivals()
            tick = latest.formal_tick
            now_holding = bool(harness.events and harness.events[-1].kind.value == "starvation")
            if now_holding != holding:
                emit("hold_start" if now_holding else "hold_end", tick, (tick,), {})
                holding = now_holding
            with recorder.measure(
                "physics" if plant is not None else "simulation_step_and_capture"
            ):
                latest = runtime.executor.step_formal(action)
            reward = float(runtime.env.boundary_reward)
            if not reward.is_integer() or reward < 0:
                raise ValueError("conveyor reward must be an integer delivery count")
            recorder.add_reward(tick, int(reward))
            emit(
                "execute",
                latest.formal_tick,
                (tick, latest.formal_tick),
                dict(action=action, proprio=latest.state),
            )

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
            starvation_mode=action_spec.starvation_mode if action_spec is not None else "guarded",
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
