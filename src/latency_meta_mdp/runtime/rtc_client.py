"""Timed RTC plan execution using the existing controlled logical request lifecycle.

Action generation/guidance is supplied by the policy callback. This module only
owns observation-time indexing, public request snapshots and buffer installation.
"""

from __future__ import annotations

import numpy as np

from latency_meta_mdp.runtime.action_chunk_client import (
    BootstrapRecord,
    ChunkClientEvent,
    ChunkClientEventKind,
)
from latency_meta_mdp.runtime.rtc_protocol import (
    RollingDelayHistory,
    RtcActionChunkClientConfig,
    RtcDecisionState,
    RtcInferenceContext,
    TimedActionPlan,
)


class RtcActionChunkClient:
    def __init__(
        self,
        *,
        action_contract,
        config,
        harness,
        simulation_time_reader,
        monotonic_ns,
        on_chunk_install=None,
    ):
        if not isinstance(config, RtcActionChunkClientConfig):
            raise TypeError("RTC client requires its own configuration")
        if action_contract.formal_tick_us != harness.formal_tick_us:
            raise ValueError("RTC and control clocks disagree")
        self.action_contract = action_contract
        self.config = config
        self.harness = harness
        self._simulation_time_reader = simulation_time_reader
        self._clock = monotonic_ns
        self._on_chunk_install = on_chunk_install
        self.delay_history = RollingDelayHistory(
            capacity=config.delay_history_capacity,
            initial_delays=config.initial_delay_ticks,
        )
        self._plan = None
        self._buffer_version = -1
        self._pending_context = None
        self._events = []
        self._actions_consumed = 0
        self._faulted = False
        self.last_gripper_command = action_contract.gripper_open_command
        self.timeout_count = 0

    @property
    def events(self):
        return tuple(self._events)

    def _validate_controls(self, actions):
        for action in actions:
            self.action_contract.split_action(action)

    def _install(self, plan, tick, *, bootstrap=False):
        remaining = plan.remaining_from(formal_tick=tick)
        if not len(remaining):
            raise ValueError("returned plan has expired before installation")
        self._validate_controls(remaining)
        old_version = self._buffer_version if self._plan is not None else None
        discarded = 0 if self._plan is None else len(self._plan.remaining_from(formal_tick=tick))
        self._buffer_version += 1
        self._plan = plan
        self._actions_consumed = 0
        event = ChunkClientEvent(
            kind=ChunkClientEventKind.BOOTSTRAP_INSTALL
            if bootstrap
            else ChunkClientEventKind.CHUNK_INSTALL,
            formal_tick=None if bootstrap else tick,
            time_us=None if bootstrap else tick * 20_000,
            chunk_id=self._buffer_version,
            source_request_id=plan.request_id,
            old_chunk_id=old_version,
            discarded_action_count=discarded,
            installed_chunk_index=tick - plan.origin_tick,
        )
        self._events.append(event)
        if self._on_chunk_install is not None:
            self._on_chunk_install(event)

    def bootstrap(self, *, observation, infer):
        if self._faulted or self._plan is not None or self._simulation_time_reader() != 0:
            raise RuntimeError("RTC bootstrap requires a fresh tick-zero runtime")
        try:
            start = self._clock()
            actions = infer(observation)
            end = self._clock()
            after = self._simulation_time_reader()
            if after != 0:
                raise RuntimeError("simulation advanced during RTC bootstrap")
            record = BootstrapRecord(
                protocol_id=self.config.protocol_id,
                installed_chunk_id=0,
                simulation_time_before_us=0,
                simulation_time_after_us=after,
                wall_start_ns=start,
                wall_end_ns=end,
                wall_duration_ns=end - start,
            )
            plan = TimedActionPlan(
                origin_tick=0,
                actions=actions,
                valid_mask=np.ones(50, dtype=bool),
                request_id=None,
                buffer_version=0,
            )
            self._install(plan, 0, bootstrap=True)
            return record
        except BaseException:
            self._faulted = True
            raise

    def _hold_action(self):
        return self.action_contract.compose_action(
            arm_reference=np.zeros(6),
            gripper_command=self.last_gripper_command,
        )

    def _state(self, tick):
        remaining = self._plan.remaining_from(formal_tick=tick)
        buffer = np.repeat(self._hold_action()[None], 50, axis=0)
        buffer[: len(remaining)] = remaining
        if len(remaining):
            buffer[len(remaining) :, -1] = remaining[-1, -1]
        return RtcDecisionState(
            formal_tick=tick,
            plan_origin_tick=self._plan.origin_tick,
            buffer_version=self._buffer_version,
            active_cursor=tick - self._plan.origin_tick,
            remaining_actions=len(remaining),
            actions_consumed=self._actions_consumed,
            estimated_delay_ticks=self.delay_history.estimate_ticks(),
            delay_history_ticks=self.delay_history.delays,
            unread_action_buffer=buffer,
            unread_action_mask=np.arange(50) < len(remaining),
        )

    def _accept(self, arrival):
        plan = arrival.payload
        context = self._pending_context
        if context is None or (
            plan.request_id != context.request_id
            or plan.origin_tick != context.origin_tick
            or plan.buffer_version != self._buffer_version
        ):
            raise ValueError("RTC response identity does not match pending snapshot")
        if not len(plan.remaining_from(formal_tick=arrival.arrival_formal_tick)):
            raise ValueError("RTC response expired before arrival")
        self.harness.mark_eligible(arrival)
        self.harness.mark_activated(arrival)
        self._install(plan, arrival.arrival_formal_tick)
        self.delay_history.record_completed(
            request_id=plan.request_id,
            origin_tick=plan.origin_tick,
            completion_tick=arrival.arrival_formal_tick,
        )
        self._pending_context = None

    def run_boundary(self, *, formal_tick, observation, infer, execute, decide_launch=None):
        if self._faulted or self._plan is None:
            raise RuntimeError("RTC client is faulted or not bootstrapped")
        try:
            if getattr(observation, "formal_tick", formal_tick) != formal_tick:
                raise ValueError("RTC observation must belong to the request source boundary")
            arrival = self.harness.open_boundary(formal_tick)
            if (
                self._pending_context is not None
                and formal_tick - self._pending_context.origin_tick
                > self.config.maximum_delay_ticks
            ):
                self.timeout_count += 1
                raise TimeoutError(
                    "RTC request exceeded declared 400ms support; request not cancelled"
                )
            if arrival is not None:
                self._accept(arrival)
            if not self.harness.pending:
                state = self._state(formal_tick)
                launch = (
                    state.plan_age >= self.config.launch_trigger_horizon
                    if decide_launch is None
                    else decide_launch(state)
                )
                if type(launch) is not bool:
                    raise TypeError("RTC scheduler must return a boolean")
                if launch:

                    def run_inference(launch_context):
                        context = RtcInferenceContext(
                            request_id=launch_context.request_id,
                            origin_tick=formal_tick,
                            observation=observation,
                            buffer_version=state.buffer_version,
                            previous_actions=state.unread_action_buffer,
                            previous_action_mask=state.unread_action_mask,
                            estimated_delay_ticks=state.estimated_delay_ticks,
                        )
                        self._pending_context = context
                        plan = infer(context)
                        if not isinstance(plan, TimedActionPlan):
                            raise TypeError("RTC inference must return a TimedActionPlan")
                        if (
                            plan.origin_tick != context.origin_tick
                            or plan.request_id != context.request_id
                            or plan.buffer_version != context.buffer_version
                        ):
                            raise ValueError("RTC response identity does not match request")
                        self._validate_controls(plan.actions[plan.valid_mask])
                        return plan

                    immediate = self.harness.launch(observation=observation, infer=run_inference)
                    if immediate is not None:
                        self._accept(immediate)
            remaining = self._plan.remaining_from(formal_tick=formal_tick)
            if len(remaining):
                action = remaining[0]
                execute(action)
                self.harness.mark_buffered_execute(
                    source_request_id=self._plan.request_id, action=action
                )
                self._events.append(
                    ChunkClientEvent(
                        kind=ChunkClientEventKind.CHUNK_ACTION_EXECUTE,
                        formal_tick=formal_tick,
                        time_us=formal_tick * 20_000,
                        chunk_id=self._buffer_version,
                        source_request_id=self._plan.request_id,
                        executed_chunk_index=formal_tick - self._plan.origin_tick,
                        action=action,
                    )
                )
                self._actions_consumed += 1
            else:
                action = self._hold_action()
                self.harness.mark_starvation(action=action)
                execute(action)
                self.harness.mark_execute(request_id=None, action=action)
            self.last_gripper_command = float(action[-1])
            self.harness.close_boundary()
            return action
        except BaseException:
            self._faulted = True
            self.harness.mark_faulted()
            raise
