from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from latency_meta_mdp.expert_realization.contracts import ExpertRealizationKey
from latency_meta_mdp.expert_realization.shared_prefix import (
    _compare_shared_prefix_anchors,
    replay_shared_prefix,
)
from latency_meta_mdp.expert_realization.task_instance import (
    TaskInstanceReplayMismatch,
    _compare_initial_states,
    materialize_task_instance,
)

pytestmark = pytest.mark.integration


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_all_closed(environments: list[object]) -> None:
    assert environments
    assert all(getattr(env, "task4_close_count") == 1 for env in environments)


def test_task_instance_and_k6_replay_are_bitwise_same_host_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gate: fresh EGL runtimes reproduce boundary 0 and K6 exactly, or planning stops."""
    from latency_meta_mdp.expert_realization import task_instance as module

    project_root = Path.cwd()
    original_factory = module._environment_factory
    environments: list[object] = []

    def tracked_factory(**kwargs: object) -> object:
        env = original_factory(**kwargs)
        env.task4_close_count = 0
        original_close = env.close

        def close_once(self: object) -> None:
            self.task4_close_count += 1
            original_close()

        env.close = MethodType(close_once, env)
        environments.append(env)
        return env

    monkeypatch.setattr(module, "_environment_factory", tracked_factory)

    first = materialize_task_instance(project_root=project_root, level=1, task_instance_seed=4000)
    _assert_all_closed(environments)
    second = materialize_task_instance(project_root=project_root, level=1, task_instance_seed=4000)
    _assert_all_closed(environments)

    assert first.motion_profile_bytes == second.motion_profile_bytes
    assert first.initial_state_bytes == second.initial_state_bytes
    assert first.task_instance_id == second.task_instance_id
    _compare_initial_states(first.initial_state, second.initial_state)
    assert dict(first.expected_anchor_fingerprints) == dict(second.expected_anchor_fingerprints)

    invalid_publication_values = (
        ("motion_profile_bytes", first.motion_profile_bytes + b"x", "motion_profile_bytes"),
        ("motion_profile_mapping", {"corrupt": True}, "motion_profile_mapping"),
        ("initial_state_bytes", first.initial_state_bytes + b"x", "initial_state_bytes"),
        (
            "initial_state",
            dataclasses.replace(
                first.initial_state,
                robot_qpos=np.array(first.initial_state.robot_qpos) + 1.0,
            ),
            "initial_state",
        ),
        ("shared_endpoint_bytes", first.shared_endpoint_bytes + b"x", "shared_endpoint_bytes"),
        (
            "shared_endpoints_xy",
            np.array(first.shared_endpoints_xy) + 1.0,
            "shared_endpoints_xy",
        ),
    )
    for field, bad_value, expected_field in invalid_publication_values:
        with pytest.raises(TaskInstanceReplayMismatch) as error:
            dataclasses.replace(first, **{field: bad_value})
        assert error.value.field == expected_field

    l2 = materialize_task_instance(project_root=project_root, level=2, task_instance_seed=4000)
    _assert_all_closed(environments)
    l3 = materialize_task_instance(project_root=project_root, level=3, task_instance_seed=4000)
    _assert_all_closed(environments)
    assert first.shared_endpoint_bytes == l2.shared_endpoint_bytes == l3.shared_endpoint_bytes
    assert (
        len(
            {
                first.task_instance_id.motion_profile_sha256,
                l2.task_instance_id.motion_profile_sha256,
                l3.task_instance_id.motion_profile_sha256,
            }
        )
        == 3
    )
    assert len({first.task_instance_id, l2.task_instance_id, l3.task_instance_id}) == 3

    replay_one = replay_shared_prefix(first, decision_source_tick=5)
    _assert_all_closed(environments)
    replay_two = replay_shared_prefix(first, decision_source_tick=5)
    _assert_all_closed(environments)
    _compare_shared_prefix_anchors(replay_one, replay_two)
    assert np.array_equal(
        replay_one.shared_actions,
        np.tile([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], (5, 1)),
    )
    assert np.array_equal(replay_one.boundary_formal_ticks, np.arange(6, dtype=np.int64))

    from latency_meta_mdp.expert_realization import shared_prefix as shared_prefix_module

    original_execute = shared_prefix_module._execute_shared_prefix

    def corrupted_execute(*args: object, **kwargs: object):
        anchor = original_execute(*args, **kwargs)
        outcome = json.loads(anchor.outcome_state_json_utf8)
        outcome["last_boundary_time_us"] += 20_000
        return dataclasses.replace(
            anchor,
            outcome_state_json_utf8=(json.dumps(outcome, sort_keys=True) + "\n").encode(),
        )

    monkeypatch.setattr(shared_prefix_module, "_execute_shared_prefix", corrupted_execute)
    with pytest.raises(TaskInstanceReplayMismatch) as error:
        replay_shared_prefix(first, decision_source_tick=5)
    assert error.value.field == "outcome.last_boundary_time_us"
    _assert_all_closed(environments)
    monkeypatch.setattr(shared_prefix_module, "_execute_shared_prefix", original_execute)

    structured_hash = _sha(project_root / "configs/expert_realization/panda_ball_structured.yaml")
    keys = tuple(
        ExpertRealizationKey(first.task_instance_id, index, structured_hash) for index in range(8)
    )
    assert {key.task_instance_id for key in keys} == {first.task_instance_id}
    for key in keys:
        assert (
            key.task_instance_id.motion_profile_sha256
            == first.task_instance_id.motion_profile_sha256
        )
        assert (
            key.task_instance_id.initial_state_sha256 == first.task_instance_id.initial_state_sha256
        )
        assert dict(first.expected_anchor_fingerprints) == dict(second.expected_anchor_fingerprints)

    environment_count = len(environments)
    for invalid_tick in (0, 4, 6):
        with pytest.raises(ValueError, match="decision_source_tick"):
            replay_shared_prefix(first, decision_source_tick=invalid_tick)
    assert len(environments) == environment_count

    corrupted_hashes = dict(first.expected_anchor_fingerprints)
    corrupted_hashes["planning_start_sha256"] = "0" * 64
    with pytest.raises(TaskInstanceReplayMismatch) as error:
        dataclasses.replace(first, expected_anchor_fingerprints=corrupted_hashes)
    assert error.value.field == "expected_anchor.planning_start_sha256"

    bypassed = dataclasses.replace(first)
    object.__setattr__(bypassed, "motion_profile_bytes", first.motion_profile_bytes + b"x")
    environment_count = len(environments)
    with pytest.raises(TaskInstanceReplayMismatch) as error:
        replay_shared_prefix(bypassed, decision_source_tick=5)
    assert error.value.field == "motion_profile_bytes"
    assert len(environments) == environment_count

    class FailingRuntime:
        def __init__(self, initialize: object) -> None:
            self.executor = SimpleNamespace(initialize=initialize)
            self.close_count = 0

        def close(self) -> None:
            self.close_count += 1

    def fail_initialize() -> object:
        raise RuntimeError("initialize failure")

    initialize_failure = FailingRuntime(fail_initialize)
    monkeypatch.setattr(
        shared_prefix_module,
        "_build_task_instance_runtime",
        lambda task_instance: initialize_failure,
    )
    with pytest.raises(RuntimeError, match="initialize failure"):
        replay_shared_prefix(first, decision_source_tick=5)
    assert initialize_failure.close_count == 1

    boundary = SimpleNamespace()
    replay_failure = FailingRuntime(lambda: boundary)
    monkeypatch.setattr(
        shared_prefix_module,
        "_build_task_instance_runtime",
        lambda task_instance: replay_failure,
    )
    monkeypatch.setattr(
        shared_prefix_module,
        "_initial_state_from_runtime",
        lambda runtime, actual_boundary: first.initial_state,
    )

    def fail_capture(**kwargs: object) -> object:
        raise RuntimeError("camera capture failure")

    monkeypatch.setattr(shared_prefix_module, "_execute_shared_prefix", fail_capture)
    with pytest.raises(RuntimeError, match="camera capture failure"):
        replay_shared_prefix(first, decision_source_tick=5)
    assert replay_failure.close_count == 1
