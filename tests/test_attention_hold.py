import numpy as np
import pytest
from test_rtc_client import _client, _infer

from latency_meta_mdp.meta.information.memory_schema import ActionSpec


@pytest.mark.parametrize("remaining", [50, 7, 0])
def test_hold_prefix_only_promotes_declared_control_horizon(remaining):
    from latency_meta_mdp.runtime.rtc_client import executable_prefix

    client, clock, _ = _client(20)
    state = client._state(50 - remaining)
    controls, mask = executable_prefix(state, allow_hold=True, horizon=20)
    assert int(mask.sum()) == max(remaining, 20)
    np.testing.assert_array_equal(controls[:remaining], state.unread_action_buffer[:remaining])
    if remaining < 20:
        assert not controls[remaining:20, :6].any()
        assert (controls[remaining:20, -1] == -1).all()
    assert not mask[max(remaining, 20) :].any()


def test_hold_wait_and_late_request_preserve_gripper_then_install_elapsed_response():
    spec = ActionSpec(starvation_mode="allow_hold")
    assert spec.legal("G", 0)[:8].all()
    client, clock, harness = _client(4)
    client.allow_hold = True
    output = []
    contexts = []

    def infer(context):
        contexts.append(context)
        return _infer(context)

    for tick in range(62):
        clock.tick = tick
        client.run_boundary(
            formal_tick=tick,
            observation=None,
            infer=infer,
            decide_launch=lambda state: state.formal_tick == 55,
            execute=lambda action: output.append(action.copy()),
        )
    np.testing.assert_array_equal(np.asarray(output[50:59])[:, :6], 0)
    assert (np.asarray(output[50:59])[:, -1] == -1).all()
    assert contexts[0].previous_action_mask[:20].all()
    assert not contexts[0].previous_action_mask[20:].any()
    assert output[59][0] == pytest.approx(0.54)
    assert sum(e.kind.value == "starvation" for e in harness.events) == 9
