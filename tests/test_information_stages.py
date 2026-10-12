import numpy as np
import pytest

from latency_meta_mdp.meta.information.schema import ActorPacket, legal_actions


def test_sleep_reserve_and_stage_actions():
    assert legal_actions("G", 24, 20, "split", False).tolist() == [
        True,
        True,
        True,
        False,
        False,
        True,
        False,
        False,
    ]
    assert np.flatnonzero(legal_actions("O", 30, 20, "bound", False)).tolist() == [6]
    assert np.flatnonzero(legal_actions("P", 20, 20, "split", True)).tolist() == [7]
    with pytest.raises(ValueError, match="forecast"):
        legal_actions("P", 20, 20, "split", False)
    with pytest.raises(ValueError, match="coverage"):
        legal_actions("G", 19, 20, "split", False)


def test_packet_copies_and_rejects_unavailable_visual_information():
    x = np.ones((2, 196, 384), np.float32)
    p = ActorPacket(tick=10, stage="O", current_tokens=x, current_tick=10)
    x[:] = 2
    assert p.current_tokens[0, 0, 0] == 1
    assert not p.current_tokens.flags.writeable
    with pytest.raises(ValueError, match="future"):
        ActorPacket(tick=10, stage="G", current_tokens=x, current_tick=11)
    with pytest.raises(ValueError, match="source"):
        ActorPacket(tick=10, stage="G", current_tokens=x)
