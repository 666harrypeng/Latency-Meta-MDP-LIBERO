import numpy as np
import torch

from latency_meta_mdp.meta.information.model import StageQ
from latency_meta_mdp.meta.information.schema import ActorPacket


def test_recurrent_sequence_equals_deployed_step_and_has_no_privileged_input():
    torch.manual_seed(12)
    model = StageQ().eval()
    packets = [
        ActorPacket(10, "G"),
        ActorPacket(
            10,
            "P",
            current_tick=10,
            current_tokens=np.ones((2, 196, 384)),
            forecast_tick=10,
            forecast_tokens=np.zeros((2, 196, 384)),
        ),
    ]
    with torch.no_grad():
        sequence, _ = model.sequence(packets)
        first, hidden = model.step(packets[0])
        second, _ = model.step(packets[1], hidden)
    torch.testing.assert_close(sequence, torch.stack([first, second]))
    assert sequence.shape == (2, 8)
    assert torch.isfinite(sequence).all()


def test_q_zero_initialization_and_episode_reset():
    from latency_meta_mdp.meta.information.fqi import FqiScheduler
    from latency_meta_mdp.meta.information.schema import legal_actions

    model = StageQ().eval()
    packet = ActorPacket(10, "G")
    with torch.no_grad():
        q, _ = model.step(packet)
    torch.testing.assert_close(q, torch.zeros(8))
    scheduler = FqiScheduler(model)
    mask = legal_actions("G", 50)
    scheduler.decide(packet, mask)
    first = scheduler.hidden.clone()
    scheduler.decide(packet, mask)
    scheduler.reset(episode_id="next")
    assert scheduler.hidden is None
    scheduler.decide(packet, mask)
    torch.testing.assert_close(first, scheduler.hidden)
