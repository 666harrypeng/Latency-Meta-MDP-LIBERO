import numpy as np
import pytest


def test_lazy_online_history_uses_real_stride4_frames_and_actual_controls(tmp_path):
    from pathlib import Path

    import torch
    from test_action_conditioned_jepa_data import _normalization, _record

    from latency_meta_mdp.belief.action_conditioned_jepa.config import load_jepa_temporal_sampling
    from latency_meta_mdp.belief.action_conditioned_jepa.contracts import FutureLatentRollout
    from latency_meta_mdp.policy_belief_runtime import FrozenJepaBelief
    from latency_meta_mdp.policy_execution import PolicyObservation

    record = _record(tmp_path, terminal_tick=30)
    sampling = load_jepa_temporal_sampling(
        Path("configs/belief/action_conditioned_jepa/stride4_80ms_history_160ms.yaml")
    )

    class Encoder:
        counts = []

        def encode(self, images):
            self.counts.append(len(images))
            value = torch.tensor(images[:, 0, 0, 0], dtype=torch.float16)
            return value[:, None, None].expand(-1, 196, 384).clone()

    class Model:
        contexts = []

        def rollout_native(self, context):
            self.contexts.append(context)
            return FutureLatentRollout(
                native_delay_ticks=torch.tensor([4, 8, 12, 16, 20]),
                future_visual_latents=context.vision_history[:, -1:]
                .expand(1, 5, 2, 196, 384)
                .clone(),
                future_proprio=torch.zeros((1, 5, 16), dtype=torch.float32),
            )

    encoder = Encoder()
    model = Model()
    provider = FrozenJepaBelief(
        model=model,
        encoder=encoder,
        normalization=_normalization(record),
        sampling=sampling,
        probabilities=np.full(20, 0.05),
        device=torch.device("cpu"),
    )
    for tick in range(15):
        obs = PolicyObservation(
            formal_tick=tick,
            image=np.full((8, 8, 3), tick, np.uint8),
            wrist_image=np.full((8, 8, 3), tick, np.uint8),
            state=record.proprio_physical[tick],
        )
        provider.observe(obs, None if tick == 0 else record.controls[tick - 1])
        if tick in (10, 14):
            result = provider(obs, np.zeros((20, 7), np.float32))
            assert result["visual"].shape == (5, 2, 196, 384)
    assert encoder.counts == [6, 2]
    np.testing.assert_array_equal(model.contexts[0].vision_history[0, :, 0, 0, 0], [2, 6, 10])
    np.testing.assert_allclose(
        model.contexts[0].executed_controls[0], record.controls[2:10].reshape(2, 4, 7)
    )
    np.testing.assert_array_equal(model.contexts[1].vision_history[0, :, 0, 0, 0], [6, 10, 14])


def test_privileged_provider_requires_explicit_runtime_lane():
    from latency_meta_mdp.policy_execution import LogicalPolicyRuntime

    with pytest.raises(ValueError, match="privileged"):
        LogicalPolicyRuntime(
            action_contract=None,
            client_config=None,
            simulation_time_reader=None,
            delay_sampler=None,
            policy=None,
            scheduler=None,
            belief_provider=type("Oracle", (), {"privileged": True})(),
            scheduler_uses_belief=True,
        )


def test_oracle_peek_and_harness_consume_share_exactly_one_delay_draw():
    from latency_meta_mdp.policy_belief_runtime import PrivilegedDelayCoupler

    values = iter([2, 7])
    coupled = PrivilegedDelayCoupler(lambda: next(values))
    assert coupled.peek() == coupled.peek() == 2
    assert coupled() == 2
    assert coupled() == 7


def test_gt_replay_uses_actual_future_controls_and_exact_oracle_delay():
    from pathlib import Path
    from types import SimpleNamespace

    from latency_meta_mdp.belief.action_conditioned_jepa.config import load_jepa_temporal_sampling
    from latency_meta_mdp.policy_belief_runtime import ReplayGroundTruthBelief
    from latency_meta_mdp.policy_execution import policy_observation_from_snapshot

    class Runtime:
        def __init__(self):
            self.x = 0.0
            self.tick = 0
            self.tracker = SimpleNamespace(status=SimpleNamespace(value="running"))
            self.executor = self

        def snapshot(self):
            q = np.array([self.x] + [0.0] * 6)
            return SimpleNamespace(
                formal_tick_index=self.tick,
                qpos=q,
                qvel=np.zeros(7),
                actuator_ctrl=np.zeros(7),
                robot_qpos=q,
                robot_qvel=np.zeros(7),
                robot_gripper_qpos=np.array([0.02, -0.02]),
                robot_gripper_qvel=np.zeros(2),
                cameras={
                    k: SimpleNamespace(rgb=np.zeros((8, 8, 3), np.uint8))
                    for k in ("agentview", "robot0_eye_in_hand")
                },
            )

        def initialize(self):
            return self.snapshot()

        def step_formal(self, action):
            self.x += float(action[0])
            self.tick += 1
            return self.snapshot()

        def close(self):
            pass

    class Encoder:
        def encode_numpy(self, images):
            return np.zeros((len(images), 196, 384), np.float16)

    live = Runtime()
    s0 = live.initialize()
    action = np.array([0.1, 0, 0, 0, 0, 0, -1])
    s1 = live.step_formal(action)
    sampling = load_jepa_temporal_sampling(
        Path("configs/belief/action_conditioned_jepa/stride4_80ms_history_160ms.yaml")
    )
    gt = ReplayGroundTruthBelief(
        runtime_factory=Runtime,
        current_snapshot=lambda: s1,
        encoder=Encoder(),
        sampling=sampling,
        probabilities=np.full(20, 0.05),
    )
    gt.observe(policy_observation_from_snapshot(s0), None)
    gt.observe(policy_observation_from_snapshot(s1), action)
    future = np.zeros((20, 7), np.float32)
    future[:, 0] = 0.2
    result = gt(policy_observation_from_snapshot(s1), future)
    assert result["proprio"][0, 0] == pytest.approx(0.9)
    result = gt(policy_observation_from_snapshot(s1), np.zeros((20, 7), np.float32))
    assert result["proprio"][0, 0] == pytest.approx(0.1)
    gt.known_delay_reader = lambda: 2
    result = gt(policy_observation_from_snapshot(s1), future)
    assert result["known_delay_ticks"] == 2 and result["proprio"][0] == pytest.approx(0.5)
