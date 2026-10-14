"""A public-history recurrent Q model, independent of simulation and privileged truth."""

import numpy as np
import torch
from torch import nn

from latency_meta_mdp.meta.information.schema import STAGES, ActorPacket


def build_model(config, *, root=None):
    if config.get("schema_version", 1) == 1:
        return StageQ()
    import json
    from pathlib import Path

    from latency_meta_mdp.meta.information.attention_model import AttentionStageQ
    from latency_meta_mdp.meta.information.config import sha256

    path = Path(root or ".") / config["assets"]["paths"]["normalization"]
    expected = config["assets"]["identity"]["forecast_identity"]["jepa_normalization_sha256"]
    if sha256(path) != expected:
        raise ValueError("frozen attention proprio normalization differs")
    return AttentionStageQ(config, normalization=json.loads(path.read_text()))


def scalar_features(packet):
    delays = np.zeros(32, np.float32)
    valid = np.zeros(32, np.float32)
    delays[: len(packet.completed_delays)] = np.asarray(packet.completed_delays) / 20
    valid[: len(packet.completed_delays)] = 1
    # Declared physical scales only. No validation/test statistics are fitted.
    metadata = np.array(
        [
            packet.tick / 2200,
            packet.plan_age / 50,
            packet.estimated_delay / 20,
            packet.elapsed_ticks / 50,
            float(packet.current_tokens is not None),
            float(packet.forecast_tokens is not None),
            float(packet.forecast_fresh),
            (packet.tick - packet.current_tick) / 2200 if packet.current_tick >= 0 else 0,
            (packet.tick - packet.forecast_tick) / 2200 if packet.forecast_tick >= 0 else 0,
        ],
        np.float32,
    )
    return np.concatenate(
        (
            packet.current_proprio,
            packet.observed_proprio,
            packet.forecast_proprio,
            packet.buffer.flatten(),
            packet.buffer_mask,
            delays,
            valid,
            metadata,
            np.eye(9, dtype=np.float32)[packet.previous_action + 1],
            np.eye(3, dtype=np.float32)[STAGES.index(packet.stage)],
        )
    ).astype(np.float32)


class StageQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, 16), nn.GELU())
        self.visual = nn.Sequential(nn.Linear(4 * 196 * 16, 128), nn.GELU())
        n_scalar = len(scalar_features(ActorPacket(0, "G")))
        self.fusion = nn.Sequential(nn.Linear(128 + n_scalar, 128), nn.LayerNorm(128), nn.GELU())
        self.gru = nn.GRUCell(128, 128)
        self.heads = nn.ModuleList([nn.Linear(128, 8) for _ in STAGES])
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def step(self, packet, hidden=None):
        if not isinstance(packet, ActorPacket):
            raise TypeError("Q accepts only public ActorPacket inputs")
        device = next(self.parameters()).device
        tokens = np.stack(
            [
                np.zeros((2, 196, 384), np.float32) if x is None else x
                for x in (packet.current_tokens, packet.forecast_tokens)
            ]
        )
        visual = self.visual(self.patch(torch.as_tensor(tokens, device=device)).flatten())
        scalar = torch.as_tensor(scalar_features(packet), device=device)
        features = self.fusion(torch.cat((visual, scalar)))
        if hidden is None:
            hidden = torch.zeros(128, device=device)
        hidden = self.gru(features, hidden)
        return self.heads[STAGES.index(packet.stage)](hidden), hidden

    def block(self, packets, hidden=None):
        """Batch stateless encoders; preserve the causal GRU and checkpoint layout."""
        if not packets or any(not isinstance(p, ActorPacket) for p in packets):
            raise TypeError("Q requires a nonempty sequence of public ActorPackets")
        device = next(self.parameters()).device
        zero = np.zeros((2, 196, 384), np.float32)
        tokens = np.stack(
            [
                np.stack([zero if x is None else x for x in (p.current_tokens, p.forecast_tokens)])
                for p in packets
            ]
        )
        visual = self.visual(self.patch(torch.as_tensor(tokens, device=device)).flatten(1))
        scalars = torch.as_tensor(np.stack([scalar_features(p) for p in packets]), device=device)
        features = self.fusion(torch.cat((visual, scalars), dim=1))
        if hidden is None:
            hidden = torch.zeros(128, device=device)
        # The public sequence module shares the existing cell's Parameters.
        # It is local: no extra trainable weights or checkpoint keys are introduced.
        recurrent = nn.GRU(128, 128, device="meta")
        recurrent.weight_ih_l0 = self.gru.weight_ih
        recurrent.weight_hh_l0 = self.gru.weight_hh
        recurrent.bias_ih_l0 = self.gru.bias_ih
        recurrent.bias_hh_l0 = self.gru.bias_hh
        recurrent.train(self.training)
        states, last = recurrent(features[:, None, :], hidden[None, None, :])
        states, hidden = states[:, 0, :], last[0, 0, :]
        values = states.new_zeros((len(packets), 8))
        for stage, head in zip(STAGES, self.heads, strict=True):
            indices = [i for i, packet in enumerate(packets) if packet.stage == stage]
            if indices:
                values[indices] = head(states[indices])
        return values, hidden

    def sequence(self, packets):
        hidden, values = None, []
        for start in range(0, len(packets), 64):
            q, hidden = self.block(packets[start : start + 64], hidden)
            values.append(q)
        return torch.cat(values), hidden
