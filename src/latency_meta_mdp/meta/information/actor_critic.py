"""In-sample IQL objectives and masked categorical PPO, in success-return units."""

import copy
import math

import numpy as np
import torch
from torch import nn

from latency_meta_mdp.meta.information.attention_model import AttentionStageQ


class AttentionValue(AttentionStageQ):
    """Same public-history encoder, independent scalar heads and parameters."""

    def __init__(self, config, *, normalization=None):
        super().__init__(config, normalization=normalization)
        self.heads = nn.ModuleList([nn.Linear(512, 1) for _ in range(2)])
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, batch):
        return super().forward(batch).squeeze(-1)


def initialize_from_encoder(source, *, value=False):
    model = (AttentionValue if value else AttentionStageQ)(copy.deepcopy(source.config))
    state = model.state_dict()
    state.update(
        {
            k: v.detach().clone()
            for k, v in source.state_dict().items()
            if not k.startswith("heads.")
        }
    )
    model.load_state_dict(state, strict=True)
    return model


def categorical(logits, legal):
    if logits.shape != legal.shape or legal.dtype != torch.bool or not legal.any(-1).all():
        raise ValueError("nonempty matching legal action mask required")
    if not torch.isfinite(logits).all():
        raise ValueError("nonfinite actor logits")
    return torch.distributions.Categorical(logits=logits.float().masked_fill(~legal, -torch.inf))


def expectile_loss(value, teacher, tau):
    if not 0 < tau < 1:
        raise ValueError("expectile must be inside (0,1)")
    residual = teacher.detach() - value
    return torch.where(residual >= 0, tau, 1 - tau) * residual.square()


def advantage_weights(advantage, temperature, maximum):
    if temperature <= 0 or maximum <= 0:
        raise ValueError("positive advantage temperature and cap required")
    return (advantage.detach() / temperature).clamp(max=math.log(maximum)).exp()


def ppo_loss(logits, old_logits, legal, actions, advantage, *, clip, entropy_coefficient):
    policy, old = categorical(logits, legal), categorical(old_logits.detach(), legal)
    if not legal.gather(1, actions[:, None]).all():
        raise ValueError("recorded actor action is illegal")
    ratio = (policy.log_prob(actions) - old.log_prob(actions)).exp()
    advantage = advantage.detach()
    unclipped = ratio * advantage
    clipped = ratio.clamp(1 - clip, 1 + clip) * advantage
    entropy = policy.entropy()
    # Avoid 0 * (-inf - -inf) at illegal categories.
    delta = old.logits.masked_fill(~legal, 0) - policy.logits.masked_fill(~legal, 0)
    kl = (old.probs * delta).sum(-1)
    return -torch.minimum(unclipped, clipped) - entropy_coefficient * entropy, dict(
        kl=kl.detach(),
        entropy=entropy.detach(),
        ratio=ratio.detach(),
        clip_fraction=((ratio - 1).abs() > clip).float().detach(),
    )


def returns_to_go(rewards, terminated):
    if not rewards or len(rewards) != len(terminated) or not terminated[-1] or any(terminated[:-1]):
        raise ValueError("complete terminal episode required for Monte Carlo returns")
    result = np.cumsum(np.asarray(rewards, dtype=np.float64)[::-1])[::-1].copy()
    if not np.isfinite(result).all():
        raise ValueError("nonfinite episode rewards")
    return result.tolist()


@torch.no_grad()
def update_target(target, source, rate):
    if not 0 < rate <= 1:
        raise ValueError("target EMA rate must be inside (0,1]")
    for to, value in zip(target.parameters(), source.parameters(), strict=True):
        to.lerp_(value, rate)
    for to, value in zip(target.buffers(), source.buffers(), strict=True):
        to.copy_(value)
