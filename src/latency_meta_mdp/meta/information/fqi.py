"""Finite-episode recurrent fitted Q iteration with fixed, history-recomputed targets."""

import copy
import hashlib
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from latency_meta_mdp.meta.information.behaviors import Decision
from latency_meta_mdp.meta.information.schema import legal_actions


def targets(reward, cost, terminated, next_q, next_legal, *, budget_lambda=0.0, cost_scale=1.0):
    if budget_lambda < 0 or cost_scale <= 0:
        raise ValueError("invalid cost objective")
    if torch.any(~terminated & ~next_legal.any(-1)):
        raise ValueError("nonterminal transition has no legal continuation")
    best = next_q.masked_fill(~next_legal, -torch.inf).max(-1).values
    best = torch.where(terminated, torch.zeros_like(best), best)
    return reward - budget_lambda * cost / cost_scale + best


def model_hash(model):
    digest = hashlib.sha256()
    for k, v in model.state_dict().items():
        digest.update(k.encode())
        digest.update(v.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _mask(packet, mode):
    return legal_actions(
        packet.stage, packet.remaining, mode=mode, forecast_fresh=packet.forecast_fresh
    )


def replay_weights(lengths, sources=None):
    if not lengths or any(n < 1 for n in lengths):
        raise ValueError("positive episode lengths required")
    if sources is None:
        return [len(lengths) / sum(lengths)] * len(lengths)
    if len(sources) != len(lengths) or set(sources) != {"offline", "online"}:
        raise ValueError("balanced replay requires both offline and online sources")
    totals = {
        s: sum(n for n, group in zip(lengths, sources, strict=True) if group == s)
        for s in ("offline", "online")
    }
    return [len(lengths) * 0.5 / totals[s] for s in sources]


def fit_iteration(
    model,
    episodes,
    *,
    mode,
    epochs=1,
    learning_rate=3e-4,
    budget_lambda=0.0,
    cost_scale=1.0,
    conservative_weight=0.0,
    chunk_size=64,
    progress=None,
    sources=None,
):
    """Fit one frozen Bellman iteration; no target encoder state comes from online Q.

    Optimizer steps occur at episode boundaries. Recurrent gradients truncate at
    chunk_size, while the public prefix is always replayed with current weights.
    """
    if not episodes or epochs < 1 or chunk_size < 1 or conservative_weight < 0:
        raise ValueError("invalid fitted iteration")
    device = next(model.parameters()).device
    frozen = copy.deepcopy(model).eval().requires_grad_(False)
    before = model_hash(frozen)
    fixed, lengths = [], []
    target_started = perf_counter()
    with torch.no_grad():
        for episode_index, ep in enumerate(episodes):
            if not ep or not ep[-1]["terminated"]:
                raise ValueError("complete terminal episodes are required")
            lengths.append(len(ep))
            q, _ = frozen.sequence([t["packet"] for t in ep])
            next_q = torch.cat((q[1:], torch.zeros_like(q[:1])))
            masks = np.stack([_mask(t["packet"], mode) for t in ep])
            next_mask = torch.as_tensor(
                np.concatenate((masks[1:], np.zeros_like(masks[:1]))), device=device
            )
            for i, t in enumerate(ep):
                if not masks[i, t["action"]] or (i < len(ep) - 1 and t["terminated"]):
                    raise ValueError("invalid action or premature terminal in replay")
            fixed.append(
                targets(
                    torch.tensor([t["task_reward"] for t in ep], device=device),
                    torch.tensor([t["cost"] for t in ep], device=device),
                    torch.tensor([t["terminated"] for t in ep], device=device),
                    next_q,
                    next_mask,
                    budget_lambda=budget_lambda,
                    cost_scale=cost_scale,
                )
            )
            if not torch.isfinite(fixed[-1]).all():
                raise ValueError("nonfinite fitted target")
            if progress is not None:
                progress(dict(phase="target", episode=episode_index + 1, episodes=len(episodes)))
    target_seconds = perf_counter() - target_started
    fit_started = perf_counter()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    weights = replay_weights(lengths, sources)
    q_min, q_max, above_one, legal_count = float("inf"), -float("inf"), 0, 0
    for epoch in range(epochs):
        order = (
            np.random.permutation(len(episodes)) if sources is not None else range(len(episodes))
        )
        stream = (
            ((int(i), episodes[int(i)], fixed[int(i)]) for i in order)
            if sources is not None
            else (
                (i, ep, target) for i, (ep, target) in enumerate(zip(episodes, fixed, strict=True))
            )
        )
        for position, (episode_index, ep, target) in enumerate(stream):
            if len(ep) != len(target):
                raise ValueError("replay changed during fitted iteration")
            optimizer.zero_grad(set_to_none=True)
            hidden = None
            episode_loss, huber_sum, conservative_sum = 0.0, 0.0, 0.0
            for start in range(0, len(ep), chunk_size):
                transitions = ep[start : start + chunk_size]
                q, hidden = model.block([t["packet"] for t in transitions], hidden)
                actions = torch.tensor([t["action"] for t in transitions], device=device)
                chosen = q.gather(1, actions[:, None]).squeeze(1)
                loss = F.smooth_l1_loss(
                    chosen, target[start : start + len(transitions)], reduction="none"
                )
                mask = torch.as_tensor(
                    np.stack([_mask(t["packet"], mode) for t in transitions]), device=device
                )
                huber = loss.sum() * weights[episode_index]
                conservative = (
                    (torch.logsumexp(q.masked_fill(~mask, -torch.inf), dim=1) - chosen).sum()
                    * weights[episode_index]
                    if conservative_weight
                    else q.new_zeros(())
                )
                block_loss = huber + conservative_weight * conservative
                if not torch.isfinite(block_loss):
                    raise ValueError("nonfinite fitted loss")
                block_loss.backward()
                values = q[mask].detach().cpu().numpy()
                q_min, q_max = min(q_min, float(values.min())), max(q_max, float(values.max()))
                above_one += int((values > 1).sum())
                legal_count += len(values)
                loss_values = (
                    torch.stack([block_loss.detach(), huber.detach(), conservative.detach()])
                    .cpu()
                    .tolist()
                )
                episode_loss += loss_values[0]
                huber_sum += loss_values[1]
                conservative_sum += loss_values[2]
                hidden = hidden.detach()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5, error_if_nonfinite=True)
            optimizer.step()
            losses.append(episode_loss)
            if progress is not None:
                progress(
                    dict(
                        phase="fit",
                        epoch=epoch + 1,
                        episode=position + 1,
                        replay_episode=episode_index,
                        episodes=len(episodes),
                        loss=episode_loss,
                        huber_loss=huber_sum,
                        conservative_loss=conservative_sum,
                        source="offline" if sources is None else sources[episode_index],
                    )
                )
    return dict(
        target_seconds=target_seconds,
        regression_seconds=perf_counter() - fit_started,
        loss=float(np.mean(losses)),
        transitions=sum(lengths),
        q_min=q_min,
        q_max=q_max,
        q_above_one_fraction=above_one / legal_count,
        target_min=min(float(t.min()) for t in fixed),
        target_max=max(float(t.max()) for t in fixed),
        target_hash_before=before,
        target_hash_after=model_hash(frozen),
    )


class FqiScheduler:
    component = "meta_q"

    def __init__(self, model, *, epsilon=0.0, seed=0):
        if not 0 <= epsilon <= 1:
            raise ValueError("invalid exploration probability")
        self.model = model.eval()
        self.version = model_hash(model)
        self.epsilon, self.rng, self.hidden = epsilon, np.random.default_rng(seed), None

    def reset(self, *, episode_id):
        self.hidden = None
        self.diagnostics = []

    def decide(self, packet, legal_mask):
        with torch.inference_mode():
            q, self.hidden = self.model.step(packet, self.hidden)
            q = q.cpu().numpy()  # Completes GPU inference before caller ends its cost timer.
        if not np.isfinite(q).all():
            raise ValueError("nonfinite deployed Q")
        choices = np.flatnonzero(legal_mask)
        greedy = int(choices[np.argmax(q[choices])])
        action = int(self.rng.choice(choices)) if self.rng.random() < self.epsilon else greedy
        probability = self.epsilon / len(choices) + (1 - self.epsilon) * (action == greedy)
        if hasattr(self, "diagnostics"):
            self.diagnostics.append(
                dict(
                    tick=packet.tick,
                    stage=packet.stage,
                    legal_q={int(i): float(q[i]) for i in choices},
                    greedy_action=greedy,
                    action=action,
                    probability=float(probability),
                    policy_version=self.version,
                )
            )
        return Decision(action, probability, self.version)
