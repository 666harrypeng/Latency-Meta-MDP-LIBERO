"""Categorical scheduler using the same causal public memory as FQI."""

import numpy as np
import torch

from latency_meta_mdp.meta.information.actor_critic import categorical
from latency_meta_mdp.meta.information.behaviors import Decision
from latency_meta_mdp.meta.information.deployment import AttentionScheduler
from latency_meta_mdp.meta.information.memory import collate_contexts
from latency_meta_mdp.meta.information.memory_schema import DecisionRef


class ActorScheduler(AttentionScheduler):
    algorithm = "actor"

    def __init__(self, model, *, deterministic=True, seed=0, epsilon=0.0):
        if epsilon != 0:
            raise ValueError("categorical actor does not use epsilon-greedy exploration")
        super().__init__(model, seed=seed)
        self.deterministic = deterministic

    def decide(self, packet, legal_mask):
        context = self.memory.context(DecisionRef(self.episode_id, self.decisions), packet)
        self.decisions += 1
        with torch.inference_mode():
            batch = collate_contexts(
                [context],
                device=next(self.model.parameters()).device,
                config=self.model.config["memory"],
            )
            logits = self.model(batch).float().cpu()
            legal = torch.as_tensor(np.asarray(legal_mask, dtype=bool)[None])
            distribution = categorical(logits, legal)
            probabilities = distribution.probs[0].double().numpy()
        probabilities /= probabilities.sum()
        action = int(
            np.argmax(probabilities)
            if self.deterministic
            else self.rng.choice(len(probabilities), p=probabilities)
        )
        probability = 1.0 if self.deterministic else float(probabilities[action])
        self.diagnostics.append(
            dict(
                tick=packet.tick,
                stage=packet.stage,
                decision_index=self.decisions - 1,
                logits=logits[0].tolist(),
                legal=legal[0].tolist(),
                action=action,
                probability=probability,
                log_probability=float(np.log(probability)),
                deterministic=self.deterministic,
                policy_version=self.version,
            )
        )
        return Decision(action, probability, self.version)
