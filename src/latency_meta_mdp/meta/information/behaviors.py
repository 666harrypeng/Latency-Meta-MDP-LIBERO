"""Deployable rule collectors over the same public packet and legal action mask."""

from dataclasses import dataclass

import numpy as np

from latency_meta_mdp.meta.information.schema import LAUNCH, OBSERVE, PREDICT, SLEEP_TICKS


@dataclass(frozen=True)
class Decision:
    action: int
    probability: float = 1.0
    policy_version: str = "rule-v1"


class RuleScheduler:
    component = "rule_scheduler"

    def __init__(self, kind, *, period=25, seed=0):
        if kind not in ("periodic", "active", "lazy", "exploration"):
            raise ValueError("unknown collection behavior")
        if type(period) is not int or not 1 <= period <= 30:
            raise ValueError("period exceeds the safe RTC reserve")
        self.kind, self.period = kind, period
        self.rng = np.random.default_rng(seed)

    def reset(self, *, episode_id):
        # No recurrent information state; the caller owns the declared RNG stream.
        pass

    def decide(self, packet, legal_mask):
        choices = np.flatnonzero(legal_mask)
        if not len(choices):
            raise ValueError("no legal behavior action")
        information = {"G": OBSERVE, "O": PREDICT, "P": LAUNCH}[packet.stage]
        if self.kind == "exploration":
            probabilities = np.zeros(8)
            sleeps = np.flatnonzero(legal_mask[:5])
            probabilities[information] = 0.5 if len(sleeps) else 1.0
            if len(sleeps):
                probabilities[sleeps] = 0.5 / len(sleeps)
            action = int(self.rng.choice(8, p=probabilities))
            return Decision(action, float(probabilities[action]), "balanced-gate-v1")
        if self.kind == "active":
            return Decision(information)
        target = 30 if self.kind == "lazy" else self.period
        if packet.stage != "G" or packet.plan_age >= target:
            return Decision(information)
        sleeps = [
            i for i, d in enumerate(SLEEP_TICKS) if legal_mask[i] and d <= target - packet.plan_age
        ]
        return Decision(max(sleeps) if sleeps else information)
