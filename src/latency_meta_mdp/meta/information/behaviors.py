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


class AttentionRuleScheduler(RuleScheduler):
    """The same rules plus a grid over the *available* guarded request window."""

    def __init__(
        self,
        kind,
        *,
        action_spec,
        period=25,
        seed=0,
        rho=None,
        refresh=None,
        forecast_mode="predict",
    ):
        if type(period) is not int or period < 1:
            raise ValueError("rule period must be a positive integer")
        super().__init__(
            "active" if kind == "grid" else kind,
            period=25 if action_spec.version == 3 else period,
            seed=seed,
        )
        self.period = period
        if forecast_mode not in ("predict", "current_only"):
            raise ValueError("unknown rule forecast mode")
        if forecast_mode == "current_only" and action_spec.version != 3:
            raise ValueError("current-only rules require the optional prediction protocol")
        if kind == "grid" and (
            rho not in (0, 0.25, 0.5, 0.75, 1) or refresh not in ("on_request", "every_4_ticks")
        ):
            raise ValueError("invalid request-window grid pattern")
        self.kind, self.spec, self.rho, self.refresh = kind, action_spec, rho, refresh
        self.forecast_mode = forecast_mode

    def reset(self, *, episode_id):
        self.buffer_version = None
        self.target = None

    def decide(self, packet, legal_mask):
        spec = self.spec
        information = spec.id_for(dict(G="observe", O="predict", P="launch")[packet.stage])
        if (
            spec.version == 3
            and packet.stage == "O"
            and (self.forecast_mode == "current_only" or not legal_mask[information])
        ):
            information = spec.id_for("launch")
        sleeps = np.flatnonzero(legal_mask[: len(spec.sleep_ticks)])
        if self.kind == "exploration":
            probabilities = np.zeros(spec.num_actions)
            choices = (
                np.flatnonzero(legal_mask[len(spec.sleep_ticks) :]) + len(spec.sleep_ticks)
                if spec.version == 3 and self.forecast_mode == "predict"
                else [information]
            )
            probabilities[choices] = (0.5 if len(sleeps) else 1.0) / len(choices)
            if len(sleeps):
                probabilities[sleeps] = 0.5 / len(sleeps)
            action = int(self.rng.choice(spec.num_actions, p=probabilities))
            return Decision(action, float(probabilities[action]), f"balanced-gate-v{spec.version}")
        if self.kind == "active":
            return Decision(information, policy_version="rule-v2")
        if self.kind == "grid":
            if self.buffer_version != packet.buffer_version:
                self.buffer_version = packet.buffer_version
                window = max(packet.remaining - spec.reserve, 0)
                self.target = packet.tick + round(self.rho * window)
            wait = max(0, self.target - packet.tick)
            if packet.stage == "O" or not wait:
                return Decision(information, policy_version="grid-v2")
            if self.refresh == "every_4_ticks":
                if packet.stage == "G":
                    return Decision(information, policy_version="grid-v2")
                wait = min(wait, 4)
        else:
            target = 30 if self.kind == "lazy" else self.period
            wait = max(0, target - packet.plan_age)
            if packet.stage != "G" or not wait:
                return Decision(information, policy_version="rule-v2")
        choices = [int(i) for i in sleeps if spec.sleep_ticks[i] <= wait]
        return Decision(max(choices) if choices else information, policy_version="rule-v2")
