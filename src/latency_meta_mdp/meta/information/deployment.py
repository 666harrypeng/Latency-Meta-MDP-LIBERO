"""Read-only deployment restore; deliberately independent of training resume."""

import json
from pathlib import Path

import torch

from latency_meta_mdp.meta.information.config import identity_hash, sha256, validate_binding
from latency_meta_mdp.meta.information.fqi import FqiScheduler, model_hash
from latency_meta_mdp.meta.information.model import StageQ


def deployment_binding(config, *, algorithm="fqi"):
    if algorithm not in ("fqi", "actor"):
        raise ValueError("unknown deployed scheduler algorithm")
    return dict(
        mode=config["mode"],
        algorithm=algorithm,
        schema_version=config.get("schema_version", 1),
        assets=config["assets"]["identity"],
        config_hash=identity_hash(config),
    )


def scheduler_identity(scheduler):
    identity = dict(
        checkpoint=scheduler.checkpoint,
        receipt_sha256=scheduler.receipt_sha256,
        model_hash=scheduler.version,
    )
    if scheduler.source_manifest is not None:
        identity["source_manifest"] = scheduler.source_manifest
    if getattr(scheduler, "algorithm", None) == "actor":
        identity["algorithm"] = "actor"
    return identity


def load_scheduler(
    checkpoint,
    expected,
    *,
    device="cpu",
    epsilon=0.0,
    seed=0,
    source_manifest=None,
    deterministic=True,
):
    checkpoint = Path(checkpoint)
    receipt = json.loads((checkpoint / "complete.json").read_text())
    for key in ("mode", "algorithm", "schema_version", "assets"):
        validate_binding(expected[key], receipt["binding"][key])
    if receipt["binding"].get("normalization") != "fixed-physical-scales-and-learned-LayerNorm":
        raise ValueError("deployment normalization differs")
    if "config_hash" in expected:
        if "config_hash" in receipt["binding"]:
            validate_binding(expected["config_hash"], receipt["binding"]["config_hash"])
        else:
            if source_manifest is None or sha256(source_manifest) != receipt["binding"].get(
                "cohort_sha256"
            ):
                raise ValueError("deployment source cohort hash differs")
            original = json.loads(Path(source_manifest).read_text())
            validate_binding(expected["config_hash"], original["binding"]["config_hash"])
    if receipt["files"] != {"state.pt": sha256(checkpoint / "state.pt")}:
        raise ValueError("deployment checkpoint hash differs")
    state = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=True)
    if expected["schema_version"] == 2:
        from latency_meta_mdp.meta.information.attention_model import AttentionStageQ

        model = AttentionStageQ(state["model_config"])
    else:
        model = StageQ()
    model.load_state_dict(state["model"], strict=True)
    if model_hash(model) != receipt["model_hash"]:
        raise ValueError("deployment model hash differs")
    model.to(device).eval().requires_grad_(False)
    scheduler_class = AttentionScheduler if expected["schema_version"] == 2 else FqiScheduler
    if expected["algorithm"] == "actor":
        from latency_meta_mdp.meta.information.actor_deployment import ActorScheduler

        if expected["schema_version"] != 2:
            raise ValueError("actor requires attention schema 2")
        scheduler = ActorScheduler(model, epsilon=epsilon, seed=seed, deterministic=deterministic)
    else:
        scheduler = scheduler_class(model, epsilon=epsilon, seed=seed)
    scheduler.source_manifest = (
        str(Path(source_manifest).resolve()) if source_manifest is not None else None
    )
    scheduler.checkpoint = str(checkpoint.resolve())
    scheduler.receipt_sha256 = sha256(checkpoint / "complete.json")
    return scheduler


class AttentionScheduler(FqiScheduler):
    """Public events enter memory even when a stage has no learned choice."""

    def __init__(self, model, *, epsilon=0.0, seed=0):
        from latency_meta_mdp.meta.information.memory import MemoryStore

        super().__init__(model, epsilon=epsilon, seed=seed)
        self.memory = MemoryStore(model.action_spec, config=model.config["memory"])

    def reset(self, *, episode_id):
        super().reset(episode_id=episode_id)
        self.memory.reset(episode_id)
        self.episode_id = episode_id
        self.decisions = 0

    def ingest(self, event):
        self.memory.ingest(event)

    def decide(self, packet, legal_mask):
        import numpy as np

        from latency_meta_mdp.meta.information.behaviors import Decision
        from latency_meta_mdp.meta.information.memory import collate_contexts
        from latency_meta_mdp.meta.information.memory_schema import DecisionRef

        context = self.memory.context(DecisionRef(self.episode_id, self.decisions), packet)
        self.decisions += 1
        with torch.inference_mode():
            batch = collate_contexts(
                [context],
                device=next(self.model.parameters()).device,
                config=self.model.config["memory"],
            )
            q = self.model(batch)[0].float().cpu().numpy()
        choices = np.flatnonzero(legal_mask)
        if not len(choices) or not np.isfinite(q).all():
            raise ValueError("invalid deployed attention Q or action mask")
        greedy = int(choices[np.argmax(q[choices])])
        action = int(self.rng.choice(choices)) if self.rng.random() < self.epsilon else greedy
        probability = self.epsilon / len(choices) + (1 - self.epsilon) * (action == greedy)
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
        return Decision(action, float(probability), self.version)
