import copy
import json

import pytest
import torch

from latency_meta_mdp.meta.information.config import sha256
from latency_meta_mdp.meta.information.fqi import model_hash
from latency_meta_mdp.meta.information.model import StageQ
from latency_meta_mdp.meta.information.schema import ActorPacket, legal_actions


def checkpoint(tmp_path):
    model = StageQ().eval()
    with torch.no_grad():
        for head in model.heads:
            head.weight.normal_(0, 0.2)
    binding = dict(
        mode="bound",
        algorithm="fqi",
        schema_version=1,
        assets={"vla": "fixed"},
        normalization="fixed-physical-scales-and-learned-LayerNorm",
    )
    path = tmp_path / "iteration-0020"
    path.mkdir()
    torch.save({"model": model.state_dict()}, path / "state.pt")
    receipt = dict(
        binding=binding, model_hash=model_hash(model), files={"state.pt": sha256(path / "state.pt")}
    )
    (path / "complete.json").write_text(json.dumps(receipt))
    return model, path, binding


def test_deployment_replays_history_and_resets_without_training(tmp_path):
    from latency_meta_mdp.meta.information.deployment import load_scheduler

    model, path, binding = checkpoint(tmp_path)
    scheduler = load_scheduler(path, binding, device="cpu")
    packets = [ActorPacket(10, "G"), ActorPacket(14, "G")]
    q, _ = model.sequence(packets)
    for i, packet in enumerate(packets):
        mask = legal_actions("G", 50, mode="bound")
        result = scheduler.decide(packet, mask)
        assert result.action == int(q[i].masked_fill(~torch.tensor(mask), -torch.inf).argmax())
        assert result.probability == 1
    scheduler.reset(episode_id="fresh")
    assert scheduler.hidden is None
    assert model_hash(scheduler.model) == model_hash(model)
    assert not any(p.requires_grad for p in scheduler.model.parameters())


def test_deployment_rejects_changed_assets_and_corrupt_state(tmp_path):
    from latency_meta_mdp.meta.information.deployment import load_scheduler

    _, path, binding = checkpoint(tmp_path)
    changed = copy.deepcopy(binding)
    changed["assets"]["vla"] = "different"
    with pytest.raises(ValueError, match="binding"):
        load_scheduler(path, changed, device="cpu")
    (path / "state.pt").write_bytes(b"broken")
    with pytest.raises(ValueError, match="hash"):
        load_scheduler(path, binding, device="cpu")


def test_deployment_checks_training_config_and_normalization(tmp_path):
    from latency_meta_mdp.meta.information.deployment import load_scheduler

    _, path, binding = checkpoint(tmp_path)
    manifest = tmp_path / "cohort.json"
    manifest.write_text(json.dumps(dict(binding={"config_hash": "original-config"})))
    receipt = json.loads((path / "complete.json").read_text())
    receipt["binding"]["cohort_sha256"] = sha256(manifest)
    receipt["binding"]["normalization"] = "wrong-scales"
    (path / "complete.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="normalization"):
        load_scheduler(
            path, {**binding, "config_hash": "original-config"}, source_manifest=manifest
        )
    receipt["binding"]["normalization"] = "fixed-physical-scales-and-learned-LayerNorm"
    (path / "complete.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="binding"):
        load_scheduler(path, {**binding, "config_hash": "different-RTC"}, source_manifest=manifest)
    restored = load_scheduler(
        path, {**binding, "config_hash": "original-config"}, source_manifest=manifest
    )
    assert restored.source_manifest == str(manifest.resolve())
