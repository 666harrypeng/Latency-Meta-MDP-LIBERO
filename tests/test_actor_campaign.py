import json

import pytest
from attention_fixtures import small_config

from latency_meta_mdp.meta.information.actor_campaign import generation, prune_batch, reset_batch
from latency_meta_mdp.meta.information.actor_learning import initial_state, save_state


def test_batch_reset_preserves_online_adam_but_not_previous_batch_state():
    state = dict(
        phase="ppo",
        step=400,
        epoch=5,
        actor_stopped=True,
        fit_binding="old-data",
        binding={"batch": 1},
        optimizers={"actor": 1, "v": 2},
    )
    new = reset_batch(state, {"batch": 2})
    assert new["optimizers"] == state["optimizers"]
    assert new["step"] == new["epoch"] == 0 and not new["actor_stopped"]
    assert new["binding"] == {"batch": 2} and "fit_binding" not in new
    assert state["step"] == 400


def test_generation_rotates_only_train_scenes_and_never_adds_episodes_for_gpu_count():
    config = small_config()
    first = generation(config, "online1", 96, 0)
    second = generation(config, "online2", 96, 1)
    assert len(first["scenes"]) == len(second["scenes"]) == 32
    assert set(first["scenes"]).isdisjoint(second["scenes"])
    assert set(first["scenes"] + second["scenes"]) == set(config["splits"]["train"])
    with pytest.raises(ValueError):
        generation(config, "bad", 95, 0)


def test_raw_batch_retirement_requires_committed_cycle_and_valid_successor(tmp_path):
    collection = tmp_path / "cycle-0001" / "collection"
    payload = collection / "workers" / "0"
    payload.mkdir(parents=True)
    (payload / "episode.bin").write_bytes(b"needed for resume")
    (payload / "records.json").write_text('{"task_reward": 0.1, "terminated": true}')
    (payload / "complete.json").write_text('{"original": "receipt"}')
    (payload / "worker.log").write_text("episode complete\n")
    successor = save_state(tmp_path / "checkpoint", initial_state(small_config()))
    with pytest.raises(ValueError, match="completed"):
        prune_batch(collection, successor_checkpoint=successor, receipts={})
    assert (payload / "episode.bin").exists()
    (collection.parent / "complete.json").write_text("{}")
    (collection / "rollout-summary.json").write_text("{}")
    prune_batch(collection, successor_checkpoint=successor, receipts={"data": "pinned"})
    assert not payload.exists()
    assert (collection / "rollout-summary.json").exists()
    assert json.loads((collection.parent / "retired-batch.json").read_text())["receipts"] == {
        "data": "pinned"
    }
    audit = collection.parent / "collection-audit"
    assert json.loads((audit / "workers/0/records.json").read_text())["task_reward"] == 0.1
    assert json.loads((audit / "workers/0/complete.json").read_text())["original"] == "receipt"
    assert (audit / "workers/0/worker.log").read_text() == "episode complete\n"
    from latency_meta_mdp.meta.information.config import sha256

    receipt = json.loads((audit / "manifest.json").read_text())
    assert receipt["files"]["workers/0/records.json"] == sha256(audit / "workers/0/records.json")
    prune_batch(collection, successor_checkpoint=successor, receipts={"data": "pinned"})
    assert (audit / "workers/0/records.json").exists()
