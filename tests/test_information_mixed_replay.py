import json

import pytest

from latency_meta_mdp.io.paths import repository_root
from latency_meta_mdp.meta.information.cohort import complete_attempt
from latency_meta_mdp.meta.information.dataset import save_episode
from latency_meta_mdp.meta.information.records import StageRecorder
from latency_meta_mdp.meta.information.schema import ActorPacket
from latency_meta_mdp.runtime.evaluate_conveyor import conveyor_delay_sampler


def make_replay(path, *, scope, scene, behavior):
    policy = dict(checkpoint="checkpoint", receipt_sha256="receipt", model_hash="hash")
    binding = dict(scope=scope, assets={"identity": {"vla": "fixed"}}, mode="bound", code_hash="v1")
    if scope == "online_training":
        binding.update(policy=policy, phase="online", schema_version=1)
    ep = dict(
        episode_id=f"{scene}-nominal-{behavior}",
        scene=scene,
        regime="nominal",
        behavior=behavior,
        scheduler_seed=2,
        policy_seed=0,
        split="train",
        epsilon=0.2,
    )
    manifest = dict(binding=binding, episodes=[ep])
    attempt = path / "episodes" / ep["episode_id"] / "attempt-000"
    attempt.mkdir(parents=True)
    (path / "cohort.json").write_text(json.dumps(manifest))
    _, delay = conveyor_delay_sampler(
        repository_root(), regime="nominal", scene_seed=scene, policy_seed=0
    )
    inner = dict(
        assets=binding["assets"]["identity"],
        mode="bound",
        scene=scene,
        scope=scope,
        code_hash="v1",
        scheduler_seed=2,
        behavior=behavior,
        policy_seed=0,
        delay=delay,
    )
    if behavior == "fqi":
        inner["learned_policy"] = dict(policy, epsilon=0.2)
    record = StageRecorder("test")
    record.start(ActorPacket(10, "G"), 5, forced=False, probability=0.8)
    record.add_reward(10, 1)
    record.finish(11, terminated=True)
    save_episode(attempt / "episode", record, report={}, binding=inner)
    report = dict(
        terminated=True,
        truncated=False,
        starvation_ticks=0,
        scope=scope,
        task_reward=0.1,
        parcels=dict(spawned=10, successes=1, misses=9, timeouts=0, active=0),
    )
    (attempt / "summary.json").write_text(json.dumps(report))
    complete_attempt(attempt, binding=dict(cohort=binding, episode=ep))
    return manifest, attempt


def test_mixed_replay_reads_both_sources_and_rejects_eval_and_changed_policy(tmp_path):
    from latency_meta_mdp.meta.information.online_training import MixedReplay
    from latency_meta_mdp.meta.information.training import VerifiedReplay

    old, _ = make_replay(tmp_path / "old", scope="starter_training", scene=10000, behavior="active")
    new, attempt = make_replay(
        tmp_path / "new", scope="online_training", scene=10016, behavior="fqi"
    )
    offline = VerifiedReplay(tmp_path / "old", old)
    online = VerifiedReplay(tmp_path / "new", new)
    mixed = MixedReplay(offline, online)
    assert mixed.sources == ["offline", "online"]
    assert len(mixed) == 2
    assert mixed[1][0]["task_reward"] == 0.1
    assert [ep[0]["task_reward"] for ep in mixed] == [0.1, 0.1]
    wrong = json.loads(json.dumps(new))
    wrong["binding"]["scope"] = "evaluation"
    with pytest.raises(ValueError, match="training"):
        VerifiedReplay(tmp_path / "new", wrong)
    inner = attempt / "episode/complete.json"
    receipt = json.loads(inner.read_text())
    receipt["binding"]["learned_policy"]["model_hash"] = "another"
    inner.write_text(json.dumps(receipt))
    complete_attempt(attempt, binding=dict(cohort=new["binding"], episode=new["episodes"][0]))
    with pytest.raises(ValueError, match="binding"):
        VerifiedReplay(tmp_path / "new", new)


def test_mixture_requires_explicit_collector_compatibility_and_train_scene_membership():
    from latency_meta_mdp.meta.information.config import identity_hash
    from latency_meta_mdp.meta.information.online_training import validate_mixture

    config = dict(mode="bound", assets={"identity": {"vla": "fixed"}}, splits={"train": [1, 2, 3]})

    def manifest(scope, code, scene):
        return dict(
            binding=dict(
                scope=scope,
                code_hash=code,
                mode="bound",
                assets=config["assets"],
                config_hash=identity_hash(config),
                schema_version=1,
            ),
            episodes=[dict(episode_id=str(scene), scene=scene, split="train")],
        )

    old, new = manifest("starter_training", "old", 1), manifest("online_training", "new", 2)
    compatibility = dict(config_hash=identity_hash(config), collection_code_hashes=["new", "old"])
    validate_mixture(config, [old, new], compatibility)
    extra = manifest("online_training", "new", 3)
    validate_mixture(config, [old, new, extra], compatibility)
    with pytest.raises(ValueError, match="duplicate"):
        validate_mixture(config, [old, new, new], compatibility)
    with pytest.raises(ValueError, match="compatibility"):
        validate_mixture(config, [old, new], {**compatibility, "collection_code_hashes": ["old"]})
    new["episodes"][0]["scene"] = 99
    with pytest.raises(ValueError, match="train"):
        validate_mixture(config, [old, new], compatibility)


def test_cumulative_online_preserves_provenance_and_shared_cache(tmp_path, monkeypatch):
    from latency_meta_mdp.meta.information.online_training import CumulativeReplay, MixedReplay
    from latency_meta_mdp.meta.information.replay_cache import EpisodeCache
    from latency_meta_mdp.meta.information.training import VerifiedReplay

    streams = []
    for i, scene in enumerate([10016, 10032]):
        path = tmp_path / str(i)
        manifest, _ = make_replay(path, scope="online_training", scene=scene, behavior="fqi")
        streams.append(VerifiedReplay(path, manifest))
    data = CumulativeReplay(streams)
    assert len(data.entries) == len(data.receipts) == len(data) == 2
    assert data[1][0]["task_reward"] == 0.1
    cache = EpisodeCache()
    data.cache = cache
    data[0]
    data[0]
    assert cache.metrics()["hits"] == 1
    assert MixedReplay([[{}]], data).sources == ["offline", "online", "online"]
    from latency_meta_mdp.meta.information import dataset
    from latency_meta_mdp.meta.information.replay_preparation import prepare_replay

    prepared = prepare_replay(data, tmp_path / "prepared")
    visited = []
    load = dataset.load_episode

    def record_path(path, **kwargs):
        visited.append(path)
        return load(path, **kwargs)

    monkeypatch.setattr(dataset, "load_episode", record_path)
    assert prepared[1][0]["task_reward"] == 0.1
    assert visited == [prepared.entries[1][0]]
