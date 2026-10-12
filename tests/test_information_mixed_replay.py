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

    config = dict(mode="bound", assets={"identity": {"vla": "fixed"}}, splits={"train": [1, 2]})

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
            episodes=[dict(scene=scene, split="train")],
        )

    old, new = manifest("starter_training", "old", 1), manifest("online_training", "new", 2)
    compatibility = dict(config_hash=identity_hash(config), collection_code_hashes=["new", "old"])
    validate_mixture(config, [old, new], compatibility)
    with pytest.raises(ValueError, match="compatibility"):
        validate_mixture(config, [old, new], {**compatibility, "collection_code_hashes": ["old"]})
    new["episodes"][0]["scene"] = 99
    with pytest.raises(ValueError, match="train"):
        validate_mixture(config, [old, new], compatibility)
