import io
import json
import tarfile

import pytest

from latency_meta_mdp.meta.information.config import sha256


def fixture_cohort(root):
    root.mkdir()
    episodes = [dict(episode_id=f"scene-{i}") for i in range(3)]
    binding = dict(schema_version=3, scope="starter_training")
    (root / "cohort.json").write_text(json.dumps(dict(binding=binding, episodes=episodes)))
    for episode in episodes:
        folder = root / "workers/0/episodes" / episode["episode_id"] / "attempt-000"
        folder.mkdir(parents=True)
        (folder / "data.npz").write_bytes(b"test data")
        (folder / "complete.json").write_text(
            json.dumps(
                dict(
                    binding=dict(cohort=binding, episode=episode),
                    files={"data.npz": sha256(folder / "data.npz")},
                )
            )
        )
        link = root / "episodes" / episode["episode_id"]
        link.parent.mkdir(exist_ok=True)
        link.symlink_to(folder.parent)
    (root / "secret.env").write_text("not part of receipts")


def test_bundle_flattens_completed_data_and_restores_once(tmp_path):
    from latency_meta_mdp.io.fqi_dataset import pack_dataset, restore_dataset

    source = tmp_path / "source"
    fixture_cohort(source)
    bundle = tmp_path / "bundle"
    index = pack_dataset(source, bundle, episodes_per_shard=2)
    assert len(index["shards"]) == 2
    assert index["episodes"] == 3
    destination = tmp_path / "restored"
    restore_dataset(bundle, destination)
    assert (destination / "episodes/scene-0/attempt-000/data.npz").read_bytes() == b"test data"
    assert not (destination / "secret.env").exists()
    assert not (destination / "workers").exists()
    with pytest.raises(FileExistsError):
        restore_dataset(bundle, destination)
    shard = bundle / index["shards"][0]["name"]
    shard.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash"):
        restore_dataset(bundle, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_restore_rejects_escape_even_with_matching_archive_hash(tmp_path):
    from latency_meta_mdp.io.fqi_dataset import pack_dataset, restore_dataset

    source, bundle = tmp_path / "source", tmp_path / "bundle"
    fixture_cohort(source)
    index = pack_dataset(source, bundle)
    shard = bundle / index["shards"][0]["name"]
    with tarfile.open(shard, "w") as archive:
        member = tarfile.TarInfo("../escape")
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))
    index["shards"][0]["sha256"] = sha256(shard)
    (bundle / "index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="member"):
        restore_dataset(bundle, tmp_path / "bad")
    assert not (tmp_path / "escape").exists()
