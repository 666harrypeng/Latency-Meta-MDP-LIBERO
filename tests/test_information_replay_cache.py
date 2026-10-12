import json

import numpy as np
import pytest

from latency_meta_mdp.meta.information.dataset import load_episode, save_episode
from latency_meta_mdp.meta.information.records import StageRecorder
from latency_meta_mdp.meta.information.schema import ActorPacket


def saved_episode(path):
    r = StageRecorder("cache")
    for tick in (10, 11):
        r.start(
            ActorPacket(tick, "G", current_tick=9, current_tokens=np.ones((2, 196, 384))),
            0,
            forced=False,
            probability=1,
        )
        r.add_reward(tick, 0)
    r.finish(12, terminated=True)
    save_episode(path, r, report={}, binding={"asset": "fixed"})


def test_loaded_visuals_share_immutable_storage_without_changing_values(tmp_path):
    saved_episode(tmp_path / "episode")
    rows = load_episode(tmp_path / "episode", binding={"asset": "fixed"})
    first, second = (r["packet"].current_tokens for r in rows)
    assert np.shares_memory(first, second)
    assert not first.flags.writeable
    np.testing.assert_array_equal(first, np.ones((2, 196, 384)))


def test_cache_reuses_verified_snapshot_but_rejects_changed_content_or_binding(tmp_path):
    from latency_meta_mdp.meta.information.replay_cache import EpisodeCache

    path = tmp_path / "episode"
    saved_episode(path)
    cache = EpisodeCache(max_bytes=2 * 2**20)
    first = cache.load(path, binding={"asset": "fixed"})
    again = cache.load(path, binding={"asset": "fixed"})
    assert again is first and cache.hits == 1 and cache.misses == 1
    with pytest.raises(ValueError, match="binding"):
        cache.load(path, binding={"asset": "other"})
    record = path / "records.json"
    record.write_text(record.read_text().replace('"tick": 10', '"tick": 19'))
    with pytest.raises(ValueError, match="hash"):
        cache.load(path, binding={"asset": "fixed"})


def test_cache_is_bounded_and_eviction_does_not_change_reward(tmp_path):
    from latency_meta_mdp.meta.information.replay_cache import EpisodeCache

    for name in ("a", "b"):
        saved_episode(tmp_path / name)
    cache = EpisodeCache(max_bytes=800_000)
    first = cache.load(tmp_path / "a", binding={"asset": "fixed"})
    cache.load(tmp_path / "b", binding={"asset": "fixed"})
    repeated = cache.load(tmp_path / "a", binding={"asset": "fixed"})
    assert repeated is not first and cache.misses == 3
    assert cache.bytes_used <= cache.max_bytes
    assert sum(t["task_reward"] for t in repeated) == 0
    # A changed, re-signed receipt must not be mistaken for the cached snapshot.
    receipt = tmp_path / "a/complete.json"
    data = json.loads(receipt.read_text())
    data["binding"] = {"asset": "different"}
    receipt.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="binding"):
        cache.load(tmp_path / "a", binding={"asset": "fixed"})
