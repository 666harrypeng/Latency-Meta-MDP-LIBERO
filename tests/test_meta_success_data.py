import json

import numpy as np
import pytest
from test_meta_training_data import write_episode


def manifest_fixture(tmp_path):
    from latency_meta_mdp.meta_success_data import PHYSICAL_KEYS

    entries = []
    for master, part in [(1, "train"), (2, "validation")]:
        path = tmp_path / f"{master}.npz"
        write_episode(path, master, part)
        with np.load(path, allow_pickle=False) as d:
            arrays = {k: d[k] for k in d.files}
        meta = json.loads(arrays["metadata_utf8"].tobytes())
        meta["identity"].update(dict.fromkeys(PHYSICAL_KEYS, "same"))
        meta["identity"]["maximum_steps"] = 1000
        arrays["metadata_utf8"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
        # Exercise old replay compatibility: raw reward remains in the episode audit JSON.
        del arrays["undiscounted_reward"]
        np.savez_compressed(path, **arrays)
        result = path.with_suffix(".json")
        result.write_text(json.dumps({
            "identity": meta["identity"], "case": meta["case"],
            "success": True, "terminated": True, "truncated": False,
            "decision_transitions": [
                {"reward": 0., "undiscounted_reward": 0., "duration_ticks": 4},
                {"reward": 1., "undiscounted_reward": 1., "duration_ticks": 1},
            ],
        }))
        entries.append({"result": str(result), "replay": str(path)})
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps({
        "schema": 2, "status": "completed", "episodes": entries,
        "expected_episodes": 2, "train_masters": [1], "validation_masters": [2],
    }))
    return m


def test_success_view_changes_objective_without_changing_source(tmp_path):
    from latency_meta_mdp.meta_success_data import load_success_replay

    manifest = manifest_fixture(tmp_path)
    before = (tmp_path / "1.npz").read_bytes()
    x, v, legal, data, binding, inventory, masters = load_success_replay(manifest)
    np.testing.assert_array_equal(data["reward"], [0, 1, 0, 1])
    np.testing.assert_array_equal(data["discount"], [1, 0, 1, 0])
    assert binding["discount_per_tick"] == 1 and binding["task_horizon_terminal"]
    assert masters == {"train": [1], "validation": [2]}
    assert (tmp_path / "1.npz").read_bytes() == before
    assert len(x) == len(v) == len(legal) == 4 and len(inventory) == 2


def test_success_view_rejects_episode_outcome_reward_disagreement(tmp_path):
    from latency_meta_mdp.meta_success_data import load_success_replay

    manifest = manifest_fixture(tmp_path)
    p = tmp_path / "1.json"
    r = json.loads(p.read_text())
    r["success"] = False
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError, match="reward"):
        load_success_replay(manifest)


def test_success_view_rejects_duplicate_sources(tmp_path):
    from latency_meta_mdp.meta_success_data import load_success_replay

    p = manifest_fixture(tmp_path)
    m = json.loads(p.read_text())
    m["episodes"][1] = m["episodes"][0]
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="duplicate"):
        load_success_replay(p)
