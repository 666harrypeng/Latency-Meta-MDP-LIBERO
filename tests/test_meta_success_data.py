import json
from pathlib import Path

import numpy as np
import pytest
from test_meta_training_data import write_episode


def manifest_fixture(tmp_path):
    from latency_meta_mdp.meta.success_data import PHYSICAL_KEYS

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
        result.write_text(
            json.dumps(
                {
                    "identity": meta["identity"],
                    "case": meta["case"],
                    "success": True,
                    "terminated": True,
                    "truncated": False,
                    "decision_transitions": [
                        {"reward": 0.0, "undiscounted_reward": 0.0, "duration_ticks": 4},
                        {"reward": 1.0, "undiscounted_reward": 1.0, "duration_ticks": 1},
                    ],
                }
            )
        )
        entries.append({"result": str(result), "replay": str(path)})
    m = tmp_path / "manifest.json"
    m.write_text(
        json.dumps(
            {
                "schema": 2,
                "status": "completed",
                "episodes": entries,
                "expected_episodes": 2,
                "train_masters": [1],
                "validation_masters": [2],
            }
        )
    )
    return m


def test_success_view_changes_objective_without_changing_source(tmp_path):
    from latency_meta_mdp.meta.success_data import load_success_replay

    manifest = manifest_fixture(tmp_path)
    before = (tmp_path / "1.npz").read_bytes()
    x, v, legal, data, binding, inventory, masters = load_success_replay(manifest)
    np.testing.assert_array_equal(data["reward"], [0, 1, 0, 1])
    np.testing.assert_array_equal(data["discount"], [1, 0, 1, 0])
    assert binding["discount_per_tick"] == 1 and binding["task_horizon_terminal"]
    assert masters == {"train": [1], "validation": [2]}
    assert (tmp_path / "1.npz").read_bytes() == before
    assert len(x) == len(v) == len(legal) == 4 and len(inventory) == 2


def test_sequence_view_keeps_duration_and_episode_links(tmp_path):
    from latency_meta_mdp.meta.success_data import load_success_replay

    data = load_success_replay(manifest_fixture(tmp_path))[3]
    np.testing.assert_array_equal(data["next_transition"], [1, -1, 3, -1])
    np.testing.assert_array_equal(data["duration_ticks"], [4, 1, 4, 1])
    np.testing.assert_array_equal(data["episode_id"], [0, 0, 1, 1])


def test_cost_view_preserves_original_reward_and_arrays(tmp_path):
    from test_meta_cost import episode, profile

    from latency_meta_mdp.meta.success_data import load_success_replay

    manifest = manifest_fixture(tmp_path)
    for p in (tmp_path / "1.json", tmp_path / "2.json"):
        d = json.loads(p.read_text())
        e = episode()
        for key in (
            "stage_events",
            "bootstrap_calls",
            "policy_calls",
            "forecast_calls",
            "forecast_decodes",
        ):
            d[key] = e[key]
        for row, timestamps in zip(d["decision_transitions"], e["decision_transitions"]):
            row.update(timestamps)
        p.write_text(json.dumps(d))
        source = p.with_suffix(".npz")
        with np.load(source) as a:
            arrays = {k: a[k] for k in a.files}
        metadata = json.loads(arrays["metadata_utf8"].tobytes())
        d["identity"]["conditioning"] = "native_rtc_forecast_rgb_v1"
        metadata["identity"] = d["identity"]
        arrays["metadata_utf8"] = np.frombuffer(json.dumps(metadata).encode(), np.uint8)
        np.savez_compressed(source, **arrays)
        p.write_text(json.dumps(d))
    before = (tmp_path / "1.npz").read_bytes()
    p = profile()
    p["binding"] = {}
    data = load_success_replay(manifest, cost_profile=p)[3]
    np.testing.assert_allclose(data["cost"], [0.21, 1.24, 0.21, 1.24])
    np.testing.assert_array_equal(data["task_reward"], [0, 1, 0, 1])
    assert (tmp_path / "1.npz").read_bytes() == before


def test_success_view_rejects_broken_physical_sequence(tmp_path):
    from latency_meta_mdp.meta.success_data import load_success_replay

    manifest = manifest_fixture(tmp_path)
    p = tmp_path / "1.npz"
    with np.load(p) as d:
        arrays = {k: d[k] for k in d.files}
    arrays["next_state_index"][0] = 0
    np.savez_compressed(p, **arrays)
    with pytest.raises(ValueError, match="sequence"):
        load_success_replay(manifest)


def test_legacy_sequence_timestamps_are_read_from_existing_result_audit(tmp_path):
    from latency_meta_mdp.meta.success_data import load_success_replay

    manifest = manifest_fixture(tmp_path)
    p = tmp_path / "1.npz"
    with np.load(p) as d:
        arrays = {k: d[k] for k in d.files}
    result = json.loads(p.with_suffix(".json").read_text())
    for key in ("start_tick", "end_tick"):
        for row, value in zip(result["decision_transitions"], arrays.pop(key)):
            row[key] = int(value)
    np.savez_compressed(p, **arrays)
    p.with_suffix(".json").write_text(json.dumps(result))
    data = load_success_replay(manifest)[3]
    np.testing.assert_array_equal(data["next_transition"], [1, -1, 3, -1])
    np.testing.assert_array_equal(data["duration_ticks"], [4, 1, 4, 1])


def test_success_view_rejects_episode_outcome_reward_disagreement(tmp_path):
    from latency_meta_mdp.meta.success_data import load_success_replay

    manifest = manifest_fixture(tmp_path)
    p = tmp_path / "1.json"
    r = json.loads(p.read_text())
    r["success"] = False
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError, match="reward"):
        load_success_replay(manifest)


def test_success_view_rejects_duplicate_sources(tmp_path):
    from latency_meta_mdp.meta.success_data import load_success_replay

    p = manifest_fixture(tmp_path)
    m = json.loads(p.read_text())
    m["episodes"][1] = m["episodes"][0]
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="duplicate"):
        load_success_replay(p)


def test_shared_visual_cache_preserves_random_batch_order_without_corpus_copy(tmp_path):
    import torch

    from latency_meta_mdp.meta.success_data import load_success_replay

    manifest = manifest_fixture(tmp_path)
    expected = load_success_replay(manifest)[0]
    cached = load_success_replay(manifest, visual_cache=tmp_path / "shared-cache")[0]
    indices = torch.tensor([3, 0, 3, 1])
    np.testing.assert_array_equal(cached[indices].numpy(), expected[indices.numpy()])
    np.testing.assert_array_equal(cached.to_tensor("cpu").numpy(), expected)
    paths = list((tmp_path / "shared-cache").glob("*.npy"))
    mtimes = {p: p.stat().st_mtime_ns for p in paths}
    load_success_replay(manifest, visual_cache=tmp_path / "shared-cache")
    assert len(paths) == 2 and mtimes == {p: p.stat().st_mtime_ns for p in paths}


def test_active_replay_append_keeps_previous_snapshot_and_rejects_eval_data(tmp_path):
    from latency_meta_mdp.meta.success_data import append_success_replay

    parent = manifest_fixture(tmp_path)
    original = parent.read_bytes()
    entries = json.loads(parent.read_text())["episodes"]
    # A copied shard from a validation master must not become online training data.
    entry = dict(entries[1])
    copied = tmp_path / "copied.npz"
    copied.write_bytes(Path(entry["replay"]).read_bytes())
    entry["replay"] = str(copied)
    with pytest.raises(ValueError, match="training"):
        append_success_replay(parent, [entry], tmp_path / "next.json")
    with pytest.raises(ValueError, match="duplicate"):
        append_success_replay(parent, [entries[0]], tmp_path / "next.json")
    assert parent.read_bytes() == original and not (tmp_path / "next.json").exists()
    entry = dict(entries[0])
    copied = tmp_path / "new-train.npz"
    copied.write_bytes(Path(entry["replay"]).read_bytes())
    entry["replay"] = str(copied)
    next_path = tmp_path / "next.json"
    append_success_replay(parent, [entry], next_path)
    assert json.loads(next_path.read_text())["expected_episodes"] == 3
    assert parent.read_bytes() == original
