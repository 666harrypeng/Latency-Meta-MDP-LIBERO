from attention_fixtures import small_config

from latency_meta_mdp.meta.information.cycle import build_cycle_manifest, grid_generation


def test_grid_and_repeated_train_scene_have_unique_generation_identity(tmp_path):
    config = small_config()
    audit = dict(
        local_inventory_pass=True,
        candidate_splits=config["splits"],
        excluded_scenes=config["excluded_scenes"],
        sources=[],
    )
    scenes = config["splits"]["train"][32:40]
    generation = grid_generation("startup-grid", scenes)
    kwargs = dict(root=tmp_path, code_hash="code", policy={}, phase="online", baselines=False)
    first = build_cycle_manifest(config, audit, generation=generation, **kwargs)
    assert len(first["episodes"]) == 288
    assert len({e["episode_id"] for e in first["episodes"]}) == 288
    second = build_cycle_manifest(
        config, audit, generation={**generation, "name": "repeat-grid"}, **kwargs
    )
    assert not {e["episode_id"] for e in first["episodes"]} & {
        e["episode_id"] for e in second["episodes"]
    }
    assert {e["scene"] for e in first["episodes"]} == set(scenes)
    assert len({str(e.get("pattern", e["behavior"])) for e in first["episodes"]}) == 12
