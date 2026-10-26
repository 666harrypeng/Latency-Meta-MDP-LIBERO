import json

import pytest

from latency_meta_mdp.io.fqi_handoff import build_recipe, checkpoint_payload
from latency_meta_mdp.meta.information.config import identity_hash, sha256


def test_portable_recipe_preserves_science_and_has_no_imported_meta(tmp_path):
    from test_optional_prediction import optional_config

    config = optional_config()
    config["splits"]["train"] = list(range(10000, 10064))
    starter = tmp_path / "data"
    starter.mkdir()
    (starter / "cohort.json").write_text(
        json.dumps(
            dict(
                binding=dict(
                    config_hash=identity_hash(config), scope="starter_training", schema_version=3
                )
            )
        )
    )
    recipe = build_recipe(config, starter, root=tmp_path)
    assert recipe["startup_sources"][0]["path"] == "data"
    assert recipe["initial_checkpoint"] is None
    assert recipe["startup"]["fitted_iterations"] == 40
    assert len(recipe["cycles"]) == recipe["replay_curriculum"]["total_cycles"] == 100
    assert recipe["global_batch"] == 256
    assert all(
        len(c["generation"]["scenes"]) * len(c["generation"]["regimes"]) == 24
        for c in recipe["cycles"]
    )
    assert all(
        set(c["generation"]["scenes"]) <= set(config["splits"]["train"]) for c in recipe["cycles"]
    )
    assert recipe["retention"]["recent_batches"] == 20
    limits = recipe["retention"]
    assert limits["max_run_bytes"] > limits["max_online_bytes"] + limits["max_cache_bytes"]
    acceptance = build_recipe(config, starter, root=tmp_path, acceptance=True)
    assert acceptance["startup"]["optimizer_updates"] == 64
    assert len(acceptance["cycles"]) == 1
    assert acceptance["replay_curriculum"] == recipe["replay_curriculum"]


def test_checkpoint_publish_allowlist_rejects_missing_or_changed_weights(tmp_path):
    import yaml
    from test_optional_prediction import optional_config

    config = optional_config()
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(config))
    recipe = tmp_path / "recipe.json"
    recipe.write_text("{}")
    checkpoint = tmp_path / "fit/iteration-0002"
    checkpoint.mkdir(parents=True)
    weights = checkpoint / "state.pt"
    weights.write_bytes(b"fixture")
    receipt = dict(
        files={"state.pt": sha256(weights)},
        binding=dict(
            schema_version=3,
            experiment_id=config["experiment_id"],
            config_hash=identity_hash(config),
            initialization=dict(kind="scratch"),
        ),
    )
    (checkpoint / "complete.json").write_text(json.dumps(receipt))
    journal = dict(
        binding=dict(recipe={}, config_sha256=sha256(cfg)),
        phases={"fit": dict(result=dict(artifacts={
            str(weights): sha256(weights),
            str(checkpoint / "complete.json"): sha256(checkpoint / "complete.json"),
        }))},
    )
    (checkpoint.parent / "campaign-state.json").write_text(json.dumps(journal))
    (checkpoint / "unrelated.secret").write_text("must not be uploaded")
    files = checkpoint_payload(checkpoint, cfg, recipe)
    assert set(files) == {"state.pt", "complete.json", "config.yaml", "recipe.json"}
    recipe.write_text('{"initial_checkpoint":"legacy/S05"}')
    with pytest.raises(ValueError, match="recipe"):
        checkpoint_payload(checkpoint, cfg, recipe)
    recipe.write_text("{}")
    weights.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint"):
        checkpoint_payload(checkpoint, cfg, recipe)


def test_asset_restore_rejects_existing_wrong_destination(tmp_path):
    from latency_meta_mdp.io.fqi_handoff import install_asset

    cached = tmp_path / "cache"
    cached.write_bytes(b"correct")
    target = tmp_path / "assets/model"
    target.parent.mkdir()
    target.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="destination"):
        install_asset(cached, target, sha256(cached))
    assert target.read_bytes() == b"wrong"


def test_reward_revision_reuses_only_explicit_compatible_starter(tmp_path):
    import copy

    import yaml
    from test_optional_prediction import optional_config

    old = optional_config()
    old['splits']['train'] = list(range(10000, 10064))
    source = tmp_path / 'recording.yaml'
    source.write_text(yaml.safe_dump(old))
    new = copy.deepcopy(old)
    new['experiment_id'] = 'budget-cost002'
    new['training']['reward'] = {'cost_weight': 0.02}
    new['training']['replay_source_configs'] = [
        {'path': 'recording.yaml', 'sha256': sha256(source)}]
    starter = tmp_path / 'starter'
    starter.mkdir()
    (starter / 'cohort.json').write_text(json.dumps({'binding': {
        'config_hash': identity_hash(old), 'scope': 'starter_training', 'schema_version': 3}}))
    recipe = build_recipe(new, starter, root=tmp_path)
    assert recipe['initial_checkpoint'] is None
    incompatible = copy.deepcopy(new)
    incompatible['actions']['reserve'] = 1
    with pytest.raises(ValueError):
        build_recipe(incompatible, starter, root=tmp_path)
    incompatible = copy.deepcopy(new)
    incompatible['splits']['train'] = list(range(10001, 10065))
    with pytest.raises(ValueError):
        build_recipe(incompatible, starter, root=tmp_path)
