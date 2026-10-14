import pytest
from attention_fixtures import small_config

from latency_meta_mdp.meta.information.attention_replay import validate_sources
from latency_meta_mdp.meta.information.config import identity_hash


def test_versioned_sources_reject_eval_duplicates_and_protocol_changes():
    config = small_config()
    manifest = dict(
        binding=dict(
            schema_version=1, assets=config["assets"], mode="bound", scope="starter_training"
        ),
        episodes=[dict(episode_id="a", scene=10000, split="train")],
    )
    validate_sources(config, [manifest])
    for bad in [
        {**manifest, "binding": {**manifest["binding"], "scope": "evaluation"}},
        {**manifest, "episodes": [dict(episode_id="a", scene=11000, split="train")]},
        {**manifest, "episodes": manifest["episodes"] * 2},
    ]:
        with pytest.raises(ValueError):
            validate_sources(config, [bad])
    v2 = {
        **manifest,
        "binding": {
            **manifest["binding"],
            "schema_version": 2,
            "config_hash": identity_hash(config),
            "actions": config["actions"],
        },
    }
    validate_sources(config, [v2])
    with pytest.raises(ValueError):
        validate_sources(config, [{**v2, "binding": {**v2["binding"], "config_hash": "changed"}}])
