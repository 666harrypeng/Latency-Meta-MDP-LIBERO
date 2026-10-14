import numpy as np
import pytest


def test_action_mapping_preserves_old_wait_durations_and_stage_legality():
    from latency_meta_mdp.meta.information.memory_schema import ActionSpec

    spec = ActionSpec()
    assert [spec.from_v1(i) for i in range(8)] == [0, 1, 3, 5, 7, 8, 9, 10]
    assert spec.name_for(3) == "sleep:4"
    assert np.flatnonzero(spec.legal("G", 23)).tolist() == [0, 1, 2, 8]
    assert np.flatnonzero(spec.legal("O", 50)).tolist() == [9]
    assert np.flatnonzero(spec.legal("P", 20, forecast_fresh=True)).tolist() == [10]
    with pytest.raises(ValueError):
        spec.legal("P", 50, forecast_fresh=False)
    with pytest.raises(ValueError):
        spec.legal("G", 19)
    small = ActionSpec(sleep_ticks=(1, 4))
    assert small.num_actions == 5
    assert small.id_for("launch") == 4


@pytest.mark.parametrize("ticks", [(1, True), (1, 1), (2, 1), (), (0, 1)])
def test_sleep_spec_rejects_ambiguous_vocabulary(ticks):
    from latency_meta_mdp.meta.information.memory_schema import ActionSpec

    with pytest.raises(ValueError):
        ActionSpec(sleep_ticks=ticks)


def test_public_events_reject_privileged_data_and_future_real_observations():
    from latency_meta_mdp.meta.information.memory_schema import PublicEvent

    payload = dict(visual=np.zeros((1, 2, 196, 384)), proprio=np.zeros((1, 16)))
    event = PublicEvent("obs0", "observe", (10, 2), (10,), payload)
    assert not event.payload["visual"].flags.writeable
    payload["visual"][0, 0, 0, 0] = 100
    assert event.payload["visual"][0, 0, 0, 0] == 0
    with pytest.raises(ValueError, match="future"):
        PublicEvent("obs1", "observe", (10, 2), (11,), payload)
    with pytest.raises(ValueError, match="fields"):
        PublicEvent("obs2", "observe", (10, 2), (10,), {**payload, "success": True})


def test_batch_checker_keeps_global_batch_independent_of_gpu_count():
    from latency_meta_mdp.meta.information.memory_schema import validate_batch

    for world, accumulation in [(1, 8), (2, 4), (4, 2), (8, 1)]:
        validate_batch(global_batch=64, world_size=world, microbatch=8, accumulation=accumulation)
    for kwargs in [
        dict(global_batch=7, world_size=2, microbatch=1, accumulation=3),
        dict(global_batch=8, world_size=2, microbatch=2, accumulation=1),
    ]:
        with pytest.raises(ValueError):
            validate_batch(**kwargs)


def test_attention_config_is_versioned_and_rejects_unknown_fields(tmp_path):
    import yaml

    from latency_meta_mdp.meta.information.config import load_config

    config = load_config("configs/experiments/conveyor_sort/meta_attention_v2.yaml")
    assert config["schema_version"] == 2
    assert config["model"]["num_queries"] == 8
    config["model"]["privileged_state"] = True
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        load_config(path)
