from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_action_conditioned_jepa_data import _record

from latency_meta_mdp.belief.jepa.corpus import (
    JepaProprioNormalization,
    compute_jepa_proprio_normalization,
)
from latency_meta_mdp.belief.jepa.data import DirectPredictionDataset, materialize_direct_sample


def conveyor_record(tmp_path, *, split="train", episode_id="train-seed1000"):
    return replace(
        _record(tmp_path, terminal_tick=40, split=split, episode_id=episode_id),
        level=None,
        task_id="conveyor_sort",
        action_contract_id="panda_osc_pose_delta_conveyor_v2",
    )


def test_conveyor_normalization_and_direct_windows(tmp_path):
    record = conveyor_record(tmp_path)
    norm = compute_jepa_proprio_normalization(
        records=(record,), source_manifest_sha256="a" * 64, split_manifest_sha256="b" * 64
    )
    assert norm.level is None and norm.task_id == "conveyor_sort"
    payload = norm.to_mapping()
    assert payload["schema_version"] == 2 and "level" not in payload
    restored = JepaProprioNormalization.from_mapping(payload)
    np.testing.assert_array_equal(norm.mean, restored.mean)
    dataset = DirectPredictionDataset(records=(record,), normalization=restored)
    assert dataset.horizon_counts[0] == 30
    sample = materialize_direct_sample(record, source_tick=20, query_ticks=20, normalization=norm)
    np.testing.assert_array_equal(sample.query.vision_history[0, :, 0, 0, 0], [12, 16, 20])
    np.testing.assert_array_equal(sample.query.executable_controls[0], record.controls[20:40])
    assert (sample.target_visual == 40).all()
    with pytest.raises(ValueError):
        materialize_direct_sample(record, source_tick=21, query_ticks=20, normalization=norm)
    val = conveyor_record(tmp_path, split="validation", episode_id="val-seed2000")
    DirectPredictionDataset(records=(val,), normalization=norm)
    with pytest.raises(ValueError, match="only train"):
        compute_jepa_proprio_normalization(
            records=(val,), source_manifest_sha256="a" * 64, split_manifest_sha256="b" * 64
        )
    legacy = _record(tmp_path, terminal_tick=40, episode_id="legacy")
    with pytest.raises(ValueError, match="identity|domain"):
        DirectPredictionDataset(records=(legacy,), normalization=norm)


def test_conveyor_model_config_has_no_level_alias():
    from latency_meta_mdp.belief.jepa.config import load_action_conditioned_jepa_config

    cfg = load_action_conditioned_jepa_config(
        model_path=Path("configs/models/jepa/model.yaml"),
        task_id="conveyor_sort",
        control_path=Path("configs/runtime/control/panda_osc_pose_delta_conveyor_v2.yaml"),
        temporal_sampling_path=Path("configs/models/jepa/stride4_80ms_history_160ms.yaml"),
    )
    assert cfg.level is None and cfg.task_id == "conveyor_sort"
    assert cfg.action_contract.contract_id == "panda_osc_pose_delta_conveyor_v2"
    assert cfg.history_ticks == 3 and cfg.model_stride_ticks == 4


def test_conveyor_adapter_fits_train_only_and_loads_validation(tmp_path):
    import json
    from dataclasses import asdict

    import yaml
    from test_conveyor_collection import job as collection_job
    from test_conveyor_vision import Encoder

    from latency_meta_mdp.belief.jepa.conveyor import prepare_normalization
    from latency_meta_mdp.belief.jepa.job import load_direct_data, load_direct_job
    from latency_meta_mdp.data.conveyor.collection import collect
    from latency_meta_mdp.data.conveyor.source import ConveyorRecorder
    from latency_meta_mdp.data.conveyor.vision import prepare_features
    from latency_meta_mdp.envs.conveyor.config import load_conveyor_spec
    from latency_meta_mdp.envs.conveyor.expert import load_expert_spec
    from latency_meta_mdp.runtime.policy_execution import PolicyObservation

    def runner(scene, expert, output, *, seed, **kwargs):
        output.mkdir(parents=True)
        image = np.zeros((256, 256, 3), np.uint8)
        summary = dict(spawned=10, successes=10, misses=0, timeouts=0, active=0, end_tick=40)
        with ConveyorRecorder(
            output / "source",
            seed=seed,
            action_contract_id="panda_osc_pose_delta_conveyor_v2",
            purpose="training_source",
            context=dict(
                scene=asdict(load_conveyor_spec(scene)), expert=asdict(load_expert_spec(expert))
            ),
        ) as record:
            record.start(PolicyObservation(0, image, image, np.full(16, seed, dtype=np.float32)))
            for h in range(40):
                record.append(
                    np.zeros(7),
                    PolicyObservation(
                        h + 1, image, image, np.full(16, seed + (h + 1) / 100, dtype=np.float32)
                    ),
                    success_delta=10 if h == 39 else 0,
                    done=h == 39,
                )
            record.finish(summary)
        status = dict(
            status="completed", expert_admitted=True, all_scheduled_spawned=True, **summary
        )
        (output / "status.json").write_text(json.dumps(status))
        return status

    path = collection_job(tmp_path)
    value = yaml.safe_load(path.read_text())
    value.update(purpose="training_source", collect_splits=["train", "validation"])
    path.write_text(yaml.safe_dump(value))
    source = collect(path, tmp_path / "sources", run_episode=runner)
    features = prepare_features(source, tmp_path / "features", encoder=Encoder())
    value = yaml.safe_load(Path("configs/experiments/conveyor_sort/belief.yaml").read_text())
    value.update(
        source_root=str(source.parent),
        vision_cache_manifest=str(features),
        normalization=str(tmp_path / "norm.json"),
    )
    path = tmp_path / "belief.yaml"
    path.write_text(yaml.safe_dump(value))
    job = load_direct_job(path, project_root=Path.cwd())
    prepare_normalization(job)
    config, norm, corpus, data = load_direct_data(job, split="train")
    assert len(corpus.records) == 2 and len(norm.episode_ids) == 2
    np.testing.assert_allclose(norm.mean, 1000.7, atol=1e-4)
    assert norm.boundary_count == 82
    _, _, validation, valdata = load_direct_data(job, split="validation")
    assert len(validation.records) == 1
    assert set(norm.episode_ids).isdisjoint(r.episode_id for r in validation.records)
    assert valdata[0].query.query_ticks.item() == 1
    assert config.task_id == "conveyor_sort" and job.label == "conveyor_sort"


def test_task_checkpoint_cannot_load_as_legacy(tmp_path):
    from types import SimpleNamespace

    import torch
    from safetensors.torch import save_file

    from latency_meta_mdp.belief.jepa.model import load_direct_prediction_weights

    path = tmp_path / "wrong.safetensors"
    save_file(
        {"unused": torch.zeros(1)},
        str(path),
        metadata={"architecture_id": "jepa_direct_q20_history_stride4_w3_v1", "level": "3"},
    )
    model = SimpleNamespace(
        architecture_id="jepa_direct_q20_history_stride4_w3_v1",
        trunk=SimpleNamespace(
            config=SimpleNamespace(
                level=None,
                task_id="conveyor_sort",
                action_contract=SimpleNamespace(contract_id="panda_osc_pose_delta_conveyor_v2"),
            )
        ),
    )
    with pytest.raises(ValueError, match="task/controller"):
        load_direct_prediction_weights(model, path)


def test_conveyor_predictor_checkpoint_roundtrip(tmp_path):
    from safetensors import safe_open

    from latency_meta_mdp.belief.jepa.config import load_action_conditioned_jepa_config
    from latency_meta_mdp.belief.jepa.model import (
        DirectJepaPredictor,
        load_direct_prediction_weights,
        save_direct_prediction_weights,
    )

    record = conveyor_record(tmp_path)
    norm = compute_jepa_proprio_normalization(
        records=(record,), source_manifest_sha256="a" * 64, split_manifest_sha256="b" * 64
    )
    cfg = load_action_conditioned_jepa_config(
        model_path=Path("configs/models/jepa/model.yaml"),
        task_id="conveyor_sort",
        control_path=Path("configs/runtime/control/panda_osc_pose_delta_conveyor_v2.yaml"),
        temporal_sampling_path=Path("configs/models/jepa/stride4_80ms_history_160ms.yaml"),
    )
    model = DirectJepaPredictor(
        backbone_config=cfg, proprio_normalization=norm, project_root=Path.cwd()
    )
    path = tmp_path / "direct.safetensors"
    save_direct_prediction_weights(model, path)
    with safe_open(str(path), framework="pt", device="cpu") as stream:
        assert stream.metadata()["task_id"] == "conveyor_sort"
        assert "level" not in stream.metadata()
    load_direct_prediction_weights(model, path)
    wrong = replace(norm, mean=norm.mean + 1)
    mismatched = DirectJepaPredictor(
        backbone_config=cfg, proprio_normalization=wrong, project_root=Path.cwd()
    )
    with pytest.raises(ValueError, match="normalization"):
        load_direct_prediction_weights(mismatched, path)
