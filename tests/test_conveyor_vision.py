import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml
from test_conveyor_collection import episode_runner, job

from latency_meta_mdp.data.conveyor.collection import collect
from latency_meta_mdp.data.conveyor.corpus import split_sources
from latency_meta_mdp.data.source.parquet import encode_png
from latency_meta_mdp.data.vision.contracts import load_vision_encoder_spec


class Encoder:
    spec = load_vision_encoder_spec(
        Path("configs/models/vision/dinov3_vits16_lvd1689m_224_v1.yaml")
    )

    def encode_numpy(self, images):
        # Preserve image values so tests catch swapped boundary/view ordering.
        values = images[:, 0, 0, 0].astype(np.float16)
        return np.broadcast_to(values[:, None, None], (len(images), 196, 384)).copy()


def colored_episode(scene, expert, output, *, seed, **kwargs):
    status = episode_runner(scene, expert, output, seed=seed, **kwargs)
    path = Path(output) / "source/frames.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    for tick, row in enumerate(rows):
        for view, key in enumerate(("agentview_rgb", "wrist_rgb")):
            row[key] = encode_png(
                np.full((256, 256, 3), seed % 100 + 2 * tick + view, np.uint8), compress_level=1
            )
    pq.write_table(table.from_pylist(rows, schema=table.schema), path)
    manifest = path.parent / "manifest.json"
    data = json.loads(manifest.read_text())
    data["files"]["frames.parquet"] = path.stat().st_size
    manifest.write_text(json.dumps(data))
    return status


def test_feature_cache_records_all_boundaries_and_preserves_splits(tmp_path):
    from latency_meta_mdp.data.conveyor.vision import prepare_features, verify_feature_cache

    config = job(tmp_path)
    value = yaml.safe_load(config.read_text())
    value["collect_splits"] = ["train", "validation"]
    config.write_text(yaml.safe_dump(value))
    manifest = collect(config, tmp_path / "corpus", run_episode=colored_episode)
    result = prepare_features(manifest, tmp_path / "features", encoder=Encoder())
    cache = verify_feature_cache(result, corpus_manifest=manifest)
    assert cache["status"] == "completed"
    assert cache["action_contract"] == "panda_osc_pose_delta_conveyor_v2"
    assert len(cache["episodes"]) == 3
    for row in cache["episodes"]:
        features = np.load(result.parent / row["features"], mmap_mode="r")
        assert features.shape == (2, 2, 196, 384)
        assert features.dtype == np.float16
        assert row["split"] == ("train" if row["seed"] < 2000 else "validation")
        assert row["boundary_count"] == 2
        assert row["transition_count"] == 1
        for tick in range(2):
            for view in range(2):
                np.testing.assert_array_equal(
                    features[tick, view], row["seed"] % 100 + 2 * tick + view
                )
    assert len(list(split_sources(manifest, split="validation"))) == 1
    with pytest.raises(ValueError, match="train/validation"):
        list(split_sources(manifest, split="test"))

    class NoEncode:
        spec = Encoder.spec

        def encode_numpy(self, images):
            raise AssertionError("resume should not recompute completed features")

    prepare_features(manifest, result.parent, encoder=NoEncode(), resume=True)
    data = json.loads(manifest.read_text())
    data["identity"] = "changed"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="identity"):
        verify_feature_cache(result, corpus_manifest=manifest)


def test_feature_cache_checks_truncated_payload(tmp_path):
    from latency_meta_mdp.data.conveyor.vision import prepare_features, verify_feature_cache

    manifest = collect(job(tmp_path), tmp_path / "corpus", run_episode=episode_runner)
    result = prepare_features(manifest, tmp_path / "features", encoder=Encoder())
    data = json.loads(result.read_text())
    (result.parent / data["episodes"][0]["features"]).write_bytes(b"truncated")
    with pytest.raises(ValueError, match="size"):
        verify_feature_cache(result, corpus_manifest=manifest)
