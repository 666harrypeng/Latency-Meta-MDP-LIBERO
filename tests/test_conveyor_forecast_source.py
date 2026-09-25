import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def test_clean_episode_restores_real_terminal_boundary_and_retains_all_actions(tmp_path):
    from latency_meta_mdp.data.conveyor.forecast_source import CleanForecastEpisode
    from latency_meta_mdp.data.source.parquet import encode_png

    def image(value):
        return encode_png(np.full((256, 256, 3), value, np.uint8), compress_level=1)

    path = tmp_path / "episode.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                dict(
                    frame_index=i,
                    state=[float(i)] * 16,
                    actions=[i / 4] * 7,
                    image={"bytes": image(i), "path": None},
                    wrist_image={"bytes": image(i + 1), "path": None},
                )
                for i in range(4)
            ]
        ),
        path,
        row_group_size=2,
    )
    row = dict(episode_id="example", frame_count=4, seed=1000)
    terminal = dict(
        episode_id="example",
        formal_tick=4,
        seed=1000,
        state=[4.0] * 16,
        image=image(4),
        wrist_image=image(5),
    )
    source = CleanForecastEpisode(path, row, terminal)
    assert source.states.shape == (5, 16) and source.actions.shape == (4, 7)
    np.testing.assert_array_equal(source.states[-1], 4)
    np.testing.assert_array_equal(source.rgb(3)[0], 3)
    np.testing.assert_array_equal(source.rgb(4)[1], 5)
    np.testing.assert_array_equal(source.actions[-1], 0.75)
    with pytest.raises(ValueError, match="terminal"):
        CleanForecastEpisode(path, row, {**terminal, "formal_tick": 3})


def test_forecast_source_rejects_terminal_assets_from_other_export(tmp_path):
    from latency_meta_mdp.data.conveyor.forecast_source import load_terminal_assets

    export = tmp_path / "export.json"
    export.write_text(json.dumps({"episodes": []}))
    inputs = SimpleNamespace(policy_export_manifest=export, bundle_identity="a" * 64)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "format_id": "conveyor_forecast_terminal_boundaries_v1",
                "policy_export_manifest_sha256": "b" * 64,
                "training_bundle_manifest_sha256": "a" * 64,
            }
        )
    )
    with pytest.raises(ValueError, match="identity"):
        load_terminal_assets(inputs, tmp_path)


def test_clean_feature_preparation_includes_terminal_and_resumes_without_reencoding(
    tmp_path, monkeypatch
):
    from latency_meta_mdp.data.conveyor import forecast_prepare as module

    policy_manifest = tmp_path / "export.json"
    row = dict(episode_id="example", seed=1000, frame_count=8)
    export = dict(episodes=[row], source_manifest_sha256="a" * 64)
    policy_manifest.write_text(json.dumps(export))
    terminal_root = tmp_path / "terminal"
    terminal_root.mkdir()
    (terminal_root / "manifest.json").write_text("{}")
    inputs = SimpleNamespace(policy_export_manifest=policy_manifest)
    source = SimpleNamespace(
        states=np.zeros((9, 16)), rgb=lambda h: np.full((2, 256, 256, 3), h, np.uint8)
    )
    monkeypatch.setattr(module, "load_terminal_assets", lambda *_: (export, {}))
    monkeypatch.setattr(module, "clean_forecast_episodes", lambda *_: iter([(row, source)]))
    calls = []

    class Encoder:
        spec = SimpleNamespace(fingerprint="encoder")

        def encode_numpy(self, images):
            calls.append(len(images))
            return np.broadcast_to(images[:, 0, 0, 0, None, None], (len(images), 196, 384)).astype(
                np.float16
            )

    path = module.prepare_clean_features(
        inputs, terminal_root, tmp_path / "features", encoder=Encoder(), boundary_batch_size=4
    )
    data = np.load(path.parent / "seed-1000.npy")
    assert data.shape == (9, 2, 196, 384)
    np.testing.assert_array_equal(data[-1], 8)
    assert calls == [8, 8, 2]
    module.prepare_clean_features(
        inputs, terminal_root, tmp_path / "features", encoder=Encoder(), boundary_batch_size=4
    )
    assert calls == [8, 8, 2]
