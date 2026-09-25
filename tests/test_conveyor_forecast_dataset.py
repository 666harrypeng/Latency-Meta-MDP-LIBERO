import dataclasses
import json

import numpy as np
import pytest


@pytest.mark.parametrize("image_codec", ["png", "webp_lossless"])
def test_conveyor_forecast_view_reuses_clean_export_and_checks_controller(tmp_path, image_codec):
    from test_action_conditioned_jepa_data import _normalization, _record

    from latency_meta_mdp.data.forecast.cache import write_forecast_cache
    from latency_meta_mdp.data.forecast.dataset import load_forecast_policy_dataset

    record = dataclasses.replace(
        _record(tmp_path, terminal_tick=32),
        level=None,
        task_id="conveyor_sort",
        action_contract_id="panda_osc_pose_delta_conveyor_v2",
    )

    class Engine:
        def predict(self, query):
            n = len(query.query_ticks)
            rgb = np.broadcast_to(
                np.arange(224, dtype=np.uint8)[None, None, :, None, None], (n, 2, 224, 224, 3)
            ).copy()
            rgb[..., 1] = query.query_ticks.numpy()[:, None, None, None]
            return rgb, np.zeros((n, 16), np.float32)

    class Native:
        def __len__(self):
            return 32

        def __getitem__(self, index):
            return dict(
                image=np.zeros((4, 4, 3), np.uint8),
                wrist_image=np.zeros((4, 4, 3), np.uint8),
                state=record.proprio_physical[index],
                prompt="sort",
            )

    bindings = {
        k: "a" * 64
        for k in (
            "predictor_sha256",
            "decoder_sha256",
            "jepa_normalization_sha256",
            "source_manifest_sha256",
            "split_manifest_sha256",
            "vision_cache_manifest_sha256",
        )
    }
    bindings["predictor_architecture"] = "jepa_direct_q20_history_stride4_w3_v1"
    cache_root = tmp_path / "cache"
    write_forecast_cache(
        records=(record,),
        normalization=_normalization(record),
        engine=Engine(),
        output_dir=cache_root,
        bindings=bindings,
        image_codec=image_codec,
    )
    export = dict(
        format_id="metamdp_lerobot_v21",
        task_id="conveyor_sort",
        variant="surface",
        split="train",
        action_contract=record.action_contract_id,
        action_target_contract="masked_h50_real_actions_v1",
        source_manifest_sha256="a" * 64,
        episodes=[dict(episode_id=record.episode_id, seed=0, frame_count=32)],
    )
    path = tmp_path / "metamdp_dataset.json"
    path.write_text(json.dumps(export))
    spec = dict(cache_root=str(cache_root), policy_export_manifest=str(path), bindings=bindings)
    data = load_forecast_policy_dataset(Native(), spec)
    sample = data[10 * 20 + 6]
    assert sample["forecast"]["query_ticks"] == 7
    assert sample["forecast"]["rgb"].shape == (2, 224, 224, 3)
    np.testing.assert_array_equal(sample["forecast"]["rgb"][0, :, 0, 0], np.arange(224))
    assert (sample["forecast"]["rgb"][..., 1] == 7).all()
    np.testing.assert_array_equal(sample["actions"][:22], record.controls[10:])
    assert sample["actions_is_pad"].sum() == 28
    assert data[0]["forecast"]["rgb"] is None

    class FullNative:
        def __len__(self):
            return 64

        def __getitem__(self, index):
            assert index >= 32  # Only the cached second episode may enter the pilot.
            return {**Native()[index - 32], "frame_index": index - 32, "episode_index": 1}

    export["episodes"].insert(0, dict(episode_id="uncached", seed=999, frame_count=32))
    path.write_text(json.dumps(export))
    with pytest.raises(ValueError, match="inventory"):
        load_forecast_policy_dataset(FullNative(), spec)
    smoke = load_forecast_policy_dataset(FullNative(), {**spec, "smoke_subset": True})
    assert len(smoke) == 32 * 20
    np.testing.assert_array_equal(smoke[10 * 20 + 6]["actions"][:22], record.controls[10:])
    export["episodes"] = export["episodes"][1:]
    export["action_contract"] = "panda_osc_pose_delta_v1"
    path.write_text(json.dumps(export))
    with pytest.raises(ValueError, match="identit"):
        load_forecast_policy_dataset(Native(), spec)
