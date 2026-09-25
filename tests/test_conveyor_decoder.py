import json

import pytest
import yaml
from test_conveyor_collection import job
from test_conveyor_vision import Encoder, colored_episode


def test_decoder_adapter_pairs_frozen_features_with_original_rgb(tmp_path):
    from latency_meta_mdp.belief.decoder.conveyor import prepare_decoder_data
    from latency_meta_mdp.belief.decoder.data import (
        load_decoder_dataset,
        verify_visual_decoder_data,
    )
    from latency_meta_mdp.data.conveyor.collection import collect
    from latency_meta_mdp.data.conveyor.vision import prepare_features

    config = job(tmp_path)
    value = yaml.safe_load(config.read_text())
    value.update(purpose="training_source", collect_splits=["train", "validation"])
    config.write_text(yaml.safe_dump(value))
    source = collect(config, tmp_path / "sources", run_episode=colored_episode)
    cache = prepare_features(source, tmp_path / "vision", encoder=Encoder())
    manifest = prepare_decoder_data(
        source, cache, tmp_path / "decoder", project_root=tmp_path, stride=5
    )
    verify_visual_decoder_data(manifest, project_root=tmp_path)
    fit = load_decoder_dataset(manifest, project_root=tmp_path, partition="fit")
    validation = load_decoder_dataset(manifest, project_root=tmp_path, partition="holdout")
    assert len(fit) == 4 and len(validation) == 2
    z, rgb = fit[1]
    assert z.shape == (2, 196, 384) and rgb.shape == (2, 3, 224, 224)
    assert (z[0] == 2).all() and (rgb[0] == 2).all()
    assert (z[1] == 3).all() and (rgb[1] == 3).all()
    assert not list(manifest.parent.glob("*.npy"))
    data = json.loads(manifest.read_text())
    data["episodes"][0]["partition"] = "holdout"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="partition"):
        verify_visual_decoder_data(manifest, project_root=tmp_path)
