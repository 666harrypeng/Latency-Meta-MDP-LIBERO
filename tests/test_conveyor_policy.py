import json

import pytest
from test_conveyor_collection import episode_runner, job


def test_train_source_selection_excludes_other_splits(tmp_path):
    from latency_meta_mdp.data.conveyor.collection import collect
    from latency_meta_mdp.data.conveyor.policy import train_sources

    config = job(tmp_path)
    import yaml

    value = yaml.safe_load(config.read_text())
    value["collect_splits"] = ["train", "validation"]
    config.write_text(yaml.safe_dump(value))
    path = collect(config, tmp_path / "corpus", run_episode=episode_runner)
    assert [row["seed"] for row, _ in train_sources(path)] == [1000, 1001]
    corpus = json.loads(path.read_text())
    corpus["episodes"][-1]["split"] = "train"
    path.write_text(json.dumps(corpus))
    with pytest.raises(ValueError, match="split"):
        list(train_sources(path))
