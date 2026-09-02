from __future__ import annotations

from pathlib import Path

import numpy as np
from test_expert_source_loader import _publish


def test_k6_d20_h50_and_visual_frames_derive_without_simulator(
    tmp_path: Path, monkeypatch
) -> None:
    """Break caught: a model-data change needs simulator state that source storage omitted."""
    from latency_meta_mdp.expert_realization import task_instance
    from latency_meta_mdp.expert_realization.source_corpus.loader import (
        load_verified_source_corpus,
    )
    from latency_meta_mdp.expert_realization.source_corpus.parquet import decode_png

    def forbidden(*args, **kwargs):
        raise AssertionError("offline derivation called the simulator")

    monkeypatch.setattr(task_instance, "_build_task_instance_runtime", forbidden)
    corpus = load_verified_source_corpus(_publish(tmp_path, boundary_count=61))
    episode = corpus.read_episode("source-l1-task000-r000")
    frames = episode.frames
    launch_tick = 5

    k6 = frames.slice(0, 6)
    past_actions = frames["expert_action"].slice(0, 5).to_pylist()
    d20_actions = frames["expert_action"].slice(launch_tick, 20).to_pylist()
    d20_future_qpos = frames["robot_qpos"].slice(launch_tick + 1, 20).to_pylist()
    h50_actions = frames["expert_action"].slice(launch_tick, 50).to_pylist()
    latency_weights = np.full(20, 1.0 / 20.0, dtype=np.float64)
    first_rgb = decode_png(k6["agentview_rgb"][0].as_py()["bytes"])
    last_rgb = decode_png(frames["wrist_rgb"][60].as_py()["bytes"])

    assert k6.num_rows == 6
    assert len(past_actions) == 5
    assert len(d20_actions) == len(d20_future_qpos) == 20
    assert len(h50_actions) == 50
    assert np.isclose(latency_weights.sum(), 1.0)
    assert first_rgb.shape == last_rgb.shape == (256, 256, 3)
    assert int(first_rgb[0, 0, 0]) == 0
    assert int(last_rgb[0, 0, 0]) == 61


def test_terminal_state_supports_offline_absorbing_tail_materialization(tmp_path: Path) -> None:
    """Break caught: late H50 targets require restarting the simulator after collection."""
    from latency_meta_mdp.expert_realization.source_corpus.loader import (
        load_verified_source_corpus,
    )

    corpus = load_verified_source_corpus(_publish(tmp_path, boundary_count=61))
    frames = corpus.read_episode("source-l1-task000-r000").frames
    terminal = frames.slice(60, 1)
    requested_boundaries = 80
    tail_count = requested_boundaries - frames.num_rows
    absorbing_tail = [terminal for _ in range(tail_count)]

    assert frames["outcome_status"][60].as_py() == "success"
    assert frames["expert_action"][60].as_py() is None
    assert len(absorbing_tail) == 19
    expected_qpos = terminal["robot_qpos"][0].as_py()
    assert all(row["robot_qpos"][0].as_py() == expected_qpos for row in absorbing_tail)
