from pathlib import Path
from types import SimpleNamespace

import pytest
from test_action_conditioned_jepa_data import _normalization, _record


def test_query_subsampling_preserves_episode_boundaries_and_terminal_endpoint(tmp_path):
    from latency_meta_mdp.belief.jepa.data import DirectPredictionDataset
    from latency_meta_mdp.belief.jepa.evaluation_data import horizon_indices

    record = _record(tmp_path, terminal_tick=35)
    dataset = DirectPredictionDataset(records=(record,), normalization=_normalization(record))
    for q in (1, 7, 20):
        indices = horizon_indices(dataset, q, source_stride=10)
        triples = [dataset.pair_at(i) for i in indices]
        assert triples[0] == (0, 10, q)
        assert triples[-1] == (0, 35 - q, q)
        assert len(indices) == len(set(indices))
        assert all(e == 0 and h + q <= 35 and query == q for e, h, query in triples)
    with pytest.raises(ValueError):
        horizon_indices(dataset, 20, source_stride=0)


def test_conveyor_evaluation_never_looks_for_a_legacy_ar_checkpoint():
    from latency_meta_mdp.belief.jepa.evaluate import load_legacy_reference

    assert load_legacy_reference(
        SimpleNamespace(task_id="conveyor_sort"), None, None, None, Path(".")
    ) == (None, None)


def test_review_loads_conveyor_rgb_without_moving_ball_metadata(tmp_path):
    from test_conveyor_vision import colored_episode

    from latency_meta_mdp.belief.jepa.evaluation_data import review_rgb

    folder = tmp_path / "episodes/seed-1000"
    colored_episode(
        Path("configs/tasks/conveyor_sort/surface.yaml"),
        Path("configs/data/expert/conveyor_surface.yaml"),
        folder,
        seed=1000,
    )
    images = review_rgb(
        SimpleNamespace(task_id="conveyor_sort", source_root=tmp_path),
        SimpleNamespace(logical_master_task_index=1000),
        [0, 1],
    )
    assert images[0][0].shape == (224, 224, 3)
    assert (images[1][0] == 2).all() and (images[1][1] == 3).all()


def test_conveyor_visual_selection_does_not_require_legacy_phase_labels(tmp_path):
    from dataclasses import replace

    from latency_meta_mdp.belief.jepa.evaluation_data import select_review_sources

    record = replace(_record(tmp_path, terminal_tick=35), phases=(None,) * 36)
    chosen = select_review_sources(SimpleNamespace(records=(record,)), task_id="conveyor_sort")
    assert len(chosen) == 1 and 10 <= chosen[0][1] <= 15


def test_metrics_only_finishes_without_running_gpu_timing(tmp_path, monkeypatch):
    import json

    from latency_meta_mdp.belief.jepa import review
    from latency_meta_mdp.belief.jepa.evaluate import finish_review

    def forbidden(**kwargs):
        raise AssertionError("prediction-only evaluation must not benchmark shared GPU")

    monkeypatch.setattr(review, "review_direct_outputs", forbidden)
    args = SimpleNamespace(metrics_only=True, output_dir=tmp_path, decoder_dir=None)
    job = SimpleNamespace(level=None, task_id="conveyor_sort")
    finish_review(args, job, None, None, None)
    result = json.loads((tmp_path / "completion.json").read_text())
    assert result["status"] == "metrics_complete"
    assert result["runtime_measured"] is False


def test_review_only_checks_saved_metrics_protocol(tmp_path):
    import json

    from latency_meta_mdp.belief.jepa.evaluate import load_saved_metrics

    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps(
            dict(
                level=None,
                task_id="conveyor_sort",
                source_stride_ticks=10,
                results=[{"query_ticks": q} for q in range(1, 21)],
            )
        )
    )
    job = SimpleNamespace(level=None, task_id="conveyor_sort")
    load_saved_metrics(tmp_path, job, 10)
    with pytest.raises(ValueError):
        load_saved_metrics(tmp_path, job, 1)
    value = json.loads(path.read_text())
    value["results"].pop()
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_saved_metrics(tmp_path, job, 10)
