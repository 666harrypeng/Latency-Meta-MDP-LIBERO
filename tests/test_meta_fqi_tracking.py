import io

import pytest

from latency_meta_mdp.meta.tracking import event_metrics, read_complete_lines


def test_fqi_progress_axes_and_loss_are_explicit():
    step, values = event_metrics(
        dict(phase="fit", iteration=2, epoch=1, episode=3, episodes=128, loss=0.12),
        total=20,
        epochs=2,
    )
    assert step == 2
    assert values["train/update"] == 259
    assert values["train/combined_loss"] == 0.12
    assert values["progress/percent"] == pytest.approx(100 * (384 + 128 + 3) / (20 * 384))
    assert not any("reward" in key for key in values)
    _, end = event_metrics(
        dict(
            phase="iteration_complete",
            iteration=2,
            metrics={
                "loss": 0.1,
                "target_hash_before": "x",
                "target_hash_after": "x",
                "q_min": -0.3,
                "q_max": 0.2,
                "seconds": 5,
                "transitions": 20,
                "q_above_one_fraction": 0,
                "target_min": 0,
                "target_max": 0.1,
                "cuda_peak_allocated_bytes": 100,
            },
        ),
        total=20,
        epochs=2,
    )
    assert end["fqi/combined_loss"] == 0.1
    assert end["fqi/target_unchanged"] == 1
    assert end["progress/percent"] == 10


def test_partial_tail_is_retried_not_discarded():
    stream = io.StringIO("first\npart")
    assert list(read_complete_lines(stream)) == ["first\n"]
    assert stream.tell() == 6
    stream.seek(0, 2)
    stream.write("ial\n")
    stream.seek(6)
    assert list(read_complete_lines(stream)) == ["partial\n"]


def test_tracker_backfills_and_finishes_without_touching_trainer(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace

    from latency_meta_mdp.meta.tracking import track_training

    class Run:
        step = 0
        url = "https://wandb.ai/test/project/runs/test"
        summary = {}
        values = []
        finished = []

        def define_metric(self, *args, **kwargs):
            pass

        def log(self, values, *, step):
            self.values.append(values)
            self.step = step + 1

        def finish(self, *, exit_code):
            self.finished.append(exit_code)

    run = Run()
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(
            util=SimpleNamespace(),
            init=lambda **kwargs: run,
            Settings=lambda **kwargs: kwargs,
            Table=lambda **kwargs: kwargs,
            Histogram=lambda values: values,
        ),
    )
    train = tmp_path / "train"
    cohort = tmp_path / "cohort"
    (train / "iteration-0000").mkdir(parents=True)
    cohort.mkdir()
    binding = dict(
        recipe=dict(total_iterations=1, epochs=2),
        seed=0,
        mode="bound",
        cohort_sha256="data",
        trainer_code_hash="code",
    )
    (train / "iteration-0000/complete.json").write_text(json.dumps({"binding": binding}))
    row = dict(
        reward=0.1,
        free_decisions=3,
        episode=dict(scene=1, regime="nominal", behavior="lazy"),
        parcels=dict(successes=1, spawned=10),
    )
    (cohort / "collection-audit.json").write_text(
        json.dumps(dict(remaining=0, verified_complete=128, episodes=[row] * 128))
    )
    log = tmp_path / "train.log"
    log.write_text(
        json.dumps(
            dict(
                phase="iteration_complete",
                iteration=1,
                metrics=dict(loss=0.1, target_hash_before="x", target_hash_after="x"),
            )
        )
        + "\n"
    )
    track_training(
        training_dir=train,
        log_path=log,
        cohort_dir=cohort,
        entity="test",
        project="project",
        name="test",
        trainer_pid=0,
    )
    assert run.finished == [0]
    assert any("fqi/combined_loss" in v for v in run.values)
    assert not any("evaluation/reward" in v for v in run.values)
    assert json.loads((train / "tracking/status.json").read_text())["complete"]
