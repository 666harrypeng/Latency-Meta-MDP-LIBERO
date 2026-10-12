import copy
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from latency_meta_mdp.meta.information.model import StageQ
from latency_meta_mdp.meta.information.schema import ActorPacket


def replay():
    result = []
    for i in range(5):
        rows = []
        for t in range(i % 2 + 1):
            rows.append(
                dict(
                    packet=ActorPacket(10 + t, "G"),
                    action=5,
                    task_reward=0.1 if t == i % 2 else 0.0,
                    cost=0.0,
                    terminated=t == i % 2,
                )
            )
        result.append(rows)
    return result


def test_global_batch_matches_two_ranks_with_ragged_tail_and_unused_heads(tmp_path):
    from latency_meta_mdp.meta.information.distributed_fqi import fit_distributed

    torch.manual_seed(17)
    one = StageQ()
    two = copy.deepcopy(one)
    data = replay()
    common = dict(
        mode="bound",
        epochs=1,
        learning_rate=0.001,
        conservative_weight=0.01,
        global_batch=4,
        sources=["offline"] * 3 + ["online"] * 2,
    )
    events = []
    np.random.seed(42)
    a = fit_distributed(one, data, devices=["cpu"], progress=events.append, **common)
    np.random.seed(42)
    b = fit_distributed(two, data, devices=["cpu", "cpu"], **common)
    for key, value in one.state_dict().items():
        torch.testing.assert_close(value, two.state_dict()[key], rtol=1e-4, atol=1e-6)
    assert a["optimizer_updates"] == b["optimizer_updates"] == 2
    assert a["transition_visits"] == b["transition_visits"] == 14
    assert a["target_hash_before"] == a["target_hash_after"] == b["target_hash_before"]
    assert a["target_min"] == b["target_min"]
    assert a["loss"] == pytest.approx(b["loss"], abs=1e-6)
    assert len([e for e in events if e["phase"] == "fit"]) == 2


def test_worker_error_is_propagated_without_publishing_weights():
    from latency_meta_mdp.meta.information.distributed_fqi import fit_distributed

    model = StageQ()
    original = copy.deepcopy(model.state_dict())
    data = replay()
    data[-1][-1]["terminated"] = False
    with pytest.raises(Exception, match="terminal"):
        fit_distributed(
            model,
            data,
            devices=["cpu", "cpu"],
            mode="bound",
            epochs=1,
            learning_rate=0.001,
            conservative_weight=0.01,
            global_batch=4,
        )
    for k, v in original.items():
        torch.testing.assert_close(v, model.state_dict()[k], rtol=0, atol=0)


def test_peer_exiting_first_does_not_hide_original_rank_failure(monkeypatch):
    from latency_meta_mdp.meta.information import distributed_fqi as module

    @contextmanager
    def workers(*args):
        directory = Path(args[-1])
        (directory / "failure-0.txt").write_text("complete terminal episodes required")
        (directory / "failure-1.txt").write_text("connection closed by peer")
        yield [SimpleNamespace(exitcode=None), SimpleNamespace(exitcode=1)]

    monkeypatch.setattr(module, "worker_processes", workers)
    with pytest.raises(RuntimeError, match="terminal"):
        module.fit_distributed(StageQ(), replay(), devices=["cpu", "cpu"], mode="bound")


def test_learner_threads_are_runtime_options_and_must_be_positive(monkeypatch):
    from latency_meta_mdp.meta.information import distributed_fqi as module

    @contextmanager
    def workers(devices, state, episodes, options, orders, directory):
        assert options["cpu_threads"] == 2
        raise RuntimeError("reached configured worker launch")
        yield

    monkeypatch.setattr(module, "worker_processes", workers)
    with pytest.raises(ValueError, match="recipe"):
        module.fit_distributed(StageQ(), replay(), devices=["cpu"], mode="bound", cpu_threads=0)
    with pytest.raises(RuntimeError, match="configured worker"):
        module.fit_distributed(StageQ(), replay(), devices=["cpu"], mode="bound", cpu_threads=2)


def test_distributed_iteration_resume_matches_continuous(tmp_path):
    from latency_meta_mdp.meta.information.distributed_fqi import fit_distributed
    from latency_meta_mdp.meta.information.training import train_iterations

    torch.manual_seed(13)
    continuous = StageQ()
    split = copy.deepcopy(continuous)
    kwargs = dict(
        binding={"mode": "bound"},
        total_iterations=2,
        epochs=1,
        learning_rate=0.001,
        conservative_weight=0.01,
        fitter=fit_distributed,
        fitter_options={"devices": ["cpu"], "global_batch": 4},
    )
    np.random.seed(9)
    train_iterations(continuous, replay(), tmp_path / "full", **kwargs)
    np.random.seed(9)
    train_iterations(split, replay(), tmp_path / "split", stop_after_iterations=1, **kwargs)
    resumed = StageQ()
    kwargs["fitter_options"] = dict(devices=["cpu", "cpu"], global_batch=4, cpu_threads=2)
    train_iterations(resumed, replay(), tmp_path / "split", **kwargs)
    for key, value in continuous.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=1e-4, atol=1e-6)

    kwargs["fitter_options"]["global_batch"] = 2
    with pytest.raises(ValueError, match="binding"):
        train_iterations(resumed, replay(), tmp_path / "split", **kwargs)


def test_global_batch_one_preserves_original_episode_optimizer_semantics():
    from latency_meta_mdp.meta.information.distributed_fqi import fit_distributed
    from latency_meta_mdp.meta.information.fqi import fit_iteration

    torch.manual_seed(14)
    original = StageQ()
    distributed = copy.deepcopy(original)
    data = replay()
    sources = ["offline"] * 3 + ["online"] * 2
    kwargs = dict(
        mode="bound", epochs=1, learning_rate=0.001, conservative_weight=0.01, sources=sources
    )
    np.random.seed(3)
    a = fit_iteration(original, data, **kwargs)
    np.random.seed(3)
    b = fit_distributed(distributed, data, devices=["cpu"], global_batch=1, **kwargs)
    for k, v in original.state_dict().items():
        torch.testing.assert_close(v, distributed.state_dict()[k], rtol=1e-4, atol=1e-6)
    assert a["loss"] == pytest.approx(b["loss"], abs=1e-6)


def test_partial_rank_start_failure_cleans_started_worker(monkeypatch, tmp_path):
    from latency_meta_mdp.meta.information import distributed_fqi as module

    processes = []

    class Process:
        def __init__(self, **kwargs):
            self.pid = None
            self.alive = False
            self.terminated = False
            processes.append(self)

        def start(self):
            if len(processes) == 2:
                raise OSError("second rank launch failed")
            self.pid = 123
            self.alive = True

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminated = True
            self.alive = False

        def join(self, timeout=None):
            pass

    class Context:
        def Process(self, **kwargs):
            return Process(**kwargs)

    monkeypatch.setattr(module.mp, "get_context", lambda _: Context())
    with pytest.raises(OSError, match="second rank"):
        with module.worker_processes(["cpu", "cpu"], {}, [], {}, [], str(tmp_path)):
            pass
    assert processes[0].terminated
