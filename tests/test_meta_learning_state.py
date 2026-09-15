import copy

import pytest
import torch


def setup_model():
    torch.manual_seed(27)
    model = torch.nn.Linear(3, 2)
    target = copy.deepcopy(model).requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    return model, target, optimizer


def update(model, target, optimizer, step):
    x = torch.rand(4, 3)
    y = torch.rand(4, 2) + target(x).detach()
    optimizer.zero_grad()
    (model(x) - y).square().mean().backward()
    optimizer.step()
    if step % 3 == 0:
        target.load_state_dict(model.state_dict())


def test_training_resume_restores_optimizer_target_and_rng_exactly(tmp_path):
    from latency_meta_mdp.meta.learning_state import load_learning_state, save_learning_state

    model, target, optimizer = setup_model()
    config = {"use_future": True, "policy_binding": {"policy": "frozen"}}
    for step in range(1, 5):
        update(model, target, optimizer, step)
    path = tmp_path / "resume.pt"
    save_learning_state(path, model, target, optimizer, 4, config)
    for step in range(5, 9):
        update(model, target, optimizer, step)
    restored, restored_target, restored_optimizer = setup_model()
    assert load_learning_state(path, restored, restored_target, restored_optimizer, config) == 4
    for step in range(5, 9):
        update(restored, restored_target, restored_optimizer, step)
    for a, b in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(target.parameters(), restored_target.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError, match="use_future"):
        load_learning_state(
            path, restored, restored_target, restored_optimizer, {**config, "use_future": False}
        )


def test_learning_rate_has_a_settling_phase_and_is_extendable():
    from latency_meta_mdp.meta.learning_state import learning_rate_at

    cfg = {
        "learning_rate": 3e-4,
        "minimum_learning_rate": 3e-5,
        "lr_decay_start": 6000,
        "lr_decay_end": 24000,
    }
    assert learning_rate_at(6000, cfg) == 3e-4
    assert 3e-5 < learning_rate_at(12000, cfg) < 3e-4
    assert learning_rate_at(24000, cfg) == learning_rate_at(36000, cfg) == 3e-5


@pytest.mark.parametrize("budget_objective", [False, True])
def test_trainer_command_resume_matches_uninterrupted_training(
    tmp_path, monkeypatch, budget_objective
):
    import json
    import sys
    from types import SimpleNamespace

    import wandb
    import yaml
    from safetensors.torch import load_file
    from test_meta_success_data import manifest_fixture

    from latency_meta_mdp.meta.train import main

    monkeypatch.setattr(
        wandb,
        "init",
        lambda **kw: SimpleNamespace(
            url="test", log=lambda *args, **kwargs: None, finish=lambda **kw: None
        ),
    )
    manifest = manifest_fixture(tmp_path)
    cfg = dict(
        objective="finite_horizon_success_v2",
        gamma=1.0,
        task_horizon_ticks=1000,
        call_cost=0.0,
        forecast_cost=0.0,
        q_precision="float32",
        seed=27,
        batch_size=2,
        learning_rate=0.0003,
        minimum_learning_rate=0.00003,
        lr_decay_start=1,
        lr_decay_end=4,
        recoverable_training=True,
        save_interval=2,
        target_update_interval=2,
        log_interval=1,
    )
    extra = []
    if budget_objective:
        from test_meta_cost import episode, profile

        cost_profile = profile()
        cost_profile["binding"] = {}
        cost_file = tmp_path / "cost.json"
        cost_file.write_text(json.dumps(cost_profile))
        for entry in json.loads(manifest.read_text())["episodes"]:
            from pathlib import Path

            import numpy as np

            p = Path(entry["result"])
            d = json.loads(p.read_text())
            e = episode()
            for key in (
                "stage_events",
                "bootstrap_calls",
                "policy_calls",
                "forecast_calls",
                "forecast_decodes",
            ):
                d[key] = e[key]
            for row, timestamps in zip(d["decision_transitions"], e["decision_transitions"]):
                row.update(timestamps)
            d["identity"]["conditioning"] = "native_rtc_forecast_rgb_v1"
            p.write_text(json.dumps(d))
            with np.load(entry["replay"]) as source:
                arrays = {k: source[k] for k in source.files}
            meta = json.loads(arrays["metadata_utf8"].tobytes())
            meta["identity"] = d["identity"]
            arrays["metadata_utf8"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
            np.savez_compressed(entry["replay"], **arrays)
        cfg.update(
            objective="rtc_budget_success_v1",
            target_method="greedy_trace",
            max_trace_steps=8,
            trace_decay=0.8,
        )
        extra = ["--cost-profile", str(cost_file), "--budget", "10", "--cost-multiplier", ".1"]

    def run(name, updates, resume=False, init=None):
        config = tmp_path / f"{name}-{updates}.yaml"
        config.write_text(yaml.safe_dump({**cfg, "updates": updates}))
        out = tmp_path / name
        args = [
            "train",
            "--replay-manifest",
            str(manifest),
            "--training-config",
            str(config),
            "--output-dir",
            str(out),
            "--device",
            "cpu",
            *extra,
        ]
        if resume:
            args += ["--resume-from", str(out / "recovery.pt")]
        if init is not None:
            args += ["--init-from", str(init)]
        monkeypatch.setattr(sys, "argv", args)
        main()
        return out

    uninterrupted = run("uninterrupted", 4)
    run("resumed", 2)
    resumed = run("resumed", 4, True)
    for key, value in load_file(uninterrupted / "model.safetensors").items():
        torch.testing.assert_close(
            value, load_file(resumed / "model.safetensors")[key], rtol=0, atol=0
        )
    assert json.loads((resumed / "training-summary.json").read_text())["segment_start_step"] == 2
    # Interruption after the last recovery save but before final export must be resumable.
    final_weights = load_file(resumed / "model.safetensors")
    (resumed / "model.safetensors").unlink()
    (resumed / "training-summary.json").unlink()
    run("resumed", 4, True)
    for key, value in final_weights.items():
        torch.testing.assert_close(
            value, load_file(resumed / "model.safetensors")[key], rtol=0, atol=0
        )
    if budget_objective:
        extra[-1] = ".2"
        with pytest.raises(ValueError, match="cost_multiplier"):
            run("resumed", 6, True)
        warm = run("new-objective", 2, init=uninterrupted)
        for key in ("vector_mean", "vector_scale"):
            torch.testing.assert_close(
                load_file(warm / "model.safetensors")[key],
                load_file(uninterrupted / "model.safetensors")[key],
            )
        assert json.loads((warm / "training-summary.json").read_text())["segment_start_step"] == 0
