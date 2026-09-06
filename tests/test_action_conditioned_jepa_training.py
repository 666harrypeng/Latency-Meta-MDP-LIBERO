from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from latency_meta_mdp.belief.action_conditioned_jepa.contracts import LaunchContextBatch
from latency_meta_mdp.belief.action_conditioned_jepa.temporal_view import (
    SharedJepaSampleIndex,
    TemporalJepaBatch,
)


class _TinyPredictor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))
        self.rollout_horizons: list[int] = []

    def predict_next(self, **kwargs):
        context = kwargs["vision_history"]
        batch = context.shape[0]
        visual = torch.zeros(batch, 2, 196, 384) + self.bias
        proprio = torch.zeros(batch, 16) + self.bias
        return visual, proprio

    def rollout_endpoint_for_loss(self, context, horizon):
        self.rollout_horizons.append(horizon)
        batch = context.batch_size
        visual = torch.zeros(batch, 2, 196, 384) + self.bias + 1.0
        proprio = torch.zeros(batch, 16) + self.bias + 1.0
        return visual, proprio


def _batch() -> TemporalJepaBatch:
    context = LaunchContextBatch(
        vision_history=torch.zeros(2, 3, 2, 196, 384, dtype=torch.float16),
        proprio_history=torch.zeros(2, 3, 16, dtype=torch.float32),
        executed_controls=torch.zeros(2, 2, 4, 7, dtype=torch.float32),
        executable_controls=torch.zeros(2, 5, 4, 7, dtype=torch.float32),
    )
    indices = tuple(
        SharedJepaSampleIndex(
            level=3,
            split="train",
            episode_id=f"episode-{index}",
            source_tick=10,
            boundary_disposition="recorded_complete",
        )
        for index in range(2)
    )
    return TemporalJepaBatch(
        indices=indices,
        temporal_config_id="stride4_80ms_history_160ms",
        context=context,
        target_visual_latents=torch.zeros(2, 5, 2, 196, 384, dtype=torch.float16),
        target_proprio_normalized=torch.zeros(2, 5, 16, dtype=torch.float32),
        target_proprio_physical=torch.full((2, 5, 16), 10_000.0, dtype=torch.float32),
        target_absorbing=torch.zeros(2, 5, dtype=torch.bool),
    )


def test_training_loss_uses_teacher_forced_d1_and_k2_normalized_targets() -> None:
    """Catches random long rollout loss or accidental physical-unit proprio supervision."""

    from latency_meta_mdp.belief.action_conditioned_jepa.training import (
        compute_temporal_jepa_loss,
    )

    model = _TinyPredictor()
    terms = compute_temporal_jepa_loss(model=model, batch=_batch())

    assert model.rollout_horizons == [2]
    torch.testing.assert_close(terms.teacher_visual, torch.tensor(0.0))
    torch.testing.assert_close(terms.teacher_proprio, torch.tensor(0.0))
    torch.testing.assert_close(terms.rollout_visual, torch.tensor(1.0))
    torch.testing.assert_close(terms.rollout_proprio, torch.tensor(1.0))
    torch.testing.assert_close(terms.total, torch.tensor(2.0))
    terms.total.backward()
    torch.testing.assert_close(model.bias.grad, torch.tensor(4.0))


def test_training_loss_rejects_nonfinite_predictions() -> None:
    """Catches publishing a training step after latent or proprio loss becomes nonfinite."""

    from latency_meta_mdp.belief.action_conditioned_jepa.training import (
        compute_temporal_jepa_loss,
    )

    model = _TinyPredictor()
    model.bias.data.fill_(float("nan"))

    try:
        compute_temporal_jepa_loss(model=model, batch=_batch())
    except FloatingPointError as error:
        assert "nonfinite" in str(error)
    else:
        raise AssertionError("nonfinite JEPA loss was accepted")


def test_training_config_matches_upstream_metaworld_optimizer_semantics() -> None:
    """Catches changing model-quality settings to accommodate one hardware target."""

    from latency_meta_mdp.belief.action_conditioned_jepa.training import (
        load_temporal_jepa_training_config,
    )

    config = load_temporal_jepa_training_config(
        Path("configs/training/action_conditioned_jepa/temporal_selection_l3.yaml")
    )

    assert config.max_epochs == 50
    assert config.logical_global_batch_size == 256
    assert config.learning_rate_start == 5e-4
    assert config.learning_rate_reference == 5e-4
    assert config.learning_rate_final == 5e-4
    assert config.weight_decay_start == 1e-7
    assert config.weight_decay_final == 1e-6
    assert config.schedule_scale == 1.0
    assert config.warmup_epochs == 0
    assert config.adamw_betas == (0.9, 0.999)
    assert config.adamw_epsilon == 1e-8
    assert config.gradient_clip_norm == 1.0
    assert config.selection_seed == 7
    assert config.confirmation_seeds == (17, 27)
    assert config.monitor_period_epochs == 5
    assert config.monitor_contexts_per_episode == 4
    assert config.checkpoint_period_epochs == 1
    assert config.milestone_epochs == (20, 40, 50)
    assert config.console_progress_period_optimizer_steps == 5


def test_optimizer_excludes_bias_and_norm_and_reproduces_upstream_schedules() -> None:
    """Catches applying weight decay to bias/norm or a hardware-dependent learning rate."""

    from latency_meta_mdp.belief.action_conditioned_jepa.training import (
        apply_upstream_optimizer_schedule,
        build_upstream_aligned_optimizer,
        load_temporal_jepa_training_config,
    )

    config = load_temporal_jepa_training_config(
        Path("configs/training/action_conditioned_jepa/temporal_selection_l3.yaml")
    )
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.LayerNorm(4),
    )
    optimizer = build_upstream_aligned_optimizer(model=model, config=config)

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["weight_decay"] == 1e-7
    assert optimizer.param_groups[1]["weight_decay"] == 0.0
    assert optimizer.param_groups[1]["WD_exclude"] is True
    lr1, wd1 = apply_upstream_optimizer_schedule(
        optimizer=optimizer,
        config=config,
        optimizer_step=1,
        total_optimizer_steps=100,
    )
    lr100, wd100 = apply_upstream_optimizer_schedule(
        optimizer=optimizer,
        config=config,
        optimizer_step=100,
        total_optimizer_steps=100,
    )

    assert lr1 == 5e-4
    assert lr100 == 5e-4
    assert wd1 == pytest.approx(1.002220478322758e-7)
    assert wd100 == 1e-6
    assert optimizer.param_groups[1]["weight_decay"] == 0.0
    with pytest.raises(ValueError, match="schedule step"):
        apply_upstream_optimizer_schedule(
            optimizer=optimizer,
            config=config,
            optimizer_step=0,
            total_optimizer_steps=100,
        )


def test_epoch_microbatches_preserve_logical_batch_and_resume_order() -> None:
    """Catches hardware microbatch changes that alter the logical sample order."""

    from latency_meta_mdp.belief.action_conditioned_jepa.training import (
        build_epoch_microbatch_indices,
    )

    first = build_epoch_microbatch_indices(
        dataset_size=1_025,
        logical_global_batch_size=256,
        microbatch_size=64,
        seed=7,
        epoch=3,
        optimizer_steps_per_epoch=4,
    )
    repeated = build_epoch_microbatch_indices(
        dataset_size=1_025,
        logical_global_batch_size=256,
        microbatch_size=64,
        seed=7,
        epoch=3,
        optimizer_steps_per_epoch=4,
    )
    next_epoch = build_epoch_microbatch_indices(
        dataset_size=1_025,
        logical_global_batch_size=256,
        microbatch_size=64,
        seed=7,
        epoch=4,
        optimizer_steps_per_epoch=4,
    )

    assert first == repeated
    assert first != next_epoch
    assert len(first) == 16
    assert all(len(values) == 64 for values in first)
    assert len(set(value for values in first for value in values)) == 1_024
    with pytest.raises(ValueError, match="divide"):
        build_epoch_microbatch_indices(
            dataset_size=1_025,
            logical_global_batch_size=256,
            microbatch_size=30,
            seed=7,
            epoch=0,
            optimizer_steps_per_epoch=4,
        )


def test_epoch_checkpoint_restores_model_optimizer_and_progress_exactly(tmp_path: Path) -> None:
    """Catches resume that restores weights but loses Adam moments or progress counters."""

    from latency_meta_mdp.belief.action_conditioned_jepa.training import (
        JepaTrainingProgress,
        build_upstream_aligned_optimizer,
        load_temporal_jepa_training_checkpoint,
        load_temporal_jepa_training_config,
        write_temporal_jepa_training_checkpoint,
    )

    config = load_temporal_jepa_training_config(
        Path("configs/training/action_conditioned_jepa/temporal_selection_l3.yaml")
    )
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4))
    optimizer = build_upstream_aligned_optimizer(model=model, config=config)
    inputs = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    model(inputs).square().mean().backward()
    optimizer.step()
    expected_parameters = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    expected_optimizer = optimizer.state_dict()
    progress = JepaTrainingProgress(
        completed_epochs=1,
        optimizer_steps=4,
        examples_seen=1_024,
        model_seed=7,
        fold_index=0,
        temporal_config_id="stride4_80ms_history_160ms",
    )
    output = tmp_path / "checkpoint"
    manifest = write_temporal_jepa_training_checkpoint(
        output_dir=output,
        model=model,
        optimizer=optimizer,
        progress=progress,
        training_config=config,
        input_sha256={
            "source_manifest": "a" * 64,
            "cache_manifest": "b" * 64,
            "split_manifest": "c" * 64,
            "selection_manifest": "d" * 64,
            "model_config": "e" * 64,
            "temporal_config": "f" * 64,
            "training_config": "0" * 64,
        },
    )
    restored_model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4))
    restored_optimizer = build_upstream_aligned_optimizer(
        model=restored_model,
        config=config,
    )
    restored = load_temporal_jepa_training_checkpoint(
        output_dir=output,
        model=restored_model,
        optimizer=restored_optimizer,
        expected_training_config=config,
        expected_input_sha256={
            "source_manifest": "a" * 64,
            "cache_manifest": "b" * 64,
            "split_manifest": "c" * 64,
            "selection_manifest": "d" * 64,
            "model_config": "e" * 64,
            "temporal_config": "f" * 64,
            "training_config": "0" * 64,
        },
        expected_temporal_config_id=progress.temporal_config_id,
        expected_fold_index=0,
        expected_model_seed=7,
    )

    assert manifest == output / "manifest.json"
    assert restored == progress
    for name, value in restored_model.state_dict().items():
        torch.testing.assert_close(value, expected_parameters[name], atol=0.0, rtol=0.0)
    assert restored_optimizer.state_dict()["param_groups"] == expected_optimizer["param_groups"]
    for key, state in restored_optimizer.state_dict()["state"].items():
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(
                    value,
                    expected_optimizer["state"][key][name],
                    atol=0.0,
                    rtol=0.0,
                )
    mismatched_optimizer = build_upstream_aligned_optimizer(
        model=restored_model,
        config=config,
    )
    with pytest.raises(ValueError, match="checkpoint inputs"):
        load_temporal_jepa_training_checkpoint(
            output_dir=output,
            model=restored_model,
            optimizer=mismatched_optimizer,
            expected_training_config=config,
            expected_input_sha256={
                "source_manifest": "1" * 64,
                "cache_manifest": "b" * 64,
                "split_manifest": "c" * 64,
                "selection_manifest": "d" * 64,
                "model_config": "e" * 64,
                "temporal_config": "f" * 64,
                "training_config": "0" * 64,
            },
            expected_temporal_config_id=progress.temporal_config_id,
            expected_fold_index=0,
            expected_model_seed=7,
        )
    with pytest.raises(FileExistsError):
        write_temporal_jepa_training_checkpoint(
            output_dir=output,
            model=model,
            optimizer=optimizer,
            progress=progress,
            training_config=config,
            input_sha256={
                "source_manifest": "a" * 64,
                "cache_manifest": "b" * 64,
                "split_manifest": "c" * 64,
                "selection_manifest": "d" * 64,
                "model_config": "e" * 64,
                "temporal_config": "f" * 64,
                "training_config": "0" * 64,
            },
        )


def test_one_epoch_accumulates_to_logical_global_batch_independent_of_microbatch() -> None:
    """Catches stepping Adam once per hardware microbatch instead of per logical batch."""

    from latency_meta_mdp.belief.action_conditioned_jepa.training import (
        JepaTrainingProgress,
        load_temporal_jepa_training_config,
        run_temporal_jepa_epoch,
    )

    config = load_temporal_jepa_training_config(
        Path("configs/training/action_conditioned_jepa/temporal_selection_l3.yaml")
    )
    model = _TinyPredictor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate_reference)
    batch = _batch()
    logged = []
    result = run_temporal_jepa_epoch(
        model=model,
        optimizer=optimizer,
        microbatches=(batch for _ in range(128)),
        training_config=config,
        progress=JepaTrainingProgress(
            completed_epochs=0,
            optimizer_steps=0,
            examples_seen=0,
            model_seed=7,
            fold_index=0,
            temporal_config_id=batch.temporal_config_id,
        ),
        optimizer_steps_this_epoch=1,
        total_optimizer_steps=50,
        optimizer_step_callback=logged.append,
    )

    assert result.progress.completed_epochs == 1
    assert result.progress.optimizer_steps == 1
    assert result.progress.examples_seen == 256
    assert result.microbatch_count == 128
    assert result.optimizer_step_count == 1
    assert result.mean_total_loss > 0
    assert optimizer.state_dict()["state"]
    assert len(logged) == 1
    assert logged[0]["optimizer_step"] == 1
    assert logged[0]["examples_seen"] == 256


def test_admission_epoch_updates_stage_progress_without_fabricating_a_fold() -> None:
    """Catches representing the all-80-master final refit as a nonexistent fifth fold."""

    from latency_meta_mdp.belief.action_conditioned_jepa.admission_runner import (
        load_jepa_admission_training_config,
    )
    from latency_meta_mdp.belief.action_conditioned_jepa.training import (
        JepaAdmissionProgress,
        run_temporal_jepa_epoch,
    )

    config = load_jepa_admission_training_config(
        Path("configs/training/action_conditioned_jepa/l3_admission.yaml")
    )
    model = _TinyPredictor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate_reference)
    batch = _batch()
    result = run_temporal_jepa_epoch(
        model=model,
        optimizer=optimizer,
        microbatches=(batch for _ in range(128)),
        training_config=config,
        progress=JepaAdmissionProgress(
            completed_epochs=0,
            optimizer_steps=0,
            examples_seen=0,
            model_seed=17,
            temporal_config_id=batch.temporal_config_id,
            stage_id="l3-stride4-final-admission-v1",
            level=3,
        ),
        optimizer_steps_this_epoch=1,
        total_optimizer_steps=13_200,
    )

    assert isinstance(result.progress, JepaAdmissionProgress)
    assert result.progress.completed_epochs == 1
    assert result.progress.optimizer_steps == 1
    assert result.progress.examples_seen == 256
    assert result.progress.stage_id == "l3-stride4-final-admission-v1"
    assert result.progress.level == 3
    assert not hasattr(result.progress, "fold_index")


def test_rolling_checkpoint_keeps_only_latest_and_20_40_50_milestones(tmp_path: Path) -> None:
    """Catches retaining every epoch or deleting a declared permanent milestone."""

    from latency_meta_mdp.belief.action_conditioned_jepa.training import (
        JepaTrainingProgress,
        build_upstream_aligned_optimizer,
        load_temporal_jepa_training_config,
        publish_rolling_temporal_jepa_checkpoint,
    )

    config = load_temporal_jepa_training_config(
        Path("configs/training/action_conditioned_jepa/temporal_selection_l3.yaml")
    )
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4))
    optimizer = build_upstream_aligned_optimizer(model=model, config=config)
    run_root = tmp_path / "run"
    hashes = {
        "source_manifest": "a" * 64,
        "cache_manifest": "b" * 64,
        "split_manifest": "c" * 64,
        "selection_manifest": "d" * 64,
        "model_config": "e" * 64,
        "temporal_config": "f" * 64,
        "training_config": "0" * 64,
    }
    for epoch in (1, 2, 20, 21, 40, 41, 50):
        publish_rolling_temporal_jepa_checkpoint(
            run_root=run_root,
            model=model,
            optimizer=optimizer,
            progress=JepaTrainingProgress(
                completed_epochs=epoch,
                optimizer_steps=epoch,
                examples_seen=epoch * 256,
                model_seed=7,
                fold_index=0,
                temporal_config_id="stride4_80ms_history_160ms",
            ),
            training_config=config,
            input_sha256=hashes,
        )

    checkpoints = sorted(path.name for path in (run_root / "checkpoints").iterdir())
    assert checkpoints == ["epoch-020", "epoch-040", "epoch-050"]
    latest = json.loads((run_root / "latest.json").read_text(encoding="utf-8"))
    assert latest == {
        "checkpoint": "checkpoints/epoch-050",
        "completed_epochs": 50,
    }


def test_resume_history_loads_every_completed_epoch_in_order(tmp_path: Path) -> None:
    """Catches a resumed final manifest silently dropping pre-resume epoch metrics."""

    from latency_meta_mdp.belief.action_conditioned_jepa.training import (
        load_completed_temporal_jepa_history,
    )

    metrics = tmp_path / "metrics"
    metrics.mkdir()
    for epoch in (1, 2, 3):
        (metrics / f"epoch-{epoch:03d}.json").write_text(
            json.dumps({"epoch": epoch, "mean_total_loss": 1.0 / epoch}) + "\n",
            encoding="utf-8",
        )

    assert [
        item["epoch"]
        for item in load_completed_temporal_jepa_history(
            run_root=tmp_path,
            completed_epochs=3,
        )
    ] == [1, 2, 3]

    (metrics / "epoch-002.json").unlink()
    with pytest.raises(ValueError, match="epoch history"):
        load_completed_temporal_jepa_history(
            run_root=tmp_path,
            completed_epochs=3,
        )
