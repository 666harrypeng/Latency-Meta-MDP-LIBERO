from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("level", [1, 2])
def test_final_preflight_counts_level_sources_without_l3_fold_reuse(tmp_path, level):
    import pyarrow.parquet as pq

    from latency_meta_mdp.belief.action_conditioned_jepa.admission_runner import (
        build_jepa_admission_preflight,
    )

    source = Path("outputs/source_corpus/panda-ball-structured-source-quota-formal-100x4-v1")
    split = json.loads(
        Path(
            "outputs/derived/source_splits/panda-ball-structured-source-quota-formal-100x4-v1/"
            "train80-validation20-seed20260903-v1.json"
        ).read_text()
    )
    rows = pq.read_table(
        source / "meta/episodes.parquet", columns=["episode_id", "level", "terminal_tick"]
    ).to_pylist()
    expected_contexts = sum(
        row["terminal_tick"] - 10
        for row in rows
        if row["level"] == level and row["episode_id"] in split["train_episode_ids"]
    )
    value = build_jepa_admission_preflight(
        project_root=Path.cwd(),
        level=level,
        model_seed=27,
        microbatch_size=32,
        num_workers=4,
        device="cuda:0",
        output_dir=tmp_path / "run",
    )
    assert value.level == level
    assert value.train_context_count == expected_contexts
    assert value.train_episode_count == 320
    assert value.optimizer_steps_per_epoch == expected_contexts // 256
    assert value.total_optimizer_steps == 75 * (expected_contexts // 256)
    assert value.formal_validation_opened is False


def test_non_l3_validation_gate_uses_its_own_training_budget():
    from latency_meta_mdp.belief.action_conditioned_jepa.admission_runner import (
        load_jepa_admission_training_config,
        require_jepa_formal_validation_gate,
    )
    from latency_meta_mdp.belief.action_conditioned_jepa.training import JepaAdmissionProgress

    config = load_jepa_admission_training_config(
        Path("configs/training/action_conditioned_jepa/final_admission.yaml")
    )
    progress = JepaAdmissionProgress(
        level=1,
        completed_epochs=75,
        optimizer_steps=75 * 171,
        examples_seen=75 * 171 * 256,
        model_seed=27,
        temporal_config_id="stride4_80ms_history_160ms",
        stage_id="l1-stride4-final-admission-v1",
    )
    require_jepa_formal_validation_gate(progress, config=config, expected_optimizer_steps=75 * 171)
    with pytest.raises(RuntimeError):
        require_jepa_formal_validation_gate(
            progress, config=config, expected_optimizer_steps=75 * 172
        )


@pytest.mark.parametrize("level", [1, 2])
def test_level_checkpoint_roundtrip_rejects_cross_level_resume(tmp_path, level):
    import torch

    from latency_meta_mdp.belief.action_conditioned_jepa.admission_runner import (
        load_jepa_admission_training_checkpoint,
        load_jepa_admission_training_config,
        write_jepa_admission_training_checkpoint,
    )
    from latency_meta_mdp.belief.action_conditioned_jepa.training import JepaAdmissionProgress

    config = load_jepa_admission_training_config(
        Path("configs/training/action_conditioned_jepa/final_admission.yaml")
    )
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    hashes = {
        name: "a" * 64
        for name in (
            "source_manifest",
            "cache_manifest",
            "split_manifest",
            "selection_manifest",
            "model_config",
            "temporal_config",
            "training_config",
        )
    }
    progress = JepaAdmissionProgress(
        level=level,
        completed_epochs=1,
        optimizer_steps=1,
        examples_seen=256,
        model_seed=27,
        temporal_config_id="stride4_80ms_history_160ms",
        stage_id=f"l{level}-stride4-final-admission-v1",
    )
    write_jepa_admission_training_checkpoint(
        output_dir=tmp_path / "checkpoint",
        model=model,
        optimizer=optimizer,
        progress=progress,
        training_config=config,
        input_sha256=hashes,
    )
    restored = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(restored.parameters(), lr=5e-4)
    kwargs = dict(
        output_dir=tmp_path / "checkpoint",
        model=restored,
        optimizer=opt,
        expected_training_config=config,
        expected_input_sha256=hashes,
        expected_model_seed=27,
    )
    actual = load_jepa_admission_training_checkpoint(**kwargs, expected_level=level)
    assert actual == progress
    torch.testing.assert_close(restored.weight, model.weight)
    with pytest.raises(ValueError):
        load_jepa_admission_training_checkpoint(**kwargs, expected_level=3)


def test_l3_admission_training_config_locks_the_approved_budget() -> None:
    """Catches silently falling back to the 50-epoch cross-validation budget."""

    from latency_meta_mdp.belief.action_conditioned_jepa.admission_runner import (
        load_jepa_admission_training_config,
    )

    config = load_jepa_admission_training_config(
        Path("configs/training/action_conditioned_jepa/l3_admission.yaml")
    )

    assert config.config_id == "l3_stride4_admission_jepa_wm_recipe"
    assert config.max_epochs == 75
    assert config.logical_global_batch_size == 256
    assert config.milestone_epochs == (25, 50, 75)
    assert config.model_seeds == (7, 17, 27)
    assert config.learning_rate_reference == 5e-4
    assert config.weight_decay_start == 1e-7
    assert config.weight_decay_final == 1e-6


def test_l3_admission_preflight_uses_all_train_masters_and_keeps_validation_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches a final refit that still uses a 60-master fold or opens validation early."""

    import latency_meta_mdp.belief.action_conditioned_jepa.admission_runner as admission_runner

    def forbid_full_data_verification(**_kwargs):
        raise AssertionError("preflight must not scan source frames or feature payloads")

    monkeypatch.setattr(
        admission_runner,
        "load_verified_jepa_inputs",
        forbid_full_data_verification,
        raising=False,
    )

    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    output = tmp_path / "seed-17"
    preflight = admission_runner.build_jepa_admission_preflight(
        project_root=Path.cwd(),
        model_seed=17,
        microbatch_size=32,
        num_workers=4,
        device="cuda:1",
        output_dir=output,
    )

    assert preflight.stage_id == "l3-stride4-final-admission-v1"
    assert preflight.temporal_config_id == "stride4_80ms_history_160ms"
    assert preflight.model_seed == 17
    assert preflight.train_master_count == 80
    assert preflight.train_episode_count == 320
    assert preflight.formal_validation_master_count == 20
    assert preflight.formal_validation_episode_count == 80
    assert preflight.train_context_count == 45171
    assert preflight.optimizer_steps_per_epoch == 176
    assert preflight.max_epochs == 75
    assert preflight.total_optimizer_steps == 13200
    assert preflight.total_examples_seen == 3379200
    assert preflight.microbatch_size == 32
    assert preflight.gradient_accumulation_steps == 8
    assert preflight.formal_validation_opened is False
    assert preflight.wandb_api_key == "UNSET"
    assert not output.exists()


def test_l3_admission_preflight_rejects_an_unapproved_seed(tmp_path: Path) -> None:
    """Catches multiplying the final campaign with undeclared random seeds."""

    from latency_meta_mdp.belief.action_conditioned_jepa.admission_runner import (
        build_jepa_admission_preflight,
    )

    with pytest.raises(ValueError, match="model seed"):
        build_jepa_admission_preflight(
            project_root=Path.cwd(),
            model_seed=9,
            microbatch_size=32,
            num_workers=4,
            device="cuda:0",
            output_dir=tmp_path / "seed-9",
        )


def test_l3_admission_cli_preflight_is_read_only(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches launching final training before its complete identity is reviewable."""

    from latency_meta_mdp.cli.train_action_conditioned_jepa_admission import main

    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    output = tmp_path / "seed-7"
    status = main(
        [
            "--project-root",
            ".",
            "--model-seed",
            "7",
            "--microbatch-size",
            "32",
            "--num-workers",
            "4",
            "--device",
            "cuda:0",
            "--output-dir",
            str(output),
            "--preflight-only",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert status == 0
    assert payload["stage_id"] == "l3-stride4-final-admission-v1"
    assert payload["train_episode_count"] == 320
    assert payload["formal_validation_episode_count"] == 80
    assert payload["total_optimizer_steps"] == 13200
    assert payload["formal_validation_opened"] is False
    assert not output.exists()


def test_l3_admission_checkpoint_restores_without_a_fold_identity(tmp_path: Path) -> None:
    """Catches final resume losing optimizer state or inheriting cross-validation semantics."""

    import torch

    from latency_meta_mdp.belief.action_conditioned_jepa.admission_runner import (
        load_jepa_admission_training_checkpoint,
        load_jepa_admission_training_config,
        write_jepa_admission_training_checkpoint,
    )
    from latency_meta_mdp.belief.action_conditioned_jepa.training import (
        JepaAdmissionProgress,
        build_upstream_aligned_optimizer,
    )

    config = load_jepa_admission_training_config(
        Path("configs/training/action_conditioned_jepa/l3_admission.yaml")
    )
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4))
    optimizer = build_upstream_aligned_optimizer(model=model, config=config)
    model(torch.ones(2, 3)).square().mean().backward()
    optimizer.step()
    expected_parameters = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    progress = JepaAdmissionProgress(
        completed_epochs=25,
        optimizer_steps=4_400,
        examples_seen=1_126_400,
        model_seed=7,
        temporal_config_id="stride4_80ms_history_160ms",
        stage_id="l3-stride4-final-admission-v1",
        level=3,
    )
    hashes = {
        "source_manifest": "a" * 64,
        "cache_manifest": "b" * 64,
        "split_manifest": "c" * 64,
        "selection_manifest": "d" * 64,
        "model_config": "e" * 64,
        "temporal_config": "f" * 64,
        "training_config": "0" * 64,
    }
    output = tmp_path / "checkpoint"
    manifest = write_jepa_admission_training_checkpoint(
        output_dir=output,
        model=model,
        optimizer=optimizer,
        progress=progress,
        training_config=config,
        input_sha256=hashes,
    )
    restored_model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4))
    restored_optimizer = build_upstream_aligned_optimizer(
        model=restored_model,
        config=config,
    )
    restored = load_jepa_admission_training_checkpoint(
        output_dir=output,
        model=restored_model,
        optimizer=restored_optimizer,
        expected_training_config=config,
        expected_input_sha256=hashes,
        expected_model_seed=7,
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["format_id"] == "action_conditioned_jepa_l3_admission_checkpoint_v1"
    assert "fold_index" not in payload["progress"]
    assert restored == progress
    for name, value in restored_model.state_dict().items():
        torch.testing.assert_close(value, expected_parameters[name], atol=0.0, rtol=0.0)


def test_l3_admission_rolling_checkpoint_keeps_25_50_75(tmp_path: Path) -> None:
    """Catches deleting an approved milestone or retaining every final-refit epoch."""

    import torch

    from latency_meta_mdp.belief.action_conditioned_jepa.admission_runner import (
        load_jepa_admission_training_config,
        publish_rolling_jepa_admission_checkpoint,
    )
    from latency_meta_mdp.belief.action_conditioned_jepa.training import (
        JepaAdmissionProgress,
        build_upstream_aligned_optimizer,
    )

    config = load_jepa_admission_training_config(
        Path("configs/training/action_conditioned_jepa/l3_admission.yaml")
    )
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4))
    optimizer = build_upstream_aligned_optimizer(model=model, config=config)
    hashes = {
        "source_manifest": "a" * 64,
        "cache_manifest": "b" * 64,
        "split_manifest": "c" * 64,
        "selection_manifest": "d" * 64,
        "model_config": "e" * 64,
        "temporal_config": "f" * 64,
        "training_config": "0" * 64,
    }
    for epoch in (1, 2, 25, 26, 50, 51, 75):
        publish_rolling_jepa_admission_checkpoint(
            run_root=tmp_path / "run",
            model=model,
            optimizer=optimizer,
            progress=JepaAdmissionProgress(
                completed_epochs=epoch,
                optimizer_steps=epoch * 176,
                examples_seen=epoch * 176 * 256,
                model_seed=7,
                temporal_config_id="stride4_80ms_history_160ms",
                stage_id="l3-stride4-final-admission-v1",
                level=3,
            ),
            training_config=config,
            input_sha256=hashes,
        )

    assert sorted(path.name for path in (tmp_path / "run/checkpoints").iterdir()) == [
        "epoch-025",
        "epoch-050",
        "epoch-075",
    ]
    assert json.loads((tmp_path / "run/latest.json").read_text(encoding="utf-8")) == {
        "checkpoint": "checkpoints/epoch-075",
        "completed_epochs": 75,
    }


def test_formal_validation_gate_opens_only_at_the_fixed_epoch_75() -> None:
    """Catches reading formal-validation data before the final checkpoint is fixed."""

    from latency_meta_mdp.belief.action_conditioned_jepa.admission_runner import (
        load_jepa_admission_training_config,
        require_jepa_formal_validation_gate,
    )
    from latency_meta_mdp.belief.action_conditioned_jepa.training import (
        JepaAdmissionProgress,
    )

    config = load_jepa_admission_training_config(
        Path("configs/training/action_conditioned_jepa/l3_admission.yaml")
    )

    def progress(epoch: int) -> JepaAdmissionProgress:
        return JepaAdmissionProgress(
            completed_epochs=epoch,
            optimizer_steps=epoch * 176,
            examples_seen=epoch * 176 * 256,
            model_seed=27,
            temporal_config_id="stride4_80ms_history_160ms",
            stage_id="l3-stride4-final-admission-v1",
            level=3,
        )

    with pytest.raises(RuntimeError, match="epoch-75"):
        require_jepa_formal_validation_gate(progress(74), config=config)
    require_jepa_formal_validation_gate(progress(75), config=config)


def test_formal_validation_records_are_not_loaded_before_the_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches a runner that materializes held-out rows and only then checks epoch progress."""

    import latency_meta_mdp.belief.action_conditioned_jepa.admission_runner as admission_runner

    config = admission_runner.load_jepa_admission_training_config(
        Path("configs/training/action_conditioned_jepa/l3_admission.yaml")
    )
    calls = []

    def fake_loader(inputs, *, episode_id: str, level: int, split: str):
        calls.append((inputs, episode_id, level, split))
        return episode_id

    monkeypatch.setattr(admission_runner, "load_verified_jepa_record", fake_loader, raising=False)

    def progress(epoch: int):
        return admission_runner.JepaAdmissionProgress(
            completed_epochs=epoch,
            optimizer_steps=epoch * 176,
            examples_seen=epoch * 176 * 256,
            model_seed=7,
            temporal_config_id="stride4_80ms_history_160ms",
            stage_id="l3-stride4-final-admission-v1",
            level=3,
        )

    with pytest.raises(RuntimeError, match="epoch-75"):
        admission_runner.load_jepa_formal_validation_records(
            inputs="inputs",
            episode_ids=("validation-a", "validation-b"),
            progress=progress(74),
            config=config,
        )
    assert calls == []

    assert admission_runner.load_jepa_formal_validation_records(
        inputs="inputs",
        episode_ids=("validation-a", "validation-b"),
        progress=progress(75),
        config=config,
    ) == ("validation-a", "validation-b")
    assert calls == [
        ("inputs", "validation-a", 3, "validation"),
        ("inputs", "validation-b", 3, "validation"),
    ]


def test_l3_admission_cli_executes_the_final_runner(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches a non-preflight invocation that silently performs only another preflight."""

    import latency_meta_mdp.cli.train_action_conditioned_jepa_admission as cli

    output = tmp_path / "seed-27"
    expected = output / "manifest.json"
    calls = []

    def fake_execute(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(cli, "execute_jepa_admission_job", fake_execute, raising=False)
    assert (
        cli.main(
            [
                "--project-root",
                ".",
                "--model-seed",
                "27",
                "--microbatch-size",
                "32",
                "--num-workers",
                "4",
                "--device",
                "cuda:2",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )

    assert capsys.readouterr().out.strip() == str(expected)
    assert calls == [
        {
            "level": 3,
            "project_root": Path("."),
            "model_seed": 27,
            "microbatch_size": 32,
            "num_workers": 4,
            "device": "cuda:2",
            "output_dir": output,
            "resume": False,
            "qualification_max_epochs": None,
            "enable_wandb": True,
        }
    ]
