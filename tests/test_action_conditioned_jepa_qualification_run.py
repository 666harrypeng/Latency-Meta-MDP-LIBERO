from __future__ import annotations

import json
from pathlib import Path

import pytest


def _completed_run(root: Path, *, epoch: int = 75, formal_opened: bool = True) -> Path:
    checkpoint = root / "checkpoints/epoch-075"
    checkpoint.mkdir(parents=True)
    (checkpoint / "manifest.json").write_text("{}\n", encoding="utf-8")
    payload = {
        "format_id": "action_conditioned_jepa_l3_admission_run_v1",
        "qualification_only": False,
        "preflight": {
            "model_seed": 17,
            "temporal_config_id": "stride4_80ms_history_160ms",
        },
        "progress": {
            "completed_epochs": epoch,
            "model_seed": 17,
            "temporal_config_id": "stride4_80ms_history_160ms",
        },
        "formal_validation_opened": formal_opened,
        "final_formal_validation": {"native": {}, "deployed_d20": {}},
    }
    (root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


def test_final_qualification_requires_completed_epoch75_and_existing_formal_results(
    tmp_path: Path,
) -> None:
    """Catches evaluating an intermediate checkpoint as the final L3 Belief."""

    from latency_meta_mdp.belief.action_conditioned_jepa.qualification_run import (
        load_completed_l3_admission_run,
    )

    complete = load_completed_l3_admission_run(
        _completed_run(tmp_path / "complete"),
        expected_model_seed=17,
    )
    assert complete.checkpoint_dir.name == "epoch-075"
    assert set(complete.existing_formal_validation) == {"native", "deployed_d20"}

    with pytest.raises(ValueError, match="completed epoch-75"):
        load_completed_l3_admission_run(
            _completed_run(tmp_path / "incomplete", epoch=74, formal_opened=False),
            expected_model_seed=17,
        )
    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / "manifest.json").write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest is invalid"):
        load_completed_l3_admission_run(malformed, expected_model_seed=17)


def test_final_qualification_report_is_no_overwrite(tmp_path: Path) -> None:
    """Catches silently replacing the one-shot formal qualification result."""

    from latency_meta_mdp.belief.action_conditioned_jepa.qualification_run import (
        write_l3_qualification_report,
    )

    path = write_l3_qualification_report(
        output_path=tmp_path / "report.json",
        report={"format_id": "action_conditioned_jepa_l3_qualification_v1", "model_seed": 7},
    )
    assert json.loads(path.read_text(encoding="utf-8"))["model_seed"] == 7
    with pytest.raises(FileExistsError):
        write_l3_qualification_report(
            output_path=path,
            report={"format_id": "action_conditioned_jepa_l3_qualification_v1"},
        )


def test_final_qualification_cli_exposes_only_the_locked_stride4_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Catches reintroducing fold/config selection into the final qualification command."""

    import latency_meta_mdp.belief.action_conditioned_jepa.qualification_run as run_module
    from latency_meta_mdp.cli.qualify_action_conditioned_jepa_l3 import main

    output = tmp_path / "qualification.json"
    observed = {}

    def execute(**kwargs):
        observed.update(kwargs)
        output.write_text("{}\n", encoding="utf-8")
        return output

    monkeypatch.setattr(run_module, "execute_l3_final_qualification", execute)
    status = main(
        [
            "--project-root",
            ".",
            "--model-seed",
            "17",
            "--device",
            "cuda:4",
            "--batch-size",
            "8",
            "--output",
            str(output),
        ]
    )

    assert status == 0
    assert observed == {
        "project_root": Path("."),
        "model_seed": 17,
        "device": "cuda:4",
        "batch_size": 8,
        "output_path": output,
    }
    assert capsys.readouterr().out.strip() == str(output)
