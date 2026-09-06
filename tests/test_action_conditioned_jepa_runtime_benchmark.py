from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_runtime_summary_uses_measured_end_to_end_samples_not_component_p95_sum() -> None:
    """Catches reporting a synthetic sum instead of the measured full-path latency."""

    from latency_meta_mdp.belief.action_conditioned_jepa.runtime_benchmark import (
        summarize_runtime_timings,
    )

    result = summarize_runtime_timings(
        {
            "dual_camera_dino": [1.0, 2.0, 3.0, 4.0],
            "history_and_context": [0.1, 0.2, 0.3, 0.4],
            "ar5_rollout": [10.0, 20.0, 30.0, 40.0],
            "pmf_assembly": [0.5, 0.6, 0.7, 0.8],
            "full_belief": [12.0, 22.0, 32.0, 42.0],
        }
    )

    assert result["sample_count"] == 4
    assert result["components"]["full_belief"]["p95_ms"] == pytest.approx(
        np.percentile([12.0, 22.0, 32.0, 42.0], 95)
    )
    component_sum = sum(
        result["components"][name]["p95_ms"]
        for name in (
            "dual_camera_dino",
            "history_and_context",
            "ar5_rollout",
            "pmf_assembly",
        )
    )
    assert result["components"]["full_belief"]["p95_ms"] != component_sum


def test_runtime_summary_rejects_misaligned_or_nonfinite_samples() -> None:
    """Catches percentile reports formed from different launch inventories."""

    from latency_meta_mdp.belief.action_conditioned_jepa.runtime_benchmark import (
        summarize_runtime_timings,
    )

    with pytest.raises(ValueError, match="aligned"):
        summarize_runtime_timings(
            {
                "dual_camera_dino": [1.0],
                "history_and_context": [1.0],
                "ar5_rollout": [1.0],
                "pmf_assembly": [1.0],
                "full_belief": [1.0, 2.0],
            }
        )
    with pytest.raises(ValueError, match="finite"):
        summarize_runtime_timings(
            {
                name: [float("nan")]
                for name in (
                    "dual_camera_dino",
                    "history_and_context",
                    "ar5_rollout",
                    "pmf_assembly",
                    "full_belief",
                )
            }
        )


def test_runtime_benchmark_cli_runs_the_locked_three_seed_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Catches adding arbitrary seed/config selection to the final J6a command."""

    import latency_meta_mdp.belief.action_conditioned_jepa.runtime_benchmark as module
    from latency_meta_mdp.cli.benchmark_action_conditioned_jepa_l3_runtime import main

    output = tmp_path / "runtime.json"
    observed = {}

    def execute(**kwargs):
        observed.update(kwargs)
        output.write_text("{}\n", encoding="utf-8")
        return output

    monkeypatch.setattr(module, "execute_l3_runtime_benchmark", execute)
    status = main(
        [
            "--project-root",
            ".",
            "--device",
            "cuda:0",
            "--output",
            str(output),
        ]
    )

    assert status == 0
    assert observed == {
        "project_root": Path("."),
        "device": "cuda:0",
        "output_path": output,
    }
    assert capsys.readouterr().out.strip() == str(output)
