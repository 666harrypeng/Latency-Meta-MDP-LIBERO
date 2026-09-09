from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from latency_meta_mdp.cli import certify_conditional_return_branches as certify_cli
from latency_meta_mdp.cli import collect_conditional_return_branches as collect_cli


def test_collect_cli_forwards_explicit_paths_and_per_level_bound(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    captured = {}
    manifest = tmp_path / "manifest.json"
    monkeypatch.setattr(
        collect_cli,
        "collect_control_branch_corpus",
        lambda **kwargs: captured.update(kwargs) or manifest,
    )

    result = collect_cli.main(
        [
            "--project-root",
            "/project",
            "--source-bulk-manifest",
            "source/manifest.json",
            "--split-config",
            "split.yaml",
            "--temporal-config",
            "temporal.yaml",
            "--branch-config",
            "branches.yaml",
            "--output-dir",
            "outputs/review",
            "--levels",
            "1",
            "2",
            "3",
            "--maximum-contexts-per-level",
            "1",
            "--scene-seed-range",
            "1000:1060",
            "--scene-seed-range",
            "1100:1120",
        ]
    )

    assert result == 0
    assert captured["levels"] == (1, 2, 3)
    assert captured["maximum_contexts_per_level"] == 1
    assert captured["allowed_scene_seed_ranges"] == ((1000, 1060), (1100, 1120))
    assert captured["project_root"] == Path("/project")
    assert capsys.readouterr().out.strip() == str(manifest)


def test_certify_cli_prints_machine_readable_summary(monkeypatch, capsys) -> None:
    loaded = SimpleNamespace(
        scientific_gate_pass=True,
        artifact_eligible=False,
        source_context_index=__import__("numpy").array([0, 0, 0]),
        branch_kind=__import__("numpy").array(["nominal", "hold", "arm_scale_0.5"]),
        target_absorbing=__import__("numpy").zeros((3, 20), dtype=bool),
        source_replay_max_abs=__import__("numpy").zeros(3),
        source_fingerprint_match=__import__("numpy").ones(3, dtype=bool),
        nominal_future_valid=__import__("numpy").array([True, False, False]),
        nominal_future_max_abs=__import__("numpy").zeros(3),
    )
    monkeypatch.setattr(
        certify_cli,
        "load_verified_control_branch_corpus",
        lambda _path: loaded,
    )

    result = certify_cli.main(["--manifest", "/artifact/manifest.json"])
    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert payload["scientific_gate_pass"] is True
    assert payload["artifact_eligible"] is False
    assert payload["context_count"] == 1
    assert payload["branch_count"] == 3
    assert payload["branch_kinds"] == ["arm_scale_0.5", "hold", "nominal"]
