from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from latency_meta_mdp.artifacts import preflight_gate_output, publish_gate_report, sha256_file


def write_g0_prerequisite(root: Path) -> dict[str, str]:
    run_root = root / "certification/g0/g0_run"
    run_root.mkdir(parents=True)
    report_path = run_root / "runtime_report.json"
    report_path.write_text(json.dumps({"eligible": True}))
    manifest_path = run_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifacts": {"runtime_report.json": sha256_file(report_path)},
                "blockers": [],
                "eligible": True,
                "gate": "g0",
                "run_id": "g0_run",
                "schema_version": 2,
            }
        )
    )
    reference = {
        "manifest": "certification/g0/g0_run/manifest.json",
        "sha256": sha256_file(manifest_path),
    }
    (root / "artifact_index.json").write_text(
        json.dumps({"gates": {"g0": reference}, "schema_version": 1})
    )
    return reference


def test_publish_g1_preserves_g0_and_indexes_manifest_hash() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        g0_reference = write_g0_prerequisite(root)

        manifest_path = publish_gate_report(
            output_root=root,
            gate="g1",
            run_id="timing_run",
            report_name="timing_report.json",
            report={
                "blockers": [],
                "eligible": True,
                "prerequisites": {"g0": g0_reference},
                "schema_version": 1,
            },
        )

        index = json.loads((root / "artifact_index.json").read_text())
        assert index["gates"]["g0"] == g0_reference
        assert index["gates"]["g1"]["manifest"] == (
            "certification/g1/timing_run/manifest.json"
        )
        assert index["gates"]["g1"]["sha256"] == sha256_file(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        report_path = manifest_path.parent / "timing_report.json"
        assert manifest["artifacts"]["timing_report.json"] == sha256_file(report_path)


def test_existing_gate_is_rejected_before_new_run_directory() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "artifact_index.json").write_text(
            json.dumps(
                {
                    "gates": {
                        "g1": {"manifest": "existing/manifest.json", "sha256": "0" * 64}
                    },
                    "schema_version": 1,
                }
            )
        )

        with pytest.raises(FileExistsError, match="g1 artifact index entry already exists"):
            publish_gate_report(
                output_root=root,
                gate="g1",
                run_id="other",
                report_name="timing_report.json",
                report={"blockers": [], "eligible": True},
            )

        assert not (root / "certification").exists()


def test_preflight_rejects_existing_gate_without_creating_directories() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "artifact_index.json").write_text(
            json.dumps(
                {
                    "gates": {
                        "g1": {"manifest": "existing/manifest.json", "sha256": "0" * 64}
                    },
                    "schema_version": 1,
                }
            )
        )

        with pytest.raises(FileExistsError, match="g1 artifact index entry already exists"):
            preflight_gate_output(output_root=root, gate="g1", run_id="other")

        assert not (root / "certification").exists()


def test_report_write_failure_removes_partial_directory() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        g0_reference = write_g0_prerequisite(root)
        with (
            patch("latency_meta_mdp.artifacts._write_json", side_effect=OSError("disk failure")),
            pytest.raises(OSError, match="disk failure"),
        ):
            publish_gate_report(
                output_root=root,
                gate="g1",
                run_id="failed",
                report_name="timing_report.json",
                report={
                    "blockers": [],
                    "eligible": True,
                    "prerequisites": {"g0": g0_reference},
                },
            )

        assert list(root.rglob("*.building-*")) == []
        assert not (root / "certification/g1/failed").exists()


def test_g1_preflight_requires_hash_verified_eligible_g0() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with pytest.raises(ValueError, match="required prerequisite g0 is missing"):
            preflight_gate_output(output_root=root, gate="g1", run_id="timing")

        reference = write_g0_prerequisite(root)
        reference["sha256"] = "f" * 64
        (root / "artifact_index.json").write_text(
            json.dumps({"gates": {"g0": reference}, "schema_version": 1})
        )
        with pytest.raises(ValueError, match="g0 manifest hash mismatch"):
            preflight_gate_output(output_root=root, gate="g1", run_id="timing")
