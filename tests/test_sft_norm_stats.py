from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from latency_meta_mdp.io.artifacts import sha256_file
from latency_meta_mdp.policy.norm_stats import (
    NormStatsComputation,
    compute_level_sft_norm_stats,
)

_PROFILE = Path("configs/legacy/policy/pi05_panda_ball_full_sft_h50_v2.yaml")
_PATCH = Path("patches/openpi/0001-filter-incomplete-action-chunks.patch")
_REPO = "yypeng666/metamdp-robosuite-franka-moving_ball-l1-clean-50hz-h50-v2"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    derived_root = tmp_path / "derived"
    dataset_manifest = derived_root / _REPO / "metamdp_dataset.json"
    _write_json(
        dataset_manifest,
        {
            "schema_version": 1,
            "format_id": "metamdp_lerobot_v21",
            "repo_id": _REPO,
            "level": 1,
            "episode_count": 2,
            "frame_count": 102,
            "valid_action_chunk_source_count": 4,
            "artifacts": {"data/chunk.txt": "1" * 64},
        },
    )
    derived_manifest = derived_root / "manifest.json"
    _write_json(
        derived_manifest,
        {
            "schema_version": 1,
            "format_id": "metamdp_lerobot_formal_corpus_v1",
            "sft_profile_id": "pi05_panda_ball_full_sft_h50_v2",
            "sft_profile_sha256": sha256_file(_PROFILE),
            "openpi_revision": "15a9616a00943ada6c20a0f158e3adb39df2ccac",
            "openpi_patch_sha256": sha256_file(_PATCH),
            "episode_count": 2,
            "datasets": [
                {
                    "level": 1,
                    "repo_id": _REPO,
                    "episode_count": 2,
                    "frame_count": 102,
                    "valid_action_chunk_source_count": 4,
                    "dataset_manifest": f"{_REPO}/metamdp_dataset.json",
                    "dataset_manifest_sha256": sha256_file(dataset_manifest),
                }
            ],
        },
    )
    certification_manifest = tmp_path / "certification" / "manifest.json"
    _write_json(
        certification_manifest,
        {
            "schema_version": 1,
            "format_id": "metamdp_openpi_formal_certification_v1",
            "eligible": True,
            "implementation_dirty": False,
            "derived_manifest_sha256": sha256_file(derived_manifest),
            "sft_profile_id": "pi05_panda_ball_full_sft_h50_v2",
            "sft_profile_sha256": sha256_file(_PROFILE),
            "openpi_revision": "15a9616a00943ada6c20a0f158e3adb39df2ccac",
            "openpi_patch_sha256": sha256_file(_PATCH),
            "levels": [
                {
                    "level": 1,
                    "repo_id": _REPO,
                    "episode_count": 2,
                    "frame_count": 102,
                    "source_count": 4,
                    "norm_source_count": 4,
                    "no_action_padding": True,
                    "dataset_manifest": f"{_REPO}/metamdp_dataset.json",
                    "dataset_manifest_sha256": sha256_file(dataset_manifest),
                    "train_state_shape": [4, 32],
                    "train_action_shape": [4, 50, 32],
                }
            ],
        },
    )
    return derived_manifest, certification_manifest


def _norm_payload() -> dict:
    return {
        "norm_stats": {
            "state": {
                "mean": [0.0] * 8,
                "std": [1.0] * 8,
                "q01": [-1.0] * 8,
                "q99": [1.0] * 8,
            },
            "actions": {
                "mean": [0.0] * 7,
                "std": [1.0] * 7,
                "q01": [-1.0] * 7,
                "q99": [1.0] * 7,
            },
        }
    }


def test_compute_level_norm_stats_publishes_verified_atomic_artifact(tmp_path: Path) -> None:
    derived, certification = _inputs(tmp_path)
    calls = []

    def backend(**kwargs) -> NormStatsComputation:
        calls.append(kwargs)
        _write_json(kwargs["output_path"], _norm_payload())
        return NormStatsComputation(source_count=4)

    output = tmp_path / "norm" / "L1"
    manifest_path = compute_level_sft_norm_stats(
        project_root=Path.cwd(),
        derived_manifest=derived,
        certification_manifest=certification,
        profile_path=_PROFILE,
        patch_path=_PATCH,
        level=1,
        data_revision="a" * 40,
        output_dir=output,
        compute_backend=backend,
    )

    assert manifest_path == output / "manifest.json"
    assert len(calls) == 1
    assert calls[0]["level"] == 1
    assert calls[0]["repo_id"] == _REPO
    assert calls[0]["expected_source_count"] == 4
    assert calls[0]["output_path"].name == "norm_stats.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format_id"] == "metamdp_openpi_norm_stats_v1"
    assert manifest["level"] == 1
    assert manifest["repo_id"] == _REPO
    assert manifest["data_revision"] == "a" * 40
    assert manifest["source_count"] == 4
    assert manifest["state_dim"] == 8
    assert manifest["action_dim"] == 7
    assert manifest["artifacts"] == {"norm_stats.json": sha256_file(output / "norm_stats.json")}


def test_compute_level_norm_stats_rejects_wrong_backend_source_count(tmp_path: Path) -> None:
    derived, certification = _inputs(tmp_path)
    output = tmp_path / "norm" / "L1"

    def backend(**kwargs) -> NormStatsComputation:
        _write_json(kwargs["output_path"], _norm_payload())
        return NormStatsComputation(source_count=3)

    with pytest.raises(ValueError, match="source count"):
        compute_level_sft_norm_stats(
            project_root=Path.cwd(),
            derived_manifest=derived,
            certification_manifest=certification,
            profile_path=_PROFILE,
            patch_path=_PATCH,
            level=1,
            data_revision="a" * 40,
            output_dir=output,
            compute_backend=backend,
        )

    assert not output.exists()
    assert not list(output.parent.glob("*.building-*"))


def test_compute_level_norm_stats_rejects_nonfinite_stats(tmp_path: Path) -> None:
    derived, certification = _inputs(tmp_path)
    output = tmp_path / "norm" / "L1"

    def backend(**kwargs) -> NormStatsComputation:
        payload = _norm_payload()
        payload["norm_stats"]["actions"]["mean"][0] = float("nan")
        _write_json(kwargs["output_path"], payload)
        return NormStatsComputation(source_count=4)

    with pytest.raises(ValueError, match="finite"):
        compute_level_sft_norm_stats(
            project_root=Path.cwd(),
            derived_manifest=derived,
            certification_manifest=certification,
            profile_path=_PROFILE,
            patch_path=_PATCH,
            level=1,
            data_revision="a" * 40,
            output_dir=output,
            compute_backend=backend,
        )

    assert not output.exists()


def test_compute_level_norm_stats_rejects_unqualified_inputs(tmp_path: Path) -> None:
    derived, certification = _inputs(tmp_path)
    certification_payload = json.loads(certification.read_text(encoding="utf-8"))
    certification_payload["eligible"] = False
    _write_json(certification, certification_payload)

    with pytest.raises(ValueError, match="eligible clean"):
        compute_level_sft_norm_stats(
            project_root=Path.cwd(),
            derived_manifest=derived,
            certification_manifest=certification,
            profile_path=_PROFILE,
            patch_path=_PATCH,
            level=1,
            data_revision="a" * 40,
            output_dir=tmp_path / "norm" / "L1",
            compute_backend=lambda **_: NormStatsComputation(source_count=4),
        )


@pytest.mark.parametrize("revision", ["main", "a" * 39, "A" * 40])
def test_compute_level_norm_stats_requires_full_lowercase_data_revision(
    tmp_path: Path,
    revision: str,
) -> None:
    derived, certification = _inputs(tmp_path)

    with pytest.raises(ValueError, match="data revision"):
        compute_level_sft_norm_stats(
            project_root=Path.cwd(),
            derived_manifest=derived,
            certification_manifest=certification,
            profile_path=_PROFILE,
            patch_path=_PATCH,
            level=1,
            data_revision=revision,
            output_dir=tmp_path / "norm" / "L1",
            compute_backend=lambda **_: NormStatsComputation(source_count=4),
        )


def test_compute_norm_stats_cli_emits_clean_json_and_progress(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    derived, certification = _inputs(tmp_path)
    cli = importlib.import_module("latency_meta_mdp.data.tools.compute_sft_norm_stats")

    def backend(**kwargs) -> NormStatsComputation:
        _write_json(kwargs["output_path"], _norm_payload())
        return NormStatsComputation(source_count=4)

    output = tmp_path / "norm" / "L1"
    result = cli.main(
        [
            "--project-root",
            str(Path.cwd()),
            "--derived-manifest",
            str(derived),
            "--certification-manifest",
            str(certification),
            "--profile",
            str(_PROFILE),
            "--patch",
            str(_PATCH),
            "--level",
            "1",
            "--data-revision",
            "a" * 40,
            "--output-dir",
            str(output),
        ],
        compute_backend=backend,
    )

    captured = capsys.readouterr()
    assert result == 0
    assert json.loads(captured.out) == {"manifest": str(output / "manifest.json")}
    assert "[sft-norm][L1] start" in captured.err
    assert "[sft-norm][L1] done source_count=4" in captured.err
