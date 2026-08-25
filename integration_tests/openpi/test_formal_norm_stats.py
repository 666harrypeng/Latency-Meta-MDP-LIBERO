from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from latency_meta_mdp.cli.compute_sft_norm_stats import main


def _git_status(root: Path) -> str:
    return subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def test_real_patched_openpi_norm_stats_cover_every_l1_pilot_source(
    tmp_path: Path,
) -> None:
    derived = Path(
        "outputs/pilots/lerobot/panda-ball-formal-streaming-smoke-0ac4f2b/manifest.json"
    )
    certification = Path(
        "outputs/certification/data/panda-ball-formal-streaming-smoke-0ac4f2b/manifest.json"
    )
    if not derived.is_file() or not certification.is_file():
        pytest.skip("real norm-stat integration requires the local certified formal pilot")
    openpi_root = Path("third_party/openpi").resolve()
    before = _git_status(openpi_root)
    output = tmp_path / "L1"

    assert (
        main(
            [
                "--project-root",
                str(Path.cwd()),
                "--derived-manifest",
                str(derived),
                "--certification-manifest",
                str(certification),
                "--profile",
                "configs/policy/pi05_panda_ball_full_sft_h50_v2.yaml",
                "--patch",
                "patches/openpi/0001-filter-incomplete-action-chunks.patch",
                "--openpi-root",
                str(openpi_root),
                "--level",
                "1",
                "--data-revision",
                "f" * 40,
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    stats = json.loads((output / "norm_stats.json").read_text(encoding="utf-8"))[
        "norm_stats"
    ]
    assert manifest["source_count"] == 1882
    assert len(stats["state"]["mean"]) == 8
    assert len(stats["actions"]["mean"]) == 7
    assert _git_status(openpi_root) == before
