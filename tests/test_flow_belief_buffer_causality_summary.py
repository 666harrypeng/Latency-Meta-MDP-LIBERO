import json

from latency_meta_mdp.io.artifacts import sha256_file
from latency_meta_mdp.legacy.belief.flow.buffer_causality_summary import (
    summarize_buffer_causality_levels,
)


def _evaluation(tmp_path, level: int, *, gain: float, false_coupling: float):
    root = tmp_path / f"L{level}"
    root.mkdir()
    summary = {
        "branches": {
            "expert": {
                "qpos_conditioning_gain_rad": 0.0,
                "qpos_conditioning_gain_latency_weighted_rad": 0.0,
                "invariant_object_false_coupling_m": 0.0,
                "invariant_object_false_coupling_latency_weighted_m": 0.0,
            },
            "hold": {
                "qpos_conditioning_gain_rad": gain,
                "qpos_conditioning_gain_latency_weighted_rad": gain / 2,
                "invariant_object_false_coupling_m": false_coupling,
                "invariant_object_false_coupling_latency_weighted_m": false_coupling / 2,
            },
        },
        "phases": {
            "pregrasp": {
                "expert": {"qpos_conditioning_gain_rad": 0.0},
                "hold": {
                    "qpos_conditioning_gain_rad": gain,
                    "qpos_conditioning_gain_latency_weighted_rad": gain / 2,
                },
            }
        },
    }
    (root / "summary.json").write_text(json.dumps(summary))
    manifest = {
        "format_id": "level_flow_belief_buffer_causality_evaluation_v1",
        "eligible": True,
        "blockers": [],
        "level": level,
        "branch_ids": ["expert", "hold"],
        "artifacts": {"summary.json": sha256_file(root / "summary.json")},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root / "manifest.json"


def test_summary_preserves_levels_and_lists_negative_regimes(tmp_path) -> None:
    manifests = tuple(
        _evaluation(
            tmp_path,
            level,
            gain={1: 0.1, 2: -0.2, 3: -0.3}[level],
            false_coupling={1: 0.01, 2: 0.02, 3: 0.03}[level],
        )
        for level in (1, 2, 3)
    )

    output = tmp_path / "combined"
    manifest_path = summarize_buffer_causality_levels(
        project_root=tmp_path,
        evaluation_manifests=manifests,
        output_dir=output,
        collect_provenance=False,
    )

    report = json.loads((output / "summary.json").read_text())
    assert tuple(report["levels"]) == ("L1", "L2", "L3")
    assert report["maximum_invariant_object_false_coupling_m"] == 0.03
    assert report["maximum_invariant_object_false_coupling_latency_weighted_m"] == 0.015
    assert report["negative_qpos_conditioning_gain_regimes"] == [
        {"branch": "hold", "gain_rad": -0.2, "level": 2, "phase": "pregrasp"},
        {"branch": "hold", "gain_rad": -0.3, "level": 3, "phase": "pregrasp"},
    ]
    assert report["negative_qpos_conditioning_gain_latency_weighted_regimes"] == [
        {"branch": "hold", "gain_rad": -0.1, "level": 2, "phase": "pregrasp"},
        {"branch": "hold", "gain_rad": -0.15, "level": 3, "phase": "pregrasp"},
    ]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["eligible"] is True
    assert manifest["levels"] == [1, 2, 3]
