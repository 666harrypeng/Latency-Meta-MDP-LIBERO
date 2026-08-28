from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml


def test_fresh_decoder_probe_runs_three_isolated_decoders(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("fresh Decoder integration smoke requires CUDA")
    source = Path("outputs/bulk/expert/panda-ball-first-tranche-e58eee5/manifest.json")
    cache = Path(
        "outputs/derived/vision_features/dinov3-vits16-first-tranche-54b9562/manifest.json"
    )
    flow_run = Path("outputs/analysis/flow_belief/dinov3-first-tranche-e5e8745/manifest.json")
    if not source.is_file() or not cache.is_file() or not flow_run.is_file():
        pytest.skip("fresh Decoder integration smoke requires first-tranche artifacts")
    from latency_meta_mdp.belief.flow.fresh_decoder_run import (
        run_fresh_decoder_probe,
    )

    probe_config = tmp_path / "probe.yaml"
    probe_config.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "probe_id": "dinov3_flow_belief_fresh_decoder_probe_v1",
                "decoder_seeds": [101, 103, 107],
                "max_epochs": 1,
                "early_stopping_patience": 1,
                "rmse_ratio_max": 1.2,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    manifest_path = run_fresh_decoder_probe(
        project_root=Path.cwd(),
        source_bulk_manifest=source,
        cache_run_manifest=cache,
        flow_run_manifest=flow_run,
        vision_config_path=Path("configs/vision/dinov3_vits16_lvd1689m_224_v1.yaml"),
        temporal_config_path=Path("configs/temporal/h50_e25_d20_k6_v1.yaml"),
        latency_law_path=Path("configs/latency/truncated_beta_5_26_400ms_v1.yaml"),
        flow_config_path=Path("configs/belief/dinov3_flow_belief_v1.yaml"),
        probe_config_path=probe_config,
        output_dir=tmp_path / "probe-run",
        levels=(1,),
        device="cuda",
        training_context_limit=8,
        validation_context_limit=4,
        evaluation_sample_count=2,
        solver_step_count=2,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads((manifest_path.parent / "L1/summary.json").read_text(encoding="utf-8"))
    assert manifest["format_id"] == "flow_belief_fresh_decoder_probe_run_v1"
    assert manifest["levels"] == [1]
    assert manifest["decoder_seeds"] == [101, 103, 107]
    assert summary["decoder_seeds"] == [101, 103, 107]
    assert set(summary["per_seed_rmse_ratios"]) == {"101", "103", "107"}
    assert (manifest_path.parent / "L1/joint_decoder_metrics.json").is_file()
    for seed in (101, 103, 107):
        seed_dir = manifest_path.parent / f"L1/seed_{seed:06d}"
        assert (seed_dir / "vector_field.safetensors").is_file()
        assert (seed_dir / "evaluation_metrics.json").is_file()
