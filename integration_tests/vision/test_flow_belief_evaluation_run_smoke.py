from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml


def test_flow_evaluation_run_binds_training_run_and_level_metrics(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Flow evaluation run smoke requires CUDA")
    source = Path("outputs/bulk/expert/panda-ball-first-tranche-e58eee5/manifest.json")
    cache = Path(
        "outputs/derived/vision_features/dinov3-vits16-first-tranche-54b9562/manifest.json"
    )
    if not source.is_file() or not cache.is_file():
        pytest.skip("Flow evaluation run smoke requires the local first tranche")
    from latency_meta_mdp.legacy.belief.flow.evaluation_run import evaluate_flow_belief_run
    from latency_meta_mdp.legacy.belief.flow.run import train_flow_belief_run

    config = yaml.safe_load(
        Path("configs/legacy/belief/dinov3_flow_belief_v1.yaml").read_text(encoding="utf-8")
    )
    config["max_epochs"] = 1
    config["early_stopping_patience"] = 1
    smoke_config = tmp_path / "smoke.yaml"
    smoke_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    training_manifest = train_flow_belief_run(
        project_root=Path.cwd(),
        source_bulk_manifest=source,
        cache_run_manifest=cache,
        vision_config_path=Path("configs/models/vision/dinov3_vits16_lvd1689m_224_v1.yaml"),
        temporal_config_path=Path("configs/contracts/temporal/h50_e25_d20_k6_v1.yaml"),
        latency_law_path=Path("configs/runtime/latency/truncated_beta_5_26_400ms_v1.yaml"),
        flow_config_path=smoke_config,
        output_dir=tmp_path / "training",
        levels=(1,),
        device="cuda",
    )

    manifest_path = evaluate_flow_belief_run(
        project_root=Path.cwd(),
        source_bulk_manifest=source,
        cache_run_manifest=cache,
        flow_run_manifest=training_manifest,
        vision_config_path=Path("configs/models/vision/dinov3_vits16_lvd1689m_224_v1.yaml"),
        temporal_config_path=Path("configs/contracts/temporal/h50_e25_d20_k6_v1.yaml"),
        latency_law_path=Path("configs/runtime/latency/truncated_beta_5_26_400ms_v1.yaml"),
        flow_config_path=smoke_config,
        output_dir=tmp_path / "evaluation",
        levels=(1,),
        device="cuda",
        context_limit=2,
        sample_count=4,
        step_count=2,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["levels"] == [1]
    assert manifest["level_manifests"] == {"L1": "L1/manifest.json"}
    assert manifest["flow_run_manifest_sha256"]
    assert (manifest_path.parent / "L1/metrics.json").is_file()
