from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml


def test_gaussian_belief_run_keeps_level_checkpoint_isolation(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Gaussian belief run smoke requires CUDA")
    source = Path(
        "outputs/bulk/expert/panda-ball-first-tranche-e58eee5/manifest.json"
    )
    cache = Path(
        "outputs/derived/vision_features/"
        "dinov3-vits16-first-tranche-54b9562/manifest.json"
    )
    if not source.is_file() or not cache.is_file():
        pytest.skip("Gaussian belief run smoke requires the local first tranche")
    from latency_meta_mdp.gaussian_belief_run import train_gaussian_belief_run

    config = yaml.safe_load(
        Path("configs/belief/dinov3_gaussian_belief_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    config["max_epochs"] = 1
    config["early_stopping_patience"] = 1
    smoke_config = tmp_path / "smoke.yaml"
    smoke_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    manifest_path = train_gaussian_belief_run(
        project_root=Path.cwd(),
        source_bulk_manifest=source,
        cache_run_manifest=cache,
        vision_config_path=Path(
            "configs/vision/dinov3_vits16_lvd1689m_224_v1.yaml"
        ),
        temporal_config_path=Path("configs/temporal/h50_e25_d20_k6_v1.yaml"),
        latency_law_path=Path("configs/latency/truncated_beta_5_26_400ms_v1.yaml"),
        belief_config_path=smoke_config,
        output_dir=tmp_path / "run",
        levels=(1,),
        device="cuda",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["checkpoint_scope"] == "per_level_only"
    assert manifest["levels"] == [1]
    assert manifest["level_manifests"] == {"L1": "L1/manifest.json"}
    assert (manifest_path.parent / "L1/encoder.safetensors").is_file()
    assert "L2" not in manifest["level_manifests"]
    assert "L3" not in manifest["level_manifests"]
