"""Generate frozen forecast inputs locally before conditioned policy training."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from latency_meta_mdp.data.forecast.assets import (
    download_forecast_models,
    forecast_bindings,
    load_forecast_job,
    stage_source,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", "--work-dir", dest="work_dir", type=Path, required=True)
    p.add_argument("--check-access", action="store_true")
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    root = Path.cwd()
    job = load_forecast_job(a.config, project_root=root)
    a.work_dir.mkdir(parents=True, exist_ok=True)
    predictor, decoder = download_forecast_models(job, a.work_dir)
    from huggingface_hub import snapshot_download

    from latency_meta_mdp.data.vision.contracts import load_vision_encoder_spec

    spec = load_vision_encoder_spec(
        root / "configs/models/vision/dinov3_vits16_lvd1689m_224_v1.yaml"
    )
    # Access is tested before a long SFT run; this also caches the small frozen encoder.
    snapshot_download(
        spec.model_id, revision=spec.revision, allow_patterns=["config.json", "model.safetensors"]
    )
    if a.check_access:
        print("Frozen model and DINO access verified", flush=True)
        return
    source = stage_source(job, a.work_dir / "source")
    from latency_meta_mdp.belief.jepa.corpus import (
        load_jepa_proprio_normalization,
    )

    norm = load_jepa_proprio_normalization(predictor / "proprio_normalization.json")
    if norm.level != job["clean"]["level"]:
        raise ValueError("Predictor and policy levels disagree")
    from latency_meta_mdp.io.artifacts import sha256_file

    if (
        sha256_file(job["split_manifest"]) != norm.split_manifest_sha256
        or sha256_file(source / "manifest.json") != norm.source_manifest_sha256
    ):
        raise ValueError("Predictor source/split identity mismatch")
    vision = a.work_dir / "vision"
    if not (vision / "manifest.json").exists():
        from latency_meta_mdp.data.source.loader import (
            load_verified_source_corpus,
        )
        from latency_meta_mdp.data.vision.dino import HfDinoPatchEncoder
        from latency_meta_mdp.data.vision.extract import write_vision_feature_cache_run

        encoder = HfDinoPatchEncoder.from_pretrained(spec=spec, device=a.device)
        write_vision_feature_cache_run(
            project_root=root,
            source_root=source,
            encoder=encoder,
            output_dir=vision,
            levels=(norm.level,),
            episode_range=None,
            selected_episode_ids=norm.episode_ids,
            boundary_batch_size=job["boundary_batch_size"],
            load_fn=lambda p: load_verified_source_corpus(p, verify_payloads=False),
            progress_fn=lambda s: print(s, flush=True),
        )
        del encoder
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
    decision = a.work_dir / "forecast-identity.json"
    decision.write_text(json.dumps(forecast_bindings(predictor, decoder), indent=2))
    output = a.work_dir / "forecasts"
    command = [
        sys.executable,
        "-u",
        "-m",
        "latency_meta_mdp.data.forecast.build",
        "--source-root",
        str(source),
        "--vision-cache-manifest",
        str(vision / "manifest.json"),
        "--split-manifest",
        str(job["split_manifest"]),
        "--predictor-dir",
        str(predictor),
        "--decoder-dir",
        str(decoder),
        "--decision",
        str(decision),
        "--output-dir",
        str(output),
        "--level",
        str(norm.level),
        "--batch-size",
        str(job["forecast_batch_size"]),
        "--device",
        a.device,
    ]
    if output.exists():
        command.append("--resume")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
