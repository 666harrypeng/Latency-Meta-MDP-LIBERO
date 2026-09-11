"""Materialize frozen Direct/decoder inputs for the native RTC-conditioned VLA."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from latency_meta_mdp.artifacts import sha256_file
from latency_meta_mdp.belief.action_conditioned_jepa.config import (
    load_action_conditioned_jepa_config,
)
from latency_meta_mdp.belief.action_conditioned_jepa.data_adapter import (
    load_jepa_proprio_normalization,
    load_verified_jepa_inputs,
    load_verified_jepa_record,
)
from latency_meta_mdp.belief.action_conditioned_jepa.direct_prediction import (
    DirectJepaPredictor,
    load_direct_prediction_weights,
)
from latency_meta_mdp.belief.action_conditioned_jepa.forecast_provider import FrozenForecastEngine
from latency_meta_mdp.belief.action_conditioned_jepa.visual_decoder_evaluation import (
    load_visual_decoder,
)
from latency_meta_mdp.policy_forecast_cache import write_forecast_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--vision-cache-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--predictor-dir", type=Path, required=True)
    parser.add_argument("--decoder-dir", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--level", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit-episodes", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.limit_episodes is not None and args.limit_episodes < 1:
        raise ValueError("episode limit must be positive")
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parents[3]
    config = load_action_conditioned_jepa_config(
        model_path=root / "configs/belief/action_conditioned_jepa/model.yaml",
        level_path=root / f"configs/belief/action_conditioned_jepa/l{args.level}.yaml",
        temporal_sampling_path=root
        / "configs/belief/action_conditioned_jepa/stride4_80ms_history_160ms.yaml",
    )
    norm_path = args.predictor_dir / "proprio_normalization.json"
    weights = args.predictor_dir / "checkpoints/epoch-075/model.safetensors"
    decision = json.loads(args.decision.read_text())
    for path, key in (
        (weights, "direct_checkpoint_sha256"),
        (norm_path, "normalization_sha256"),
        (args.decoder_dir / "model.safetensors", "decoder_sha256"),
    ):
        if sha256_file(path) != decision.get(key):
            raise ValueError("forecast checkpoint differs from declared decision")
    norm = load_jepa_proprio_normalization(norm_path)
    print("Verifying existing source/cache/split; no new data collection or training", flush=True)
    inputs = load_verified_jepa_inputs(
        source_root=args.source_root,
        cache_run_manifest=args.vision_cache_manifest,
        split_manifest_path=args.split_manifest,
        config=config,
    )
    ids = sorted(
        set(inputs.split.train_episode_ids) & set(inputs.source.episode_ids(level=args.level))
    )
    if tuple(ids) != norm.episode_ids:
        raise ValueError("normalization does not bind the complete train inventory")
    if args.limit_episodes is not None:
        ids = ids[: args.limit_episodes]
    records = tuple(
        load_verified_jepa_record(inputs, episode_id=e, level=args.level, split="train")
        for e in ids
    )
    count = sum(max(0, r.terminal_tick - q - 9) for r in records for q in range(1, 21))
    # Conservative lossless PNG + SQLite bound; actual compressed usage is logged.
    parent = args.output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    required = count * (2 * 224 * 224 * 3 + 8192) + 2 * 1024**3
    if shutil.disk_usage(parent).free < required:
        raise OSError("insufficient disk for conservative forecast cache bound")
    model = DirectJepaPredictor(
        backbone_config=config, proprio_normalization=norm, project_root=root
    )
    load_direct_prediction_weights(model, weights)
    decoder = load_visual_decoder(args.decoder_dir, device=args.device)
    engine = FrozenForecastEngine(model, decoder, device=args.device)
    bindings = {
        "predictor_architecture": model.architecture_id,
        "predictor_sha256": sha256_file(weights),
        "decoder_sha256": sha256_file(args.decoder_dir / "model.safetensors"),
        "jepa_normalization_sha256": sha256_file(norm_path),
        "source_manifest_sha256": inputs.source_manifest_sha256,
        "split_manifest_sha256": inputs.split_manifest_sha256,
        "vision_cache_manifest_sha256": inputs.cache_manifest_sha256,
    }
    print(
        json.dumps(
            {
                "episodes": len(records),
                "forecast_pairs": count,
                "batch_size": args.batch_size,
                "partial_inventory": args.limit_episodes is not None,
                "optimizer_steps": 0,
            }
        ),
        flush=True,
    )
    manifest = write_forecast_cache(
        records=records,
        normalization=norm,
        engine=engine,
        output_dir=args.output_dir,
        bindings=bindings,
        batch_size=args.batch_size,
    )
    print(json.dumps({"completed_manifest": str(manifest), "optimizer_steps": 0}), flush=True)


if __name__ == "__main__":
    main()
