"""Cache frozen DINO spatial features for a selected expert-episode range."""

from __future__ import annotations

import argparse
from pathlib import Path

from latency_meta_mdp.hf_dino_encoder import HfDinoPatchEncoder
from latency_meta_mdp.vision_encoder import load_vision_encoder_spec
from latency_meta_mdp.vision_feature_cache_run import write_vision_feature_cache_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bulk-manifest", type=Path, required=True)
    parser.add_argument(
        "--vision-config",
        type=Path,
        default=Path("configs/vision/dinov3_vits16_lvd1689m_224_v1.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--boundary-batch-size", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-download", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    spec = load_vision_encoder_spec(args.vision_config)
    encoder = HfDinoPatchEncoder.from_pretrained(
        spec=spec,
        device=args.device,
        local_files_only=not args.allow_download,
    )
    manifest = write_vision_feature_cache_run(
        project_root=Path.cwd(),
        source_bulk_manifest=args.source_bulk_manifest,
        encoder=encoder,
        output_dir=args.output_dir,
        levels=tuple(args.levels),
        seed_start=args.seed_start,
        seed_count=args.seed_count,
        boundary_batch_size=args.boundary_batch_size,
    )
    print(manifest)


if __name__ == "__main__":
    main()
