"""Cache frozen DINO spatial features from a structured source corpus."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--vision-config",
        type=Path,
        default=Path("configs/models/vision/dinov3_vits16_lvd1689m_224_v1.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument(
        "--episode-range",
        type=int,
        nargs=2,
        metavar=("START", "STOP"),
        default=None,
        help="Optional half-open episode-index range applied independently per level.",
    )
    parser.add_argument("--boundary-batch-size", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-download", action="store_true")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    encoder_factory: Callable[..., Any] | None = None,
    writer_fn: Callable[..., Path] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    from latency_meta_mdp.data.vision.contracts import load_vision_encoder_spec

    if encoder_factory is None:
        from latency_meta_mdp.data.vision.dino import HfDinoPatchEncoder

        encoder_factory = HfDinoPatchEncoder.from_pretrained
    if writer_fn is None:
        from latency_meta_mdp.data.vision.extract import (
            write_vision_feature_cache_run,
        )

        writer_fn = write_vision_feature_cache_run
    spec = load_vision_encoder_spec(args.vision_config)
    encoder = encoder_factory(
        spec=spec,
        device=args.device,
        local_files_only=not args.allow_download,
    )
    manifest = writer_fn(
        project_root=Path.cwd(),
        source_root=args.source_root,
        encoder=encoder,
        output_dir=args.output_dir,
        levels=tuple(args.levels),
        episode_range=None if args.episode_range is None else tuple(args.episode_range),
        boundary_batch_size=args.boundary_batch_size,
        progress_fn=lambda message: print(message, file=sys.stderr, flush=True),
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
