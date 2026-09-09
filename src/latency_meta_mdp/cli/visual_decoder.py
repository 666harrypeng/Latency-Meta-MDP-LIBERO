"""Prepare, train and evaluate the independent frozen-latent RGB decoder."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        from latency_meta_mdp.belief.action_conditioned_jepa.visual_decoder_data import (
            prepare_visual_decoder_data,
        )

        manifest = prepare_visual_decoder_data(
            project_root=Path.cwd(), output=args.output_dir, seed=args.seed
        )
        print(f"Decoder targets ready: {manifest['boundary_count']} real boundaries")


if __name__ == "__main__":
    main()
