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
    train = commands.add_parser("train")
    train.add_argument("--data-manifest", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--batch-per-device", type=int, default=64)
    train.add_argument("--workers", type=int, default=2)
    train.add_argument("--seed", type=int, default=27)
    train.add_argument("--limit-steps", type=int)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--wandb", action="store_true")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--data-manifest", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--prediction-root", type=Path)
    evaluate.add_argument("--device", default="cuda:0")
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--run-dir", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        from latency_meta_mdp.belief.decoder.data import (
            prepare_visual_decoder_data,
        )

        manifest = prepare_visual_decoder_data(
            project_root=Path.cwd(), output=args.output_dir, seed=args.seed
        )
        print(f"Decoder targets ready: {manifest['boundary_count']} real boundaries")

    elif args.command == "train":
        from latency_meta_mdp.belief.decoder.training import (
            train_visual_decoder,
        )

        train_visual_decoder(
            project_root=Path.cwd(),
            data_manifest=args.data_manifest,
            output=args.output_dir,
            epochs=args.epochs,
            batch_per_device=args.batch_per_device,
            workers=args.workers,
            seed=args.seed,
            limit_steps=args.limit_steps,
            resume=args.resume,
            wandb_enabled=args.wandb,
        )
    else:
        import json

        from latency_meta_mdp.belief.decoder.evaluation import (
            benchmark_visual_decoder,
            evaluate_reconstruction,
            load_visual_decoder,
        )
        from latency_meta_mdp.legacy.belief.decoder_evaluation import (
            evaluate_predicted_reconstruction,
        )

        model = load_visual_decoder(args.run_dir, device=args.device)
        if args.command == "benchmark":
            if args.output.exists():
                raise FileExistsError(args.output)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(benchmark_visual_decoder(model), indent=2) + "\n")
        else:
            from latency_meta_mdp.belief.decoder.data import (
                VisualDecoderDataset,
            )

            args.output_dir.mkdir(parents=True, exist_ok=False)
            dataset = VisualDecoderDataset(
                args.data_manifest, project_root=Path.cwd(), partition="holdout"
            )
            report = {
                "ground_truth_latent_reconstruction": evaluate_reconstruction(model, dataset),
                "runtime": benchmark_visual_decoder(model),
            }
            if args.prediction_root:
                report["predicted_latent_reconstruction"] = evaluate_predicted_reconstruction(
                    model,
                    dataset,
                    prediction_root=args.prediction_root,
                    data_manifest=args.data_manifest,
                    output=args.output_dir / "montages",
                )
            (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
