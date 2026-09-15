"""Verify the direct-query candidate on real train-only data; never save smoke weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from latency_meta_mdp.belief.action_conditioned_jepa.config import (
    load_action_conditioned_jepa_config,
)
from latency_meta_mdp.belief.action_conditioned_jepa.data_adapter import (
    load_action_conditioned_jepa_corpus,
    load_jepa_proprio_normalization,
)
from latency_meta_mdp.belief.action_conditioned_jepa.direct_prediction import DirectJepaPredictor
from latency_meta_mdp.belief.action_conditioned_jepa.direct_prediction_data import (
    BalancedQuerySampler,
    DirectPredictionDataset,
    collate_direct_samples,
)
from latency_meta_mdp.belief.action_conditioned_jepa.direct_prediction_training import (
    DirectTrainingConfig,
    direct_prediction_loss,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, help="Level-aware Direct job; omit only for historical L3 checks"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--microbatch-size", type=int, default=16)
    parser.add_argument("--optimizer-steps", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    root = Path.cwd()
    if args.config is not None:
        from latency_meta_mdp.belief.action_conditioned_jepa.direct_prediction_run import (
            load_direct_data,
            load_direct_job,
        )

        job = load_direct_job(args.config, project_root=root)
        config_path = job.training_config
        training = job.training
        args.microbatch_size, args.num_workers, args.device = (
            job.microbatch_size,
            job.num_workers,
            job.device,
        )
        norm_path = job.normalization
        print(f"Preparing L{job.level} train records", flush=True)
        config, normalization, corpus, dataset = load_direct_data(job, split="train")
    else:
        config_path = root / "configs/training/action_conditioned_jepa/direct_query_l3_v1.yaml"
        training = DirectTrainingConfig(**yaml.safe_load(config_path.read_text()))
        count = args.optimizer_steps * training.global_batch_size
        if args.optimizer_steps <= 0 or count % 20:
            parser.error(
                "optimizer steps must give exactly balanced exposure across twenty queries"
            )
        if args.microbatch_size <= 0 or training.global_batch_size % args.microbatch_size:
            parser.error("microbatch must divide global batch size")
        if args.num_workers < 0:
            parser.error("num workers cannot be negative")
        args.output_dir.mkdir(parents=True, exist_ok=False)
        torch.manual_seed(training.seed)
        np.random.seed(training.seed)
        config = load_action_conditioned_jepa_config(
            model_path=root / "configs/belief/action_conditioned_jepa/model.yaml",
            level_path=root / "configs/belief/action_conditioned_jepa/l3.yaml",
            temporal_sampling_path=root
            / "configs/belief/action_conditioned_jepa/stride4_80ms_history_160ms.yaml",
        )
        norm_path = root / (
            "outputs/training/action_conditioned_jepa/l3-final-admission/"
            "stride4_80ms_history_160ms/seed-27/proprio_normalization.json"
        )
        normalization = load_jepa_proprio_normalization(norm_path)
        print("Verifying shared source/cache and loading train episodes only", flush=True)
        corpus = load_action_conditioned_jepa_corpus(
            source_root=root
            / "outputs/source_corpus/panda-ball-structured-source-quota-formal-100x4-v1",
            cache_run_manifest=root
            / "outputs/derived/vision_features"
            / "dinov3-vits16-structured-source-100x4-v1/manifest.json",
            split_manifest_path=root
            / (
                "outputs/derived/source_splits/"
                "panda-ball-structured-source-quota-formal-100x4-v1/"
                "train80-validation20-seed20260903-v1.json"
            ),
            level=3,
            split="train",
            config=config,
            normalization=normalization,
        )
        dataset = DirectPredictionDataset(records=corpus.records, normalization=normalization)

    count = args.optimizer_steps * training.global_batch_size
    if args.optimizer_steps <= 0 or count % 20:
        parser.error("optimizer steps must give balanced query exposure")
    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True, exist_ok=False)
    elif args.config is not None:
        raise FileExistsError(args.output_dir)
    torch.manual_seed(training.seed)
    np.random.seed(training.seed)
    sampler = BalancedQuerySampler(dataset, sample_count=count, seed=training.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.microbatch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_direct_samples,
    )
    device = torch.device(args.device)
    model = DirectJepaPredictor(
        backbone_config=config, proprio_normalization=normalization, project_root=root
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=training.weight_decay_start,
    )
    print(
        f"real_pairs={len(dataset)} microbatch={args.microbatch_size} "
        f"parameters={model.parameter_count}",
        flush=True,
    )
    accumulation = training.global_batch_size // args.microbatch_size
    optimizer.zero_grad(set_to_none=True)
    updates, step_metrics, gradients = [], [], {}
    histogram = np.zeros(21, dtype=np.int64)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    update_started = started
    last_query = None
    for microstep, sample in enumerate(loader):
        histogram += np.bincount(sample.query.query_ticks.numpy(), minlength=21)
        query = sample.query.to(device)
        query.validate_finite()
        last_query = query
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            prediction = model(query)
            loss, metrics = direct_prediction_loss(
                prediction,
                sample.target_visual.to(device),
                sample.target_proprio.to(device),
                proprio_mean=model.trunk.proprio_mean,
                proprio_scale=model.trunk.proprio_scale,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite direct endpoint loss")
        (loss / accumulation).backward()
        step_metrics.append({key: value.item() for key, value in metrics.items()})
        if (microstep + 1) % accumulation == 0:
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), training.gradient_clip_norm, error_if_nonfinite=True
            )
            if not gradients:
                gradients = {
                    name: float(parameter.grad.norm())
                    for name, parameter in model.named_parameters()
                    if parameter.grad is not None
                }
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            now = time.perf_counter()
            row = {
                "optimizer_step": len(updates) + 1,
                "seconds": now - update_started,
                "gradient_norm_before_clip": float(norm),
                **{key: float(np.mean([r[key] for r in step_metrics])) for key in step_metrics[0]},
            }
            updates.append(row)
            print(json.dumps(row), flush=True)
            step_metrics.clear()
            update_started = now
    train_seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    model.eval()
    # Batch-1 full consumer output, equal q workload. Excludes frozen DINO and decoder.
    from latency_meta_mdp.belief.action_conditioned_jepa.contracts import ForecastQuery

    base = {name: value[:1] for name, value in vars(last_query).items()}
    timings = []
    for repetition in range(4):
        for q in range(1, 21):
            single = ForecastQuery(
                **{
                    **base,
                    "query_ticks": torch.tensor([q], device=device),
                    "control_mask": torch.arange(20, device=device)[None] < q,
                }
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            tick = time.perf_counter()
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                result = model.predict_at(single)
            if (
                not torch.isfinite(result.visual_latents).all()
                or not torch.isfinite(result.proprio).all()
            ):
                raise FloatingPointError("nonfinite inference output")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if repetition:
                timings.append((time.perf_counter() - tick) * 1000)
    source_files = [
        Path(__file__),
        config_path,
        *sorted(
            (root / "src/latency_meta_mdp/belief/action_conditioned_jepa").glob(
                "direct_prediction*.py"
            )
        ),
        root / "src/latency_meta_mdp/belief/action_conditioned_jepa/contracts.py",
    ]
    report = {
        "status": "preflight_passed",
        "level": config.level,
        "scientific_admission": False,
        "initialization": "scratch",
        "weights_saved": False,
        "validation_loaded": False,
        "training_config": asdict(training),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).splitlines(),
        "source_code_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_files
        },
        "source_manifest_sha256": corpus.source_manifest_sha256,
        "cache_manifest_sha256": corpus.cache_manifest_sha256,
        "split_manifest_sha256": corpus.split_manifest_sha256,
        "normalization_sha256": hashlib.sha256(norm_path.read_bytes()).hexdigest(),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "microbatch_size": args.microbatch_size,
        "accumulation_steps": accumulation,
        "train_episode_count": len(dataset.records),
        "unique_pair_count": len(dataset),
        "pairs_by_query": dataset.horizon_counts,
        "examples_seen": int(histogram.sum()),
        "seen_by_query": histogram.tolist(),
        "optimizer_steps": len(updates),
        "parameter_count": model.parameter_count,
        "train_seconds": train_seconds,
        "peak_training_allocated_bytes": peak,
        "first_update_gradient_norms": gradients,
        "updates": updates,
        "batch1_predictor_only_ms": {
            "median": float(np.median(timings)),
            "p95": float(np.percentile(timings, 95)),
            "samples": len(timings),
            "includes_dino_decoder_transport": False,
        },
    }
    with (args.output_dir / "preflight.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print("Preflight complete; no checkpoint saved and no downstream quality claim", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
