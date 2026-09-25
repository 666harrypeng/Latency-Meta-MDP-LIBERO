"""Review a completed Direct run on real grouped validation endpoints."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset

from latency_meta_mdp.belief.jepa.backbone import ActionConditionedJepaPredictor
from latency_meta_mdp.belief.jepa.data import (
    collate_direct_samples,
)
from latency_meta_mdp.belief.jepa.job import (
    atomic_json,
    load_direct_data,
    load_direct_job,
)
from latency_meta_mdp.belief.jepa.metrics import (
    copy_current_prediction,
    predict_legacy_endpoint,
    prediction_mse_by_example,
)
from latency_meta_mdp.belief.jepa.model import (
    DirectJepaPredictor,
    load_direct_prediction_weights,
)
from latency_meta_mdp.io.artifacts import sha256_file


def load_legacy_reference(job, config, norm, direct, root):
    if job.task_id is not None:
        return None, None
    legacy = (
        ActionConditionedJepaPredictor(config=config, proprio_normalization=norm, project_root=root)
        .to(job.device)
        .eval()
    )
    legacy_path = job.normalization.parent / "checkpoints/epoch-075/model.safetensors"
    legacy.load_state_dict(load_file(str(legacy_path), device=job.device), strict=True)
    for key in ("proprio_mean", "proprio_scale"):
        if not torch.equal(getattr(legacy, key), getattr(direct.trunk, key)):
            raise ValueError("AR normalization mismatch")
    return legacy, legacy_path


def load_saved_metrics(output, job, source_stride):
    report = json.loads((output / "metrics.json").read_text())
    if (
        report.get("level") != job.level
        or report.get("task_id") != job.task_id
        or report.get("source_stride_ticks") != source_stride
        or [r["query_ticks"] for r in report.get("results", [])] != list(range(1, 21))
    ):
        raise ValueError("Saved metrics are incomplete or use a different evaluation protocol")
    return report


def finish_review(args, job, dataset, direct, legacy):
    if not args.metrics_only:
        from latency_meta_mdp.belief.jepa.review import review_direct_outputs

        review_direct_outputs(
            job=job,
            dataset=dataset,
            direct=direct,
            legacy=legacy,
            decoder_dir=args.decoder_dir,
            output=args.output_dir,
        )
    atomic_json(
        args.output_dir / "completion.json",
        {
            "status": "metrics_complete" if args.metrics_only else "review_ready",
            "level": job.level,
            "task_id": job.task_id,
            "runtime_measured": not args.metrics_only,
            "hf_uploaded": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--decoder-dir", type=Path)
    parser.add_argument(
        "--source-stride", type=int, help="validation source spacing in 20 ms ticks"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--metrics-only",
        action="store_true",
        help="Compute prediction errors without GPU timing or RGB review",
    )
    mode.add_argument(
        "--review-only",
        action="store_true",
        help="Reuse completed metrics for isolated GPU timing and optional RGB review",
    )
    args = parser.parse_args()
    if args.metrics_only and args.decoder_dir is not None:
        parser.error("metrics-only does not use an RGB decoder")
    root = Path.cwd()
    job = load_direct_job(args.config, project_root=root)
    source_stride = (
        args.source_stride if args.source_stride is not None else (10 if job.task_id else 1)
    )
    if source_stride < 1:
        parser.error("source stride must be positive")
    previous_integrity = None
    if args.review_only:
        load_saved_metrics(args.output_dir, job, source_stride)
        previous_integrity = json.loads((args.output_dir / "integrity.json").read_text())
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    config, norm, corpus, dataset = load_direct_data(job, split="validation")
    run = json.loads((args.run_dir / "run.json").read_text())
    completion = json.loads((args.run_dir / "completion.json").read_text())
    if (
        completion["completed_epochs"] != job.training.max_epochs
        or completion["identity"] != run["identity"]
    ):
        raise ValueError("Incomplete or inconsistent run")
    if sha256_file(args.run_dir / "proprio_normalization.json") != sha256_file(job.normalization):
        raise ValueError("Evaluation normalization changed")
    for name, value in [
        ("source", corpus.source_manifest_sha256),
        ("cache", corpus.cache_manifest_sha256),
        ("split", corpus.split_manifest_sha256),
    ]:
        if run["identity"][name + "_sha256"] != value:
            raise ValueError("Evaluation data identity changed")
    direct = (
        DirectJepaPredictor(backbone_config=config, proprio_normalization=norm, project_root=root)
        .to(job.device)
        .eval()
    )
    milestones = []
    for epoch in run["milestones"]:
        path = args.run_dir / f"checkpoints/epoch-{epoch:03d}"
        manifest = json.loads((path / "manifest.json").read_text())
        if (
            manifest["identity"] != run["identity"]
            or manifest["epoch"] != epoch
            or manifest["model_sha256"] != sha256_file(path / "model.safetensors")
        ):
            raise ValueError("Milestone identity mismatch")
        load_direct_prediction_weights(direct, path / "model.safetensors")
        milestones.append({"epoch": epoch, "sha256": manifest["model_sha256"]})
    latest = torch.load(args.run_dir / "latest.pt", map_location="cpu", weights_only=True)
    if latest[
        "optimizer_steps"
    ] != job.training.optimizer_steps_per_epoch * job.training.max_epochs or any(
        not torch.equal(v.cpu(), latest["model"][k]) for k, v in direct.state_dict().items()
    ):
        raise ValueError("Rolling checkpoint differs from final inference checkpoint")
    if {int(v["step"]) for v in latest["optimizer"]["state"].values()} != {
        latest["optimizer_steps"]
    }:
        raise ValueError("Optimizer step inventory mismatch")
    ledger = [json.loads(line) for line in (args.run_dir / "epochs.jsonl").read_text().splitlines()]
    if [r["completed_epochs"] for r in ledger] != list(
        range(1, job.training.max_epochs + 1)
    ) or any(
        r["result"]["seen_by_query"] != [0] + [job.training.examples_per_epoch // 20] * 20
        for r in ledger
    ):
        raise ValueError("Epoch exposure ledger incomplete")
    del latest
    legacy, legacy_path = load_legacy_reference(job, config, norm, direct, root)
    integrity = {
        "status": "verified",
        "level": job.level,
        "milestones": milestones,
        "optimizer_steps": completion["optimizer_steps"],
        "examples_seen": completion["examples_seen"],
        "epochs": len(ledger),
        "training_seconds": sum(r["result"]["seconds"] for r in ledger),
        "legacy_sha256": None if legacy_path is None else sha256_file(legacy_path),
    }
    if args.review_only:
        if integrity != previous_integrity:
            raise ValueError("Saved metrics checkpoint/data integrity differs from this run")
        finish_review(args, job, dataset, direct, legacy)
        return
    atomic_json(args.output_dir / "integrity.json", integrity)
    from latency_meta_mdp.belief.jepa.evaluation_data import horizon_indices

    metrics = ("visual", "qpos", "qvel", "gripper_width", "gripper_width_velocity")
    results = []
    with torch.inference_mode():
        for q in range(1, 21):
            indices = horizon_indices(dataset, q, source_stride=source_stride)
            loader = DataLoader(
                Subset(dataset, indices),
                batch_size=args.batch_size,
                num_workers=2,
                collate_fn=collate_direct_samples,
            )
            sums = {
                name: np.zeros(len(metrics))
                for name in ("direct", "copy_current")
                + (("ar",) if legacy is not None and q % 4 == 0 else ())
            }
            counts = 0
            started = time.monotonic()
            for sample in loader:
                query = sample.query.to(job.device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    predictions = {
                        "direct": direct.predict_at(query),
                        "copy_current": copy_current_prediction(
                            query,
                            proprio_mean=direct.trunk.proprio_mean,
                            proprio_scale=direct.trunk.proprio_scale,
                        ),
                    }
                    if legacy is not None and q % 4 == 0:
                        predictions["ar"] = predict_legacy_endpoint(legacy, query)
                for name, prediction in predictions.items():
                    mse = prediction_mse_by_example(prediction, sample)
                    sums[name] += np.array([mse[k].sum() for k in metrics])
                counts += len(sample.query.query_ticks)
            row = {
                "query_ticks": q,
                "milliseconds": 20 * q,
                "real_pairs": counts,
                "coordinate_rmse": {
                    name: {
                        key: float(np.sqrt(v / counts))
                        for key, v in zip(metrics, total, strict=True)
                    }
                    for name, total in sums.items()
                },
                "evaluation_seconds": time.monotonic() - started,
            }
            results.append(row)
            atomic_json(
                args.output_dir / "progress.json", {"completed_queries": q, "results": results}
            )
            print(json.dumps(row), flush=True)
    report = {
        "level": job.level,
        "split": "validation_development_not_independent_paper_test",
        "masters": len({r.logical_master_task_index for r in corpus.records}),
        "episodes": len(corpus.records),
        "sampling": (
            "per-episode fixed source stride plus last real endpoint; identical for all models"
        ),
        "source_stride_ticks": source_stride,
        "task_id": job.task_id,
        "results": results,
        "downstream_control_benefit_proven": False,
    }
    atomic_json(args.output_dir / "metrics.json", report)
    finish_review(args, job, dataset, direct, legacy)


if __name__ == "__main__":
    main()
