"""Bounded runtime and RGB review of a completed Direct predictor."""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageDraw

from latency_meta_mdp.belief.decoder.evaluation import (
    decode,
    load_visual_decoder,
)
from latency_meta_mdp.belief.jepa.data import (
    materialize_direct_sample,
)
from latency_meta_mdp.belief.jepa.job import atomic_json
from latency_meta_mdp.belief.jepa.metrics import (
    predict_legacy_endpoint,
)


def review_direct_outputs(*, job, dataset, direct, legacy, decoder_dir: Path | None, output: Path):
    decoder = None if decoder_dir is None else load_visual_decoder(decoder_dir, device=job.device)
    record = next(r for r in dataset.records if r.terminal_tick >= 30)
    timings = []
    with torch.inference_mode():
        for q in range(1, 21):
            query = materialize_direct_sample(
                record, source_tick=10, query_ticks=q, normalization=dataset.normalization
            ).query.to(job.device)
            for name, model in [("direct", direct)] + ([("ar", legacy)] if q % 4 == 0 else []):
                predictor_ms = []
                total_ms = []
                for i in range(25):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        prediction = (
                            model.predict_at(query)
                            if name == "direct"
                            else predict_legacy_endpoint(model, query)
                        )
                    torch.cuda.synchronize()
                    middle = time.perf_counter()
                    if decoder is not None:
                        decode(decoder, prediction.visual_latents)
                    torch.cuda.synchronize()
                    end = time.perf_counter()
                    if i >= 5:
                        predictor_ms.append((middle - start) * 1000)
                        total_ms.append((end - start) * 1000)
                timings.append(
                    {
                        "model": name,
                        "milliseconds": q * 20,
                        "predictor_median_ms": float(np.median(predictor_ms)),
                        "predictor_p95_ms": float(np.percentile(predictor_ms, 95)),
                        "predictor_decoder_median_ms": None
                        if decoder is None
                        else float(np.median(total_ms)),
                    }
                )
    atomic_json(
        output / "runtime.json",
        {
            "scope": (
                "warm batch1; inputs already on GPU; endpoint prediction and optional RGB decoder; "
                "excludes DINO/history/CPU image conversion/transport/concurrency"
            ),
            "steady_samples_per_query": 20,
            "device": torch.cuda.get_device_name(job.device),
            "rows": timings,
        },
    )
    report = json.loads((output / "metrics.json").read_text())
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for model in ("direct", "ar", "copy_current"):
        rows = [r for r in report["results"] if model in r["coordinate_rmse"]]
        for ax, key, scale, label in (
            (axes[0, 0], "visual", 1, "Visual coordinate RMSE"),
            (axes[0, 1], "qpos", 1000, "Joint position coordinate RMSE (mrad)"),
        ):
            ax.plot(
                [r["milliseconds"] for r in rows],
                [r["coordinate_rmse"][model][key] * scale for r in rows],
                marker="o",
                label=model,
            )
            ax.set_ylabel(label)
    for model in ("direct", "ar"):
        rows = [r for r in timings if r["model"] == model]
        axes[1, 0].plot(
            [r["milliseconds"] for r in rows],
            [r["predictor_median_ms"] for r in rows],
            marker="o",
            label=model,
        )
        if decoder is not None:
            axes[1, 1].plot(
                [r["milliseconds"] for r in rows],
                [r["predictor_decoder_median_ms"] for r in rows],
                marker="o",
                label=model,
            )
    axes[1, 0].set_ylabel("Predictor median latency (ms)")
    axes[1, 1].set_ylabel("Predictor + decoder median latency (ms)")
    for ax in axes.flat:
        ax.set_xlabel("Forecast horizon (ms)")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle(f"L{job.level} Direct and AR: real validation endpoints / isolated GPU runtime")
    fig.tight_layout()
    fig.savefig(output / "accuracy-runtime.png", dpi=160)
    fig.savefig(output / "accuracy-runtime.pdf")
    plt.close(fig)
    if decoder is None:
        return
    metadata = {
        r["episode_id"]: r
        for r in pq.read_table(job.source_root / "meta/episodes.parquet").to_pylist()
    }
    chosen = []
    used = set()
    for r in dataset.records:
        if r.logical_master_task_index in used:
            continue
        candidates = [
            h
            for h in range(10, r.terminal_tick - 19)
            if r.phases[h] in ("approach", "grasp_funnel")
        ]
        if not candidates:
            continue
        # Select visible motion using GT feature change, before inspecting decoder errors.
        scores = [
            np.square(
                np.asarray(r.cache.features[h + 20, 0], dtype=np.float32)
                - np.asarray(r.cache.features[h, 0], dtype=np.float32)
            ).mean()
            for h in candidates
        ]
        h = candidates[int(np.argmax(scores))]
        chosen.append((r, h))
        used.add(r.logical_master_task_index)
        if len(chosen) == 4:
            break
    selections = []
    for r, h in chosen:
        meta = metadata[r.episode_id]
        rows = pq.read_table(
            job.source_root / meta["data_shard"],
            filters=[("episode_id", "=", r.episode_id)],
            columns=["formal_tick", "agentview_rgb", "wrist_rgb"],
        ).to_pylist()
        by_tick = {row["formal_tick"]: row for row in rows}
        qs = (1, 5, 10, 15, 20)
        canvas = Image.new("RGB", (len(qs) * 224, 6 * 248), "white")
        draw = ImageDraw.Draw(canvas)
        for col, q in enumerate(qs):
            sample = materialize_direct_sample(
                r, source_tick=h, query_ticks=q, normalization=dataset.normalization
            )
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = direct.predict_at(sample.query.to(job.device))
            pred = decode(decoder, prediction.visual_latents).cpu().numpy()[0]
            gt_decode = decode(decoder, sample.target_visual).cpu().numpy()[0]
            for camera, field in enumerate(("agentview_rgb", "wrist_rgb")):
                gt = (
                    Image.open(io.BytesIO(by_tick[h + q][field]["bytes"]))
                    .convert("RGB")
                    .resize((224, 224), Image.Resampling.BILINEAR)
                )
                images = [gt] + [
                    Image.fromarray(
                        (x[camera].transpose(1, 2, 0) * 255).round().clip(0, 255).astype("uint8")
                    )
                    for x in (gt_decode, pred)
                ]
                for kind, image in enumerate(images):
                    y = (camera * 3 + kind) * 248
                    canvas.paste(image, (col * 224, y + 24))
                    draw.text(
                        (col * 224 + 3, y + 3),
                        f"{'main' if camera == 0 else 'wrist'} "
                        f"{('GT', 'GT decode', 'Direct decode')[kind]} {20 * q}ms",
                        fill="black",
                    )
        name = f"{r.episode_id}-h{h}.png"
        canvas.save(output / name)
        selections.append(
            {
                "episode_id": r.episode_id,
                "master": r.logical_master_task_index,
                "source_tick": h,
                "phase": r.phases[h],
                "image": name,
            }
        )
    atomic_json(
        output / "visual-review.json",
        {
            "selection": (
                "first four distinct validation masters with approach/grasp; "
                "largest GT main-feature change within each selected episode"
            ),
            "cases": selections,
            "decoder": str(decoder_dir),
        },
    )
