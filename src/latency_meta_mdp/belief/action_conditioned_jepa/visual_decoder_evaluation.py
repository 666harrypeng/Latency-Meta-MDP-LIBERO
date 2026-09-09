"""RGB quality and isolated decode timing; neither is a downstream policy admission."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file

from latency_meta_mdp.belief.action_conditioned_jepa.visual_decoder import (
    DualViewVisualDecoder,
    VisualDecoderConfig,
)


class RGBMetrics:
    def __init__(self):
        self.sums = np.zeros((2, 5), np.float64)

    def add(self, predicted, target):
        predicted = predicted.float().reshape(-1, 2, 3, 224, 224)
        target = target.float().reshape_as(predicted)
        error = predicted - target
        red = (
            (target[:, :, 0] > 0.45)
            & (target[:, :, 0] - target[:, :, 1] > 0.25)
            & (target[:, :, 0] - target[:, :, 2] > 0.25)
        )
        count = error.shape[0] * 3 * 224 * 224
        values = torch.stack(
            [
                error.abs().sum((0, 2, 3, 4)),
                error.square().sum((0, 2, 3, 4)),
                torch.full((2,), count, device=error.device),
                (error.abs() * red[:, :, None]).sum((0, 2, 3, 4)),
                red.sum((0, 2, 3)) * 3,
            ],
            dim=-1,
        )
        self.sums += values.double().cpu().numpy()

    def report(self):
        result = {}
        for name, row in zip(("agentview", "wrist"), self.sums, strict=True):
            if row[2] == 0:
                result[name] = None
                continue
            mse = float(row[1] / row[2])
            result[name] = {
                "mae": float(row[0] / row[2]),
                "rmse": mse**0.5,
                "psnr_db": float(-10 * np.log10(max(mse, 1e-12))),
                "red_pixel_proxy_mae": None if row[4] == 0 else float(row[3] / row[4]),
                "rgb_values": int(row[2]),
                "red_rgb_values": int(row[4]),
            }
        return result


def load_visual_decoder(run_root: Path, *, device: str):
    from latency_meta_mdp.artifacts import sha256_file

    receipt = json.loads((run_root / "checkpoint.json").read_text())
    if sha256_file(run_root / "model.safetensors") != receipt["model_sha256"]:
        raise ValueError("visual decoder checkpoint hash mismatch")
    config = json.loads((run_root / "run.json").read_text())["model"]
    model = DualViewVisualDecoder(VisualDecoderConfig(**config)).to(device)
    model.load_state_dict(
        load_file(str(run_root / "model.safetensors"), device=device), strict=True
    )
    model.eval()
    return model


def decode(model, latents):
    device = next(model.parameters()).device
    with (
        torch.inference_mode(),
        torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"),
    ):
        return model(latents.to(device, non_blocking=True))


def evaluate_reconstruction(model, dataset, *, batch_size: int = 64, workers: int = 2):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        shuffle=False,
        multiprocessing_context="spawn" if workers else None,
    )
    metric = RGBMetrics()
    device = next(model.parameters()).device
    for z, rgb in loader:
        target = rgb.to(device, non_blocking=True).float() / 255
        metric.add(decode(model, z), target)
    return metric.report()


def save_comparison(target, reconstruction, output: Path, *, caption: str):
    target = target.detach().float().cpu().numpy()
    reconstruction = reconstruction.detach().float().cpu().numpy()
    canvas = Image.new("RGB", (4 * 224, 252), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 5), caption, fill="black")
    for i, image in enumerate([target[0], reconstruction[0], target[1], reconstruction[1]]):
        image = np.clip(image.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
        canvas.paste(Image.fromarray(image), (i * 224, 28))
    canvas.save(output)


def benchmark_visual_decoder(model, *, repeats: int = 100, warmups: int = 20):
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise ValueError("latency qualification requires CUDA")
    results = {}
    for anchors in [1, 5]:
        cpu = torch.randn(1, anchors, 2, 196, 384, dtype=torch.float16)
        gpu = cpu.to(device)
        for transfer in [False, True]:
            times = []
            for i in range(warmups + repeats):
                torch.cuda.synchronize(device)
                started = time.perf_counter_ns()
                value = decode(model, cpu if transfer else gpu)
                if transfer:
                    value = value.mul(255).clamp(0, 255).to(torch.uint8).cpu()
                torch.cuda.synchronize(device)
                elapsed = (time.perf_counter_ns() - started) / 1e6
                if i >= warmups:
                    times.append(elapsed)
            results[f"{anchors}_anchors_" + ("cpu_to_rgb_cpu" if transfer else "gpu_resident")] = {
                "median_ms": float(np.median(times)),
                "p95_ms": float(np.percentile(times, 95)),
                "mean_ms": float(np.mean(times)),
                "repeats": repeats,
                "samples_ms": times,
            }
    return {
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "precision": "FP32 weights with BF16 autocast, float32 RGB",
        "timings": results,
        "scope": "isolated decoder, excludes JEPA/DINO/SigLIP/VLA/network/control concurrency",
    }


def evaluate_predicted_reconstruction(
    model,
    dataset,
    *,
    prediction_root: Path,
    data_manifest: Path,
    output: Path,
    batch_size: int = 16,
):
    from latency_meta_mdp.policy_return_cache import NominalReturnPredictionCache

    manifest = json.loads((prediction_root / "manifest.json").read_text())
    data = json.loads(data_manifest.read_text())
    decision = json.loads(
        (
            dataset.root / "outputs/analysis/action_conditioned_jepa/l3-admission-decision.json"
        ).read_text()
    )
    expected = {
        "checkpoint_sha256": decision["canonical_checkpoint_sha256"],
        "source_manifest_sha256": data["source_manifest_sha256"],
        "split_manifest_sha256": data["split_manifest_sha256"],
        "vision_cache_manifest_sha256": data["vision_manifest_sha256"],
    }
    if any(manifest["bindings"][k] != v for k, v in expected.items()):
        raise ValueError("prediction cache is not matched to the frozen decoder data/JEPA")
    predictions = NominalReturnPredictionCache(
        prediction_root, expected_bindings=manifest["bindings"]
    )
    metrics = {name: [RGBMetrics() for _ in range(5)] for name in ["true_dino", "jepa_predicted"]}
    device = next(model.parameters()).device
    source_count = 0
    pictured = set()
    output.mkdir(parents=True, exist_ok=True)
    for episode, row in enumerate(dataset.rows):
        features, rgb = dataset._episode(episode)
        terminal = row["boundary_count"] - 1
        anchors = np.array([4, 8, 12, 16, 20])
        phases = row.get("phases", [])
        funnel = next(
            (h for h, phase in enumerate(phases) if phase == "grasp_funnel" and h >= 10), 10
        )
        picture_ticks = {min(funnel, terminal - 20), terminal - 20}
        for start in range(10, terminal, batch_size):
            hs = np.arange(start, min(start + batch_size, terminal))
            future = hs[:, None] + anchors
            clipped = np.minimum(future, terminal)
            gt_z = torch.from_numpy(np.array(features[clipped], copy=True))
            pred_z = torch.from_numpy(
                np.stack([predictions.read(row["episode_id"], int(h))["visual"] for h in hs])
            )
            target = torch.from_numpy(np.array(rgb[clipped], copy=True)).to(device).float() / 255
            decoded_gt = decode(model, gt_z)
            decoded_pred = decode(model, pred_z)
            for k in range(5):
                valid = torch.as_tensor(future[:, k] <= terminal, device=device)
                metrics["true_dino"][k].add(decoded_gt[:, k][valid], target[:, k][valid])
                metrics["jepa_predicted"][k].add(decoded_pred[:, k][valid], target[:, k][valid])
            source_count += len(hs)
            master = row["master_index"]
            if len({m for m, _ in pictured}) < 2 or master in {m for m, _ in pictured}:
                for j, h in enumerate(hs):
                    if h not in picture_ticks or (master, int(h)) in pictured:
                        continue
                    for view in range(2):
                        canvas = Image.new("RGB", (5 * 224, 3 * 244 + 30), "white")
                        draw = ImageDraw.Draw(canvas)
                        draw.text(
                            (5, 5),
                            f"{row['episode_id']} h={h}; view={view}; heldout decoder master",
                            fill="black",
                        )
                        for line, (label, values) in enumerate(
                            [
                                ("GT future", target[j]),
                                ("Decode true DINO", decoded_gt[j]),
                                ("Decode JEPA", decoded_pred[j]),
                            ]
                        ):
                            for k in range(5):
                                draw.text(
                                    (k * 224 + 3, line * 244 + 32),
                                    f"{label}: {anchors[k] * 20}ms",
                                    fill="black",
                                )
                                image = (
                                    values[k, view]
                                    .detach()
                                    .float()
                                    .cpu()
                                    .numpy()
                                    .transpose(1, 2, 0)
                                )
                                canvas.paste(
                                    Image.fromarray(np.clip(image * 255, 0, 255).astype(np.uint8)),
                                    (k * 224, line * 244 + 50),
                                )
                        canvas.save(output / f"master-{master:03d}-h{h}-view{view}.png")
                    pictured.add((master, int(h)))
        print(f"Decoded predicted futures {episode + 1}/{len(dataset.rows)} episodes", flush=True)
    return {
        "sources": source_count,
        "anchor_ms": [80, 160, 240, 320, 400],
        "metrics": {name: [m.report() for m in values] for name, values in metrics.items()},
        "scope": (
            "same decoder; decoder-heldout train-pool masters; "
            "only real future targets, no absorbing extension; recorded nominal expert controls"
        ),
    }
