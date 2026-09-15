"""Historical five-anchor AR decoder evaluation on cached return mixtures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from latency_meta_mdp.belief.decoder.evaluation import RGBMetrics, decode


def evaluate_predicted_reconstruction(
    model,
    dataset,
    *,
    prediction_root: Path,
    data_manifest: Path,
    output: Path,
    batch_size: int = 16,
):
    from latency_meta_mdp.legacy.policy.policy_return_cache import NominalReturnPredictionCache

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
