"""Frozen conveyor forecasts from the existing clean bundle and terminal observations."""

import io
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from latency_meta_mdp.data.conveyor.forecast_source import (
    clean_forecast_episodes,
    load_terminal_assets,
)
from latency_meta_mdp.io.artifacts import sha256_file


def _write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def prepare_clean_features(
    inputs, terminal_root, output, *, encoder, boundary_batch_size=6, limit_episodes=None
):
    """Stream the exact clean RGB plus real terminal RGB through frozen DINO."""
    if boundary_batch_size < 1:
        raise ValueError("boundary batch size must be positive")
    export, _ = load_terminal_assets(inputs, terminal_root)
    selected = export["episodes"][:limit_episodes]
    identity = dict(
        policy_export_manifest_sha256=sha256_file(inputs.policy_export_manifest),
        terminal_manifest_sha256=sha256_file(Path(terminal_root) / "manifest.json"),
        encoder_fingerprint=encoder.spec.fingerprint,
        episode_ids=[r["episode_id"] for r in selected],
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    report = (
        json.loads(path.read_text())
        if path.exists()
        else dict(
            format_id="conveyor_clean_forecast_features_v1",
            identity=identity,
            source_manifest_sha256=export["source_manifest_sha256"],
            complete=False,
            episodes=[],
        )
    )
    if report["identity"] != identity:
        raise ValueError("clean feature-cache preparation identity changed")
    done = {e["episode_id"]: e for e in report["episodes"]}
    for i, (row, source) in enumerate(clean_forecast_episodes(inputs, terminal_root)):
        if i >= len(selected):
            break
        if row["episode_id"] in done:
            entry = done[row["episode_id"]]
            if (output / entry["features"]).stat().st_size != entry["file_size"]:
                raise ValueError("feature-cache payload size changed")
            continue
        target = output / f"seed-{row['seed']}.npy"
        temporary = target.with_suffix(".tmp.npy")
        n = len(source.states)
        features = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.float16, shape=(n, 2, 196, 384)
        )
        for start in range(0, n, boundary_batch_size):
            stop = min(n, start + boundary_batch_size)
            images = np.stack([source.rgb(h) for h in range(start, stop)])
            z = np.asarray(encoder.encode_numpy(images.reshape(-1, 256, 256, 3)))
            if z.shape != ((stop - start) * 2, 196, 384) or not np.isfinite(z).all():
                raise ValueError("invalid conveyor DINO features")
            features[start:stop] = z.reshape(stop - start, 2, 196, 384)
        features.flush()
        del features
        temporary.replace(target)
        report["episodes"].append(
            dict(
                episode_id=row["episode_id"],
                seed=row["seed"],
                features=target.name,
                file_size=target.stat().st_size,
                frame_count=n,
            )
        )
        _write(path, report)
        print(
            json.dumps({"features_completed": len(report["episodes"]), "planned": len(selected)}),
            flush=True,
        )
    report["complete"] = True
    _write(path, report)
    return path


def _verify_normalization(export, normalization):
    if (
        normalization.task_id != "conveyor_sort"
        or normalization.action_contract_id != export["action_contract"]
        or normalization.source_manifest_sha256 != export["source_manifest_sha256"]
        or normalization.split_manifest_sha256 != export["source_manifest_sha256"]
        or tuple(sorted(r["episode_id"] for r in export["episodes"])) != normalization.episode_ids
    ):
        raise ValueError("forecast source/normalization identity mismatch")


def load_clean_forecast_records(inputs, terminal_root, feature_manifest, normalization):
    from latency_meta_mdp.belief.jepa.corpus import JepaEpisodeRecord
    from latency_meta_mdp.data.vision.cache import EpisodeVisionFeatureCache

    export, _ = load_terminal_assets(inputs, terminal_root)
    _verify_normalization(export, normalization)
    path = Path(feature_manifest)
    report = json.loads(path.read_text())
    if (
        not report["complete"]
        or report["identity"]["policy_export_manifest_sha256"]
        != sha256_file(inputs.policy_export_manifest)
        or report["identity"]["terminal_manifest_sha256"]
        != sha256_file(Path(terminal_root) / "manifest.json")
    ):
        raise ValueError("incomplete or mismatched clean forecast features")
    entries = {e["episode_id"]: e for e in report["episodes"]}
    records = []
    for row, source in clean_forecast_episodes(inputs, terminal_root):
        if row["episode_id"] not in entries:
            continue  # A limited preparation is a smoke artifact, never a full train view.
        e = entries[row["episode_id"]]
        target = path.parent / e["features"]
        if target.stat().st_size != e["file_size"]:
            raise ValueError("clean forecast feature size mismatch")
        n = len(source.actions)
        records.append(
            JepaEpisodeRecord(
                episode_id=row["episode_id"],
                task_instance_id=row["group_id"],
                logical_master_task_index=row["seed"],
                level=None,
                split="train",
                terminal_tick=n,
                task_id="conveyor_sort",
                action_contract_id=export["action_contract"],
                cache=EpisodeVisionFeatureCache(
                    manifest=e, features=np.load(target, mmap_mode="r")
                ),
                proprio_physical=source.states,
                controls=source.actions,
                phases=(None,) * (n + 1),
                statuses=("running",) * n + ("success",),
            )
        )
    if len(records) != len(entries):
        raise ValueError("feature/source inventory mismatch")
    return tuple(records)


def prepare_conveyor_forecasts(job, work, *, predictor, decoder, device, limit_episodes=None):
    import torch

    from latency_meta_mdp.belief.decoder.evaluation import load_visual_decoder
    from latency_meta_mdp.belief.jepa.config import load_action_conditioned_jepa_config
    from latency_meta_mdp.belief.jepa.corpus import load_jepa_proprio_normalization
    from latency_meta_mdp.belief.jepa.forecast_provider import FrozenForecastEngine
    from latency_meta_mdp.belief.jepa.model import (
        DirectJepaPredictor,
        load_direct_prediction_weights,
    )
    from latency_meta_mdp.data.forecast.cache import write_forecast_cache
    from latency_meta_mdp.data.vision.dino import HfDinoPatchEncoder
    from latency_meta_mdp.io.paths import repository_root
    from latency_meta_mdp.policy.clean_data import require_training_source, resolve_clean_inputs

    root = repository_root()
    torch.set_num_threads(8)
    inputs = resolve_clean_inputs(job["clean"], work / "clean_data")
    require_training_source(inputs, checking=False)
    terminal = job.get("terminal_dir")
    if terminal is None:
        from huggingface_hub import snapshot_download

        terminal = Path(
            snapshot_download(
                job["terminal_repo"], repo_type="dataset", revision=job["terminal_revision"]
            )
        )
    norm_path = predictor / "proprio_normalization.json"
    norm = load_jepa_proprio_normalization(norm_path)
    export, _ = load_terminal_assets(inputs, terminal)
    _verify_normalization(export, norm)
    selected = export["episodes"][:limit_episodes]
    source_count = sum(row["frame_count"] for row in export["episodes"])
    selected_count = sum(row["frame_count"] for row in selected)
    if "forecast_cache_budget_gib" in job:
        cache_budget = float(job["forecast_cache_budget_gib"])
        if not np.isfinite(cache_budget) or cache_budget <= 0:
            raise ValueError("forecast cache space budget must be positive")
        # Check the declared task estimate before materializing a large feature cache.
        expected = cache_budget * 1024**3 * selected_count / source_count
        expected += sum((row["frame_count"] + 1) * 2 * 196 * 384 * 2 for row in selected)
        existing = sum(p.stat().st_size for p in (work / "vision").glob("*.npy"))
        database = work / "forecasts/forecasts.sqlite"
        existing += database.stat().st_size if database.exists() else 0
        if shutil.disk_usage(work).free < max(0, expected - existing) + 32 * 1024**3:
            raise OSError("insufficient space for declared forecast budget, DINO cache and reserve")
    config = load_action_conditioned_jepa_config(
        model_path=root / "configs/models/jepa/model.yaml",
        temporal_sampling_path=root / "configs/models/jepa/stride4_80ms_history_160ms.yaml",
        task_id="conveyor_sort",
        control_path=root / "configs/runtime/control/panda_osc_pose_delta_conveyor_v2.yaml",
    )
    encoder = HfDinoPatchEncoder.from_pretrained(spec=config.vision_encoder, device=device)
    features = prepare_clean_features(
        inputs,
        terminal,
        work / "vision",
        encoder=encoder,
        boundary_batch_size=job["boundary_batch_size"],
        limit_episodes=limit_episodes,
    )
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    records = load_clean_forecast_records(inputs, terminal, features, norm)
    model = DirectJepaPredictor(
        backbone_config=config, proprio_normalization=norm, project_root=root
    )
    weights = predictor / "checkpoints/epoch-075/model.safetensors"
    load_direct_prediction_weights(model, weights)
    engine = FrozenForecastEngine(model, load_visual_decoder(decoder, device=device), device=device)
    bindings = dict(
        predictor_architecture=model.architecture_id,
        predictor_sha256=sha256_file(weights),
        decoder_sha256=sha256_file(decoder / "model.safetensors"),
        jepa_normalization_sha256=sha256_file(norm_path),
        source_manifest_sha256=norm.source_manifest_sha256,
        split_manifest_sha256=norm.split_manifest_sha256,
        vision_cache_manifest_sha256=sha256_file(features),
    )
    output = work / "forecasts"
    image_codec = job.get("image_codec", "webp_lossless")
    # Calibrate space against real task predictions, not an uncompressed multi-TiB bound.
    from latency_meta_mdp.belief.jepa.data import materialize_direct_query

    sizes = []
    for index in np.linspace(0, len(records) - 1, min(20, len(records)), dtype=int):
        r = records[int(index)]
        for h in (10, (r.terminal_tick - 20) // 2, r.terminal_tick - 20):
            rgb, _ = engine.predict(
                materialize_direct_query(r, source_tick=h, query_ticks=20, normalization=norm)
            )
            stream = io.BytesIO()
            im = Image.fromarray(np.concatenate(rgb[0], axis=1))
            if image_codec == "webp_lossless":
                im.save(stream, format="WEBP", lossless=True, method=0)
            elif image_codec == "png":
                im.save(stream, format="PNG", compress_level=1)
            else:
                raise ValueError("unsupported forecast codec")
            sizes.append(len(stream.getvalue()))
    pairs = sum(max(0, r.terminal_tick - q - 9) for r in records for q in range(1, 21))
    estimated = int(pairs * np.mean(sizes) * 1.25)
    existing = (
        (output / "forecasts.sqlite").stat().st_size
        if (output / "forecasts.sqlite").exists()
        else 0
    )
    if shutil.disk_usage(work).free < max(0, estimated - existing) + 32 * 1024**3:
        raise OSError("insufficient free space for sampled forecast estimate plus 32 GiB reserve")
    _write(
        work / "storage-estimate.json",
        dict(
            prediction_pairs=pairs,
            estimated_cache_bytes=estimated,
            image_codec=image_codec,
            partial=limit_episodes is not None,
        ),
    )
    return write_forecast_cache(
        records=records,
        normalization=norm,
        engine=engine,
        output_dir=output,
        bindings=bindings,
        batch_size=job["forecast_batch_size"],
        resume=output.exists(),
        image_codec=image_codec,
    )
