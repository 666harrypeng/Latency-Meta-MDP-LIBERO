"""Stream continuous conveyor RGB into frozen, memory-mapped DINO features."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from latency_meta_mdp.data.conveyor.corpus import split_sources
from latency_meta_mdp.data.vision.contracts import load_vision_encoder_spec
from latency_meta_mdp.io.artifacts import sha256_file
from latency_meta_mdp.io.paths import repository_root


def _write(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(path)


def verify_feature_cache(path, *, corpus_manifest, require_complete=True):
    path, corpus_manifest = Path(path).resolve(), Path(corpus_manifest).resolve()
    report = json.loads(path.read_text())
    corpus = json.loads(corpus_manifest.read_text())
    if (
        report["format_id"] != "conveyor_dino_features_v1"
        or report["source_manifest_sha256"] != sha256_file(corpus_manifest)
        or report["action_contract"] != corpus["action_contract"]
        or report["purpose"] != corpus["purpose"]
    ):
        raise ValueError("feature cache source identity mismatch")
    expected = {r["episode_id"]: r for r in corpus["episodes"] if r["split"] != "test"}
    seen = set()
    for row in report["episodes"]:
        source = expected.get(row["episode_id"])
        if source is None or row["episode_id"] in seen:
            raise ValueError("feature cache has an unexpected or duplicate episode")
        seen.add(row["episode_id"])
        if any(row[k] != source[k] for k in ("seed", "split", "group_id", "transition_count")):
            raise ValueError("feature cache episode identity mismatch")
        source_manifest = (
            corpus_manifest.parent / source["source_root"] / "manifest.json"
        ).resolve()
        if not source_manifest.is_relative_to(corpus_manifest.parent):
            raise ValueError("source path escapes corpus")
        if row["source_manifest_sha256"] != sha256_file(source_manifest):
            raise ValueError("feature cache episode source changed")
        file = (path.parent / row["features"]).resolve()
        if not file.is_relative_to(path.parent) or file.stat().st_size != row["file_size"]:
            raise ValueError("feature cache payload path/size mismatch")
        features = np.load(file, mmap_mode="r", allow_pickle=False)
        if (
            row["boundary_count"] != row["transition_count"] + 1
            or features.shape != (row["boundary_count"], 2, 196, 384)
            or features.dtype != np.float16
        ):
            raise ValueError("feature cache array contract mismatch")
    if require_complete and (report["status"] != "completed" or seen != set(expected)):
        raise ValueError("feature cache is incomplete")
    return report


def prepare_features(corpus_manifest, output_dir, *, encoder, boundary_batch_size=6, resume=False):
    if type(boundary_batch_size) is not int or boundary_batch_size < 1:
        raise ValueError("boundary batch size must be positive")
    corpus_manifest, output = Path(corpus_manifest).resolve(), Path(output_dir).resolve()
    corpus = json.loads(corpus_manifest.read_text())
    # Validate source admission and grouping before any expensive encoding.
    for split in ("train", "validation"):
        for _row, _source in split_sources(corpus_manifest, split=split):
            pass
    identity = dict(
        source_manifest_sha256=sha256_file(corpus_manifest),
        encoder_fingerprint=encoder.spec.fingerprint,
        preparation_sha256=sha256_file(Path(__file__)),
    )
    manifest = output / "manifest.json"
    if output.exists():
        if not resume:
            raise FileExistsError("feature output exists; use --resume")
        report = verify_feature_cache(
            manifest, corpus_manifest=corpus_manifest, require_complete=False
        )
        if report["identity"] != identity:
            raise ValueError("feature preparation identity changed")
    else:
        output.mkdir(parents=True)
        report = dict(
            format_id="conveyor_dino_features_v1",
            schema_version=1,
            task_id="conveyor_sort",
            variant="surface",
            purpose=corpus["purpose"],
            action_contract=corpus["action_contract"],
            source_manifest_sha256=identity["source_manifest_sha256"],
            encoder=asdict(encoder.spec),
            identity=identity,
            status="running",
            episodes=[],
        )
        _write(manifest, report)
    done = {r["episode_id"] for r in report["episodes"]}
    for split in ("train", "validation"):
        for row, source in split_sources(corpus_manifest, split=split):
            if row["episode_id"] in done:
                continue
            folder = output / "episodes" / f"seed-{row['seed']}"
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / "features.npy"
            temporary = folder / "features.tmp.npy"
            count = len(source.states)
            features = np.lib.format.open_memmap(
                temporary, mode="w+", dtype=np.float16, shape=(count, 2, 196, 384)
            )
            for start in range(0, count, boundary_batch_size):
                stop = min(count, start + boundary_batch_size)
                images = np.stack([source.rgb(t) for t in range(start, stop)])
                encoded = np.asarray(encoder.encode_numpy(images.reshape(-1, 256, 256, 3)))
                if (
                    encoded.shape != ((stop - start) * 2, 196, 384)
                    or not np.isfinite(encoded).all()
                ):
                    raise ValueError("invalid DINO output")
                features[start:stop] = encoded.reshape(stop - start, 2, 196, 384)
            features.flush()
            del features
            temporary.replace(target)
            report["episodes"].append(
                dict(
                    episode_id=row["episode_id"],
                    group_id=row["group_id"],
                    seed=row["seed"],
                    split=split,
                    transition_count=len(source.actions),
                    boundary_count=count,
                    source_manifest_sha256=sha256_file(source.root / "manifest.json"),
                    features=str(target.relative_to(output)),
                    file_size=target.stat().st_size,
                )
            )
            _write(manifest, report)
            print(
                json.dumps(
                    {
                        "cached_episodes": len(report["episodes"]),
                        "seed": row["seed"],
                        "split": split,
                        "boundaries": count,
                    }
                ),
                flush=True,
            )
    report["status"] = "completed"
    _write(manifest, report)
    verify_feature_cache(manifest, corpus_manifest=corpus_manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--vision-config",
        type=Path,
        default=repository_root() / "configs/models/vision/dinov3_vits16_lvd1689m_224_v1.yaml",
    )
    parser.add_argument("--boundary-batch-size", type=int, default=6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    from latency_meta_mdp.data.vision.dino import HfDinoPatchEncoder

    encoder = HfDinoPatchEncoder.from_pretrained(
        spec=load_vision_encoder_spec(args.vision_config),
        device=args.device,
        local_files_only=not args.allow_download,
    )
    print(
        prepare_features(
            args.corpus_manifest,
            args.output_dir,
            encoder=encoder,
            boundary_batch_size=args.boundary_batch_size,
            resume=args.resume,
        )
    )
    return 0
