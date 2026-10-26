"""Portable completed-episode archives, excluding worker and cache duplicates."""

import json
import shutil
import tarfile
import tempfile
from pathlib import Path

from latency_meta_mdp.meta.information.cohort import verify_complete
from latency_meta_mdp.meta.information.config import sha256


def pack_dataset(source, output, *, episodes_per_shard=16):
    source, output = Path(source).resolve(), Path(output).resolve()
    manifest = json.loads((source / "cohort.json").read_text())
    if manifest["binding"].get("scope") != "starter_training":
        raise ValueError("only immutable starter cohorts can be packaged")
    if episodes_per_shard < 1:
        raise ValueError("positive shard size required")
    if len({e["episode_id"] for e in manifest["episodes"]}) != len(manifest["episodes"]):
        raise ValueError("duplicate episode identity")
    attempts = []
    for episode in manifest["episodes"]:
        parent = source / "episodes" / episode["episode_id"]
        binding = dict(cohort=manifest["binding"], episode=episode)
        completed = [
            p
            for p in sorted(parent.glob("attempt-*"))
            if verify_complete(
                p, binding=binding, verification_cache=source / ".verification-cache"
            )
        ]
        if len(completed) != 1:
            raise ValueError(f"expected one completed attempt: {episode['episode_id']}")
        attempts.append(completed[0])
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source / "cohort.json", output / "cohort.json")
    index = dict(
        version=1, episodes=len(attempts), cohort_sha256=sha256(output / "cohort.json"), shards=[]
    )
    for start in range(0, len(attempts), episodes_per_shard):
        archive_path = output / f"episodes-{start // episodes_per_shard:04d}.tar"
        selected = attempts[start : start + episodes_per_shard]
        with tarfile.open(archive_path, "w", dereference=True) as archive:
            for attempt in selected:
                receipt = json.loads((attempt / "complete.json").read_text())
                for name in [*sorted(receipt["files"]), "complete.json"]:
                    path = attempt / name
                    archive.add(path, arcname=str(path.relative_to(source)), recursive=False)
        index["shards"].append(
            dict(
                name=archive_path.name,
                sha256=sha256(archive_path),
                bytes=archive_path.stat().st_size,
                episode_ids=[p.parent.name for p in selected],
            )
        )
        print(f"PACKED {archive_path.name} episodes={len(selected)}", flush=True)
    (output / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return index


def restore_dataset(bundle, destination):
    bundle, destination = Path(bundle).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("restore requires a new destination; existing cohorts stay unchanged")
    index = json.loads((bundle / "index.json").read_text())
    manifest = json.loads((bundle / "cohort.json").read_text())
    expected = {e["episode_id"] for e in manifest["episodes"]}
    received = [e for s in index["shards"] for e in s["episode_ids"]]
    if (
        index["version"] != 1
        or index["episodes"] != len(expected)
        or len(received) != len(expected)
        or set(received) != expected
        or sha256(bundle / "cohort.json") != index["cohort_sha256"]
    ):
        raise ValueError("dataset index or manifest differs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fqi-restore-", dir=destination.parent) as temp:
        staging = Path(temp) / "data"
        staging.mkdir()
        for shard in index["shards"]:
            path = bundle / shard["name"]
            if path.parent != bundle or sha256(path) != shard["sha256"]:
                raise ValueError("dataset shard hash differs")
            with tarfile.open(path) as archive:
                for member in archive:
                    target = staging / member.name
                    if (
                        not member.isfile()
                        or not target.resolve().is_relative_to(staging)
                        or not member.name.startswith("episodes/")
                    ):
                        raise ValueError("invalid dataset archive member")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("xb") as stream, archive.extractfile(member) as origin:
                        shutil.copyfileobj(origin, stream)
            print(f"RESTORED {shard['name']}", flush=True)
        shutil.copy2(bundle / "cohort.json", staging / "cohort.json")
        for episode in manifest["episodes"]:
            completed = list((staging / "episodes" / episode["episode_id"]).glob("*/complete.json"))
            if len(completed) != 1:
                raise ValueError("restored cohort is incomplete")
        staging.rename(destination)
