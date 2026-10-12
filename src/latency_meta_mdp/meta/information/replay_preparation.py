"""Verified lossless uncompressed replay copies, reusable across learner launches."""

import copy
import errno
import json
import shutil
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from latency_meta_mdp.meta.information.config import sha256, validate_binding


def prepare_episode(source, cache_root, *, expected_receipt=None, binding=None):
    source, cache_root = Path(source), Path(cache_root)
    source_hash = sha256(source / "complete.json")
    if expected_receipt is not None:
        validate_binding(source_hash, expected_receipt)
    receipt = json.loads((source / "complete.json").read_text())
    if binding is not None:
        validate_binding(receipt["binding"], binding)
    if set(receipt["files"]) != {"visual.npz", "records.json", "audit.json"}:
        raise ValueError("invalid source replay manifest")
    for name, value in receipt["files"].items():
        if sha256(source / name) != value:
            raise ValueError("source replay hash differs")
    cache_root.mkdir(parents=True, exist_ok=True)
    target = cache_root / source_hash
    if target.exists():
        cached = json.loads((target / "complete.json").read_text())
        validate_binding(cached["source_receipt_sha256"], source_hash)
        validate_binding(cached["binding"], receipt["binding"])
        if set(cached["files"]) != set(receipt["files"]):
            raise ValueError("cache file manifest differs")
        for name, value in cached["files"].items():
            if sha256(target / name) != value:
                raise ValueError("prepared cache hash differs")
        return target
    pending = Path(tempfile.mkdtemp(prefix=".pending-", dir=cache_root))
    try:
        with zipfile.ZipFile(source / "visual.npz") as original:
            with zipfile.ZipFile(
                pending / "visual.npz", "w", compression=zipfile.ZIP_STORED
            ) as out:
                for name in original.namelist():
                    if not name.startswith("visual_") or not name.endswith(".npy") or "/" in name:
                        raise ValueError("invalid visual array entry")
                    with original.open(name) as inp, out.open(name, "w", force_zip64=True) as dst:
                        shutil.copyfileobj(inp, dst, length=2**20)
        for name in ("records.json", "audit.json"):
            shutil.copy2(source / name, pending / name)
            if sha256(pending / name) != receipt["files"][name]:
                raise ValueError("source replay changed during preparation")
        cached = dict(
            receipt,
            source_receipt_sha256=source_hash,
            files={name: sha256(pending / name) for name in receipt["files"]},
        )
        (pending / "complete.json").write_text(json.dumps(cached, indent=2) + "\n")
        # Reject a source that changed during preparation, before publication.
        if sha256(source / "complete.json") != source_hash:
            raise ValueError("source receipt changed during preparation")
        for name, value in receipt["files"].items():
            if sha256(source / name) != value:
                raise ValueError("source replay changed during preparation")
        try:
            pending.rename(target)
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            # Another run published the same source while this copy was built.
            return prepare_episode(
                source, cache_root, expected_receipt=source_hash, binding=binding
            )
    finally:
        if pending.exists():
            shutil.rmtree(pending)
    return target


def prepare_replay(replay, cache_root, *, workers=4, progress=None):
    """Preserve admitted cohort identities; only substitute lossless storage paths."""
    result = copy.copy(replay)
    streams = (replay.offline, replay.online) if hasattr(replay, "offline") else (replay,)
    prepared = []
    entries = [entry for stream in streams for entry in stream.entries]
    with ThreadPoolExecutor(max_workers=workers) as pool:

        def prepare(entry):
            return prepare_episode(
                entry[0], cache_root, expected_receipt=entry[2], binding=entry[1]
            )

        for i, path in enumerate(pool.map(prepare, entries)):
            prepared.append((path, entries[i][1], sha256(path / "complete.json")))
            if progress is not None:
                progress(dict(phase="prepare_replay", episodes=i + 1, expected=len(entries)))
    if len(streams) == 2:
        result.offline, result.online = copy.copy(streams[0]), copy.copy(streams[1])
        n = len(streams[0])
        result.offline.entries, result.online.entries = prepared[:n], prepared[n:]
        result.offline.cache = result.online.cache = result.cache = None
    else:
        result.entries, result.cache = prepared, None
    return result
