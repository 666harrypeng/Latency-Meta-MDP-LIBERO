"""Reuse content hashes for unchanged files in our local replay storage."""

import json
import tempfile
from pathlib import Path

from latency_meta_mdp.meta.information.config import identity_hash, sha256


def _signature(path):
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def cached_sha256(path, cache_dir):
    """Hash once, then stat only; edits and replacements invalidate the entry."""
    path = Path(path).resolve()
    if cache_dir is None:
        return sha256(path)
    cache_dir = Path(cache_dir)
    target = cache_dir / (identity_hash(str(path)) + ".json")
    before = _signature(path)
    if target.exists():
        saved = json.loads(target.read_text())
        if saved["signature"] == before:
            return saved["sha256"]
    digest = sha256(path)
    if _signature(path) != before:
        raise ValueError("replay file changed during verification")
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=cache_dir, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(dict(signature=before, sha256=digest), stream)
    try:
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return digest
