"""Disposable binary metadata caches for already admitted, operator-owned replay.

The canonical files remain JSON. Marshal stores only JSON builtins, never model
or Python class objects. Cache identity includes the Python version and file stat.
"""

import hashlib
import json
import marshal
import sys
import tempfile
from pathlib import Path


def load_metadata(path, cache, *, fields=None):
    path, cache = Path(path), Path(cache)
    fields = None if fields is None else tuple(fields)
    identity = (str(path.resolve()), fields, sys.implementation.cache_tag, marshal.version)
    key = hashlib.sha256(repr(identity).encode()).hexdigest()
    cached = cache / (key + ".marshal")

    def signature():
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    before = signature()
    try:
        saved, value = marshal.loads(cached.read_bytes())
        if saved == before:
            return value
    except (FileNotFoundError, EOFError, ValueError, TypeError):
        pass
    value = json.loads(path.read_text())
    if fields is not None:
        value = {key: value[key] for key in fields}
    if signature() != before:
        raise ValueError("replay metadata changed while reading")
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache, prefix=".pending-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            marshal.dump((before, value), stream)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(cached)
    return value
