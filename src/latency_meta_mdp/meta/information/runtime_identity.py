"""Keep immutable experiment identities across an explicitly audited runtime repair.

A pinned receipt verifies the ENTIRE executed Python tree. Only the resource
admission fix and its provenance plumbing may retain a prior experiment identity;
model, data, reward and optimizer changes cannot use this compatibility path.
"""

import json
import os
from pathlib import Path

from latency_meta_mdp.meta.information.config import identity_hash, sha256

_OPERATIONAL_FILES = {
    "src/latency_meta_mdp/meta/information/" + name + ".py"
    for name in (
        "resources",
        "preflight",
        "attention_campaign",
        "attention_training",
        "runtime_identity",
        "monitor",
    )
}


def source_identity(root):
    root = Path(root).resolve()
    current = {
        str(p.relative_to(root)): sha256(p)
        for folder in ("src/latency_meta_mdp", "scripts")
        for p in sorted((root / folder).rglob("*.py"))
    }
    actual = identity_hash(current)
    result = dict(code_hash=actual, execution_code_hash=actual, runtime_patch_sha256=None)
    path = os.environ.get("CONVEYOR_RUNTIME_PATCH")
    if not path:
        return result
    digest = sha256(path)
    if digest != os.environ.get("CONVEYOR_RUNTIME_PATCH_SHA256"):
        raise ValueError("runtime patch receipt differs from pinned SHA256")
    receipt = json.loads(Path(path).read_text())
    if receipt["root"] != str(root) or receipt["after"] != current:
        raise ValueError("execution source differs from audited runtime repair")
    before = receipt["before"]
    changed = {p for p in before.keys() | current.keys() if before.get(p) != current.get(p)}
    if not changed or not changed <= _OPERATIONAL_FILES:
        raise ValueError("runtime compatibility is limited to audited operational files")
    result.update(code_hash=identity_hash(before), runtime_patch_sha256=digest)
    # Preserve protocols already embedded in immutable checkpoints by the first
    # runtime repair. The full prior tree must differ only in operational files.
    previous = receipt.get("previous_execution")
    if previous is not None:
        changed = {p for p in before.keys() | previous.keys() if before.get(p) != previous.get(p)}
        if not changed <= _OPERATIONAL_FILES:
            raise ValueError("previous execution differs outside operational files")
        result["previous_execution_code_hash"] = identity_hash(previous)
    return result
