"""Keep immutable experiment identities across an explicitly audited runtime repair.

A pinned receipt verifies the ENTIRE executed Python tree. Audited resource
repairs, lossless input transport and numerically checked compiled execution may
retain a prior experiment identity. Architecture, data, reward and optimizer
changes cannot use this compatibility path.
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


_INPUT_TRANSPORT_FILES = _OPERATIONAL_FILES | {
    "src/latency_meta_mdp/meta/campaign_tracking.py",
    "src/latency_meta_mdp/meta/information/attention_fqi.py",
    "src/latency_meta_mdp/meta/information/fqi_inputs.py",
    "src/latency_meta_mdp/meta/information/memory.py",
    "src/latency_meta_mdp/meta/information/replay_metadata.py",
    "src/latency_meta_mdp/meta/information/packed_replay.py",
    "src/latency_meta_mdp/meta/information/attention_replay.py",
}

_COMPILED_EXECUTION_FILES = _INPUT_TRANSPORT_FILES | {
    "scripts/run_conveyor_meta.py",
    "src/latency_meta_mdp/meta/information/campaign.py",
    "src/latency_meta_mdp/meta/information/training.py",
    "src/latency_meta_mdp/meta/information/distributed_fqi.py",
    "src/latency_meta_mdp/meta/information/fqi_batch.py",
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
    allowed = _OPERATIONAL_FILES
    checked_execution = receipt.get("kind") in (
        "lossless-fqi-input-transport",
        "compiled-fqi-execution",
    )
    if checked_execution:
        allowed = (
            _COMPILED_EXECUTION_FILES
            if receipt["kind"] == "compiled-fqi-execution"
            else _INPUT_TRANSPORT_FILES
        )
        evidence = receipt.get("equivalence_evidence", {})
        if not evidence or any(sha256(root / p) != digest for p, digest in evidence.items()):
            raise ValueError("execution equivalence evidence differs")
    if not changed or not changed <= allowed:
        raise ValueError("runtime compatibility is limited to audited operational files")
    result.update(code_hash=identity_hash(before), runtime_patch_sha256=digest)
    if checked_execution:
        name = "src/latency_meta_mdp/meta/information/attention_replay.py"
        if name in before:
            result["replay_converter_sha256"] = before[name]
        # Pinned collation changes transport, not the causal history selector.
        name = "src/latency_meta_mdp/meta/information/memory.py"
        if name in before:
            result["replay_selector_sha256"] = before[name]
    # Preserve protocols already embedded in immutable checkpoints by the first
    # runtime repair. The full prior tree must differ only in operational files.
    previous = receipt.get("previous_execution")
    if previous is not None:
        changed = {p for p in before.keys() | previous.keys() if before.get(p) != previous.get(p)}
        if not changed <= _OPERATIONAL_FILES:
            raise ValueError("previous execution differs outside operational files")
        result["previous_execution_code_hash"] = identity_hash(previous)
    return result
