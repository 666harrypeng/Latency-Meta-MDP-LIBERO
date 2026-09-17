"""Verified train/validation source selection for conveyor preparation."""

import json
from pathlib import Path

from latency_meta_mdp.data.conveyor.source import ConveyorSource


def split_sources(corpus_manifest, *, split):
    if split not in {"train", "validation"}:
        raise ValueError("training preparation accepts train/validation only")
    path = Path(corpus_manifest).resolve()
    corpus = json.loads(path.read_text())
    if (
        corpus.get("format_id") != "conveyor_expert_corpus_v1"
        or corpus["status"] != "completed"
        or corpus["admitted_count"] != corpus["requested_count"]
    ):
        raise ValueError("preparation requires a complete, fully admitted corpus")
    roles = {}
    for split_name, seeds in corpus["splits"].items():
        for seed in seeds:
            if seed in roles:
                raise ValueError("corpus split groups overlap")
            roles[seed] = split_name
    expected = {
        (split, seed) for split in corpus["collect_splits"] for seed in corpus["splits"][split]
    }
    actual = {(row["split"], row["seed"]) for row in corpus["episodes"]}
    if actual != expected or len(corpus["episodes"]) != len(expected):
        raise ValueError("corpus split inventory is incomplete or duplicated")
    seen = set()
    for row in corpus["episodes"]:
        seed = row["seed"]
        if seed in seen or roles.get(seed) != row["split"]:
            raise ValueError("corpus episode group/split mismatch")
        seen.add(seed)
        if row["split"] != split:
            continue
        if not row["admitted"]:
            raise ValueError("non-admitted episode in policy source")
        root = (path.parent / row["source_root"]).resolve()
        if not root.is_relative_to(path.parent):
            raise ValueError("source path escapes corpus")
        source = ConveyorSource(root)
        if any(
            source.manifest["context"].get(k) != corpus["bindings"][k] for k in ("scene", "expert")
        ):
            raise ValueError("source task/expert configuration differs from corpus")
        if (
            source.manifest["seed"] != seed
            or source.manifest["purpose"] != corpus["purpose"]
            or source.manifest["group_id"] != row["group_id"]
            or source.manifest["action_contract"] != corpus["action_contract"]
            or len(source.actions) != row["transition_count"]
        ):
            raise ValueError("source identity differs from corpus")
        yield row, source
