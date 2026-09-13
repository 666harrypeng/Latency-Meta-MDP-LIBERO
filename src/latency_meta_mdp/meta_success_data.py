"""Audited, zero-copy-on-disk views of completed terminal-success Meta replay."""

import json
import os
from pathlib import Path

import numpy as np

from latency_meta_mdp.meta_replay import FEATURE_FORMAT

PHYSICAL_KEYS = (
    "checkpoint_step", "checkpoint_verification_sha256", "forecast_identity",
    "client_config_sha256", "rtc_calibration", "rtc_max_guidance_weight",
    "decision_interval_ticks", "runtime_patch_sha256", "maximum_steps",
    "profile_sha256", "bootstrap_verification_sha256", "protocol_id", "conditioning",
)


def append_success_replay(parent_path, entries, output_path):
    """Commit a new immutable snapshot of the same logical replay after complete episodes."""
    manifest = json.loads(Path(parent_path).read_text())
    if manifest["status"] != "completed" or not entries:
        raise ValueError("append requires a completed parent and new episodes")
    paths = {str(Path(e["replay"]).resolve()) for e in manifest["episodes"]}
    reference, _ = audited_episode(manifest["episodes"][0])
    for entry in entries:
        path = str(Path(entry["replay"]).resolve())
        if path in paths:
            raise ValueError("duplicate replay source")
        paths.add(path)
        meta, _ = audited_episode(entry)
        if (meta["case"]["meta_partition"] != "train"
                or meta["case"]["master_index"] not in manifest["train_masters"]):
            raise ValueError("online append requires declared training masters")
        if any(meta["identity"][k] != reference["identity"][k] for k in PHYSICAL_KEYS):
            raise ValueError("online append changes the frozen physical environment")
    manifest["episodes"] += entries
    manifest["expected_episodes"] = len(manifest["episodes"])
    manifest["revision"] = manifest.get("revision", 0) + 1
    manifest["parent_snapshot"] = str(Path(parent_path).resolve())
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix('.tmp')
    with temporary.open('x') as f:
        json.dump(manifest, f, indent=2)
    os.replace(temporary, output_path)


def audited_episode(entry):
    """Read and validate labels before materializing large visual arrays."""
    result = json.loads(Path(entry["result"]).read_text())
    if not result["terminated"] or result["truncated"]:
        raise ValueError("success view requires a completed task episode, not capture truncation")
    with np.load(entry["replay"], allow_pickle=False) as d:
        meta = json.loads(d["metadata_utf8"].tobytes())
        if meta["feature_format"] != FEATURE_FORMAT:
            raise ValueError("Meta replay feature format differs")
        if meta["identity"] != result["identity"] or meta["case"] != result["case"]:
            raise ValueError("replay/result identity mismatch")
        fields = {k: d[k] for k in (
            "state_index", "next_state_index", "action", "reward", "bootstrap_discount",
            "duration_ticks", "terminated", "truncated", "shielded", "vector", "legal_actions",
        )}
        # Earlier immutable shards kept timestamps in the paired result JSON only.
        for key in ("start_tick", "end_tick"):
            fields[key] = (d[key] if key in d else np.asarray(
                [row[key] for row in result["decision_transitions"]], dtype=np.int64))
        raw = np.asarray([r["undiscounted_reward"] for r in result["decision_transitions"]])
        if "undiscounted_reward" in d and not np.array_equal(raw, d["undiscounted_reward"]):
            raise ValueError("raw replay/audit reward mismatch")
    n = len(fields["action"])
    term, duration = fields["terminated"], fields["duration_ticks"]
    if (not n or raw.shape != (n,) or fields["truncated"].any() or
            np.any(duration <= 0) or term.sum() != 1 or not term[-1]):
        raise ValueError("invalid completed decision sequence")
    expected = np.zeros(n)
    expected[-1] = float(result["success"])
    gamma = meta["identity"]["discount_per_tick"]
    if not 0 <= gamma <= 1 or not np.array_equal(raw, expected):
        raise ValueError("terminal-only reward and episode outcome disagree")
    if not np.allclose(fields["reward"], expected * gamma ** (duration - 1)):
        raise ValueError("source reward is not the declared terminal-success reward")
    if not np.allclose(fields["bootstrap_discount"], np.where(term, 0, gamma ** duration)):
        raise ValueError("source duration discount mismatch")
    idx, nxt = fields["state_index"], fields["next_state_index"]
    size = len(fields["vector"])
    if (np.any(idx < 0) or np.any(idx >= size) or np.any(nxt >= size)
            or np.any((nxt < 0) != term) or not np.isin(fields["action"], [0, 1]).all()):
        raise ValueError("invalid replay state/action indices")
    if not fields["legal_actions"][idx, fields["action"]].all():
        raise ValueError("illegal executed Meta action")
    if (not np.array_equal(nxt[:-1], idx[1:])
            or not np.array_equal(fields["end_tick"][:-1], fields["start_tick"][1:])
            or not np.array_equal(fields["end_tick"] - fields["start_tick"], duration)):
        raise ValueError("invalid contiguous decision sequence")
    for i, row in enumerate(result["decision_transitions"]):
        for key in ("start_tick", "end_tick", "duration_ticks"):
            if key in row and row[key] != fields[key][i]:
                raise ValueError("decision sequence differs from result audit")
    fields["success_reward"] = raw.astype(np.float32)
    return meta, fields


def load_success_replay(manifest_path, *, visual_cache=None, cost_profile=None):
    manifest = json.loads(Path(manifest_path).read_text())
    entries = manifest["episodes"]
    if (manifest["schema"] != 2 or manifest["status"] != "completed"
            or len(entries) != manifest["expected_episodes"]):
        raise ValueError("incomplete success replay manifest")
    paths = [str(Path(e["replay"]).resolve()) for e in entries]
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate replay source")
    states, vectors, masks, rows, inventory = [], [], [], [], []
    masters = {"train": set(), "validation": set()}
    binding, offset, transition_offset = None, 0, 0
    for episode_id, entry in enumerate(entries):
        meta, d = audited_episode(entry)
        identity = {k: meta["identity"][k] for k in PHYSICAL_KEYS}
        if identity["maximum_steps"] != 1000:
            raise ValueError("success objective requires the declared1000tick task horizon")
        if binding is not None and binding != identity:
            raise ValueError("success view mixes physical policy/timing identities")
        binding = identity
        part = meta["case"]["meta_partition"]
        masters[part].add(meta["case"]["master_index"])
        if visual_cache is None:
            with np.load(entry["replay"], allow_pickle=False) as source:
                x = source["visual"]
        else:
            from latency_meta_mdp.meta_visual_replay import cache_visual_shard

            cached = cache_visual_shard(Path(entry["replay"]), Path(visual_cache))
            x = np.load(cached, mmap_mode='r', allow_pickle=False)
        v, legal = d["vector"], d["legal_actions"]
        if x.shape[1:] != (2, 2, 196, 384) or v.shape != (len(x), 501):
            raise ValueError("invalid replay feature shape")
        if (legal.shape != (len(x), 2) or not np.isfinite(v).all()
                or (visual_cache is None and not np.isfinite(x).all())):
            raise ValueError("invalid replay features or legal mask")
        states.append(x if visual_cache is None else cached)
        vectors.append(v)
        masks.append(legal)
        nxt = d["next_state_index"]
        cost_report = None
        if cost_profile is not None:
            from latency_meta_mdp.meta_cost import episode_cost_components

            cost_report = episode_cost_components(
                json.loads(Path(entry["result"]).read_text()), cost_profile
            )
            if (cost_report["transition_costs"] is None
                    or len(cost_report["transition_costs"]) != len(nxt)):
                raise ValueError("cost ledger does not match replay transitions")
        rows.append({
            "state": d["state_index"] + offset,
            "next": np.where(nxt < 0, 0, nxt + offset),
            "action": d["action"], "reward": d["success_reward"],
            "discount": (~d["terminated"]).astype(np.float32),
            "train": np.full(len(nxt), part == "train", bool),
            "task_reward": d["success_reward"],
            "cost": (np.zeros(len(nxt), np.float32) if cost_report is None
                     else np.asarray(cost_report["transition_costs"], np.float32)),
            "duration_ticks": d["duration_ticks"],
            "terminated": d["terminated"],
            "episode_id": np.full(len(nxt), episode_id, np.int64),
            "next_transition": np.where(
                d["terminated"], -1, np.arange(len(nxt)) + transition_offset + 1),
        })
        inventory.append({
            **entry, "partition": part, "states": len(x), "transitions": len(nxt),
            "source_gamma": meta["identity"]["discount_per_tick"],
            "conversion": "verified_undiscounted_terminal_success",
            **({} if cost_report is None else {"cost": cost_report}),
        })
        offset += len(x)
        transition_offset += len(nxt)
    if (not all(masters.values()) or masters["train"] & masters["validation"] or
            any(sorted(masters[k]) != sorted(manifest[k + "_masters"]) for k in masters)):
        raise ValueError("success replay grouped split differs from manifest")
    binding.update(discount_per_tick=1.0, task_horizon_terminal=True)
    if visual_cache is not None:
        from latency_meta_mdp.meta_visual_replay import VisualReplay

        visual = VisualReplay(states)
    else:
        visual = np.concatenate(states)
    return (
        visual, np.concatenate(vectors), np.concatenate(masks),
        {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}, binding, inventory,
        {k: sorted(v) for k, v in masters.items()},
    )
