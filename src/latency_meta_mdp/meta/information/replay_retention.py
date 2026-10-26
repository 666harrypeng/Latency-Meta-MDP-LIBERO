"""Bounded episode replay with explicit, restartable payload retirement receipts."""

import json
import shutil
from pathlib import Path

from latency_meta_mdp.meta.information.config import identity_hash, sha256, validate_binding
from latency_meta_mdp.meta.information.validation import atomic_json


def tree_bytes(directory):
    return sum(
        p.stat().st_size for p in Path(directory).rglob("*") if p.is_file() and not p.is_symlink()
    )


def validate_retention(policy):
    fields = {
        "recent_batches",
        "history_episodes",
        "max_online_bytes",
        "max_cache_bytes",
        "max_run_bytes",
        "min_free_bytes",
        "episode_cache_size",
    }
    if set(policy) != fields or any(type(v) is not int or v < 0 for v in policy.values()):
        raise ValueError("invalid replay retention limits")
    if any(policy[k] < 1 for k in fields - {"min_free_bytes", "history_episodes"}):
        raise ValueError("positive replay retention limits required")
    if policy["history_episodes"] % 3:
        raise ValueError("history capacity must split across three delay regimes")


class BoundedReplay:
    def __init__(self, directory, policy, *, seed, cache_dir=None):
        validate_retention(policy)
        self.root = Path(directory).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.policy, self.seed = policy, seed
        self.cache = Path(cache_dir).resolve() if cache_dir else self.root / "replay-cache"
        self.state_dir = self.root / "replay-state"
        self.inventory_dir = self.root / "replay-inventory"
        self.state_dir.mkdir(exist_ok=True)
        self.inventory_dir.mkdir(exist_ok=True)
        self.binding = dict(policy=policy, seed=seed)

    def check_storage(self):
        size = tree_bytes(self.root)
        if size > self.policy["max_run_bytes"]:
            raise RuntimeError("campaign storage ceiling reached; checkpoint retained")
        if shutil.disk_usage(self.root).free < self.policy["min_free_bytes"]:
            raise RuntimeError("insufficient free storage for next phase; checkpoint retained")
        return size

    def _inventory(self, cycle, source):
        path = self.inventory_dir / f"{cycle:04d}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            validate_binding(source, saved["source"])
            return saved["records"]
        directory = Path(source["path"]).resolve()
        if not directory.is_relative_to(self.root):
            raise ValueError("retention only admits campaign-owned online collections")
        validate_binding(source["manifest_sha256"], sha256(directory / "cohort.json"))
        manifest = json.loads((directory / "cohort.json").read_text())
        records = []
        for ep in manifest["episodes"]:
            if ep["split"] != "train" or ep["regime"] not in ("zero", "nominal", "family"):
                raise ValueError("only training episodes may enter bounded replay")
            attempts = list((directory / "episodes" / ep["episode_id"]).glob("attempt-*"))
            complete = [p.resolve() for p in attempts if (p / "complete.json").is_file()]
            if len(complete) != 1 or not complete[0].is_relative_to(self.root):
                raise ValueError("one owned completed attempt required")
            attempt = complete[0]
            records.append(
                dict(
                    key=sha256(attempt / "complete.json"),
                    source=source,
                    batch=cycle,
                    episode_id=ep["episode_id"],
                    regime=ep["regime"],
                    attempt=str(attempt),
                    bytes=tree_bytes(attempt),
                    packed_key=sha256(attempt / "episode/complete.json"),
                )
            )
        atomic_json(path, dict(source=source, records=records))
        return records

    def _history(self, candidates, max_bytes=None):
        # Stable random priorities are independent of outcome and episode length.
        ordered = sorted(candidates, key=lambda r: identity_hash([self.seed, r["key"]]))
        groups = [
            [r for r in ordered if r["regime"] == regime]
            for regime in ("zero", "nominal", "family")
        ]
        kept, size = [], 0
        budget = self.policy["max_online_bytes"] if max_bytes is None else max_bytes
        for i in range(self.policy["history_episodes"] // 3):
            for group in groups:
                if i < len(group) and size + group[i]["bytes"] <= budget:
                    kept.append(group[i])
                    size += group[i]["bytes"]
        return kept

    def select(self, cycle, source):
        path = self.state_dir / f"{cycle:04d}.json"
        if path.exists():
            state = json.loads(path.read_text())
            validate_binding(self.binding, state["binding"])
            validate_binding(source, state["incoming"])
            return state["sources"]
        previous = self.state_dir / f"{cycle - 1:04d}.json"
        state = (
            json.loads(previous.read_text()) if previous.exists() else dict(recent=[], history=[])
        )
        if cycle > 1 and not previous.exists():
            raise ValueError("missing preceding replay snapshot")
        if previous.exists():
            validate_binding(self.binding, state["binding"])
        recent = [*state["recent"], self._inventory(cycle, source)]
        history = state["history"]
        while (
            len(recent) > self.policy["recent_batches"]
            or sum(r["bytes"] for b in recent for r in b) + sum(r["bytes"] for r in history)
            > self.policy["max_online_bytes"]
        ):
            if len(recent) == 1:
                available = self.policy["max_online_bytes"] - sum(r["bytes"] for r in recent[0])
                if available < 0:
                    raise RuntimeError("latest batch exceeds online replay byte ceiling")
                history = self._history(history, max_bytes=available)
                break
            history = self._history([*history, *recent.pop(0)])
        active = [r for batch in recent for r in batch] + history
        sources = {}
        for record in active:
            original = record["source"]
            key = original["path"]
            group = "latest" if record["batch"] == cycle else "old"
            item = sources.setdefault(key, dict(original, group=group, episode_ids=[]))
            item["episode_ids"].append(record["episode_id"])
        sources = [dict(s, episode_ids=sorted(s["episode_ids"])) for s in sources.values()]
        atomic_json(
            path,
            dict(
                binding=self.binding,
                incoming=source,
                recent=recent,
                history=history,
                sources=sources,
                active_bytes=sum(r["bytes"] for r in active),
            ),
        )
        return sources

    def cleanup(self, cycle):
        """Called only after fit+validation publish; keep two cycle recovery inputs."""
        snapshots = sorted(self.state_dir.glob("*.json"))
        if not snapshots or int(snapshots[-1].stem) != cycle:
            return
        protected = set()
        for path in snapshots[-2:]:
            state = json.loads(path.read_text())
            protected.update(r["key"] for b in state["recent"] for r in b)
            protected.update(r["key"] for r in state["history"])
        retired_path = self.root / "replay-retired.json"
        retired = json.loads(retired_path.read_text()) if retired_path.exists() else dict(files={})
        doomed_cache = []
        for path in sorted(self.inventory_dir.glob("*.json")):
            for record in json.loads(path.read_text())["records"]:
                if record["key"] in protected:
                    continue
                attempt = Path(record["attempt"])
                if not attempt.resolve().is_relative_to(self.root):
                    raise ValueError("retirement escaped owned campaign")
                validate_binding(record["key"], sha256(attempt / "complete.json"))
                receipt = json.loads((attempt / "complete.json").read_text())
                for name, digest in receipt["files"].items():
                    # Keep outcome summaries, provenance and lightweight action audit.
                    if Path(name).suffix not in (".npz", ".npy", ".mp4") and name not in (
                        "episode/records.json",
                        "episode/audit.json",
                    ):
                        continue
                    file = (attempt / name).resolve()
                    if not file.is_relative_to(attempt):
                        raise ValueError("retirement payload escaped episode")
                    if str(file) not in retired["files"]:
                        validate_binding(digest, sha256(file))
                        retired["files"][str(file)] = digest
                doomed_cache.extend(
                    self.cache / name / record["packed_key"]
                    for name in (
                        "mmap-raw-v2",
                        "mmap-raw-f16-v1",
                        "mmap-raw-f16-v2",
                        "mmap-raw-direct-v5",
                    )
                )
        # Publish intent first; repeated cleanup after interruption is safe.
        atomic_json(retired_path, retired)
        for name, digest in retired["files"].items():
            file = Path(name)
            if not file.is_relative_to(self.root):
                raise ValueError("retired payload outside campaign")
            if file.exists():
                validate_binding(digest, sha256(file))
            file.unlink(missing_ok=True)
        for path in doomed_cache:
            if path.exists():
                shutil.rmtree(path)
        self.check_storage()
