"""Attention decision contexts derived from verified raw stage timelines."""

import copy
import hashlib
import json
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path
from types import MappingProxyType

import numpy as np

from latency_meta_mdp.meta.information.memory import DEFAULT_MEMORY, MemoryStore
from latency_meta_mdp.meta.information.memory_schema import (
    AttentionPacket,
    DecisionRef,
    PublicEvent,
)
from latency_meta_mdp.meta.information.records import decision_view
from latency_meta_mdp.meta.information.schema import ActorPacket


def contexts_from_raw(raw, action_spec, *, config=None):
    """Preserve forced events before folding rewards; never fabricate unavailable history."""
    config = DEFAULT_MEMORY if config is None else config
    events, rows = list(raw["events"]), []
    current_tick = forecast_tick = -1
    for i, row in enumerate(raw["rows"]):
        packet = row["packet"]
        action = row["action"]
        if not isinstance(packet, AttentionPacket):
            values = {f.name: getattr(packet, f.name) for f in fields(ActorPacket)}
            values["previous_action"] = (
                -1 if packet.previous_action == -1 else action_spec.from_v1(packet.previous_action)
            )
            # ActorPacket was validated on admission. Preserve read-only mmap visual views.
            values.update(
                action_spec=action_spec,
                decision_key=(packet.tick, 4 * i + 2),
                episode_remaining=max(0, config["episode_ticks"] - packet.tick),
            )
            packet = object.__new__(AttentionPacket)
            for key, value in values.items():
                object.__setattr__(packet, key, value)
            if packet.current_tick > current_tick:
                events.append(
                    PublicEvent(
                        f"legacy-observe-{i}",
                        "observe",
                        (packet.tick, 4 * i),
                        (packet.current_tick,),
                        dict(
                            visual=packet.current_tokens[None],
                            proprio=packet.observed_proprio[None],
                        ),
                    )
                )
                payload = dict(events[-1].payload, visual=packet.current_tokens[None])
                object.__setattr__(events[-1], "payload", MappingProxyType(payload))
                current_tick = packet.current_tick
            if packet.forecast_tick > forecast_tick:
                # The first raw packet exposing a forecast is its causal availability bound.
                prefix = hashlib.sha256(
                    packet.buffer.tobytes() + packet.buffer_mask.tobytes()
                ).hexdigest()
                events.append(
                    PublicEvent(
                        f"legacy-predict-{i}",
                        "predict",
                        (packet.tick, 4 * i + 1),
                        (packet.forecast_tick + packet.estimated_delay,),
                        dict(
                            visual=packet.forecast_tokens,
                            proprio=packet.forecast_proprio,
                            source_tick=packet.forecast_tick,
                            query_ticks=packet.estimated_delay,
                            buffer_version=packet.forecast_buffer_version,
                            prefix_hash=prefix,
                        ),
                    )
                )
                payload = dict(events[-1].payload, visual=packet.forecast_tokens)
                object.__setattr__(events[-1], "payload", MappingProxyType(payload))
                forecast_tick = packet.forecast_tick
            action = action_spec.from_v1(action)
            events.append(
                PublicEvent(
                    f"legacy-decision-{i}",
                    "decision",
                    (packet.tick, 4 * i + 3),
                    (packet.tick,),
                    dict(action=action, stage=packet.stage, probability=row["probability"]),
                )
            )
        elif packet.action_spec != action_spec:
            raise ValueError("replay action vocabulary differs")
        rows.append({**row, "packet": packet, "action": action})
    events.sort(key=lambda e: e.available_at)
    memory = MemoryStore(action_spec, config=config)
    memory.reset(raw["episode_id"])
    cursor = 0
    decisions = decision_view(rows)
    raw_indices = {id(row["packet"]): i for i, row in enumerate(rows)}
    for row in decisions:
        while cursor < len(events) and events[cursor].available_at < row["packet"].decision_key:
            memory.ingest(events[cursor])
            cursor += 1
        row["context"] = memory.context(
            DecisionRef(raw["episode_id"], raw_indices[id(row["packet"])]), row["packet"]
        )
        row["legal"] = action_spec.legal(
            row["packet"].stage,
            row["packet"].remaining,
            forecast_fresh=row["packet"].forecast_fresh,
        )
    for event in events[cursor:]:
        memory.ingest(event)  # Validate the remaining timeline, including terminal events.
    for i, row in enumerate(decisions):
        row["next_context"] = decisions[i + 1]["context"] if i + 1 < len(decisions) else None
        row["next_legal"] = (
            decisions[i + 1]["legal"]
            if i + 1 < len(decisions)
            else np.zeros(action_spec.num_actions, bool)
        )
    return decisions


class AttentionEpisodes:
    """Lazy context reconstruction; process spawn sends receipts, never feature arrays."""

    def __init__(self, entries, action_spec, config, cache_size=128):
        if type(cache_size) is not int or cache_size < 1:
            raise ValueError("positive episode cache size required")
        self.entries, self.action_spec, self.config = entries, action_spec, config
        self.cache_size = cache_size
        self.loaded = OrderedDict()
        self.lengths = []
        for entry in entries:
            rows = json.loads((entry.path / "records.json").read_text())["rows"]
            self.lengths.append(sum(not r["forced"] for r in rows))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        self.entries[index].verify_unchanged()
        return self._load_verified(index)

    def _load_verified(self, index):
        from latency_meta_mdp.meta.information.packed_replay import load_packed_raw

        if index not in self.loaded:
            while len(self.loaded) >= self.cache_size:
                self.loaded.popitem(last=False)
            raw = load_packed_raw(self.entries[index])
            # Old recorder names collide between independent cohorts; source receipt is unique.
            raw["episode_id"] = self.entries[index].path.name + "/" + raw["episode_id"]
            self.loaded[index] = contexts_from_raw(raw, self.action_spec, config=self.config)
        self.loaded.move_to_end(index)
        return self.loaded[index]

    def __getstate__(self):
        return dict(
            entries=self.entries,
            action_spec=self.action_spec,
            config=self.config,
            cache_size=self.cache_size,
            lengths=self.lengths,
        )

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.loaded = OrderedDict()


class DecisionReplay:
    """Transition-indexed view; source groups remain disjoint as the buffer grows."""

    def __init__(self, episodes, groups=None):
        self.episodes = episodes
        self.groups = list(groups or ["startup"] * len(episodes))
        if len(self.groups) != len(episodes) or any(
            g not in ("startup", "old", "latest") for g in self.groups
        ):
            raise ValueError("invalid replay groups")
        self.lengths = np.array(
            episodes.lengths
            if isinstance(episodes, AttentionEpisodes)
            else [len(episodes[i]) for i in range(len(episodes))],
            dtype=np.int64,
        )
        if not len(self.lengths) or np.any(self.lengths <= 0):
            raise ValueError("replay requires nonempty decision episodes")
        self.ends = self.lengths.cumsum()
        starts = np.r_[0, self.ends[:-1]]
        self.group_indices = {
            name: np.concatenate(
                [
                    np.arange(a, b, dtype=np.int64)
                    for a, b, g in zip(starts, self.ends, self.groups, strict=True)
                    if g == name
                ]
            )
            for name in dict.fromkeys(self.groups)
        }

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = int(np.searchsorted(self.ends, index, side="right"))
        start = 0 if episode == 0 else self.ends[episode - 1]
        return self.episodes[episode][int(index - start)]

    @contextmanager
    def batch(self, indices, *, executor=None):
        """Verify each touched source around a batch read; preserve sampling order."""
        indices = [int(i) for i in indices]
        if any(not 0 <= i < len(self) for i in indices):
            raise IndexError("decision batch index outside replay")
        if not isinstance(self.episodes, AttentionEpisodes):
            yield [self[i] for i in indices]
            return
        episodes = np.searchsorted(self.ends, indices, side="right")
        unique = list(dict.fromkeys(int(i) for i in episodes))
        entries = [self.episodes.entries[i] for i in unique]

        def verify():
            apply = map if executor is None else executor.map
            list(apply(lambda entry: entry.verify_unchanged(), entries))

        verify()
        loaded = {i: self.episodes._load_verified(i) for i in unique}
        yield [
            loaded[episode][index - (int(self.ends[episode - 1]) if episode else 0)]
            for index, episode in zip(indices, episodes, strict=True)
        ]
        verify()


class DecisionSampler:
    def __init__(self, replay, *, seed):
        self.replay = replay
        self.rng = np.random.default_rng(seed)
        self.visits = 0
        if "startup" not in replay.group_indices:
            raise ValueError("startup replay must remain available")
        self.names = [g for g in ("startup", "old", "latest") if g in replay.group_indices]
        weights = dict(startup=0.5, old=0.25, latest=0.25)
        if "old" not in self.names:
            weights["latest"] += weights["old"]
        if "latest" not in self.names:
            weights["old"] += weights["latest"]
        values = np.array([weights[g] for g in self.names])
        self.probabilities = values / values.sum()

    def sample(self, count):
        if type(count) is not int or count < 1:
            raise ValueError("positive sample count required")
        groups = self.rng.choice(len(self.names), size=count, p=self.probabilities)
        result = np.empty(count, dtype=np.int64)
        for i, name in enumerate(self.names):
            selected = groups == i
            result[selected] = self.rng.choice(
                self.replay.group_indices[name], size=int(selected.sum())
            )
        self.visits += count
        return result

    def state_dict(self):
        return dict(rng=copy.deepcopy(self.rng.bit_generator.state), visits=self.visits)

    def load_state_dict(self, state):
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
        self.visits = state["visits"]


def validate_sources(config, manifests):
    from latency_meta_mdp.meta.information.config import identity_hash, validate_binding

    if not manifests:
        raise ValueError("explicit source manifests required")
    for manifest in manifests:
        binding = manifest["binding"]
        if binding["scope"] not in ("starter_training", "online_training"):
            raise ValueError("evaluation and diagnostic data cannot enter replay")
        validate_binding(config["assets"]["identity"], binding["assets"]["identity"])
        validate_binding(config["mode"], binding["mode"])
        if binding["schema_version"] == 2:
            validate_binding(identity_hash(config), binding["config_hash"])
            validate_binding(config["actions"], binding["actions"])
        elif binding["schema_version"] != 1 or config["actions"]["starvation_mode"] != "guarded":
            raise ValueError("legacy replay only qualifies for guarded protocol")
        episodes = manifest["episodes"]
        if not episodes or len({e["episode_id"] for e in episodes}) != len(episodes):
            raise ValueError("empty or duplicate episode identities in source")
        if any(
            e["split"] != "train" or e["scene"] not in config["splits"]["train"] for e in episodes
        ):
            raise ValueError("training source contains evaluation scene")


def prepare_attention_replay(
    config,
    sources,
    *,
    root,
    cache_dir,
    workers=2,
    progress=None,
    episode_cache_size=128,
    cache_max_bytes=None,
):
    """Only named, hash-pinned manifests; never discover additional cohorts on disk."""
    from concurrent.futures import ThreadPoolExecutor

    from latency_meta_mdp.meta.information.config import identity_hash, sha256, validate_binding
    from latency_meta_mdp.meta.information.memory_schema import ActionSpec
    from latency_meta_mdp.meta.information.packed_replay import (
        CacheBudget,
        compact_packed_episode,
        prepare_packed_episode,
    )
    from latency_meta_mdp.meta.information.training import VerifiedReplay
    from latency_meta_mdp.meta.information.validation import atomic_json

    directories = [(Path(root) / s["path"]).resolve() for s in sources]
    manifests = []
    for source, directory in zip(sources, directories, strict=True):
        if set(source) - {"episode_ids"} != {"path", "manifest_sha256", "group"}:
            raise ValueError("source requires path, manifest hash and replay group")
        validate_binding(source["manifest_sha256"], sha256(directory / "cohort.json"))
        manifest = json.loads((directory / "cohort.json").read_text())
        if "episode_ids" in source:
            ids = source["episode_ids"]
            if not ids or len(ids) != len(set(ids)):
                raise ValueError("nonempty distinct replay episode subset required")
            selected = [e for e in manifest["episodes"] if e["episode_id"] in ids]
            if len(selected) != len(ids):
                raise ValueError("replay subset contains unknown episodes")
            manifest = dict(manifest, episodes=selected)
        manifests.append(manifest)
    validate_sources(config, manifests)
    entries, groups, provenance = [], [], []
    seen = set()
    for source, directory, manifest in zip(sources, directories, manifests, strict=True):
        admitted = VerifiedReplay(directory, manifest, progress=progress, workers=workers)
        for entry in admitted.entries:
            if entry[2] in seen:
                raise ValueError("same physical rollout copied into multiple sources")
            seen.add(entry[2])
            entries.append(entry)
            groups.append(source["group"])
            provenance.append(
                dict(
                    manifest_sha256=source["manifest_sha256"],
                    source_receipt_sha256=entry[2],
                    schema_version=manifest["binding"]["schema_version"],
                    public_history="paid_three_frame_and_execution_events"
                    if manifest["binding"]["schema_version"] == 2
                    else "current_frame_at_first_visibility; execution_history_missing",
                )
            )

    budget = CacheBudget(cache_dir, cache_max_bytes) if cache_max_bytes is not None else None

    def prepare(entry):
        return prepare_packed_episode(
            entry[0],
            cache_dir,
            binding=entry[1],
            expected_receipt=entry[2],
            raw=True,
            storage_budget=budget,
        )

    packed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for entry in pool.map(prepare, entries):
            packed.append(entry)
            if progress:
                progress(dict(phase="prepare_replay", episodes=len(packed), expected=len(entries)))
        # Required FP32 caches have first claim on the shared byte allowance.
        def compact(entry):
            return compact_packed_episode(entry, cache_dir, storage_budget=budget)

        packed = list(pool.map(compact, packed))
    episodes = AttentionEpisodes(
        packed, ActionSpec(**config["actions"]), config["memory"], cache_size=episode_cache_size
    )
    replay = DecisionReplay(episodes, groups)
    receipt = dict(
        format_version=2,
        sources=sources,
        provenance=provenance,
        converter_sha256=sha256(__file__),
        selector_sha256=sha256(Path(__file__).with_name("memory.py")),
        memory=config["memory"],
        actions=config["actions"],
        episode_decisions=episodes.lengths,
        group_decisions={k: len(v) for k, v in replay.group_indices.items()},
        action_mapping=[episodes.action_spec.from_v1(i) for i in range(8)]
        if any(p["schema_version"] == 1 for p in provenance)
        else None,
    )
    target = Path(cache_dir) / "conversions" / f"{identity_hash(receipt)}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(target, receipt)
    return replay, receipt
