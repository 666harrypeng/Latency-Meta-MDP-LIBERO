"""Matched nominal-source D20 policy views with explicit privileged oracle separation.

All delays are enumerated virtually; images and frozen features are not replicated.
The main actor never receives the supervision delay. CUDA prediction belongs in an
offline cache writer, not in DataLoader workers. These recorded GT futures are only
valid under the recorded nominal executable controls, not arbitrary policy buffers.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np

_ANCHORS = np.array([4, 8, 12, 16, 20], dtype=np.int64)
_START_TICK = 10


class UniformDelaySampler:
    """Balance valid D20 labels in shuffled blocks, covering each delay's index pool.

    Each epoch visits every valid pair at least once. Shorter delay pools are
    repeated only as needed to equalize counts. No episode-PMF weighting is used.
    """

    def __init__(self, indices_by_delay, *, seed: int, batch_size: int = 20, start_batch: int = 0):
        if type(seed) is not int or seed < 0:
            raise ValueError("sampler seed must be a nonnegative integer")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("sampler batch_size must be a positive integer")
        if type(start_batch) is not int or start_batch < 0:
            raise ValueError("sampler start_batch must be a nonnegative integer")
        self.indices = tuple(np.asarray(x, dtype=np.int64) for x in indices_by_delay)
        if len(self.indices) != 20 or any(x.ndim != 1 or not len(x) for x in self.indices):
            raise ValueError("balanced sampling requires real action labels at every D20 delay")
        multiple = batch_size // gcd(batch_size, 20)
        self.per_delay_count = ((max(map(len, self.indices)) + multiple - 1) // multiple) * multiple
        self.seed = seed
        self.epoch, self.offset = divmod(start_batch * batch_size, 20 * self.per_delay_count)

    def __len__(self):
        return 20 * self.per_delay_count - self.offset

    def __iter__(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        self.epoch += 1
        offset, self.offset = self.offset, 0
        columns = []
        for indices in self.indices:
            parts = []
            remaining = self.per_delay_count
            while remaining:
                part = rng.permutation(indices)[:remaining]
                parts.append(part)
                remaining -= len(part)
            columns.append(np.concatenate(parts))
        # Every consecutive block has one sample of each delay, in random order.
        ordered = rng.permuted(np.column_stack(columns), axis=1)
        yield from map(int, ordered.ravel()[offset:])


@dataclass(frozen=True)
class _Episode:
    episode_id: str
    master_index: int
    level: int
    terminal_tick: int
    features_path: str
    features_offset: int
    proprio: np.ndarray
    controls: np.ndarray
    phases: tuple[str | None, ...]
    base_offset: int


class MatchedReturnPolicyDataset:
    """Virtual return labels: legacy PMF-weighted or balanced prefix post-training."""

    def __init__(
        self,
        *,
        native_dataset,
        records,
        episode_rows,
        episode_probabilities,
        mode: str,
        predictions=None,
        conditioning: str = "late",
    ):
        import torch

        from latency_meta_mdp.belief.action_conditioned_jepa.config import (
            load_jepa_temporal_sampling,
        )
        from latency_meta_mdp.belief.action_conditioned_jepa.data_adapter import JepaEpisodeRecord
        from latency_meta_mdp.belief.action_conditioned_jepa.latent_return import (
            quantize_d20_probabilities,
        )

        if conditioning not in {"late", "prefix"}:
            raise ValueError("return-policy conditioning must be late or prefix")
        if mode not in {
            "predicted_mixture",
            "gt_mixture",
            "known_delay_oracle",
            "no_future_control",
        }:
            raise ValueError("unsupported return-policy view mode")
        if not records or any(
            not isinstance(r, JepaEpisodeRecord) or r.split != "train" for r in records
        ):
            raise ValueError("return-policy development requires verified train records")
        if len({r.level for r in records}) != 1 or len({r.episode_id for r in records}) != len(
            records
        ):
            raise ValueError("return-policy records must have unique IDs and one level")
        if mode == "predicted_mixture" and not callable(getattr(predictions, "read", None)):
            raise ValueError("predicted mixture requires a frozen nominal prediction lookup")
        offsets = {}
        offset = 0
        for row in episode_rows:
            if row["episode_id"] in offsets or row["frame_count"] <= 0:
                raise ValueError("native episode inventory is invalid")
            offsets[row["episode_id"]] = (offset, row)
            offset += row["frame_count"]
        if len(native_dataset) != offset:
            raise ValueError("native dataset length differs from episode inventory")
        self.native_dataset = native_dataset
        self.mode = mode
        self.conditioning = conditioning
        self.predictions = predictions
        self.episodes = []
        self.probabilities = []
        self.macro_probabilities = []
        self.contexts = []
        self._features = {}
        sampling = load_jepa_temporal_sampling(
            Path(__file__).resolve().parents[2]
            / "configs/belief/action_conditioned_jepa/stride4_80ms_history_160ms.yaml"
        )
        for record in records:
            base_offset, row = offsets[record.episode_id]
            if (
                row["frame_count"] != record.terminal_tick
                or row["level"] != record.level
                or row["logical_master_task_index"] != record.logical_master_task_index
            ):
                raise ValueError("native/JEPA episode identities do not match")
            pmf = np.asarray(episode_probabilities[record.episode_id], dtype=np.float64)
            if (
                pmf.shape != (20,)
                or not np.isfinite(pmf).all()
                or np.any(pmf < 0)
                or not np.isclose(pmf.sum(), 1, atol=1e-6, rtol=0)
            ):
                raise ValueError("episode probabilities must be a normalized D20 law")
            _, macro = quantize_d20_probabilities(
                probabilities=torch.tensor(pmf[None], dtype=torch.float32), sampling=sampling
            )
            features = record.cache.features
            if not isinstance(features, np.memmap) or not features.flags.c_contiguous:
                raise ValueError("source features must stay contiguous memory mappings")
            index = len(self.episodes)
            self.episodes.append(
                _Episode(
                    record.episode_id,
                    record.logical_master_task_index,
                    record.level,
                    record.terminal_tick,
                    str(features.filename),
                    features.offset,
                    record.proprio_physical,
                    record.controls,
                    record.phases,
                    base_offset,
                )
            )
            self.probabilities.append(pmf.copy())
            self.macro_probabilities.append(macro[0].numpy())
            self.contexts.extend((index, tick) for tick in range(_START_TICK, record.terminal_tick))
        if not self.contexts:
            raise ValueError("return-policy data has no real history-ready source")

    def __getstate__(self):
        # Spawned workers reopen mappings instead of pickling tens of GiB.
        return {**self.__dict__, "_features": {}}

    def __len__(self):
        return len(self.contexts) * 20

    def training_sampler(self, *, seed: int = 0, batch_size: int = 20, start_batch: int = 0):
        if self.conditioning != "prefix":
            return None
        indices = [
            np.array(
                [
                    i * 20 + delay - 1
                    for i, (episode, tick) in enumerate(self.contexts)
                    if tick + delay < self.episodes[episode].terminal_tick
                ],
                dtype=np.int64,
            )
            for delay in range(1, 21)
        ]
        return UniformDelaySampler(
            indices, seed=seed, batch_size=batch_size, start_batch=start_batch
        )

    def _loss_weight(self, episode_index, delay):
        if self.conditioning == "prefix":
            return 1.0
        return float(20 * self.probabilities[episode_index][delay - 1])

    def _coordinates(self, index):
        if (
            not isinstance(index, (int, np.integer))
            or isinstance(index, bool)
            or not 0 <= index < len(self)
        ):
            raise IndexError("return-policy sample index is out of range")
        context, delay_index = divmod(int(index), 20)
        episode_index, source_tick = self.contexts[context]
        return episode_index, source_tick, delay_index + 1

    def sample_identity(self, index) -> dict[str, Any]:
        episode_index, source_tick, delay = self._coordinates(index)
        episode = self.episodes[episode_index]
        return {
            "episode_id": episode.episode_id,
            "master_index": episode.master_index,
            "source_tick": source_tick,
            "source_phase": episode.phases[source_tick],
            "supervision_delay_ticks": delay,
            "loss_weight": self._loss_weight(episode_index, delay),
            "nominal_recorded_controls_only": True,
            "privileged_oracle": self.mode == "known_delay_oracle",
        }

    def _ground_truth(self, episode_index, ticks):
        episode = self.episodes[episode_index]
        if episode_index not in self._features:
            self._features[episode_index] = np.memmap(
                episode.features_path,
                mode="r",
                dtype=np.float16,
                shape=(episode.terminal_tick + 1, 2, 196, 384),
                offset=episode.features_offset,
            )
        rows = np.minimum(ticks, episode.terminal_tick)
        visual = np.asarray(self._features[episode_index][rows])
        proprio = np.array(episode.proprio[rows], copy=True)
        absorbing = ticks > episode.terminal_tick
        proprio[absorbing, 7:14] = 0
        proprio[absorbing, 15] = 0
        return visual, proprio

    def __getitem__(self, index):
        episode_index, source_tick, delay = self._coordinates(index)
        episode = self.episodes[episode_index]
        native = self.native_dataset[episode.base_offset + source_tick]
        if not np.allclose(
            np.asarray(native["state"]), episode.proprio[source_tick], atol=1e-6, rtol=0
        ):
            raise ValueError("native observation and JEPA source boundary are misaligned")
        result = {key: native[key] for key in ("image", "wrist_image", "state", "prompt")}
        target_start = source_tick + delay
        count = max(0, min(50, episode.terminal_tick - target_start))
        target = np.zeros((50, 7), np.float32)
        target[:count] = episode.controls[target_start : target_start + count]
        result.update(
            actions=target,
            actions_is_pad=np.arange(50) >= count,
            action_loss_weight=np.float32(self._loss_weight(episode_index, delay)),
        )
        if self.mode == "known_delay_oracle" and self.conditioning == "late":
            visual, proprio = self._ground_truth(episode_index, np.array([target_start]))
            result["known_delay_oracle"] = {
                "visual": visual[0],
                "proprio": proprio[0],
                "known_delay_ticks": delay,
            }
        else:
            if self.mode == "no_future_control":
                visual = np.zeros((5, 2, 196, 384), np.float16)
                proprio = np.zeros((5, 16), np.float32)
            elif self.mode in {"gt_mixture", "known_delay_oracle"}:
                visual, proprio = self._ground_truth(episode_index, source_tick + _ANCHORS)
            else:
                predicted = self.predictions.read(episode.episode_id, source_tick)
                if set(predicted) != {"visual", "proprio"}:
                    raise ValueError("prediction lookup must provide only frozen physical futures")
                visual, proprio = predicted["visual"], predicted["proprio"]
            result["return_belief"] = {
                "visual": visual,
                "proprio": proprio,
                "delay_ticks": _ANCHORS.copy(),
                "probabilities": self.macro_probabilities[episode_index].copy(),
            }
            if self.conditioning == "prefix":
                result["return_belief"]["latency_probabilities"] = self.probabilities[
                    episode_index
                ].astype(np.float32)
                if self.mode == "known_delay_oracle":
                    result["known_delay_oracle"] = {"known_delay_ticks": delay}
        return result


def load_matched_return_policy_dataset(native_dataset, spec: dict) -> MatchedReturnPolicyDataset:
    """OpenPI loader hook: verify source/export/cache joins before constructing a view."""
    import json

    from latency_meta_mdp.belief.action_conditioned_jepa.config import (
        load_action_conditioned_jepa_config,
    )
    from latency_meta_mdp.belief.action_conditioned_jepa.data_adapter import (
        load_verified_jepa_inputs,
        load_verified_jepa_record,
    )
    from latency_meta_mdp.expert_realization.artifacts import _hash_file
    from latency_meta_mdp.latency_law_family import load_episode_latency_law_family
    from latency_meta_mdp.policy_return_cache import NominalReturnPredictionCache

    required = {
        "source_root",
        "vision_cache_manifest",
        "split_manifest",
        "policy_export_manifest",
        "latency_family",
        "level",
        "mode",
        "belief_decision",
        "prediction_cache_root",
    }
    if isinstance(spec, dict) and "conditioning" in spec:
        required.add("conditioning")
    if (
        not isinstance(spec, dict)
        or set(spec) != required
        or type(spec["level"]) is not int
        or spec["level"] not in (1, 2, 3)
        or spec.get("conditioning", "late") not in {"late", "prefix"}
    ):
        raise ValueError("return-policy view specification is invalid")
    root = Path(__file__).resolve().parents[2]
    level = spec["level"]
    temporal = root / "configs/belief/action_conditioned_jepa/stride4_80ms_history_160ms.yaml"
    config = load_action_conditioned_jepa_config(
        model_path=root / "configs/belief/action_conditioned_jepa/model.yaml",
        level_path=root / f"configs/belief/action_conditioned_jepa/l{level}.yaml",
        temporal_sampling_path=temporal,
    )
    inputs = load_verified_jepa_inputs(
        source_root=Path(spec["source_root"]),
        cache_run_manifest=Path(spec["vision_cache_manifest"]),
        split_manifest_path=Path(spec["split_manifest"]),
        config=config,
    )
    export_path = Path(spec["policy_export_manifest"]).resolve()
    export = json.loads(export_path.read_text())
    if (
        export.get("format_id") != "metamdp_lerobot_structured_train_v1"
        or export.get("split") != "train"
        or export.get("source_manifest_sha256") != inputs.source_manifest_sha256
        or export.get("split_manifest_sha256") != inputs.split_manifest_sha256
    ):
        raise ValueError("policy export is not the matched structured train source")
    selected = [row for row in export["datasets"] if row["level"] == level]
    if len(selected) != 1:
        raise ValueError("policy export level inventory is invalid")
    row = selected[0]
    nested_path = (export_path.parent / row["dataset_manifest"]).resolve()
    if (
        not nested_path.is_relative_to(export_path.parent)
        or _hash_file(nested_path) != row["dataset_manifest_sha256"]
    ):
        raise ValueError("policy dataset manifest binding is invalid")
    native = json.loads(nested_path.read_text())
    ids = tuple(
        sorted(set(inputs.split.train_episode_ids) & set(inputs.source.episode_ids(level=level)))
    )
    if {row["episode_id"] for row in native["episodes"]} != set(ids):
        raise ValueError("native dataset does not contain the exact grouped train inventory")
    records = tuple(
        load_verified_jepa_record(inputs, episode_id=episode_id, level=level, split="train")
        for episode_id in ids
    )
    family = load_episode_latency_law_family(Path(spec["latency_family"]))
    probabilities = {}
    for episode_id in ids:
        metadata = inputs.source.episode_metadata(episode_id)
        key = 4 * metadata["logical_master_task_index"] + metadata["accepted_slot"]
        probabilities[episode_id] = family.sample_for_key(assignment_key=key).probabilities
    predictions = None
    if spec["mode"] == "predicted_mixture":
        decision = json.loads(Path(spec["belief_decision"]).read_text())
        if (
            decision.get("level") != level
            or decision.get("status") != "admitted_for_policy_integration"
        ):
            raise ValueError("predicted policy view requires the level's admitted Belief decision")
        bindings = {
            "checkpoint_sha256": decision["canonical_checkpoint_sha256"],
            "source_manifest_sha256": inputs.source_manifest_sha256,
            "split_manifest_sha256": inputs.split_manifest_sha256,
            "vision_cache_manifest_sha256": inputs.cache_manifest_sha256,
            "normalization_sha256": _hash_file(root / decision["normalization"]),
            "temporal_config_sha256": _hash_file(temporal),
            "model_config_sha256": _hash_file(
                root / "configs/belief/action_conditioned_jepa/model.yaml"
            ),
        }
        predictions = NominalReturnPredictionCache(
            Path(spec["prediction_cache_root"]), expected_bindings=bindings
        )
    result = MatchedReturnPolicyDataset(
        native_dataset=native_dataset,
        records=records,
        episode_rows=tuple(native["episodes"]),
        episode_probabilities=probabilities,
        mode=spec["mode"],
        predictions=predictions,
        conditioning=spec.get("conditioning", "late"),
    )
    result.provenance = {
        "source_manifest_sha256": inputs.source_manifest_sha256,
        "split_manifest_sha256": inputs.split_manifest_sha256,
        "export_manifest_sha256": _hash_file(export_path),
        "latency_family_sha256": _hash_file(Path(spec["latency_family"])),
        "control_source": "recorded_nominal_only",
    }
    if result.conditioning == "prefix":
        valid_counts = [
            sum(max(episode.terminal_tick - _START_TICK - delay, 0) for episode in result.episodes)
            for delay in range(1, 21)
        ]
        result.provenance.update(
            conditioning="prefix",
            training_delay_objective="uniform_d20_nonempty_pairs",
            action_loss_weight=1.0,
            valid_pairs_per_delay=valid_counts,
            balanced_sampler_minimum_epoch_examples=20 * max(valid_counts),
        )
    return result
