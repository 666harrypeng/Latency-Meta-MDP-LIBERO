"""Immutable indexed nominal JEPA futures shared by all policy supervision delays."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np

from latency_meta_mdp.expert_realization.artifacts import _hash_file

_BINDINGS = {
    "checkpoint_sha256",
    "source_manifest_sha256",
    "split_manifest_sha256",
    "vision_cache_manifest_sha256",
    "normalization_sha256",
    "temporal_config_sha256",
    "model_config_sha256",
}


def _check_bindings(bindings):
    if set(bindings) != _BINDINGS or any(
        not isinstance(x, str) or not re.fullmatch(r"[a-f0-9]{64}", x) for x in bindings.values()
    ):
        raise ValueError("nominal prediction cache bindings are invalid")


def write_nominal_return_predictions(
    *,
    records,
    normalization,
    sampling,
    model,
    device,
    batch_size: int,
    output_dir: Path,
    bindings: dict,
) -> Path:
    import torch

    from latency_meta_mdp.belief.action_conditioned_jepa.temporal_view import (
        SharedJepaSampleIndex,
        collate_temporal_jepa_evaluation_samples,
        materialize_temporal_jepa_sample,
    )

    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(target)
    _check_bindings(bindings)
    if getattr(model, "training", False):
        raise ValueError("nominal prediction cache requires an eval-mode frozen predictor")
    if (
        not records
        or any(r.split != "train" for r in records)
        or len({r.level for r in records}) != 1
    ):
        raise ValueError("policy prediction cache requires one level of train records")
    if (
        sampling.native_future_offsets != (4, 8, 12, 16, 20)
        or type(batch_size) is not int
        or batch_size <= 0
    ):
        raise ValueError("policy prediction cache requires stride4 and a positive batch size")
    episodes = []
    indices = []
    by_id = {}
    for record in sorted(records, key=lambda r: r.episode_id):
        if record.episode_id in by_id or record.terminal_tick <= 10:
            raise ValueError("prediction-cache episode inventory is invalid")
        by_id[record.episode_id] = record
        episodes.append(
            {
                "episode_id": record.episode_id,
                "first_index": len(indices),
                "source_start_tick": 10,
                "source_count": record.terminal_tick - 10,
                "master_index": record.logical_master_task_index,
            }
        )
        indices.extend(
            SharedJepaSampleIndex(
                level=record.level,
                split="train",
                episode_id=record.episode_id,
                source_tick=t,
                boundary_disposition="recorded_complete"
                if t + 20 <= record.terminal_tick
                else "certified_absorbing_extension",
            )
            for t in range(10, record.terminal_tick)
        )
    n = len(indices)
    needed = n * (5 * 2 * 196 * 384 * 2 + 5 * 16 * 4)
    target.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(target.parent).free < needed + 1024**3:
        raise OSError("insufficient disk for one shared frozen-prediction cache")
    staging = target.parent / f".{target.name}.building-{os.getpid()}"
    staging.mkdir()
    try:
        visual = np.lib.format.open_memmap(
            staging / "visual.npy", mode="w+", dtype=np.float16, shape=(n, 5, 2, 196, 384)
        )
        proprio = np.lib.format.open_memmap(
            staging / "proprio.npy", mode="w+", dtype=np.float32, shape=(n, 5, 16)
        )
        with torch.inference_mode():
            for start in range(0, n, batch_size):
                selected = indices[start : start + batch_size]
                samples = [
                    materialize_temporal_jepa_sample(
                        record=by_id[index.episode_id],
                        index=index,
                        sampling=sampling,
                        normalization=normalization,
                    )
                    for index in selected
                ]
                batch = collate_temporal_jepa_evaluation_samples(samples).to(device)
                scope = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if device.type == "cuda"
                    else contextlib.nullcontext()
                )
                with scope:
                    predicted = model.rollout_native(batch.context)
                if tuple(predicted.native_delay_ticks.cpu().tolist()) != (4, 8, 12, 16, 20):
                    raise ValueError("predictor returned a different native delay grid")
                visual[start : start + len(selected)] = (
                    predicted.future_visual_latents.cpu().numpy()
                )
                proprio[start : start + len(selected)] = predicted.future_proprio.cpu().numpy()
                if start == 0 or (start // batch_size + 1) % 100 == 0 or start + len(selected) == n:
                    print(f"[policy-return-cache] contexts={start + len(selected)}/{n}", flush=True)
        visual.flush()
        proprio.flush()
        del visual, proprio
        artifacts = {
            name: {"bytes": (staging / name).stat().st_size, "sha256": _hash_file(staging / name)}
            for name in ("visual.npy", "proprio.npy")
        }
        manifest = {
            "format_id": "nominal_policy_return_predictions_v1",
            "complete": True,
            "control_source": "recorded_nominal_expert_prefix_with_certified_absorbing_hold",
            "split": "train",
            "level": records[0].level,
            "bindings": bindings,
            "source_count": n,
            "episodes": episodes,
            "delay_ticks": [4, 8, 12, 16, 20],
            "batch_size": batch_size,
            "autocast": "bfloat16" if device.type == "cuda" else "disabled",
            "artifacts": artifacts,
        }
        with (staging / "manifest.json").open("x") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target / "manifest.json"


class NominalReturnPredictionCache:
    def __init__(self, root: Path, *, expected_bindings: dict):
        self.root = Path(root).resolve()
        self.expected_bindings = dict(expected_bindings)
        self._load(verify_payloads=True)

    def _load(self, *, verify_payloads):
        _check_bindings(self.expected_bindings)
        raw = (self.root / "manifest.json").read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if hasattr(self, "manifest_sha256") and digest != self.manifest_sha256:
            raise ValueError("prediction cache manifest changed while spawning a worker")
        self.manifest_sha256 = digest
        self.manifest = json.loads(raw)
        m = self.manifest
        if (
            m.get("format_id") != "nominal_policy_return_predictions_v1"
            or m.get("complete") is not True
            or m.get("split") != "train"
            or m.get("bindings") != self.expected_bindings
            or m.get("delay_ticks") != [4, 8, 12, 16, 20]
            or m.get("control_source")
            != "recorded_nominal_expert_prefix_with_certified_absorbing_hold"
        ):
            raise ValueError("prediction cache semantics or bindings disagree")
        self.episodes = {}
        expected_start = 0
        for row in m["episodes"]:
            if (
                row["episode_id"] in self.episodes
                or row["first_index"] != expected_start
                or row["source_start_tick"] != 10
                or row["source_count"] <= 0
            ):
                raise ValueError("prediction cache episode index is invalid")
            self.episodes[row["episode_id"]] = row
            expected_start += row["source_count"]
        if expected_start != m["source_count"] or set(m["artifacts"]) != {
            "visual.npy",
            "proprio.npy",
        }:
            raise ValueError("prediction cache inventory is invalid")
        for name, item in m["artifacts"].items():
            path = self.root / name
            if path.stat().st_size != item["bytes"] or (
                verify_payloads and _hash_file(path) != item["sha256"]
            ):
                raise ValueError("prediction cache artifact verification failed")
        self.visual = np.load(self.root / "visual.npy", mmap_mode="r", allow_pickle=False)
        self.proprio = np.load(self.root / "proprio.npy", mmap_mode="r", allow_pickle=False)
        if (
            self.visual.shape != (expected_start, 5, 2, 196, 384)
            or self.visual.dtype != np.float16
            or self.proprio.shape != (expected_start, 5, 16)
            or self.proprio.dtype != np.float32
        ):
            raise ValueError("prediction cache array contract is invalid")

    def __getstate__(self):
        return {
            "root": self.root,
            "expected_bindings": self.expected_bindings,
            "manifest_sha256": self.manifest_sha256,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Parent verified payload hashes. Workers reopen the same immutable files.
        self._load(verify_payloads=False)

    def read(self, episode_id: str, source_tick: int) -> dict:
        row = self.episodes[episode_id]
        offset = source_tick - row["source_start_tick"]
        if type(source_tick) is not int or not 0 <= offset < row["source_count"]:
            raise ValueError("source tick is absent from the nominal prediction cache")
        index = row["first_index"] + offset
        visual, proprio = self.visual[index], self.proprio[index]
        if not np.isfinite(visual).all() or not np.isfinite(proprio).all():
            raise ValueError("frozen prediction contains non-finite values")
        return {"visual": visual, "proprio": proprio}
