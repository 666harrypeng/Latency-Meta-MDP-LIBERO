"""Verified RGB targets and grouped decoder-only holdout over the existing train pool."""

from __future__ import annotations

import io
import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from latency_meta_mdp.io.artifacts import sha256_file


def partition_decoder_masters(masters, *, holdout_count: int, seed: int):
    masters = tuple(sorted(masters))
    if len(set(masters)) != len(masters) or not 0 < holdout_count < len(masters):
        raise ValueError("decoder holdout must partition unique training masters")
    order = np.random.default_rng(seed).permutation(masters)
    return tuple(sorted(map(int, order[holdout_count:]))), tuple(
        sorted(map(int, order[:holdout_count]))
    )


def prepare_visual_decoder_data(*, project_root: Path, output: Path, seed: int = 20260909):
    from latency_meta_mdp.belief.jepa.config import (
        load_action_conditioned_jepa_config,
    )
    from latency_meta_mdp.belief.jepa.corpus import (
        load_verified_jepa_inputs,
    )
    from latency_meta_mdp.data.source.schema import SourceFieldRole

    root = project_root.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    source_id = "panda-ball-structured-source-quota-formal-100x4-v1"
    source = root / "outputs/source_corpus" / source_id
    cache = root / "outputs/derived/vision_features/dinov3-vits16-structured-source-100x4-v1"
    split = (
        root
        / "outputs/derived/source_splits"
        / source_id
        / "train80-validation20-seed20260903-v1.json"
    )
    config = load_action_conditioned_jepa_config(
        model_path=root / "configs/models/jepa/model.yaml",
        level_path=root / "configs/models/jepa/l3.yaml",
        temporal_sampling_path=root / "configs/models/jepa/stride4_80ms_history_160ms.yaml",
    )
    inputs = load_verified_jepa_inputs(
        source_root=source,
        cache_run_manifest=cache / "manifest.json",
        split_manifest_path=split,
        config=config,
    )
    fit, heldout = partition_decoder_masters(
        inputs.split.train_master_task_indices, holdout_count=16, seed=seed
    )
    admitted = sorted(set(inputs.split.train_episode_ids) & set(inputs.source.episode_ids(level=3)))
    staging = output.parent / f".{output.name}.building-{os.getpid()}"
    staging.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, episode_id in enumerate(admitted):
        meta = inputs.source.episode_metadata(episode_id)
        row = inputs.cache_rows_by_episode[episode_id]
        table = inputs.source.read_fields(
            episode_id,
            fields=("formal_tick", "agentview_rgb", "wrist_rgb", "phase_id"),
            allowed_roles=frozenset(
                {
                    SourceFieldRole.IDENTITY,
                    SourceFieldRole.DEPLOYMENT_INPUT,
                    SourceFieldRole.AUDIT_ONLY,
                }
            ),
        ).to_pylist()
        count = row["boundary_count"]
        if [x["formal_tick"] for x in table] != list(range(count)):
            raise ValueError("RGB and latent source boundaries disagree")
        target = staging / f"{episode_id}.npy"
        rgb = np.lib.format.open_memmap(
            target, mode="w+", dtype=np.uint8, shape=(count, 2, 3, 224, 224)
        )
        for tick, item in enumerate(table):
            for view, field in enumerate(("agentview_rgb", "wrist_rgb")):
                with Image.open(io.BytesIO(item[field]["bytes"])) as image:
                    image = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
                    rgb[tick, view] = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)
        rgb.flush()
        del rgb
        feature = cache / Path(row["cache_manifest"]).parent / "features.npy"
        artifact = inputs.cache_manifest["artifacts"][str(feature.relative_to(cache))]
        master = meta["logical_master_task_index"]
        rows.append(
            {
                "episode_id": episode_id,
                "master_index": master,
                "partition": "holdout" if master in heldout else "fit",
                "boundary_count": count,
                "phases": [x["phase_id"] for x in table],
                "feature_path": str(feature.relative_to(root)),
                "feature_sha256": artifact["sha256"],
                "feature_bytes": artifact["bytes"],
                "rgb_path": str((output / target.name).relative_to(root)),
                "rgb_sha256": sha256_file(target),
                "rgb_bytes": target.stat().st_size,
            }
        )
        if index % 20 == 0 or index + 1 == len(admitted):
            print(f"RGB target cache {index + 1}/{len(admitted)}", flush=True)
    manifest = {
        "format_id": "frozen_visual_decoder_rgb_targets_v1",
        "complete": True,
        "source_manifest_sha256": inputs.source_manifest_sha256,
        "vision_manifest_sha256": inputs.cache_manifest_sha256,
        "split_manifest_sha256": inputs.split_manifest_sha256,
        "fit_master_indices": fit,
        "holdout_master_indices": heldout,
        "split_seed": seed,
        "rgb_preprocessing": "PIL RGB bilinear224, before DINO mean/std; uint8 CHW",
        "pillow_version": Image.__version__,
        "episodes": rows,
        "boundary_count": sum(x["boundary_count"] for x in rows),
        "scope": "L3 original train pool; decoder-only grouped holdout, not unseen JEPA test",
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    staging.rename(output)
    return manifest


def verify_visual_decoder_data(manifest_path: Path, *, project_root: Path):
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("format_id") != "frozen_visual_decoder_rgb_targets_v1"
        or not manifest["complete"]
    ):
        raise ValueError("decoder cache must be complete")
    fit, heldout = set(manifest["fit_master_indices"]), set(manifest["holdout_master_indices"])
    if len(fit) != 64 or len(heldout) != 16 or fit & heldout:
        raise ValueError("decoder master partition is invalid")
    if len(manifest["episodes"]) != 320:
        raise ValueError("decoder targets must cover all320 L3 training episodes")
    seen = set()
    for row in manifest["episodes"]:
        if row["partition"] not in {"fit", "holdout"}:
            raise ValueError("unknown decoder episode partition")
        if row["episode_id"] in seen:
            raise ValueError("duplicate decoder episode")
        seen.add(row["episode_id"])
        allowed = fit if row["partition"] == "fit" else heldout
        if row["master_index"] not in allowed:
            raise ValueError("decoder episode crosses master partition")
        for kind in ("rgb", "feature"):
            p = project_root / row[f"{kind}_path"]
            if p.stat().st_size != row[f"{kind}_bytes"] or sha256_file(p) != row[f"{kind}_sha256"]:
                raise ValueError(f"decoder {kind} payload changed: {p}")
    return manifest


class VisualDecoderDataset(torch.utils.data.Dataset):
    """Lazy per-episode RGB/feature mappings, with a bounded per-worker open set."""

    def __init__(self, manifest_path: Path, *, project_root: Path, partition: str):
        if partition not in {"fit", "holdout"}:
            raise ValueError("unknown decoder partition")
        self.root = project_root
        manifest = json.loads(manifest_path.read_text())
        if not manifest["complete"]:
            raise ValueError("decoder targets are incomplete")
        self.rows = [x for x in manifest["episodes"] if x["partition"] == partition]
        self.indices = [
            (i, t) for i, row in enumerate(self.rows) for t in range(row["boundary_count"])
        ]
        self._maps = OrderedDict()

    def __len__(self):
        return len(self.indices)

    def __getstate__(self):
        return {**self.__dict__, "_maps": OrderedDict()}

    def _episode(self, index):
        if index not in self._maps:
            row = self.rows[index]
            z = np.load(self.root / row["feature_path"], mmap_mode="r", allow_pickle=False)
            rgb = np.load(self.root / row["rgb_path"], mmap_mode="r", allow_pickle=False)
            n = row["boundary_count"]
            if (
                z.shape != (n, 2, 196, 384)
                or z.dtype != np.float16
                or rgb.shape != (n, 2, 3, 224, 224)
                or rgb.dtype != np.uint8
            ):
                raise ValueError("RGB/latent decoder cache shape or dtype disagrees")
            self._maps[index] = (z, rgb)
            if len(self._maps) > 8:
                _, expired = self._maps.popitem(last=False)
                for value in expired:
                    value._mmap.close()
        self._maps.move_to_end(index)
        return self._maps[index]

    def __getitem__(self, index):
        episode, tick = self.indices[index]
        z, rgb = self._episode(episode)
        return torch.from_numpy(np.array(z[tick], copy=True)), torch.from_numpy(
            np.array(rgb[tick], copy=True)
        )
