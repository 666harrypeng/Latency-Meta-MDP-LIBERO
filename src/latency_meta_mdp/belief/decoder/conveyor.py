"""Decoder adaptation views over existing conveyor PNG and frozen DINO sources."""

import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from latency_meta_mdp.data.conveyor.source import ConveyorSource
from latency_meta_mdp.data.conveyor.vision import verify_feature_cache
from latency_meta_mdp.io.artifacts import sha256_file

FORMAT = "conveyor_decoder_data_v1"


def prepare_decoder_data(corpus_manifest, feature_manifest, output, *, project_root, stride=5):
    if type(stride) is not int or stride < 1:
        raise ValueError("decoder stride must be positive")
    root = Path(project_root).resolve()
    corpus_manifest, feature_manifest = (
        Path(corpus_manifest).resolve(),
        Path(feature_manifest).resolve(),
    )
    cache = verify_feature_cache(feature_manifest, corpus_manifest=corpus_manifest)
    if cache["purpose"] != "training_source":
        raise ValueError("decoder adaptation requires training_source")
    rows = [
        dict(
            episode_id=r["episode_id"],
            seed=r["seed"],
            partition="fit" if r["split"] == "train" else "holdout",
            boundary_count=r["boundary_count"],
        )
        for r in cache["episodes"]
    ]
    if {r["partition"] for r in rows} != {"fit", "holdout"}:
        raise ValueError("decoder adaptation needs train and validation sources")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = output / "manifest.json"
    manifest.write_text(
        json.dumps(
            dict(
                format_id=FORMAT,
                complete=True,
                corpus_manifest=os.path.relpath(corpus_manifest, root),
                feature_manifest=os.path.relpath(feature_manifest, root),
                source_sha256=sha256_file(corpus_manifest),
                cache_sha256=sha256_file(feature_manifest),
                stride=stride,
                episodes=rows,
                rgb_preprocessing="PIL RGB bilinear224, uint8 CHW",
                scope=(
                    "conveyor train fit; validation holdout; test excluded; no copied RGB/features"
                ),
            ),
            indent=2,
        )
    )
    verify_decoder_data(manifest, project_root=root)
    return manifest


def verify_decoder_data(manifest, *, project_root):
    root = Path(project_root)
    data = json.loads(Path(manifest).read_text())
    if (
        data["format_id"] != FORMAT
        or not data["complete"]
        or type(data["stride"]) is not int
        or data["stride"] < 1
    ):
        raise ValueError("invalid conveyor decoder manifest")
    source = root / data["corpus_manifest"]
    features = root / data["feature_manifest"]
    if (
        sha256_file(source) != data["source_sha256"]
        or sha256_file(features) != data["cache_sha256"]
    ):
        raise ValueError("decoder source/cache identity mismatch")
    cache = verify_feature_cache(features, corpus_manifest=source)
    expected = {r["episode_id"]: r for r in cache["episodes"]}
    if len(data["episodes"]) != len(expected) or len(
        {r["episode_id"] for r in data["episodes"]}
    ) != len(expected):
        raise ValueError("decoder episode inventory mismatch")
    if cache["purpose"] != "training_source":
        raise ValueError("decoder requires training_source")
    for row in data["episodes"]:
        reference = expected.get(row["episode_id"])
        if (
            reference is None
            or row["seed"] != reference["seed"]
            or row["boundary_count"] != reference["boundary_count"]
        ):
            raise ValueError("decoder episode identity mismatch")
        partition = "fit" if reference["split"] == "train" else "holdout"
        if row["partition"] != partition:
            raise ValueError("decoder episode crosses train/validation partition")
    return data


class ConveyorDecoderDataset(torch.utils.data.Dataset):
    def __init__(self, manifest, *, project_root, partition):
        if partition not in ("fit", "holdout"):
            raise ValueError("unknown decoder partition")
        root = Path(project_root)
        self.manifest = verify_decoder_data(manifest, project_root=root)
        self.source_manifest = root / self.manifest["corpus_manifest"]
        self.feature_manifest = root / self.manifest["feature_manifest"]
        self.sources = {
            r["episode_id"]: r for r in json.loads(self.source_manifest.read_text())["episodes"]
        }
        self.features = {
            r["episode_id"]: r for r in json.loads(self.feature_manifest.read_text())["episodes"]
        }
        self.rows = [r for r in self.manifest["episodes"] if r["partition"] == partition]
        self.indices = []
        for i, row in enumerate(self.rows):
            ticks = list(range(0, row["boundary_count"], self.manifest["stride"]))
            if ticks[-1] != row["boundary_count"] - 1:
                ticks.append(row["boundary_count"] - 1)
            self.indices.extend((i, t) for t in ticks)
        self._maps = OrderedDict()

    def __len__(self):
        return len(self.indices)

    def __getstate__(self):
        return {**self.__dict__, "_maps": OrderedDict()}

    def __getitem__(self, index):
        episode, tick = self.indices[index]
        if episode not in self._maps:
            key = self.rows[episode]["episode_id"]
            source = ConveyorSource(self.source_manifest.parent / self.sources[key]["source_root"])
            z = np.load(
                self.feature_manifest.parent / self.features[key]["features"], mmap_mode="r"
            )
            self._maps[episode] = (source, z)
            if len(self._maps) > 8:
                _, (expired, array) = self._maps.popitem(last=False)
                array._mmap.close()
                expired.frames.close()
        self._maps.move_to_end(episode)
        source, z = self._maps[episode]
        rgb = np.stack(
            [
                np.asarray(Image.fromarray(im).resize((224, 224), Image.Resampling.BILINEAR))
                for im in source.rgb(tick)
            ]
        ).transpose(0, 3, 1, 2)
        return torch.from_numpy(np.array(z[tick], copy=True)), torch.from_numpy(rgb.copy())
