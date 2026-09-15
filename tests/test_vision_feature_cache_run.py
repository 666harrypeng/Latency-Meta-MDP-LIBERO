from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest

from latency_meta_mdp.artifacts import ImplementationProvenance, sha256_file
from latency_meta_mdp.expert_realization.source_corpus.parquet import encode_png
from latency_meta_mdp.vision_encoder import load_vision_encoder_spec


@dataclass(frozen=True)
class _RuntimeInfo:
    torch_version: str = "test-torch"
    transformers_version: str = "test-transformers"
    device: str = "test-device"
    compute_dtype: str = "float16"


class _DeterministicEncoder:
    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.spec = load_vision_encoder_spec(
            Path("configs/vision/dinov3_vits16_lvd1689m_224_v1.yaml")
        )
        self.runtime_info = _RuntimeInfo()
        self.fail_on_call = fail_on_call
        self.call_count = 0

    def encode_numpy(self, images: np.ndarray) -> np.ndarray:
        self.call_count += 1
        if self.call_count == self.fail_on_call:
            raise RuntimeError("injected extraction failure")
        values = images[:, 0, 0, 0].astype(np.float16)
        return np.broadcast_to(
            values[:, None, None],
            (len(images), 196, 384),
        ).copy()


def _image(value: int) -> dict[str, object]:
    rgb = np.full((256, 256, 3), value, dtype=np.uint8)
    return {"bytes": encode_png(rgb, compress_level=1), "path": f"frame-{value}.png"}


def _episode(*, level: int, master: int, slot: int, boundary_count: int = 2):
    episode_id = f"source-L{level}-task{master:06d}-s{slot:02d}-d{slot:04d}"
    metadata = {
        "episode_id": episode_id,
        "task_instance_id": f"task-instance-{master}-level-{level}",
        "logical_master_task_index": master,
        "level": level,
        "accepted_slot": slot,
        "realization_draw_index": slot,
        "row_count": boundary_count,
    }
    frames = pa.table(
        {
            "agentview_rgb": [_image(10 + tick) for tick in range(boundary_count)],
            "wrist_rgb": [_image(100 + tick) for tick in range(boundary_count)],
        }
    )
    return SimpleNamespace(metadata=MappingProxyType(metadata), frames=frames)


class _Corpus:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir()
        (self.root / "manifest.json").write_bytes(b'{"canonical":true}\n')
        episodes = (
            _episode(level=1, master=10, slot=0),
            _episode(level=1, master=11, slot=0),
            _episode(level=2, master=10, slot=0),
            _episode(level=2, master=11, slot=0),
        )
        self._episodes = {episode.metadata["episode_id"]: episode for episode in episodes}
        self.manifest = SimpleNamespace(corpus_id="structured-source-demo")

    def episode_ids(self, *, level: int) -> tuple[str, ...]:
        return tuple(
            sorted(
                episode_id
                for episode_id, episode in self._episodes.items()
                if episode.metadata["level"] == level
            )
        )

    def read_episode(self, episode_id: str):
        return self._episodes[episode_id]


def _clean_provenance(_root: Path) -> ImplementationProvenance:
    return ImplementationProvenance(
        revision="1" * 40,
        source_sha256="2" * 64,
        dirty=False,
    )


def test_structured_source_cache_run_round_trips_exact_inventory(tmp_path: Path) -> None:
    from latency_meta_mdp.vision_feature_cache_run import (
        load_verified_vision_feature_cache_run,
        write_vision_feature_cache_run,
    )

    corpus = _Corpus(tmp_path / "source")
    encoder = _DeterministicEncoder()
    output = tmp_path / "cache"

    result = write_vision_feature_cache_run(
        project_root=Path.cwd(),
        source_root=corpus.root,
        encoder=encoder,
        output_dir=output,
        levels=(1, 2),
        episode_range=(0, 1),
        boundary_batch_size=2,
        load_fn=lambda _root: corpus,
        provenance_fn=_clean_provenance,
    )
    manifest = load_verified_vision_feature_cache_run(
        output,
        expected_source_manifest_sha256=sha256_file(corpus.root / "manifest.json"),
        expected_spec=encoder.spec,
    )

    assert result == output / "manifest.json"
    assert manifest["schema_version"] == 2
    assert manifest["format_id"] == "vision_feature_cache_run_v2"
    assert manifest["eligible"] is True
    assert manifest["blockers"] == []
    assert manifest["implementation"] == {
        "revision": "1" * 40,
        "source_sha256": "2" * 64,
        "dirty": False,
    }
    assert manifest["source_corpus_id"] == "structured-source-demo"
    assert manifest["levels"] == [1, 2]
    assert manifest["episode_range"] == [0, 1]
    assert manifest["episode_count"] == 2
    assert manifest["boundary_count"] == 4
    assert manifest["feature_payload_bytes"] == 4 * 2 * 196 * 384 * 2
    assert manifest["feature_artifact_bytes"] == manifest["feature_payload_bytes"] + 256
    assert [row["level"] for row in manifest["episodes"]] == [1, 2]
    assert all(row["accepted_slot"] == 0 for row in manifest["episodes"])
    assert set(manifest["artifacts"]) == {
        relative
        for row in manifest["episodes"]
        for relative in (
            row["cache_manifest"],
            row["cache_manifest"].replace("manifest.json", "features.npy"),
        )
    }


def test_cache_run_rejects_overwrite_and_cleans_failed_staging(tmp_path: Path) -> None:
    from latency_meta_mdp.vision_feature_cache_run import write_vision_feature_cache_run

    corpus = _Corpus(tmp_path / "source")
    output = tmp_path / "cache"
    kwargs = {
        "project_root": Path.cwd(),
        "source_root": corpus.root,
        "output_dir": output,
        "levels": (1,),
        "episode_range": (0, 2),
        "boundary_batch_size": 2,
        "load_fn": lambda _root: corpus,
        "provenance_fn": _clean_provenance,
    }

    with pytest.raises(RuntimeError, match="injected extraction failure"):
        write_vision_feature_cache_run(
            encoder=_DeterministicEncoder(fail_on_call=2),
            **kwargs,
        )
    assert not output.exists()
    assert list(tmp_path.glob(".cache.building-*")) == []

    write_vision_feature_cache_run(encoder=_DeterministicEncoder(), **kwargs)
    with pytest.raises(FileExistsError, match="already exists"):
        write_vision_feature_cache_run(encoder=_DeterministicEncoder(), **kwargs)


def test_cache_run_loader_rejects_child_tamper(tmp_path: Path) -> None:
    from latency_meta_mdp.vision_feature_cache_run import (
        load_verified_vision_feature_cache_run,
        write_vision_feature_cache_run,
    )

    corpus = _Corpus(tmp_path / "source")
    output = tmp_path / "cache"
    write_vision_feature_cache_run(
        project_root=Path.cwd(),
        source_root=corpus.root,
        encoder=_DeterministicEncoder(),
        output_dir=output,
        levels=(1,),
        episode_range=(0, 1),
        boundary_batch_size=2,
        load_fn=lambda _root: corpus,
        provenance_fn=_clean_provenance,
    )
    run = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    feature = output / run["episodes"][0]["cache_manifest"].replace("manifest.json", "features.npy")
    with feature.open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"x")

    with pytest.raises(ValueError, match="artifact verification"):
        load_verified_vision_feature_cache_run(output)


def test_cache_cli_exposes_structured_source_selection(tmp_path: Path, capsys) -> None:
    from latency_meta_mdp.cli.cache_vision_features import main

    source = tmp_path / "source"
    output = tmp_path / "cache"
    captured: dict[str, object] = {}
    encoder = _DeterministicEncoder()

    def encoder_factory(*, spec, device: str, local_files_only: bool):
        captured["encoder"] = (spec, device, local_files_only)
        return encoder

    def writer_fn(**kwargs):
        captured["writer"] = kwargs
        return output / "manifest.json"

    assert (
        main(
            [
                "--source-root",
                str(source),
                "--output-dir",
                str(output),
                "--levels",
                "1",
                "3",
                "--episode-range",
                "4",
                "9",
                "--boundary-batch-size",
                "8",
                "--device",
                "cuda:0",
            ],
            encoder_factory=encoder_factory,
            writer_fn=writer_fn,
        )
        == 0
    )
    terminal = capsys.readouterr()
    assert terminal.out == f"{output / 'manifest.json'}\n"
    assert terminal.err == ""
    assert captured["encoder"][1:] == ("cuda:0", True)
    assert captured["writer"]["source_root"] == source
    assert captured["writer"]["levels"] == (1, 3)
    assert captured["writer"]["episode_range"] == (4, 9)
    assert captured["writer"]["boundary_batch_size"] == 8


def test_cache_can_select_explicit_training_episode_inventory(tmp_path):
    from latency_meta_mdp.vision_feature_cache_run import (
        load_verified_vision_feature_cache_run,
        write_vision_feature_cache_run,
    )

    corpus = _Corpus(tmp_path / "source")
    selected = (corpus.episode_ids(level=2)[1],)
    output = tmp_path / "selected"
    path = write_vision_feature_cache_run(
        project_root=tmp_path,
        source_root=corpus.root,
        encoder=_DeterministicEncoder(),
        output_dir=output,
        levels=(2,),
        episode_range=None,
        boundary_batch_size=2,
        load_fn=lambda p: corpus,
        provenance_fn=_clean_provenance,
        selected_episode_ids=selected,
    )
    manifest = load_verified_vision_feature_cache_run(path.parent)
    assert [row["episode_id"] for row in manifest["episodes"]] == list(selected)
