from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from latency_meta_mdp.io.paths import repository_root

_PROJECT_ROOT = repository_root()
_SOURCE = (
    _PROJECT_ROOT / "outputs/bulk/expert/panda-ball-formal-train-1000-1199-7571a4c/manifest.json"
)
_CACHE = (
    _PROJECT_ROOT
    / "outputs/derived/vision_features/dinov3-vits16-formal-1000-1199-408dbe3/manifest.json"
)
_FLOW_ROOT = _PROJECT_ROOT / "outputs/analysis/flow_belief/dinov3-formal-180-20-f6caf55"
_EVALUATION_ROOT = (
    _PROJECT_ROOT / "outputs/analysis/flow_belief_evaluation/dinov3-formal-180-20-f6caf55"
)


def _require_formal_artifacts() -> None:
    for path in (
        _SOURCE,
        _CACHE,
        _FLOW_ROOT / "manifest.json",
        _EVALUATION_ROOT / "manifest.json",
    ):
        if not path.is_file():
            pytest.skip("quality sample integration requires formal Flow artifacts")


def _load_l1_inputs():
    from latency_meta_mdp.data.vision.contracts import load_vision_encoder_spec
    from latency_meta_mdp.legacy.belief.common.feature_corpus import (
        load_level_feature_belief_corpus,
    )
    from latency_meta_mdp.legacy.belief.flow.config import load_flow_belief_config
    from latency_meta_mdp.legacy.belief.flow.quality_config import (
        load_flow_belief_quality_sample_config,
    )
    from latency_meta_mdp.legacy.belief.flow.quality_samples import (
        load_flow_belief_normalization,
    )
    from latency_meta_mdp.legacy.belief.flow.quality_selection import (
        load_formal_flow_summary,
        score_validation_contexts,
        select_quality_contexts,
    )
    from latency_meta_mdp.legacy.vision_probe_data import ProbeSplit

    flow_config = load_flow_belief_config(
        _PROJECT_ROOT / "configs/legacy/belief/dinov3_flow_belief_v1.yaml"
    )
    quality_config = load_flow_belief_quality_sample_config(
        _PROJECT_ROOT / "configs/legacy/analysis/flow_belief_quality_samples_v1.yaml"
    )
    corpus = load_level_feature_belief_corpus(
        project_root=_PROJECT_ROOT,
        source_bulk_manifest=_SOURCE,
        cache_run_manifest=_CACHE,
        expected_spec=load_vision_encoder_spec(
            _PROJECT_ROOT / "configs/models/vision/dinov3_vits16_lvd1689m_224_v1.yaml"
        ),
        temporal_config_path=_PROJECT_ROOT / "configs/contracts/temporal/h50_e25_d20_k6_v1.yaml",
        latency_law_path=_PROJECT_ROOT
        / "configs/runtime/latency/truncated_beta_5_26_400ms_v1.yaml",
        split_plan_path=_PROJECT_ROOT / "configs/legacy/data/formal_belief_train_val_v1.yaml",
        level=1,
    )
    normalization = load_flow_belief_normalization(_FLOW_ROOT / "L1/normalization.npz")
    summary = load_formal_flow_summary(
        _EVALUATION_ROOT / "L1/summary_arrays.npz",
        expected_context_count=len(corpus.sample_references[ProbeSplit.VALIDATION]),
    )
    scored = score_validation_contexts(
        corpus=corpus,
        summary=summary,
        normalization=normalization,
    )
    selection = select_quality_contexts(scored=scored, config=quality_config)[0]
    return flow_config, quality_config, corpus, summary, selection


def test_selected_sampling_reproduces_formal_summary(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("quality sample integration requires CUDA")
    _require_formal_artifacts()
    from latency_meta_mdp.legacy.belief.flow.quality_samples import (
        export_level_quality_samples,
    )

    flow_config, quality_config, corpus, summary, selection = _load_l1_inputs()
    manifest_path = export_level_quality_samples(
        corpus=corpus,
        config=quality_config,
        flow_config=flow_config,
        flow_checkpoint_dir=_FLOW_ROOT / "L1",
        evaluation_summary=summary,
        selections=(selection,),
        output_dir=tmp_path / "L1",
        device="cuda",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parity = json.loads((manifest_path.parent / "parity.json").read_text(encoding="utf-8"))
    with np.load(manifest_path.parent / "samples.npz", allow_pickle=False) as bundle:
        assert bundle["normalized_samples"].shape == (1, 5, 32, 22)
        assert bundle["physical_samples"].shape == (1, 5, 32, 22)
        assert bundle["validation_offsets"].tolist() == [selection.identity.validation_offset]

    assert manifest["format_id"] == "level_flow_belief_quality_samples_v1"
    assert manifest["level"] == 1
    assert manifest["context_count"] == 1
    assert manifest["sample_count"] == 32
    assert manifest["sampling_wall_seconds"] > 0.0
    assert manifest["seconds_per_selected_context"] > 0.0
    assert manifest["wall_seconds"] >= manifest["sampling_wall_seconds"]
    assert set(manifest["generated_noise_sha256"]) == {str(selection.identity.validation_offset)}
    assert len(next(iter(manifest["generated_noise_sha256"].values()))) == 64
    assert parity["mean_allclose"] is True
    assert parity["std_allclose"] is True
    assert parity["mean_physical_max_abs_error"] >= 0.0
    assert parity["std_physical_max_abs_error"] >= 0.0
    with pytest.raises(FileExistsError):
        export_level_quality_samples(
            corpus=corpus,
            config=quality_config,
            flow_config=flow_config,
            flow_checkpoint_dir=_FLOW_ROOT / "L1",
            evaluation_summary=summary,
            selections=(selection,),
            output_dir=tmp_path / "L1",
            device="cuda",
        )


def test_verified_level_loader_rejects_checkpoint_hash_mismatch(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    _require_formal_artifacts()
    from latency_meta_mdp.legacy.belief.flow.config import load_flow_belief_config
    from latency_meta_mdp.legacy.belief.flow.quality_samples import (
        load_verified_flow_quality_level,
    )

    source = _FLOW_ROOT / "L1"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in ("model.safetensors", "encoder.safetensors", "normalization.npz"):
        os.symlink((source / name).resolve(), checkpoint / name)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest["artifacts"]["model.safetensors"] = "0" * 64
    (checkpoint / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        load_verified_flow_quality_level(
            checkpoint_dir=checkpoint,
            flow_config=load_flow_belief_config(
                _PROJECT_ROOT / "configs/legacy/belief/dinov3_flow_belief_v1.yaml"
            ),
            expected_level=1,
            device="cpu",
        )
