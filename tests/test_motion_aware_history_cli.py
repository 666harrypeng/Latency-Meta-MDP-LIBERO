from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from latency_meta_mdp.io.artifacts import sha256_file
from latency_meta_mdp.legacy.belief.causal_return.motion_aware_run import (
    evaluate_motion_aware_history_run,
    limit_motion_aware_history_corpus,
    train_motion_aware_history_run,
)
from latency_meta_mdp.legacy.cli.evaluate_motion_aware_history import (
    build_parser as build_evaluation_parser,
)
from latency_meta_mdp.legacy.cli.train_motion_aware_history import (
    build_parser as build_training_parser,
)
from latency_meta_mdp.legacy.vision_probe_data import ProbeSplit


@dataclass(frozen=True)
class _FakeCorpus:
    sample_references: dict
    level: int = 1


def test_training_cli_uses_versioned_semantic_defaults() -> None:
    args = build_training_parser().parse_args(
        [
            "--source-bulk-manifest",
            "source.json",
            "--cache-run-manifest",
            "cache.json",
            "--output-dir",
            "output",
        ]
    )
    assert args.levels == (1, 2, 3)
    assert args.vision_config == Path("configs/models/vision/dinov3_vits16_lvd1689m_224_v1.yaml")
    assert args.temporal_config == Path("configs/contracts/temporal/h50_e25_d20_k6_v1.yaml")
    assert args.split_config == Path("configs/legacy/data/formal_belief_train_val_v1.yaml")
    assert args.motion_aware_config == Path(
        "configs/legacy/belief/causal_return/motion_aware_history.yaml"
    )
    assert args.max_epochs is None
    assert args.training_context_limit is None
    assert args.validation_context_limit is None


def test_evaluation_cli_requires_training_and_baseline_manifests() -> None:
    args = build_evaluation_parser().parse_args(
        [
            "--source-bulk-manifest",
            "source.json",
            "--cache-run-manifest",
            "cache.json",
            "--training-run-manifest",
            "training/manifest.json",
            "--baseline-evaluation-manifest",
            "baseline/manifest.json",
            "--output-dir",
            "evaluation",
        ]
    )
    assert args.training_run_manifest == Path("training/manifest.json")
    assert args.baseline_evaluation_manifest == Path("baseline/manifest.json")
    assert args.levels == (1, 2, 3)


def test_bounded_context_selection_is_evenly_spaced() -> None:
    corpus = _FakeCorpus(
        sample_references={
            ProbeSplit.TRAIN: tuple((index, 0) for index in range(10)),
            ProbeSplit.VALIDATION: tuple((index, 0) for index in range(6)),
            ProbeSplit.HOLDOUT: (),
        }
    )
    limited = limit_motion_aware_history_corpus(
        corpus,
        training_context_limit=4,
        validation_context_limit=3,
    )
    assert limited.sample_references[ProbeSplit.TRAIN] == (
        (0, 0),
        (3, 0),
        (6, 0),
        (9, 0),
    )
    assert limited.sample_references[ProbeSplit.VALIDATION] == (
        (0, 0),
        (2, 0),
        (5, 0),
    )


def test_new_production_names_have_no_generation_tags() -> None:
    names = (
        "motion_aware_history",
        "train_motion_aware_history",
        "evaluate_motion_aware_history",
        "causal_return_motion_aware_history",
    )
    assert all("v2" not in name and "v3" not in name for name in names)


def _inputs(tmp_path: Path) -> dict[str, Path]:
    source = tmp_path / "source.json"
    cache = tmp_path / "cache.json"
    source.write_text("{}\n", encoding="utf-8")
    cache.write_text("{}\n", encoding="utf-8")
    return {
        "source_bulk_manifest": source,
        "cache_run_manifest": cache,
        "vision_config_path": Path("configs/models/vision/dinov3_vits16_lvd1689m_224_v1.yaml"),
        "temporal_config_path": Path("configs/contracts/temporal/h50_e25_d20_k6_v1.yaml"),
        "split_config_path": Path("configs/legacy/data/formal_belief_train_val_v1.yaml"),
        "motion_aware_config_path": Path(
            "configs/legacy/belief/causal_return/motion_aware_history.yaml"
        ),
    }


def _fake_corpus_loader(**kwargs):
    level = kwargs["level"]
    return _FakeCorpus(
        level=level,
        sample_references={
            ProbeSplit.TRAIN: tuple((index, 0) for index in range(4)),
            ProbeSplit.VALIDATION: tuple((index, 0) for index in range(2)),
            ProbeSplit.HOLDOUT: (),
        },
    )


def _fake_level_trainer(*, corpus, config, output_dir, device, provenance):
    assert device == "cpu"
    output_dir.mkdir()
    value = {
        "format_id": "causal_return_motion_aware_history_checkpoint",
        "level": corpus.level,
        "artifact_eligible": provenance.artifact_eligible,
        "artifacts": {},
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return path


def _provenance(_root: Path):
    return SimpleNamespace(revision="1" * 40, source_sha256="2" * 64, dirty=False)


def test_training_runner_binds_inputs_bounds_and_no_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "training"
    manifest = train_motion_aware_history_run(
        project_root=Path.cwd(),
        output_dir=output,
        levels=(1,),
        device="cpu",
        max_epochs=2,
        training_context_limit=2,
        validation_context_limit=1,
        corpus_loader=_fake_corpus_loader,
        level_trainer=_fake_level_trainer,
        provenance_collector=_provenance,
        **_inputs(tmp_path),
    )
    value = json.loads(manifest.read_text(encoding="utf-8"))
    assert value["format_id"] == "causal_return_motion_aware_history_training_run"
    assert value["artifact_eligible"] is False
    assert value["scientific_gate_pass"] is False
    assert value["levels"] == [1]
    assert set(value["input_sha256"]) == {
        "source_bulk_manifest",
        "cache_run_manifest",
        "vision_config",
        "temporal_config",
        "split_config",
        "motion_aware_config",
    }
    assert Path(value["input_paths"]["source_bulk_manifest"]).is_absolute()
    with pytest.raises(FileExistsError, match="exists"):
        train_motion_aware_history_run(
            project_root=Path.cwd(),
            output_dir=output,
            levels=(1,),
            device="cpu",
            corpus_loader=_fake_corpus_loader,
            level_trainer=_fake_level_trainer,
            provenance_collector=_provenance,
            **_inputs(tmp_path),
        )


def _baseline(tmp_path: Path, *, source_sha256: str) -> Path:
    root = tmp_path / "baseline"
    level = root / "L1"
    level.mkdir(parents=True)
    np.savez(
        level / "predictions.npz",
        episode_id=np.asarray(["episode-0"]),
        source_tick=np.asarray([25]),
        source_phase=np.asarray(["pregrasp"]),
        object_state_mean=np.zeros((1, 6), dtype=np.float32),
        object_state_target=np.zeros((1, 6), dtype=np.float32),
    )
    level_manifest = {
        "format_id": "causal_return_information_state_evaluation",
        "level": 1,
        "artifacts": {
            "predictions.npz": sha256_file(level / "predictions.npz"),
        },
    }
    (level / "manifest.json").write_text(json.dumps(level_manifest) + "\n", encoding="utf-8")
    top = {
        "format_id": "causal_return_information_state_evaluation_run",
        "levels": [1],
        "level_manifests": {"L1": "L1/manifest.json"},
        "input_sha256": {"source_bulk_manifest": source_sha256},
        "artifacts": {
            "L1/manifest.json": sha256_file(level / "manifest.json"),
            "L1/predictions.npz": sha256_file(level / "predictions.npz"),
        },
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(top) + "\n", encoding="utf-8")
    return path


def _fake_level_evaluator(*, corpus, output_dir, **kwargs):
    output_dir.mkdir()
    path = output_dir / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "format_id": "causal_return_motion_aware_history_evaluation",
                "level": corpus.level,
                "scientific_gate_pass": True,
                "artifact_eligible": False,
                "artifacts": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _fake_failing_level_evaluator(*, corpus, output_dir, **kwargs):
    output_dir.mkdir()
    path = output_dir / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "format_id": "causal_return_motion_aware_history_evaluation",
                "level": corpus.level,
                "scientific_gate_pass": False,
                "artifact_eligible": True,
                "artifacts": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_evaluation_runner_separates_scientific_and_artifact_status(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    training = train_motion_aware_history_run(
        project_root=Path.cwd(),
        output_dir=tmp_path / "training",
        levels=(1,),
        device="cpu",
        max_epochs=2,
        corpus_loader=_fake_corpus_loader,
        level_trainer=_fake_level_trainer,
        provenance_collector=_provenance,
        **inputs,
    )
    baseline = _baseline(
        tmp_path,
        source_sha256=sha256_file(inputs["source_bulk_manifest"]),
    )
    manifest = evaluate_motion_aware_history_run(
        project_root=Path.cwd(),
        training_run_manifest=training,
        baseline_evaluation_manifest=baseline,
        output_dir=tmp_path / "evaluation",
        levels=(1,),
        device="cpu",
        corpus_loader=_fake_corpus_loader,
        level_evaluator=_fake_level_evaluator,
        provenance_collector=_provenance,
        **inputs,
    )
    value = json.loads(manifest.read_text(encoding="utf-8"))
    assert value["scientific_gate_pass"] is True
    assert value["artifact_eligible"] is False
    assert Path(value["training_run_manifest_path"]).is_absolute()
    assert Path(value["baseline_run_manifest_path"]).is_absolute()


def test_canonical_failed_science_remains_provenance_eligible(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    training = train_motion_aware_history_run(
        project_root=Path.cwd(),
        output_dir=tmp_path / "training",
        levels=(1,),
        device="cpu",
        corpus_loader=_fake_corpus_loader,
        level_trainer=_fake_level_trainer,
        provenance_collector=_provenance,
        **inputs,
    )
    baseline = _baseline(
        tmp_path,
        source_sha256=sha256_file(inputs["source_bulk_manifest"]),
    )
    manifest = evaluate_motion_aware_history_run(
        project_root=Path.cwd(),
        training_run_manifest=training,
        baseline_evaluation_manifest=baseline,
        output_dir=tmp_path / "evaluation",
        levels=(1,),
        device="cpu",
        corpus_loader=_fake_corpus_loader,
        level_evaluator=_fake_failing_level_evaluator,
        provenance_collector=_provenance,
        **inputs,
    )
    value = json.loads(manifest.read_text(encoding="utf-8"))
    assert value["scientific_gate_pass"] is False
    assert value["artifact_eligible"] is True
    assert value["provenance_blockers"] == []


def test_evaluation_runner_rejects_corrupted_top_level_hash_chain(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    training = train_motion_aware_history_run(
        project_root=Path.cwd(),
        output_dir=tmp_path / "training",
        levels=(1,),
        device="cpu",
        corpus_loader=_fake_corpus_loader,
        level_trainer=_fake_level_trainer,
        provenance_collector=_provenance,
        **inputs,
    )
    value = json.loads(training.read_text(encoding="utf-8"))
    value["artifacts"]["L1/manifest.json"] = "0" * 64
    training.write_text(json.dumps(value) + "\n", encoding="utf-8")
    baseline = _baseline(
        tmp_path,
        source_sha256=sha256_file(inputs["source_bulk_manifest"]),
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        evaluate_motion_aware_history_run(
            project_root=Path.cwd(),
            training_run_manifest=training,
            baseline_evaluation_manifest=baseline,
            output_dir=tmp_path / "evaluation",
            levels=(1,),
            device="cpu",
            corpus_loader=_fake_corpus_loader,
            level_evaluator=_fake_level_evaluator,
            provenance_collector=_provenance,
            **inputs,
        )
