from __future__ import annotations

from pathlib import Path


def _request():
    import hashlib

    from latency_meta_mdp.expert_realization.config import load_formal_corpus_config
    from latency_meta_mdp.expert_realization.contracts import build_formal_request_universe
    from latency_meta_mdp.expert_realization.strategy import StructuredStrategyConfig

    root = Path.cwd()
    path = root / "configs/source_corpus/panda_ball_formal_source_pilot.yaml"
    structured = StructuredStrategyConfig.from_path(
        root / "configs/expert_realization/panda_ball_structured.yaml"
    )
    return build_formal_request_universe(
        load_formal_corpus_config(path),
        corpus_config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        structured_expert_config_sha256=structured.source_sha256,
    )


def test_formal_collection_identity_binds_every_scientific_input() -> None:
    """Break caught: resume accepts changed split, execution, gate, lock, or implementation."""
    from latency_meta_mdp.expert_realization.recording_contracts import ImplementationIdentity
    from latency_meta_mdp.expert_realization.source_corpus.config import (
        load_master_task_split_plan,
        load_source_corpus_config,
        load_source_execution_config,
    )
    from latency_meta_mdp.expert_realization.source_corpus.formal_runtime import (
        build_formal_collection_identity,
    )

    root = Path.cwd()
    identity = build_formal_collection_identity(
        formal_request=_request(),
        source_config=load_source_corpus_config(
            root / "configs/source_corpus/panda_ball_source_parquet.yaml"
        ),
        split_plan=load_master_task_split_plan(
            root / "configs/source_corpus/panda_ball_formal_source_pilot_split.yaml"
        ),
        execution_config=load_source_execution_config(
            root / "configs/source_corpus/panda_ball_formal_source_execution.yaml"
        ),
        qualification_gate_sha256="a" * 64,
        planner_environment_sha256="b" * 64,
        implementation=ImplementationIdentity("revision", "c" * 64, False),
    )

    assert set(identity) == {
        "schema_version",
        "format_id",
        "formal_request_sha256",
        "source_config_sha256",
        "split_plan_sha256",
        "execution_config_sha256",
        "qualification_gate_sha256",
        "planner_environment_sha256",
        "implementation",
    }
    assert identity["formal_request_sha256"] == _request().request_sha256
    assert identity["implementation"]["source_sha256"] == "c" * 64


def test_planner_launcher_keeps_venv_symlink_lexically() -> None:
    """Break caught: Path.resolve changes the venv launcher into the bare uv interpreter."""
    from latency_meta_mdp.expert_realization.source_corpus.formal_runtime import (
        lexical_planner_launcher,
    )

    root = Path.cwd()
    launcher = lexical_planner_launcher(
        root, Path(".venv-expert-realization/bin/python")
    )
    assert launcher == (root / ".venv-expert-realization/bin/python").absolute()
    assert launcher != launcher.resolve()
