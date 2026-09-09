from dataclasses import dataclass
from pathlib import Path

from latency_meta_mdp.belief.causal_return.information_run import (
    limit_information_state_corpus,
)
from latency_meta_mdp.cli.evaluate_causal_return_information_state import (
    build_parser as build_evaluation_parser,
)
from latency_meta_mdp.cli.train_causal_return_information_state import (
    build_parser as build_training_parser,
)
from latency_meta_mdp.vision_probe_data import ProbeSplit


@dataclass(frozen=True)
class _FakeCorpus:
    sample_references: dict


def test_training_cli_uses_semantic_defaults_and_all_levels() -> None:
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
    assert args.information_config == Path(
        "configs/belief/causal_return/information_state.yaml"
    )
    assert args.temporal_config == Path("configs/temporal/h50_e25_d20_k6_v1.yaml")
    assert args.split_config == Path("configs/data/formal_belief_train_val_v1.yaml")
    assert args.max_epochs is None
    assert args.training_context_limit is None
    assert args.validation_context_limit is None


def test_evaluation_cli_requires_matching_training_run() -> None:
    args = build_evaluation_parser().parse_args(
        [
            "--source-bulk-manifest",
            "source.json",
            "--cache-run-manifest",
            "cache.json",
            "--training-run-manifest",
            "training/manifest.json",
            "--output-dir",
            "evaluation",
        ]
    )

    assert args.levels == (1, 2, 3)
    assert args.training_run_manifest == Path("training/manifest.json")
    assert args.output_dir == Path("evaluation")


def test_production_cli_names_do_not_use_generation_tags() -> None:
    names = (
        "train_causal_return_information_state",
        "evaluate_causal_return_information_state",
        "causal_return_information_state",
    )
    assert all("v2" not in name and "v3" not in name for name in names)


def test_bounded_context_selection_is_evenly_spaced_not_prefix_biased() -> None:
    corpus = _FakeCorpus(
        sample_references={
            ProbeSplit.TRAIN: tuple((index, 0) for index in range(10)),
            ProbeSplit.VALIDATION: tuple((index, 0) for index in range(6)),
            ProbeSplit.HOLDOUT: (),
        }
    )

    limited = limit_information_state_corpus(
        corpus,
        training_context_limit=4,
        validation_context_limit=3,
    )

    assert limited.sample_references[ProbeSplit.TRAIN] == ((0, 0), (3, 0), (6, 0), (9, 0))
    assert limited.sample_references[ProbeSplit.VALIDATION] == ((0, 0), (2, 0), (5, 0))
