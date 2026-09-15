"""Cache admitted L3 nominal JEPA futures once for matched policy views; no SFT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latency_meta_mdp.data.collection.artifacts import _hash_file


def main() -> None:
    import torch
    from safetensors.torch import load_file

    from latency_meta_mdp.belief.jepa.ar.qualify import (
        load_completed_l3_admission_run,
    )
    from latency_meta_mdp.belief.jepa.backbone import (
        ActionConditionedJepaPredictor,
    )
    from latency_meta_mdp.belief.jepa.config import (
        load_action_conditioned_jepa_config,
    )
    from latency_meta_mdp.belief.jepa.corpus import (
        load_action_conditioned_jepa_corpus,
        load_jepa_proprio_normalization,
    )
    from latency_meta_mdp.legacy.policy.policy_return_cache import write_nominal_return_predictions

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    root = args.project_root.resolve()
    decision = json.loads(
        (root / "outputs/analysis/action_conditioned_jepa/l3-admission-decision.json").read_text()
    )
    if decision["level"] != 3 or decision["status"] != "admitted_for_policy_integration":
        raise ValueError("L3 is not admitted for policy integration")
    checkpoint = root / decision["canonical_checkpoint"]
    run_root = checkpoint.parents[2]
    completed = load_completed_l3_admission_run(
        run_root, expected_model_seed=decision["canonical_model_seed"]
    )
    if (
        checkpoint.parent != completed.checkpoint_dir
        or _hash_file(checkpoint) != decision["canonical_checkpoint_sha256"]
    ):
        raise ValueError("canonical checkpoint and completed run disagree")
    norm_path = root / decision["normalization"]
    normalization = load_jepa_proprio_normalization(norm_path)
    temporal = root / "configs/models/jepa/stride4_80ms_history_160ms.yaml"
    config = load_action_conditioned_jepa_config(
        model_path=root / "configs/models/jepa/model.yaml",
        level_path=root / "configs/models/jepa/l3.yaml",
        temporal_sampling_path=temporal,
    )
    source_id = "panda-ball-structured-source-quota-formal-100x4-v1"
    corpus = load_action_conditioned_jepa_corpus(
        source_root=root / "outputs/source_corpus" / source_id,
        cache_run_manifest=root
        / "outputs/derived/vision_features/dinov3-vits16-structured-source-100x4-v1/manifest.json",
        split_manifest_path=root
        / "outputs/derived/source_splits"
        / source_id
        / "train80-validation20-seed20260903-v1.json",
        level=3,
        split="train",
        config=config,
        normalization=normalization,
    )
    device = torch.device(args.device)
    model = ActionConditionedJepaPredictor(
        config=config, proprio_normalization=normalization, project_root=root
    ).to(device)
    model.load_state_dict(load_file(str(checkpoint), device=str(device)), strict=True)
    model.eval()
    bindings = {
        "checkpoint_sha256": decision["canonical_checkpoint_sha256"],
        "source_manifest_sha256": corpus.source_manifest_sha256,
        "split_manifest_sha256": corpus.split_manifest_sha256,
        "vision_cache_manifest_sha256": corpus.cache_manifest_sha256,
        "normalization_sha256": _hash_file(norm_path),
        "temporal_config_sha256": _hash_file(temporal),
        "model_config_sha256": _hash_file(root / "configs/models/jepa/model.yaml"),
    }
    checkpoint_inputs = json.loads((checkpoint.parent / "manifest.json").read_text())[
        "input_sha256"
    ]
    for checkpoint_key, binding_key in (
        ("model_config", "model_config_sha256"),
        ("temporal_config", "temporal_config_sha256"),
        ("source_manifest", "source_manifest_sha256"),
        ("split_manifest", "split_manifest_sha256"),
        ("cache_manifest", "vision_cache_manifest_sha256"),
    ):
        if checkpoint_inputs[checkpoint_key] != bindings[binding_key]:
            raise ValueError("prediction inputs differ from the admitted checkpoint provenance")
    print(
        write_nominal_return_predictions(
            records=corpus.records,
            normalization=normalization,
            sampling=config.temporal_sampling,
            model=model,
            device=device,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            bindings=bindings,
        )
    )


if __name__ == "__main__":
    main()
