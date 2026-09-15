from __future__ import annotations

import json
from pathlib import Path


def test_hf_bundle_is_inference_only_with_empty_readme(tmp_path: Path) -> None:
    """Catches optimizer, training provenance, or descriptive README leaking to a public repo."""

    from latency_meta_mdp.belief.jepa.hf_publication import (
        build_hf_inference_bundle,
        load_hf_publication_config,
    )

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"model-weights")
    normalization = tmp_path / "normalization.json"
    normalization.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format_id": "action_conditioned_jepa_proprio_normalization_v1",
                "level": 3,
                "mean": [0.0] * 16,
                "scale": [1.0] * 16,
                "constant_dimension_mask": [False] * 16,
                "episode_ids": ["private-episode-id"],
                "boundary_count": 123,
                "source_manifest_sha256": "a" * 64,
                "split_manifest_sha256": "b" * 64,
                "sample_index_sha256": "c" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = load_hf_publication_config(
        Path("configs/legacy/publication/action_conditioned_jepa/hf_inference.yaml")
    )
    output = tmp_path / "bundle"
    bundle = build_hf_inference_bundle(
        output_dir=output,
        publication_config=config,
        level=3,
        model_checkpoint=checkpoint / "model.safetensors",
        proprio_normalization=normalization,
        inference_config={
            "schema_version": 1,
            "format_id": "action_conditioned_jepa_inference_config",
            "level": 3,
            "temporal_config_id": "stride4_80ms_history_160ms",
            "model_stride_ticks": 4,
            "history_observation_count": 3,
            "native_rollout_steps": 5,
            "vision_encoder_id": "dinov3_vits16_lvd1689m_224_v1",
            "camera_order": ["agentview", "wrist"],
            "proprio_dim": 16,
            "action_dim": 7,
        },
    )

    assert bundle.repo_id == "yypeng666/metamdp-jepa-return-l3-s4-h160ms-t400ms-final-v1"
    assert {path.name for path in output.iterdir()} == {
        "README.md",
        "checksums.json",
        "inference_config.json",
        "model.safetensors",
        "proprio_normalization.json",
    }
    assert (output / "README.md").read_bytes() == b""
    assert (output / "model.safetensors").read_bytes() == b"model-weights"
    payload = json.loads((output / "inference_config.json").read_text(encoding="utf-8"))
    public_normalization = json.loads(
        (output / "proprio_normalization.json").read_text(encoding="utf-8")
    )
    assert set(public_normalization) == {
        "schema_version",
        "format_id",
        "level",
        "mean",
        "scale",
        "constant_dimension_mask",
    }
    encoded = json.dumps({"config": payload, "normalization": public_normalization})
    for forbidden in (
        "optimizer",
        "wandb",
        "fold",
        "seed",
        "source_manifest",
        "cache_manifest",
        "private-episode-id",
        "sample_index",
        "/home/",
    ):
        assert forbidden not in encoded.lower()
    assert set(json.loads((output / "checksums.json").read_text())) == {
        "README.md",
        "inference_config.json",
        "model.safetensors",
        "proprio_normalization.json",
    }


def test_hf_commands_are_public_manual_gated_and_never_embed_token() -> None:
    """Catches a private repo, ungated public weights, or token passed on the command line."""

    from latency_meta_mdp.belief.jepa.hf_publication import (
        build_hf_publication_commands,
        load_hf_publication_config,
    )

    config = load_hf_publication_config(
        Path("configs/legacy/publication/action_conditioned_jepa/hf_inference.yaml")
    )
    commands = build_hf_publication_commands(
        config=config,
        level=2,
        bundle_dir=Path("/tmp/inference-bundle"),
    )
    rendered = "\n".join(" ".join(command) for command in commands)

    assert commands[0] == (
        "hf",
        "repos",
        "create",
        "yypeng666/metamdp-jepa-return-l2-s4-h160ms-t400ms-final-v1",
        "--type",
        "model",
        "--public",
        "--exist-ok",
    )
    assert "--gated manual" in rendered
    assert (
        "hf upload yypeng666/metamdp-jepa-return-l2-s4-h160ms-t400ms-final-v1 /tmp/inference-bundle"
    ) in rendered
    assert "--private" not in rendered
    assert "token" not in rendered.lower()
