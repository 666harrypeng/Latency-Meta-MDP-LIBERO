from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest

from latency_meta_mdp.belief.jepa.ar import qualify_cli as qualify_cli
from latency_meta_mdp.belief.jepa.check_upstream import (
    OfficialPredictorQualification,
    official_predictor_config,
    qualify_official_predictor,
    qualify_upstream_reference,
    validate_locked_runtime,
    write_upstream_qualification_manifest,
)
from latency_meta_mdp.belief.jepa.upstream_adapter import (
    CheckpointMirror,
    UpstreamCheckout,
    load_upstream_primitives,
    load_upstream_reference,
    verify_checkpoint_mirror,
    verify_upstream_checkout,
)
from latency_meta_mdp.io.paths import repository_root

REFERENCE_PATH = Path("configs/models/jepa/upstream_reference.yaml")


def _valid_upstream_yaml() -> str:
    return REFERENCE_PATH.read_text(encoding="utf-8")


def test_upstream_reference_is_exact(tmp_path: Path) -> None:
    path = tmp_path / "upstream.yaml"
    path.write_text(_valid_upstream_yaml(), encoding="utf-8")

    value = load_upstream_reference(path)

    assert value.commit == "13cf1d9c7e476f53c17714d2e0f1dc239a883ce0"
    assert value.license_sha256 == (
        "1b0556efdc7e72b17b706c041002c6cdc1e0aa257f5e5676ea0f887b2f0854ec"
    )
    assert value.reference_config_sha256 == (
        "db0c9cab6c12cb6541c222026358e7ccb482aa2276ed12b5a597ff4b547d0bba"
    )
    assert value.droid_reference_config_sha256 == (
        "be981788c58515ba6be1eea84ee0fbfe4e535f7da374a95b7b2e47dc4a241c4e"
    )
    assert value.metaworld_checkpoint_sha256 == (
        "c3297772c7af4e84c28bd0c0937e22398a888848305014bb3ab3f1508a310bd8"
    )
    assert value.droid_checkpoint_sha256 == (
        "daa69198aef764932f1cb809239a4e19c71da20a93c6a0b9f3869cb30a13f4aa"
    )


def test_upstream_reference_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "upstream.yaml"
    path.write_text(_valid_upstream_yaml() + "mutable_branch: main\n", encoding="utf-8")

    with pytest.raises(ValueError, match="fields"):
        load_upstream_reference(path)


@pytest.mark.parametrize("value", ("true", "1.0"))
def test_upstream_reference_rejects_noninteger_schema_version(
    tmp_path: Path,
    value: str,
) -> None:
    path = tmp_path / "upstream.yaml"
    payload = _valid_upstream_yaml().replace("schema_version: 1", f"schema_version: {value}")
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported"):
        load_upstream_reference(path)


def test_upstream_checkout_verifies_pinned_submodule() -> None:
    project_root = repository_root()
    reference = load_upstream_reference(
        project_root / "configs/models/jepa/upstream_reference.yaml"
    )

    checkout = verify_upstream_checkout(reference=reference, project_root=project_root)

    assert checkout.root == project_root / "third_party/jepa-wms"
    assert checkout.commit == reference.commit
    assert checkout.license_sha256 == reference.license_sha256
    assert checkout.reference_config_sha256 == reference.reference_config_sha256
    assert checkout.droid_reference_config_sha256 == reference.droid_reference_config_sha256


def test_upstream_primitive_bundle_exposes_adaln_and_rope_contract(monkeypatch) -> None:
    project_root = repository_root()
    reference = load_upstream_reference(REFERENCE_PATH)

    # The eval environment exposes only our src directory, not the JEPA checkout.
    monkeypatch.setattr(
        sys, "path", [p for p in sys.path if p != str(project_root / "third_party/jepa-wms")]
    )
    bundle = load_upstream_primitives(reference=reference, project_root=project_root)

    assert bundle.rope_attention_type.__name__ == "RoPEAttention"
    assert bundle.fw_adaln_block_type.__name__ == "FWAdaLNBlock"
    assert bundle.vision_transformer_adaln_type.__name__ == "VisionTransformerAdaLN"
    assert bundle.mlp_type.__name__ == "MLP"
    assert bundle.drop_path_type.__name__ == "DropPath"
    assert callable(bundle.modulate)
    assert callable(bundle.rotate_queries_or_keys)
    assert callable(bundle.trunc_normal)


def test_upstream_primitives_reject_modules_outside_verified_checkout(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = repository_root()
    reference = load_upstream_reference(REFERENCE_PATH)
    fake = ModuleType("fake")
    fake.__file__ = str(tmp_path / "fake.py")
    fake.RoPEAttention = type("RoPEAttention", (), {})
    fake.MLP = type("MLP", (), {})
    fake.DropPath = type("DropPath", (), {})
    fake.rotate_queries_or_keys = lambda *args: args
    fake.FWAdaLNBlock = type("FWAdaLNBlock", (), {})
    fake.VisionTransformerAdaLN = type("VisionTransformerAdaLN", (), {})
    fake.modulate = lambda *args: args
    fake.trunc_normal_ = lambda *args: args
    monkeypatch.setitem(sys.modules, "src.models.utils.modules", fake)
    monkeypatch.setitem(sys.modules, "app.plan_common.models.AdaLN_vit", fake)
    monkeypatch.setitem(sys.modules, "src.utils.tensors", fake)

    with pytest.raises(ValueError, match="outside"):
        load_upstream_primitives(reference=reference, project_root=project_root)


def test_checkpoint_mirror_requires_identical_bytes(tmp_path: Path) -> None:
    direct = tmp_path / "direct.pth.tar"
    hub = tmp_path / "hub.pth.tar"
    direct.write_bytes(b"same-checkpoint")
    hub.write_bytes(b"same-checkpoint")

    expected = hashlib.sha256(b"same-checkpoint").hexdigest()
    mirror = verify_checkpoint_mirror(
        name="metaworld",
        direct_path=direct,
        hub_path=hub,
        expected_sha256=expected,
    )

    assert mirror.name == "metaworld"
    assert mirror.size_bytes == len(b"same-checkpoint")
    assert len(mirror.sha256) == 64


def test_checkpoint_mirror_rejects_disagreement(tmp_path: Path) -> None:
    direct = tmp_path / "direct.pth.tar"
    hub = tmp_path / "hub.pth.tar"
    direct.write_bytes(b"direct")
    hub.write_bytes(b"hub")

    with pytest.raises(ValueError, match="mirror"):
        verify_checkpoint_mirror(
            name="droid",
            direct_path=direct,
            hub_path=hub,
            expected_sha256=hashlib.sha256(b"direct").hexdigest(),
        )


def test_checkpoint_mirror_rejects_identical_unpinned_bytes(tmp_path: Path) -> None:
    direct = tmp_path / "direct.pth.tar"
    hub = tmp_path / "hub.pth.tar"
    direct.write_bytes(b"identical-but-untrusted")
    hub.write_bytes(b"identical-but-untrusted")

    with pytest.raises(ValueError, match="pinned"):
        verify_checkpoint_mirror(
            name="droid",
            direct_path=direct,
            hub_path=hub,
            expected_sha256="0" * 64,
        )


def _qualification(name: str, sha256: str) -> OfficialPredictorQualification:
    config = official_predictor_config(name)
    output_visual_shape = (
        1,
        config.frame_count,
        config.grid_size**2,
        config.embed_dim,
    )
    output_proprio_shape = (
        None
        if config.proprio_feature_dim == 0
        else (
            1,
            config.frame_count,
            config.grid_size**2,
            config.proprio_feature_dim,
        )
    )
    return OfficialPredictorQualification(
        name=name,
        checkpoint_sha256=sha256,
        checkpoint_size_bytes=10,
        checkpoint_epoch=config.checkpoint_epoch,
        predictor_parameter_count=config.predictor_parameter_count,
        auxiliary_parameter_count=config.auxiliary_parameter_count,
        predictor_state_key_count=config.predictor_state_key_count,
        input_visual_shape=config.input_visual_shape,
        input_action_shape=config.input_action_shape,
        input_proprio_shape=config.input_proprio_shape,
        output_visual_shape=output_visual_shape,
        output_proprio_shape=output_proprio_shape,
        output_dtype="bfloat16",
        two_step_output_shape=output_visual_shape,
        finite=True,
        warmup_iterations=20,
        benchmark_iterations=100,
        one_step_p50_ms=1.0,
        one_step_p95_ms=1.2,
        one_step_p99_ms=1.3,
        two_step_p50_ms=2.0,
        two_step_p95_ms=2.2,
        two_step_p99_ms=2.3,
        peak_reserved_bytes=1024,
    )


def _locked_runtime() -> dict[str, object]:
    return {
        "python": "3.10.20",
        "torch": "2.7.1+cu126",
        "torchvision": "0.22.1+cu126",
        "tensordict": "0.11.0",
        "timm": "1.0.19",
        "device": "cuda",
        "cuda_available": True,
        "cuda_device_name": "fake-cuda",
        "cuda_capability": [8, 9],
    }


@pytest.mark.parametrize("capability", ([8, 0], [9, 0]))
def test_locked_runtime_accepts_zero_minor_compute_capability(capability: list[int]) -> None:
    runtime = _locked_runtime()
    runtime["cuda_capability"] = capability

    validate_locked_runtime(runtime)


def test_locked_runtime_rejects_version_prefix_collision() -> None:
    runtime = _locked_runtime()
    runtime["torch"] = "2.7.10+cu999"

    with pytest.raises(ValueError, match="runtime"):
        validate_locked_runtime(runtime)


def test_predictor_qualification_rejects_model_specific_topology_mismatch() -> None:
    metaworld = _qualification("metaworld", "a" * 64)

    with pytest.raises(ValueError, match="topology"):
        replace(metaworld, name="droid")


def test_qualification_manifest_binds_upstream_and_mirrors(tmp_path: Path) -> None:
    reference_path = Path("configs/models/jepa/upstream_reference.yaml")
    reference = load_upstream_reference(reference_path)
    checkout = UpstreamCheckout(
        root=Path("/checkout"),
        commit=reference.commit,
        license_sha256=reference.license_sha256,
        reference_config_sha256=reference.reference_config_sha256,
        droid_reference_config_sha256=reference.droid_reference_config_sha256,
    )
    mirrors = (
        CheckpointMirror(
            "metaworld",
            Path("/direct/mw"),
            Path("/hub/mw"),
            reference.metaworld_checkpoint_sha256,
            10,
        ),
        CheckpointMirror(
            "droid",
            Path("/direct/droid"),
            Path("/hub/droid"),
            reference.droid_checkpoint_sha256,
            10,
        ),
    )

    manifest = write_upstream_qualification_manifest(
        output_dir=tmp_path / "qualification",
        reference=reference,
        reference_config_path=reference_path,
        environment_lock_path=Path("requirements/action-conditioned-jepa.lock"),
        checkout=checkout,
        mirrors=mirrors,
        qualifications=(
            _qualification("metaworld", reference.metaworld_checkpoint_sha256),
            _qualification("droid", reference.droid_checkpoint_sha256),
        ),
        runtime=_locked_runtime(),
    )
    payload = __import__("json").loads(manifest.read_text(encoding="utf-8"))

    assert payload["format_id"] == "action_conditioned_jepa_upstream_qualification_v1"
    assert payload["qualification_pass"] is True
    assert payload["reference"]["commit"] == reference.commit
    assert len(payload["environment_lock"]["sha256"]) == 64
    assert set(payload["checkpoints"]) == {"metaworld", "droid"}
    assert set(payload["predictors"]) == {"metaworld", "droid"}


def test_qualification_manifest_rejects_unlocked_cpu_runtime(tmp_path: Path) -> None:
    reference = load_upstream_reference(REFERENCE_PATH)
    checkout = UpstreamCheckout(
        root=Path("/checkout"),
        commit=reference.commit,
        license_sha256=reference.license_sha256,
        reference_config_sha256=reference.reference_config_sha256,
        droid_reference_config_sha256=reference.droid_reference_config_sha256,
    )
    mirrors = (
        CheckpointMirror(
            "metaworld",
            Path("/direct/mw"),
            Path("/hub/mw"),
            reference.metaworld_checkpoint_sha256,
            10,
        ),
        CheckpointMirror(
            "droid",
            Path("/direct/droid"),
            Path("/hub/droid"),
            reference.droid_checkpoint_sha256,
            10,
        ),
    )
    runtime = _locked_runtime()
    runtime.update(device="cpu", cuda_available=False)

    with pytest.raises(ValueError, match="runtime"):
        write_upstream_qualification_manifest(
            output_dir=tmp_path / "qualification",
            reference=reference,
            reference_config_path=REFERENCE_PATH,
            environment_lock_path=Path("requirements/action-conditioned-jepa.lock"),
            checkout=checkout,
            mirrors=mirrors,
            qualifications=(
                _qualification("metaworld", reference.metaworld_checkpoint_sha256),
                _qualification("droid", reference.droid_checkpoint_sha256),
            ),
            runtime=runtime,
        )


def test_official_predictor_configs_match_checkpoint_topology() -> None:
    metaworld = official_predictor_config("metaworld")
    droid = official_predictor_config("droid")

    assert metaworld.input_visual_shape == (1, 4, 1, 16, 16, 384)
    assert metaworld.input_action_shape == (1, 4, 20)
    assert metaworld.input_proprio_shape == (1, 4, 4)
    assert metaworld.depth == 6
    assert metaworld.proprio_feature_dim == 16
    assert droid.input_visual_shape == (1, 4, 1, 16, 16, 1024)
    assert droid.input_action_shape == (1, 4, 7)
    assert droid.input_proprio_shape is None
    assert droid.depth == 12
    assert droid.proprio_feature_dim == 0


def test_upstream_qualification_orchestrates_both_predictors(tmp_path: Path) -> None:
    project_root = repository_root()
    mirror_root = tmp_path / "mirrors"
    (mirror_root / "direct").mkdir(parents=True)
    (mirror_root / "hf").mkdir()
    payloads = (
        ("metaworld.pth.tar", "jepa_wm_metaworld.pth.tar", b"metaworld!"),
        ("droid.pth.tar", "jepa_wm_droid.pth.tar", b"droiddata!"),
    )
    for direct_name, hub_name, payload in payloads:
        (mirror_root / "direct" / direct_name).write_bytes(payload)
        (mirror_root / "hf" / hub_name).write_bytes(payload)
    reference_path = tmp_path / "upstream.yaml"
    reference_payload = _valid_upstream_yaml()
    reference_payload = reference_payload.replace(
        "c3297772c7af4e84c28bd0c0937e22398a888848305014bb3ab3f1508a310bd8",
        hashlib.sha256(payloads[0][2]).hexdigest(),
    ).replace(
        "daa69198aef764932f1cb809239a4e19c71da20a93c6a0b9f3869cb30a13f4aa",
        hashlib.sha256(payloads[1][2]).hexdigest(),
    )
    reference_path.write_text(reference_payload, encoding="utf-8")
    calls = []

    def fake_qualifier(**kwargs):
        calls.append(kwargs)
        return _qualification(kwargs["name"], kwargs["checkpoint_sha256"])

    manifest = qualify_upstream_reference(
        reference_config_path=reference_path,
        project_root=project_root,
        mirror_root=mirror_root,
        output_dir=tmp_path / "qualification",
        device="cuda",
        predictor_qualifier=fake_qualifier,
        runtime_collector=lambda _device: _locked_runtime(),
    )

    assert manifest.is_file()
    assert [row["name"] for row in calls] == ["metaworld", "droid"]
    assert all(row["warmup_iterations"] == 20 for row in calls)
    assert all(row["benchmark_iterations"] == 100 for row in calls)
    assert all(row["upstream_root"] == project_root / "third_party/jepa-wms" for row in calls)


def test_upstream_qualification_cli_prints_only_manifest(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    captured = {}
    manifest = tmp_path / "qualification/manifest.json"

    def fake_qualify(**kwargs):
        print("upstream-info")
        captured.update(kwargs)
        return manifest

    monkeypatch.setattr(qualify_cli, "qualify_upstream_reference", fake_qualify)

    result = qualify_cli.main(
        [
            "--project-root",
            "/project",
            "--reference-config",
            "upstream.yaml",
            "--mirror-root",
            "mirrors",
            "--output-dir",
            "qualification",
            "--device",
            "cpu",
        ]
    )

    assert result == 0
    assert captured["project_root"] == Path("/project")
    assert captured["reference_config_path"] == Path("upstream.yaml")
    assert captured["mirror_root"] == Path("mirrors")
    assert captured["output_dir"] == Path("qualification")
    assert captured["device"] == "cpu"
    output = capsys.readouterr()
    assert output.out.strip() == str(manifest)
    assert output.err.strip() == "upstream-info"


@pytest.mark.skipif(
    os.environ.get("RUN_JEPA_UPSTREAM_INTEGRATION") != "1",
    reason="explicit GPU/upstream checkpoint integration lane",
)
@pytest.mark.parametrize(
    ("name", "relative_checkpoint", "parameter_count", "auxiliary_count"),
    (
        (
            "metaworld",
            "mirror_identity/hf/jepa_wm_metaworld.pth.tar",
            17_630_480,
            80,
        ),
        (
            "droid",
            "mirror_identity/hf/jepa_wm_droid.pth.tar",
            228_835_328,
            0,
        ),
    ),
)
def test_official_predictor_checkpoint_forward_and_two_step_rollout(
    name: str,
    relative_checkpoint: str,
    parameter_count: int,
    auxiliary_count: int,
) -> None:
    root = Path("outputs/upstream/action_conditioned_jepa")
    checkpoint = root / relative_checkpoint
    if not checkpoint.is_file():
        pytest.skip(f"missing ignored upstream checkpoint: {checkpoint}")

    result = qualify_official_predictor(
        name=name,
        checkpoint_path=checkpoint,
        checkpoint_sha256=(
            "c3297772c7af4e84c28bd0c0937e22398a888848305014bb3ab3f1508a310bd8"
            if name == "metaworld"
            else "daa69198aef764932f1cb809239a4e19c71da20a93c6a0b9f3869cb30a13f4aa"
        ),
        upstream_root=Path("third_party/jepa-wms").resolve(),
        device="cuda",
        warmup_iterations=1,
        benchmark_iterations=3,
    )

    assert result.predictor_parameter_count == parameter_count
    assert result.auxiliary_parameter_count == auxiliary_count
    assert result.output_visual_shape == official_predictor_config(name).input_visual_shape[:2] + (
        256,
        official_predictor_config(name).embed_dim,
    )
    assert result.two_step_output_shape == result.output_visual_shape
    assert result.output_dtype == "bfloat16"
    assert result.finite
