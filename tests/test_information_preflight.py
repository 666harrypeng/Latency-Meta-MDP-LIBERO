from pathlib import Path

import pytest
import yaml

from latency_meta_mdp.meta.information.config import load_config
from latency_meta_mdp.meta.information.preflight import verify_assets


def test_hash_verification_rejects_missing_or_changed_asset(tmp_path):
    from latency_meta_mdp.meta.information.config import sha256

    file = tmp_path / "weights"
    file.write_bytes(b"original")
    assets = {"files": {"weights": sha256(file)}}
    assert verify_assets(tmp_path, assets)["verified_files"] == 1
    file.write_bytes(b"changed")
    with pytest.raises(ValueError, match="asset"):
        verify_assets(tmp_path, assets)


def test_experiment_config_rejects_algorithm_and_unknown_nested_keys(tmp_path):
    source = Path("configs/experiments/conveyor_sort/meta_information.yaml")
    config = load_config(source)
    assert config["training"]["algorithm"] == "fqi"
    config["training"]["privileged_actor"] = True
    target = tmp_path / "bad.yaml"
    target.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="training"):
        load_config(target)


def test_actual_restore_paths_must_be_covered_by_pinned_manifest(tmp_path):
    source = Path("configs/experiments/conveyor_sort/meta_information.yaml")
    original = yaml.safe_load(source.read_text())
    import copy

    for change in ("checkpoint", "step", "predictor", "normalization", "task_config"):
        config = copy.deepcopy(original)
        if change == "checkpoint":
            config["assets"]["paths"]["conditioned"] = "/tmp/unverified/15236"
        elif change == "step":
            config["assets"]["identity"]["conditioned"]["step"] = 999
        elif change == "task_config":
            config["task_config"] = "/tmp/unverified.yaml"
        else:
            field = "predictor_weights" if change == "predictor" else change
            config["assets"]["paths"][field] = "/tmp/unverified.bin"
        target = tmp_path / f"{change}.yaml"
        target.write_text(yaml.safe_dump(config))
        with pytest.raises(ValueError, match="binding"):
            load_config(target)


def test_runtime_dependency_coverage_rejects_unbound_loaded_configs():
    import copy

    from latency_meta_mdp.meta.information.preflight import verify_runtime_dependencies

    root = Path.cwd()
    config = load_config(root / "configs/experiments/conveyor_sort/meta_information.yaml")
    for missing in (
        "configs/models/jepa/model.yaml",
        "configs/models/vision/dinov3_vits16_lvd1689m_224_v1.yaml",
        "configs/contracts/temporal/h50_e25_d20_k6_v1.yaml",
        "configs/runtime/latency/truncated_beta_family_8_65_400ms_v1.yaml",
        config["assets"]["paths"]["decoder_dir"] + "/run.json",
    ):
        broken = copy.deepcopy(config)
        broken["assets"]["files"].pop(missing, None)
        with pytest.raises(ValueError, match="runtime dependency"):
            verify_runtime_dependencies(root, broken)
