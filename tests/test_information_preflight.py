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


def test_asset_digest_cache_reuses_only_unchanged_files_and_expected_hash(tmp_path, monkeypatch):
    import os

    from latency_meta_mdp.meta.information import preflight

    file, cache = tmp_path / "weights", tmp_path / "verified"
    file.write_bytes(b"original")
    assets = {"files": {"weights": preflight.sha256(file)}}
    digest, calls = preflight.sha256, []

    def counted(path):
        calls.append(path)
        return digest(path)

    monkeypatch.setattr(preflight, "sha256", counted)
    verify_assets(tmp_path, assets, cache_dir=cache)
    verify_assets(tmp_path, assets, cache_dir=cache)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="asset"):
        verify_assets(tmp_path, {"files": {"weights": "wrong"}}, cache_dir=cache)
    before = file.stat()
    file.write_bytes(b"mutated!")
    os.utime(file, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match="asset"):
        verify_assets(tmp_path, assets, cache_dir=cache)
    assert len(calls) == 2  # ctime catches changes even with restored size/mtime.


def test_asset_digest_cache_rejects_mutation_during_hash(tmp_path, monkeypatch):
    from latency_meta_mdp.meta.information import preflight

    file = tmp_path / "weights"
    file.write_bytes(b"original")
    digest = preflight.sha256(file)

    def changed(path):
        path.write_bytes(b"mutated!")
        return digest

    monkeypatch.setattr(preflight, "sha256", changed)
    with pytest.raises(ValueError, match="changed during"):
        verify_assets(tmp_path, {"files": {"weights": digest}}, cache_dir=tmp_path / "verified")
    assert not list((tmp_path / "verified").glob("*.json"))


def test_asset_digest_cache_concurrent_first_publish(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from latency_meta_mdp.meta.information import preflight

    file, cache = tmp_path / "weights", tmp_path / "verified"
    file.write_bytes(b"original")
    digest, barrier = preflight.sha256(file), Barrier(3)

    def simultaneous(path):
        barrier.wait(timeout=5)
        return digest

    monkeypatch.setattr(preflight, "sha256", simultaneous)
    assets = {"files": {"weights": digest}}
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(
            pool.map(lambda _: verify_assets(tmp_path, assets, cache_dir=cache), range(3))
        )
    assert all(result == results[0] for result in results)
    assert verify_assets(tmp_path, assets, cache_dir=cache) == results[0]
    assert len(list(cache.iterdir())) == 1


def test_asset_digest_cache_does_not_skip_checkpoint_coverage(tmp_path):
    from latency_meta_mdp.meta.information.config import sha256

    for name in ("conditioned", "clean"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "weights").write_bytes(b"original")
    assets = {
        "paths": {name: name for name in ("conditioned", "clean")},
        "files": {
            f"{name}/weights": sha256(tmp_path / name / "weights")
            for name in ("conditioned", "clean")
        },
    }
    cache = tmp_path / "verified"
    verify_assets(tmp_path, assets, cache_dir=cache)
    (tmp_path / "conditioned" / "unbound").write_bytes(b"new file")
    with pytest.raises(ValueError, match="coverage"):
        verify_assets(tmp_path, assets, cache_dir=cache)


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
