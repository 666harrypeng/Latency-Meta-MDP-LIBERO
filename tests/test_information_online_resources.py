import json
import os

import pytest


def test_learner_uses_physical_gpu_order_before_source_admission(tmp_path, monkeypatch):
    from latency_meta_mdp.meta.information import config, online_training

    class AdmissionReached(Exception):
        pass

    def admit(*args):
        assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "4,5,6,7"
        raise AdmissionReached

    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(config, "load_config", lambda _: {})
    monkeypatch.setattr(online_training, "validate_mixture", admit)
    for name in ("offline", "online"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "cohort.json").write_text(json.dumps({}))
    compatibility = tmp_path / "compatibility.json"
    compatibility.write_text("{}")
    with pytest.raises(AdmissionReached):
        online_training.run_online_training(
            tmp_path / "config.yaml",
            tmp_path / "offline",
            tmp_path / "online",
            tmp_path / "output",
            root=tmp_path,
            checkpoint=tmp_path / "checkpoint",
            compatibility_path=compatibility,
            audit_path=tmp_path / "audit.json",
            training_gpus="4,5,6,7",
        )
