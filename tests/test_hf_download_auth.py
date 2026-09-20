"""Downloads respect the Hub client's configured auth, including anonymous use."""

import json

import pytest
from huggingface_hub.utils import _headers


@pytest.fixture(params=[None, "unit-test-token"])
def check_auth(request, monkeypatch):
    monkeypatch.setattr(_headers, "get_token", lambda: request.param)
    monkeypatch.setattr(_headers.constants, "HF_HUB_DISABLE_IMPLICIT_TOKEN", False)

    def check(kwargs):
        headers = _headers.build_hf_headers(token=kwargs.get("token"))
        expected = None if request.param is None else f"Bearer {request.param}"
        assert headers.get("authorization") == expected

    return check


@pytest.mark.parametrize("schema", [1, 2])
def test_clean_dataset_download_uses_configured_auth(tmp_path, monkeypatch, check_auth, schema):
    import huggingface_hub

    from latency_meta_mdp.policy.clean_data import resolve_clean_inputs

    class DownloadReached(Exception):
        pass

    def download(**kwargs):
        check_auth(kwargs)
        raise DownloadReached

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    job = {"schema_version": schema}
    if schema == 1:
        job.update(dataset_repo="owner/data", dataset_revision="a" * 40)
    else:
        job["dataset"] = dict(kind="huggingface", repo_id="owner/data", revision="a" * 40)
    with pytest.raises(DownloadReached):
        resolve_clean_inputs(job, tmp_path)


def test_source_and_forecast_models_use_configured_auth(tmp_path, monkeypatch, check_auth):
    import huggingface_hub

    from latency_meta_mdp.data.forecast.assets import download_forecast_models, stage_source

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "data.bin").write_bytes(b"data")
    (snapshot / "manifest.json").write_text(json.dumps({"artifacts": {"data.bin": {"bytes": 4}}}))
    calls = []

    def single(*args, **kwargs):
        check_auth(kwargs)
        calls.append("manifest")
        return str(snapshot / "manifest.json")

    def download(*args, **kwargs):
        check_auth(kwargs)
        calls.append("snapshot")
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", single)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    job = dict(
        source_repo="owner/data",
        source_revision="a" * 40,
        predictor_repo="owner/belief",
        predictor_revision="b" * 40,
        decoder_repo="owner/decoder",
        decoder_revision="c" * 40,
    )
    target = stage_source(job, tmp_path / "source")
    assert (target / "data.bin").read_bytes() == b"data"
    assert len(download_forecast_models(job, tmp_path / "work")) == 2
    assert calls == ["manifest", "snapshot", "snapshot", "snapshot"]
