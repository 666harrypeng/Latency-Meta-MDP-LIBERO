import os

import pytest

from latency_meta_mdp.meta.information import config


def test_unchanged_file_hash_is_reused_across_calls(tmp_path, monkeypatch):
    from latency_meta_mdp.meta.information import verification_cache as module

    path = tmp_path / "payload"
    path.write_bytes(b"original")
    cache = tmp_path / "cache"
    digest = module.cached_sha256(path, cache)
    monkeypatch.setattr(module, "sha256", lambda path: pytest.fail("unchanged file was reread"))
    assert module.cached_sha256(path, cache) == digest


def test_same_size_edit_with_restored_mtime_invalidates_cache(tmp_path):
    from latency_meta_mdp.meta.information.verification_cache import cached_sha256

    path = tmp_path / "payload"
    path.write_bytes(b"original")
    cache = tmp_path / "cache"
    previous = cached_sha256(path, cache)
    stat = path.stat()
    path.write_bytes(b"modified")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert cached_sha256(path, cache) == config.sha256(path) != previous


def test_cohort_admission_reuses_unchanged_payloads(tmp_path, monkeypatch):
    from latency_meta_mdp.meta.information import cohort, verification_cache

    attempt = tmp_path / "attempt"
    attempt.mkdir()
    (attempt / "payload").write_bytes(b"original")
    cohort.complete_attempt(attempt, binding={"sample": 1})
    cache = tmp_path / "cache"
    assert cohort.verify_complete(attempt, binding={"sample": 1}, verification_cache=cache)
    monkeypatch.setattr(verification_cache, "sha256", lambda p: pytest.fail("payload reread"))
    assert cohort.verify_complete(attempt, binding={"sample": 1}, verification_cache=cache)


def test_training_admission_checks_consumed_files_not_archive_media(tmp_path, monkeypatch):
    from latency_meta_mdp.meta.information import cohort, verification_cache

    attempt = tmp_path / "attempt"
    attempt.mkdir()
    (attempt / "payload").write_bytes(b"original")
    (attempt / "video.mp4").write_bytes(b"archive media")
    cohort.complete_attempt(attempt, binding={"sample": 1})
    calls = []
    real_hash = verification_cache.sha256

    def count(path):
        calls.append(path.name)
        return real_hash(path)

    monkeypatch.setattr(verification_cache, "sha256", count)
    cache = tmp_path / "cache"
    assert cohort.verify_complete(
        attempt, binding={"sample": 1}, verification_cache=cache, required_files=("payload",)
    )
    assert calls == ["payload"]
    with pytest.raises(ValueError, match="missing"):
        cohort.verify_complete(attempt, binding={"sample": 1}, required_files=("absent",))
    (attempt / "payload").write_bytes(b"modified")
    with pytest.raises(ValueError, match="hash"):
        cohort.verify_complete(
            attempt, binding={"sample": 1}, verification_cache=cache, required_files=("payload",)
        )
