from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from latency_meta_mdp.legacy.policy.sft_asset_lock import load_sft_asset_lock
from latency_meta_mdp.policy.profile import load_sft_profile

_PROFILE_PATH = Path("configs/legacy/policy/pi05_panda_ball_full_sft_h50_v2.yaml")


def _payload() -> dict:
    profile = load_sft_profile(_PROFILE_PATH)
    return {
        "schema_version": 1,
        "lock_id": "franka_moving_ball_sft_assets_v1",
        "profile_id": profile.profile_id,
        "levels": {
            level: {
                "repo_id": profile.levels[level].repo_id,
                "data_revision": str(level) * 40,
                "asset_revision": ("a", "b", "c")[level - 1] * 40,
                "dataset_manifest_sha256": str(level) * 64,
                "norm_stats_sha256": ("d", "e", "f")[level - 1] * 64,
                "norm_manifest_sha256": ("7", "8", "9")[level - 1] * 64,
            }
            for level in (1, 2, 3)
        },
    }


def _write(path: Path, value: dict) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def test_load_sft_asset_lock_binds_three_profile_specific_revisions(tmp_path: Path) -> None:
    path = tmp_path / "assets.yaml"
    _write(path, _payload())
    profile = load_sft_profile(_PROFILE_PATH)

    lock = load_sft_asset_lock(path, profile=profile)

    assert lock.lock_id == "franka_moving_ball_sft_assets_v1"
    assert set(lock.levels) == {1, 2, 3}
    assert lock.level(2).repo_id == profile.levels[2].repo_id
    assert lock.level(2).data_revision == "2" * 40
    assert lock.level(2).asset_revision == "b" * 40
    with pytest.raises(ValueError, match="level"):
        lock.level(0)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["levels"].pop(3), "L1, L2, and L3"),
        (
            lambda value: value["levels"][1].update(repo_id="yypeng666/wrong"),
            "repo",
        ),
        (
            lambda value: value["levels"][1].update(asset_revision="main"),
            "revision",
        ),
        (
            lambda value: value["levels"][1].update(norm_stats_sha256="0" * 63),
            "SHA256",
        ),
        (
            lambda value: value["levels"][1].update(asset_revision="1" * 40),
            "differ",
        ),
    ],
)
def test_load_sft_asset_lock_rejects_inconsistent_or_mutable_inputs(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    payload = _payload()
    mutation(payload)
    path = tmp_path / "assets.yaml"
    _write(path, payload)

    with pytest.raises(ValueError, match=message):
        load_sft_asset_lock(path, profile=load_sft_profile(_PROFILE_PATH))
