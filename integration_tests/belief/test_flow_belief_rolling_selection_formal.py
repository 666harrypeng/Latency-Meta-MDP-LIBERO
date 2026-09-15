from __future__ import annotations

import pytest

from latency_meta_mdp.data.vision.contracts import load_vision_encoder_spec
from latency_meta_mdp.io.paths import repository_root
from latency_meta_mdp.legacy.belief.common.feature_corpus import (
    load_level_feature_belief_corpus,
)
from latency_meta_mdp.legacy.belief.flow.rolling_config import (
    load_flow_belief_rolling_config,
)
from latency_meta_mdp.legacy.belief.flow.rolling_selection import (
    select_rolling_seed_windows,
)
from latency_meta_mdp.legacy.vision_probe_data import ProbeSplit

_PROJECT_ROOT = repository_root()
_SOURCE = (
    _PROJECT_ROOT / "outputs/bulk/expert/panda-ball-formal-train-1000-1199-7571a4c/manifest.json"
)
_CACHE = (
    _PROJECT_ROOT
    / "outputs/derived/vision_features/dinov3-vits16-formal-1000-1199-408dbe3/manifest.json"
)


def test_formal_l1_seed_1180_rolling_windows_are_validation_aligned() -> None:
    if not _SOURCE.is_file() or not _CACHE.is_file():
        pytest.skip("rolling selection integration requires formal source and cache")
    corpus = load_level_feature_belief_corpus(
        project_root=_PROJECT_ROOT,
        source_bulk_manifest=_SOURCE,
        cache_run_manifest=_CACHE,
        expected_spec=load_vision_encoder_spec(
            _PROJECT_ROOT / "configs/models/vision/dinov3_vits16_lvd1689m_224_v1.yaml"
        ),
        temporal_config_path=_PROJECT_ROOT / "configs/contracts/temporal/h50_e25_d20_k6_v1.yaml",
        latency_law_path=_PROJECT_ROOT
        / "configs/runtime/latency/truncated_beta_5_26_400ms_v1.yaml",
        split_plan_path=_PROJECT_ROOT / "configs/legacy/data/formal_belief_train_val_v1.yaml",
        level=1,
    )

    selection = select_rolling_seed_windows(
        corpus=corpus,
        scene_seed=1180,
        config=load_flow_belief_rolling_config(
            _PROJECT_ROOT / "configs/legacy/analysis/flow_belief_rolling_inspection_v1.yaml"
        ),
    )

    assert selection.level == 1
    assert selection.scene_seed == 1180
    assert corpus.temporal_contract.launch_trigger_horizon == 25
    assert selection.critical_start_tick == 25
    assert selection.approach_start_tick == 56
    assert selection.critical_end_tick == 88
    assert selection.handoff_tick == 97
    assert tuple(row.source_tick for row in selection.windows) == (
        25,
        35,
        45,
        55,
        56,
        65,
        68,
    )
    assert all(row.level == 1 and row.scene_seed == 1180 for row in selection.windows)
    assert all(row.source_tick + 20 <= row.critical_end_tick for row in selection.windows)
    assert all(row.source_phase in ("pregrasp", "approach") for row in selection.windows)
    assert all(
        corpus.materialize(
            split=ProbeSplit.VALIDATION,
            offset=row.validation_offset,
        ).source_tick
        == row.source_tick
        for row in selection.windows
    )
