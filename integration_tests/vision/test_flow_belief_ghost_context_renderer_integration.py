from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from latency_meta_mdp.belief.flow.ghost_config import load_flow_belief_ghost_config
from latency_meta_mdp.belief.flow.ghost_context_renderer import (
    GhostContextRenderInput,
    render_flow_belief_ghost_context,
)
from latency_meta_mdp.belief.flow.ghost_environment import GhostEnvironmentAdapter
from latency_meta_mdp.belief_data import load_belief_episode
from latency_meta_mdp.control import load_action_contract
from latency_meta_mdp.task import load_task_spec, make_dynamic_grasp_lift_environment
from latency_meta_mdp.temporal_contract import load_temporal_contract
from latency_meta_mdp.terminal_absorbing_tail import build_terminal_absorbing_tail

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EPISODE = (
    _PROJECT_ROOT
    / "outputs/bulk/expert/panda-ball-formal-train-1000-1199-7571a4c/episodes/L1/seed_001180"
)
_QUALITY = (
    _PROJECT_ROOT / "outputs/analysis/flow_belief_quality_samples/dinov3-formal-v1-6a85f90/L1"
)


def test_context_renderer_preserves_real_l1_future_state_metrics(tmp_path: Path) -> None:
    if not _EPISODE.is_dir() or not (_QUALITY / "samples.npz").is_file():
        pytest.skip("ghost context integration requires formal L1 quality samples")
    selection = json.loads((_QUALITY / "selection.json").read_text(encoding="utf-8"))[0]
    with np.load(_QUALITY / "samples.npz", allow_pickle=False) as source:
        samples = {
            name: np.array(source[name][0], copy=True)
            for name in source.files
            if source[name].ndim > 1
        }
        delays = np.array(source["display_delay_ticks"], copy=True)
    identity = selection["identity"]
    context = GhostContextRenderInput(
        level=identity["level"],
        episode_id=identity["episode_id"],
        scene_seed=identity["scene_seed"],
        validation_offset=identity["validation_offset"],
        source_tick=identity["source_tick"],
        source_phase=selection["source_phase"],
        roles=tuple(selection["roles"]),
        display_tag="role pre_handoff",
        delay_ticks=delays,
        normalized_samples=samples["normalized_samples"],
        physical_samples=samples["physical_samples"],
        physical_targets=samples["physical_targets"],
        absorbing=samples["absorbing"],
    )
    episode = load_belief_episode(_EPISODE)
    temporal = load_temporal_contract(_PROJECT_ROOT / "configs/temporal/h50_e25_d20_k6_v1.yaml")
    action_contract = load_action_contract(
        _PROJECT_ROOT / "configs/control/panda_osc_pose_delta_v1.yaml"
    )
    tail = build_terminal_absorbing_tail(
        episode=episode,
        temporal_contract=temporal,
        action_contract=action_contract,
    )
    config = load_flow_belief_ghost_config(
        _PROJECT_ROOT / "configs/analysis/flow_belief_agentview_ghost_v1.yaml"
    )
    env = make_dynamic_grasp_lift_environment(
        spec=load_task_spec(_PROJECT_ROOT / "configs/task/dynamic_grasp_lift_l0.yaml"),
        seed=episode.scene_seed,
        offscreen=True,
        controller_config=action_contract.to_robosuite_config(),
    )
    try:
        result = render_flow_belief_ghost_context(
            context=context,
            output_dir=tmp_path / "h_0025",
            episode=episode,
            tail=tail,
            adapter=GhostEnvironmentAdapter(
                env=env,
                camera_name=config.camera_name,
                width=config.width,
                height=config.height,
            ),
            config=config,
        )
    finally:
        env.close()

    assert result.source_tick == 25
    assert result.panel.shape == (1200, 1920, 3)
    assert len(result.delay_metrics) == 5
    assert all(row.ground_truth_rgb_mae <= 1.0 for row in result.delay_metrics)
    assert (tmp_path / "h_0025/panel.png").is_file()
    assert (tmp_path / "h_0025/geometry.npz").is_file()
    assert (tmp_path / "h_0025/metrics.json").is_file()
