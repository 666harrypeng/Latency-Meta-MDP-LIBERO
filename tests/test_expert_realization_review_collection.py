from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def test_review_request_is_exactly_three_by_three_by_three_and_nontraining() -> None:
    """Break caught: the requested review silently becomes a family-balanced or training corpus."""
    from latency_meta_mdp.expert_realization.review_collection import load_review_request

    request = load_review_request(Path("configs/collection/panda_ball_smooth_review_3x3x3.yaml"))

    assert request.levels == (1, 2, 3)
    assert request.task_instance_count == 3
    assert request.reserve_task_instance_count == 3
    assert request.realizations_per_task == 3
    assert request.requested_trajectory_count == 27
    assert request.bounded_review_only is True
    assert request.training_authorized is False
    assert request.review_video_fps == 25
    assert request.to_formal_config().reserve_task_indices == (3, 4, 5)


def test_real_ffmpeg_review_video_contains_both_views_and_all_frames(tmp_path: Path) -> None:
    """Break caught: review publication writes an unreadable or single-view video."""
    from latency_meta_mdp.expert_realization.review_collection import encode_review_video

    agentview = np.zeros((4, 16, 20, 3), dtype=np.uint8)
    wrist = np.zeros((4, 16, 20, 3), dtype=np.uint8)
    agentview[..., 0] = 255
    wrist[..., 1] = 255
    target = tmp_path / "review.mp4"
    encode_review_video(
        agentview_rgb=agentview,
        wrist_rgb=wrist,
        phase_by_tick=("shared_prefix", "smooth_approach", "grasp_funnel", "lift"),
        level=2,
        logical_task_index=1,
        realization_slot=0,
        family="lateral_arc",
        fps=25,
        target=target,
    )

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_read_frames,r_frame_rate",
            "-of",
            "json",
            str(target),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream == {
        "width": 40,
        "height": 48,
        "r_frame_rate": "25/1",
        "nb_read_frames": "4",
    }


def test_final_manifest_assembles_exactly_three_complete_groups_per_level(tmp_path: Path) -> None:
    """Break caught: successful rows from incomplete task groups enter the final 27-video view."""
    from latency_meta_mdp.expert_realization.review_collection import assemble_review_manifest

    for level in (1, 2, 3):
        for task_index in range(3):
            group = tmp_path / f"level-{level}" / f"task-{task_index:03d}-seed-{task_index}"
            group.mkdir(parents=True)
            (group / "group_result.json").write_text(
                json.dumps(
                    {
                        "admitted": True,
                        "selected_realization_slots": [0, 1, 2],
                        "trajectory_statuses": [],
                    }
                )
            )
            for slot in range(3):
                realization = group / f"realization-{slot:02d}"
                realization.mkdir()
                (realization / "summary.json").write_text(
                    json.dumps(
                        {
                            "level": level,
                            "logical_task_index": task_index,
                            "master_task_seed": task_index,
                            "realization_slot": slot,
                            "family": "canonical_direct",
                            "terminal_status": "success",
                            "terminal_reason": "lift_succeeded",
                            "video": f"level-{level}/task-{task_index}/r{slot}.mp4",
                        }
                    )
                )

    manifest_path = assemble_review_manifest(
        target=tmp_path,
        config_path=Path("configs/collection/panda_ball_smooth_review_3x3x3.yaml"),
    )
    manifest = json.loads(manifest_path.read_text())

    assert manifest["complete"] is True
    assert manifest["admitted_video_count"] == 27
    assert len(manifest["trajectories"]) == 27
    assert manifest["admitted_task_instance_count_by_level"] == {"1": 3, "2": 3, "3": 3}


def test_phase_timeline_covers_prefix_decisions_and_terminal_frame() -> None:
    """Break caught: video labels are shifted one tick relative to recorded actions."""
    from latency_meta_mdp.expert_realization.review_collection import phase_timeline

    decisions = tuple(
        SimpleNamespace(
            source_formal_tick=tick,
            phase=SimpleNamespace(value=phase),
        )
        for tick, phase in ((5, "smooth_approach"), (6, "grasp_funnel"), (7, "lift"))
    )

    assert phase_timeline(decisions, boundary_count=9) == (
        "shared_prefix",
        "shared_prefix",
        "shared_prefix",
        "shared_prefix",
        "shared_prefix",
        "smooth_approach",
        "grasp_funnel",
        "lift",
        "terminal",
    )
