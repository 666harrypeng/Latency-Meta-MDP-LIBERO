from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from latency_meta_mdp.rendering import render_task_setup

_TASK_CONFIG = Path("configs/task/dynamic_grasp_lift_l0.yaml")


def test_render_task_setup_writes_two_same_boundary_camera_images(tmp_path: Path) -> None:
    output_dir = tmp_path / "camera_setup"
    manifest_path = render_task_setup(
        task_config_path=_TASK_CONFIG,
        output_dir=output_dir,
        seed=7,
        width=128,
        height=128,
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == 2
    assert manifest["task_id"] == "dynamic_grasp_lift"
    assert manifest["agentview_resource_id"] == "libero_tabletop_agentview_v1"
    assert manifest["source_physics_step"] == 0
    assert manifest["source_formal_tick"] == 0
    assert manifest["source_time_us"] == 0
    assert set(manifest["cameras"]) == {"agentview", "robot0_eye_in_hand"}
    for camera_name, camera in manifest["cameras"].items():
        image_path = output_dir / camera["image"]
        assert Image.open(image_path).size == (128, 128)
        assert len(camera["sha256"]) == 64
        assert camera["ball_visible_pixels"] >= 25
        assert camera["model_pose_frame"] in {"world", "parent_body"}
        assert len(camera["model_position"]) == 3
        assert len(camera["model_quaternion_wxyz"]) == 4
        assert len(camera["world_position_at_capture"]) == 3
        assert len(camera["world_rotation_matrix_at_capture"]) == 3
    assert manifest["cameras"]["agentview"]["model_pose_frame"] == "world"
    assert manifest["cameras"]["robot0_eye_in_hand"]["model_pose_frame"] == "parent_body"


def test_render_task_setup_refuses_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "camera_setup"
    render_task_setup(
        task_config_path=_TASK_CONFIG,
        output_dir=output_dir,
        seed=7,
        width=64,
        height=64,
    )

    with pytest.raises(FileExistsError, match="render output already exists"):
        render_task_setup(
            task_config_path=_TASK_CONFIG,
            output_dir=output_dir,
            seed=7,
            width=64,
            height=64,
        )
