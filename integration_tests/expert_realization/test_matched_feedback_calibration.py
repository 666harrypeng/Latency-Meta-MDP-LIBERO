from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_matched_k6_feedback_worker_succeeds_on_all_three_levels(tmp_path: Path) -> None:
    """Gate: exact shared-prefix feedback execution yields complete physical timing rows."""
    from latency_meta_mdp.data.collection.calibration import (
        TimingCalibrationAttemptRequest,
        load_timing_calibration_attempt_result,
        run_timing_calibration_attempt_process,
        write_timing_calibration_attempt_request,
    )

    root = Path.cwd()
    config_paths = {
        "task_config_sha256": root / "configs/tasks/moving_ball/task/dynamic_grasp_lift_l0.yaml",
        "runtime_config_sha256": root / "configs/runtime/robosuite_v1.yaml",
        "controller_config_sha256": root / "configs/runtime/control/panda_osc_pose_delta_v1.yaml",
        "expert_config_sha256": root / "configs/data/expert/panda_ball_feedback_v1.yaml",
    }
    for level in (1, 2, 3):
        request = TimingCalibrationAttemptRequest(
            calibration_id="panda-ball-feedback-calibration-v1",
            logical_task_index=0,
            level=level,
            master_task_seed=12985087823104956951,
            motion_config_sha256=hashlib.sha256(
                (
                    root / f"configs/tasks/moving_ball/motion/dynamic_grasp_lift_l{level}.yaml"
                ).read_bytes()
            ).hexdigest(),
            **{
                name: hashlib.sha256(path.read_bytes()).hexdigest()
                for name, path in config_paths.items()
            },
        )
        request_path = tmp_path / f"request-l{level}.json"
        result_path = tmp_path / f"result-l{level}.json"
        write_timing_calibration_attempt_request(request, request_path)
        run_timing_calibration_attempt_process(
            request_path,
            result_path,
            worker_python=Path(sys.executable),
        )
        row = load_timing_calibration_attempt_result(result_path)
        assert row.level == level
        assert row.terminal_status == "success"
        assert row.handoff_time_us is not None
        assert row.terminal_time_us is not None
        assert row.k6_planning_start_sha256
