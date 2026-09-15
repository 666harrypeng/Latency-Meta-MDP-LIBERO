import json
import shutil
import subprocess

import numpy as np
import pytest

from latency_meta_mdp.runtime.policy_execution import PolicyObservation


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg required"
)
def test_two_camera_video_keeps_formal_frame_count_and_both_views(tmp_path):
    from latency_meta_mdp.io.policy_video import DualCameraVideoWriter

    destination = tmp_path / "episode.mp4"
    main = np.zeros((32, 32, 3), np.uint8)
    main[..., 0] = 255
    wrist = np.zeros_like(main)
    wrist[..., 2] = 255
    with DualCameraVideoWriter(destination) as writer:
        for tick in range(4):
            writer.write(PolicyObservation(tick, main, wrist, np.zeros(16)))
    info = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate,nb_frames",
                "-of",
                "json",
                str(destination),
            ]
        )
    )["streams"][0]
    assert info["width"] == 64 and info["height"] == 56
    assert info["r_frame_rate"] == "50/1" and info["nb_frames"] == "4"
    frame = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(destination),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
    )
    image = np.frombuffer(frame, np.uint8).reshape(56, 64, 3)
    assert image[40, 16, 0] > 240 and image[40, 48, 2] > 240
    np.testing.assert_array_equal(main[..., 0], 255)
    with pytest.raises(FileExistsError):
        DualCameraVideoWriter(destination)
