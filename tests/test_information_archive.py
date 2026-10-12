from dataclasses import replace

import numpy as np
import pytest
from test_rtc_forecast_provider import _context, _obs

from latency_meta_mdp.meta.information.archive import SourceArchive


def test_source_archive_deduplicates_frames_and_rejects_conflicting_reuse(tmp_path):
    archive = SourceArchive(tmp_path / "source")
    history = [_obs(t) for t in (2, 6, 10)]
    controls = np.zeros((8, 7))
    archive.record(history, controls, _context(10), kind="observe")
    archive.record(history, controls, _context(10), kind="predict")
    assert len(list((tmp_path / "source/frames").glob("*.npz"))) == 3
    assert len(list((tmp_path / "source/queries").glob("*.npz"))) == 2
    different = replace(history[-1], image=np.ones((8, 8, 3), np.uint8))
    with pytest.raises(ValueError, match="frame"):
        archive.record([different], controls, _context(10), kind="observe")
