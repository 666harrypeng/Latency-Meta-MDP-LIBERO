from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from latency_meta_mdp.legacy.belief.flow.ghost_context_renderer import (
    GhostContextRenderInput,
)


def _context() -> GhostContextRenderInput:
    return GhostContextRenderInput(
        level=1,
        episode_id="l1-seed-001180-attempt-000",
        scene_seed=1180,
        validation_offset=0,
        source_tick=25,
        source_phase="pregrasp",
        roles=("pre_handoff",),
        display_tag="role pre_handoff",
        delay_ticks=np.asarray([1, 5, 10, 15, 20]),
        normalized_samples=np.zeros((5, 32, 22), dtype=np.float32),
        physical_samples=np.zeros((5, 32, 22), dtype=np.float32),
        physical_targets=np.zeros((5, 22), dtype=np.float32),
        absorbing=np.zeros(5, dtype=np.bool_),
    )


def test_ghost_context_input_is_typed_readonly_and_shape_checked() -> None:
    context = _context()

    assert context.delay_ticks.tolist() == [1, 5, 10, 15, 20]
    assert context.normalized_samples.flags.writeable is False
    assert context.physical_samples.shape == (5, 32, 22)

    with pytest.raises(ValueError, match="shape"):
        replace(context, physical_targets=np.zeros((4, 22), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        replace(context, absorbing=np.ones(4, dtype=np.bool_))
