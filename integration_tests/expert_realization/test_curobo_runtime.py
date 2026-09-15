from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def test_real_curobo_worker_runs_official_franka_fk_and_motiongen() -> None:
    """A CuRobo install that only imports must not qualify the expert runtime."""
    from latency_meta_mdp.data.collection.curobo_runtime import run_worker_qualification

    qualification = run_worker_qualification()

    assert qualification.official_franka_fk_passed is True
    assert qualification.motion_gen_smoke_passed is True
    assert qualification.peak_vram_bytes > 0
    assert qualification.peak_vram_measurement_scope == "torch_allocator_max_memory_allocated"
    assert qualification.eligible is True
