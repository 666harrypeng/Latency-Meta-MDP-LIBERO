"""Reloaded upstream classes must not reuse a previous source context's type hints."""

import importlib
import sys
from pathlib import Path

import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")


def test_type_hints_follow_reloaded_openpi_source():
    from latency_meta_mdp.policy.openpi.source import temporary_patched_openpi_copy

    source = """from typing import Any
from beartype import beartype
from openpi.training.utils import TrainState
@beartype
def typed(value: Any) -> tuple[TrainState, Any]:
    return value, None
"""
    for _ in range(2):
        with temporary_patched_openpi_copy(
            openpi_root=Path("third_party/openpi"),
            patch_paths=(),
            expected_revision="15a9616a00943ada6c20a0f158e3adb39df2ccac",
        ) as root:
            (root / "src/_openpi_type_fixture.py").write_text(source)
            sys.path.insert(0, str(root / "src"))
            module = importlib.import_module("_openpi_type_fixture")
            # This fixture checks class identity, without creating a model or optimizer.
            state = object.__new__(module.TrainState)
            for field in module.TrainState.__dataclass_fields__:
                object.__setattr__(state, field, None)
            assert module.typed(state)[0] is state
