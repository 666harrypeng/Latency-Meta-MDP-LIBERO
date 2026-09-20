"""Reloaded upstream classes must not reuse a previous source context's type hints."""

import ast
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")


@pytest.mark.parametrize("override", [False, True])
def test_training_cache_respects_environment_in_temporary_copy(tmp_path, monkeypatch, override):
    from latency_meta_mdp.policy.openpi.source import temporary_patched_openpi_copy

    target = tmp_path / "cache 'quoted'"
    if override:
        monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(target))
    else:
        monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    canonical = Path("third_party/openpi/scripts/train.py").read_bytes()
    with temporary_patched_openpi_copy(
        openpi_root=Path("third_party/openpi"),
        patch_paths=(),
        expected_revision="15a9616a00943ada6c20a0f158e3adb39df2ccac",
    ) as root:
        tree = ast.parse((root / "scripts/train.py").read_text())
        # Execute the actual cache configuration call without starting model training.
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "jax.config.update":
                eval(
                    compile(ast.Expression(node), "cache-call", "eval"),
                    {
                        "jax": SimpleNamespace(
                            config=SimpleNamespace(update=lambda *a: calls.append(a))
                        ),
                        "epath": SimpleNamespace(Path=Path),
                    },
                )
        expected = str(target) if override else str(Path("~/.cache/jax").expanduser())
        assert ("jax_compilation_cache_dir", expected) in calls
    assert Path("third_party/openpi/scripts/train.py").read_bytes() == canonical


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
