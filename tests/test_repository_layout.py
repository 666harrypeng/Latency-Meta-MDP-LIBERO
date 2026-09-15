"""Release-facing namespace boundaries, independent of GPU execution."""

import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRIDGES = {
    "openpi_belief_adapter.py",
    "openpi_belief_prefix.py",
    "openpi_forecast.py",
    "openpi_rtc.py",
    "policy_forecast_dataset.py",
    "policy_return_data.py",
}


def test_package_root_contains_only_package_and_pinned_patch_bridges():
    assert {p.name for p in (ROOT / "src/latency_meta_mdp").glob("*.py")} == BRIDGES | {
        "__init__.py"
    }


def test_primary_packages_do_not_import_legacy_implementations():
    failures = []
    for directory in ("envs", "data", "belief", "policy", "meta", "runtime", "io"):
        for p in (ROOT / "src/latency_meta_mdp" / directory).rglob("*.py"):
            for node in ast.walk(ast.parse(p.read_text())):
                refs = (
                    [node.module]
                    if isinstance(node, ast.ImportFrom)
                    else [a.name for a in node.names]
                    if isinstance(node, ast.Import)
                    else []
                )
                for ref in refs:
                    if ref and ref.startswith("latency_meta_mdp.legacy"):
                        failures.append(str(p.relative_to(ROOT)) + ":" + ref)
    assert not failures, failures


def test_published_data_contract_bytes_are_preserved():
    p = ROOT / "configs/contracts/policy/pi05_state16_h50.yaml"
    assert (
        hashlib.sha256(p.read_bytes()).hexdigest()
        == "8edce4aaeab6cac49f19f4195ff6687817540b28d036a0cbe21b6c36c1e6b9d8"
    )


def test_training_scripts_are_thin_module_entrypoints():
    for name in (
        "train_clean_policy",
        "train_belief",
        "train_conditioned_policy",
        "train_meta_policy",
    ):
        path = ROOT / "scripts" / f"{name}.py"
        assert path.is_file()
        assert len(path.read_text().splitlines()) < 20
