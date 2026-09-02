from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src/latency_meta_mdp/expert_realization"
TASK3_MODULES = (
    PACKAGE_ROOT / "recording_contracts.py",
    PACKAGE_ROOT / "recording_artifacts.py",
    PACKAGE_ROOT / "artifacts.py",
)
FORBIDDEN = {
    "latency_meta_mdp.expert",
    "latency_meta_mdp.expert_collection",
    "latency_meta_mdp.bulk_collection",
    "latency_meta_mdp.pilot_collection",
    "latency_meta_mdp.recording",
    "latency_meta_mdp.episode_artifacts",
    "latency_meta_mdp.formal_corpus",
    "latency_meta_mdp.vision_feature_cache",
    "latency_meta_mdp.vision_feature_cache_run",
    "latency_meta_mdp.belief_data",
    "latency_meta_mdp.belief_data_artifact",
}


def test_task3_modules_have_no_forbidden_high_level_import_edges() -> None:
    """Break caught: the supposedly independent schema transitively revives historical semantics."""
    imported: set[str] = set()
    for path in TASK3_MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
    assert imported.isdisjoint(FORBIDDEN)


def test_clean_process_import_graph_does_not_load_historical_recording_or_collection_modules() -> (
    None
):
    """Break caught: an allowed-looking direct import has a forbidden transitive dependency."""
    code = """
import sys
import latency_meta_mdp.expert_realization.recording_contracts
import latency_meta_mdp.expert_realization.recording_artifacts
import latency_meta_mdp.expert_realization.artifacts
forbidden = {
    'latency_meta_mdp.expert',
    'latency_meta_mdp.expert_collection',
    'latency_meta_mdp.bulk_collection',
    'latency_meta_mdp.pilot_collection',
    'latency_meta_mdp.recording',
    'latency_meta_mdp.episode_artifacts',
    'latency_meta_mdp.formal_corpus',
    'latency_meta_mdp.vision_feature_cache',
    'latency_meta_mdp.vision_feature_cache_run',
    'latency_meta_mdp.belief_data',
    'latency_meta_mdp.belief_data_artifact',
}
loaded = sorted(forbidden.intersection(sys.modules))
if loaded:
    raise SystemExit('forbidden modules loaded: ' + ','.join(loaded))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env={"PYTHONPATH": str(PROJECT_ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
