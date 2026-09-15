from __future__ import annotations

import ast
import subprocess
import sys

from latency_meta_mdp.io.paths import repository_root

PROJECT_ROOT = repository_root()
PACKAGE_ROOT = PROJECT_ROOT / "src/latency_meta_mdp/data/collection"
TASK3_MODULES = (
    PACKAGE_ROOT / "recording_contracts.py",
    PACKAGE_ROOT / "recording_artifacts.py",
    PACKAGE_ROOT / "artifacts.py",
)
FORBIDDEN = {
    "latency_meta_mdp.envs.expert",
    "latency_meta_mdp.data.expert_collection",
    "latency_meta_mdp.data.bulk_collection",
    "latency_meta_mdp.data.pilot_collection",
    "latency_meta_mdp.data.recording",
    "latency_meta_mdp.io.episode_artifacts",
    "latency_meta_mdp.data.formal_corpus",
    "latency_meta_mdp.data.vision.cache",
    "latency_meta_mdp.data.vision.extract",
    "latency_meta_mdp.legacy.belief_data",
    "latency_meta_mdp.legacy.belief_data_artifact",
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
import latency_meta_mdp.data.collection.recording_contracts
import latency_meta_mdp.data.collection.recording_artifacts
import latency_meta_mdp.data.collection.artifacts
forbidden = {
    'latency_meta_mdp.envs.expert',
    'latency_meta_mdp.data.expert_collection',
    'latency_meta_mdp.data.bulk_collection',
    'latency_meta_mdp.data.pilot_collection',
    'latency_meta_mdp.data.recording',
    'latency_meta_mdp.io.episode_artifacts',
    'latency_meta_mdp.data.formal_corpus',
    'latency_meta_mdp.data.vision.cache',
    'latency_meta_mdp.data.vision.extract',
    'latency_meta_mdp.legacy.belief_data',
    'latency_meta_mdp.legacy.belief_data_artifact',
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
