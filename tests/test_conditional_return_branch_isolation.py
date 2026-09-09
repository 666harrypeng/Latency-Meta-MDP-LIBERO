from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

FORBIDDEN_PREFIXES = (
    "latency_meta_mdp.belief.flow",
    "latency_meta_mdp.belief.gaussian",
    "latency_meta_mdp.belief.common",
    "latency_meta_mdp.belief.causal_return",
)


def test_branch_contract_import_does_not_load_historical_beliefs() -> None:
    project_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project_root / "src")
    script = """
import json
import sys
import latency_meta_mdp.belief.conditional_return_flow.branch_artifacts
import latency_meta_mdp.belief.conditional_return_flow.branch_collection
import latency_meta_mdp.belief.conditional_return_flow.branch_contracts
import latency_meta_mdp.belief.conditional_return_flow.branch_runtime
import latency_meta_mdp.belief.conditional_return_flow.control_continuations
import latency_meta_mdp.belief.conditional_return_flow.executable_prefix
import latency_meta_mdp.belief.conditional_return_flow.source_corpus
print(json.dumps(sorted(sys.modules)))
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    loaded = json.loads(completed.stdout)

    assert not any(
        module_name.startswith(prefix)
        for module_name in loaded
        for prefix in FORBIDDEN_PREFIXES
    )
