from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_new_package_does_not_import_historical_learned_models() -> None:
    forbidden = (
        "latency_meta_mdp.belief.flow",
        "latency_meta_mdp.belief.gaussian",
        "latency_meta_mdp.belief.common",
        "latency_meta_mdp.belief.causal_return",
        "latency_meta_mdp.belief.conditional_return_flow",
    )
    project_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project_root / "src")
    code = """
import json
import sys
import latency_meta_mdp.belief.action_conditioned_jepa
print(json.dumps(sorted(sys.modules)))
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    loaded = set(json.loads(result.stdout))

    assert not any(
        name == prefix or name.startswith(prefix + ".") for prefix in forbidden for name in loaded
    )
