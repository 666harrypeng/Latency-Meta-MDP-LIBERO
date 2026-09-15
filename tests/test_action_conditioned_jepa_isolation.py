from __future__ import annotations

import json
import os
import subprocess
import sys

from latency_meta_mdp.io.paths import repository_root


def test_new_package_does_not_import_historical_learned_models() -> None:
    forbidden = (
        "latency_meta_mdp.legacy.belief.flow",
        "latency_meta_mdp.legacy.belief.gaussian",
        "latency_meta_mdp.legacy.belief.common",
        "latency_meta_mdp.legacy.belief.causal_return",
        "latency_meta_mdp.legacy.belief.conditional_return_flow",
    )
    project_root = repository_root()
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project_root / "src")
    code = """
import json
import sys
import latency_meta_mdp.belief.jepa
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
