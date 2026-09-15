import json
import os
import sys
import time

import pytest


def test_local_workers_write_disjoint_cases_and_resume_existing_results(tmp_path, monkeypatch):
    from latency_meta_mdp.meta import rollouts

    cohort = tmp_path / "cases.json"
    cases = [{"master_index": n, "policy_seed": 0} for n in range(3)]
    cohort.write_text(json.dumps({"cases": cases}))
    worker = tmp_path / "worker.py"
    worker.write_text("""import json,sys,os
from pathlib import Path
cohort,output,index,count=sys.argv[1:]
output=Path(output)
for case in json.loads(Path(cohort).read_text())["cases"][int(index)::int(count)]:
    name=f"master-{case['master_index']:03d}-seed-0-nominal"
    target=output/(name+".json")
    if target.exists(): continue
    (output/(name+".npz")).write_bytes(b"fixture")
    target.write_text(json.dumps({"case":case,"identity":{"regime":"nominal"},
        "terminated":True,"truncated":False,"meta_replay":{"path":name+".npz"},
        "gpu":os.environ["CUDA_VISIBLE_DEVICES"]}))
""")
    monkeypatch.setattr(
        rollouts,
        "rollout_command",
        lambda cfg, **kw: [
            sys.executable,
            str(worker),
            str(kw["cohort"]),
            str(kw["output"]),
            str(kw["worker"]),
            str(kw["workers"]),
        ],
    )
    cfg = {"gpu_ids": [2, 5], "record_video": False, "cohorts": {"train": {"nominal": str(cohort)}}}
    output = tmp_path / "results"
    entries = rollouts.run_rollouts(cfg, "train", output)
    assert len(entries) == 3
    rows = [json.loads(open(e["result"]).read()) for e in entries]
    assert [r["gpu"] for r in rows] == ["2", "5", "2"]
    assert rollouts.run_rollouts(cfg, "train", output) == entries
    (output / "nominal/master-001-seed-0-nominal.npz").unlink()
    with pytest.raises(FileNotFoundError):
        rollouts.collect_results(cohort, output / "nominal", "nominal", collect=True, video=False)


def test_worker_failure_stops_its_running_sibling(tmp_path, monkeypatch):
    from latency_meta_mdp.meta import rollouts

    cohort = tmp_path / "cases.json"
    cohort.write_text(json.dumps({"cases": [{"master_index": 1}, {"master_index": 2}]}))
    worker = tmp_path / "worker.py"
    ready = tmp_path / "ready"
    worker.write_text("""import sys,os,time
from pathlib import Path
ready=Path(sys.argv[2])
if sys.argv[1]=="0":
    for _ in range(100):
        if ready.exists(): sys.exit(3)
        time.sleep(.01)
    sys.exit(4)
ready.write_text(str(os.getpid()))
time.sleep(60)
""")
    monkeypatch.setattr(
        rollouts,
        "rollout_command",
        lambda cfg, **kw: [sys.executable, str(worker), str(kw["worker"]), str(ready)],
    )
    cfg = {"gpu_ids": [0, 1], "record_video": False, "cohorts": {"train": {"nominal": str(cohort)}}}
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="worker failed"):
        rollouts.run_rollouts(cfg, "train", tmp_path / "results")
    assert time.monotonic() - started < 15
    with pytest.raises(ProcessLookupError):
        os.kill(int(ready.read_text()), 0)
