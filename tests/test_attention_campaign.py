from pathlib import Path

import pytest

from latency_meta_mdp.meta.information.attention_campaign import attention_cycle_loop
from latency_meta_mdp.meta.information.campaign import CampaignJournal


def test_accumulated_online_replay_latest_parent_and_report_only_regression(tmp_path):
    recipe = dict(
        startup_sources=[dict(path="old", group="startup")],
        reward_regression_policy="report_only",
        cycles=[
            dict(generation={"name": f"g{i}"}, fitted_iterations=2, optimizer_updates=3)
            for i in range(2)
        ],
    )
    calls = []
    fail = [True]

    def collect(index, definition, parent):
        calls.append(("collect", index, str(parent)))
        return dict(artifacts={}, source=dict(path=f"new{index}", group="latest"))

    def fit(index, definition, parent, sources):
        calls.append(("fit", index, [s["group"] for s in sources]))
        if fail[0]:
            fail[0] = False
            raise RuntimeError("validation failed after saved update")
        candidates = [
            dict(
                checkpoint=str(tmp_path / f"{index}-{i}"),
                metrics={"selection_score": (5 - index - i) / 60},
            )
            for i in range(2)
        ]
        return dict(
            artifacts={},
            latest=candidates[-1]["checkpoint"],
            candidates=candidates,
            optimizer_updates=6,
            validation_episodes=18,
        )

    journal = CampaignJournal(tmp_path, binding={"v": 2})
    args = dict(
        journal=journal,
        recipe=recipe,
        initial=Path("initial"),
        initial_metrics=None,
        collect=collect,
        fit=fit,
    )
    with pytest.raises(RuntimeError):
        attention_cycle_loop(**args)
    result = attention_cycle_loop(**args)
    assert len([c for c in calls if c[0] == "collect"]) == 2
    assert calls[-2] == ("collect", 2, str(tmp_path / "1-1"))
    assert calls[-1] == ("fit", 2, ["startup", "old", "latest"])
    assert result["best"] == str(tmp_path / "1-0")
    assert result["latest"] == str(tmp_path / "2-1")
    assert result["validation_episodes"] == 36
    assert attention_cycle_loop(**args) == result


def test_external_checkpoint_binding_detects_replacement_and_corruption(tmp_path):
    import json

    from latency_meta_mdp.meta.information.attention_campaign import checkpoint_artifacts
    from latency_meta_mdp.meta.information.config import sha256

    checkpoint = tmp_path / "input"
    checkpoint.mkdir()
    weights = checkpoint / "state.pt"
    receipt = checkpoint / "complete.json"
    weights.write_bytes(b"first")
    receipt.write_text(json.dumps({"files": {"state.pt": sha256(weights)}}))
    journal = CampaignJournal(tmp_path / "run", binding=checkpoint_artifacts(checkpoint))
    assert journal is not None
    weights.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint artifact"):
        checkpoint_artifacts(checkpoint)
    receipt.write_text(json.dumps({"files": {"state.pt": sha256(weights)}}))
    with pytest.raises(ValueError, match="binding"):
        CampaignJournal(tmp_path / "run", binding=checkpoint_artifacts(checkpoint))
