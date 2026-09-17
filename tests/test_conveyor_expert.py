from dataclasses import replace
from pathlib import Path

import numpy as np

from latency_meta_mdp.envs.conveyor.config import load_conveyor_spec
from latency_meta_mdp.envs.conveyor.task import make_conveyor_runtime


def test_expert_physically_delivers_single_parcel_with_open_release():
    from latency_meta_mdp.envs.conveyor.expert import ConveyorExpert, load_expert_spec

    spec = replace(
        load_conveyor_spec(Path("configs/tasks/conveyor_sort/surface.yaml")),
        spawn_count=None,
        supply_ticks=100,
    )
    runtime = make_conveyor_runtime(spec, seed=27, offscreen=False)
    try:
        expert = ConveyorExpert(
            runtime, load_expert_spec(Path("configs/data/expert/conveyor_surface.yaml"))
        )
        runtime.executor.initialize()
        actions, positions = [], []
        while not runtime.env.done:
            decision = expert.next_action()
            actions.append(decision.action)
            if runtime.world.ledger.statuses.get(0) == "active":
                positions.append(runtime.world.position(0))
            runtime.executor.step_formal(decision.action)
        summary = runtime.world.ledger.summary()
        assert summary["spawned"] == summary["successes"] == 1, (summary, expert.events)
        assert summary["misses"] == summary["timeouts"] == 0
        assert np.max(np.abs(actions)) <= 1
        assert max(p[2] for p in positions) > spec.goal_center[2] + 0.05
        assert [e["event"] for e in runtime.world.goal.events] == [
            "held_in_goal",
            "release_requested",
            "released_in_goal",
        ]
        phases = [e["phase"] for e in expert.events]
        for phase in ("close", "lift", "transfer", "lower", "release"):
            assert phase in phases
        assert phases.index("lift") < phases.index("transfer") < phases.index("release")
    finally:
        runtime.env.close()


def test_expert_continues_between_deliveries_without_reset():
    from latency_meta_mdp.envs.conveyor.expert import ConveyorExpert, load_expert_spec

    spec = replace(
        load_conveyor_spec(Path("configs/tasks/conveyor_sort/surface.yaml")),
        spawn_count=None,
        supply_ticks=400,
    )
    runtime = make_conveyor_runtime(spec, seed=27, offscreen=False)
    try:
        expert = ConveyorExpert(
            runtime, load_expert_spec(Path("configs/data/expert/conveyor_surface.yaml"))
        )
        runtime.executor.initialize()
        ticks = []
        while not runtime.env.done:
            runtime.executor.step_formal(expert.next_action().action)
            ticks.append(runtime.executor.ledger.formal_tick_index)
        summary = runtime.world.ledger.summary()
        assert summary["spawned"] >= 2
        assert summary["successes"] == summary["spawned"], (summary, expert.events)
        assert "retreat" in [e["phase"] for e in expert.events]
        assert ticks == list(range(1, ticks[-1] + 1))
        assert np.isclose(runtime.env.sim.data.time, ticks[-1] * 0.02)
    finally:
        runtime.env.close()


def test_expert_delivers_close_pair_before_either_ball_leaves():
    from latency_meta_mdp.envs.conveyor.expert import ConveyorExpert, load_expert_spec

    spec = replace(
        load_conveyor_spec(Path("configs/tasks/conveyor_sort/surface.yaml")),
        spawn_count=None,
        supply_ticks=100,
        intervals={k: (45, 45) for k in ("moderate", "sparse", "burst")},
    )
    runtime = make_conveyor_runtime(spec, seed=0, offscreen=False)
    try:
        config = replace(
            load_expert_spec(Path("configs/data/expert/conveyor_surface.yaml")),
            translation_clip_m=0.05,
        )
        expert = ConveyorExpert(runtime, config)
        runtime.executor.initialize()
        while not runtime.env.done:
            runtime.executor.step_formal(expert.next_action().action)
        summary = runtime.world.ledger.summary()
        assert summary["spawned"] == 2
        assert summary["successes"] == 2, (summary, expert.events)
        assert summary["misses"] == summary["timeouts"] == 0
    finally:
        runtime.env.close()


def test_expert_delivers_every_parcel_in_the_fixed_ten_protocol():
    from latency_meta_mdp.envs.conveyor.expert import ConveyorExpert, load_expert_spec

    spec = load_conveyor_spec(Path("configs/tasks/conveyor_sort/surface.yaml"))
    assert spec.spawn_count == 10
    runtime = make_conveyor_runtime(spec, seed=4, offscreen=False)
    try:
        expert = ConveyorExpert(
            runtime, load_expert_spec(Path("configs/data/expert/conveyor_surface.yaml"))
        )
        runtime.executor.initialize()
        while not runtime.env.done:
            runtime.executor.step_formal(expert.next_action().action)
        summary = runtime.world.ledger.summary()
        assert summary["successes"] == summary["spawned"] == 10, summary
        assert summary["misses"] == summary["timeouts"] == 0
    finally:
        runtime.env.close()
