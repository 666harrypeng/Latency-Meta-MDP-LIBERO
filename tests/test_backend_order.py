from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from latency_meta_mdp.backend import FormalStepExecutor
from latency_meta_mdp.timing import ClockLedger


@dataclass
class FakePlant:
    physics_dt_seconds: float = 0.002
    fail_step2_call: int | None = None
    calls: list[str] = field(default_factory=list)
    step2_calls: int = 0
    sim_time_seconds: float = 0.0

    def write_world_at(self, current_time_us: int) -> None:
        assert current_time_us == round(self.sim_time_seconds * 1_000_000)
        self.calls.append(f"world@{current_time_us}")

    def step1(self) -> None:
        self.calls.append(f"step1@{round(self.sim_time_seconds * 1_000_000)}")

    def materialize_boundary(self, ledger: ClockLedger) -> str:
        self.calls.append(f"snapshot@{ledger.time_us}")
        return f"boundary-{ledger.formal_tick_index}"

    def apply_control(self, action: object, *, policy_step: bool) -> None:
        del action
        self.calls.append(
            f"control@{round(self.sim_time_seconds * 1_000_000)}:policy={policy_step}"
        )

    def step2(self) -> None:
        self.step2_calls += 1
        self.calls.append(f"step2@{round(self.sim_time_seconds * 1_000_000)}")
        if self.step2_calls == self.fail_step2_call:
            raise RuntimeError("injected step2 failure")
        self.sim_time_seconds += self.physics_dt_seconds


def test_formal_step_obeys_world_step1_snapshot_control_step2_order() -> None:
    ledger = ClockLedger(physics_dt_us=2_000, formal_tick_us=20_000)
    plant = FakePlant()
    executor = FormalStepExecutor(plant=plant, ledger=ledger)

    initial = executor.initialize()
    final = executor.step_formal(action="hold")

    assert initial == "boundary-0"
    assert final == "boundary-1"
    assert plant.calls[:5] == [
        "world@0",
        "step1@0",
        "snapshot@0",
        "control@0:policy=True",
        "step2@0",
    ]
    assert plant.calls[-3:] == ["world@20000", "step1@20000", "snapshot@20000"]
    assert sum(call.startswith("step2@") for call in plant.calls) == 10
    assert sum(":policy=True" in call for call in plant.calls) == 1
    assert sum(":policy=False" in call for call in plant.calls) == 9


def test_failed_step2_does_not_advance_ledger_and_faults_executor() -> None:
    ledger = ClockLedger(physics_dt_us=2_000, formal_tick_us=20_000)
    plant = FakePlant(fail_step2_call=1)
    executor = FormalStepExecutor(plant=plant, ledger=ledger)
    executor.initialize()

    with pytest.raises(RuntimeError, match="injected step2 failure"):
        executor.step_formal(action="hold")

    assert ledger.physics_step_index == 0
    assert ledger.formal_tick_index == 0
    assert executor.faulted
    with pytest.raises(RuntimeError, match="faulted"):
        executor.step_formal(action="hold")


def test_initialize_is_required_and_cannot_be_repeated() -> None:
    executor = FormalStepExecutor(
        plant=FakePlant(),
        ledger=ClockLedger(physics_dt_us=2_000, formal_tick_us=20_000),
    )

    with pytest.raises(RuntimeError, match="not initialized"):
        executor.step_formal(action="hold")
    executor.initialize()
    with pytest.raises(RuntimeError, match="already initialized"):
        executor.initialize()
