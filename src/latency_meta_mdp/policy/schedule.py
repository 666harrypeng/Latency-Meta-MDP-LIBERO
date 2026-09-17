"""Policy training requests and exposure-preserving optimization schedules."""

from __future__ import annotations

import re
from dataclasses import dataclass

from latency_meta_mdp.policy.profile import SFTProfile

_EXPERIMENT_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


@dataclass(frozen=True)
class SFTLaunchRequest:
    level: int | None
    experiment_name: str
    mode: str
    resume: bool
    device_count: int
    batch_size_override: int | None = None
    task_id: str | None = None

    def __post_init__(self) -> None:
        if self.level is None and self.task_id == "conveyor_sort":
            pass
        elif self.level not in (1, 2, 3):
            raise ValueError("SFT launch level must be 1, 2, or 3")
        if _EXPERIMENT_NAME.fullmatch(self.experiment_name) is None:
            raise ValueError("SFT experiment name is invalid")
        if self.mode not in {"smoke", "formal"}:
            raise ValueError("SFT launch mode must be smoke or formal")
        if type(self.device_count) is not int or self.device_count <= 0:
            raise ValueError("SFT device count must be a positive integer")
        if self.batch_size_override is not None:
            if type(self.batch_size_override) is not int or self.batch_size_override <= 0:
                raise ValueError("global batch-size override must be a positive integer")


@dataclass(frozen=True)
class SFTSchedule:
    num_train_steps: int
    rolling_save_interval: int
    milestone_interval: int
    expected_checkpoint_steps: tuple[int, ...]
    batch_size: int
    warmup_steps: int
    decay_steps: int


def resolve_sft_schedule(*, profile: SFTProfile, request: SFTLaunchRequest) -> SFTSchedule:
    """Keep sample exposure when a replicated-data-parallel run changes its batch.

    Each formal milestone rounds up to a whole global batch. The full run can
    therefore exceed the profile budget by fewer than three global batches.
    Equal sample exposure does not imply identical optimizer trajectories.
    """

    if profile.fsdp_devices != 1:
        raise ValueError("SFT currently requires replicated data parallelism (fsdp_devices=1)")
    batch = request.batch_size_override or profile.batch_size
    if batch % request.device_count:
        raise ValueError("global batch size must be divisible by the SFT device count")

    def scaled(steps: int) -> int:
        return (steps * profile.batch_size + batch - 1) // batch

    milestone = scaled(profile.keep_period)
    decay = 3 * milestone
    warmup = scaled(profile.warmup_steps)

    if request.mode == "smoke":
        if request.resume:
            return SFTSchedule(120, 20, 20, (100, 120), batch, warmup, decay)
        return SFTSchedule(100, 100, 100, (100,), batch, warmup, decay)
    if scaled(profile.save_interval) >= milestone:
        raise ValueError("global batch is too large to preserve distinct checkpoint intervals")
    return SFTSchedule(
        num_train_steps=decay,
        rolling_save_interval=scaled(profile.save_interval),
        milestone_interval=milestone,
        expected_checkpoint_steps=tuple(milestone * i for i in (1, 2, 3)),
        batch_size=batch,
        warmup_steps=warmup,
        decay_steps=decay,
    )
