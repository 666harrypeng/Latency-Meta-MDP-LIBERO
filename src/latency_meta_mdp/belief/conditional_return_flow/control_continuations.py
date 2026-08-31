"""Deterministic, controller-valid executable-control branch families."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from latency_meta_mdp.belief.conditional_return_flow.branch_contracts import (
    BranchCorpusConfig,
    ControlContinuationSpec,
    ExecutablePrefix,
)
from latency_meta_mdp.control import ActionContract


@dataclass(frozen=True)
class ControlContinuation:
    source_context_id: str
    spec: ControlContinuationSpec
    prefix: ExecutablePrefix

    def __post_init__(self) -> None:
        if not self.source_context_id:
            raise ValueError("control continuation source context cannot be empty")


def _validate_controls(value: np.ndarray, *, action_contract: ActionContract) -> np.ndarray:
    controls = np.asarray(value, dtype=np.float32)
    if controls.shape != (20, action_contract.action_dim) or not np.all(np.isfinite(controls)):
        raise ValueError("control continuation must be finite with shape [20, action_dim]")
    for row in controls:
        try:
            action_contract.split_action(row)
        except ValueError as error:
            raise ValueError("control continuation violates ActionContract") from error
    return controls


def _prefix(
    *,
    controls: np.ndarray,
    mask: np.ndarray,
    source_gripper: float,
    action_contract: ActionContract,
) -> ExecutablePrefix:
    return ExecutablePrefix(
        controls=_validate_controls(controls, action_contract=action_contract),
        from_active_buffer_mask=np.asarray(mask, dtype=bool),
        last_executed_gripper_command=source_gripper,
    )


def build_control_continuations(
    *,
    nominal_prefix: ExecutablePrefix,
    config: BranchCorpusConfig,
    action_contract: ActionContract,
    source_context_id: str,
) -> tuple[ControlContinuation, ...]:
    if not source_context_id:
        raise ValueError("source context ID cannot be empty")
    nominal_controls = _validate_controls(
        nominal_prefix.controls,
        action_contract=action_contract,
    )
    source_gripper = nominal_prefix.last_executed_gripper_command
    rows = [
        ControlContinuation(
            source_context_id=source_context_id,
            spec=ControlContinuationSpec.nominal(),
            prefix=nominal_prefix,
        )
    ]

    hold = np.zeros_like(nominal_controls)
    hold[:, -1] = source_gripper
    rows.append(
        ControlContinuation(
            source_context_id=source_context_id,
            spec=ControlContinuationSpec(kind="hold", arm_scale=None, prefix_real_ticks=None),
            prefix=_prefix(
                controls=hold,
                mask=np.zeros(20, dtype=bool),
                source_gripper=source_gripper,
                action_contract=action_contract,
            ),
        )
    )

    for scale in config.arm_scale_factors:
        scaled = np.array(nominal_controls, copy=True)
        scaled[:, : action_contract.arm_dim] *= scale
        rows.append(
            ControlContinuation(
                source_context_id=source_context_id,
                spec=ControlContinuationSpec(
                    kind=f"arm_scale_{scale:g}",
                    arm_scale=float(scale),
                    prefix_real_ticks=None,
                ),
                prefix=_prefix(
                    controls=scaled,
                    mask=nominal_prefix.from_active_buffer_mask,
                    source_gripper=source_gripper,
                    action_contract=action_contract,
                ),
            )
        )

    for real_ticks in config.prefix_hold_ticks:
        prefix_hold = np.zeros_like(nominal_controls)
        prefix_hold[:real_ticks] = nominal_controls[:real_ticks]
        prefix_hold[real_ticks:, -1] = nominal_controls[real_ticks - 1, -1]
        mask = np.zeros(20, dtype=bool)
        mask[:real_ticks] = nominal_prefix.from_active_buffer_mask[:real_ticks]
        rows.append(
            ControlContinuation(
                source_context_id=source_context_id,
                spec=ControlContinuationSpec(
                    kind=f"prefix_hold_{real_ticks}",
                    arm_scale=None,
                    prefix_real_ticks=real_ticks,
                ),
                prefix=_prefix(
                    controls=prefix_hold,
                    mask=mask,
                    source_gripper=source_gripper,
                    action_contract=action_contract,
                ),
            )
        )
    return tuple(rows)
