from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from test_action_conditioned_jepa_admission_evaluation import (
    _batch,
    _constant_velocity_target,
    _FixedProprioRollout,
)
from test_action_conditioned_jepa_data import _record
from test_action_conditioned_jepa_physical_readout import (
    _LatentEncodedStateReadout,
    _matching_signal_episode,
)


class _CountingRollout(_FixedProprioRollout):
    def __init__(self, proprio: torch.Tensor) -> None:
        super().__init__(proprio)
        self.calls = 0

    def rollout_native(self, context):
        self.calls += 1
        return super().rollout_native(context)


def test_stride4_qualification_reuses_one_rollout_for_j2_and_j3(tmp_path: Path) -> None:
    """Catches running the AR5 predictor independently for each qualification metric."""

    from latency_meta_mdp.belief.jepa.ar.qualification import (
        evaluate_stride4_qualification_batch,
        summarize_stride4_qualification,
    )
    from latency_meta_mdp.belief.jepa.diagnostics.readout import (
        ObjectStateNormalization,
    )

    target = _constant_velocity_target()
    batch = _batch(target=target)
    model = _CountingRollout(target)
    record = _record(tmp_path, episode_id="episode-0", terminal_tick=30)
    episode = _matching_signal_episode(record)
    result = evaluate_stride4_qualification_batch(
        model=model,
        readout=_LatentEncodedStateReadout(),
        object_normalization=ObjectStateNormalization(
            mean=torch.zeros(6).numpy(),
            scale=torch.ones(6).numpy(),
        ),
        batch=batch,
        signal_episodes={record.episode_id: episode},
    )
    summary = summarize_stride4_qualification(
        (result,),
        signal_episodes={record.episode_id: episode},
    )

    assert model.calls == 1
    assert summary["j1_future_latent"]["source_count"] == 1
    assert summary["j2_future_proprio"]["source_count"] == 1
    assert summary["j3_gt_latent_ceiling"]["source_count"] == 1
    assert summary["j3_predicted_latent"]["source_count"] == 1
    assert summary["per_master"]["j2_qpos_rmse_rad"][0]["master_task_index"] == 0
    assert summary["per_master"]["j1_latent_rmse"][0]["master_task_index"] == 0
    assert summary["per_master"]["j3_predicted_position_rmse_m"][0]["master_task_index"] == 0


def test_j5_changes_only_mixture_weights_on_one_fixed_stride4_rollout() -> None:
    """Catches recomputing futures or losing D20 probability mass between latency laws."""

    from latency_meta_mdp.belief.jepa.ar.qualification import (
        evaluate_latency_mixture_invariants,
    )
    from latency_meta_mdp.belief.jepa.config import (
        load_jepa_temporal_sampling,
    )

    target = _constant_velocity_target()
    batch = _batch(target=target)
    rollout = _FixedProprioRollout(target).rollout_native(batch.context)
    laws = {
        "uniform": torch.full((1, 20), 0.05, dtype=torch.float32),
        "fast": torch.nn.functional.one_hot(torch.tensor([0]), 20).to(torch.float32),
    }
    result = evaluate_latency_mixture_invariants(
        rollout=rollout,
        probabilities_by_law=laws,
        sampling=load_jepa_temporal_sampling(
            Path("configs/models/jepa/stride4_80ms_history_160ms.yaml")
        ),
    )

    assert result["law_names"] == ["fast", "uniform"]
    assert result["fixed_future_tensors_shared"] is True
    assert result["maximum_probability_mass_error"] == 0.0
    assert result["expected_delay_ticks"]["fast"] == [4.0]
    assert result["expected_delay_ticks"]["uniform"] == pytest.approx([11.2])


def test_metric_aggregation_balances_contexts_then_episodes_then_masters(
    tmp_path: Path,
) -> None:
    """Catches long episodes or four realizations receiving extra statistical weight."""

    from latency_meta_mdp.belief.jepa.ar.qualification import (
        aggregate_metric_by_master,
    )

    episode_specs = (("a", 0), ("b", 0), ("c", 1), ("d", 1))
    signals = {}
    for episode_id, master in episode_specs:
        record = replace(
            _record(tmp_path, episode_id=episode_id, terminal_tick=30),
            logical_master_task_index=master,
        )
        signals[episode_id] = _matching_signal_episode(record)
    indices = tuple(
        replace(
            _batch(target=_constant_velocity_target()).indices[0],
            episode_id=episode_id,
        )
        for episode_id in ("a", "a", "b", "c", "d", "d")
    )
    result = aggregate_metric_by_master(
        values=np.asarray([[0.0], [10.0], [100.0], [20.0], [40.0], [60.0]]),
        valid=np.ones((6, 1), dtype=np.bool_),
        indices=indices,
        signal_episodes=signals,
    )

    assert result == [
        {"master_task_index": 0, "value": 52.5},
        {"master_task_index": 1, "value": 35.0},
    ]
