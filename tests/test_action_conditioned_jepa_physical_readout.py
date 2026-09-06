from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from test_action_conditioned_jepa_data import _record

from latency_meta_mdp.belief.action_conditioned_jepa.temporal_view import (
    SharedJepaSampleIndex,
)


def _matching_signal_episode(record):
    from latency_meta_mdp.belief.action_conditioned_jepa.temporal_signal_audit import (
        TemporalSignalEpisode,
    )

    count = record.terminal_tick + 1
    ticks = np.arange(count, dtype=np.float32)
    return TemporalSignalEpisode(
        record=record,
        object_position=np.stack((ticks, ticks + 1.0, ticks + 2.0), axis=1),
        object_linear_velocity=np.stack((ticks + 3.0, ticks + 4.0, ticks + 5.0), axis=1),
        eef_position=np.zeros((count, 3), dtype=np.float32),
        motion_segment_index=np.zeros(count, dtype=np.int64),
        left_pad_contact=np.zeros(count, dtype=np.bool_),
        right_pad_contact=np.zeros(count, dtype=np.bool_),
        handoff_state=tuple("driven" for _ in range(count)),
    )


def test_dual_view_readout_uses_both_full_patch_views_without_latent_gradients() -> None:
    """Catches dropping one camera or updating frozen JEPA latents."""

    from latency_meta_mdp.belief.action_conditioned_jepa.physical_readout import (
        DualViewObjectStateReadout,
    )

    torch.manual_seed(4)
    model = DualViewObjectStateReadout(project_root=Path(__file__).resolve().parents[1])
    latents = torch.randn(1, 1, 2, 196, 384, dtype=torch.float32, requires_grad=True)
    base = model(latents)
    changed_agent = latents.detach().clone()
    changed_agent[:, :, 0, 17] += 1.0
    changed_wrist = latents.detach().clone()
    changed_wrist[:, :, 1, 29] += 1.0

    assert base.shape == (1, 1, 6)
    assert not torch.equal(base, model(changed_agent))
    assert not torch.equal(base, model(changed_wrist))
    base.sum().backward()
    assert latents.grad is None
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_object_state_dataset_reads_each_gt_frame_once_and_normalizes_targets(
    tmp_path: Path,
) -> None:
    """Catches training the probe on repeated rollout contexts or unnormalized mixed-unit state."""

    from latency_meta_mdp.belief.action_conditioned_jepa.physical_readout import (
        ObjectStateReadoutDataset,
        compute_object_state_normalization,
    )

    record = _record(tmp_path, terminal_tick=6)
    episode = _matching_signal_episode(record)
    normalization = compute_object_state_normalization((episode,))
    dataset = ObjectStateReadoutDataset(
        records=(record,),
        episodes={record.episode_id: episode},
        normalization=normalization,
    )
    latent, target = dataset[4]

    assert len(dataset) == 7
    assert latent.shape == (2, 196, 384)
    assert latent.dtype == torch.float16
    torch.testing.assert_close(
        target,
        normalization.normalize(torch.tensor([4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=torch.float32)),
    )
    torch.testing.assert_close(
        normalization.denormalize(target),
        torch.tensor([4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=torch.float32),
    )


class _LatentEncodedStateReadout(torch.nn.Module):
    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return latents[:, :, 0, 0, :6].float()


def test_object_state_evaluation_uses_one_frozen_readout_for_gt_and_predicted_latents() -> None:
    """Catches mixing the GT-latent probe ceiling with predicted-latent error."""

    from latency_meta_mdp.belief.action_conditioned_jepa.physical_readout import (
        ObjectStateNormalization,
        evaluate_object_state_latents,
    )

    normalization = ObjectStateNormalization(
        mean=np.zeros(6, dtype=np.float32),
        scale=np.ones(6, dtype=np.float32),
    )
    target = torch.zeros(1, 2, 6, dtype=torch.float32)
    gt_latents = torch.zeros(1, 2, 2, 196, 384, dtype=torch.float16)
    predicted_latents = gt_latents.clone()
    predicted_latents[:, :, 0, 0, :3] = 0.002
    predicted_latents[:, :, 0, 0, 3:6] = 0.04

    ceiling = evaluate_object_state_latents(
        readout=_LatentEncodedStateReadout(),
        normalization=normalization,
        visual_latents=gt_latents,
        target_object_state=target,
        absorbing=torch.zeros(1, 2, dtype=torch.bool),
    )
    predicted = evaluate_object_state_latents(
        readout=_LatentEncodedStateReadout(),
        normalization=normalization,
        visual_latents=predicted_latents,
        target_object_state=target,
        absorbing=torch.zeros(1, 2, dtype=torch.bool),
    )

    np.testing.assert_allclose(ceiling.position_rmse_m, 0.0, atol=0.0)
    np.testing.assert_allclose(ceiling.velocity_rmse_m_s, 0.0, atol=0.0)
    np.testing.assert_allclose(predicted.position_rmse_m, 0.002, atol=2e-6)
    np.testing.assert_allclose(predicted.velocity_rmse_m_s, 0.04, atol=2e-5)


def test_object_state_targets_follow_stride4_ticks_and_terminal_absorption(tmp_path: Path) -> None:
    """Catches reading object targets from the source tick or beyond the terminal boundary."""

    from latency_meta_mdp.belief.action_conditioned_jepa.physical_readout import (
        gather_object_state_targets,
    )

    record = _record(tmp_path, terminal_tick=10)
    episode = _matching_signal_episode(record)
    index = SharedJepaSampleIndex(
        level=3,
        split="train",
        episode_id=record.episode_id,
        source_tick=5,
        boundary_disposition="certified_absorbing_extension",
    )
    targets = gather_object_state_targets(
        indices=(index,),
        native_delay_ticks=(4, 8, 12, 16, 20),
        episodes={record.episode_id: episode},
    )

    torch.testing.assert_close(targets[0, :, 0], torch.tensor([9.0, 10.0, 10.0, 10.0, 10.0]))
    torch.testing.assert_close(targets[0, :, 3], torch.tensor([12.0, 0.0, 0.0, 0.0, 0.0]))


def test_object_state_summary_keeps_dynamic_and_absorbing_errors_separate() -> None:
    """Catches repeated terminal targets making grounded rollout quality look better."""

    from latency_meta_mdp.belief.action_conditioned_jepa.physical_readout import (
        ObjectStateBatchMetrics,
        summarize_object_state_metrics,
    )

    metrics = ObjectStateBatchMetrics(
        position_rmse_m=np.asarray([[0.001, 0.002, 1.0]], dtype=np.float64),
        velocity_rmse_m_s=np.asarray([[0.01, 0.02, 2.0]], dtype=np.float64),
        dynamic_mask=np.asarray([[True, True, False]], dtype=np.bool_),
        absorbing_mask=np.asarray([[False, False, True]], dtype=np.bool_),
    )
    summary = summarize_object_state_metrics((metrics,))

    assert summary["dynamic_position_rmse_m"] == 0.0015
    assert summary["dynamic_velocity_rmse_m_s"] == 0.015
    assert summary["absorbing_position_rmse_m"] == 1.0
    assert summary["absorbing_velocity_rmse_m_s"] == 2.0
    assert summary["per_anchor_dynamic_position_rmse_m"] == [0.001, 0.002, None]
