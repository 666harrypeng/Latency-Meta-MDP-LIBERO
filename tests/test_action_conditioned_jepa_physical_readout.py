from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from test_action_conditioned_jepa_data import _record


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
        normalization.normalize(
            torch.tensor([4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=torch.float32)
        ),
    )
    torch.testing.assert_close(normalization.denormalize(target), torch.tensor(
        [4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=torch.float32
    ))
