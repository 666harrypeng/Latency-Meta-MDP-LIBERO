from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
from test_action_conditioned_jepa_data import _record


class _NativeRows:
    def __init__(self, record):
        self.states = record.proprio_physical[:-1]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        return {
            "state": self.states[index].copy(),
            "image": np.zeros((8, 8, 3), np.uint8),
            "wrist_image": np.zeros((8, 8, 3), np.uint8),
            "prompt": "Grasp the moving ball and lift it.",
            "object_pose": np.full(7, 999.0),
        }


class _Predictions:
    def read(self, episode_id, source_tick):
        return {
            "visual": np.full((5, 2, 196, 384), source_tick, np.float16),
            "proprio": np.full((5, 16), source_tick, np.float32),
        }


def _dataset(tmp_path, mode, *, split="train"):
    from latency_meta_mdp.policy_return_data import MatchedReturnPolicyDataset

    record = _record(tmp_path, terminal_tick=30, split=split)
    pmf = np.arange(1, 21, dtype=float)
    pmf /= pmf.sum()
    dataset = MatchedReturnPolicyDataset(
        native_dataset=_NativeRows(record),
        records=(record,),
        episode_rows=(
            {
                "episode_id": record.episode_id,
                "frame_count": 30,
                "level": 3,
                "logical_master_task_index": record.logical_master_task_index,
            },
        ),
        episode_probabilities={record.episode_id: pmf},
        mode=mode,
        predictions=_Predictions() if mode == "predicted_mixture" else None,
    )
    return dataset, record, pmf


@pytest.mark.parametrize("mode", ["predicted_mixture", "gt_mixture"])
def test_main_inputs_do_not_reveal_the_supervision_delay(tmp_path, mode):
    dataset, record, _ = _dataset(tmp_path, mode)
    first, second = dataset[0], dataset[1]
    assert len(dataset) == (30 - 10) * 20
    assert set(first) == {
        "image",
        "wrist_image",
        "state",
        "prompt",
        "actions",
        "actions_is_pad",
        "action_loss_weight",
        "return_belief",
    }
    for key in first["return_belief"]:
        np.testing.assert_array_equal(first["return_belief"][key], second["return_belief"][key])
    np.testing.assert_array_equal(first["actions"][0], record.controls[11])
    np.testing.assert_array_equal(second["actions"][0], record.controls[12])
    assert dataset.sample_identity(0)["supervision_delay_ticks"] == 1


def test_oracle_uses_exact_dense_delay_and_preserves_pairing(tmp_path):
    oracle, record, _ = _dataset(tmp_path, "known_delay_oracle")
    sample = oracle[0]
    assert "return_belief" not in sample
    assert sample["known_delay_oracle"]["known_delay_ticks"] == 1
    np.testing.assert_array_equal(sample["known_delay_oracle"]["visual"], 11)
    np.testing.assert_array_equal(sample["actions"][0], record.controls[11])


def test_uniform_delay_enumeration_recovers_episode_pmf_objective(tmp_path):
    dataset, _, pmf = _dataset(tmp_path, "gt_mixture")
    weights = np.array([dataset.sample_identity(i)["loss_weight"] for i in range(20)])
    costs = np.arange(20, dtype=float) ** 2
    np.testing.assert_allclose(np.mean(weights * costs), np.dot(pmf, costs), rtol=1e-6)


def test_success_tail_has_absorbing_state_but_no_invented_action_label(tmp_path):
    dataset, record, _ = _dataset(tmp_path, "gt_mixture")
    final = dataset[len(dataset) - 1]
    assert final["actions_is_pad"].all()
    np.testing.assert_array_equal(final["actions"], 0)
    np.testing.assert_array_equal(final["return_belief"]["visual"], 30)
    np.testing.assert_array_equal(final["return_belief"]["proprio"][:, 7:14], 0)
    np.testing.assert_array_equal(final["return_belief"]["proprio"][:, 15], 0)
    np.testing.assert_array_equal(final["state"], record.proprio_physical[29])


def test_spawn_serialization_does_not_copy_the_feature_cache(tmp_path):
    dataset, _, _ = _dataset(tmp_path, "gt_mixture")
    dataset[0]
    serialized = pickle.dumps(dataset)
    assert len(serialized) < 100_000
    restored = pickle.loads(serialized)
    np.testing.assert_array_equal(
        restored[0]["return_belief"]["visual"], dataset[0]["return_belief"]["visual"]
    )


def test_policy_development_rejects_validation_records(tmp_path):
    with pytest.raises(ValueError, match="train"):
        _dataset(tmp_path, "gt_mixture", split="validation")


def test_episode_law_key_preserves_existing_generation():
    from latency_meta_mdp.latency_law_family import load_episode_latency_law_family

    family = load_episode_latency_law_family(
        Path("configs/latency/truncated_beta_family_8_65_400ms_v1.yaml")
    )
    by_key = family.sample_for_key(assignment_key=12)
    legacy = family.sample_for_episode(
        level=3, episode_id="l3-seed-000012-attempt-000", scene_seed=12
    )
    np.testing.assert_array_equal(by_key.probabilities, legacy.probabilities)
