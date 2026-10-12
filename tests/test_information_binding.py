import pytest

from latency_meta_mdp.meta.information.config import validate_binding, validate_splits


def test_binding_does_not_accept_partial_or_changed_identity():
    validate_binding(
        {"mode": "bound", "assets": {"norm": "a"}}, {"mode": "bound", "assets": {"norm": "a"}}
    )
    with pytest.raises(ValueError, match="binding"):
        validate_binding({"assets": {"norm": "a"}}, {"assets": {"norm": "b"}})
    with pytest.raises(ValueError, match="binding"):
        validate_binding({"mode": "bound"}, {"mode": "bound", "extra": 1})


def test_splits_reject_cross_scene_leakage_and_boolean_seeds():
    validate_splits({"train": [1, 2], "validation": [3], "test": [4]}, excluded=[5])
    for data in (
        {"train": [1], "validation": [1], "test": [3]},
        {"train": [True], "validation": [2], "test": [3]},
        {"train": [1, 1], "validation": [2], "test": [3]},
    ):
        with pytest.raises(ValueError):
            validate_splits(data, excluded=[])
    with pytest.raises(ValueError, match="exposed"):
        validate_splits({"train": [1], "validation": [2], "test": [3]}, excluded=[3])
