from __future__ import annotations

import sys


def test_gaussian_canonical_package_and_legacy_imports_share_objects() -> None:
    from latency_meta_mdp.belief.common.feature_data import FeatureBeliefSample
    from latency_meta_mdp.belief.gaussian.config import GaussianBeliefConfig
    from latency_meta_mdp.belief_feature_data import FeatureBeliefSample as LegacySample
    from latency_meta_mdp.gaussian_belief_config import (
        GaussianBeliefConfig as LegacyConfig,
    )

    assert LegacySample is FeatureBeliefSample
    assert LegacyConfig is GaussianBeliefConfig


def test_importing_gaussian_package_does_not_import_flow_package() -> None:
    from latency_meta_mdp.belief.gaussian.config import GaussianBeliefConfig

    assert GaussianBeliefConfig.__module__ == "latency_meta_mdp.belief.gaussian.config"
    assert not any(
        name == "latency_meta_mdp.belief.flow"
        or name.startswith("latency_meta_mdp.belief.flow.")
        for name in sys.modules
    )
