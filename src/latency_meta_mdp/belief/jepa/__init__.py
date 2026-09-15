"""Action-conditioned JEPA return-belief implementation."""

from latency_meta_mdp.belief.jepa.ar.return_belief import (
    LoadedReturnLatentBelief,
    assemble_return_latent_belief,
    load_return_latent_belief,
    weighted_future_proprio,
    weighted_future_visual_latents,
    write_return_latent_belief,
)
from latency_meta_mdp.belief.jepa.ar.runtime import JepaRuntimeHistory
from latency_meta_mdp.belief.jepa.backbone import (
    ActionConditionedJepaPredictor,
)
from latency_meta_mdp.belief.jepa.config import (
    ActionConditionedJepaConfig,
    JepaSourceProtocol,
    JepaTemporalSampling,
    load_action_conditioned_jepa_config,
    load_jepa_source_protocol,
    load_jepa_temporal_sampling,
)
from latency_meta_mdp.belief.jepa.contracts import (
    FutureLatentRollout,
    JepaLaunchSupportContract,
    LaunchContextBatch,
    ReturnLatentBeliefBatch,
)
from latency_meta_mdp.belief.jepa.corpus import (
    ActionConditionedJepaCorpus,
    JepaEpisodeRecord,
    JepaProprioNormalization,
    JepaSampleIndex,
    JepaTrainingSample,
    compute_jepa_proprio_normalization,
    load_action_conditioned_jepa_corpus,
    load_jepa_proprio_normalization,
    write_jepa_proprio_normalization,
)

__all__ = [
    "ActionConditionedJepaConfig",
    "ActionConditionedJepaCorpus",
    "ActionConditionedJepaPredictor",
    "FutureLatentRollout",
    "JepaEpisodeRecord",
    "JepaLaunchSupportContract",
    "JepaProprioNormalization",
    "JepaRuntimeHistory",
    "JepaSourceProtocol",
    "JepaTemporalSampling",
    "JepaSampleIndex",
    "JepaTrainingSample",
    "LaunchContextBatch",
    "LoadedReturnLatentBelief",
    "ReturnLatentBeliefBatch",
    "assemble_return_latent_belief",
    "compute_jepa_proprio_normalization",
    "load_action_conditioned_jepa_config",
    "load_action_conditioned_jepa_corpus",
    "load_jepa_proprio_normalization",
    "load_jepa_source_protocol",
    "load_jepa_temporal_sampling",
    "load_return_latent_belief",
    "weighted_future_proprio",
    "weighted_future_visual_latents",
    "write_return_latent_belief",
    "write_jepa_proprio_normalization",
]
