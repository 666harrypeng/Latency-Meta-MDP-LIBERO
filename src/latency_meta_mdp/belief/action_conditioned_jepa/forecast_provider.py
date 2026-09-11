"""Shared frozen batch prediction and request-time RGB-history encoding for RTC."""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

from latency_meta_mdp.belief.action_conditioned_jepa.contracts import ForecastQuery
from latency_meta_mdp.policy_forecast import DecodedForecast


def load_forecast_components(assets, *, project_root, device):
    """Load exactly the frozen predictor/decoder pair bound to the policy training."""
    from pathlib import Path

    from latency_meta_mdp.artifacts import sha256_file
    from latency_meta_mdp.belief.action_conditioned_jepa.config import (
        load_action_conditioned_jepa_config,
    )
    from latency_meta_mdp.belief.action_conditioned_jepa.data_adapter import (
        load_jepa_proprio_normalization,
    )
    from latency_meta_mdp.belief.action_conditioned_jepa.direct_prediction import (
        DirectJepaPredictor,
        load_direct_prediction_weights,
    )
    from latency_meta_mdp.belief.action_conditioned_jepa.visual_decoder_evaluation import (
        load_visual_decoder,
    )
    from latency_meta_mdp.hf_dino_encoder import HfDinoPatchEncoder

    root = Path(project_root)
    identity = assets["forecast_identity"]
    if identity["predictor_architecture"] != "jepa_direct_q20_history_stride4_w3_v1":
        raise ValueError("forecast evaluation requires the trained Direct architecture")
    weights = root / assets["predictor_weights"]
    normalization = root / assets["normalization"]
    decoder_root = root / assets["decoder_dir"]
    for path, key in (
        (weights, "predictor_sha256"),
        (normalization, "jepa_normalization_sha256"),
        (decoder_root / "model.safetensors", "decoder_sha256"),
    ):
        if sha256_file(path) != identity[key]:
            raise ValueError(f"forecast evaluation artifact mismatch: {key}")
    level = assets["level"]
    config = load_action_conditioned_jepa_config(
        model_path=root / "configs/belief/action_conditioned_jepa/model.yaml",
        level_path=root / f"configs/belief/action_conditioned_jepa/l{level}.yaml",
        temporal_sampling_path=(
            root / "configs/belief/action_conditioned_jepa/stride4_80ms_history_160ms.yaml"
        ),
    )
    norm = load_jepa_proprio_normalization(normalization)
    if norm.level != level:
        raise ValueError("forecast normalization level mismatch")
    predictor = DirectJepaPredictor(
        backbone_config=config, proprio_normalization=norm, project_root=root
    )
    load_direct_prediction_weights(predictor, weights)
    decoder = load_visual_decoder(decoder_root, device=device)
    engine = FrozenForecastEngine(predictor, decoder, device=device)
    encoder = HfDinoPatchEncoder.from_pretrained(
        spec=config.vision_encoder, device=device, local_files_only=True
    )
    return engine, encoder, norm


class FrozenForecastEngine:
    def __init__(self, predictor, decoder, *, device):
        self.device = torch.device(device)
        self.predictor, self.decoder = predictor, decoder
        for model in (predictor, decoder):
            model.to(self.device).eval().requires_grad_(False)

    def predict(self, query: ForecastQuery):
        """Return lossless RGB storage values and physical proprio for each query."""
        query = query.to(self.device)
        query.validate_finite()
        with (
            torch.inference_mode(),
            torch.autocast(
                self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"
            ),
        ):
            prediction = self.predictor.predict_at(query)
            if not torch.equal(prediction.source_ticks, query.source_ticks) or not torch.equal(
                prediction.target_ticks, query.source_ticks + query.query_ticks
            ):
                raise ValueError("Direct forecast output does not match the requested endpoint")
            if (
                not torch.isfinite(prediction.visual_latents).all()
                or not torch.isfinite(prediction.proprio).all()
            ):
                raise ValueError("forecast prediction must be finite")
            decoded = self.decoder(prediction.visual_latents)
            if (
                decoded.shape != (len(query.query_ticks), 2, 3, 224, 224)
                or not torch.isfinite(decoded).all()
            ):
                raise ValueError("forecast decoder returned invalid RGB")
            if torch.any((decoded < 0) | (decoded > 1)):
                raise ValueError("decoder RGB must be in0..1")
            rgb = decoded.permute(0, 1, 3, 4, 2).mul(255).round().to(torch.uint8).cpu().numpy()
            proprio = prediction.proprio.float().cpu().numpy()
        return rgb, proprio


class DirectForecastProvider:
    """Nine real RGB boundaries, eight executed controls; encode three frames on request.

    No DINO work occurs between requests. Source history is still collected at50Hz.
    The policy must receive the measured cost of this full request-time operation.
    """

    def __init__(self, engine, *, encoder, normalization, episode_id: str):
        trunk = getattr(engine.predictor, "trunk", None)
        if trunk is not None and (
            not np.array_equal(trunk.proprio_mean.cpu().numpy(), normalization.mean)
            or not np.array_equal(trunk.proprio_scale.cpu().numpy(), normalization.scale)
        ):
            raise ValueError("forecast history normalization differs from predictor normalization")
        self.engine, self.encoder, self.normalization = engine, encoder, normalization
        self._observations = deque(maxlen=9)
        self._controls = deque(maxlen=8)
        self.reset(episode_id=episode_id)

    def reset(self, *, episode_id: str):
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("forecast provider requires an episode identity")
        self.episode_id = episode_id
        self._observations.clear()
        self._controls.clear()

    def observe(self, observation, previous_action):
        if self._observations:
            if observation.formal_tick != self._observations[-1].formal_tick + 1:
                raise ValueError("forecast history must have consecutive real boundaries")
            control = np.asarray(previous_action, dtype=np.float32)
            if (
                control.shape != (7,)
                or not np.isfinite(control).all()
                or np.any(np.abs(control) > 1)
            ):
                raise ValueError("history requires actual controller-native previous action")
            self._controls.append(control.copy())
        elif previous_action is not None:
            raise ValueError("first history boundary cannot have an unknown previous action")
        self._observations.append(observation)

    def predict(self, context):
        h, q = context.origin_tick, context.estimated_delay_ticks
        if not self._observations or self._observations[-1].formal_tick != h:
            raise ValueError("forecast source is not the latest observed boundary")
        observed = self._observations[-1]
        if any(
            not np.array_equal(getattr(observed, key), getattr(context.observation, key))
            for key in ("image", "wrist_image", "state")
        ):
            raise ValueError("forecast request source differs from the recorded observation")
        if type(q) is not int or not 0 <= q <= 20:
            raise ValueError("forecast query must lie in0..20")
        buffer, valid = (
            np.asarray(context.previous_actions),
            np.asarray(context.previous_action_mask),
        )
        if buffer.shape != (50, 7) or valid.shape != (50,) or valid.dtype != np.bool_:
            raise ValueError("forecast request has invalid buffer shape/validity")
        if h < 10 or len(self._observations) < 9 or not valid[:q].all():
            return DecodedForecast(h, h + q, q, context.buffer_version, None, None)
        history = list(self._observations)[::4]
        images = np.stack([image for obs in history for image in (obs.image, obs.wrist_image)])
        with torch.inference_mode():
            visual = self.encoder.encode(images).reshape(1, 3, 2, 196, 384).half()
        device = visual.device
        prop = self.normalization.normalize(np.stack([obs.state for obs in history]))
        prefix = np.zeros((20, 7), np.float32)
        prefix[:q] = buffer[:q]
        query = ForecastQuery(
            vision_history=visual,
            proprio_history=torch.tensor(prop, dtype=torch.float32, device=device)[None],
            executed_controls=torch.tensor(
                np.stack(self._controls).reshape(2, 4, 7), device=device
            )[None],
            executable_controls=torch.tensor(prefix, device=device)[None],
            control_mask=(torch.arange(20, device=device) < q)[None],
            query_ticks=torch.tensor([q], device=device),
            source_ticks=torch.tensor([h], device=device),
        )
        rgb, proprio = self.engine.predict(query)
        return DecodedForecast(h, h + q, q, context.buffer_version, rgb[0], proprio[0])
