"""Pure latency-mixture assembly and lossless Return Latent Belief artifacts."""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

from latency_meta_mdp.belief.action_conditioned_jepa.contracts import (
    FutureLatentRollout,
    ReturnLatentBeliefBatch,
)
from latency_meta_mdp.expert_realization.artifacts import (
    _fsync_directory,
    _hash_file,
    _rename_noreplace,
    _write_file_fsynced,
)

_FORMAT_ID = "action_conditioned_jepa_return_latent_belief_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_FIELDS = {
    "schema_version",
    "format_id",
    "latency_law_sha256",
    "batch_size",
    "delay_count",
    "visual_shape",
    "proprio_shape",
    "artifacts",
}


def _require_sha256(value: str, *, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def assemble_return_latent_belief(
    rollout: FutureLatentRollout,
    probabilities: torch.Tensor,
) -> ReturnLatentBeliefBatch:
    """Attach a valid D20 PMF without recomputing or copying fixed-delay futures."""

    if not isinstance(rollout, FutureLatentRollout):
        raise TypeError("rollout must be a FutureLatentRollout")
    if not isinstance(probabilities, torch.Tensor):
        raise TypeError("probabilities must be a torch.Tensor")
    delay_ticks = torch.arange(1, 21, dtype=torch.int64, device=rollout.device)
    return ReturnLatentBeliefBatch(
        delay_ticks=delay_ticks,
        delay_probabilities=probabilities,
        future_visual_latents=rollout.future_visual_latents,
        future_proprio=rollout.future_proprio,
    )


def weighted_future_proprio(belief: ReturnLatentBeliefBatch) -> torch.Tensor:
    if not isinstance(belief, ReturnLatentBeliefBatch):
        raise TypeError("belief must be a ReturnLatentBeliefBatch")
    return torch.sum(
        belief.delay_probabilities[..., None] * belief.future_proprio,
        dim=1,
    )


def weighted_future_visual_latents(belief: ReturnLatentBeliefBatch) -> torch.Tensor:
    if not isinstance(belief, ReturnLatentBeliefBatch):
        raise TypeError("belief must be a ReturnLatentBeliefBatch")
    return torch.sum(
        belief.delay_probabilities[..., None, None, None] * belief.future_visual_latents,
        dim=1,
    )


@dataclass(frozen=True)
class LoadedReturnLatentBelief:
    belief: ReturnLatentBeliefBatch
    latency_law_sha256: str
    manifest: dict[str, Any]


def write_return_latent_belief(
    output_dir: Path,
    belief: ReturnLatentBeliefBatch,
    *,
    latency_law_sha256: str,
) -> Path:
    """Publish one no-overwrite, consumer-independent Belief artifact."""

    if not isinstance(belief, ReturnLatentBeliefBatch):
        raise TypeError("belief must be a ReturnLatentBeliefBatch")
    law_sha = _require_sha256(latency_law_sha256, name="latency_law_sha256")
    target = Path(output_dir).absolute()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f".{target.name}.building-{os.getpid()}-{uuid.uuid4().hex}"
    building.mkdir()
    try:
        tensor_payload = save_safetensors(
            {
                "delay_ticks": belief.delay_ticks.detach().cpu().contiguous(),
                "delay_probabilities": belief.delay_probabilities.detach().cpu().contiguous(),
                "future_visual_latents": belief.future_visual_latents.detach().cpu().contiguous(),
                "future_proprio": belief.future_proprio.detach().cpu().contiguous(),
            }
        )
        tensor_path = building / "belief.safetensors"
        _write_file_fsynced(tensor_path, tensor_payload)
        manifest = {
            "schema_version": 1,
            "format_id": _FORMAT_ID,
            "latency_law_sha256": law_sha,
            "batch_size": belief.batch_size,
            "delay_count": belief.maximum_delay_ticks,
            "visual_shape": list(belief.future_visual_latents.shape),
            "proprio_shape": list(belief.future_proprio.shape),
            "artifacts": {
                "belief.safetensors": {
                    "bytes": tensor_path.stat().st_size,
                    "sha256": _hash_file(tensor_path),
                }
            },
        }
        payload = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        _write_file_fsynced(building / "manifest.json", payload)
        _fsync_directory(building)
        _rename_noreplace(building, target)
        _fsync_directory(target.parent)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return target / "manifest.json"


def load_return_latent_belief(output_dir: Path) -> LoadedReturnLatentBelief:
    root = Path(output_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        type(manifest) is not dict
        or set(manifest) != _MANIFEST_FIELDS
        or manifest["schema_version"] != 1
        or manifest["format_id"] != _FORMAT_ID
    ):
        raise ValueError("Return Latent Belief manifest fields are invalid")
    law_sha = _require_sha256(
        manifest["latency_law_sha256"],
        name="latency_law_sha256",
    )
    artifacts = manifest["artifacts"]
    tensor_path = root / "belief.safetensors"
    if type(artifacts) is not dict or set(artifacts) != {"belief.safetensors"}:
        raise ValueError("Return Latent Belief artifact inventory is invalid")
    metadata = artifacts["belief.safetensors"]
    if (
        type(metadata) is not dict
        or set(metadata) != {"bytes", "sha256"}
        or type(metadata["bytes"]) is not int
        or type(metadata["sha256"]) is not str
        or not tensor_path.is_file()
        or tensor_path.stat().st_size != metadata["bytes"]
        or _hash_file(tensor_path) != metadata["sha256"]
    ):
        raise ValueError("Return Latent Belief artifact verification failed")
    tensors = load_safetensors(tensor_path.read_bytes())
    if set(tensors) != {
        "delay_ticks",
        "delay_probabilities",
        "future_visual_latents",
        "future_proprio",
    }:
        raise ValueError("Return Latent Belief tensor inventory is invalid")
    belief = ReturnLatentBeliefBatch(
        delay_ticks=tensors["delay_ticks"],
        delay_probabilities=tensors["delay_probabilities"],
        future_visual_latents=tensors["future_visual_latents"],
        future_proprio=tensors["future_proprio"],
    )
    if (
        manifest["batch_size"] != belief.batch_size
        or manifest["delay_count"] != belief.maximum_delay_ticks
        or manifest["visual_shape"] != list(belief.future_visual_latents.shape)
        or manifest["proprio_shape"] != list(belief.future_proprio.shape)
    ):
        raise ValueError("Return Latent Belief tensor shapes disagree with manifest")
    return LoadedReturnLatentBelief(
        belief=belief,
        latency_law_sha256=law_sha,
        manifest=manifest,
    )
