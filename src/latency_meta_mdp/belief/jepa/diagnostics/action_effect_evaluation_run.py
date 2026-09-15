"""Evaluate all final L3 seeds on one immutable J4 action-effect bank."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.numpy import load_file as load_numpy_safetensors

from latency_meta_mdp.belief.jepa.ar.benchmark import (
    _load_final_model,
)
from latency_meta_mdp.belief.jepa.ar.qualify import (
    _load_readout,
)
from latency_meta_mdp.belief.jepa.ar.temporal_view import (
    SharedJepaSampleIndex,
    materialize_temporal_jepa_sample,
)
from latency_meta_mdp.belief.jepa.config import (
    load_action_conditioned_jepa_config,
)
from latency_meta_mdp.belief.jepa.contracts import LaunchContextBatch
from latency_meta_mdp.belief.jepa.corpus import (
    load_verified_jepa_inputs,
    load_verified_jepa_record,
)
from latency_meta_mdp.belief.jepa.diagnostics.action_effect_evaluation import (
    summarize_j4_effects,
)
from latency_meta_mdp.data.collection.artifacts import (
    _fsync_directory,
    _write_file_fsynced,
)
from latency_meta_mdp.io.artifacts import sha256_file

_FORMAT_ID = "action_conditioned_jepa_l3_j4_evaluation_v1"
_BANK_FORMAT_ID = "action_conditioned_jepa_l3_j4_bank_v1"
_SEEDS = (7, 17, 27)
_BRANCH_NAMES = ("nominal", "hold", "scale_0.5", "prefix4_then_hold")
_ANCHORS = (4, 8, 12, 16, 20)


def _paths(root: Path) -> dict[str, Path]:
    source_id = "panda-ball-structured-source-quota-formal-100x4-v1"
    return {
        "model": root / "configs/models/jepa/model.yaml",
        "level": root / "configs/models/jepa/l3.yaml",
        "temporal": (root / "configs/models/jepa/stride4_80ms_history_160ms.yaml"),
        "source": root / "outputs/source_corpus" / source_id,
        "cache": (
            root
            / "outputs/derived/vision_features"
            / "dinov3-vits16-structured-source-100x4-v1/manifest.json"
        ),
        "split": (
            root
            / "outputs/derived/source_splits"
            / source_id
            / "train80-validation20-seed20260903-v1.json"
        ),
        "bank": root / "outputs/analysis/action_conditioned_jepa/l3-final-action-effect-bank",
        "readout": root / "outputs/training/action_conditioned_jepa/l3-gt-latent-object-readout",
    }


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_bank(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        contexts = json.loads((root / "contexts.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("J4 bank metadata is invalid") from error
    if (
        type(manifest) is not dict
        or manifest.get("format_id") != _BANK_FORMAT_ID
        or manifest.get("context_count") != 80
        or manifest.get("branch_count") != 320
        or manifest.get("formal_validation_only") is not True
        or manifest.get("training_eligible") is not False
        or type(contexts) is not dict
        or tuple(contexts.get("branch_names", ())) != _BRANCH_NAMES
        or tuple(contexts.get("native_delay_ticks", ())) != _ANCHORS
        or type(contexts.get("contexts")) is not list
        or len(contexts["contexts"]) != 80
    ):
        raise ValueError("J4 bank contract is incompatible")
    for name, metadata in manifest["artifacts"].items():
        path = root / name
        if (
            not path.is_file()
            or path.stat().st_size != metadata["bytes"]
            or _hash_file(path) != metadata["sha256"]
        ):
            raise ValueError("J4 bank artifact verification failed")
    arrays = load_numpy_safetensors(root / "bank.safetensors")
    expected = {
        "controls": (80, 4, 20, 7),
        "future_proprio": (80, 4, 5, 16),
        "future_object": (80, 4, 5, 6),
        "future_visual_latents": (80, 4, 5, 2, 196, 384),
        "left_pad_contact": (80, 4, 5),
        "right_pad_contact": (80, 4, 5),
        "handoff_physical": (80, 4, 5),
        "absorbing": (80, 4, 5),
        "terminal_step": (80, 4),
        "source_replay_max_abs": (80, 4),
        "nominal_future_max_abs": (80,),
    }
    if set(arrays) != set(expected) or any(
        arrays[name].shape != shape for name, shape in expected.items()
    ):
        raise ValueError("J4 bank tensor inventory is incompatible")
    return manifest, contexts, arrays


@torch.inference_mode()
def _decode_object_latents(
    *,
    readout: torch.nn.Module,
    normalization: Any,
    latents: np.ndarray,
    device: torch.device,
    batch_size: int = 8,
) -> np.ndarray:
    flat = latents.reshape(-1, 5, 2, 196, 384)
    decoded = np.empty((flat.shape[0], 5, 6), dtype=np.float32)
    for start in range(0, flat.shape[0], batch_size):
        stop = min(start + batch_size, flat.shape[0])
        value = torch.from_numpy(np.array(flat[start:stop], copy=True)).to(device)
        predicted = normalization.denormalize(readout(value).float())
        decoded[start:stop] = predicted.cpu().numpy()
    return decoded.reshape(*latents.shape[:3], 6)


def _launch_batch(
    *,
    context_rows: list[tuple[int, int]],
    context_metadata: list[dict[str, Any]],
    controls: np.ndarray,
    records: dict[str, Any],
    normalization: Any,
    sampling: Any,
    device: torch.device,
) -> LaunchContextBatch:
    vision = []
    proprio = []
    executed = []
    executable = []
    samples = {}
    for context_index, branch_index in context_rows:
        metadata = context_metadata[context_index]
        key = (metadata["episode_id"], metadata["source_tick"])
        if key not in samples:
            record = records[metadata["episode_id"]]
            samples[key] = materialize_temporal_jepa_sample(
                record=record,
                index=SharedJepaSampleIndex(
                    level=3,
                    split="validation",
                    episode_id=metadata["episode_id"],
                    source_tick=metadata["source_tick"],
                    boundary_disposition="recorded_complete",
                ),
                sampling=sampling,
                normalization=normalization,
            )
        sample = samples[key]
        vision.append(sample.vision_history)
        proprio.append(sample.proprio_history)
        executed.append(sample.past_macro_controls)
        executable.append(
            torch.from_numpy(
                np.array(controls[context_index, branch_index], copy=True).reshape(5, 4, 7)
            )
        )
    return LaunchContextBatch(
        vision_history=torch.stack(vision).to(device),
        proprio_history=torch.stack(proprio).to(device),
        executed_controls=torch.stack(executed).to(device),
        executable_controls=torch.stack(executable).to(device),
    )


def _absolute_metrics(
    *,
    predicted_proprio: np.ndarray,
    predicted_object: np.ndarray,
    decoded_gt_object: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> dict[str, float]:
    gt_proprio = arrays["future_proprio"]
    gt_object = arrays["future_object"]
    return {
        "predicted_qpos_coordinate_rmse_rad": float(
            np.sqrt(np.mean((predicted_proprio[..., :7] - gt_proprio[..., :7]) ** 2))
        ),
        "predicted_object_position_coordinate_rmse_m": float(
            np.sqrt(np.mean((predicted_object[..., :3] - gt_object[..., :3]) ** 2))
        ),
        "gt_latent_readout_position_coordinate_rmse_m": float(
            np.sqrt(np.mean((decoded_gt_object[..., :3] - gt_object[..., :3]) ** 2))
        ),
    }


def _effect_breakdown(
    *,
    predicted_proprio: np.ndarray,
    predicted_object: np.ndarray,
    arrays: dict[str, np.ndarray],
    context_metadata: list[dict[str, Any]],
) -> dict[str, object]:
    categories = tuple(value["category"] for value in context_metadata)

    def summarize(selected: np.ndarray) -> dict[str, object]:
        return summarize_j4_effects(
            predicted_proprio=predicted_proprio[selected],
            predicted_object=predicted_object[selected],
            gt_proprio=arrays["future_proprio"][selected],
            gt_object=arrays["future_object"][selected],
            handoff_physical=arrays["handoff_physical"][selected],
            context_categories=tuple(
                category for category, keep in zip(categories, selected, strict=True) if keep
            ),
            branch_names=_BRANCH_NAMES,
            native_delay_ticks=_ANCHORS,
        )

    return {
        "overall": summarize(np.ones(len(categories), dtype=np.bool_)),
        "by_phase": {
            category: summarize(np.asarray([value == category for value in categories]))
            for category in (
                "smooth_approach",
                "grasp_funnel",
                "close_stabilize",
                "post_handoff_lift",
            )
        },
    }


@torch.inference_mode()
def evaluate_l3_j4_bank(
    *,
    project_root: Path,
    device: str,
    output_path: Path,
) -> Path:
    root = Path(project_root).resolve()
    target = Path(output_path).absolute()
    if target.exists():
        raise FileExistsError(target)
    if type(device) is not str or re.fullmatch(r"cuda:[0-9]+", device) is None:
        raise ValueError("device must identify one CUDA device")
    target_device = torch.device(device)
    paths = _paths(root)
    config = load_action_conditioned_jepa_config(
        model_path=paths["model"],
        level_path=paths["level"],
        temporal_sampling_path=paths["temporal"],
    )
    bank_manifest, context_payload, arrays = _load_bank(paths["bank"])
    inputs = load_verified_jepa_inputs(
        source_root=paths["source"],
        cache_run_manifest=paths["cache"],
        split_manifest_path=paths["split"],
        config=config,
    )
    provenance = bank_manifest["provenance"]
    if provenance != {
        "source_manifest_sha256": inputs.source_manifest_sha256,
        "cache_manifest_sha256": inputs.cache_manifest_sha256,
        "split_manifest_sha256": inputs.split_manifest_sha256,
        "model_config_sha256": sha256_file(paths["model"]),
        "temporal_config_sha256": sha256_file(paths["temporal"]),
        "vision_encoder_fingerprint": config.vision_encoder.fingerprint,
    }:
        raise ValueError("J4 bank provenance disagrees with the final model inputs")
    context_metadata = context_payload["contexts"]
    episode_ids = tuple(sorted({value["episode_id"] for value in context_metadata}))
    records = {
        episode_id: load_verified_jepa_record(
            inputs,
            episode_id=episode_id,
            level=3,
            split="validation",
        )
        for episode_id in episode_ids
    }
    readout, object_normalization, readout_summary = _load_readout(
        root=root,
        artifact_root=paths["readout"],
        device=target_device,
    )
    decoded_gt_object = _decode_object_latents(
        readout=readout,
        normalization=object_normalization,
        latents=arrays["future_visual_latents"],
        device=target_device,
    )
    gt_latent_readout_effect_control = _effect_breakdown(
        predicted_proprio=arrays["future_proprio"],
        predicted_object=decoded_gt_object,
        arrays=arrays,
        context_metadata=context_metadata,
    )
    seed_results = []
    all_rows = [(context, branch) for context in range(80) for branch in range(4)]
    for seed in _SEEDS:
        model, normalization, checkpoint_sha = _load_final_model(
            root=root,
            seed=seed,
            config=config,
            device=target_device,
        )
        predicted_proprio = np.empty((80, 4, 5, 16), dtype=np.float32)
        predicted_object = np.empty((80, 4, 5, 6), dtype=np.float32)
        model.eval()
        for start in range(0, len(all_rows), 8):
            rows = all_rows[start : start + 8]
            launch = _launch_batch(
                context_rows=rows,
                context_metadata=context_metadata,
                controls=arrays["controls"],
                records=records,
                normalization=normalization,
                sampling=config.temporal_sampling,
                device=target_device,
            )
            rollout = model.rollout_native(launch)
            object_state = object_normalization.denormalize(
                readout(rollout.future_visual_latents).float()
            )
            for row_index, (context_index, branch_index) in enumerate(rows):
                predicted_proprio[context_index, branch_index] = (
                    rollout.future_proprio[row_index].cpu().numpy()
                )
                predicted_object[context_index, branch_index] = (
                    object_state[row_index].cpu().numpy()
                )
        effects = _effect_breakdown(
            predicted_proprio=predicted_proprio,
            predicted_object=predicted_object,
            arrays=arrays,
            context_metadata=context_metadata,
        )
        seed_results.append(
            {
                "model_seed": seed,
                "checkpoint_sha256": checkpoint_sha,
                "absolute": _absolute_metrics(
                    predicted_proprio=predicted_proprio,
                    predicted_object=predicted_object,
                    decoded_gt_object=decoded_gt_object,
                    arrays=arrays,
                ),
                "effects": effects,
            }
        )
        del model
        torch.cuda.empty_cache()
    report = {
        "format_id": _FORMAT_ID,
        "bank_manifest_sha256": sha256_file(paths["bank"] / "manifest.json"),
        "context_count": 80,
        "branch_count": 320,
        "branch_names": list(_BRANCH_NAMES),
        "native_delay_ticks": list(_ANCHORS),
        "readout": {
            "format_id": readout_summary["format_id"],
            "final_train_loss": readout_summary["final_train_loss"],
        },
        "bank_quality": {
            "source_replay_max_abs": float(arrays["source_replay_max_abs"].max()),
            "nominal_future_max_abs": float(arrays["nominal_future_max_abs"].max()),
        },
        "gt_latent_readout_effect_control": gt_latent_readout_effect_control,
        "results": seed_results,
        "j4_gate_pass": None,
        "gate_status": "metrics_ready_for_review",
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_file_fsynced(
        target,
        (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )
    _fsync_directory(target.parent)
    return target
