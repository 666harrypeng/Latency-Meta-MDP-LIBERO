"""Local RTX steady-state runtime benchmark for final L3 JEPA checkpoints."""

from __future__ import annotations

import gc
import io
import json
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from latency_meta_mdp.artifacts import sha256_file
from latency_meta_mdp.belief.action_conditioned_jepa.config import (
    ActionConditionedJepaConfig,
    load_action_conditioned_jepa_config,
)
from latency_meta_mdp.belief.action_conditioned_jepa.data_adapter import (
    JepaEpisodeRecord,
    load_jepa_proprio_normalization,
    load_verified_jepa_inputs,
    load_verified_jepa_record,
)
from latency_meta_mdp.belief.action_conditioned_jepa.latent_return import (
    assemble_return_latent_belief,
)
from latency_meta_mdp.belief.action_conditioned_jepa.qualification_run import (
    load_completed_l3_admission_run,
)
from latency_meta_mdp.belief.action_conditioned_jepa.rollout import (
    ActionConditionedJepaPredictor,
)
from latency_meta_mdp.belief.action_conditioned_jepa.runtime import JepaRuntimeHistory
from latency_meta_mdp.expert_realization.artifacts import (
    _fsync_directory,
    _write_file_fsynced,
)
from latency_meta_mdp.expert_realization.source_corpus.schema import SourceFieldRole
from latency_meta_mdp.hf_dino_encoder import HfDinoPatchEncoder
from latency_meta_mdp.latency_law import load_latency_law

_FORMAT_ID = "action_conditioned_jepa_l3_runtime_benchmark_v1"
_SOURCE_EPISODE_ID = "source-L3-task000000-s00-d0000"
_SEEDS = (7, 17, 27)
_WARMUP_LAUNCHES = 10
_TIMED_LAUNCHES = 100
_COMPONENTS = (
    "dual_camera_dino",
    "history_and_context",
    "ar5_rollout",
    "pmf_assembly",
    "full_belief",
)


def summarize_runtime_timings(
    samples: Mapping[str, list[float]],
) -> dict[str, object]:
    if not isinstance(samples, Mapping) or set(samples) != set(_COMPONENTS):
        raise ValueError("runtime samples must contain the canonical components")
    arrays = {name: np.asarray(samples[name], dtype=np.float64) for name in _COMPONENTS}
    counts = {value.size for value in arrays.values()}
    if len(counts) != 1 or next(iter(counts), 0) <= 0:
        raise ValueError("runtime samples must contain one aligned launch inventory")
    if any(value.ndim != 1 or not np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("runtime samples must contain finite vectors")
    return {
        "sample_count": next(iter(counts)),
        "components": {
            name: {
                "mean_ms": float(value.mean()),
                "p50_ms": float(np.percentile(value, 50)),
                "p95_ms": float(np.percentile(value, 95)),
                "p99_ms": float(np.percentile(value, 99)),
            }
            for name, value in arrays.items()
        },
    }


def _paths(root: Path, seed: int) -> dict[str, Path]:
    source_id = "panda-ball-structured-source-quota-formal-100x4-v1"
    return {
        "model": root / "configs/belief/action_conditioned_jepa/model.yaml",
        "level": root / "configs/belief/action_conditioned_jepa/l3.yaml",
        "temporal": (
            root / "configs/belief/action_conditioned_jepa/stride4_80ms_history_160ms.yaml"
        ),
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
        "run": (
            root
            / "outputs/training/action_conditioned_jepa/l3-final-admission"
            / "stride4_80ms_history_160ms"
            / f"seed-{seed}"
        ),
        "law": root / "configs/latency/truncated_beta_8_65_400ms_v1.yaml",
    }


def _decode_rgb(payload: dict[str, object]) -> np.ndarray:
    with Image.open(io.BytesIO(payload["bytes"])) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _load_source_stream(
    *,
    root: Path,
    config: ActionConditionedJepaConfig,
) -> tuple[JepaEpisodeRecord, tuple[np.ndarray, ...]]:
    paths = _paths(root, 7)
    inputs = load_verified_jepa_inputs(
        source_root=paths["source"],
        cache_run_manifest=paths["cache"],
        split_manifest_path=paths["split"],
        config=config,
    )
    record = load_verified_jepa_record(
        inputs,
        episode_id=_SOURCE_EPISODE_ID,
        level=3,
        split="train",
    )
    table = inputs.source.read_fields(
        _SOURCE_EPISODE_ID,
        fields=("formal_tick", "agentview_rgb", "wrist_rgb"),
        allowed_roles=frozenset({SourceFieldRole.IDENTITY, SourceFieldRole.DEPLOYMENT_INPUT}),
    )
    rows = table.to_pylist()
    if [row["formal_tick"] for row in rows] != list(range(record.terminal_tick + 1)):
        raise ValueError("runtime source images are not aligned to the episode")
    images = tuple(
        np.stack((_decode_rgb(row["agentview_rgb"]), _decode_rgb(row["wrist_rgb"]))) for row in rows
    )
    return record, images


def _load_final_model(
    *,
    root: Path,
    seed: int,
    config: ActionConditionedJepaConfig,
    device: torch.device,
) -> tuple[ActionConditionedJepaPredictor, Any, str]:
    paths = _paths(root, seed)
    completed = load_completed_l3_admission_run(paths["run"], expected_model_seed=seed)
    normalization = load_jepa_proprio_normalization(
        completed.run_root / "proprio_normalization.json"
    )
    checkpoint = completed.checkpoint_dir
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    metadata = manifest["artifacts"]["model.safetensors"]
    model_path = checkpoint / "model.safetensors"
    model_sha = sha256_file(model_path)
    if model_path.stat().st_size != metadata["bytes"] or model_sha != metadata["sha256"]:
        raise ValueError("runtime checkpoint model artifact is invalid")
    model = ActionConditionedJepaPredictor(
        config=config,
        proprio_normalization=normalization,
        project_root=root,
    ).to(device)
    state = load_file(str(model_path), device=str(device))
    model.load_state_dict(state, strict=True)
    model.eval()
    del state
    return model, normalization, model_sha


def _synchronize(device: torch.device) -> float:
    torch.cuda.synchronize(device)
    return time.perf_counter()


@torch.inference_mode()
def _benchmark_seed(
    *,
    root: Path,
    seed: int,
    config: ActionConditionedJepaConfig,
    record: JepaEpisodeRecord,
    images: tuple[np.ndarray, ...],
    encoder: HfDinoPatchEncoder,
    probability: torch.Tensor,
    device: torch.device,
) -> dict[str, object]:
    model, normalization, model_sha = _load_final_model(
        root=root,
        seed=seed,
        config=config,
        device=device,
    )
    history = JepaRuntimeHistory(
        normalization,
        temporal_sampling=config.temporal_sampling,
    )
    timings = {name: [] for name in _COMPONENTS}
    launch_count = 0
    torch.cuda.reset_peak_memory_stats(device)
    for tick in range(record.terminal_tick):
        start = _synchronize(device)
        features = encoder.encode(images[tick])
        after_encode = _synchronize(device)
        history.append_boundary(
            vision_features=features,
            proprio=torch.from_numpy(np.array(record.proprio_physical[tick], copy=True)).to(device),
            executed_control_from_previous=(
                None
                if tick == 0
                else torch.from_numpy(np.array(record.controls[tick - 1], copy=True)).to(device)
            ),
        )
        if not history.ready or tick + 20 > len(record.controls):
            continue
        context = history.build_launch_context(
            torch.from_numpy(np.array(record.controls[tick : tick + 20], copy=True)).to(device)
        )
        after_context = _synchronize(device)
        rollout = model.rollout_native(context)
        after_rollout = _synchronize(device)
        assemble_return_latent_belief(
            rollout,
            probability,
            sampling=config.temporal_sampling,
        )
        after_assembly = _synchronize(device)
        launch_count += 1
        if launch_count <= _WARMUP_LAUNCHES:
            continue
        for name, value in (
            ("dual_camera_dino", after_encode - start),
            ("history_and_context", after_context - after_encode),
            ("ar5_rollout", after_rollout - after_context),
            ("pmf_assembly", after_assembly - after_rollout),
            ("full_belief", after_assembly - start),
        ):
            timings[name].append(value * 1_000.0)
        if len(timings["full_belief"]) == _TIMED_LAUNCHES:
            break
    if len(timings["full_belief"]) != _TIMED_LAUNCHES:
        raise RuntimeError("runtime source episode cannot provide the fixed launch inventory")
    summary = summarize_runtime_timings(timings)
    full_p95 = summary["components"]["full_belief"]["p95_ms"]
    result = {
        "model_seed": seed,
        "checkpoint_sha256": model_sha,
        "warmup_launches": _WARMUP_LAUNCHES,
        **summary,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "isolated_steady_state_p95_target_ms": 40.0,
        "isolated_steady_state_gate_pass": full_p95 <= 40.0,
    }
    del model, history
    gc.collect()
    torch.cuda.empty_cache()
    return result


def execute_l3_runtime_benchmark(
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
    if not torch.cuda.is_available():
        raise RuntimeError("J6a runtime benchmark requires CUDA")
    target_device = torch.device(device)
    paths = _paths(root, 7)
    config = load_action_conditioned_jepa_config(
        model_path=paths["model"],
        level_path=paths["level"],
        temporal_sampling_path=paths["temporal"],
    )
    record, images = _load_source_stream(root=root, config=config)
    encoder = HfDinoPatchEncoder.from_pretrained(
        spec=config.vision_encoder,
        device=device,
        local_files_only=True,
    )
    law = load_latency_law(paths["law"])
    probability = (
        torch.from_numpy(np.array(law.probabilities, copy=True))
        .to(device=target_device, dtype=torch.float32)
        .unsqueeze(0)
    )
    results = tuple(
        _benchmark_seed(
            root=root,
            seed=seed,
            config=config,
            record=record,
            images=images,
            encoder=encoder,
            probability=probability,
            device=target_device,
        )
        for seed in _SEEDS
    )
    report = {
        "format_id": _FORMAT_ID,
        "device": torch.cuda.get_device_name(target_device),
        "temporal_config_id": config.temporal_sampling.config_id,
        "source_episode_id": _SOURCE_EPISODE_ID,
        "png_decode_timed": False,
        "simulator_callback_or_queueing_timed": False,
        "scope": "isolated client-local steady-state Belief core",
        "results": list(results),
        "all_seeds_isolated_gate_pass": all(
            value["isolated_steady_state_gate_pass"] for value in results
        ),
        "j6a_complete": False,
        "remaining_j6a_measurement": "concurrent simulator/control load and queueing",
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_file_fsynced(
        target,
        (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )
    _fsync_directory(target.parent)
    return target
