"""Durable qualification artifacts for official JEPA-WM predictors."""

from __future__ import annotations

import ctypes
import errno
import gc
import importlib
import json
import os
import shutil
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from latency_meta_mdp.belief.jepa.upstream_adapter import (
    CheckpointMirror,
    UpstreamCheckout,
    UpstreamReference,
    load_upstream_reference,
    verify_checkpoint_mirror,
    verify_upstream_checkout,
)
from latency_meta_mdp.io.artifacts import sha256_file


@dataclass(frozen=True)
class OfficialPredictorConfig:
    name: str
    image_size: int
    patch_size: int
    frame_count: int
    embed_dim: int
    predictor_dim: int
    depth: int
    head_count: int
    action_dim: int
    proprio_dim: int
    proprio_feature_dim: int
    checkpoint_epoch: int
    predictor_parameter_count: int
    auxiliary_parameter_count: int
    predictor_state_key_count: int

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    @property
    def input_visual_shape(self) -> tuple[int, ...]:
        return (1, self.frame_count, 1, self.grid_size, self.grid_size, self.embed_dim)

    @property
    def input_action_shape(self) -> tuple[int, ...]:
        return (1, self.frame_count, self.action_dim)

    @property
    def input_proprio_shape(self) -> tuple[int, ...] | None:
        if self.proprio_feature_dim == 0:
            return None
        return (1, self.frame_count, self.proprio_dim)


def official_predictor_config(name: str) -> OfficialPredictorConfig:
    if name == "metaworld":
        return OfficialPredictorConfig(
            name=name,
            image_size=224,
            patch_size=14,
            frame_count=4,
            embed_dim=384,
            predictor_dim=384,
            depth=6,
            head_count=16,
            action_dim=20,
            proprio_dim=4,
            proprio_feature_dim=16,
            checkpoint_epoch=50,
            predictor_parameter_count=17_630_480,
            auxiliary_parameter_count=80,
            predictor_state_key_count=92,
        )
    if name == "droid":
        return OfficialPredictorConfig(
            name=name,
            image_size=256,
            patch_size=16,
            frame_count=4,
            embed_dim=1024,
            predictor_dim=1024,
            depth=12,
            head_count=16,
            action_dim=7,
            proprio_dim=7,
            proprio_feature_dim=0,
            checkpoint_epoch=315,
            predictor_parameter_count=228_835_328,
            auxiliary_parameter_count=0,
            predictor_state_key_count=176,
        )
    raise ValueError("unsupported official JEPA-WM predictor")


def _clean_checkpoint_state(value: dict[str, Any]) -> dict[str, Any]:
    return {name.removeprefix("module."): tensor for name, tensor in value.items()}


def _qualification_modules(upstream_root: Path) -> tuple[Any, Any]:
    adaln = importlib.import_module("app.plan_common.models.AdaLN_vit")
    proprio = importlib.import_module("app.plan_common.models.prop_embedding")
    root = upstream_root.resolve()
    for module in (adaln, proprio):
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            raise ValueError("JEPA-WM qualification module lacks a source file")
        try:
            Path(module_file).resolve().relative_to(root)
        except ValueError as error:
            raise ValueError("JEPA-WM qualification module resolves outside checkout") from error
    return adaln.vit_predictor_AdaLN, proprio.ProprioceptiveEmbedding


def collect_locked_runtime(device: str) -> dict[str, Any]:
    import platform

    import tensordict
    import timm
    import torch
    import torchvision

    target = torch.device(device)
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "tensordict": str(tensordict.__version__),
        "timm": str(timm.__version__),
        "device": str(target),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": (
            torch.cuda.get_device_name(target)
            if target.type == "cuda" and torch.cuda.is_available()
            else None
        ),
        "cuda_capability": (
            list(torch.cuda.get_device_capability(target))
            if target.type == "cuda" and torch.cuda.is_available()
            else None
        ),
    }


def validate_locked_runtime(runtime: dict[str, Any]) -> None:
    required = {
        "python",
        "torch",
        "torchvision",
        "tensordict",
        "timm",
        "device",
        "cuda_available",
        "cuda_device_name",
        "cuda_capability",
    }
    torch_version = str(runtime.get("torch", "")).split("+", maxsplit=1)[0]
    torchvision_version = str(runtime.get("torchvision", "")).split("+", maxsplit=1)[0]
    capability = runtime.get("cuda_capability")
    capability_valid = (
        isinstance(capability, list)
        and len(capability) == 2
        and not isinstance(capability[0], bool)
        and isinstance(capability[0], int)
        and capability[0] > 0
        and not isinstance(capability[1], bool)
        and isinstance(capability[1], int)
        and capability[1] >= 0
    )
    if (
        not isinstance(runtime, dict)
        or set(runtime) != required
        or not str(runtime["python"]).startswith("3.10.")
        or torch_version != "2.7.1"
        or torchvision_version != "0.22.1"
        or runtime["tensordict"] != "0.11.0"
        or runtime["timm"] != "1.0.19"
        or not str(runtime["device"]).startswith("cuda")
        or runtime["cuda_available"] is not True
        or not isinstance(runtime["cuda_device_name"], str)
        or not runtime["cuda_device_name"]
        or not capability_valid
    ):
        raise ValueError("official JEPA-WM qualification runtime is not the locked CUDA stack")


def qualify_official_predictor(
    *,
    name: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    upstream_root: Path,
    device: str,
    warmup_iterations: int,
    benchmark_iterations: int,
) -> OfficialPredictorQualification:
    import torch

    config = official_predictor_config(name)
    vit_predictor_AdaLN, ProprioceptiveEmbedding = _qualification_modules(upstream_root)
    checkpoint_file = checkpoint_path.resolve()
    if not checkpoint_file.is_file() or sha256_file(checkpoint_file) != checkpoint_sha256:
        raise ValueError("official JEPA-WM checkpoint identity is invalid")
    if (
        isinstance(warmup_iterations, bool)
        or not isinstance(warmup_iterations, int)
        or warmup_iterations <= 0
        or isinstance(benchmark_iterations, bool)
        or not isinstance(benchmark_iterations, int)
        or benchmark_iterations <= 0
    ):
        raise ValueError("official JEPA-WM benchmark counts must be positive integers")
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("official JEPA-WM qualification requires the locked CUDA runtime")

    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    predictor_state = checkpoint.get("predictor")
    epoch = checkpoint.get("epoch")
    if (
        not isinstance(predictor_state, dict)
        or isinstance(epoch, bool)
        or not isinstance(epoch, int)
    ):
        raise ValueError("official JEPA-WM checkpoint structure is invalid")
    predictor = vit_predictor_AdaLN(
        img_size=config.image_size,
        patch_size=config.patch_size,
        num_frames=config.frame_count,
        tubelet_size=1,
        embed_dim=config.embed_dim,
        predictor_embed_dim=config.predictor_dim,
        depth=config.depth,
        num_heads=config.head_count,
        use_silu=False,
        use_rope=True,
        local_window=(3, -1, -1),
        use_activation_checkpointing=False,
        action_dim=config.action_dim,
        proprio_dim=config.proprio_dim,
        use_proprio=config.proprio_feature_dim > 0,
        act_mlp=False,
        prop_mlp=False,
        proprio_encoder_inpred=False,
        action_encoder_inpred=True,
        proprio_encoding="feature" if config.proprio_feature_dim else "none",
        proprio_emb_dim=config.proprio_feature_dim,
        proprio_tokens=0,
        init_scale_factor_adaln=10,
    )
    predictor.load_state_dict(_clean_checkpoint_state(predictor_state), strict=True)
    auxiliary = None
    if config.proprio_feature_dim:
        auxiliary_state = checkpoint.get("proprio_encoder")
        if not isinstance(auxiliary_state, dict):
            raise ValueError("MetaWorld JEPA-WM checkpoint lacks its proprio encoder")
        auxiliary = ProprioceptiveEmbedding(
            num_frames=config.frame_count,
            tubelet_size=1,
            in_chans=config.proprio_dim,
            tokens_per_step=1,
            embed_dim=config.proprio_feature_dim,
            shift_input=False,
            use_mlp=False,
        )
        auxiliary.load_state_dict(_clean_checkpoint_state(auxiliary_state), strict=True)
    state_key_count = len(predictor_state)
    del checkpoint, predictor_state
    gc.collect()

    predictor = predictor.to(target).eval()
    if auxiliary is not None:
        auxiliary = auxiliary.to(target).eval()
    torch.manual_seed(0)
    visual = torch.randn(config.input_visual_shape, device=target)
    actions = torch.randn(config.input_action_shape, device=target)
    raw_proprio = (
        None
        if config.input_proprio_shape is None
        else torch.randn(config.input_proprio_shape, device=target)
    )
    if auxiliary is None:
        proprio_features = None
    else:
        with torch.inference_mode():
            encoded = auxiliary(raw_proprio)
        proprio_features = encoded.expand(-1, -1, config.grid_size**2, -1)

    use_amp = target.type == "cuda"

    def forward_once(
        visual_value: Any,
        action_value: Any,
        proprio_value: Any,
    ) -> tuple[Any, Any]:
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=target.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ),
        ):
            predicted_visual, _, predicted_proprio = predictor(
                visual_value,
                action_value,
                proprio_value,
            )
        return predicted_visual, predicted_proprio

    def forward_two_step() -> tuple[Any, Any, Any, Any]:
        first_visual_value, first_proprio_value = forward_once(
            visual,
            actions,
            proprio_features,
        )
        predicted_block = first_visual_value[:, -1:].reshape(
            1,
            1,
            1,
            config.grid_size,
            config.grid_size,
            config.embed_dim,
        )
        second_visual_input = torch.cat((visual[:, 1:], predicted_block.float()), dim=1)
        second_action_input = torch.cat((actions[:, 1:], actions[:, -1:]), dim=1)
        second_proprio_input = (
            None
            if proprio_features is None
            else torch.cat(
                (proprio_features[:, 1:], first_proprio_value[:, -1:].float()),
                dim=1,
            )
        )
        second_visual_value, second_proprio_value = forward_once(
            second_visual_input,
            second_action_input,
            second_proprio_input,
        )
        return (
            first_visual_value,
            first_proprio_value,
            second_visual_value,
            second_proprio_value,
        )

    torch.cuda.reset_peak_memory_stats(target)
    first_visual, first_proprio, second_visual, second_proprio = forward_two_step()
    torch.cuda.synchronize(target)
    finite = bool(torch.isfinite(first_visual).all() and torch.isfinite(second_visual).all())
    if first_proprio is not None and second_proprio is not None:
        finite = finite and bool(
            torch.isfinite(first_proprio).all() and torch.isfinite(second_proprio).all()
        )

    for _ in range(warmup_iterations):
        forward_two_step()
    torch.cuda.synchronize(target)
    one_step_timings = []
    for _ in range(benchmark_iterations):
        started_at = time.perf_counter()
        forward_once(visual, actions, proprio_features)
        torch.cuda.synchronize(target)
        one_step_timings.append((time.perf_counter() - started_at) * 1_000.0)
    two_step_timings = []
    for _ in range(benchmark_iterations):
        started_at = time.perf_counter()
        forward_two_step()
        torch.cuda.synchronize(target)
        two_step_timings.append((time.perf_counter() - started_at) * 1_000.0)
    predictor_parameters = sum(parameter.numel() for parameter in predictor.parameters())
    auxiliary_parameters = (
        0 if auxiliary is None else sum(parameter.numel() for parameter in auxiliary.parameters())
    )
    peak_reserved = int(torch.cuda.max_memory_reserved(target))
    result = OfficialPredictorQualification(
        name=name,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_size_bytes=checkpoint_file.stat().st_size,
        checkpoint_epoch=epoch,
        predictor_parameter_count=predictor_parameters,
        auxiliary_parameter_count=auxiliary_parameters,
        predictor_state_key_count=state_key_count,
        input_visual_shape=config.input_visual_shape,
        input_action_shape=config.input_action_shape,
        input_proprio_shape=config.input_proprio_shape,
        output_visual_shape=tuple(first_visual.shape),
        output_proprio_shape=(None if first_proprio is None else tuple(first_proprio.shape)),
        output_dtype=str(first_visual.dtype).removeprefix("torch."),
        two_step_output_shape=tuple(second_visual.shape),
        finite=finite,
        warmup_iterations=warmup_iterations,
        benchmark_iterations=benchmark_iterations,
        one_step_p50_ms=float(np.percentile(one_step_timings, 50)),
        one_step_p95_ms=float(np.percentile(one_step_timings, 95)),
        one_step_p99_ms=float(np.percentile(one_step_timings, 99)),
        two_step_p50_ms=float(np.percentile(two_step_timings, 50)),
        two_step_p95_ms=float(np.percentile(two_step_timings, 95)),
        two_step_p99_ms=float(np.percentile(two_step_timings, 99)),
        peak_reserved_bytes=peak_reserved,
    )
    return result


@dataclass(frozen=True)
class OfficialPredictorQualification:
    name: str
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    checkpoint_epoch: int
    predictor_parameter_count: int
    auxiliary_parameter_count: int
    predictor_state_key_count: int
    input_visual_shape: tuple[int, ...]
    input_action_shape: tuple[int, ...]
    input_proprio_shape: tuple[int, ...] | None
    output_visual_shape: tuple[int, ...]
    output_proprio_shape: tuple[int, ...] | None
    output_dtype: str
    two_step_output_shape: tuple[int, ...]
    finite: bool
    warmup_iterations: int
    benchmark_iterations: int
    one_step_p50_ms: float
    one_step_p95_ms: float
    one_step_p99_ms: float
    two_step_p50_ms: float
    two_step_p95_ms: float
    two_step_p99_ms: float
    peak_reserved_bytes: int

    def __post_init__(self) -> None:
        if self.name not in {"metaworld", "droid"}:
            raise ValueError("official JEPA-WM predictor name is invalid")
        config = official_predictor_config(self.name)
        expected_output_visual_shape = (
            1,
            config.frame_count,
            config.grid_size**2,
            config.embed_dim,
        )
        expected_output_proprio_shape = (
            None
            if config.proprio_feature_dim == 0
            else (
                1,
                config.frame_count,
                config.grid_size**2,
                config.proprio_feature_dim,
            )
        )
        if (
            self.checkpoint_epoch != config.checkpoint_epoch
            or self.predictor_parameter_count != config.predictor_parameter_count
            or self.auxiliary_parameter_count != config.auxiliary_parameter_count
            or self.predictor_state_key_count != config.predictor_state_key_count
            or self.input_visual_shape != config.input_visual_shape
            or self.input_action_shape != config.input_action_shape
            or self.input_proprio_shape != config.input_proprio_shape
            or self.output_visual_shape != expected_output_visual_shape
            or self.output_proprio_shape != expected_output_proprio_shape
            or self.two_step_output_shape != expected_output_visual_shape
        ):
            raise ValueError("official JEPA-WM predictor topology does not match its model")
        if (
            not isinstance(self.checkpoint_sha256, str)
            or len(self.checkpoint_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.checkpoint_sha256)
        ):
            raise ValueError("official JEPA-WM checkpoint digest is invalid")
        for name in (
            "checkpoint_size_bytes",
            "predictor_parameter_count",
            "predictor_state_key_count",
            "warmup_iterations",
            "benchmark_iterations",
            "peak_reserved_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"official JEPA-WM {name} must be a positive integer")
        if (
            isinstance(self.checkpoint_epoch, bool)
            or not isinstance(self.checkpoint_epoch, int)
            or self.checkpoint_epoch < 0
            or isinstance(self.auxiliary_parameter_count, bool)
            or not isinstance(self.auxiliary_parameter_count, int)
            or self.auxiliary_parameter_count < 0
        ):
            raise ValueError("official JEPA-WM epoch or auxiliary parameter count is invalid")
        for name in (
            "input_visual_shape",
            "input_action_shape",
            "output_visual_shape",
            "two_step_output_shape",
        ):
            value = getattr(self, name)
            if not value or any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value
            ):
                raise ValueError(f"official JEPA-WM {name} is invalid")
        for name in ("input_proprio_shape", "output_proprio_shape"):
            value = getattr(self, name)
            if value is not None and (
                not value
                or any(
                    isinstance(item, bool) or not isinstance(item, int) or item <= 0
                    for item in value
                )
            ):
                raise ValueError(f"official JEPA-WM {name} is invalid")
        one_step_timings = (
            self.one_step_p50_ms,
            self.one_step_p95_ms,
            self.one_step_p99_ms,
        )
        two_step_timings = (
            self.two_step_p50_ms,
            self.two_step_p95_ms,
            self.two_step_p99_ms,
        )
        if (
            self.output_dtype != "bfloat16"
            or type(self.finite) is not bool
            or not all(
                np.isfinite(value) and value > 0.0
                for value in (*one_step_timings, *two_step_timings)
            )
            or not one_step_timings[0] <= one_step_timings[1] <= one_step_timings[2]
            or not two_step_timings[0] <= two_step_timings[1] <= two_step_timings[2]
        ):
            raise ValueError("official JEPA-WM output or runtime qualification is invalid")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_no_replace(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"upstream qualification output exists: {target}")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace publication requires renameat2")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1)
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(f"upstream qualification output exists: {target}")
    raise OSError(error, os.strerror(error), str(target))


def write_upstream_qualification_manifest(
    *,
    output_dir: Path,
    reference: UpstreamReference,
    reference_config_path: Path,
    environment_lock_path: Path,
    checkout: UpstreamCheckout,
    mirrors: tuple[CheckpointMirror, ...],
    qualifications: tuple[OfficialPredictorQualification, ...],
    runtime: dict[str, Any],
) -> Path:
    target = output_dir.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"upstream qualification output exists: {target}")
    mirror_by_name = {row.name: row for row in mirrors}
    qualification_by_name = {row.name: row for row in qualifications}
    if (
        set(mirror_by_name) != {"metaworld", "droid"}
        or len(mirror_by_name) != len(mirrors)
        or set(qualification_by_name) != {"metaworld", "droid"}
        or len(qualification_by_name) != len(qualifications)
        or any(
            qualification_by_name[name].checkpoint_sha256 != mirror_by_name[name].sha256
            or qualification_by_name[name].checkpoint_size_bytes != mirror_by_name[name].size_bytes
            or not qualification_by_name[name].finite
            for name in mirror_by_name
        )
        or mirror_by_name["metaworld"].sha256 != reference.metaworld_checkpoint_sha256
        or mirror_by_name["droid"].sha256 != reference.droid_checkpoint_sha256
    ):
        raise ValueError("official JEPA-WM mirror and predictor qualifications disagree")
    if (
        checkout.commit != reference.commit
        or checkout.license_sha256 != reference.license_sha256
        or checkout.reference_config_sha256 != reference.reference_config_sha256
        or checkout.droid_reference_config_sha256 != reference.droid_reference_config_sha256
    ):
        raise ValueError("official JEPA-WM checkout or runtime identity is invalid")
    validate_locked_runtime(runtime)
    reference_path = reference_config_path.resolve()
    environment_lock = environment_lock_path.resolve()
    if not reference_path.is_file() or not environment_lock.is_file():
        raise FileNotFoundError("upstream reference config or environment lock is missing")
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f".{target.name}.building-{uuid.uuid4().hex}"
    building.mkdir()
    try:
        payload = {
            "schema_version": 1,
            "format_id": "action_conditioned_jepa_upstream_qualification_v1",
            "qualification_pass": True,
            "reference": asdict(reference),
            "reference_file": {
                "path": str(reference_path),
                "sha256": sha256_file(reference_path),
            },
            "environment_lock": {
                "path": str(environment_lock),
                "sha256": sha256_file(environment_lock),
            },
            "checkout": {
                **asdict(checkout),
                "root": str(checkout.root),
            },
            "checkpoints": {
                name: {
                    **asdict(row),
                    "direct_path": str(row.direct_path),
                    "hub_path": str(row.hub_path),
                }
                for name, row in sorted(mirror_by_name.items())
            },
            "predictors": {
                name: asdict(row) for name, row in sorted(qualification_by_name.items())
            },
            "runtime": runtime,
            "runtime_adaptations": {
                "torchvision_patch_pair": "torch-2.7.1_torchvision-0.22.1",
                "checkpoint_loading": "pinned_official_sha256_weights_only_true",
            },
        }
        manifest = building / "manifest.json"
        _write_json(manifest, payload)
        _fsync_directory(building)
        _rename_directory_no_replace(building, target)
        _fsync_directory(target.parent)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return target / "manifest.json"


def qualify_upstream_reference(
    *,
    reference_config_path: Path,
    project_root: Path,
    mirror_root: Path,
    output_dir: Path,
    device: str,
    predictor_qualifier: Callable[..., OfficialPredictorQualification] = (
        qualify_official_predictor
    ),
    runtime_collector: Callable[[str], dict[str, Any]] = collect_locked_runtime,
) -> Path:
    reference_path = reference_config_path.resolve()
    reference = load_upstream_reference(reference_path)
    checkout = verify_upstream_checkout(reference=reference, project_root=project_root.resolve())
    runtime = runtime_collector(device)
    validate_locked_runtime(runtime)
    root = mirror_root.resolve()
    mirrors = (
        verify_checkpoint_mirror(
            name="metaworld",
            direct_path=root / "direct/metaworld.pth.tar",
            hub_path=root / "hf/jepa_wm_metaworld.pth.tar",
            expected_sha256=reference.metaworld_checkpoint_sha256,
        ),
        verify_checkpoint_mirror(
            name="droid",
            direct_path=root / "direct/droid.pth.tar",
            hub_path=root / "hf/jepa_wm_droid.pth.tar",
            expected_sha256=reference.droid_checkpoint_sha256,
        ),
    )
    qualifications = tuple(
        predictor_qualifier(
            name=mirror.name,
            checkpoint_path=mirror.direct_path,
            checkpoint_sha256=mirror.sha256,
            upstream_root=checkout.root,
            device=device,
            warmup_iterations=20,
            benchmark_iterations=100,
        )
        for mirror in mirrors
    )
    return write_upstream_qualification_manifest(
        output_dir=output_dir,
        reference=reference,
        reference_config_path=reference_path,
        environment_lock_path=project_root.resolve() / "requirements/action-conditioned-jepa.lock",
        checkout=checkout,
        mirrors=mirrors,
        qualifications=qualifications,
        runtime=runtime,
    )
