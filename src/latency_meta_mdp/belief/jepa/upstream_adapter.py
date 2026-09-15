"""Verified boundary around the pinned upstream JEPA-WM implementation."""

from __future__ import annotations

import importlib
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import torch
import torch.nn.functional as F
import yaml
from torch.nn.attention import SDPBackend, sdpa_kernel

from latency_meta_mdp.io.artifacts import sha256_file

_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REFERENCE_ID = "jepa_wms_adaln_depth6_metaworld"
_REPOSITORY = "https://github.com/facebookresearch/jepa-wms.git"


def _is_relative_file_path(value: str) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts and path.name != ""


def _is_https_url(value: str) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc) and not parsed.username


@dataclass(frozen=True)
class UpstreamReference:
    schema_version: int
    reference_id: str
    repository: str
    commit: str
    license_path: str
    license_sha256: str
    reference_config_path: str
    reference_config_sha256: str
    droid_reference_config_path: str
    droid_reference_config_sha256: str
    metaworld_checkpoint_url: str
    droid_checkpoint_url: str
    metaworld_checkpoint_sha256: str
    droid_checkpoint_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.reference_id != _REFERENCE_ID
        ):
            raise ValueError("unsupported JEPA-WM upstream reference")
        if self.repository != _REPOSITORY or _GIT_SHA.fullmatch(self.commit) is None:
            raise ValueError("JEPA-WM repository or commit is invalid")
        for name in (
            "license_path",
            "reference_config_path",
            "droid_reference_config_path",
        ):
            if not _is_relative_file_path(getattr(self, name)):
                raise ValueError(f"JEPA-WM {name} must be a safe relative file path")
        for name in (
            "license_sha256",
            "reference_config_sha256",
            "droid_reference_config_sha256",
            "metaworld_checkpoint_sha256",
            "droid_checkpoint_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"JEPA-WM {name} must be a SHA256 digest")
        for name in ("metaworld_checkpoint_url", "droid_checkpoint_url"):
            if not _is_https_url(getattr(self, name)):
                raise ValueError(f"JEPA-WM {name} must be an HTTPS URL")


@dataclass(frozen=True)
class UpstreamCheckout:
    root: Path
    commit: str
    license_sha256: str
    reference_config_sha256: str
    droid_reference_config_sha256: str


@dataclass(frozen=True)
class UpstreamPrimitiveBundle:
    rope_attention_type: type[Any]
    fw_adaln_block_type: type[Any]
    vision_transformer_adaln_type: type[Any]
    mlp_type: type[Any]
    drop_path_type: type[Any]
    modulate: Callable[..., Any]
    rotate_queries_or_keys: Callable[..., Any]
    trunc_normal: Callable[..., Any]


@dataclass(frozen=True)
class DualViewUpstreamTypes:
    coordinate_attention_type: type[Any]
    coordinate_adaln_block_type: type[Any]


@dataclass(frozen=True)
class CheckpointMirror:
    name: str
    direct_path: Path
    hub_path: Path
    sha256: str
    size_bytes: int


def load_upstream_reference(path: Path) -> UpstreamReference:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != set(UpstreamReference.__dataclass_fields__):
        raise ValueError("JEPA-WM upstream reference fields are invalid")
    return UpstreamReference(**raw)


def _git_output(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def verify_upstream_checkout(
    *,
    reference: UpstreamReference,
    project_root: Path,
) -> UpstreamCheckout:
    root = (project_root / "third_party/jepa-wms").resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"JEPA-WM submodule is missing: {root}")
    commit = _git_output(root, "rev-parse", "HEAD")
    remote = _git_output(root, "remote", "get-url", "origin")
    status = _git_output(root, "status", "--porcelain=v1")
    if commit != reference.commit or remote != reference.repository:
        raise ValueError("JEPA-WM checkout identity disagrees with the pinned reference")
    if status:
        raise ValueError("JEPA-WM submodule must remain unmodified")
    license_path = root / reference.license_path
    reference_config = root / reference.reference_config_path
    droid_reference_config = root / reference.droid_reference_config_path
    observed = (
        sha256_file(license_path),
        sha256_file(reference_config),
        sha256_file(droid_reference_config),
    )
    expected = (
        reference.license_sha256,
        reference.reference_config_sha256,
        reference.droid_reference_config_sha256,
    )
    if observed != expected:
        raise ValueError("JEPA-WM source/config digests disagree with the pinned reference")
    return UpstreamCheckout(
        root=root,
        commit=commit,
        license_sha256=observed[0],
        reference_config_sha256=observed[1],
        droid_reference_config_sha256=observed[2],
    )


def load_upstream_primitives(
    *,
    reference: UpstreamReference,
    project_root: Path,
) -> UpstreamPrimitiveBundle:
    checkout = verify_upstream_checkout(reference=reference, project_root=project_root)
    # Evaluation environments need not install the upstream Python distribution.
    # Import from the verified checkout and retain the origin checks below.
    if str(checkout.root) not in sys.path:
        sys.path.insert(0, str(checkout.root))
    modules = importlib.import_module("src.models.utils.modules")
    adaln = importlib.import_module("app.plan_common.models.AdaLN_vit")
    tensors = importlib.import_module("src.utils.tensors")
    for module in (modules, adaln, tensors):
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            raise ValueError("JEPA-WM upstream module lacks a source file")
        try:
            Path(module_file).resolve().relative_to(checkout.root)
        except ValueError as error:
            raise ValueError(
                "JEPA-WM imported module resolves outside verified checkout"
            ) from error
    return UpstreamPrimitiveBundle(
        rope_attention_type=modules.RoPEAttention,
        fw_adaln_block_type=adaln.FWAdaLNBlock,
        vision_transformer_adaln_type=adaln.VisionTransformerAdaLN,
        mlp_type=modules.MLP,
        drop_path_type=modules.DropPath,
        modulate=adaln.modulate,
        rotate_queries_or_keys=modules.rotate_queries_or_keys,
        trunc_normal=tensors.trunc_normal_,
    )


def build_dual_view_upstream_types(
    primitives: UpstreamPrimitiveBundle,
) -> DualViewUpstreamTypes:
    """Derive only coordinate-aware forwards from the verified upstream classes."""

    if not isinstance(primitives, UpstreamPrimitiveBundle):
        raise TypeError("primitives must be an UpstreamPrimitiveBundle")
    rotate = primitives.rotate_queries_or_keys
    modulate = primitives.modulate
    attention_base = primitives.rope_attention_type
    block_base = primitives.fw_adaln_block_type
    backends = [
        SDPBackend.MATH,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.CUDNN_ATTENTION,
    ]

    class DualViewCoordinateRoPEAttention(attention_base):
        def forward(self, x, *, coordinates, attn_mask=None):
            batch_size, token_count, channels = x.shape
            if (
                not isinstance(coordinates, torch.Tensor)
                or coordinates.ndim != 3
                or coordinates.shape[-1] != 3
                or coordinates.numel() // 3 != token_count
            ):
                raise ValueError("RoPE coordinates must have shape [T,K,3] aligned with tokens")
            positions = coordinates.reshape(token_count, 3).to(
                device=x.device,
                dtype=x.dtype,
            )
            qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
            query, key, value = qkv[0], qkv[1], qkv[2]
            offset = 0
            rotated_query = []
            rotated_key = []
            for coordinate_index, width in enumerate((self.d_dim, self.h_dim, self.w_dim)):
                rotated_query.append(
                    rotate(
                        query[..., offset : offset + width],
                        pos=positions[:, coordinate_index],
                    )
                )
                rotated_key.append(
                    rotate(
                        key[..., offset : offset + width],
                        pos=positions[:, coordinate_index],
                    )
                )
                offset += width
            if offset < self.head_dim:
                rotated_query.append(query[..., offset:])
                rotated_key.append(key[..., offset:])
            query = torch.cat(rotated_query, dim=-1)
            key = torch.cat(rotated_key, dim=-1)
            if attn_mask is not None:
                if tuple(attn_mask.shape) != (token_count, token_count):
                    raise ValueError("attention mask shape does not match coordinate tokens")
                with sdpa_kernel(backends):
                    attended = F.scaled_dot_product_attention(
                        query,
                        key,
                        value,
                        dropout_p=self.proj_drop_prob,
                        is_causal=self.is_causal,
                        attn_mask=attn_mask,
                    )
            else:
                weights = (query @ key.transpose(-2, -1)) * self.scale
                weights = self.attn_drop(weights.softmax(dim=-1))
                attended = weights @ value
            attended = attended.transpose(1, 2).reshape(
                batch_size,
                token_count,
                channels,
            )
            return self.proj_drop(self.proj(attended))

    class DualViewFWAdaLNBlock(block_base):
        def forward(
            self,
            x,
            z,
            *,
            coordinates,
            attn_mask,
            tokens_per_time,
        ):
            if (
                type(tokens_per_time) is not int
                or tokens_per_time <= 0
                or x.shape[1] != z.shape[1] * tokens_per_time
            ):
                raise ValueError("AdaLN token/time alignment is invalid")
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(z).repeat_interleave(tokens_per_time, dim=1).chunk(6, dim=2)
            )
            attended = self.attn(
                modulate(self.norm1(x), shift_msa, scale_msa),
                coordinates=coordinates,
                attn_mask=attn_mask,
            )
            x = x + self.drop_path(attended * gate_msa)
            return x + self.drop_path(
                gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
            )

    DualViewCoordinateRoPEAttention.__name__ = "DualViewCoordinateRoPEAttention"
    DualViewCoordinateRoPEAttention.__qualname__ = "DualViewCoordinateRoPEAttention"
    DualViewFWAdaLNBlock.__name__ = "DualViewFWAdaLNBlock"
    DualViewFWAdaLNBlock.__qualname__ = "DualViewFWAdaLNBlock"
    return DualViewUpstreamTypes(
        coordinate_attention_type=DualViewCoordinateRoPEAttention,
        coordinate_adaln_block_type=DualViewFWAdaLNBlock,
    )


def adapt_upstream_block_for_coordinates(
    block: Any,
    *,
    types: DualViewUpstreamTypes,
) -> Any:
    """Promote an already initialized upstream block without replacing its parameters."""

    if not isinstance(types, DualViewUpstreamTypes):
        raise TypeError("types must be DualViewUpstreamTypes")
    required = ("attn", "adaLN_modulation", "norm1", "norm2", "mlp", "drop_path")
    if any(not hasattr(block, name) for name in required):
        raise TypeError("block does not satisfy the upstream FWAdaLN contract")
    block.__class__ = types.coordinate_adaln_block_type
    block.attn.__class__ = types.coordinate_attention_type
    return block


def verify_checkpoint_mirror(
    *,
    name: str,
    direct_path: Path,
    hub_path: Path,
    expected_sha256: str,
) -> CheckpointMirror:
    if name not in {"metaworld", "droid"}:
        raise ValueError("JEPA-WM checkpoint mirror name is invalid")
    direct = direct_path.resolve()
    hub = hub_path.resolve()
    if not direct.is_file() or not hub.is_file():
        raise FileNotFoundError("JEPA-WM checkpoint mirror file is missing")
    direct_size = direct.stat().st_size
    hub_size = hub.stat().st_size
    direct_sha = sha256_file(direct)
    hub_sha = sha256_file(hub)
    if not isinstance(expected_sha256, str) or _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("JEPA-WM pinned checkpoint digest is invalid")
    if (
        direct_size <= 0
        or direct_size != hub_size
        or direct_sha != hub_sha
        or direct_sha != expected_sha256
    ):
        if direct_sha == hub_sha and direct_sha != expected_sha256:
            raise ValueError("JEPA-WM checkpoint mirror bytes do not match the pinned digest")
        raise ValueError("JEPA-WM checkpoint mirror bytes disagree")
    return CheckpointMirror(
        name=name,
        direct_path=direct,
        hub_path=hub,
        sha256=direct_sha,
        size_bytes=direct_size,
    )
