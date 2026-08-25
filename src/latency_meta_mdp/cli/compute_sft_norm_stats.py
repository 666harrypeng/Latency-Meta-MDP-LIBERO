"""Compute one certified level's OpenPI normalization assets locally."""

from __future__ import annotations

import argparse
import functools
import json
import sys
from collections.abc import Callable
from pathlib import Path

from latency_meta_mdp.openpi_runtime import temporary_patched_openpi_worktree
from latency_meta_mdp.sft_norm_stats import (
    NormStatsBackend,
    compute_level_sft_norm_stats,
)
from latency_meta_mdp.sft_profile import load_sft_profile


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--derived-manifest", type=Path, required=True)
    parser.add_argument("--certification-manifest", type=Path, required=True)
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("configs/policy/pi05_panda_ball_full_sft_h50_v2.yaml"),
    )
    parser.add_argument(
        "--patch",
        type=Path,
        default=Path("patches/openpi/0001-filter-incomplete-action-chunks.patch"),
    )
    parser.add_argument("--openpi-root", type=Path)
    parser.add_argument("--level", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--data-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _run(args: argparse.Namespace, *, compute_backend: NormStatsBackend) -> Path:
    print(f"[sft-norm][L{args.level}] start", file=sys.stderr, flush=True)
    manifest = compute_level_sft_norm_stats(
        project_root=args.project_root,
        derived_manifest=args.derived_manifest,
        certification_manifest=args.certification_manifest,
        profile_path=args.profile,
        patch_path=args.patch,
        level=args.level,
        data_revision=args.data_revision,
        output_dir=args.output_dir,
        compute_backend=compute_backend,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    print(
        f"[sft-norm][L{args.level}] done source_count={payload['source_count']}",
        file=sys.stderr,
        flush=True,
    )
    return manifest


def main(
    argv: list[str] | None = None,
    *,
    compute_backend: NormStatsBackend | None = None,
) -> int:
    args = _parser().parse_args(argv)
    if compute_backend is not None:
        manifest = _run(args, compute_backend=compute_backend)
    else:
        if args.openpi_root is None:
            raise ValueError("--openpi-root is required for real norm-stat computation")
        profile = load_sft_profile(args.profile)
        with temporary_patched_openpi_worktree(
            openpi_root=args.openpi_root,
            patch_path=args.patch,
            expected_revision=profile.openpi_revision,
        ) as patched_root:
            sys.path.insert(0, str(patched_root / "src"))
            try:
                from latency_meta_mdp.openpi_sft import compute_openpi_norm_stats

                backend: Callable = functools.partial(
                    compute_openpi_norm_stats,
                    profile=profile,
                    openpi_root=patched_root,
                )
                manifest = _run(args, compute_backend=backend)
            finally:
                sys.path.remove(str(patched_root / "src"))
    print(json.dumps({"manifest": manifest.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
