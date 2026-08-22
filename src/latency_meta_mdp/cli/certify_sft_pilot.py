"""Certify three level-specific pilot datasets through the pinned OpenPI loaders."""

from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

from latency_meta_mdp.openpi_patch import apply_openpi_patch
from latency_meta_mdp.sft_certification import LevelProbe, certify_lerobot_pilot_run
from latency_meta_mdp.sft_profile import load_sft_profile


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--derived-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("configs/policy/pi05_panda_ball_full_sft_v1.yaml"),
    )
    parser.add_argument("--openpi-root", type=Path)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    level_probe: LevelProbe | None = None,
) -> int:
    args = _parser().parse_args(argv)
    probe = level_probe
    if probe is None:
        if args.openpi_root is None:
            raise ValueError("--openpi-root is required for real certification")
        project_root = args.project_root.resolve()
        openpi_root = args.openpi_root.resolve()
        patch = project_root / "patches/openpi/0001-filter-incomplete-action-chunks.patch"
        profile = load_sft_profile(args.profile)
        application = apply_openpi_patch(
            openpi_root=openpi_root,
            patch_path=patch,
            expected_revision=profile.openpi_revision,
        )
        if application.patch_sha256 != profile.openpi_patch_sha256:
            raise ValueError("applied OpenPI patch does not match the SFT profile")
        sys.path.insert(0, str(openpi_root / "src"))
        from latency_meta_mdp.openpi_sft import probe_sft_pilot_level

        probe = functools.partial(
            probe_sft_pilot_level,
            openpi_root=openpi_root,
        )

    manifest = certify_lerobot_pilot_run(
        project_root=args.project_root,
        derived_manifest=args.derived_manifest,
        output_dir=args.output_dir,
        profile_path=args.profile,
        level_probe=probe,
    )
    print(json.dumps({"manifest": manifest.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
