"""Publish selected policy inference milestones without training state."""

from __future__ import annotations

import argparse
from pathlib import Path

from latency_meta_mdp.io.paths import repository_root
from latency_meta_mdp.io.policy_publish import publish_checkpoints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    revision = publish_checkpoints(
        root=args.checkpoint_dir,
        steps=tuple(args.steps),
        repo_id=args.repo,
        notices=repository_root() / "licenses/pi05",
    )
    print(revision)


if __name__ == "__main__":
    main()
