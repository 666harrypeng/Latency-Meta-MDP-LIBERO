"""Mirror a running FQI log to W&B without modifying or restarting training."""

import argparse
from pathlib import Path

from latency_meta_mdp.meta.tracking import track_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--trainer-pid", type=int, required=True)
    track_training(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
