"""Task dispatch for frozen vision feature preparation."""

import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", choices=("moving_ball", "conveyor_sort"), default="moving_ball")
    task, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if task.task == "conveyor_sort":
        from latency_meta_mdp.data.conveyor.vision import main as prepare
    else:
        from latency_meta_mdp.data.vision.prepare import main as prepare
    return prepare(remaining)
