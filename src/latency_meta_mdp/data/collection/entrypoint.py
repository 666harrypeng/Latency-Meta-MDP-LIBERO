"""Task dispatch for the expert collection CLI; legacy arguments remain valid."""

import argparse
import sys


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", choices=("moving_ball", "conveyor_sort"), default="moving_ball")
    task, remaining = parser.parse_known_args(args)
    if task.task == "conveyor_sort":
        from latency_meta_mdp.data.conveyor.collection import main as collect
    else:
        from latency_meta_mdp.data.collection.collect import main as collect
    return collect(remaining)
