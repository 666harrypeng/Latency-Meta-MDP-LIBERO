"""Task dispatch for policy dataset preparation."""

import argparse
import sys


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", choices=("moving_ball", "conveyor_sort"), default="moving_ball")
    task, remaining = parser.parse_known_args(args)
    if task.task == "conveyor_sort":
        from latency_meta_mdp.data.conveyor.prepare_policy import main as prepare

        return prepare(remaining)
    from latency_meta_mdp.data.prepare_policy import main as prepare

    return prepare(remaining)
