"""Best-effort external telemetry; durable local records remain authoritative."""

import json
from pathlib import Path
from time import time


def _attempt(directory, operation, call):
    try:
        return call()
    except Exception as error:
        # SDK error strings may contain authentication details; store only type.
        path = Path(directory) / "wandb-errors.jsonl"
        with path.open("a") as stream:
            stream.write(
                json.dumps(dict(time=time(), operation=operation, error_type=type(error).__name__))
                + "\n"
            )
        print(
            f"W&B {operation} unavailable ({type(error).__name__}); local records retained",
            flush=True,
        )
        return None


class _Summary:
    def __init__(self, run):
        self.run = run

    def update(self, values):
        return self.run._call("summary", lambda: self.run.remote.summary.update(values))

    def __setitem__(self, key, value):
        self.update({key: value})


class SafeRun:
    def __init__(self, remote, directory):
        self.remote, self.directory = remote, Path(directory)
        self.summary = _Summary(self)

    @property
    def step(self):
        return self.remote.step if self.remote is not None else 0

    @property
    def url(self):
        return self.remote.url if self.remote is not None else None

    def _call(self, operation, call):
        return _attempt(self.directory, operation, call) if self.remote is not None else None

    def log(self, *args, **kwargs):
        return self._call("log", lambda: self.remote.log(*args, **kwargs))

    def define_metric(self, *args, **kwargs):
        return self._call("define_metric", lambda: self.remote.define_metric(*args, **kwargs))

    def finish(self, *args, **kwargs):
        return self._call("finish", lambda: self.remote.finish(*args, **kwargs))


def start_run(directory, **kwargs):
    import wandb

    return SafeRun(_attempt(directory, "init", lambda: wandb.init(**kwargs)), directory)
