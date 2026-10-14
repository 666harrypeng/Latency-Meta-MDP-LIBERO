"""Explicit visible-device admission, shared-process exceptions and worker locks."""

import fcntl
import os
import socket
import subprocess
from contextlib import contextmanager
from pathlib import Path


def gpu_lock_directory(device):
    """Local GPU identities must not conflict across nodes on a shared filesystem."""
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "conveyor-meta/gpu-locks" / socket.gethostname() / device


def selected_gpu_processes(inventory, active, *, visible=None, allowed=""):
    devices = dict(
        tuple(x.strip() for x in line.split(",", 1))
        for line in inventory.splitlines()
        if line.strip()
    )
    selected = set(devices.values())
    if visible is not None:
        tokens = visible.split(",")
        if not tokens or any(t not in devices and t not in devices.values() for t in tokens):
            raise ValueError("CUDA_VISIBLE_DEVICES must contain explicit GPU indices or full UUIDs")
        selected = {devices.get(t, t) for t in tokens}
        if len(selected) != len(tokens):
            raise ValueError("duplicate visible GPU")
    exempt = set(allowed.split(",")) - {""}
    if any(not p.isdigit() or int(p) <= 0 for p in exempt):
        raise ValueError("shared GPU exceptions require explicit positive PIDs")
    rows = []
    for line in active.splitlines():
        if not line.strip():
            continue
        uuid, pid, name = (x.strip() for x in line.split(",", 2))
        if uuid in selected and pid not in exempt:
            rows.append(f"{pid}, {name}")
    return "\n".join(rows)


def gpu_processes():
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
    ).strip()
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"],
        text=True,
    ).strip()
    selected = selected_gpu_processes(
        inventory,
        active,
        visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
        allowed=os.environ.get("CONVEYOR_SHARED_GPU_PIDS", ""),
    )
    # Preserve the existing parked-learner check (only a synchronous direct parent).
    from latency_meta_mdp.meta.information.preflight import blocking_gpu_processes

    return blocking_gpu_processes(selected), active


@contextmanager
def worker_lock(directory):
    """Prevent concurrent resumes from writing the same cohort/worker directory."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    with (path / ".writer.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"collector already owns {path}") from error
        yield


@contextmanager
def interrupt_on_termination():
    """Run coordinator cleanup when its waiting learner terminates evaluation."""
    import signal

    def interrupt(signum, frame):
        raise InterruptedError("rollout coordinator terminated")

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)
