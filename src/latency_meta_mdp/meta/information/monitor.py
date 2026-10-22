"""Real-simulator validation between FQI iterations or over saved iterates."""

import gc
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

from latency_meta_mdp.meta.information.config import (
    load_config,
    sha256,
    validate_binding,
)
from latency_meta_mdp.meta.information.runtime_identity import source_identity
from latency_meta_mdp.meta.information.validation import (
    atomic_json,
    record_validation,
    retain_checkpoints,
    validation_metrics,
    verified_validation_metrics,
)


class IterationValidator:
    def __init__(
        self,
        config_path,
        output_dir,
        *,
        root,
        audit_path,
        bootstrap_dir,
        model=None,
        progress=None,
        prune=True,
        source_manifest=None,
        worker_python=None,
        gpus=None,
        cpu_threads=4,
        scratch_dir=None,
        workers_per_gpu=1,
    ):
        self.root = Path(root).resolve()
        self.worker_python = str(worker_python or sys.executable)
        if type(cpu_threads) is not int or cpu_threads < 1:
            raise ValueError("worker CPU threads must be positive")
        self.cpu_threads = cpu_threads
        self.gpus = gpus
        self.scratch_dir = scratch_dir
        self.workers_per_gpu = workers_per_gpu
        self.config_path, self.audit_path = Path(config_path).resolve(), Path(audit_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bootstrap_dir = Path(bootstrap_dir).resolve()
        self.model, self.progress, self.prune = model, progress, prune
        self.source_manifest = (
            Path(source_manifest).resolve() if source_manifest is not None else None
        )
        config = load_config(self.config_path)
        scenes = config["splits"]["validation"][:3]
        for scene in scenes:
            if not (self.bootstrap_dir / f"seed-{scene}.npz").is_file():
                raise ValueError("validation requires existing verified bootstrap files")
        source = source_identity(self.root)
        self.collection_code_hash = source["code_hash"]
        self.protocol = dict(
            scenes=scenes,
            regimes=["nominal", "family", "zero"],
            config_sha256=sha256(self.config_path),
            audit_sha256=sha256(self.audit_path),
            code_hash=source["code_hash"],
            epsilon=0.0,
            selection="mean_nominal_family_incumbent_on_ties",
            delay="actual_zero_with_unchanged_RTC_estimator_and_reserve",
        )
        if config.get("schema_version", 1) == 2:
            self.protocol["starvation_mode"] = config["actions"]["starvation_mode"]
        if self.source_manifest is not None:
            self.protocol["source_manifest_sha256"] = sha256(self.source_manifest)
        protocol_file = self.output_dir / "protocol.json"
        if protocol_file.exists():
            previous = json.loads(protocol_file.read_text())
            if previous["code_hash"] == source.get("previous_execution_code_hash"):
                self.protocol["code_hash"] = previous["code_hash"]
            validate_binding(self.protocol, previous)
        else:
            atomic_json(protocol_file, self.protocol)
        atomic_json(self.output_dir / "runtime-source.json", source)

    def __call__(self, checkpoint, receipt):
        import torch

        checkpoint = Path(checkpoint).resolve()
        iteration = receipt["iteration"]
        destination = self.output_dir / f"iteration-{iteration:04d}"
        destination.mkdir(parents=True, exist_ok=True)
        link = destination / "bootstrap"
        if not link.exists():
            link.symlink_to(self.bootstrap_dir, target_is_directory=True)
        elif link.resolve() != self.bootstrap_dir:
            raise ValueError("validation bootstrap location differs")
        state_path = checkpoint.parent / "validation-state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            validate_binding(self.protocol, state["protocol"])
            previous = state["completed"].get(str(iteration))
            if previous is not None:
                metrics = verified_validation_metrics(
                    checkpoint, destination, previous, self.protocol
                )
                if self.progress is not None:
                    self.progress(
                        dict(phase="validation_complete", iteration=iteration, metrics=metrics)
                    )
                if self.prune:
                    retain_checkpoints(checkpoint.parent, best_iteration=state["best_iteration"])
                return metrics
        print(
            json.dumps(
                dict(
                    phase="validation_started",
                    iteration=iteration,
                    episodes=9,
                    output=str(destination),
                )
            ),
            flush=True,
        )
        command = [
            self.worker_python,
            "-u",
            str(self.root / "scripts/run_conveyor_meta.py"),
            "parallel-collect" if self.gpus else "cycle-collect",
            "--config",
            str(self.config_path),
            "--scene-audit",
            str(self.audit_path),
            "--checkpoint",
            str(checkpoint),
            "--phase",
            "monitor",
            "--meta-only",
            "--no-wandb",
            "--output-dir",
            str(destination),
        ]
        if self.scratch_dir is not None:
            command.extend(["--rollout-scratch-dir", str(Path(self.scratch_dir).resolve())])
        if self.gpus:
            command.extend(
                [
                    "--gpus",
                    self.gpus,
                    "--worker-python",
                    self.worker_python,
                    "--bootstrap-dir",
                    str(self.bootstrap_dir),
                    "--cpu-threads",
                    str(self.cpu_threads),
                    "--rollout-workers-per-gpu",
                    str(self.workers_per_gpu),
                ]
            )
        if self.source_manifest is not None:
            validate_binding(self.protocol["source_manifest_sha256"], sha256(self.source_manifest))
            command.extend(["--offline-dir", str(self.source_manifest.parent)])
        env = dict(os.environ)
        # The caller supplies its runtime overlays through PYTHONPATH.
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (env.get("PYTHONPATH", ""), str(self.root / "src")) if p
        )
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[key] = str(self.cpu_threads)
        device = next(self.model.parameters()).device if self.model is not None else None
        try:
            if device is not None and device.type == "cuda":
                torch.cuda.synchronize(device)
                self.model.cpu()
                gc.collect()
                # cuBLAS workspaces survive model.cpu() and count as allocated
                # memory despite holding no live model tensors (pinned PyTorch).
                torch._C._cuda_clearCublasWorkspaces()
                torch.cuda.empty_cache()
                if torch.cuda.memory_allocated(device) != 0:
                    raise RuntimeError("learner still holds live GPU tensors before validation")
                # Only this direct, synchronously waiting parent is exempted by preflight.
                env["CONVEYOR_IDLE_LEARNER_PID"] = str(os.getpid())
            with (destination / "execution.log").open("a") as log:
                process = subprocess.Popen(
                    command,
                    cwd=self.root,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                try:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        if line.startswith(
                            ("EPISODE_COMPLETE ", "PARALLEL_PROGRESS ", "WANDB_RUN ", "tick=")
                        ):
                            print(line, end="", flush=True)
                    if process.wait() != 0:
                        raise RuntimeError(
                            f"validation failed; see {destination / 'execution.log'}"
                        )
                finally:
                    if process.poll() is None:
                        process.terminate()
                        process.wait()
        finally:
            if self.model is not None and device is not None:
                self.model.to(device)
        manifest = json.loads((destination / "cohort.json").read_text())
        validate_binding(manifest["binding"]["code_hash"], self.collection_code_hash)
        validate_binding(
            manifest["binding"]["policy"]["receipt_sha256"], sha256(checkpoint / "complete.json")
        )
        summary = json.loads((destination / "rollout-summary.json").read_text())
        if summary["completed"] != 9 or summary["expected"] != 9:
            raise ValueError("incomplete validation cohort")
        metrics = validation_metrics(
            summary["episodes"],
            scenes=self.protocol["scenes"],
            starvation_mode=self.protocol.get("starvation_mode", "guarded"),
        )
        state = record_validation(
            checkpoint.parent,
            iteration=iteration,
            metrics=metrics,
            protocol=self.protocol,
            evaluation_sha256=sha256(destination / "rollout-summary.json"),
        )
        event = dict(phase="validation_complete", iteration=iteration, metrics=metrics)
        if self.progress is not None:
            self.progress(event)
        else:
            print(json.dumps(event), flush=True)
        if self.prune:
            retain_checkpoints(checkpoint.parent, best_iteration=state["best_iteration"])
        return metrics


class ValidationTracker:
    def __init__(self, directory, *, protocol):
        import wandb
        from latency_meta_mdp.meta.tracking_transport import start_run

        directory = Path(directory)
        path = directory / "wandb-run.json"
        state = json.loads(path.read_text()) if path.exists() else dict(id=uuid.uuid4().hex[:8])
        self.run = start_run(
            directory,
            entity=os.environ.get("WANDB_ENTITY"),
            project=os.environ.get("WANDB_PROJECT", "latency-meta-mdp-robosuite"),
            id=state["id"],
            resume="allow",
            name=directory.name,
            group="conveyor-meta-bound-fqi-cycle-v1",
            job_type="retrospective-validation",
            tags=["conveyor", "meta", "fqi", "validation"],
            config=dict(protocol=protocol, retrospective=True),
            dir=str(directory),
            settings=wandb.Settings(disable_git=True, save_code=False, console="off"),
        )
        state["url"] = self.run.url
        atomic_json(path, state)
        self.run.define_metric("validation/iteration")
        self.run.define_metric("validation/*", step_metric="validation/iteration")
        print("WANDB_RUN " + str(self.run.url), flush=True)

    def __call__(self, event):
        print(json.dumps(event), flush=True)
        if event["iteration"] < self.run.step:
            return
        self.run.log(
            {
                "validation/iteration": event["iteration"],
                **{f"validation/{k}": v for k, v in event["metrics"].items()},
            },
            step=event["iteration"],
        )

    def close(self, *, success):
        self.run.finish(exit_code=0 if success else 1)


def run_monitor_series(
    config_path,
    checkpoint_dir,
    output_dir,
    *,
    root,
    audit_path,
    bootstrap_dir,
    source_manifest=None,
    worker_python=None,
    gpus=None,
    cpu_threads=4,
    scratch_dir=None,
    workers_per_gpu=1,
):
    checkpoint_dir = Path(checkpoint_dir).resolve()
    entries = sorted(checkpoint_dir.glob("iteration-*/complete.json"))
    if not entries:
        raise ValueError("no completed checkpoints to evaluate")
    if source_manifest is None and any(
        "cohort_sha256" in json.loads(entry.read_text())["binding"] for entry in entries
    ):
        raise ValueError(
            "offline checkpoints require --offline-dir with the original cohort manifest"
        )
    validator = IterationValidator(
        config_path,
        output_dir,
        root=root,
        audit_path=audit_path,
        bootstrap_dir=bootstrap_dir,
        prune=False,
        source_manifest=source_manifest,
        worker_python=worker_python,
        gpus=gpus,
        cpu_threads=cpu_threads,
        scratch_dir=scratch_dir,
        workers_per_gpu=workers_per_gpu,
    )
    tracker = ValidationTracker(validator.output_dir, protocol=validator.protocol)
    validator.progress = tracker
    success = False
    try:
        for entry in entries:
            validator(entry.parent, json.loads(entry.read_text()))
        state = json.loads((checkpoint_dir / "validation-state.json").read_text())
        retain_checkpoints(checkpoint_dir, best_iteration=state["best_iteration"])
        success = True
    finally:
        tracker.close(success=success)
