"""FQI online-refit telemetry; rewards are logged by real rollout jobs only."""

import json
import os
import uuid
from pathlib import Path


def recover_events(path):
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    boundary = data.rfind(b"\n") + 1
    history = [json.loads(line) for line in data[:boundary].splitlines()]
    if boundary < len(data):
        path.with_name(path.name + ".partial-" + uuid.uuid4().hex[:8]).write_bytes(data[boundary:])
        with path.open("r+b") as stream:
            stream.truncate(boundary)
    return history


def fit_metrics(event, *, episodes, epochs):
    phase = event["phase"]
    iteration = event["iteration"]
    if phase == "validation_complete":
        return {
            "validation/iteration": iteration,
            "validation/optimizer_updates": event["actual_update"],
            **{f"validation/{k}": v for k, v in event["metrics"].items()},
        }
    if phase == "fit":
        source = event["source"]
        if source not in ("offline", "online"):
            raise ValueError("invalid training source")
        return dict(
            {
                "train/update": event.get(
                    "actual_update",
                    ((iteration - 1) * epochs + event["epoch"] - 1) * episodes + event["episode"],
                ),
                "train/attempt": event.get("attempt", 1),
                "train/resume_iteration": event.get("resume_iteration", 0),
                f"train/attempt_{event.get('attempt', 1)}/combined_loss": event["loss"],
                "train/model_epoch": (iteration - 1) * epochs + event["epoch"],
                "train/global_epoch": (event["actual_update"] - 1) // episodes + 1
                if "actual_update" in event
                else (iteration - 1) * epochs + event["epoch"],
                "train/iteration": iteration,
                "train/combined_loss": event["loss"],
                f"train/{source}/combined_loss": event["loss"],
                f"train/{source}/huber_loss": event["huber_loss"],
                f"train/{source}/conservative_loss": event["conservative_loss"],
            }
        )
    if phase == "iteration_complete":
        return {
            "fqi/iteration": iteration,
            **{
                f"fqi/replay_cache/{k}": v
                for k, v in event["metrics"].get("replay_cache", {}).items()
            },
            **{f"fqi/{k}": v for k, v in event["metrics"].items() if isinstance(v, (float, int))},
        }
    return {"progress/iteration": iteration, "progress/target_episode": event["episode"]}


class OnlineFitTracker:
    def __init__(self, directory, *, binding, episodes, epochs):
        import wandb
        from latency_meta_mdp.meta.information.config import identity_hash

        self.directory, self.episodes, self.epochs = Path(directory), episodes, epochs
        identity_path = self.directory / "wandb-run.json"
        if identity_path.exists():
            state = json.loads(identity_path.read_text())
            if state["binding_hash"] != identity_hash(binding):
                raise ValueError("tracker training binding differs")
        else:
            state = dict(id=uuid.uuid4().hex[:8], binding_hash=identity_hash(binding))
            identity_path.write_text(json.dumps(state))
        from latency_meta_mdp.meta.tracking_transport import start_run

        self.run = start_run(
            self.directory,
            entity=os.environ.get("WANDB_ENTITY"),
            project=os.environ.get("WANDB_PROJECT", "latency-meta-mdp-robosuite"),
            id=state["id"],
            resume="allow",
            name=self.directory.name,
            group="conveyor-meta-bound-fqi-cycle-v1",
            job_type="online-refit",
            tags=["conveyor", "meta", "fqi", "bound", "lambda0"],
            config=binding,
            dir=str(self.directory),
            settings=wandb.Settings(disable_git=True, save_code=False, console="off"),
        )
        state["url"] = self.run.url
        identity_path.write_text(json.dumps(state, indent=2) + "\n")
        self.run.define_metric("train/update")
        self.run.define_metric("train/*", step_metric="train/update")
        self.run.define_metric("validation/iteration")
        self.run.define_metric("validation/*", step_metric="validation/iteration")
        self.run.define_metric("fqi/iteration")
        self.run.define_metric("fqi/*", step_metric="fqi/iteration")
        self.log_path = self.directory / "events.jsonl"
        history = recover_events(self.log_path)
        self.sequence = len(history)
        self.actual_update = sum(item["event"]["phase"] == "fit" for item in history)
        self.attempt = 1 + max((item["event"].get("attempt", 1) for item in history), default=0)
        checkpoints = sorted(self.directory.glob("iteration-*/complete.json"))
        self.resume_iteration = (
            json.loads(checkpoints[-1].read_text())["iteration"] if checkpoints else 0
        )
        discarded = [
            dict(
                sequence=item["sequence"],
                attempt=item["event"].get("attempt", 1),
                iteration=item["event"]["iteration"],
            )
            for item in history
            if item["event"]["iteration"] > self.resume_iteration
        ]
        if discarded:
            (self.directory / f"discarded-before-attempt-{self.attempt}.json").write_text(
                json.dumps(discarded, indent=2)
            )
            self.run.summary["recovery/discarded_events"] = discarded
        self.run.summary["recovery/loss_axis"] = (
            "Actual optimizer work; attempt curves distinguish rolled-back iterations"
        )
        for item in history:
            if item["sequence"] >= self.run.step:
                self.run.log(
                    fit_metrics(item["event"], episodes=episodes, epochs=epochs),
                    step=item["sequence"],
                )
        self.run.summary.update(
            {
                "evaluation/status": "validation_after_each_iteration"
                if "validation_protocol" in binding
                else "separate_greedy_rollout_required",
                "data/offline_episodes": 128,
                "data/online_episodes": 32,
            }
        )
        print("WANDB_RUN " + str(self.run.url), flush=True)

    def __call__(self, event):
        if event["phase"] == "fit":
            self.actual_update += 1
        event = dict(
            event,
            actual_update=self.actual_update,
            attempt=self.attempt,
            resume_iteration=self.resume_iteration,
        )
        if event["phase"] == "validation_complete":
            self.run.summary["evaluation/latest_iteration"] = event["iteration"]
            self.run.summary["evaluation/latest_metrics"] = event["metrics"]
        self.sequence += 1
        with self.log_path.open("a") as stream:
            stream.write(json.dumps(dict(sequence=self.sequence, event=event)) + "\n")
        print(json.dumps(event), flush=True)
        self.run.log(
            fit_metrics(event, episodes=self.episodes, epochs=self.epochs), step=self.sequence
        )

    def close(self, *, success):
        self.run.finish(exit_code=0 if success else 1)
