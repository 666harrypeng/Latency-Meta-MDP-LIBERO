"""Read-only W&B sidecar for an already-running FQI trainer."""

import json
from pathlib import Path


def read_complete_lines(stream):
    while True:
        position = stream.tell()
        line = stream.readline()
        if not line or not line.endswith("\n"):
            stream.seek(position)
            return
        yield line


def event_metrics(event, *, total, epochs):
    phase = event.get("phase")
    iteration = event.get("iteration", 0)
    if phase not in ("target", "fit", "iteration_complete"):
        return iteration, {}
    values = {"progress/iteration": iteration, "progress/phase": phase}
    if phase == "iteration_complete":
        metrics = event["metrics"]
        values.update(
            {
                f"fqi/{k}": v
                for k, v in metrics.items()
                if isinstance(v, (int, float)) and k != "loss"
            }
        )
        values["fqi/iteration"] = iteration
        values["fqi/combined_loss"] = metrics["loss"]
        values["fqi/target_unchanged"] = int(
            metrics["target_hash_before"] == metrics["target_hash_after"]
        )
        values["progress/percent"] = 100 * iteration / total
        values["progress/completed_iterations"] = iteration
    else:
        episode, count = event["episode"], event["episodes"]
        within = episode / count
        if phase == "fit":
            epoch = event["epoch"]
            within += epoch
            values.update(
                {
                    "train/update": ((iteration - 1) * epochs + epoch - 1) * count + episode,
                    "train/combined_loss": event["loss"],
                    "train/epoch": epoch,
                    "train/episode_in_epoch": episode,
                }
            )
        values["progress/percent"] = 100 * (iteration - 1 + within / (epochs + 1)) / total
    return iteration, values


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def track_training(*, training_dir, log_path, cohort_dir, entity, project, name, trainer_pid):
    import fcntl
    import hashlib
    import os
    import time
    import uuid
    from datetime import datetime, timezone

    import wandb

    training_dir, log_path, cohort_dir = map(Path, (training_dir, log_path, cohort_dir))
    directory = training_dir / "tracking"
    directory.mkdir(parents=True, exist_ok=True)
    # An extra sidecar cannot duplicate points or change trainer state.
    with (directory / "tracker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = json.loads((training_dir / "iteration-0000/complete.json").read_text())
        binding = receipt["binding"]
        recipe = binding["recipe"]
        audit = json.loads((cohort_dir / "collection-audit.json").read_text())
        if audit["remaining"] or audit["verified_complete"] != 128:
            raise ValueError("complete starter audit required for dataset reward logging")
        identity = dict(
            entity=entity,
            project=project,
            name=name,
            binding=binding,
            log_path=str(log_path.resolve()),
        )
        identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        state_path = directory / "run.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state["identity_hash"] != identity_hash:
                raise ValueError("tracker run binding differs")
        else:
            state = dict(
                id=uuid.uuid4().hex[:8],
                identity_hash=identity_hash,
                entity=entity,
                project=project,
            )
            _atomic_json(state_path, state)
        run = wandb.init(
            id=state["id"],
            resume="allow",
            entity=entity,
            project=project,
            name=name,
            job_type="offline-fqi",
            tags=["conveyor", "meta", "fqi", "bound", "lambda0"],
            dir=str(directory),
            mode="online",
            config=dict(
                **recipe,
                seed=binding["seed"],
                algorithm="fqi",
                mode=binding["mode"],
                episodes=128,
                free_transitions=sum(e["free_decisions"] for e in audit["episodes"]),
                data_manifest_sha256=binding["cohort_sha256"],
                trainer_code_hash=binding["trainer_code_hash"],
                reward_semantics="Fixed offline replay rewards; no policy evaluation yet",
                loss_semantics="Huber regression plus masked conservative regularizer",
                logging=(
                    "Backfilled and live training log; "
                    "upload timestamps are not original event timestamps"
                ),
            ),
            settings=wandb.Settings(
                disable_git=True,
                save_code=False,
                x_disable_stats=True,
                console="off",
                init_timeout=30,
            ),
        )
        state.update(url=run.url)
        _atomic_json(state_path, state)
        print("WANDB_RUN " + run.url, flush=True)
        run.define_metric("train/update", hidden=True)
        run.define_metric("train/*", step_metric="train/update")
        run.define_metric("fqi/iteration", hidden=True)
        run.define_metric("fqi/*", step_metric="fqi/iteration")
        run.define_metric("progress/event", hidden=True)
        run.define_metric("progress/*", step_metric="progress/event")
        exit_code = 1
        try:
            rewards = [e["reward"] for e in audit["episodes"]]
            run.summary.update(
                {
                    "dataset/episodes": len(rewards),
                    "dataset/reward_mean": sum(rewards) / len(rewards),
                    "dataset/zero_reward_episodes": sum(r == 0 for r in rewards),
                    "evaluation/status": "not_run",
                    "tracker/system_metrics": "Disabled: tracker metrics are not trainer metrics",
                }
            )
            if run.step == 0:
                table = wandb.Table(
                    columns=["scene", "delay", "behavior", "reward", "successes", "spawned"],
                    data=[
                        [
                            e["episode"]["scene"],
                            e["episode"]["regime"],
                            e["episode"]["behavior"],
                            e["reward"],
                            e["parcels"]["successes"],
                            e["parcels"]["spawned"],
                        ]
                        for e in audit["episodes"]
                    ],
                )
                run.log(
                    {
                        "dataset/rewards": table,
                        "dataset/reward_distribution": wandb.Histogram(rewards),
                        "dataset/reward_mean": sum(rewards) / len(rewards),
                    },
                    step=0,
                )
            minimum_step = run.step
            line_number = 0
            last_phase = None
            complete = False
            last_size = 0
            # Tail only full lines; explicit file-line steps make backfill/resume idempotent.
            with log_path.open() as stream:
                while True:
                    current_size = log_path.stat().st_size
                    if current_size < last_size:
                        raise ValueError("trainer log was truncated; refusing mixed history")
                    last_size = current_size
                    for line in read_complete_lines(stream):
                        line_number += 1
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(event, dict):
                            continue
                        iteration, values = event_metrics(
                            event, total=recipe["total_iterations"], epochs=recipe["epochs"]
                        )
                        if values:
                            last_phase = event["phase"]
                            complete = (
                                last_phase == "iteration_complete"
                                and iteration == recipe["total_iterations"]
                            )
                            if line_number >= minimum_step:
                                run.log({**values, "progress/event": line_number}, step=line_number)
                    status = dict(
                        url=run.url,
                        last_source_line=line_number,
                        last_phase=last_phase,
                        complete=complete,
                        last_sync_utc=datetime.now(timezone.utc).isoformat(),
                    )
                    _atomic_json(directory / "status.json", status)
                    run.summary["tracker/last_source_line"] = line_number
                    run.summary["tracker/last_sync_utc"] = status["last_sync_utc"]
                    if complete:
                        exit_code = 0
                        break
                    try:
                        os.kill(trainer_pid, 0)
                    except ProcessLookupError:
                        raise RuntimeError(
                            "trainer exited before a completed final iteration was observed"
                        ) from None
                    time.sleep(10)
        finally:
            run.finish(exit_code=exit_code)
