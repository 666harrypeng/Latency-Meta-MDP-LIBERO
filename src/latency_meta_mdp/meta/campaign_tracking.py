"""One console and resumable W&B history around an existing FQI controller.

Only the controller writes training artifacts. This CPU-only supervisor reads
durable stage records, owns telemetry-main, and never initializes a simulator.
"""

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
import uuid
from pathlib import Path


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_events(path):
    data = path.read_bytes() if path.exists() else b""
    return [json.loads(line) for line in data[: data.rfind(b"\n") + 1].splitlines()]


def rollout_status(progress):
    seconds = progress.get("wall_seconds", 0)
    workers = len(progress.get("per_worker", []))
    return dict(
        rollout_workers=workers,
        rollout_active_workers=sum(
            code is None for code in progress.get("exit_codes", [None] * workers)
        ),
        rollout_wall_seconds=seconds,
        rollout_episodes_per_hour=3600 * progress.get("completed", 0) / seconds if seconds else 0,
    )


def snapshot(root):
    root = Path(root)
    campaign = root / "campaign" if (root / "campaign").is_dir() else root
    records = []
    status = dict(phase="preparing", total_updates=0, total_episodes=0, cycle=0)
    updates = iterations = episodes = 0
    startup = campaign / "startup"
    folders = ([startup] if startup.is_dir() else []) + sorted(campaign.glob("cycle-[0-9]*"))
    for folder in folders:
        is_startup = folder == startup
        cycle = 0 if is_startup else int(folder.name.split("-")[-1])
        fit_folder = folder if is_startup else folder / "fit"
        status = dict(cycle=cycle, phase="collection")
        summary = read_json(folder / "collection/rollout-summary.json", {})
        rows = summary.get("episodes", [])
        manifest = read_json(folder / "collection/cohort.json", {})
        if "episodes" in manifest:
            # Workers may finish out of order; resume reconstructs manifest order.
            # Publish the completed prefix so neither order can rewrite a plotted point.
            by_id = {row["episode"]["episode_id"]: row for row in rows}
            rows = []
            for episode in manifest["episodes"]:
                if episode["episode_id"] not in by_id:
                    break
                rows.append(by_id[episode["episode_id"]])
        for i, row in enumerate(rows, 1):
            ep, report = row["episode"], row["report"]
            p = report["parcels"]
            prefix = f"collection/{ep['regime']}/{ep['behavior']}"
            values = {
                "collection/episode": episodes + i,
                "collection/cycle": cycle,
                f"{prefix}/delivery_ratio": p["successes"] / p["spawned"],
            }
            budget = report.get("budget", {})
            for label, key in (
                ("budget_return", "episode_reward"),
                ("compute_cost", "compute_cost"),
                ("drops", "drops"),
            ):
                if key in budget:
                    values[f"{prefix}/{label}"] = budget[key]
            values.update(
                {f"{prefix}/calls/{k}": v for k, v in budget.get("module_calls", {}).items()}
            )
            records.append(dict(id=f"{folder.name}/collection/{ep['episode_id']}", metrics=values))
        episodes += len(summary.get("episodes", []))
        progress = read_json(folder / "collection/parallel-progress.json", {})
        status.update(rollout_status(progress))
        status.update(
            collection_completed=progress.get("completed", summary.get("completed", 0)),
            collection_expected=progress.get(
                "expected", summary.get("expected", 0 if is_startup else 24)
            ),
        )
        local_updates = local_iterations = 0
        preparation = read_json(fit_folder / "preparation-progress.json", {})
        if preparation:
            status.update(phase=preparation["phase"])
            if "expected" in preparation:
                status.update(
                    replay_completed=preparation["episodes"],
                    replay_expected=preparation["expected"],
                )
        for item in read_events(fit_folder / "events.jsonl"):
            event = item["event"]
            phase, iteration = event["phase"], event["iteration"]
            local_updates = max(local_updates, event.get("actual_update", 0))
            local_iterations = max(local_iterations, iteration)
            status.update(phase=phase, iteration=iteration)
            values = {}
            if phase == "fit":
                values = {
                    "train/total_updates": updates + event["actual_update"],
                    "train/cycle": cycle,
                    "train/local_iteration": iteration,
                    "train/attempt": event.get("attempt", 1),
                }
                for key in (
                    "loss",
                    "grad_norm",
                    "td_loss",
                    "conservative_loss",
                    "weighted_conservative_loss",
                    "learning_rate",
                    "input_seconds",
                    "update_seconds",
                    "updates_per_second",
                    "global_batch",
                    "microbatch",
                    "accumulation",
                    "compile_model",
                ):
                    if key in event:
                        values[f"train/{key}"] = event[key]
                        status[key] = event[key]
                status.update(
                    loss=event["loss"],
                    iteration_update=event.get("optimizer_update"),
                    iteration_updates=event.get("optimizer_updates"),
                )
            elif phase == "iteration_complete":
                values = {"fqi/global_iteration": iterations + iteration}
                values.update(
                    {
                        f"fqi/{k}": v
                        for k, v in event["metrics"].items()
                        if isinstance(v, int | float)
                    }
                )
                for group, metrics in event["metrics"].get("diagnostics", {}).items():
                    values.update({f"fqi/diagnostics/{group}/{k}": v for k, v in metrics.items()})
                sampling = event["metrics"].get("sampling", {})
                for key, value in sampling.items():
                    if isinstance(value, int | float):
                        values[f"fqi/sampling/{key}"] = value
                for group, metrics in sampling.get("groups", {}).items():
                    values.update({f"fqi/sampling/{group}/{k}": v for k, v in metrics.items()})
            elif phase == "validation_complete":
                values = {
                    "validation/cycle": cycle,
                    "validation/total_updates": updates + local_updates,
                    "validation/global_iteration": iterations + iteration,
                }
                for key, value in event["metrics"].items():
                    label = (
                        key.removesuffix("/reward") + "/delivery_ratio"
                        if key.endswith("/reward")
                        else key
                    )
                    values[f"validation/{label}"] = value
            elif phase in ("target_progress", "targets"):
                status.update(
                    target_completed=event.get(
                        "completed_decisions", event.get("unique_decisions", 0)
                    ),
                    target_total=event.get("unique_decisions", 0),
                )
            if values:
                records.append(dict(id=f"{folder.name}/fit/{item['sequence']}", metrics=values))
        updates += local_updates
        iterations += local_iterations
        validation = sorted((fit_folder / "validation").glob("iteration-*/parallel-progress.json"))
        if (
            validation
            and status["phase"] != "validation_complete"
            and int(validation[-1].parent.name.split("-")[-1]) == local_iterations
        ):
            p = read_json(validation[-1])
            status.update(rollout_status(p))
            status.update(
                phase="validation",
                validation_completed=p["completed"],
                validation_expected=p["expected"],
            )
    status.update(total_updates=updates, total_episodes=episodes)
    digest = hashlib.sha256()
    for row in records:
        digest.update(json.dumps(row, sort_keys=True, allow_nan=False).encode())
        row["digest"] = digest.hexdigest()
    return records, status


def pending_records(records, next_step, last_digest):
    if next_step > len(records):
        raise ValueError("source history is shorter than mirrored history")
    if next_step and records[next_step - 1]["digest"] != last_digest:
        raise ValueError("mirrored source prefix changed")
    yield from enumerate(records[next_step:], start=next_step)


def console_line(status):
    fields = [f"cycle={status.get('cycle', 0):03d}", f"phase={status['phase']}"]
    if status["phase"] in ("collection", "validation") and status.get("rollout_workers"):
        fields.extend(
            [
                f"workers={status.get('rollout_active_workers', status['rollout_workers'])}"
                f"/{status['rollout_workers']}",
                f"episodes/h={status['rollout_episodes_per_hour']:.1f}",
                f"stage={status['rollout_wall_seconds'] / 60:.1f}m",
            ]
        )
    for done, total, label in (
        ("collection_completed", "collection_expected", "collection"),
        ("target_completed", "target_total", "targets"),
        ("validation_completed", "validation_expected", "validation"),
        ("replay_completed", "replay_expected", "replay"),
    ):
        if done in status:
            fields.append(f"{label}={status[done]}/{status[total]}")
    if "iteration" in status:
        fields.append(f"iteration={status['iteration']}")
    if status["phase"] == "fit" and status.get("iteration_update") is not None:
        fields.append(f"step={status['iteration_update']}/{status.get('iteration_updates') or '?'}")
    fields.append(f"updates={status.get('total_updates', 0)}")
    if status["phase"] == "fit" and "loss" in status:
        fields.append(f"loss={status['loss']:.5f}")
    if status["phase"] == "fit":
        for key, label in (
            ("td_loss", "td"),
            ("weighted_conservative_loss", "cql"),
            ("grad_norm", "grad_norm"),
        ):
            if key in status:
                fields.append(f"{label}={status[key]:.5f}")
        if "learning_rate" in status:
            fields.append(f"lr={status['learning_rate']:.4g}")
        if "microbatch" in status:
            fields.append(
                f"batch={status['global_batch']} micro={status['microbatch']} "
                f"accum={status['accumulation']}"
            )
            fields.append(f"compiled={status['compile_model']}")
        if "updates_per_second" in status:
            rate = status["updates_per_second"]
            fields.append(f"updates/s={rate:.2f}")
            if status.get("iteration_updates") and rate > 0:
                remaining = status["iteration_updates"] - status["iteration_update"]
                fields.append(f"fit_eta={remaining / rate:.0f}s")
        if "input_seconds" in status:
            fields.append(f"input={1000 * status['input_seconds']:.1f}ms")
    if "prepacked" in status:
        fields.append(f"historical_prepack={status['prepacked']}/{status['prepack_total']}")
    fields.append(f"elapsed={status.get('elapsed_seconds', 0) / 60:.1f}m")
    return time.strftime("[%Y-%m-%d %H:%M:%S UTC]", time.gmtime()) + " " + " | ".join(fields)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--operations", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--checkpoint-repo")
    parser.add_argument("--command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.poll_seconds <= 0 or (args.once and args.command):
        parser.error("positive poll interval required; --once cannot launch training")
    if args.checkpoint_repo and not args.recipe:
        parser.error("--checkpoint-repo requires --recipe")
    # Keep the operator's stable home alias while the physical project moves there.
    root, operations = args.root.absolute(), args.operations.absolute()
    output = root / "telemetry-main"
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "writer.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    binding = dict(
        project=args.project,
        experiment=root.name,
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
    )
    if args.recipe:
        binding["recipe_sha256"] = hashlib.sha256(args.recipe.read_bytes()).hexdigest()
        binding["recipe"] = read_json(args.recipe)
        binding["science_config"] = __import__("yaml").safe_load(args.config.read_text())
        binding["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    identity_path = output / "wandb-run.json"
    identity = read_json(identity_path, dict(id=uuid.uuid4().hex[:8], binding=binding))
    if identity["binding"] != binding:
        raise ValueError("campaign identity/config changed")
    atomic_json(identity_path, identity)
    import wandb

    run = None
    next_step, last_digest = 0, None

    def connect():
        nonlocal run, next_step, last_digest
        run = wandb.init(
            project=args.project,
            entity=os.environ.get("WANDB_ENTITY"),
            id=identity["id"],
            resume="must" if identity.get("url") else "allow",
            name=args.name,
            job_type="campaign",
            tags=["main", "budget", "fqi", "attention"],
            dir=str(output),
            config=binding,
            settings=wandb.Settings(
                disable_git=True,
                save_code=False,
                console="wrap",
                # Device telemetry uses W&B's separate system stream, so it cannot
                # advance the append-only training-event cursor used on resume.
                x_disable_stats=False,
                x_stats_sampling_interval=5,
                x_stats_disk_paths=(str(root),),
                init_timeout=30,
            ),
        )
        for prefix, axis in (
            ("train", "total_updates"),
            ("fqi", "global_iteration"),
            ("collection", "episode"),
            ("validation", "total_updates"),
        ):
            run.define_metric(f"{prefix}/{axis}")
            run.define_metric(f"{prefix}/*", step_metric=f"{prefix}/{axis}", step_sync=False)
        run.define_metric("mirror/*", hidden=True)
        identity["url"] = run.url
        atomic_json(identity_path, identity)
        next_step, last_digest = run.step, run.summary.get("mirror/digest")
        print("WANDB_MAIN " + run.url, flush=True)

    publisher = None
    if args.checkpoint_repo:
        from latency_meta_mdp.io.fqi_publication import BackgroundPublisher
        publisher = BackgroundPublisher(root, args.config, args.recipe,
                                        repo_id=args.checkpoint_repo, run_id=identity["id"])
    table_digest = None
    child = None
    log = None
    if args.command:
        log = (operations / "continuation-console.log").open("a")
        child = subprocess.Popen(
            args.command,
            env=dict(os.environ, CONVEYOR_STAGE_WANDB="0"),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    started = time.monotonic()
    last_connect = -300.0
    exit_code = 0
    draining = False
    try:
        while not stopping:
            status = dict(phase="reading_records", total_updates=0)
            try:
                records, status = snapshot(root)
                state = read_json(operations / "continuation-status.json", {})
                prepack = read_json(operations / "prepack-progress.json", {})
                if prepack:
                    status.update(
                        prepacked=len(prepack["completed"]),
                        prepack_total=len(prepack["completed"]) + prepack["remaining"],
                    )
                if state.get("phase") in ("failed", "completed"):
                    status["phase"] = state["phase"]
                elif (
                    status.get("collection_completed") == status.get("collection_expected")
                    and not status["total_updates"]
                    and status["phase"] == "collection"
                ):
                    status["phase"] = state.get("phase", "preparing_replay")
                if run is None and time.monotonic() - last_connect > 300:
                    last_connect = time.monotonic()
                    connect()
                if run is not None:
                    for step, row in pending_records(records, next_step, last_digest):
                        run.log(
                            row["metrics"]
                            | {"mirror/event_id": row["id"], "mirror/digest": row["digest"]},
                            step=step,
                            commit=True,
                        )
                        next_step, last_digest = step + 1, row["digest"]
                    run.summary.update({f"progress/{k}": v for k, v in status.items()})
                status.update(
                    url=identity.get("url"),
                    mirrored_records=next_step,
                    elapsed_seconds=time.monotonic() - started,
                    checked_at=time.time(),
                )
                atomic_json(output / "status.json", status)
            except Exception as error:
                # Telemetry faults never stop training. Do not print SDK credential-bearing errors.
                status["telemetry_error"] = type(error).__name__
                print(
                    f"TELEMETRY_RETRY {type(error).__name__}; "
                    "training continues; local logs retained",
                    flush=True,
                )
            publishing = False
            if publisher:
                publishing = publisher.poll(wandb_url=identity.get("url"),
                                            final=args.once or draining)
                published = publisher.status
                print(f"CHECKPOINT_SYNC uploaded={published.get('uploaded', 0)} "
                      f"pending={published.get('pending', 'scanning')} "
                      f"index_pending={published.get('index_pending', True)} "
                      f"error={published.get('error')}", flush=True)
                if run is not None and published:
                    try:
                        run.summary.update({f"checkpoints/{k}": published.get(k)
                                            for k in ("uploaded", "pending", "index_pending",
                                                      "index_url", "error")})
                        rows = published.get("checkpoints", [])
                        encoded = json.dumps(rows, sort_keys=True).encode()
                        digest = hashlib.sha256(encoded).hexdigest()
                        if digest != table_digest:
                            run.summary["checkpoints/table"] = wandb.Table(
                                columns=["checkpoint", "cycle", "iteration", "total_updates",
                                         "ancestor_updates", "HF", "validation"],
                                data=[[r["checkpoint"], r["cycle"], r["iteration"],
                                       r["total_updates"], r["ancestor_updates"], r.get("hf_url"),
                                       json.dumps(r["validation"])] for r in rows])
                            table_digest = digest
                    except Exception as error:
                        print(f"CHECKPOINT_TABLE_RETRY {type(error).__name__}", flush=True)
            print(console_line(status), flush=True)
            if args.once or draining:
                if publishing:
                    time.sleep(min(5, args.poll_seconds))
                    continue
                exit_code = 0 if child is None else child.returncode
                if publisher and (publisher.status.get("error") or
                                  publisher.status.get("pending") or
                                  publisher.status.get("index_pending")):
                    exit_code = exit_code or 75
                break
            if child is not None and child.poll() is not None:
                # Read once more after exit so final checkpoint/validation events are included.
                draining = True
                continue
            deadline = time.monotonic() + args.poll_seconds
            while not stopping and time.monotonic() < deadline:
                time.sleep(min(1, max(0, deadline - time.monotonic())))
    finally:
        if child is not None and child.poll() is None and stopping:
            # This process group contains only the controller launched here and its children.
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            exit_code = 130
        if publisher is not None:
            publisher.close()
        if run is not None:
            run.finish(exit_code=exit_code)
        if log is not None:
            log.close()
        lock.close()
    print(
        f"EXPERIMENT_EXIT code={exit_code}; detailed log={operations / 'continuation-console.log'}",
        flush=True,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
