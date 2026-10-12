"""Actual rollout metrics and local video index, mirrored to a dedicated W&B run."""

import json
import os
import uuid
from collections import defaultdict
from pathlib import Path


def rollout_metrics(episode, report, *, phase, completed):
    if phase in ("validation", "monitor") and episode["epsilon"] != 0:
        raise ValueError("evaluation must be greedy")
    prefix = (
        "eval"
        if phase in ("validation", "monitor")
        else ("collection" if phase == "online" else "diagnostic")
    )
    group = f"{prefix}/{episode['behavior']}/{episode['regime']}"
    parcels = report["parcels"]
    result = {
        f"{prefix}/episode": completed,
        f"{group}/task_return": report["task_reward"],
        f"{group}/successes": parcels["successes"],
        f"{group}/misses": parcels["misses"],
        f"{group}/timeouts": parcels["timeouts"],
        f"{group}/epsilon": episode["epsilon"],
        f"{group}/logical_seconds": report["executed_steps"] / 50,
        f"{group}/starvation_ticks": report["starvation_ticks"],
    }
    for component, seconds in report.get("components", {}).items():
        result[f"{group}/seconds/{component}"] = seconds
    for component, count in report.get("component_calls", {}).items():
        result[f"{group}/calls/{component}"] = count
    return result


def summarize_rollouts(rows):
    groups = defaultdict(list)
    for row in rows:
        ep = row["episode"]
        groups[f"{ep['behavior']}/{ep['regime']}"].append(row["report"])
    result = {}
    for key, reports in groups.items():
        successes = sum(r["parcels"]["successes"] for r in reports)
        spawned = sum(r["parcels"]["spawned"] for r in reports)
        result[key] = dict(
            episodes=len(reports),
            successes=successes,
            spawned=spawned,
            success_rate=successes / spawned,
            mean_reward=sum(r["task_reward"] for r in reports) / len(reports),
            zero_delivery=sum(r["parcels"]["successes"] == 0 for r in reports),
            all_delivered=sum(r["parcels"]["successes"] == 10 for r in reports),
        )
    return result


class CycleTracker:
    def __init__(self, directory, *, name, enabled=True):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "cohort.json").read_text())
        self.phase = self.manifest["binding"]["phase"]
        self.rows = []
        self.run = None
        self.enabled = enabled
        if enabled:
            import wandb

            state_path = self.directory / "wandb-run.json"
            if state_path.exists():
                identity = json.loads(state_path.read_text())
            else:
                identity = dict(id=uuid.uuid4().hex[:8])
                state_path.write_text(json.dumps(identity))
            from latency_meta_mdp.meta.tracking_transport import start_run

            self.run = start_run(
                self.directory,
                entity=os.environ.get("WANDB_ENTITY"),
                project=os.environ.get("WANDB_PROJECT", "latency-meta-mdp-robosuite"),
                id=identity["id"],
                resume="allow",
                name=name,
                group="conveyor-meta-bound-fqi-cycle-v1",
                job_type=self.phase,
                tags=["conveyor", "meta", "fqi", "bound", "lambda0"],
                config=self.manifest["binding"],
                dir=str(self.directory),
                settings=wandb.Settings(disable_git=True, save_code=False, console="off"),
            )
            identity["url"] = self.run.url
            state_path.write_text(json.dumps(identity, indent=2) + "\n")
            print("WANDB_RUN " + str(self.run.url), flush=True)
            prefix = (
                "eval"
                if self.phase in ("validation", "monitor")
                else ("collection" if self.phase == "online" else "diagnostic")
            )
            self.run.define_metric(f"{prefix}/episode")
            self.run.define_metric(f"{prefix}/*", step_metric=f"{prefix}/episode")
        # Backfill completed episodes from their original immutable files. Step is
        # manifest order; collection is ordered and only one writer may run.
        from latency_meta_mdp.meta.information.cohort import verify_complete

        for ep in self.manifest["episodes"]:
            attempts = list((self.directory / "episodes" / ep["episode_id"]).glob("attempt-*"))
            valid = [
                a
                for a in attempts
                if verify_complete(a, binding=dict(cohort=self.manifest["binding"], episode=ep))
            ]
            if len(valid) > 1:
                raise ValueError("duplicate completed rollout")
            if valid:
                self(ep, json.loads((valid[0] / "summary.json").read_text()), valid[0], {})

    def __call__(self, episode, report, attempt, progress):
        row = dict(episode=episode, report=report, video=str((attempt / "smoke.mp4").resolve()))
        self.rows.append(row)
        values = rollout_metrics(episode, report, phase=self.phase, completed=len(self.rows))
        data = dict(
            completed=len(self.rows),
            expected=len(self.manifest["episodes"]),
            groups=summarize_rollouts(self.rows),
            episodes=self.rows,
        )
        temporary = self.directory / "rollout-summary.tmp"
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        temporary.replace(self.directory / "rollout-summary.json")
        lines = [
            "# Complete rollout videos",
            "",
            "| Scene | Delay | Policy | Delivered | Video |",
            "|---|---|---|---|---|",
        ]
        for r in self.rows:
            e, p = r["episode"], r["report"]["parcels"]
            lines.append(
                f"| {e['scene']} | {e['regime']} | {e['behavior']} | "
                f"{p['successes']}/{p['spawned']} | [Video]({r['video']}) |"
            )
        (self.directory / "videos.md").write_text("\n".join(lines) + "\n")
        if self.run is not None:
            # Explicit monotonic steps avoid duplicating backfilled points.
            step = len(self.rows)
            if step >= self.run.step:
                self.run.log(values, step=step)
            self.run.summary.update(
                {
                    "rollouts/completed": step,
                    "rollouts/expected": data["expected"],
                    "rollouts/groups": data["groups"],
                }
            )

    def close(self, *, success):
        complete = success and len(self.rows) == len(self.manifest["episodes"])
        if (
            complete
            and "shard" not in self.manifest
            and self.phase in ("validation", "monitor")
            and self.rows
        ):
            from latency_meta_mdp.meta.information.evaluation import paired_summary

            groups = summarize_rollouts(self.rows)
            metrics, paired = {}, {}
            for behavior in sorted({r["episode"]["behavior"] for r in self.rows}):
                metrics[f"{behavior}/mean_task_return"] = (
                    sum(
                        groups[f"{behavior}/{regime}"]["mean_reward"]
                        for regime in ("nominal", "family")
                    )
                    / 2
                )
                if behavior != "fqi":
                    paired[behavior] = paired_summary(self.rows, reference=behavior)
            if self.phase == "monitor":
                from latency_meta_mdp.meta.information.validation import validation_metrics

                separate = validation_metrics(
                    self.rows, scenes=sorted({r["episode"]["scene"] for r in self.rows})
                )
                metrics.update({f"fqi/{k}": v for k, v in separate.items()})
            result = dict(metrics=metrics, paired=paired)
            (self.directory / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
            if self.run is not None:
                import wandb

                checkpoint = Path(self.manifest["binding"]["policy"]["checkpoint"])
                binding = json.loads((checkpoint / "complete.json").read_text())["binding"]
                self.run.define_metric("learning/online_episodes")
                self.run.define_metric("evaluation/*", step_metric="learning/online_episodes")
                self.run.log(
                    {
                        "learning/online_episodes": binding.get("online_training_episodes", 0),
                        **{f"evaluation/{k}": v for k, v in metrics.items()},
                    }
                )
                self.run.summary["evaluation/paired"] = paired
                first_scene = min(r["episode"]["scene"] for r in self.rows)
                videos = {
                    f"videos/{r['episode']['behavior']}": wandb.Video(r["video"], format="mp4")
                    for r in self.rows
                    if r["episode"]["scene"] == first_scene and r["episode"]["regime"] == "nominal"
                }
                self.run.log(videos)
        if self.run is not None:
            self.run.summary["rollouts/complete"] = complete
            self.run.finish(exit_code=0 if success else 1)
