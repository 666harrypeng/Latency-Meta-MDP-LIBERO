"""Package a prepared task dataset and generate a concrete local clean-SFT job."""

import argparse
import errno
import json
import os
import shutil
from pathlib import Path

import yaml

from latency_meta_mdp.data.bundles import verify_bundle_files
from latency_meta_mdp.io.artifacts import sha256_file


def _inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("bundle input path escapes its root")
    return path


def _copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".parquet":
        try:
            os.link(source, target)
            return
        except OSError as error:
            if error.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES}:
                raise
    shutil.copy2(source, target)


def package_task_bundle(preparation_root, output_dir):
    root, output = Path(preparation_root).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    prep = json.loads((root / "preparation.json").read_text())
    dataset_path = _inside(root, prep["dataset_manifest"])
    data = json.loads(dataset_path.read_text())
    if sha256_file(dataset_path) != prep["dataset_manifest_sha256"]:
        raise ValueError("prepared dataset manifest changed")
    if (
        prep["status"] != "completed"
        or data["split"] != "train"
        or prep["task_id"] != data["task_id"]
        or prep["action_contract"] != data["action_contract"]
        or prep["source_count"] != data["frame_count"]
        or prep["source_manifest_sha256"] != data["source_manifest_sha256"]
        or data["purpose"] not in {"development_smoke", "training_source"}
        or prep["purpose"] != data["purpose"]
    ):
        raise ValueError("preparation and dataset identities disagree")
    norm = _inside(root, prep["norm_stats"])
    if sha256_file(norm) != prep["norm_stats_sha256"]:
        raise ValueError("prepared normalization changed")
    for name, size in data["file_sizes"].items():
        file = _inside(dataset_path.parent, name)
        if not file.is_file() or file.stat().st_size != size:
            raise ValueError(f"missing or truncated dataset file: {name}")
    staged = output.parent / f".{output.name}.building-{os.getpid()}"
    staged.mkdir(parents=True, exist_ok=False)
    try:
        dataset_relative = Path("datasets") / data["repo_id"] / "metamdp_dataset.json"
        destination = _inside(staged, str(dataset_relative)).parent
        for name in data["file_sizes"]:
            _copy(_inside(dataset_path.parent, name), _inside(destination, name))
        _copy(dataset_path, destination / "metamdp_dataset.json")
        prep_dir = staged / "preparation"
        _copy(norm, prep_dir / "assets" / data["repo_id"] / "norm_stats.json")
        portable = {
            **prep,
            "dataset_manifest": str(dataset_relative),
            "norm_stats": f"assets/{data['repo_id']}/norm_stats.json",
        }
        (prep_dir / "preparation.json").write_text(json.dumps(portable, indent=2))
        manifest = dict(
            schema_version=2,
            format_id="task_pi05_training_bundle_v2",
            task_id=data["task_id"],
            variant=data["variant"],
            action_contract=data["action_contract"],
            purpose=data["purpose"],
            source_count=data["frame_count"],
            repo_id=data["repo_id"],
            dataset_root="datasets",
            dataset_manifest=str(dataset_relative),
            preparation_root="preparation",
            dataset_manifest_sha256=sha256_file(destination / "metamdp_dataset.json"),
            preparation_sha256=sha256_file(prep_dir / "preparation.json"),
            files={
                str(p.relative_to(staged)): p.stat().st_size
                for p in sorted(staged.rglob("*"))
                if p.is_file()
            },
        )
        (staged / "bundle.json").write_text(json.dumps(manifest, indent=2))
        verify_bundle_files(staged)
        staged.rename(output)
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    return output / "bundle.json"


def write_local_training_job(
    bundle_manifest, output, *, run_name="conveyor-clean", device_count=8, batch_size=256
):
    bundle_manifest, output = Path(bundle_manifest).resolve(), Path(output).resolve()
    manifest = verify_bundle_files(bundle_manifest.parent)
    if manifest["schema_version"] != 2:
        raise ValueError("task job generation requires a task bundle")
    job = dict(
        schema_version=2,
        task_id=manifest["task_id"],
        variant=manifest["variant"],
        action_contract=manifest["action_contract"],
        profile="configs/contracts/policy/pi05_state16_h50.yaml",
        training="configs/training/policy/conveyor_clean.yaml",
        dataset=dict(
            kind="local",
            path=os.path.relpath(bundle_manifest.parent, output.parent),
            manifest_sha256=sha256_file(bundle_manifest),
        ),
        run_name=run_name,
        device_count=device_count,
        batch_size=batch_size,
        publish_repo=None,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        yaml.safe_dump(job, stream, sort_keys=False)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--job-output", type=Path, required=True)
    parser.add_argument("--run-name", default="conveyor-clean")
    args = parser.parse_args()
    if args.job_output.exists():
        raise FileExistsError(args.job_output)
    path = package_task_bundle(args.preparation_dir, args.output_dir)
    print(write_local_training_job(path, args.job_output, run_name=args.run_name), flush=True)
