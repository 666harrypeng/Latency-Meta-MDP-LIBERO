"""Export a conveyor train split and validate a native OpenPI data batch."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from latency_meta_mdp.data.conveyor.policy import export_policy, train_sources
from latency_meta_mdp.io.artifacts import sha256_file
from latency_meta_mdp.io.paths import repository_root
from latency_meta_mdp.policy.openpi.source import temporary_patched_openpi_copy
from latency_meta_mdp.policy.profile import load_sft_profile


def prepare(corpus_manifest, output_dir, *, repo_id="metamdp/conveyor_sort_smoke", resume=False):
    corpus_manifest, output = Path(corpus_manifest).resolve(), Path(output_dir).resolve()
    if len(Path(repo_id).parts) != 2 or Path(repo_id).is_absolute() or ".." in Path(repo_id).parts:
        raise ValueError("local LeRobot repo_id must be owner/name")
    if output.exists() and not resume:
        raise FileExistsError("preparation exists; use --resume")
    output.mkdir(parents=True, exist_ok=True)
    dataset_home = output / "datasets"
    os.environ["HF_LEROBOT_HOME"] = str(dataset_home)
    dataset_manifest = dataset_home / repo_id / "metamdp_dataset.json"
    corpus_hash = sha256_file(corpus_manifest)
    if not dataset_manifest.exists():
        export_policy(corpus_manifest, dataset_home / repo_id, repo_id=repo_id)
    exported = json.loads(dataset_manifest.read_text())
    if exported["source_manifest_sha256"] != corpus_hash or exported["split"] != "train":
        raise ValueError("existing export differs from the train corpus")
    for name, size in exported["file_sizes"].items():
        if (dataset_manifest.parent / name).stat().st_size != size:
            raise ValueError("exported file size changed")

    root = repository_root()
    profile = load_sft_profile(root / "configs/contracts/policy/pi05_state16_h50.yaml")
    patches = tuple(
        root / "patches/openpi" / name
        for name in (
            "0001-filter-incomplete-action-chunks.patch",
            "0002-save-completed-step-checkpoints.patch",
            "0003-mask-action-tails.patch",
        )
    )
    with temporary_patched_openpi_copy(
        openpi_root=root / "third_party/openpi",
        patch_paths=patches,
        expected_revision=profile.openpi_revision,
    ) as copied:
        sys.path.insert(0, str(copied / "src"))
        try:
            import jax
            from openpi.shared import normalize
            from openpi.training.data_loader import create_torch_data_loader, create_torch_dataset

            from latency_meta_mdp.policy.openpi.training import build_task_train_config

            state_stats, action_stats = normalize.RunningStats(), normalize.RunningStats()
            count, train_seeds = 0, []
            for row, source in train_sources(corpus_manifest):
                # One contribution per real source/control, without terminal state or H50 padding.
                state_stats.update(source.states[:-1].astype(np.float64))
                action_stats.update(source.actions.astype(np.float64))
                count += len(source.actions)
                train_seeds.append(row["seed"])
            if count != exported["frame_count"]:
                raise ValueError("normalization source count differs from exported frames")
            assets = output / "assets"
            normalize.save(
                assets / repo_id,
                {"state": state_stats.get_statistics(), "actions": action_stats.get_statistics()},
            )
            config = build_task_train_config(
                profile,
                config_name="conveyor_sort_clean_smoke",
                repo_id=repo_id,
                task_id="conveyor_sort",
                action_contract_id=exported["action_contract"],
                extra_metadata={"variant": "surface"},
            )
            data = config.data.create(assets, config.model)
            raw = create_torch_dataset(data, 50, config.model)
            if len(raw) != count or data.norm_stats["state"].mean.shape != (16,):
                raise ValueError("native source length/state normalization mismatch")
            cursor, probes = 0, 0
            for episode_index, (row, source) in enumerate(train_sources(corpus_manifest)):
                n = len(source.actions)
                boundaries = {0, max(0, n - 50), n - 1}
                boundaries.update(max(0, int(t) - 5) for t in np.flatnonzero(source.success_delta))
                for h in sorted(boundaries):
                    sample = raw[cursor + h]
                    expected = source.policy_sample(h)
                    np.testing.assert_array_equal(
                        np.asarray(sample["actions_is_pad"]), expected["actions_is_pad"]
                    )
                    mask = ~expected["actions_is_pad"]
                    np.testing.assert_allclose(
                        np.asarray(sample["actions"])[mask], expected["actions"][mask], atol=1e-7
                    )
                    np.testing.assert_allclose(
                        np.asarray(sample["state"]), source.states[h], atol=1e-7
                    )
                    if (
                        int(sample["frame_index"]) != h
                        or int(sample["episode_index"]) != episode_index
                    ):
                        raise ValueError("native frame/episode indices are misaligned")
                    for key, expected_rgb in zip(("image", "wrist_image"), source.rgb(h)):
                        actual = np.asarray(sample[key])
                        if actual.shape[0] == 3:
                            actual = actual.transpose(1, 2, 0)
                        if np.issubdtype(actual.dtype, np.floating):
                            actual = np.rint(actual * 255).astype(np.uint8)
                        np.testing.assert_array_equal(actual, expected_rgb)
                    probes += 1
                cursor += n
            loader = create_torch_data_loader(
                data,
                config.model,
                50,
                2,
                num_batches=1,
                num_workers=0,
                sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0]),
            )
            observation, actions = next(iter(loader))
            mask = np.asarray(observation.action_loss_mask)
            if mask.shape != (2, 50, 32) or not mask[:, 0, :7].all() or mask[..., 7:].any():
                raise ValueError("native batch action mask mismatch")
            report = dict(
                status="completed",
                scope="dataset_preparation_and_batch_check_no_training",
                task_id="conveyor_sort",
                purpose=exported["purpose"],
                action_contract=exported["action_contract"],
                corpus_identity=exported["corpus_identity"],
                source_manifest_sha256=corpus_hash,
                dataset_manifest=str(dataset_manifest.relative_to(output)),
                dataset_manifest_sha256=sha256_file(dataset_manifest),
                norm_stats=str((assets / repo_id / "norm_stats.json").relative_to(output)),
                norm_stats_sha256=sha256_file(assets / repo_id / "norm_stats.json"),
                train_seeds=train_seeds,
                source_count=count,
                episodes=exported["episode_count"],
                boundary_probes=probes,
                state_batch_shape=list(observation.state.shape),
                action_batch_shape=list(actions.shape),
                action_mask_shape=list(mask.shape),
                token_shape=list(observation.tokenized_prompt.shape),
                policy_metadata=config.policy_metadata,
                profile_sha256=sha256_file(root / "configs/contracts/policy/pi05_state16_h50.yaml"),
                openpi_revision=profile.openpi_revision,
                patches={p.name: sha256_file(p) for p in patches},
                training_started=False,
            )
            (output / "preparation.json").write_text(json.dumps(report, indent=2))
        finally:
            sys.path.remove(str(copied / "src"))
    print(json.dumps(report), flush=True)
    return output / "preparation.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default="metamdp/conveyor_sort_smoke")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    prepare(args.corpus_manifest, args.output_dir, repo_id=args.repo_id, resume=args.resume)
