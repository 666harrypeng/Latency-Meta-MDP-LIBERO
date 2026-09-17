"""Train-only policy views over admitted continuous conveyor sources."""

import json
from pathlib import Path

from latency_meta_mdp.data.conveyor.corpus import split_sources
from latency_meta_mdp.data.lerobot_conversion import write_lerobot_frame_dataset
from latency_meta_mdp.io.artifacts import sha256_file


def train_sources(corpus_manifest):
    return split_sources(corpus_manifest, split="train")


def export_policy(corpus_manifest, output_dir, *, repo_id, dataset_factory=None):
    path = Path(corpus_manifest).resolve()
    corpus = json.loads(path.read_text())
    first = next(train_sources(path), None)
    if first is None:
        raise ValueError("no train episodes available")
    instruction = first[1].manifest["instruction"]
    del first

    def streams():
        for row, source in train_sources(path):
            n = len(source.actions)
            metadata = dict(
                episode_id=source.manifest["episode_id"],
                group_id=row["group_id"],
                seed=row["seed"],
                split="train",
                frame_count=n,
                valid_action_chunk_source_count=n,
                first_source_time_us=0,
                last_source_time_us=(n - 1) * 20000,
            )

            def frames(episode=source):
                for h, action in enumerate(episode.actions):
                    image, wrist = episode.rgb(h)
                    yield dict(
                        image=image,
                        wrist_image=wrist,
                        state=episode.states[h],
                        actions=action,
                        task=episode.manifest["instruction"],
                    )

            yield metadata, frames()
            print(f"[conveyor export] seed={row['seed']} frames={n}", flush=True)

    return write_lerobot_frame_dataset(
        episodes=streams(),
        output_dir=Path(output_dir),
        repo_id=repo_id,
        header=dict(
            task_id="conveyor_sort",
            variant="surface",
            instruction=instruction,
            source_format_id=corpus["format_id"],
            purpose=corpus["purpose"],
            split="train",
            corpus_identity=corpus["identity"],
            source_manifest_sha256=sha256_file(path),
            action_contract=corpus["action_contract"],
            state_contract="joint_qpos_qvel_gripper_width_velocity",
            state_dim=16,
            drop_n_last_frames=0,
            action_target_contract="masked_h50_real_actions_v1",
        ),
        image_shape=(256, 256, 3),
        dataset_factory=dataset_factory,
    )
