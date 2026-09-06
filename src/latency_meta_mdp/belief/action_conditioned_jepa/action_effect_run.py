"""Collect one compact formal-validation same-source action-effect bank."""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from safetensors.torch import save_file

from latency_meta_mdp.artifacts import sha256_file
from latency_meta_mdp.belief.action_conditioned_jepa.action_effect import (
    J4SourceContext,
    build_j4_control_branches,
    select_j4_source_contexts,
)
from latency_meta_mdp.belief.action_conditioned_jepa.config import (
    load_action_conditioned_jepa_config,
)
from latency_meta_mdp.belief.action_conditioned_jepa.data_adapter import (
    JepaEpisodeRecord,
    load_verified_jepa_inputs,
)
from latency_meta_mdp.belief.action_conditioned_jepa.temporal_signal_audit import (
    TemporalSignalEpisode,
    load_temporal_signal_episodes,
)
from latency_meta_mdp.expert_realization.artifacts import (
    _fsync_directory,
    _hash_file,
    _rename_noreplace,
    _write_file_fsynced,
)
from latency_meta_mdp.expert_realization.contracts import TaskInstanceId
from latency_meta_mdp.expert_realization.source_corpus.schema import SourceFieldRole
from latency_meta_mdp.expert_realization.task_instance import (
    MaterializedTaskInstance,
    _build_task_instance_runtime,
    materialize_task_instance,
)
from latency_meta_mdp.hf_dino_encoder import HfDinoPatchEncoder

_FORMAT_ID = "action_conditioned_jepa_l3_j4_bank_v1"
_BRANCHES = ("nominal", "hold", "scale_0.5", "prefix4_then_hold")
_ANCHORS = (4, 8, 12, 16, 20)
_CONTEXT_COUNT = 80
_REPLAY_MAX_ABS_TOLERANCE = 2e-8
_NOMINAL_FUTURE_MAX_ABS_TOLERANCE = 5e-7


def replay_is_numerically_equivalent(maximum_absolute_error: float) -> bool:
    return (
        type(maximum_absolute_error) is float
        and np.isfinite(maximum_absolute_error)
        and 0.0 <= maximum_absolute_error <= _REPLAY_MAX_ABS_TOLERANCE
    )


def nominal_future_is_numerically_equivalent(maximum_absolute_error: float) -> bool:
    return (
        type(maximum_absolute_error) is float
        and np.isfinite(maximum_absolute_error)
        and 0.0 <= maximum_absolute_error <= _NOMINAL_FUTURE_MAX_ABS_TOLERANCE
    )


def _paths(root: Path) -> dict[str, Path]:
    source_id = "panda-ball-structured-source-quota-formal-100x4-v1"
    return {
        "model": root / "configs/belief/action_conditioned_jepa/model.yaml",
        "level": root / "configs/belief/action_conditioned_jepa/l3.yaml",
        "temporal": (
            root / "configs/belief/action_conditioned_jepa/stride4_80ms_history_160ms.yaml"
        ),
        "source": root / "outputs/source_corpus" / source_id,
        "cache": (
            root
            / "outputs/derived/vision_features"
            / "dinov3-vits16-structured-source-100x4-v1/manifest.json"
        ),
        "split": (
            root
            / "outputs/derived/source_splits"
            / source_id
            / "train80-validation20-seed20260903-v1.json"
        ),
    }


def _physical_proprio(snapshot: Any, *, absorbing: bool) -> np.ndarray:
    value = np.concatenate(
        (
            np.asarray(snapshot.robot_qpos),
            np.asarray(snapshot.robot_qvel),
            np.asarray(
                [
                    snapshot.robot_gripper_qpos[0] - snapshot.robot_gripper_qpos[1],
                    snapshot.robot_gripper_qvel[0] - snapshot.robot_gripper_qvel[1],
                ]
            ),
        )
    ).astype(np.float32)
    if absorbing:
        value[7:14] = 0.0
        value[15] = 0.0
    return value


def _physical_object(snapshot: Any, *, absorbing: bool) -> np.ndarray:
    value = np.concatenate(
        (np.asarray(snapshot.object_qpos)[:3], np.asarray(snapshot.object_qvel)[:3])
    ).astype(np.float32)
    if absorbing:
        value[3:] = 0.0
    return value


def _source_replay_max_abs(
    *,
    snapshot: Any,
    reference: dict[str, np.ndarray],
    source_tick: int,
) -> float:
    pairs = (
        (snapshot.robot_qpos, reference["robot_qpos"][source_tick]),
        (snapshot.robot_qvel, reference["robot_qvel"][source_tick]),
        (snapshot.robot_gripper_qpos, reference["gripper_qpos"][source_tick]),
        (snapshot.robot_gripper_qvel, reference["gripper_qvel"][source_tick]),
        (snapshot.object_qpos, reference["object_pose"][source_tick]),
        (snapshot.object_qvel, reference["object_velocity"][source_tick]),
        (snapshot.eef_pos, reference["eef_position_world"][source_tick]),
    )
    return max(
        float(np.max(np.abs(np.asarray(observed) - np.asarray(expected))))
        for observed, expected in pairs
    )


def _load_replay_references(
    *,
    inputs: Any,
    episode_ids: tuple[str, ...],
) -> dict[str, dict[str, np.ndarray]]:
    fields = (
        "formal_tick",
        "robot_qpos",
        "robot_qvel",
        "gripper_qpos",
        "gripper_qvel",
        "object_pose",
        "object_velocity",
        "eef_position_world",
    )
    roles = frozenset(
        {
            SourceFieldRole.IDENTITY,
            SourceFieldRole.DEPLOYMENT_INPUT,
            SourceFieldRole.SUPERVISION_CANDIDATE,
        }
    )
    references = {}
    for episode_id in episode_ids:
        rows = inputs.source.read_fields(
            episode_id,
            fields=fields,
            allowed_roles=roles,
        ).to_pylist()
        if [row["formal_tick"] for row in rows] != list(range(len(rows))):
            raise ValueError("J4 replay reference is not boundary aligned")
        references[episode_id] = {
            name: np.asarray([row[name] for row in rows], dtype=np.float64) for name in fields[1:]
        }
    return references


def _materialize_tasks(
    *,
    project_root: Path,
    source_root: Path,
    episode_ids: tuple[str, ...],
) -> dict[str, MaterializedTaskInstance]:
    rows = pq.read_table(source_root / "meta/episodes.parquet").to_pylist()
    metadata = {row["episode_id"]: row for row in rows if row["episode_id"] in episode_ids}
    if set(metadata) != set(episode_ids):
        raise ValueError("J4 episode metadata join is incomplete")
    tasks = {}
    for episode_id in episode_ids:
        identity = TaskInstanceId.from_mapping(json.loads(metadata[episode_id]["task_instance_id"]))
        task = materialize_task_instance(
            project_root=project_root,
            level=3,
            task_instance_seed=identity.task_instance_seed,
        )
        if task.task_instance_id != identity:
            raise ValueError("J4 task materialization changed source identity")
        tasks[episode_id] = task
    return tasks


def _select_formal_episodes(inputs: Any) -> tuple[str, ...]:
    rows = pq.read_table(inputs.source.root / "meta/episodes.parquet").to_pylist()
    validation = set(inputs.split.validation_episode_ids)
    selected = tuple(
        row["episode_id"]
        for row in sorted(rows, key=lambda value: value["logical_master_task_index"])
        if row["level"] == 3 and row["accepted_slot"] == 0 and row["episode_id"] in validation
    )
    if len(selected) != 20:
        raise ValueError("J4 requires one L3 formal realization from each validation master")
    return selected


def _collect_branch(
    *,
    task: MaterializedTaskInstance,
    record: JepaEpisodeRecord,
    signal: TemporalSignalEpisode,
    replay_reference: dict[str, np.ndarray],
    context: J4SourceContext,
    controls: np.ndarray,
) -> dict[str, Any]:
    runtime = _build_task_instance_runtime(task)
    try:
        snapshot = runtime.executor.initialize()
        for tick in range(context.source_tick):
            snapshot = runtime.executor.step_formal(record.controls[tick])
        replay_error = _source_replay_max_abs(
            snapshot=snapshot,
            reference=replay_reference,
            source_tick=context.source_tick,
        )
        if not replay_is_numerically_equivalent(replay_error):
            raise RuntimeError(f"J4 source replay exceeds numerical tolerance: {replay_error}")
        future_proprio = []
        future_object = []
        images = []
        left_contact = []
        right_contact = []
        handoff_physical = []
        absorbing = []
        terminal_step = -1
        for step, action in enumerate(controls, start=1):
            if terminal_step < 0:
                snapshot = runtime.executor.step_formal(action)
                if runtime.tracker.status.value != "running":
                    terminal_step = step
            if step not in _ANCHORS:
                continue
            is_absorbing = terminal_step >= 0 and step > terminal_step
            contact = runtime.handoff.last_contact
            if contact is None:
                raise RuntimeError("J4 branch boundary lacks contact state")
            future_proprio.append(_physical_proprio(snapshot, absorbing=is_absorbing))
            future_object.append(_physical_object(snapshot, absorbing=is_absorbing))
            images.append(
                np.stack(
                    (
                        np.asarray(snapshot.cameras["agentview"].rgb, dtype=np.uint8),
                        np.asarray(
                            snapshot.cameras["robot0_eye_in_hand"].rgb,
                            dtype=np.uint8,
                        ),
                    )
                )
            )
            left_contact.append(contact.left_pad_contact)
            right_contact.append(contact.right_pad_contact)
            handoff_physical.append(runtime.handoff.state.value == "physical")
            absorbing.append(is_absorbing)
        if len(future_proprio) != len(_ANCHORS):
            raise RuntimeError("J4 branch did not produce all native anchors")
        return {
            "source_replay_max_abs": replay_error,
            "terminal_step": terminal_step,
            "future_proprio": np.asarray(future_proprio, dtype=np.float32),
            "future_object": np.asarray(future_object, dtype=np.float32),
            "images": np.asarray(images, dtype=np.uint8),
            "left_pad_contact": np.asarray(left_contact, dtype=np.bool_),
            "right_pad_contact": np.asarray(right_contact, dtype=np.bool_),
            "handoff_physical": np.asarray(handoff_physical, dtype=np.bool_),
            "absorbing": np.asarray(absorbing, dtype=np.bool_),
        }
    finally:
        runtime.close()


def _write_bank(
    *,
    output_dir: Path,
    contexts: tuple[J4SourceContext, ...],
    arrays: dict[str, np.ndarray],
    provenance: dict[str, Any],
) -> Path:
    target = Path(output_dir).absolute()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f".{target.name}.building-{os.getpid()}-{uuid.uuid4().hex}"
    building.mkdir()
    try:
        tensor_path = building / "bank.safetensors"
        save_file(
            {name: torch.from_numpy(np.ascontiguousarray(value)) for name, value in arrays.items()},
            tensor_path,
        )
        with tensor_path.open("rb") as handle:
            os.fsync(handle.fileno())
        context_payload = {
            "schema_version": 1,
            "branch_names": list(_BRANCHES),
            "native_delay_ticks": list(_ANCHORS),
            "contexts": [
                {
                    "logical_master_task_index": value.logical_master_task_index,
                    "episode_id": value.episode_id,
                    "category": value.category,
                    "source_tick": value.source_tick,
                }
                for value in contexts
            ],
        }
        _write_file_fsynced(
            building / "contexts.json",
            (json.dumps(context_payload, indent=2, sort_keys=True) + "\n").encode(),
        )
        artifacts = {
            name: {
                "bytes": (building / name).stat().st_size,
                "sha256": _hash_file(building / name),
            }
            for name in ("bank.safetensors", "contexts.json")
        }
        manifest = {
            "schema_version": 1,
            "format_id": _FORMAT_ID,
            "level": 3,
            "formal_validation_only": True,
            "training_eligible": False,
            "context_count": len(contexts),
            "branch_count": len(contexts) * len(_BRANCHES),
            "replay_max_abs_tolerance": _REPLAY_MAX_ABS_TOLERANCE,
            "nominal_future_max_abs_tolerance": _NOMINAL_FUTURE_MAX_ABS_TOLERANCE,
            "provenance": provenance,
            "artifacts": artifacts,
        }
        _write_file_fsynced(
            building / "manifest.json",
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
        )
        _fsync_directory(building)
        _rename_noreplace(building, target)
        _fsync_directory(target.parent)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise
    return target / "manifest.json"


def collect_l3_j4_bank(
    *,
    project_root: Path,
    device: str,
    output_dir: Path,
) -> Path:
    root = Path(project_root).resolve()
    if type(device) is not str or re.fullmatch(r"cuda:[0-9]+", device) is None:
        raise ValueError("device must identify one CUDA device")
    if Path(output_dir).exists():
        raise FileExistsError(output_dir)
    paths = _paths(root)
    config = load_action_conditioned_jepa_config(
        model_path=paths["model"],
        level_path=paths["level"],
        temporal_sampling_path=paths["temporal"],
    )
    inputs = load_verified_jepa_inputs(
        source_root=paths["source"],
        cache_run_manifest=paths["cache"],
        split_manifest_path=paths["split"],
        config=config,
    )
    episode_ids = _select_formal_episodes(inputs)
    signals_tuple = load_temporal_signal_episodes(
        inputs=inputs,
        episode_ids=episode_ids,
        level=3,
        split="validation",
    )
    contexts = select_j4_source_contexts(signals_tuple)
    if len(contexts) != _CONTEXT_COUNT:
        raise ValueError("J4 context selection must produce 20 masters x 4 phases")
    signals = {value.record.episode_id: value for value in signals_tuple}
    records = {value.record.episode_id: value.record for value in signals_tuple}
    replay_references = _load_replay_references(
        inputs=inputs,
        episode_ids=episode_ids,
    )
    tasks = _materialize_tasks(
        project_root=root,
        source_root=paths["source"],
        episode_ids=episode_ids,
    )
    encoder = HfDinoPatchEncoder.from_pretrained(
        spec=config.vision_encoder,
        device=device,
        local_files_only=True,
    )
    shape = (_CONTEXT_COUNT, len(_BRANCHES), len(_ANCHORS))
    arrays = {
        "controls": np.empty((_CONTEXT_COUNT, len(_BRANCHES), 20, 7), dtype=np.float32),
        "future_proprio": np.empty((*shape, 16), dtype=np.float32),
        "future_object": np.empty((*shape, 6), dtype=np.float32),
        "future_visual_latents": np.empty((*shape, 2, 196, 384), dtype=np.float16),
        "left_pad_contact": np.empty(shape, dtype=np.bool_),
        "right_pad_contact": np.empty(shape, dtype=np.bool_),
        "handoff_physical": np.empty(shape, dtype=np.bool_),
        "absorbing": np.empty(shape, dtype=np.bool_),
        "terminal_step": np.empty((_CONTEXT_COUNT, len(_BRANCHES)), dtype=np.int64),
        "source_replay_max_abs": np.empty((_CONTEXT_COUNT, len(_BRANCHES)), dtype=np.float64),
        "nominal_future_max_abs": np.empty((_CONTEXT_COUNT,), dtype=np.float64),
    }
    for context_index, context in enumerate(contexts):
        print(
            f"[jepa-j4] context={context_index + 1}/{len(contexts)} "
            f"master={context.logical_master_task_index} phase={context.category} "
            f"tick={context.source_tick}",
            flush=True,
        )
        record = records[context.episode_id]
        signal = signals[context.episode_id]
        branch_controls = build_j4_control_branches(
            nominal_controls=np.array(
                record.controls[context.source_tick : context.source_tick + 20],
                dtype=np.float32,
                copy=True,
            ),
            last_executed_control=np.array(
                record.controls[context.source_tick - 1],
                dtype=np.float32,
                copy=True,
            ),
        )
        images = []
        for branch_index, branch_name in enumerate(_BRANCHES):
            controls = branch_controls[branch_name]
            result = _collect_branch(
                task=tasks[context.episode_id],
                record=record,
                signal=signal,
                replay_reference=replay_references[context.episode_id],
                context=context,
                controls=controls,
            )
            arrays["controls"][context_index, branch_index] = controls
            for name in (
                "future_proprio",
                "future_object",
                "left_pad_contact",
                "right_pad_contact",
                "handoff_physical",
                "absorbing",
            ):
                arrays[name][context_index, branch_index] = result[name]
            arrays["terminal_step"][context_index, branch_index] = result["terminal_step"]
            arrays["source_replay_max_abs"][context_index, branch_index] = result[
                "source_replay_max_abs"
            ]
            images.append(result["images"])
        encoded = encoder.encode_numpy(np.stack(images).reshape(-1, 256, 256, 3))
        arrays["future_visual_latents"][context_index] = encoded.reshape(
            len(_BRANCHES), len(_ANCHORS), 2, 196, 384
        )
        nominal_proprio = record.proprio_physical[
            np.asarray(_ANCHORS, dtype=np.int64) + context.source_tick
        ]
        nominal_object = np.concatenate(
            (
                signal.object_position[np.asarray(_ANCHORS, dtype=np.int64) + context.source_tick],
                signal.object_linear_velocity[
                    np.asarray(_ANCHORS, dtype=np.int64) + context.source_tick
                ],
            ),
            axis=1,
        )
        arrays["nominal_future_max_abs"][context_index] = max(
            float(np.max(np.abs(arrays["future_proprio"][context_index, 0] - nominal_proprio))),
            float(np.max(np.abs(arrays["future_object"][context_index, 0] - nominal_object))),
        )
        if not nominal_future_is_numerically_equivalent(
            float(arrays["nominal_future_max_abs"][context_index])
        ):
            raise RuntimeError("J4 nominal branch exceeds numerical future tolerance")
    provenance = {
        "source_manifest_sha256": inputs.source_manifest_sha256,
        "cache_manifest_sha256": inputs.cache_manifest_sha256,
        "split_manifest_sha256": inputs.split_manifest_sha256,
        "model_config_sha256": sha256_file(paths["model"]),
        "temporal_config_sha256": sha256_file(paths["temporal"]),
        "vision_encoder_fingerprint": config.vision_encoder.fingerprint,
    }
    return _write_bank(
        output_dir=output_dir,
        contexts=contexts,
        arrays=arrays,
        provenance=provenance,
    )
