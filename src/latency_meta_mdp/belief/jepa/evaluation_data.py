"""Task-aware validation sampling and RGB review, separate from training inputs."""

import io

import numpy as np
from PIL import Image


def horizon_indices(dataset, q, *, source_stride=1):
    if type(source_stride) is not int or source_stride < 1 or not 1 <= q <= 20:
        raise ValueError("invalid validation stride or query")
    offset = 0 if q == 1 else dataset.horizon_ends[q - 2]
    previous = 0
    selected = []
    for end in dataset._episode_ends[q - 1]:
        if end > previous:
            indices = list(range(offset + previous, offset + end, source_stride))
            if indices[-1] != offset + end - 1:
                indices.append(offset + end - 1)
            selected.extend(indices)
        previous = end
    return selected


def review_rgb(job, record, ticks):
    if job.task_id == "conveyor_sort":
        from latency_meta_mdp.data.conveyor.source import ConveyorSource

        source = ConveyorSource(
            job.source_root / f"episodes/seed-{record.logical_master_task_index}/source"
        )
        return {
            h: np.stack(
                [
                    np.asarray(Image.fromarray(im).resize((224, 224), Image.Resampling.BILINEAR))
                    for im in source.rgb(h)
                ]
            )
            for h in ticks
        }
    import pyarrow.parquet as pq

    meta = next(
        r
        for r in pq.read_table(job.source_root / "meta/episodes.parquet").to_pylist()
        if r["episode_id"] == record.episode_id
    )
    rows = pq.read_table(
        job.source_root / meta["data_shard"],
        filters=[("episode_id", "=", record.episode_id)],
        columns=["formal_tick", "agentview_rgb", "wrist_rgb"],
    ).to_pylist()
    wanted = set(ticks)
    return {
        r["formal_tick"]: np.stack(
            [
                np.asarray(
                    Image.open(io.BytesIO(r[k]["bytes"]))
                    .convert("RGB")
                    .resize((224, 224), Image.Resampling.BILINEAR)
                )
                for k in ("agentview_rgb", "wrist_rgb")
            ]
        )
        for r in rows
        if r["formal_tick"] in wanted
    }


def select_review_sources(dataset, *, task_id):
    chosen = []
    used = set()
    for r in dataset.records:
        if r.logical_master_task_index in used:
            continue
        candidates = [
            h
            for h in range(10, r.terminal_tick - 19)
            if task_id is not None or r.phases[h] in ("approach", "grasp_funnel")
        ]
        if not candidates:
            continue
        # Select visible motion using GT feature change, before inspecting decoder errors.
        scores = [
            np.square(
                np.asarray(r.cache.features[h + 20, 0], dtype=np.float32)
                - np.asarray(r.cache.features[h, 0], dtype=np.float32)
            ).mean()
            for h in candidates
        ]
        h = candidates[int(np.argmax(scores))]
        chosen.append((r, h))
        used.add(r.logical_master_task_index)
        if len(chosen) == 4:
            break
    return chosen
