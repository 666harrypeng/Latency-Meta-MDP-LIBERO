"""Reuse the clean train package, adding only its real Nth observation per episode."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from latency_meta_mdp.data.source.parquet import decode_png, encode_png
from latency_meta_mdp.io.artifacts import sha256_file

FORMAT = "conveyor_forecast_terminal_boundaries_v1"


def write_terminal_assets(corpus_manifest, clean_inputs, output_dir):
    from latency_meta_mdp.data.conveyor.corpus import split_sources

    export = json.loads(clean_inputs.policy_export_manifest.read_text())
    if sha256_file(corpus_manifest) != export["source_manifest_sha256"]:
        raise ValueError("clean export and terminal source identity mismatch")
    inventory = {row["episode_id"]: row for row in export["episodes"]}
    rows = []
    for row, source in split_sources(corpus_manifest, split="train"):
        n = len(source.actions)
        if inventory[row["episode_id"]]["frame_count"] != n:
            raise ValueError("clean export/terminal action count differs")
        image, wrist = source.rgb(n)
        rows.append(
            dict(
                episode_id=row["episode_id"],
                seed=row["seed"],
                formal_tick=n,
                state=source.states[n].tolist(),
                image=encode_png(image, compress_level=1),
                wrist_image=encode_png(wrist, compress_level=1),
            )
        )
    if {row["episode_id"] for row in rows} != set(inventory):
        raise ValueError("terminal inventory differs from clean train inventory")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "terminal_boundaries.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    manifest = dict(
        format_id=FORMAT,
        task_id="conveyor_sort",
        split="train",
        action_contract=export["action_contract"],
        source_manifest_sha256=export["source_manifest_sha256"],
        training_bundle_manifest_sha256=clean_inputs.bundle_identity,
        policy_export_manifest_sha256=sha256_file(clean_inputs.policy_export_manifest),
        episodes=len(rows),
        terminal_bytes=path.stat().st_size,
        terminal_sha256=sha256_file(path),
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def load_terminal_assets(clean_inputs, root):
    root = Path(root)
    m = json.loads((root / "manifest.json").read_text())
    if (
        m["format_id"] != FORMAT
        or m["training_bundle_manifest_sha256"] != clean_inputs.bundle_identity
        or m["policy_export_manifest_sha256"] != sha256_file(clean_inputs.policy_export_manifest)
    ):
        raise ValueError("terminal asset/clean export identity mismatch")
    path = root / "terminal_boundaries.parquet"
    if path.stat().st_size != m["terminal_bytes"] or sha256_file(path) != m["terminal_sha256"]:
        raise ValueError("terminal observations are incomplete or changed")
    rows = pq.read_table(path).to_pylist()
    export = json.loads(clean_inputs.policy_export_manifest.read_text())
    if len(rows) != len(export["episodes"]) or {r["episode_id"] for r in rows} != {
        r["episode_id"] for r in export["episodes"]
    }:
        raise ValueError("terminal episode identity mismatch")
    return export, {r["episode_id"]: r for r in rows}


class CleanForecastEpisode:
    def __init__(self, path, row, terminal):
        self.row, self.terminal = row, terminal
        self.frames = pq.ParquetFile(path)
        n = row["frame_count"]
        if (
            terminal["episode_id"] != row["episode_id"]
            or terminal["seed"] != row["seed"]
            or terminal["formal_tick"] != n
            or self.frames.metadata.num_rows != n
        ):
            raise ValueError("terminal observation does not follow the clean episode")
        values = self.frames.read(columns=["frame_index", "state", "actions"]).to_pydict()
        self.states = np.asarray([*values["state"], terminal["state"]], np.float32)
        self.actions = np.asarray(values["actions"], np.float32)
        if (
            values["frame_index"] != list(range(n))
            or self.states.shape != (n + 1, 16)
            or self.actions.shape != (n, 7)
            or not np.isfinite(self.states).all()
            or not np.isfinite(self.actions).all()
        ):
            raise ValueError("clean forecast state/action alignment mismatch")
        self.group_ends = np.cumsum(
            [self.frames.metadata.row_group(i).num_rows for i in range(self.frames.num_row_groups)]
        )
        self.cached_group, self.cached_rows = None, None

    def rgb(self, tick):
        if not 0 <= tick <= len(self.actions):
            raise IndexError("forecast source boundary out of range")
        if tick == len(self.actions):
            return np.stack([decode_png(self.terminal[k]) for k in ("image", "wrist_image")])
        group = int(np.searchsorted(self.group_ends, tick, side="right"))
        if group != self.cached_group:
            self.cached_rows = self.frames.read_row_group(
                group, columns=["image", "wrist_image"]
            ).to_pylist()
            self.cached_group = group
        offset = tick - (int(self.group_ends[group - 1]) if group else 0)
        return np.stack(
            [decode_png(self.cached_rows[offset][k]["bytes"]) for k in ("image", "wrist_image")]
        )


def clean_forecast_episodes(clean_inputs, terminal_root):
    export, terminals = load_terminal_assets(clean_inputs, terminal_root)
    data_root = clean_inputs.policy_export_manifest.parent
    info = json.loads((data_root / "meta/info.json").read_text())
    for index, row in enumerate(export["episodes"]):
        path = data_root / info["data_path"].format(
            episode_chunk=index // info["chunks_size"], episode_index=index
        )
        yield row, CleanForecastEpisode(path, row, terminals[row["episode_id"]])
