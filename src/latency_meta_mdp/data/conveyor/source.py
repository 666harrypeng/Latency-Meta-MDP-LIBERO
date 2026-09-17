"""Lossless continuous episodes; deliveries are rewards, not episode boundaries.

The public sample views deliberately exclude scene truth and the arrival plan.
Images are streamed in bounded Parquet row groups, including the final boundary.
"""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from latency_meta_mdp.data.source.parquet import decode_png, encode_png

FORMAT_ID = "conveyor_continuous_source_v1"
FRAME_SCHEMA = pa.schema(
    [
        pa.field("formal_tick", pa.int64(), nullable=False),
        pa.field("state", pa.list_(pa.float32(), 16), nullable=False),
        pa.field("agentview_rgb", pa.binary(), nullable=False),
        pa.field("wrist_rgb", pa.binary(), nullable=False),
    ]
)


class ConveyorRecorder:
    """Finalize expert training sources only when every spawned parcel succeeds."""

    def __init__(
        self, root, *, seed, action_contract_id, context=None, purpose="development_smoke"
    ):
        if type(seed) is not int or seed < 0:
            raise ValueError("source seed must be a nonnegative integer")
        if not isinstance(action_contract_id, str) or not action_contract_id:
            raise ValueError("recording requires an explicit action contract identity")
        if purpose not in {"development_smoke", "training_source"}:
            raise ValueError("unsupported source purpose")
        self.purpose = purpose
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.seed, self.context = seed, context or {}
        self.action_contract_id = action_contract_id
        self.writer = pq.ParquetWriter(
            self.root / "frames.parquet", FRAME_SCHEMA, compression="zstd"
        )
        self.pending, self.actions, self.rewards, self.done = [], [], [], []
        self.last_tick, self.prompt = -1, None
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if not self.closed:
            self._flush()
            self.writer.close()
            self.closed = True

    def _flush(self):
        if self.pending:
            self.writer.write_table(pa.Table.from_pylist(self.pending, schema=FRAME_SCHEMA))
            self.pending.clear()

    def _write_observation(self, observation):
        if self.closed or observation.formal_tick != self.last_tick + 1:
            raise ValueError("source observations must be consecutive 20 ms boundaries")
        if self.prompt is not None and observation.prompt != self.prompt:
            raise ValueError("instruction changed inside the source episode")
        row = {
            "formal_tick": observation.formal_tick,
            "state": observation.state.tolist(),
            "agentview_rgb": encode_png(observation.image, compress_level=1),
            "wrist_rgb": encode_png(observation.wrist_image, compress_level=1),
        }
        self.pending.append(row)
        self.last_tick, self.prompt = observation.formal_tick, observation.prompt
        if len(self.pending) >= 32:
            self._flush()

    def start(self, observation):
        if self.last_tick != -1 or observation.formal_tick != 0:
            raise ValueError("recording must start at boundary zero exactly once")
        self._write_observation(observation)

    def append(self, action, next_observation, *, success_delta, done):
        action = np.asarray(action, dtype=np.float32)
        if self.last_tick < 0 or (self.done and self.done[-1]):
            raise ValueError("cannot append before initialization or after termination")
        if action.shape != (7,) or not np.isfinite(action).all() or (np.abs(action) > 1).any():
            raise ValueError("source actions must satisfy the 7D OSC contract")
        if type(success_delta) is not int or success_delta < 0 or type(done) is not bool:
            raise ValueError("success delta must count deliveries; done must be boolean")
        self._write_observation(next_observation)
        self.actions.append(action.copy())
        self.rewards.append(success_delta)
        self.done.append(done)

    def finish(self, summary):
        if self.closed or not self.done or not self.done[-1]:
            raise ValueError("only an episode with a real final transition may be finalized")
        if (
            summary["end_tick"] != self.last_tick
            or summary["successes"] != sum(self.rewards)
            or summary["spawned"] != sum(summary[k] for k in ("successes", "misses", "timeouts"))
        ):
            raise ValueError("source transitions disagree with the parcel ledger")
        if summary["spawned"] <= 0 or summary["successes"] != summary["spawned"]:
            raise ValueError("training source requires a fully successful expert episode")
        self.close()
        np.savez_compressed(
            self.root / "transitions.npz",
            actions=np.stack(self.actions),
            success_delta=np.asarray(self.rewards, dtype=np.int32),
            done=np.asarray(self.done, dtype=bool),
        )
        manifest = dict(
            format_id=FORMAT_ID,
            schema_version=1,
            task_id="conveyor_sort",
            variant="surface",
            episode_id=f"conveyor-surface-seed-{self.seed}",
            group_id=f"conveyor_sort/surface/seed-{self.seed}",
            seed=self.seed,
            purpose=self.purpose,
            expert_admitted=True,
            instruction=self.prompt,
            formal_tick_us=20000,
            physics_dt_us=2000,
            action_contract=self.action_contract_id,
            state_contract="joint_qpos_qvel_gripper_width_velocity",
            frame_count=self.last_tick + 1,
            transition_count=self.last_tick,
            summary=summary,
            context=self.context,
            files={
                name: (self.root / name).stat().st_size
                for name in ("frames.parquet", "transitions.npz")
            },
        )
        (self.root / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return manifest


class ConveyorSource:
    """Read one fully successful expert recording without decoding all RGB into RAM."""

    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if (
            self.manifest.get("format_id") != FORMAT_ID
            or self.manifest.get("formal_tick_us") != 20000
        ):
            raise ValueError("not a continuous conveyor source")
        summary = self.manifest["summary"]
        if (
            summary["spawned"] <= 0
            or summary["successes"] != summary["spawned"]
            or summary["misses"]
            or summary["timeouts"]
        ):
            raise ValueError("training source requires a fully successful expert episode")
        for name in ("frames.parquet", "transitions.npz"):
            if (self.root / name).stat().st_size != self.manifest["files"][name]:
                raise ValueError("source file size disagrees with its completed manifest")
        self.frames = pq.ParquetFile(self.root / "frames.parquet")
        if self.frames.schema_arrow != FRAME_SCHEMA:
            raise ValueError("source frame schema mismatch")
        values = self.frames.read(columns=["formal_tick", "state"]).to_pydict()
        self.states = np.asarray(values["state"], dtype=np.float32)
        with np.load(self.root / "transitions.npz", allow_pickle=False) as arrays:
            self.actions = arrays["actions"].copy()
            self.success_delta = arrays["success_delta"].copy()
            self.done = arrays["done"].copy()
        n = self.manifest["transition_count"]
        if (
            n <= 0
            or self.manifest["frame_count"] != n + 1
            or values["formal_tick"] != list(range(n + 1))
            or self.states.shape != (n + 1, 16)
            or self.actions.shape != (n, 7)
            or not np.isfinite(self.states).all()
            or not np.isfinite(self.actions).all()
            or (np.abs(self.actions) > 1).any()
            or self.success_delta.shape != (n,)
            or not np.issubdtype(self.success_delta.dtype, np.integer)
            or (self.success_delta < 0).any()
            or self.done.shape != (n,)
            or self.done.dtype != np.bool_
            or np.flatnonzero(self.done).tolist() != [n - 1]
            or int(self.success_delta.sum()) != self.manifest["summary"]["successes"]
        ):
            raise ValueError("source frame/action/reward/terminal alignment is invalid")
        self.group_ends = np.cumsum(
            [self.frames.metadata.row_group(i).num_rows for i in range(self.frames.num_row_groups)]
        )
        self.cached_group, self.cached_rows = None, None
        for value in (self.states, self.actions, self.success_delta, self.done):
            value.setflags(write=False)

    def rgb(self, tick):
        if type(tick) is not int or not 0 <= tick < len(self.states):
            raise IndexError("source boundary out of range")
        group = int(np.searchsorted(self.group_ends, tick, side="right"))
        if self.cached_group != group:
            self.cached_rows = self.frames.read_row_group(
                group, columns=["agentview_rgb", "wrist_rgb"]
            ).to_pylist()
            self.cached_group = group
        offset = tick - (int(self.group_ends[group - 1]) if group else 0)
        row = self.cached_rows[offset]
        return np.stack([decode_png(row[k]) for k in ("agentview_rgb", "wrist_rgb")])

    def policy_sample(self, h):
        if type(h) is not int or not 0 <= h < len(self.actions):
            raise IndexError("policy source requires a real outgoing action")
        count = min(50, len(self.actions) - h)
        actions = np.zeros((50, 7), dtype=np.float32)
        actions[:count] = self.actions[h : h + count]
        image, wrist = self.rgb(h)
        return dict(
            image=image,
            wrist_image=wrist,
            state=self.states[h].copy(),
            prompt=self.manifest["instruction"],
            actions=actions,
            actions_is_pad=np.arange(50) >= count,
        )

    def belief_window(self, h, q):
        if (
            type(h) is not int
            or type(q) is not int
            or h < 10
            or not 1 <= q <= 20
            or h + q > len(self.actions)
        ):
            raise ValueError("belief window requires real history and a recorded endpoint")
        history = np.array([h - 8, h - 4, h])
        controls = np.zeros((20, 7), dtype=np.float32)
        controls[:q] = self.actions[h : h + q]
        return {
            "inputs": dict(
                history_ticks=history,
                vision_history_rgb=np.stack([self.rgb(int(t)) for t in history]),
                proprio_history=self.states[history].copy(),
                executed_controls=self.actions[h - 8 : h].copy().reshape(2, 4, 7),
                executable_controls=controls,
                control_mask=np.arange(20) < q,
                query_ticks=q,
            ),
            "target": dict(rgb=self.rgb(h + q), proprio=self.states[h + q].copy()),
        }
