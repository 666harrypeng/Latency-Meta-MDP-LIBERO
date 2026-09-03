"""Resumable private collection ledger for formal source generation."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from latency_meta_mdp.expert_realization.artifacts import (
    _fsync_directory,
    _write_file_fsynced,
)
from latency_meta_mdp.expert_realization.contracts import FormalRequestUniverse

_ACTIVE = {"requested", "planned", "running"}
_FAILURES = {
    "planner_failure",
    "task_failure",
    "safety_failure",
    "diversity_rejection",
    "infrastructure_failure",
}
_STATUSES = _ACTIVE | _FAILURES | {"success"}
_TERMINAL_SEMANTIC = {
    "planner_failure",
    "task_failure",
    "safety_failure",
    "diversity_rejection",
    "success",
}


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _safe_relative(value: str | None) -> str | None:
    if value is None:
        return None
    if type(value) is not str or not value:
        raise ValueError("payload_path must be a non-empty relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
        or not path.parts
        or path.parts[0] != "payloads"
    ):
        raise ValueError("payload_path must be a normalized relative path")
    return value


@dataclass(frozen=True)
class RealizationRunStatus:
    logical_master_task_index: int
    level: int
    realization_index: int
    status: str
    attempt_index: int
    payload_path: str | None = None
    terminal_reason: str | None = None

    def __post_init__(self) -> None:
        for name in ("logical_master_task_index", "realization_index", "attempt_index"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if type(self.level) is not int or self.level not in (1, 2, 3):
            raise ValueError("level must be one of 1, 2, or 3")
        if self.status not in _STATUSES:
            raise ValueError("status is not a recognized collection status")
        _safe_relative(self.payload_path)
        if self.terminal_reason is not None and (
            type(self.terminal_reason) is not str or not self.terminal_reason
        ):
            raise ValueError("terminal_reason must be a non-empty string or None")
        if self.status == "success":
            if self.payload_path is None or self.terminal_reason is None:
                raise ValueError("success requires payload_path and terminal_reason")
        elif self.status in _FAILURES:
            if self.payload_path is not None or self.terminal_reason is None:
                raise ValueError("failure requires a reason and cannot retain payload_path")
        elif self.payload_path is not None or self.terminal_reason is not None:
            raise ValueError("active status cannot carry terminal fields")


@dataclass(frozen=True)
class FormalBlockStatus:
    logical_master_task_index: int
    status: str
    terminal_reason: str | None

    def __post_init__(self) -> None:
        if type(self.logical_master_task_index) is not int or self.logical_master_task_index < 0:
            raise ValueError("logical_master_task_index must be non-negative")
        if self.status not in {
            "pending",
            "filling",
            "planning",
            "plan_ready",
            "executing",
            "admitted",
            "rejected",
        }:
            raise ValueError("formal block status is invalid")
        if (self.status == "rejected") != (self.terminal_reason is not None):
            raise ValueError("only rejected blocks carry terminal_reason")


@dataclass(frozen=True)
class LevelQuotaStatus:
    logical_master_task_index: int
    level: int
    status: str
    next_draw_index: int
    accepted_count: int

    def __post_init__(self) -> None:
        if type(self.logical_master_task_index) is not int or self.logical_master_task_index < 0:
            raise ValueError("logical_master_task_index must be non-negative")
        if self.level not in (1, 2, 3):
            raise ValueError("level must be one of 1, 2, or 3")
        if self.status not in {"pending", "filling", "complete", "exhausted"}:
            raise ValueError("level quota status is invalid")
        if type(self.next_draw_index) is not int or self.next_draw_index < 0:
            raise ValueError("next_draw_index must be non-negative")
        if type(self.accepted_count) is not int or self.accepted_count < 0:
            raise ValueError("accepted_count must be non-negative")


@dataclass(frozen=True)
class DrawRunStatus:
    logical_master_task_index: int
    level: int
    realization_draw_index: int
    family: str
    realization_seed: int
    status: str
    attempt_index: int
    accepted_slot: int | None = None
    plan_path: str | None = None
    payload_path: str | None = None
    selected_candidate_fingerprint: str | None = None
    terminal_reason: str | None = None
    last_infrastructure_reason: str | None = None

    def __post_init__(self) -> None:
        for name in ("logical_master_task_index", "realization_draw_index", "attempt_index"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.level not in (1, 2, 3):
            raise ValueError("level must be one of 1, 2, or 3")
        if type(self.family) is not str or not self.family:
            raise ValueError("family must be a non-empty string")
        if type(self.realization_seed) is not int or not 0 <= self.realization_seed < 2**64:
            raise ValueError("realization_seed must be uint64")
        if self.status not in {
            "planning",
            "plan_qualified",
            "running",
            "planner_failure",
            "task_failure",
            "safety_failure",
            "diversity_rejection",
            "accepted",
        }:
            raise ValueError("draw status is invalid")
        if self.accepted_slot is not None and (
            type(self.accepted_slot) is not int or self.accepted_slot < 0
        ):
            raise ValueError("accepted_slot must be a non-negative integer or None")
        for value in (self.plan_path, self.payload_path):
            _safe_relative(value)
        if self.selected_candidate_fingerprint is not None and (
            type(self.selected_candidate_fingerprint) is not str
            or len(self.selected_candidate_fingerprint) != 64
        ):
            raise ValueError("selected_candidate_fingerprint must be a SHA-256 digest")
        for name in ("terminal_reason", "last_infrastructure_reason"):
            value = getattr(self, name)
            if value is not None and (type(value) is not str or not value):
                raise ValueError(f"{name} must be a non-empty string or None")
        terminal = {
            "planner_failure",
            "task_failure",
            "safety_failure",
            "diversity_rejection",
            "accepted",
        }
        if (self.status in terminal) != (self.terminal_reason is not None):
            raise ValueError("only terminal draw statuses carry terminal_reason")
        if self.status == "accepted":
            if self.accepted_slot is None or self.payload_path is None:
                raise ValueError("accepted draw requires slot and payload")
        elif self.accepted_slot is not None or self.payload_path is not None:
            raise ValueError("only accepted draw carries slot and payload")


class CollectionWorkspace:
    """Own one exact request ledger; large attempt payloads remain outside the database."""

    def __init__(self, root: Path, request: FormalRequestUniverse) -> None:
        self.root = Path(root)
        self.request = request
        self.database_path = self.root / "run_state.sqlite"

    @property
    def payload_root(self) -> Path:
        return self.root / "payloads"

    @staticmethod
    def _connect(path: Path) -> sqlite3.Connection:
        connection = sqlite3.connect(path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    @classmethod
    def create(cls, root: Path, request: FormalRequestUniverse) -> CollectionWorkspace:
        if not isinstance(request, FormalRequestUniverse):
            raise TypeError("request must be FormalRequestUniverse")
        root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        request_payload = _canonical_json(request.to_mapping())
        _write_file_fsynced(root / "request.json", request_payload)
        workspace = cls(root, request)
        workspace.payload_root.mkdir()
        with cls._connect(workspace.database_path) as connection:
            connection.executescript(
                """
                CREATE TABLE metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE realization_status (
                    logical_master_task_index INTEGER NOT NULL,
                    master_task_seed TEXT NOT NULL,
                    reserve INTEGER NOT NULL,
                    level INTEGER NOT NULL,
                    realization_index INTEGER NOT NULL,
                    family TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt_index INTEGER NOT NULL,
                    payload_path TEXT,
                    terminal_reason TEXT,
                    PRIMARY KEY (logical_master_task_index, level, realization_index)
                );
                """
            )
            connection.execute(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                ("request_sha256", request.request_sha256),
            )
            for task in request.primary_tasks + request.reserve_tasks:
                assignments = request.family_assignments[task.logical_task_index]
                for level in request.config.levels:
                    for assignment in assignments:
                        connection.execute(
                            """
                            INSERT INTO realization_status VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                task.logical_task_index,
                                str(task.master_task_seed),
                                int(task.reserve),
                                level,
                                assignment.realization_slot,
                                assignment.family.value,
                                "requested",
                                0,
                                None,
                                None,
                            ),
                        )
        _fsync_directory(root)
        return workspace

    @classmethod
    def create_formal(
        cls,
        root: Path,
        request: FormalRequestUniverse,
        *,
        collection_identity: Mapping[str, Any],
    ) -> CollectionWorkspace:
        if not isinstance(collection_identity, Mapping):
            raise TypeError("collection_identity must be a mapping")
        identity_json = json.dumps(
            dict(collection_identity), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        workspace = cls.create(root, request)
        with cls._connect(workspace.database_path) as connection:
            connection.execute("DELETE FROM realization_status")
            connection.executescript(
                """
                CREATE TABLE formal_block_status (
                    logical_master_task_index INTEGER PRIMARY KEY,
                    status TEXT NOT NULL,
                    terminal_reason TEXT
                );
                CREATE TABLE level_quota_status (
                    logical_master_task_index INTEGER NOT NULL,
                    level INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    next_draw_index INTEGER NOT NULL,
                    accepted_count INTEGER NOT NULL,
                    PRIMARY KEY (logical_master_task_index, level)
                );
                CREATE TABLE draw_status (
                    logical_master_task_index INTEGER NOT NULL,
                    level INTEGER NOT NULL,
                    realization_draw_index INTEGER NOT NULL,
                    family TEXT NOT NULL,
                    realization_seed TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt_index INTEGER NOT NULL,
                    accepted_slot INTEGER,
                    plan_path TEXT,
                    payload_path TEXT,
                    selected_candidate_fingerprint TEXT,
                    terminal_reason TEXT,
                    last_infrastructure_reason TEXT,
                    PRIMARY KEY (logical_master_task_index, level, realization_draw_index),
                    UNIQUE (logical_master_task_index, level, accepted_slot)
                );
                """
            )
            connection.execute(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                ("formal_collection_identity", identity_json),
            )
            for task in request.primary_tasks + request.reserve_tasks:
                connection.execute(
                    "INSERT INTO formal_block_status VALUES (?, ?, ?)",
                    (task.logical_task_index, "pending", None),
                )
                for level in request.config.levels:
                    connection.execute(
                        "INSERT INTO level_quota_status VALUES (?, ?, ?, ?, ?)",
                        (task.logical_task_index, level, "pending", 0, 0),
                    )
        _fsync_directory(workspace.root)
        return workspace

    @classmethod
    def resume(cls, root: Path, request: FormalRequestUniverse) -> CollectionWorkspace:
        if not isinstance(request, FormalRequestUniverse):
            raise TypeError("request must be FormalRequestUniverse")
        root = Path(root)
        required = {"request.json", "run_state.sqlite", "payloads"}
        allowed = required | {
            "run_state.sqlite-journal",
            "run_state.sqlite-wal",
            "run_state.sqlite-shm",
        }
        names = {path.name for path in root.iterdir()} if root.is_dir() else set()
        if not required <= names or not names <= allowed:
            raise ValueError("workspace inventory is invalid")
        payload_root = root / "payloads"
        if not payload_root.is_dir() or payload_root.is_symlink():
            raise ValueError("workspace payload root is invalid")
        if (root / "request.json").read_bytes() != _canonical_json(request.to_mapping()):
            raise ValueError("workspace request identity does not match")
        with cls._connect(root / "run_state.sqlite") as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key='request_sha256'"
            ).fetchone()
        if row is None or row["value"] != request.request_sha256:
            raise ValueError("workspace request identity does not match")
        return cls(root, request)

    @classmethod
    def resume_formal(
        cls,
        root: Path,
        request: FormalRequestUniverse,
        *,
        collection_identity: Mapping[str, Any],
    ) -> CollectionWorkspace:
        if not isinstance(collection_identity, Mapping):
            raise TypeError("collection_identity must be a mapping")
        workspace = cls.resume(root, request)
        expected = json.dumps(
            dict(collection_identity), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        with cls._connect(workspace.database_path) as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key='formal_collection_identity'"
            ).fetchone()
            table_names = {
                item["name"]
                for item in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        if row is None or row["value"] != expected:
            raise ValueError("workspace formal collection identity does not match")
        if not {"formal_block_status", "level_quota_status", "draw_status"} <= table_names:
            raise ValueError("workspace formal collection tables are missing")
        return workspace

    @staticmethod
    def _formal_block_from_row(row: sqlite3.Row) -> FormalBlockStatus:
        return FormalBlockStatus(
            logical_master_task_index=row["logical_master_task_index"],
            status=row["status"],
            terminal_reason=row["terminal_reason"],
        )

    def formal_block_statuses(self) -> tuple[FormalBlockStatus, ...]:
        with self._connect(self.database_path) as connection:
            rows = connection.execute(
                "SELECT * FROM formal_block_status ORDER BY logical_master_task_index"
            ).fetchall()
        return tuple(self._formal_block_from_row(row) for row in rows)

    def formal_block_status(self, logical_master_task_index: int) -> FormalBlockStatus:
        with self._connect(self.database_path) as connection:
            row = connection.execute(
                "SELECT * FROM formal_block_status WHERE logical_master_task_index=?",
                (logical_master_task_index,),
            ).fetchone()
        if row is None:
            raise KeyError("formal block identity is outside the workspace")
        return self._formal_block_from_row(row)

    @staticmethod
    def _level_quota_from_row(row: sqlite3.Row) -> LevelQuotaStatus:
        return LevelQuotaStatus(
            logical_master_task_index=row["logical_master_task_index"],
            level=row["level"],
            status=row["status"],
            next_draw_index=row["next_draw_index"],
            accepted_count=row["accepted_count"],
        )

    @staticmethod
    def _draw_from_row(row: sqlite3.Row) -> DrawRunStatus:
        return DrawRunStatus(
            logical_master_task_index=row["logical_master_task_index"],
            level=row["level"],
            realization_draw_index=row["realization_draw_index"],
            family=row["family"],
            realization_seed=int(row["realization_seed"]),
            status=row["status"],
            attempt_index=row["attempt_index"],
            accepted_slot=row["accepted_slot"],
            plan_path=row["plan_path"],
            payload_path=row["payload_path"],
            selected_candidate_fingerprint=row["selected_candidate_fingerprint"],
            terminal_reason=row["terminal_reason"],
            last_infrastructure_reason=row["last_infrastructure_reason"],
        )

    def level_quota_statuses(self) -> tuple[LevelQuotaStatus, ...]:
        with self._connect(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT * FROM level_quota_status
                ORDER BY logical_master_task_index, level
                """
            ).fetchall()
        return tuple(self._level_quota_from_row(row) for row in rows)

    def level_quota_status(
        self, logical_master_task_index: int, level: int
    ) -> LevelQuotaStatus:
        with self._connect(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT * FROM level_quota_status
                WHERE logical_master_task_index=? AND level=?
                """,
                (logical_master_task_index, level),
            ).fetchone()
        if row is None:
            raise KeyError("level quota identity is outside the workspace")
        return self._level_quota_from_row(row)

    def draw_statuses(self) -> tuple[DrawRunStatus, ...]:
        with self._connect(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT * FROM draw_status
                ORDER BY logical_master_task_index, level, realization_draw_index
                """
            ).fetchall()
        return tuple(self._draw_from_row(row) for row in rows)

    def draw_status(
        self, logical_master_task_index: int, level: int, realization_draw_index: int
    ) -> DrawRunStatus:
        with self._connect(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT * FROM draw_status
                WHERE logical_master_task_index=? AND level=?
                  AND realization_draw_index=?
                """,
                (logical_master_task_index, level, realization_draw_index),
            ).fetchone()
        if row is None:
            raise KeyError("draw identity is outside the workspace")
        return self._draw_from_row(row)

    def accepted_draws(
        self, logical_master_task_index: int, level: int
    ) -> tuple[DrawRunStatus, ...]:
        return tuple(
            row
            for row in self.draw_statuses()
            if row.logical_master_task_index == logical_master_task_index
            and row.level == level
            and row.status == "accepted"
        )

    @staticmethod
    def _draw_identity(draw: DrawRunStatus) -> tuple[int, int, int]:
        if not isinstance(draw, DrawRunStatus):
            raise TypeError("draw must be DrawRunStatus")
        return (
            draw.logical_master_task_index,
            draw.level,
            draw.realization_draw_index,
        )

    def begin_next_draw(self, *, logical_master_task_index: int, request: Any) -> DrawRunStatus:
        from latency_meta_mdp.expert_realization.contracts import FormalRealizationDrawRequest

        if not isinstance(request, FormalRealizationDrawRequest):
            raise TypeError("request must be FormalRealizationDrawRequest")
        masters = {
            row.logical_task_index: row
            for row in self.request.primary_tasks + self.request.reserve_tasks
        }
        master = masters.get(logical_master_task_index)
        if master is None or master.master_task_seed != request.task_instance_id.task_instance_seed:
            raise ValueError("draw request does not match the formal master task")
        identity = (
            logical_master_task_index,
            request.task_instance_id.level,
            request.realization_draw_index,
        )
        with self._connect(self.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            block = connection.execute(
                "SELECT status FROM formal_block_status WHERE logical_master_task_index=?",
                (logical_master_task_index,),
            ).fetchone()
            level = connection.execute(
                """
                SELECT * FROM level_quota_status
                WHERE logical_master_task_index=? AND level=?
                """,
                identity[:2],
            ).fetchone()
            if block is None or block["status"] != "filling":
                raise ValueError("draw requires a filling formal block")
            if level is None or level["status"] == "complete":
                raise ValueError("level quota is complete or missing")
            if level["status"] == "exhausted":
                raise ValueError("level quota is exhausted")
            if request.realization_draw_index != level["next_draw_index"]:
                raise ValueError("draw index must equal the level's next draw index")
            existing = connection.execute(
                """
                SELECT * FROM draw_status
                WHERE logical_master_task_index=? AND level=?
                  AND realization_draw_index=?
                """,
                identity,
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO draw_status VALUES (?, ?, ?, ?, ?, 'planning', 0,
                        NULL, NULL, NULL, NULL, NULL, NULL)
                    """,
                    (
                        *identity,
                        request.assigned_family.value,
                        str(request.realization_seed),
                    ),
                )
                connection.execute(
                    """
                    UPDATE level_quota_status SET status='filling'
                    WHERE logical_master_task_index=? AND level=? AND status='pending'
                    """,
                    identity[:2],
                )
                existing = connection.execute(
                    """
                    SELECT * FROM draw_status
                    WHERE logical_master_task_index=? AND level=?
                      AND realization_draw_index=?
                    """,
                    identity,
                ).fetchone()
            if existing is None:
                raise RuntimeError("draw row was not created")
            value = self._draw_from_row(existing)
            if value.family != request.assigned_family.value or (
                value.realization_seed != request.realization_seed
            ):
                raise ValueError("stored draw identity does not match request")
            return value

    def record_draw_plan_qualified(
        self,
        draw: DrawRunStatus,
        *,
        selected_candidate_fingerprint: str,
        plan_path: str,
    ) -> DrawRunStatus:
        identity = self._draw_identity(draw)
        _safe_relative(plan_path)
        if type(selected_candidate_fingerprint) is not str or (
            len(selected_candidate_fingerprint) != 64
        ):
            raise ValueError("selected_candidate_fingerprint must be a SHA-256 digest")
        with self._connect(self.database_path) as connection:
            cursor = connection.execute(
                """
                UPDATE draw_status SET status='plan_qualified', plan_path=?,
                    selected_candidate_fingerprint=?
                WHERE logical_master_task_index=? AND level=?
                  AND realization_draw_index=? AND status='planning'
                """,
                (plan_path, selected_candidate_fingerprint, *identity),
            )
            if cursor.rowcount != 1:
                raise ValueError("draw must be planning before plan qualification")
        return self.draw_status(*identity)

    def record_draw_running(self, draw: DrawRunStatus) -> DrawRunStatus:
        identity = self._draw_identity(draw)
        with self._connect(self.database_path) as connection:
            cursor = connection.execute(
                """
                UPDATE draw_status SET status='running'
                WHERE logical_master_task_index=? AND level=?
                  AND realization_draw_index=? AND status='plan_qualified'
                """,
                identity,
            )
            if cursor.rowcount != 1:
                raise ValueError("draw must be plan_qualified before rollout")
        return self.draw_status(*identity)

    def record_draw_infrastructure_retry(
        self, draw: DrawRunStatus, *, reason: str
    ) -> DrawRunStatus:
        identity = self._draw_identity(draw)
        if type(reason) is not str or not reason:
            raise ValueError("infrastructure retry reason must be non-empty")
        with self._connect(self.database_path) as connection:
            cursor = connection.execute(
                """
                UPDATE draw_status SET attempt_index=attempt_index+1,
                    last_infrastructure_reason=?
                WHERE logical_master_task_index=? AND level=?
                  AND realization_draw_index=?
                  AND status IN ('planning', 'plan_qualified', 'running')
                """,
                (reason, *identity),
            )
            if cursor.rowcount != 1:
                raise ValueError("only an active draw may retry infrastructure")
        return self.draw_status(*identity)

    def record_draw_failure(
        self, draw: DrawRunStatus, *, failure_class: str, reason: str
    ) -> DrawRunStatus:
        identity = self._draw_identity(draw)
        if failure_class not in {
            "planner_failure",
            "task_failure",
            "safety_failure",
            "diversity_rejection",
        }:
            raise ValueError("failure_class is not a terminal semantic draw failure")
        if type(reason) is not str or not reason:
            raise ValueError("draw failure reason must be non-empty")
        with self._connect(self.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE draw_status SET status=?, terminal_reason=?
                WHERE logical_master_task_index=? AND level=?
                  AND realization_draw_index=?
                  AND status IN ('planning', 'plan_qualified', 'running')
                """,
                (failure_class, reason, *identity),
            )
            if cursor.rowcount != 1:
                raise ValueError("only an active draw may fail")
            connection.execute(
                """
                UPDATE level_quota_status SET status='filling', next_draw_index=?
                WHERE logical_master_task_index=? AND level=?
                  AND next_draw_index=? AND status='filling'
                """,
                (identity[2] + 1, identity[0], identity[1], identity[2]),
            )
            if connection.total_changes != 2:
                raise RuntimeError("draw failure did not advance exactly one level cursor")
        return self.draw_status(*identity)

    def record_draw_accepted(
        self,
        draw: DrawRunStatus,
        *,
        payload_path: str,
        terminal_reason: str,
    ) -> int:
        identity = self._draw_identity(draw)
        _safe_relative(payload_path)
        if type(terminal_reason) is not str or not terminal_reason:
            raise ValueError("terminal_reason must be non-empty")
        quota = self.request.config.realizations_per_task
        with self._connect(self.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            level = connection.execute(
                """
                SELECT * FROM level_quota_status
                WHERE logical_master_task_index=? AND level=?
                """,
                identity[:2],
            ).fetchone()
            if level is None or level["status"] != "filling":
                raise ValueError("accepted draw requires a filling level quota")
            slot = level["accepted_count"]
            if slot >= quota:
                raise ValueError("level quota is already complete")
            cursor = connection.execute(
                """
                UPDATE draw_status SET status='accepted', accepted_slot=?,
                    payload_path=?, terminal_reason=?
                WHERE logical_master_task_index=? AND level=?
                  AND realization_draw_index=? AND status='running'
                """,
                (slot, payload_path, terminal_reason, *identity),
            )
            if cursor.rowcount != 1:
                raise ValueError("draw must be running before acceptance")
            accepted_count = slot + 1
            connection.execute(
                """
                UPDATE level_quota_status SET status=?, next_draw_index=?, accepted_count=?
                WHERE logical_master_task_index=? AND level=?
                  AND next_draw_index=? AND accepted_count=?
                """,
                (
                    "complete" if accepted_count == quota else "filling",
                    identity[2] + 1,
                    accepted_count,
                    identity[0],
                    identity[1],
                    identity[2],
                    slot,
                ),
            )
            if connection.total_changes != 2:
                raise RuntimeError("draw acceptance did not update exactly one level quota")
            return slot

    def begin_formal_block(self, logical_master_task_index: int) -> None:
        with self._connect(self.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM formal_block_status WHERE logical_master_task_index=?",
                (logical_master_task_index,),
            ).fetchone()
            if row is None or row["status"] != "pending":
                raise ValueError("formal block must be pending before planning")
            connection.execute(
                """
                UPDATE formal_block_status SET status='filling'
                WHERE logical_master_task_index=?
                """,
                (logical_master_task_index,),
            )

    def reject_formal_block(self, logical_master_task_index: int, *, reason: str) -> None:
        if type(reason) is not str or not reason:
            raise ValueError("formal block rejection reason must be non-empty")
        with self._connect(self.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE formal_block_status SET status='rejected', terminal_reason=?
                WHERE logical_master_task_index=? AND status IN ('filling', 'planning', 'executing')
                """,
                (reason, logical_master_task_index),
            )
            if cursor.rowcount != 1:
                raise ValueError("formal block rejection requires an active state")

    def admit_formal_block(self, logical_master_task_index: int) -> None:
        with self._connect(self.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            levels = connection.execute(
                """
                SELECT COUNT(*) AS count FROM level_quota_status
                WHERE logical_master_task_index=? AND status='complete'
                """,
                (logical_master_task_index,),
            ).fetchone()
            if levels is None or levels["count"] != len(self.request.config.levels):
                raise ValueError("formal block admission requires three complete level quotas")
            cursor = connection.execute(
                """
                UPDATE formal_block_status SET status='admitted'
                WHERE logical_master_task_index=? AND status='filling'
                """,
                (logical_master_task_index,),
            )
            if cursor.rowcount != 1:
                raise ValueError("formal block is not filling")

    @staticmethod
    def _from_row(row: sqlite3.Row) -> RealizationRunStatus:
        return RealizationRunStatus(
            logical_master_task_index=row["logical_master_task_index"],
            level=row["level"],
            realization_index=row["realization_index"],
            status=row["status"],
            attempt_index=row["attempt_index"],
            payload_path=row["payload_path"],
            terminal_reason=row["terminal_reason"],
        )

    def status_rows(self) -> tuple[RealizationRunStatus, ...]:
        with self._connect(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT * FROM realization_status
                ORDER BY logical_master_task_index, level, realization_index
                """
            ).fetchall()
        return tuple(self._from_row(row) for row in rows)

    def status_for(
        self, *, logical_master_task_index: int, level: int, realization_index: int
    ) -> RealizationRunStatus:
        with self._connect(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT * FROM realization_status
                WHERE logical_master_task_index=? AND level=? AND realization_index=?
                """,
                (logical_master_task_index, level, realization_index),
            ).fetchone()
        if row is None:
            raise KeyError("realization identity is outside the workspace request")
        return self._from_row(row)

    def record_status(self, status: RealizationRunStatus) -> None:
        if not isinstance(status, RealizationRunStatus):
            raise TypeError("status must be RealizationRunStatus")
        identity = (
            status.logical_master_task_index,
            status.level,
            status.realization_index,
        )
        with self._connect(self.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM realization_status
                WHERE logical_master_task_index=? AND level=? AND realization_index=?
                """,
                identity,
            ).fetchone()
            if row is None:
                raise KeyError("realization identity is outside the workspace request")
            previous = self._from_row(row)
            if previous.status in _TERMINAL_SEMANTIC:
                raise ValueError("terminal semantic status cannot transition")
            allowed = {
                "requested": {"planned", "planner_failure"},
                "planned": {"running", "planner_failure"},
                "running": {
                    "success",
                    "task_failure",
                    "safety_failure",
                    "diversity_rejection",
                    "infrastructure_failure",
                },
                "infrastructure_failure": {"running"},
            }[previous.status]
            if status.status not in allowed:
                raise ValueError("invalid status transition")
            expected_attempt = previous.attempt_index
            if previous.status == "infrastructure_failure" and status.status == "running":
                expected_attempt += 1
            if status.attempt_index != expected_attempt:
                raise ValueError("attempt_index does not match retry semantics")
            connection.execute(
                """
                UPDATE realization_status
                SET status=?, attempt_index=?, payload_path=?, terminal_reason=?
                WHERE logical_master_task_index=? AND level=? AND realization_index=?
                """,
                (
                    status.status,
                    status.attempt_index,
                    status.payload_path,
                    status.terminal_reason,
                    *identity,
                ),
            )

    def admitted_complete_blocks(self) -> tuple[int, ...]:
        expected_per_block = (
            len(self.request.config.levels) * self.request.config.realizations_per_task
        )
        with self._connect(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT logical_master_task_index,
                       SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) AS successes,
                       COUNT(*) AS requested
                FROM realization_status
                GROUP BY logical_master_task_index
                ORDER BY logical_master_task_index
                """
            ).fetchall()
        return tuple(
            row["logical_master_task_index"]
            for row in rows
            if row["requested"] == expected_per_block and row["successes"] == expected_per_block
        )
