"""Resumable private collection ledger for formal source generation."""

from __future__ import annotations

import json
import sqlite3
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
