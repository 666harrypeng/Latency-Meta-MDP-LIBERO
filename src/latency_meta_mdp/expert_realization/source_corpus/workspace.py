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
class FormalPlanStatus:
    logical_master_task_index: int
    level: int
    realization_index: int
    status: str
    current_candidate_index: int
    infrastructure_retry_count: int
    selected_candidate_fingerprint: str | None
    semantic_failures: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.logical_master_task_index) is not int or self.logical_master_task_index < 0:
            raise ValueError("logical_master_task_index must be non-negative")
        if self.level not in (1, 2, 3) or type(self.realization_index) is not int or (
            self.realization_index < 0
        ):
            raise ValueError("formal plan identity is invalid")
        if self.status not in {"pending", "planning", "qualified", "exhausted"}:
            raise ValueError("formal plan status is invalid")
        if not 0 <= self.current_candidate_index < 8:
            raise ValueError("current_candidate_index must be in 0..7")
        if type(self.infrastructure_retry_count) is not int or (
            self.infrastructure_retry_count < 0
        ):
            raise ValueError("infrastructure_retry_count must be non-negative")
        if self.status == "qualified":
            if (
                type(self.selected_candidate_fingerprint) is not str
                or len(self.selected_candidate_fingerprint) != 64
            ):
                raise ValueError("qualified plan requires selected candidate fingerprint")
        elif self.selected_candidate_fingerprint is not None:
            raise ValueError("only qualified plan may carry selected candidate fingerprint")
        if type(self.semantic_failures) is not tuple or any(
            type(value) is not str or not value for value in self.semantic_failures
        ):
            raise ValueError("semantic_failures must contain non-empty strings")


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
            connection.executescript(
                """
                CREATE TABLE formal_block_status (
                    logical_master_task_index INTEGER PRIMARY KEY,
                    status TEXT NOT NULL,
                    terminal_reason TEXT
                );
                CREATE TABLE formal_plan_status (
                    logical_master_task_index INTEGER NOT NULL,
                    level INTEGER NOT NULL,
                    realization_index INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    current_candidate_index INTEGER NOT NULL,
                    infrastructure_retry_count INTEGER NOT NULL,
                    selected_candidate_fingerprint TEXT,
                    semantic_failures_json TEXT NOT NULL,
                    PRIMARY KEY (logical_master_task_index, level, realization_index)
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
                    for assignment in request.family_assignments[task.logical_task_index]:
                        connection.execute(
                            "INSERT INTO formal_plan_status VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                task.logical_task_index,
                                level,
                                assignment.realization_slot,
                                "pending",
                                0,
                                0,
                                None,
                                "[]",
                            ),
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
        if not {"formal_block_status", "formal_plan_status"} <= table_names:
            raise ValueError("workspace formal collection tables are missing")
        return workspace

    @staticmethod
    def _formal_block_from_row(row: sqlite3.Row) -> FormalBlockStatus:
        return FormalBlockStatus(
            logical_master_task_index=row["logical_master_task_index"],
            status=row["status"],
            terminal_reason=row["terminal_reason"],
        )

    @staticmethod
    def _formal_plan_from_row(row: sqlite3.Row) -> FormalPlanStatus:
        failures = json.loads(row["semantic_failures_json"])
        if type(failures) is not list:
            raise ValueError("workspace semantic failure history is invalid")
        return FormalPlanStatus(
            logical_master_task_index=row["logical_master_task_index"],
            level=row["level"],
            realization_index=row["realization_index"],
            status=row["status"],
            current_candidate_index=row["current_candidate_index"],
            infrastructure_retry_count=row["infrastructure_retry_count"],
            selected_candidate_fingerprint=row["selected_candidate_fingerprint"],
            semantic_failures=tuple(failures),
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

    def formal_plan_statuses(self) -> tuple[FormalPlanStatus, ...]:
        with self._connect(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT * FROM formal_plan_status
                ORDER BY logical_master_task_index, level, realization_index
                """
            ).fetchall()
        return tuple(self._formal_plan_from_row(row) for row in rows)

    def formal_plan_status(
        self, logical_master_task_index: int, level: int, realization_index: int
    ) -> FormalPlanStatus:
        with self._connect(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT * FROM formal_plan_status
                WHERE logical_master_task_index=? AND level=? AND realization_index=?
                """,
                (logical_master_task_index, level, realization_index),
            ).fetchone()
        if row is None:
            raise KeyError("formal plan identity is outside the workspace")
        return self._formal_plan_from_row(row)

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
                UPDATE formal_block_status SET status='planning'
                WHERE logical_master_task_index=?
                """,
                (logical_master_task_index,),
            )

    def record_formal_candidate_failure(
        self,
        *,
        logical_master_task_index: int,
        level: int,
        realization_index: int,
        candidate_index: int,
        reason: str,
    ) -> None:
        if type(candidate_index) is not int or not 0 <= candidate_index < 8:
            raise ValueError("candidate index must be in 0..7")
        if type(reason) is not str or not reason:
            raise ValueError("candidate failure reason must be non-empty")
        identity = (logical_master_task_index, level, realization_index)
        with self._connect(self.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM formal_plan_status
                WHERE logical_master_task_index=? AND level=? AND realization_index=?
                """,
                identity,
            ).fetchone()
            block = connection.execute(
                "SELECT status FROM formal_block_status WHERE logical_master_task_index=?",
                (logical_master_task_index,),
            ).fetchone()
            if row is None or block is None or block["status"] != "planning":
                raise ValueError("candidate failure requires a planning formal block")
            if row["status"] not in {"pending", "planning"} or (
                row["current_candidate_index"] != candidate_index
            ):
                raise ValueError("candidate failure does not match current candidate index")
            failures = json.loads(row["semantic_failures_json"])
            failures.append(f"candidate={candidate_index}: {reason}")
            if candidate_index == 7:
                connection.execute(
                    """
                    UPDATE formal_plan_status
                    SET status='exhausted', infrastructure_retry_count=0,
                        semantic_failures_json=?
                    WHERE logical_master_task_index=? AND level=? AND realization_index=?
                    """,
                    (json.dumps(failures), *identity),
                )
                connection.execute(
                    """
                    UPDATE formal_block_status
                    SET status='rejected', terminal_reason=?
                    WHERE logical_master_task_index=?
                    """,
                    (
                        f"planner candidates exhausted at L{level} r{realization_index}",
                        logical_master_task_index,
                    ),
                )
            else:
                connection.execute(
                    """
                    UPDATE formal_plan_status
                    SET status='planning', current_candidate_index=?,
                        infrastructure_retry_count=0, semantic_failures_json=?
                    WHERE logical_master_task_index=? AND level=? AND realization_index=?
                    """,
                    (candidate_index + 1, json.dumps(failures), *identity),
                )

    def record_formal_infrastructure_retry(
        self,
        *,
        logical_master_task_index: int,
        level: int,
        realization_index: int,
        candidate_index: int,
        reason: str,
    ) -> None:
        if type(reason) is not str or not reason:
            raise ValueError("infrastructure retry reason must be non-empty")
        identity = (logical_master_task_index, level, realization_index)
        with self._connect(self.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM formal_plan_status
                WHERE logical_master_task_index=? AND level=? AND realization_index=?
                """,
                identity,
            ).fetchone()
            block = connection.execute(
                "SELECT status FROM formal_block_status WHERE logical_master_task_index=?",
                (logical_master_task_index,),
            ).fetchone()
            if block is None or block["status"] != "planning":
                raise ValueError("infrastructure retry requires a planning formal block")
            if row is None or row["status"] not in {"pending", "planning"} or (
                row["current_candidate_index"] != candidate_index
            ):
                raise ValueError("infrastructure retry does not match current candidate index")
            connection.execute(
                """
                UPDATE formal_plan_status
                SET status='planning', infrastructure_retry_count=infrastructure_retry_count+1
                WHERE logical_master_task_index=? AND level=? AND realization_index=?
                """,
                identity,
            )

    def record_formal_plan_success(
        self,
        *,
        logical_master_task_index: int,
        level: int,
        realization_index: int,
        candidate_index: int,
        selected_candidate_fingerprint: str,
    ) -> None:
        if (
            type(selected_candidate_fingerprint) is not str
            or len(selected_candidate_fingerprint) != 64
        ):
            raise ValueError("selected_candidate_fingerprint must be a SHA-256 digest")
        identity = (logical_master_task_index, level, realization_index)
        with self._connect(self.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM formal_plan_status
                WHERE logical_master_task_index=? AND level=? AND realization_index=?
                """,
                identity,
            ).fetchone()
            block = connection.execute(
                "SELECT status FROM formal_block_status WHERE logical_master_task_index=?",
                (logical_master_task_index,),
            ).fetchone()
            if block is None or block["status"] != "planning":
                raise ValueError("plan success requires a planning formal block")
            if row is None or row["status"] not in {"pending", "planning"} or (
                row["current_candidate_index"] != candidate_index
            ):
                raise ValueError("plan success does not match current candidate")
            connection.execute(
                """
                UPDATE formal_plan_status
                SET status='qualified', selected_candidate_fingerprint=?
                WHERE logical_master_task_index=? AND level=? AND realization_index=?
                """,
                (selected_candidate_fingerprint, *identity),
            )
            connection.execute(
                """
                UPDATE realization_status SET status='planned'
                WHERE logical_master_task_index=? AND level=? AND realization_index=?
                  AND status='requested'
                """,
                identity,
            )

    def mark_formal_block_plan_ready(self, logical_master_task_index: int) -> None:
        with self._connect(self.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT COUNT(*) AS count FROM formal_plan_status
                WHERE logical_master_task_index=? AND status='qualified'
                """,
                (logical_master_task_index,),
            ).fetchone()
            if row is None or row["count"] != 12:
                raise ValueError("formal block plan_ready requires all twelve qualified plans")
            connection.execute(
                """
                UPDATE formal_block_status SET status='plan_ready'
                WHERE logical_master_task_index=? AND status='planning'
                """,
                (logical_master_task_index,),
            )
            if connection.total_changes != 1:
                raise ValueError("formal block is not in planning state")

    def begin_formal_block_execution(self, logical_master_task_index: int) -> None:
        with self._connect(self.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE formal_block_status SET status='executing'
                WHERE logical_master_task_index=? AND status='plan_ready'
                """,
                (logical_master_task_index,),
            )
            if cursor.rowcount != 1:
                raise ValueError("formal block must be plan_ready before execution")

    def record_formal_execution_status(self, status: RealizationRunStatus) -> None:
        block = self.formal_block_status(status.logical_master_task_index)
        if block.status != "executing":
            raise ValueError("formal execution status requires an executing block")
        self.record_status(status)
        if status.status in {
            "planner_failure",
            "task_failure",
            "safety_failure",
            "diversity_rejection",
        }:
            with self._connect(self.database_path) as connection:
                connection.execute(
                    """
                    UPDATE formal_block_status SET status='rejected', terminal_reason=?
                    WHERE logical_master_task_index=? AND status='executing'
                    """,
                    (status.terminal_reason, status.logical_master_task_index),
                )

    def reject_formal_block(self, logical_master_task_index: int, *, reason: str) -> None:
        if type(reason) is not str or not reason:
            raise ValueError("formal block rejection reason must be non-empty")
        with self._connect(self.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE formal_block_status SET status='rejected', terminal_reason=?
                WHERE logical_master_task_index=? AND status IN ('planning', 'executing')
                """,
                (reason, logical_master_task_index),
            )
            if cursor.rowcount != 1:
                raise ValueError("formal block rejection requires planning or executing state")

    def admit_formal_block(self, logical_master_task_index: int) -> None:
        with self._connect(self.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT COUNT(*) AS count FROM realization_status
                WHERE logical_master_task_index=? AND status='success'
                """,
                (logical_master_task_index,),
            ).fetchone()
            if row is None or row["count"] != 12:
                raise ValueError("formal block admission requires all twelve successes")
            cursor = connection.execute(
                """
                UPDATE formal_block_status SET status='admitted'
                WHERE logical_master_task_index=? AND status='executing'
                """,
                (logical_master_task_index,),
            )
            if cursor.rowcount != 1:
                raise ValueError("formal block is not executing")

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
