"""Restart-persistent authoritative user strategy groups."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StrategyGroupMutation(ContractModel):
    name: str = Field(min_length=1, max_length=128)
    account_ids: list[str] = Field(min_length=1)
    member_position_ids: list[str]
    expected_version: int | None = Field(default=None, ge=1)


class StrategyGroup(ContractModel):
    group_id: str
    name: str
    account_ids: list[str]
    member_position_ids: list[str]
    created_at: datetime
    updated_at: datetime
    version: int
    member_resolution_status: Literal["current", "orphaned_or_partial"]


class StrategyGroupStore:
    def __init__(self, path: str | Path) -> None:
        self._lock = RLock()
        self._connection = sqlite3.connect(Path(path).expanduser(), check_same_thread=False, timeout=10)
        self._connection.row_factory = sqlite3.Row
        with self._connection:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS strategy_groups (
                    group_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    account_ids TEXT NOT NULL,
                    member_position_ids TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    version INTEGER NOT NULL
                )
                """
            )

    def list(self, current_position_ids: set[str] | None = None) -> list[StrategyGroup]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM strategy_groups ORDER BY name, group_id"
            ).fetchall()
        return [self._model(row, current_position_ids) for row in rows]

    def put(self, group_id: str, mutation: StrategyGroupMutation) -> StrategyGroup:
        now = datetime.now(timezone.utc).isoformat()
        # Canonicalize membership so equivalent writes compare consistently.
        accounts = sorted(dict.fromkeys(item.strip() for item in mutation.account_ids if item.strip()))
        members = sorted(dict.fromkeys(mutation.member_position_ids))
        with self._lock, self._connection:
            existing = self._connection.execute(
                "SELECT * FROM strategy_groups WHERE group_id=?", (group_id,)
            ).fetchone()
            if existing is not None and mutation.expected_version != int(existing["version"]):
                # Auto-grouping uses deterministic IDs. Re-running it against an
                # already persisted, identical group is an idempotent success,
                # not a concurrent edit. A materially different payload still
                # requires the caller to provide the current version.
                unchanged = (
                    mutation.expected_version is None
                    and mutation.name == str(existing["name"])
                    and accounts == json.loads(existing["account_ids"])
                    and members == json.loads(existing["member_position_ids"])
                )
                if unchanged:
                    return self._model(existing, None)
                raise ValueError("strategy group version changed")
            if existing is None:
                self._connection.execute(
                    "INSERT INTO strategy_groups VALUES (?,?,?,?,?,?,1)",
                    (group_id, mutation.name, json.dumps(accounts), json.dumps(members), now, now),
                )
            else:
                self._connection.execute(
                    "UPDATE strategy_groups SET name=?, account_ids=?, member_position_ids=?, "
                    "updated_at=?, version=version+1 WHERE group_id=?",
                    (mutation.name, json.dumps(accounts), json.dumps(members), now, group_id),
                )
            row = self._connection.execute(
                "SELECT * FROM strategy_groups WHERE group_id=?", (group_id,)
            ).fetchone()
        return self._model(row, None)

    def delete(self, group_id: str, expected_version: int) -> bool:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "DELETE FROM strategy_groups WHERE group_id=? AND version=?",
                (group_id, expected_version),
            )
            return cursor.rowcount == 1

    def _model(self, row: sqlite3.Row, current_position_ids: set[str] | None) -> StrategyGroup:
        members = json.loads(row["member_position_ids"])
        # Retain saved groups while flagging members absent from the current snapshot.
        partial = current_position_ids is not None and any(item not in current_position_ids for item in members)
        return StrategyGroup(
            group_id=row["group_id"], name=row["name"],
            account_ids=json.loads(row["account_ids"]), member_position_ids=members,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]), version=row["version"],
            member_resolution_status="orphaned_or_partial" if partial else "current",
        )


__all__ = ["StrategyGroup", "StrategyGroupMutation", "StrategyGroupStore"]
