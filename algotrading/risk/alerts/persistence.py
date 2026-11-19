"""SQLite alert episode lifecycle with optimistic user-state updates."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import sqlite3
from threading import RLock
from typing import Iterable
import uuid

from .models import AlertCandidate, AlertEpisode, AlertPatchRequest, AlertsResponse


class AlertVersionConflict(RuntimeError):
    pass


class AlertNotFound(KeyError):
    pass


class AlertStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass
        self._lock = RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
        self._connection.row_factory = sqlite3.Row
        with self._connection:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS alert_episodes (
                    alert_id TEXT PRIMARY KEY,
                    episode_number INTEGER NOT NULL,
                    dedup_key TEXT NOT NULL,
                    rule_id TEXT NOT NULL,
                    rule_version TEXT NOT NULL,
                    category TEXT NOT NULL,
                    title TEXT NOT NULL,
                    condition_state TEXT NOT NULL,
                    user_state TEXT NOT NULL,
                    pinned INTEGER NOT NULL,
                    severity TEXT NOT NULL,
                    observed_value REAL NULL,
                    threshold_value REAL NULL,
                    threshold_unit TEXT NULL,
                    financial_impact_usd REAL NULL,
                    affected_position_ids TEXT NOT NULL,
                    evidence TEXT NOT NULL,
                    triggered_at TEXT NOT NULL,
                    last_evaluated_at TEXT NOT NULL,
                    deadline_at TEXT NULL,
                    cleared_at TEXT NULL,
                    expired_at TEXT NULL,
                    snoozed_until TEXT NULL,
                    row_version INTEGER NOT NULL
                )
                """
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS ix_alert_episodes_dedup ON alert_episodes(dedup_key, episode_number)"
            )
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS alert_configuration (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS alert_suppressions (
                    suppression_key TEXT PRIMARY KEY,
                    source_alert_id TEXT NOT NULL,
                    suppressed_at TEXT NOT NULL
                )
                """
            )
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def apply_evaluation(
        self,
        candidates: Iterable[AlertCandidate],
        *,
        evaluated_rule_ids: set[str],
        evaluated_at: datetime,
        portfolio_snapshot_id: str,
    ) -> AlertsResponse:
        now = _utc(evaluated_at)
        with self._lock, self._connection:
            suppressed_keys = {
                str(row["suppression_key"])
                for row in self._connection.execute(
                    "SELECT suppression_key FROM alert_suppressions"
                ).fetchall()
            }
            # Apply durable suppression before opening or updating alert episodes.
            candidate_list = [
                item for item in candidates
                if _candidate_suppression_key(item) not in suppressed_keys
            ]
            active_keys = {item.dedup_key for item in candidate_list}
            for candidate in candidate_list:
                row = self._connection.execute(
                    "SELECT * FROM alert_episodes WHERE dedup_key = ? "
                    "ORDER BY episode_number DESC LIMIT 1",
                    (candidate.dedup_key,),
                ).fetchone()
                # A continuing condition updates its existing episode instead of duplicating it.
                if row is not None and row["condition_state"] in {"active", "unable_to_evaluate"}:
                    self._update_candidate(str(row["alert_id"]), candidate, now, portfolio_snapshot_id)
                elif candidate.once and row is not None:
                    # A once-only position rule is permanently consumed by its first
                    # episode, even if that episode has since cleared or was dismissed.
                    continue
                else:
                    episode = int(row["episode_number"]) + 1 if row is not None else 1
                    self._insert_candidate(candidate, episode, now, portfolio_snapshot_id)
            # Clear absent conditions only for rules evaluated in this refresh.
            if evaluated_rule_ids:
                placeholders = ",".join("?" for _ in evaluated_rule_ids)
                rows = self._connection.execute(
                    "SELECT alert_id, dedup_key, condition_state, pinned FROM alert_episodes "
                    f"WHERE rule_id IN ({placeholders}) AND condition_state IN ('active','unable_to_evaluate')",
                    tuple(sorted(evaluated_rule_ids)),
                ).fetchall()
                for row in rows:
                    if row["dedup_key"] not in active_keys and row["condition_state"] == "active" and not row["pinned"]:
                        self._connection.execute(
                            "UPDATE alert_episodes SET condition_state='cleared', cleared_at=?, "
                            "last_evaluated_at=?, row_version=row_version+1 WHERE alert_id=?",
                            (now.isoformat(), now.isoformat(), row["alert_id"]),
                        )
        return self.list("active", as_of=now, portfolio_snapshot_id=portfolio_snapshot_id)

    def list(
        self,
        view: str,
        *,
        as_of: datetime | None = None,
        portfolio_snapshot_id: str | None = None,
    ) -> AlertsResponse:
        now = _utc(as_of or datetime.now(timezone.utc))
        if view == "new":
            predicate = "user_state='unseen' AND condition_state IN ('active','unable_to_evaluate')"
        elif view == "active":
            predicate = "condition_state IN ('active','unable_to_evaluate') AND user_state NOT IN ('dismissed','suppressed')"
        elif view == "history":
            predicate = "(condition_state IN ('cleared','expired') OR user_state='dismissed') AND user_state <> 'suppressed'"
        elif view == "suppressed":
            predicate = "user_state='suppressed'"
        else:
            raise ValueError("view must be new, active, history, or suppressed")
        with self._lock, self._connection:
            # Pinning is an explicit request to keep an episode actionable. Repair
            # episodes written by older versions before applying view predicates.
            self._connection.execute(
                "UPDATE alert_episodes SET condition_state='active', user_state=CASE "
                "WHEN user_state='dismissed' THEN 'seen' ELSE user_state END, "
                "cleared_at=NULL, expired_at=NULL, snoozed_until=NULL, row_version=row_version+1 "
                "WHERE pinned=1 AND user_state <> 'suppressed' "
                "AND (condition_state IN ('cleared','expired') OR user_state='dismissed')"
            )
            self._delete_old_history(now)
            rows = self._connection.execute(
                "SELECT * FROM alert_episodes WHERE " + predicate + " "
                "ORDER BY pinned DESC, CASE severity WHEN 'critical' THEN 1 WHEN 'high' THEN 2 "
                "WHEN 'medium' THEN 3 ELSE 4 END, triggered_at DESC"
            ).fetchall()
            all_rows = self._connection.execute("SELECT condition_state, user_state, pinned FROM alert_episodes").fetchall()
        episodes = [self._model(row) for row in rows]
        counts = {
            "unseen_active": sum(row["condition_state"] in {"active", "unable_to_evaluate"} and row["user_state"] == "unseen" for row in all_rows),
            "active_non_snoozed": sum(row["condition_state"] in {"active", "unable_to_evaluate"} and row["user_state"] not in {"snoozed", "dismissed", "suppressed"} for row in all_rows),
            "pinned": sum(bool(row["pinned"]) for row in all_rows),
            "suppressed": sum(row["user_state"] == "suppressed" for row in all_rows),
        }
        return AlertsResponse(
            portfolio_snapshot_id=portfolio_snapshot_id,
            data=episodes,
            counts=counts,
            as_of=now,
        )

    def _delete_old_history(self, now: datetime) -> None:
        cutoff = now - timedelta(days=14)
        self._connection.execute(
            """
            DELETE FROM alert_episodes
            WHERE user_state <> 'suppressed'
              AND (condition_state IN ('cleared','expired') OR user_state='dismissed')
              AND COALESCE(cleared_at, expired_at, last_evaluated_at, triggered_at) < ?
            """,
            (cutoff.isoformat(),),
        )

    def patch(self, alert_id: str, patch: AlertPatchRequest) -> AlertEpisode:
        now = datetime.now(timezone.utc)
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT * FROM alert_episodes WHERE alert_id=?", (alert_id,)
            ).fetchone()
            if row is None:
                raise AlertNotFound(alert_id)
            # Reject stale user updates before changing any alert lifecycle state.
            if int(row["row_version"]) != patch.expected_version:
                raise AlertVersionConflict("alert row version changed")
            user_state = str(row["user_state"])
            pinned = bool(row["pinned"])
            snoozed_until = row["snoozed_until"]
            condition_state = str(row["condition_state"])
            cleared_at = row["cleared_at"]
            expired_at = row["expired_at"]
            last_evaluated_at = str(row["last_evaluated_at"])
            if patch.action == "mark_seen":
                user_state = "seen"
            elif patch.action == "acknowledge":
                user_state = "acknowledged"
            elif patch.action == "snooze":
                if patch.snoozed_until is None or _utc(patch.snoozed_until) <= now:
                    raise ValueError("snoozed_until must be in the future")
                user_state = "snoozed"
                snoozed_until = _utc(patch.snoozed_until).isoformat()
            elif patch.action == "dismiss":
                if pinned:
                    return self._model(row)
                user_state = "dismissed"
                last_evaluated_at = now.isoformat()
            elif patch.action == "activate":
                user_state = "seen"
                condition_state = "active"
                cleared_at = None
                expired_at = None
                snoozed_until = None
                last_evaluated_at = now.isoformat()
            elif patch.action == "pin":
                pinned = True
                user_state = "seen" if user_state in {"dismissed", "snoozed"} else user_state
                condition_state = "active"
                cleared_at = None
                expired_at = None
                snoozed_until = None
            elif patch.action == "unpin":
                pinned = False
            elif patch.action == "suppress":
                suppression_key = _row_suppression_key(row)
                self._connection.execute(
                    "INSERT INTO alert_suppressions(suppression_key, source_alert_id, suppressed_at) "
                    "VALUES(?, ?, ?) ON CONFLICT(suppression_key) DO UPDATE SET "
                    "source_alert_id=excluded.source_alert_id, suppressed_at=excluded.suppressed_at",
                    (suppression_key, alert_id, now.isoformat()),
                )
                user_state = "suppressed"
                pinned = False
            elif patch.action == "unsuppress":
                self._connection.execute(
                    "DELETE FROM alert_suppressions WHERE suppression_key=?",
                    (_row_suppression_key(row),),
                )
                user_state = "dismissed"
                pinned = False
                if condition_state in {"active", "unable_to_evaluate"}:
                    condition_state = "cleared"
                    cleared_at = now.isoformat()
                last_evaluated_at = now.isoformat()
            self._connection.execute(
                "UPDATE alert_episodes SET user_state=?, pinned=?, snoozed_until=?, condition_state=?, cleared_at=?, expired_at=?, last_evaluated_at=?, "
                "row_version=row_version+1 WHERE alert_id=? AND row_version=?",
                (user_state, int(pinned), snoozed_until, condition_state, cleared_at, expired_at, last_evaluated_at, alert_id, patch.expected_version),
            )
            updated = self._connection.execute(
                "SELECT * FROM alert_episodes WHERE alert_id=?", (alert_id,)
            ).fetchone()
        return self._model(updated)

    def load_position_configuration(self) -> dict[str, list[dict]]:
        keys = ("risk.alerts.position.conditions", "risk.alerts.position.rules")
        with self._lock:
            rows = self._connection.execute(
                "SELECT key, value FROM alert_configuration WHERE key IN (?, ?)", keys
            ).fetchall()
        result: dict[str, list[dict]] = {}
        for row in rows:
            try:
                value = json.loads(row["value"])
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(value, list):
                result[str(row["key"])] = [item for item in value if isinstance(item, dict)]
        return result

    def save_position_configuration(self, conditions: list[dict], rules: list[dict]) -> None:
        values = {
            "risk.alerts.position.conditions": conditions,
            "risk.alerts.position.rules": rules,
        }
        with self._lock, self._connection:
            for key, value in values.items():
                self._connection.execute(
                    "INSERT INTO alert_configuration(key, value) VALUES(?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (key, json.dumps(value, sort_keys=True)),
                )

    def _insert_candidate(
        self,
        candidate: AlertCandidate,
        episode: int,
        now: datetime,
        snapshot_id: str,
    ) -> None:
        evidence = dict(candidate.evidence)
        evidence["portfolio_snapshot_id"] = snapshot_id
        evidence["suppression_key"] = _candidate_suppression_key(candidate)
        self._connection.execute(
            "INSERT INTO alert_episodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                str(uuid.uuid4()), episode, candidate.dedup_key, candidate.rule_id,
                candidate.rule_version, candidate.category, candidate.title,
                candidate.condition_state, "unseen", 0, candidate.severity,
                candidate.observed_value, candidate.threshold_value,
                candidate.threshold_unit, candidate.financial_impact_usd,
                json.dumps(candidate.affected_position_ids), json.dumps(evidence, default=str),
                now.isoformat(), now.isoformat(),
                _iso(candidate.deadline_at), None, None, None, 1,
            ),
        )

    def _update_candidate(
        self,
        alert_id: str,
        candidate: AlertCandidate,
        now: datetime,
        snapshot_id: str,
    ) -> None:
        evidence = dict(candidate.evidence)
        evidence["portfolio_snapshot_id"] = snapshot_id
        evidence["suppression_key"] = _candidate_suppression_key(candidate)
        self._connection.execute(
            """
            UPDATE alert_episodes SET rule_version=?, category=?, title=?,
                condition_state=?, severity=?, observed_value=?, threshold_value=?,
                threshold_unit=?, financial_impact_usd=?, affected_position_ids=?,
                evidence=?, last_evaluated_at=?, deadline_at=?, row_version=row_version+1
            WHERE alert_id=?
            """,
            (
                candidate.rule_version, candidate.category, candidate.title,
                candidate.condition_state, candidate.severity, candidate.observed_value,
                candidate.threshold_value, candidate.threshold_unit,
                candidate.financial_impact_usd,
                json.dumps(candidate.affected_position_ids), json.dumps(evidence, default=str),
                now.isoformat(), _iso(candidate.deadline_at), alert_id,
            ),
        )

    def _model(self, row: sqlite3.Row) -> AlertEpisode:
        return AlertEpisode(
            alert_id=row["alert_id"], episode_number=row["episode_number"],
            dedup_key=row["dedup_key"], rule_id=row["rule_id"],
            rule_version=row["rule_version"], category=row["category"], title=row["title"],
            condition_state=row["condition_state"], user_state=row["user_state"],
            pinned=bool(row["pinned"]), severity=row["severity"],
            observed_value=row["observed_value"], threshold_value=row["threshold_value"],
            threshold_unit=row["threshold_unit"], financial_impact_usd=row["financial_impact_usd"],
            affected_position_ids=json.loads(row["affected_position_ids"]),
            evidence=json.loads(row["evidence"]), triggered_at=_parse(row["triggered_at"]),
            last_evaluated_at=_parse(row["last_evaluated_at"]),
            deadline_at=_parse_optional(row["deadline_at"]), cleared_at=_parse_optional(row["cleared_at"]),
            expired_at=_parse_optional(row["expired_at"]), snoozed_until=_parse_optional(row["snoozed_until"]),
            row_version=row["row_version"],
        )


def default_alert_state_path(context) -> Path:
    configured = str(context.get("risk.state.path", "") or "").strip()
    if configured:
        return Path(configured).expanduser()
    root = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return root / "algotrading" / "risk-api-state.sqlite3"


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("alert timestamps must be timezone aware")
    return value.astimezone(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return _utc(value).isoformat() if value else None


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def _parse_optional(value: str | None) -> datetime | None:
    return _parse(value) if value else None


_DAILY_SUFFIX = re.compile(r":\d{4}-\d{2}-\d{2}$")


def _candidate_suppression_key(candidate: AlertCandidate) -> str:
    # Strip the daily suffix so suppression also covers future daily episodes.
    return candidate.suppression_key or _DAILY_SUFFIX.sub("", candidate.dedup_key)


def _row_suppression_key(row: sqlite3.Row) -> str:
    try:
        evidence = json.loads(row["evidence"])
    except (TypeError, ValueError, json.JSONDecodeError):
        evidence = {}
    configured = evidence.get("suppression_key") if isinstance(evidence, dict) else None
    return str(configured or _DAILY_SUFFIX.sub("", str(row["dedup_key"])))


__all__ = [
    "AlertNotFound", "AlertStore", "AlertVersionConflict", "default_alert_state_path"
]
