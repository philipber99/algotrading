"""Alert lifecycle and API models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ConditionState = Literal["active", "cleared", "expired", "unable_to_evaluate"]
UserState = Literal["unseen", "seen", "acknowledged", "snoozed", "dismissed", "suppressed"]
Severity = Literal["critical", "high", "medium", "low"]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AlertCandidate(ContractModel):
    dedup_key: str
    rule_id: str
    rule_version: str
    category: str
    title: str
    condition_state: Literal["active", "unable_to_evaluate"]
    severity: Severity
    observed_value: float | None = None
    threshold_value: float | None = None
    threshold_unit: str | None = None
    financial_impact_usd: float | None = None
    affected_position_ids: list[str] = Field(default_factory=list)
    deadline_at: datetime | None = None
    once: bool = False
    suppression_key: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)


class AlertEpisode(ContractModel):
    alert_id: str
    episode_number: int
    dedup_key: str
    rule_id: str
    rule_version: str
    category: str
    title: str
    condition_state: ConditionState
    user_state: UserState
    pinned: bool
    severity: Severity
    observed_value: float | None = None
    threshold_value: float | None = None
    threshold_unit: str | None = None
    financial_impact_usd: float | None = None
    affected_position_ids: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    triggered_at: datetime
    last_evaluated_at: datetime
    deadline_at: datetime | None = None
    cleared_at: datetime | None = None
    expired_at: datetime | None = None
    snoozed_until: datetime | None = None
    row_version: int


class AlertsResponse(ContractModel):
    schema_version: int = 1
    portfolio_snapshot_id: str | None = None
    data: list[AlertEpisode]
    counts: dict[str, int]
    as_of: datetime


class AlertPatchRequest(ContractModel):
    action: Literal[
        "mark_seen", "acknowledge", "snooze", "dismiss", "pin", "unpin",
        "suppress", "unsuppress", "activate",
    ]
    expected_version: int = Field(ge=1)
    snoozed_until: datetime | None = None


class AlertEvaluationRequest(ContractModel):
    account_ids: list[str] = Field(default_factory=lambda: ["all"])
    portfolio_snapshot_id: str
