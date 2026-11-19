from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ..alerts.models import AlertCandidate, AlertPatchRequest
from ..alerts.persistence import AlertStore, AlertVersionConflict


NOW = datetime(2026, 8, 17, 12, tzinfo=timezone.utc)


def candidate(state="active", severity="high"):
    return AlertCandidate(dedup_key="rule:position", rule_id="rule", rule_version="v1",
                          category="Test", title="Synthetic lifecycle condition",
                          condition_state=state, severity=severity,
                          affected_position_ids=["synthetic-position"])


def test_alert_episode_dedup_user_pin_clear_retrigger_and_restart(tmp_path):
    path = tmp_path / "state.sqlite"
    store = AlertStore(path)
    first = store.apply_evaluation([candidate()], evaluated_rule_ids={"rule"}, evaluated_at=NOW, portfolio_snapshot_id="snapshot-1").data[0]
    pinned = store.patch(first.alert_id, AlertPatchRequest(action="pin", expected_version=first.row_version))
    acknowledged = store.patch(pinned.alert_id, AlertPatchRequest(action="acknowledge", expected_version=pinned.row_version))
    updated = store.apply_evaluation([candidate(severity="critical")], evaluated_rule_ids={"rule"}, evaluated_at=NOW+timedelta(minutes=1), portfolio_snapshot_id="snapshot-2").data[0]
    assert updated.alert_id == first.alert_id and updated.episode_number == 1
    assert updated.pinned and updated.user_state == "acknowledged" and updated.severity == "critical"
    store.apply_evaluation([], evaluated_rule_ids={"rule"}, evaluated_at=NOW+timedelta(minutes=2), portfolio_snapshot_id="snapshot-3")
    retained = AlertStore(path).list("active").data[0]
    assert retained.alert_id == first.alert_id and retained.condition_state == "active"
    assert store.patch(
        retained.alert_id,
        AlertPatchRequest(action="dismiss", expected_version=retained.row_version),
    ) == retained
    unpinned = store.patch(
        retained.alert_id,
        AlertPatchRequest(action="unpin", expected_version=retained.row_version),
    )
    assert not unpinned.pinned
    store.apply_evaluation([], evaluated_rule_ids={"rule"}, evaluated_at=NOW+timedelta(minutes=3), portfolio_snapshot_id="snapshot-4")
    assert AlertStore(path).list("history", as_of=NOW + timedelta(minutes=3)).data[0].condition_state == "cleared"
    retriggered = store.apply_evaluation([candidate()], evaluated_rule_ids={"rule"}, evaluated_at=NOW+timedelta(minutes=4), portfolio_snapshot_id="snapshot-5").data[0]
    assert retriggered.alert_id != first.alert_id and retriggered.episode_number == 2


def test_unable_to_evaluate_does_not_clear_and_optimistic_version_is_enforced(tmp_path):
    store = AlertStore(tmp_path / "state.sqlite")
    episode = store.apply_evaluation([candidate(state="unable_to_evaluate")], evaluated_rule_ids={"rule"}, evaluated_at=NOW, portfolio_snapshot_id="snapshot-1").data[0]
    store.apply_evaluation([], evaluated_rule_ids={"rule"}, evaluated_at=NOW+timedelta(minutes=1), portfolio_snapshot_id="snapshot-2")
    current = store.list("active").data[0]
    assert current.condition_state == "unable_to_evaluate"
    counts = store.list("new").counts
    assert counts["unseen_active"] == 1
    assert counts["active_non_snoozed"] == 1
    changed = store.patch(episode.alert_id, AlertPatchRequest(action="mark_seen", expected_version=current.row_version))
    with pytest.raises(AlertVersionConflict):
        store.patch(episode.alert_id, AlertPatchRequest(action="dismiss", expected_version=current.row_version))
    assert changed.user_state == "seen"


def test_history_alert_can_be_reactivated(tmp_path):
    store = AlertStore(tmp_path / "state.sqlite")
    episode = store.apply_evaluation(
        [candidate()], evaluated_rule_ids={"rule"}, evaluated_at=NOW,
        portfolio_snapshot_id="snapshot-1",
    ).data[0]
    dismissed = store.patch(
        episode.alert_id,
        AlertPatchRequest(action="dismiss", expected_version=episode.row_version),
    )
    assert store.list("history").data[0].alert_id == episode.alert_id

    activated = store.patch(
        dismissed.alert_id,
        AlertPatchRequest(action="activate", expected_version=dismissed.row_version),
    )
    assert activated.condition_state == "active"
    assert activated.user_state == "seen"
    assert activated.cleared_at is None
    assert activated.expired_at is None
    assert store.list("history").data == []
    assert store.list("active").data[0].alert_id == episode.alert_id


def test_once_only_candidate_never_opens_a_second_episode(tmp_path):
    store = AlertStore(tmp_path / "state.sqlite")
    once = candidate().model_copy(update={"once": True})
    first = store.apply_evaluation([once], evaluated_rule_ids={"rule"}, evaluated_at=NOW, portfolio_snapshot_id="snapshot-1").data[0]
    store.apply_evaluation([], evaluated_rule_ids={"rule"}, evaluated_at=NOW + timedelta(days=1), portfolio_snapshot_id="snapshot-2")
    response = store.apply_evaluation([once], evaluated_rule_ids={"rule"}, evaluated_at=NOW + timedelta(days=2), portfolio_snapshot_id="snapshot-3")
    assert response.data == []
    history = store.list("history", as_of=NOW + timedelta(days=2)).data
    assert len(history) == 1 and history[0].alert_id == first.alert_id


def test_position_alert_configuration_survives_store_restart(tmp_path):
    path = tmp_path / "state.sqlite"
    conditions = [{"name": "Breakdown", "predicates": [{"duration": "1M"}]}]
    rules = [{"rule_id": "long-equity", "condition_name": "Breakdown"}]
    AlertStore(path).save_position_configuration(conditions, rules)
    assert AlertStore(path).load_position_configuration() == {
        "risk.alerts.position.conditions": conditions,
        "risk.alerts.position.rules": rules,
    }


def test_suppression_drops_future_episodes_until_unsuppressed(tmp_path):
    store = AlertStore(tmp_path / "state.sqlite")
    first_candidate = candidate().model_copy(update={
        "dedup_key": "rule:position:2026-08-17",
        "suppression_key": "rule:position",
    })
    first = store.apply_evaluation([first_candidate], evaluated_rule_ids={"rule"}, evaluated_at=NOW, portfolio_snapshot_id="snapshot-1").data[0]
    suppressed = store.patch(first.alert_id, AlertPatchRequest(action="suppress", expected_version=first.row_version))
    assert suppressed.user_state == "suppressed"
    assert store.list("suppressed").data[0].alert_id == first.alert_id

    next_candidate = first_candidate.model_copy(update={"dedup_key": "rule:position:2026-08-18"})
    assert store.apply_evaluation([next_candidate], evaluated_rule_ids={"rule"}, evaluated_at=NOW + timedelta(days=1), portfolio_snapshot_id="snapshot-2").data == []
    current = store.list("suppressed").data[0]
    unsuppressed = store.patch(current.alert_id, AlertPatchRequest(action="unsuppress", expected_version=current.row_version))
    assert unsuppressed.user_state == "dismissed"
    assert unsuppressed.condition_state == "cleared"
    assert store.list("suppressed", as_of=NOW + timedelta(days=1)).data == []
    assert store.list("history", as_of=NOW + timedelta(days=1)).data[0].alert_id == first.alert_id

    active = store.apply_evaluation([next_candidate], evaluated_rule_ids={"rule"}, evaluated_at=NOW + timedelta(days=1, minutes=1), portfolio_snapshot_id="snapshot-3").data
    assert len(active) == 1
    assert active[0].alert_id != first.alert_id
    assert active[0].user_state == "unseen"


def test_history_older_than_fourteen_days_is_deleted_but_suppressed_is_retained(tmp_path):
    store = AlertStore(tmp_path / "state.sqlite")
    first = store.apply_evaluation([candidate()], evaluated_rule_ids={"rule"}, evaluated_at=NOW, portfolio_snapshot_id="snapshot-1").data[0]
    store.apply_evaluation([], evaluated_rule_ids={"rule"}, evaluated_at=NOW + timedelta(minutes=1), portfolio_snapshot_id="snapshot-2")
    assert len(store.list("history", as_of=NOW + timedelta(days=14)).data) == 1
    assert store.list("history", as_of=NOW + timedelta(days=15)).data == []

    retained = store.apply_evaluation([candidate()], evaluated_rule_ids={"rule"}, evaluated_at=NOW + timedelta(days=16), portfolio_snapshot_id="snapshot-3").data[0]
    store.patch(retained.alert_id, AlertPatchRequest(action="suppress", expected_version=retained.row_version))
    assert store.list("suppressed", as_of=NOW + timedelta(days=45)).data[0].alert_id == retained.alert_id
