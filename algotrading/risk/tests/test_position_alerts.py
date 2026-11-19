from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from algotrading.risk.alerts.engine import AlertEngine
from algotrading.risk.alerts.position_rules import (
    _finra_signal,
    _predicate_window_start,
    _signal,
)
from algotrading.risk.api.portfolio_models import (
    CacheMeta,
    PortfolioSnapshotData,
    PortfolioSnapshotResponse,
    PositionAlertPredicate,
)
from algotrading.risk.models import Coverage, DataQuality, NormalizedPosition, PortfolioKpis


NOW = datetime(2026, 8, 24, 20, tzinfo=timezone.utc)


class FallingBars:
    def bars(self, symbols, *, start, end, include_current_quote=False, remote_query=False):
        assert start == (NOW - timedelta(days=7)).date()
        assert end == NOW.date()
        assert include_current_quote is True
        assert remote_query is False
        return [
            {
                "symbol": symbol,
                "bars": [
                    {"timestamp": (NOW - timedelta(days=offset)).isoformat(), "close": close}
                    for offset, close in [(4, 100), (3, 95), (2, 85), (1, 75), (0, 70)]
                ],
            }
            for symbol in symbols
        ]


class FINRAHistory:
    def __init__(self, values):
        self.values = values

    def finra_short_volume(self, symbols, *, as_of, session_limit):
        assert symbols == ["AAPL"]
        sessions = [as_of - timedelta(days=offset) for offset in range(session_limit - 1, -1, -1)]
        return {
            "sessions": [day.isoformat() for day in sessions],
            "observations": [
                {
                    "symbol": "AAPL",
                    "timestamp": sessions[index].isoformat(),
                    "short_volume_percent": value,
                }
                for index, value in self.values
                if index < len(sessions)
            ],
        }


def quality():
    return DataQuality(status="usable", coverage=Coverage(positions_total=1, positions_modeled=1))


def equity(position_id="equity-1", quantity=10):
    return NormalizedPosition(
        position_id=position_id,
        economic_position_id=position_id,
        account_id="TEST",
        instrument_id=position_id,
        asset_type="EQUITY",
        exact_symbol="AAPL",
        underlying_symbol="AAPL",
        signed_quantity=quantity,
        multiplier=1,
        signed_cost_basis_usd=1000,
        mark=70,
        signed_market_value_usd=700,
        unrealized_pnl_usd=-300,
        current_day_pnl_usd=-50,
        data_quality=quality(),
    )


def snapshot(*positions):
    return PortfolioSnapshotResponse(
        portfolio_snapshot_id="snapshot-1",
        configuration_version="test",
        as_of=NOW,
        display_timezone="America/Los_Angeles",
        data_quality=quality(),
        data=PortfolioSnapshotData(
            account_ids=["TEST"],
            positions=list(positions),
            kpis=PortfolioKpis(
                portfolio_value_usd=700,
                gross_delta_exposure_usd=700,
                net_delta_exposure_usd=700,
                gross_market_value_usd=700,
                leverage=1,
                options_vega_per_1pct_usd=None,
                options_theta_per_day_usd=None,
                margin_utilization=0,
                modeled_market_value_percent=100,
                modeled_gross_delta_percent=100,
            ),
            balances=[],
            capabilities={},
        ),
        cache=CacheMeta(state="miss", age_seconds=0, source_as_of=NOW),
    )


def test_configured_adverse_signal_generates_position_alert_with_daily_key():
    context = {
        "risk.alerts.position.conditions": [{
            "name": "Breakdown",
            "predicates": [
                {"duration": "1W", "signal": "Channel Position", "comparator": "<", "target": 0.75, "join": "AND"},
                {"duration": "1W", "signal": "RSI", "comparator": "<", "target": 40, "join": "AND"},
            ],
        }],
        "risk.alerts.position.rules": [{"rule_id": "long-equity", "side": "Long", "position_type": "Equity", "condition_name": "Breakdown", "once": False}],
    }
    candidates, evaluated = AlertEngine(context, FallingBars()).evaluate(snapshot(equity()), evaluated_at=NOW)
    candidate = next(item for item in candidates if item.rule_id == "position_direction")
    assert candidate.dedup_key.endswith(":2026-08-24")
    assert candidate.affected_position_ids == ["equity-1"]
    assert candidate.financial_impact_usd == 300
    assert candidate.evidence["condition_name"] == "Breakdown"
    assert candidate.title == "AAPL triggered Breakdown"
    assert abs(candidate.evidence["predicates"][0]["observed"] - 0.5) < 1e-12
    assert "position_direction" in evaluated


def test_same_underlying_and_condition_generate_one_alert_for_all_positions():
    context = {
        "risk.alerts.position.conditions": [{
            "name": "Low Channel Exit",
            "predicates": [{
                "duration": "1W", "signal": "Channel Position",
                "comparator": "<", "target": 0.75, "join": "AND",
            }],
        }],
        "risk.alerts.position.rules": [{
            "rule_id": "long-equity", "side": "Long", "position_type": "Equity",
            "condition_name": "Low Channel Exit", "once": False,
        }],
    }

    candidates, _ = AlertEngine(context, FallingBars()).evaluate(
        snapshot(equity("equity-1"), equity("equity-2")), evaluated_at=NOW
    )

    position_alerts = [item for item in candidates if item.rule_id == "position_direction"]
    assert len(position_alerts) == 1
    alert = position_alerts[0]
    assert alert.title == "AAPL triggered Low Channel Exit"
    assert alert.affected_position_ids == ["equity-1", "equity-2"]
    assert alert.financial_impact_usd == 600
    assert "equity-1" not in alert.suppression_key
    assert "equity-2" not in alert.suppression_key
    assert alert.dedup_key.startswith(alert.suppression_key)


def test_different_matching_conditions_generate_distinct_alerts_for_same_position():
    context = {
        "risk.alerts.position.conditions": [
            {"name": "Low Channel", "predicates": [{
                "duration": "1W", "signal": "Channel Position",
                "comparator": "<", "target": 0.75, "join": "AND",
            }]},
            {"name": "Weak RSI", "predicates": [{
                "duration": "1W", "signal": "RSI",
                "comparator": "<", "target": 100, "join": "AND",
            }]},
        ],
        "risk.alerts.position.rules": [
            {"rule_id": "channel-rule", "side": "Long", "position_type": "Equity", "condition_name": "Low Channel", "once": False},
            {"rule_id": "rsi-rule", "side": "Long", "position_type": "Equity", "condition_name": "Weak RSI", "once": False},
        ],
    }

    candidates, _ = AlertEngine(context, FallingBars()).evaluate(
        snapshot(equity("equity-1")), evaluated_at=NOW
    )

    position_alerts = [item for item in candidates if item.rule_id == "position_direction"]
    assert {item.title for item in position_alerts} == {
        "AAPL triggered Low Channel", "AAPL triggered Weak RSI",
    }
    assert len({item.dedup_key for item in position_alerts}) == 2
    assert all(item.affected_position_ids == ["equity-1"] for item in position_alerts)


def test_side_filter_prevents_long_rule_from_matching_short_equity():
    context = {
        "risk.alerts.position.conditions": [{"name": "Breakdown", "predicates": [{"duration": "1W", "signal": "Channel Position", "comparator": "<", "target": 0.75, "join": "AND"}]}],
        "risk.alerts.position.rules": [{"rule_id": "long-equity", "side": "Long", "position_type": "Equity", "condition_name": "Breakdown", "once": True}],
    }
    candidates, _ = AlertEngine(context, FallingBars()).evaluate(snapshot(equity(quantity=-10)), evaluated_at=NOW)
    assert all(item.rule_id != "position_direction" for item in candidates)


def test_finra_one_day_alert_uses_decimal_ratio_and_strict_greater_than():
    context = {
        "risk.alerts.position.conditions": [{
            "name": "Heavy short volume",
            "predicates": [{
                "duration": "1D", "signal": "FINRA Short Volume %",
                "comparator": ">", "target": 0.6, "join": "AND",
            }],
        }],
        "risk.alerts.position.rules": [{
            "rule_id": "finra-long-equity", "side": "Long",
            "position_type": "Equity", "condition_name": "Heavy short volume",
            "once": False,
        }],
    }
    candidates, _ = AlertEngine(
        context, FINRAHistory([(0, 0.61)])
    ).evaluate(snapshot(equity()), evaluated_at=NOW)

    candidate = next(item for item in candidates if item.rule_id == "position_direction")
    predicate = candidate.evidence["predicates"][0]
    assert predicate["observed"] == 0.61
    assert predicate["observation_count"] == 1
    assert candidate.threshold_unit == "FINRA Short Volume %"


def test_finra_ema_omits_missing_samples_without_extending_the_window():
    observations = [
        (NOW.date() - timedelta(days=4), 0.2),
        (NOW.date() - timedelta(days=2), 0.8),
        (NOW.date(), 0.5),
    ]
    alpha = 1.0 - 0.5 ** (1.0 / 3.0)
    expected = alpha * 0.5 + (1 - alpha) * (
        alpha * 0.8 + (1 - alpha) * 0.2
    )
    assert _finra_signal(observations, "1W") == pytest.approx(expected)


def test_finra_alert_with_no_samples_is_silent():
    context = {
        "risk.alerts.position.conditions": [{
            "name": "Heavy short volume",
            "predicates": [{
                "duration": "1M", "signal": "FINRA Short Volume %",
                "comparator": ">", "target": 0.6, "join": "AND",
            }],
        }],
        "risk.alerts.position.rules": [{
            "rule_id": "finra-long-equity", "side": "Long",
            "position_type": "Equity", "condition_name": "Heavy short volume",
            "once": False,
        }],
    }
    candidates, _ = AlertEngine(
        context, FINRAHistory([])
    ).evaluate(snapshot(equity()), evaluated_at=NOW)

    assert all(item.rule_id != "position_direction" for item in candidates)


def test_finra_predicate_contract_restricts_window_and_decimal_scale():
    valid = PositionAlertPredicate.model_validate({
        "duration": "1D", "signal": "FINRA Short Volume %",
        "comparator": ">", "target": 0.6,
    })
    assert valid.target == 0.6
    with pytest.raises(ValueError, match="supports only"):
        PositionAlertPredicate.model_validate({
            "duration": "3M", "signal": "FINRA Short Volume %",
            "comparator": ">", "target": 0.6,
        })
    with pytest.raises(ValueError, match="between 0 and 1"):
        PositionAlertPredicate.model_validate({
            "duration": "1M", "signal": "FINRA Short Volume %",
            "comparator": ">", "target": 60,
        })


def test_alert_channel_position_matches_stock_gui_residual_percentiles_without_clipping():
    observations = [
        (NOW.date() - timedelta(days=4 - index), value)
        for index, value in enumerate([100.0, 95.0, 85.0, 75.0, 70.0])
    ]
    assert abs(_signal("Channel Position", observations) - 0.5) < 1e-12

    upper_breakout = [
        (NOW.date() - timedelta(days=4 - index), value)
        for index, value in enumerate([100.0, 101.0, 99.0, 102.0, 110.0])
    ]
    lower_breakdown = [
        (NOW.date() - timedelta(days=4 - index), value)
        for index, value in enumerate([100.0, 101.0, 99.0, 102.0, 95.0])
    ]
    assert _signal("Channel Position", upper_breakout) == 1.0
    assert _signal("Channel Position", lower_breakdown) == -1.0
    assert _signal("Channel Position", [(day, 100.0) for day, _ in observations]) is None


def test_alert_channel_position_is_not_clipped_outside_the_residual_channel():
    observations = [
        (NOW.date() - timedelta(days=9 - index), float(index * 2 + (-8 if index == 9 else (-1 if index % 2 else 1))))
        for index in range(10)
    ]

    assert _signal("Channel Position", observations) < -1.0


def test_channel_position_alert_lookbacks_match_stock_gui_chart_history():
    assert _predicate_window_start(
        {"duration": "1M", "signal": "Channel Position"}, NOW.date()
    ) == (NOW - timedelta(days=35)).date()
    assert _predicate_window_start(
        {"duration": "6M", "signal": "Channel Position"}, NOW.date()
    ) == (NOW - timedelta(days=190)).date()
