from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from algotrading.risk.exposures.portfolio import (
    portfolio_kpis,
    portfolio_risk_coverage,
)
from algotrading.risk.instruments.terms import build_instrument_terms
from algotrading.risk.market_inputs.freshness import evaluate_quote_freshness
from algotrading.risk.models import DataQuality, InstrumentTerms, PortfolioSnapshot
from algotrading.risk.payoff.engine import certify_maximum_loss
from algotrading.risk.pnl.reconstruction import (
    normalize_execution_legs,
    reconstruct_lifecycles,
)
from algotrading.risk.pnl.valuation import _lifecycle_pnl, _period_decomposition
from algotrading.risk.positions.normalization import normalize_broker_position
from algotrading.risk.scenarios.engine import ScenarioDefinition, ScenarioEngine
from algotrading.risk.tail.historical import historical_var_es
from algotrading.risk.transactions.models import TransactionItemModel, TransactionModel


UTC = timezone.utc
NOW = datetime(2026, 8, 26, 18, tzinfo=UTC)


def _position(
    symbol: str,
    asset_type: str,
    *,
    quantity: float = 1,
    mark: float = 100,
    average: float | None = None,
    classification: dict[str, str] | None = None,
    terms: InstrumentTerms | None = None,
):
    quote = {"mark": mark}
    return normalize_broker_position(
        {
            "instrument": {"symbol": symbol, "type": asset_type},
            "long_quantity": quantity,
            "short_quantity": 0,
            "average_price": average,
            "market_value": quantity * mark,
        },
        account_id="account",
        quote=quote if asset_type != "OPTION" else None,
        option_quote=(
            {"mark": mark, "underlying_price": 100}
            if asset_type == "OPTION"
            else None
        ),
        classification=classification,
        instrument_terms=terms,
    )


def _future_transaction(row_id: int, at: datetime, instruction: str, price: float):
    return TransactionModel(
        id=row_id,
        broker_transaction_id=f"future-{row_id}",
        transaction_at=at,
        transaction_type="TRADE",
        ingested_at=at,
        items=[
            TransactionItemModel(
                item_index=0,
                symbol="/ESU26",
                asset_type="FUTURE",
                instruction=instruction,
                position_effect="OPENING" if instruction == "BUY" else "CLOSING",
                quantity=Decimal("1"),
                price=Decimal(str(price)),
                currency="USD",
            )
        ],
    )


def _snapshot(*positions, portfolio_value: float = 100_000):
    return PortfolioSnapshot(
        portfolio_snapshot_id="semantic-snapshot",
        configuration_version="test",
        as_of=NOW,
        account_ids=("account",),
        portfolio_value_usd=portfolio_value,
        positions=tuple(positions),
        data_quality=DataQuality(status="usable"),
    )


def test_es_terms_notional_tick_and_variation_pnl_reconcile_without_nav_value():
    terms = build_instrument_terms("/ESU26", broker_type="FUTURE")
    position = _position(
        "/ESU26", "FUTURE", mark=6500, average=6490, terms=terms
    )

    assert terms.price_multiplier == 50
    assert terms.minimum_tick == 0.25
    assert terms.tick_value == 12.50
    assert position.signed_notional_exposure_usd == 325_000
    assert position.unrealized_pnl_usd == 500
    assert position.signed_liquidation_value_usd is None
    assert position.raw_broker_fields["broker_market_value_ignored_for_future"] == 6500

    kpis = portfolio_kpis([position], portfolio_value_usd=25_000)
    assert kpis.portfolio_value_usd == 25_000
    assert kpis.net_liquidating_value_usd == 25_000
    assert kpis.gross_market_value_usd is None


def test_futures_execution_realized_and_current_pnl_use_identical_terms():
    terms = build_instrument_terms("/ESU26", broker_type="FUTURE")
    opening = _future_transaction(1, NOW, "BUY", 6500)
    closing = _future_transaction(2, NOW + timedelta(hours=1), "SELL", 6510)
    executions, warnings = normalize_execution_legs(
        "account", [opening, closing], instrument_terms={"/ESU26": terms}
    )
    assert warnings == []
    assert {item.multiplier for item in executions} == {50}
    assert sum(item.cash_effect_usd or 0 for item in executions) == 500

    lifecycle = reconstruct_lifecycles(executions)[0]
    realized, unrealized = _period_decomposition(
        lifecycle,
        start_inventory={},
        end_inventory={},
        start_at=NOW - timedelta(seconds=1),
        end_at=NOW + timedelta(hours=2),
        marks=SimpleNamespace(),
    )
    assert realized == 500
    assert unrealized == 0

    open_executions, _ = normalize_execution_legs(
        "account", [opening], instrument_terms={"/ESU26": terms}
    )
    open_lifecycle = reconstruct_lifecycles(open_executions)[0]
    current, missing = _lifecycle_pnl(
        open_lifecycle,
        SimpleNamespace(
            as_of=NOW + timedelta(hours=1),
            current_mark=lambda _account, _symbol: 6510,
        ),
    )
    assert missing == []
    assert current == 500


def test_missing_future_terms_fail_closed_and_cannot_report_full_coverage():
    terms = build_instrument_terms("/ABCU26", broker_type="FUTURE")
    position = _position("/ABCU26", "FUTURE", mark=100, terms=terms)
    coverage = portfolio_risk_coverage([position])

    assert terms.price_multiplier is None
    assert position.signed_notional_exposure_usd is None
    assert position.delta_exposure_usd is None
    assert position.data_quality.status == "insufficient_data"
    assert coverage.instrument_terms.modeled_positions == 0
    assert coverage.instrument_terms.denominator_usd is None
    assert coverage.instrument_terms.percent is None
    assert coverage.delta.percent is None


def test_typed_scenarios_do_not_apply_broad_equity_to_crude_or_hide_missing_zn_dv01():
    crude = _position("/CLU26", "FUTURE", mark=75)
    treasury = _position("/ZNU26", "FUTURE", mark=112)
    engine = ScenarioEngine({"risk.scenarios.maximum_absolute_equity_shock": 0.5})

    broad = engine.evaluate(
        _snapshot(crude),
        ScenarioDefinition(
            scenario_id="broad-down", name="Broad equity -10%", broad_equity_shock=-0.10
        ),
    )
    assert broad["scenario_pnl_usd"] == 0
    assert broad["contributions"][0]["calculation_method"] == "no_direct_channel_for_typed_scenario"

    rates = engine.evaluate(
        _snapshot(treasury),
        ScenarioDefinition(
            scenario_id="rates-up", name="Yield curve +100 bp", rate_basis_points=100
        ),
    )
    assert rates["quality_state"] == "insufficient_data"
    assert rates["scenario_pnl_usd"] is None
    assert rates["unmodeled_position_ids"] == [treasury.position_id]
    assert rates["coverage"]["percent"] == 0


def test_sector_scenario_uses_classified_holding_and_reports_exact_contribution():
    holding = _position(
        "MSFT",
        "EQUITY",
        quantity=10,
        mark=400,
        classification={"sector": "Technology"},
    )
    result = ScenarioEngine({}).evaluate(
        _snapshot(holding),
        ScenarioDefinition(
            scenario_id="technology-down",
            name="Technology sector -8%",
            sector_shocks={"Technology": -0.08},
        ),
    )
    assert result["scenario_pnl_usd"] == -320
    assert result["coverage"]["percent"] == 100


def test_adjusted_deliverable_is_used_and_occ_assumption_cannot_certify_payoff():
    adjusted_terms = build_instrument_terms(
        "XYZ260918C00100000",
        broker_type="OPTION",
        option_terms={
            "terms_status": "adjusted",
            "adjusted_contract": True,
            "price_multiplier": 50,
            "shares_per_contract": 50,
            "deliverable": "50 XYZ shares",
            "expiration_at": "2026-09-18T20:00:00Z",
            "exercise_style": "american",
            "settlement_type": "physical",
            "source": "deterministic_fixture",
            "source_version": "fixture-v1",
        },
    )
    adjusted = _position(
        "XYZ260918C00100000", "OPTION", mark=5, terms=adjusted_terms
    )
    certified = certify_maximum_loss([adjusted])
    assert adjusted.signed_liquidation_value_usd == 250
    assert certified["status"] == "exact_bounded"
    assert certified["maximum_loss_usd"] == 250

    assumed = _position("XYZ260918C00100000", "OPTION", mark=5)
    unavailable = certify_maximum_loss([assumed])
    assert "option_terms_assumed_standard_occ" in assumed.instrument_terms.warnings
    assert unavailable["status"] == "insufficient_data"
    assert unavailable["reason"] == "option_deliverable_not_certified"


def test_payoff_certifies_long_stock_and_reports_short_stock_unbounded():
    long_stock = _position("XYZ", "EQUITY", quantity=100, mark=100)
    short_stock = normalize_broker_position(
        {
            "instrument": {"symbol": "XYZ", "type": "EQUITY"},
            "long_quantity": 0,
            "short_quantity": 100,
        },
        account_id="account",
        quote={"mark": 100},
    )
    assert certify_maximum_loss([long_stock])["maximum_loss_usd"] == 10_000
    short_result = certify_maximum_loss([short_stock])
    assert short_result["status"] == "unbounded"
    assert short_result["maximum_loss_usd"] is None


def test_weighted_es_uses_exact_fractional_mass_and_reconciling_weights():
    result = historical_var_es(
        [[-100.0, -25.0], [-50.0, 5.0], [10.0, 10.0]],
        confidence=0.80,
        half_life_sessions=10,
        unit_ids=["left", "right"],
    )
    assert result["tail_probability_mass"] == pytest.approx(0.20)
    assert sum(item["tail_weight"] for item in result["observation_weights"]) == pytest.approx(0.20)
    assert result["contribution_sum_usd"] == pytest.approx(
        result["expected_shortfall_usd"]
    )
    with pytest.raises(ValueError, match="less than one"):
        historical_var_es([[-1.0]], confidence=1.0)


def test_low_weight_extreme_changes_var_and_es_from_unweighted_order_statistics():
    result = historical_var_es(
        [[-100.0, -25.0], [-50.0, 5.0], [10.0, 10.0]],
        confidence=0.80,
        half_life_sessions=1,
        unit_ids=["left", "right"],
    )

    # An unweighted three-row order statistic would report both VaR and ES as
    # the $125 worst loss. The exact weighted tail consumes all 1/7 of that
    # row and 2/35 of the $45 row instead.
    assert result["var_usd"] == pytest.approx(45.0)
    assert result["expected_shortfall_usd"] == pytest.approx(102.14285714285715)
    assert result["observation_weights"][0]["weight"] == pytest.approx(1 / 7)
    assert sum(item["tail_weight"] for item in result["observation_weights"]) == pytest.approx(0.20)


def test_freshness_accepts_latest_completed_eod_but_rejects_stale_intraday():
    los_angeles = ZoneInfo("America/Los_Angeles")
    context = {
        "risk.freshness.intraday_max_age_seconds": 900,
        "risk.freshness.completed_session_eod_allowed": True,
        "risk.freshness.regular_session_open_local_time": "06:30",
        "risk.freshness.regular_session_close_local_time": "13:00",
    }
    eod = evaluate_quote_freshness(
        datetime(2026, 8, 25, 13, tzinfo=los_angeles),
        as_of=datetime(2026, 8, 25, 18, tzinfo=los_angeles),
        context=context,
        exchange_timezone="America/Los_Angeles",
        source_kind="daily_close",
    )
    assert eod.valid_for_risk
    assert eod.state == "valid_completed_session_eod"

    stale = evaluate_quote_freshness(
        datetime(2026, 8, 26, 9, tzinfo=los_angeles),
        as_of=datetime(2026, 8, 26, 10, tzinfo=los_angeles),
        context=context,
        exchange_timezone="America/Los_Angeles",
    )
    assert not stale.valid_for_risk
    assert stale.reason == "intraday_quote_too_old"


def test_nonfinite_and_incoherent_numeric_inputs_are_rejected_before_aggregation():
    invalid_quantity = normalize_broker_position(
        {
            "instrument": {"symbol": "XYZ", "type": "EQUITY"},
            "long_quantity": float("inf"),
            "short_quantity": 0,
        },
        account_id="account",
        quote={"mark": 100},
    )
    assert invalid_quantity.signed_quantity == 0
    assert invalid_quantity.data_quality.status == "insufficient_data"
    assert "position_quantity_nonfinite_or_invalid" in invalid_quantity.data_quality.warnings

    with pytest.raises(ValidationError):
        InstrumentTerms(
            canonical_instrument_id="BAD",
            asset_type="FUTURE",
            contract_subtype="future",
            price_multiplier=float("nan"),
            quantity_unit="contracts",
            source="fixture",
            source_version="fixture-v1",
            quality_status="standard",
        )
    with pytest.raises(ValidationError, match="minimum_tick"):
        InstrumentTerms(
            canonical_instrument_id="BAD",
            asset_type="FUTURE",
            contract_subtype="future",
            price_multiplier=50,
            minimum_tick=0.25,
            tick_value=10,
            quantity_unit="contracts",
            source="fixture",
            source_version="fixture-v1",
            quality_status="standard",
        )
