"""Snapshot-only phase-one rules; calculators never fetch data themselves."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from algotrading.risk.api.portfolio_models import PortfolioSnapshotResponse

from .models import AlertCandidate
from .position_rules import evaluate_position_rules


class AlertEngine:
    RULE_VERSION = "risk_alert_rules_v1"

    def __init__(self, context: Mapping[str, Any], reference_client: Any | None = None) -> None:
        self.context = context
        self.reference_client = reference_client

    def evaluate(
        self,
        snapshot: PortfolioSnapshotResponse,
        *,
        evaluated_at: datetime,
    ) -> tuple[list[AlertCandidate], set[str]]:
        now = evaluated_at.astimezone(timezone.utc)
        candidates: list[AlertCandidate] = []
        evaluated_rules = {
            "short_option_assignment",
            "option_expiration",
            "margin_utilization",
            "material_risk_data_failure",
            "position_direction",
        }
        # Combine configured position signals with the built-in snapshot checks.
        candidates.extend(evaluate_position_rules(
            snapshot,
            self.context,
            self.reference_client,
            evaluated_at=now,
        ))
        expiry_window = int(
            self.context.get("risk.alerts.assignment.expiry_window_trading_days", 2)
        )
        maximum_extrinsic = float(
            self.context.get("risk.alerts.assignment.maximum_extrinsic_usd", 0.10)
        )
        for position in snapshot.data.positions:
            if position.asset_type != "OPTION" or position.expiration_at is None:
                continue
            days = (position.expiration_at.astimezone(timezone.utc) - now).total_seconds() / 86400
            if days < 0:
                continue
            if days <= max(3, expiry_window):
                premium = abs(position.signed_market_value_usd or 0)
                candidates.append(
                    AlertCandidate(
                        dedup_key=f"option_expiration:{position.position_id}",
                        rule_id="option_expiration",
                        rule_version=self.RULE_VERSION,
                        category="Expiry",
                        title=f"Option expiration · {position.canonical_occ_symbol}",
                        condition_state="active",
                        severity="high" if days <= 1 else "medium",
                        observed_value=days,
                        threshold_value=float(max(3, expiry_window)),
                        threshold_unit="calendar_days_approximation",
                        financial_impact_usd=premium or None,
                        affected_position_ids=[position.position_id],
                        deadline_at=position.expiration_at,
                        evidence={
                            "current_premium_expiring_usd": premium,
                            "signed_quantity": position.signed_quantity,
                            "consequence": "exercise, assignment, or loss of current option exposure",
                            "warning": "trading-session DTE unavailable; calendar-day gate used",
                        },
                    )
                )
            # Assignment checks apply only to short options inside the expiry window.
            if position.signed_quantity >= 0 or days > expiry_window:
                continue
            spot = position.raw_broker_fields.get("underlying_price")
            if spot is None or position.strike is None or position.mark is None:
                candidates.append(
                    AlertCandidate(
                        dedup_key=f"short_option_assignment:{position.position_id}",
                        rule_id="short_option_assignment",
                        rule_version=self.RULE_VERSION,
                        category="Assignment",
                        title=f"Unable to evaluate assignment · {position.canonical_occ_symbol}",
                        condition_state="unable_to_evaluate",
                        severity="high",
                        affected_position_ids=[position.position_id],
                        deadline_at=position.expiration_at,
                        evidence={"missing": ["spot/strike/mark"]},
                    )
                )
                continue
            # Low remaining extrinsic value identifies in-the-money assignment exposure.
            intrinsic = max(float(spot) - position.strike, 0) if position.put_call == "CALL" else max(position.strike - float(spot), 0)
            extrinsic = max(0.0, position.mark - intrinsic)
            if intrinsic > 0 and extrinsic <= maximum_extrinsic:
                shares = abs(position.signed_quantity) * (position.multiplier or 0)
                candidates.append(
                    AlertCandidate(
                        dedup_key=f"short_option_assignment:{position.position_id}",
                        rule_id="short_option_assignment",
                        rule_version=self.RULE_VERSION,
                        category="Assignment",
                        title=f"Short option assignment risk · {position.canonical_occ_symbol}",
                        condition_state="active",
                        severity="critical" if days <= 1 else "high",
                        observed_value=extrinsic,
                        threshold_value=maximum_extrinsic,
                        threshold_unit="usd_per_option_unit",
                        financial_impact_usd=shares * position.strike,
                        affected_position_ids=[position.position_id],
                        deadline_at=position.expiration_at,
                        evidence={
                            "intrinsic_value": intrinsic,
                            "extrinsic_value": extrinsic,
                            "affected_shares": shares,
                            "strike_cash_flow_usd": shares * position.strike,
                            "consequence": "temporary or permanent underlying share/cash obligation",
                        },
                    )
                )
        margin = snapshot.data.kpis.margin_utilization
        warning_margin = float(self.context.get("risk.alerts.margin.warning_utilization", 0.50))
        critical_margin = float(self.context.get("risk.alerts.margin.critical_utilization", 0.70))
        if margin is not None and margin >= warning_margin:
            candidates.append(
                AlertCandidate(
                    dedup_key="margin_utilization:portfolio",
                    rule_id="margin_utilization",
                    rule_version=self.RULE_VERSION,
                    category="Margin",
                    title="Current margin utilization",
                    condition_state="active",
                    severity="critical" if margin >= critical_margin else "high",
                    observed_value=margin,
                    threshold_value=critical_margin if margin >= critical_margin else warning_margin,
                    threshold_unit="decimal_fraction_of_net_liquidating_value",
                    evidence={"consequence": "reduced liquidity buffer; broker requirements may change"},
                )
            )
        elif margin is None:
            candidates.append(
                AlertCandidate(
                    dedup_key="margin_utilization:portfolio",
                    rule_id="margin_utilization",
                    rule_version=self.RULE_VERSION,
                    category="Margin",
                    title="Unable to evaluate margin utilization",
                    condition_state="unable_to_evaluate",
                    severity="medium",
                    evidence={"missing": ["maintenance_requirement_or_net_liquidating_value"]},
                )
            )
        # Surface degraded risk inputs alongside financial exposure alerts.
        if snapshot.data_quality.status == "insufficient_data" or snapshot.data_quality.warnings:
            affected_positions = [
                item for item in snapshot.data.positions if item.data_quality.status != "usable"
            ]
            affected = [item.position_id for item in affected_positions]
            candidates.append(
                AlertCandidate(
                    dedup_key="material_risk_data_failure:portfolio",
                    rule_id="material_risk_data_failure",
                    rule_version=self.RULE_VERSION,
                    category="Data quality",
                    title="Portfolio risk data requires review",
                    condition_state="active",
                    severity="high" if snapshot.data_quality.status == "insufficient_data" else "medium",
                    affected_position_ids=affected,
                    evidence={
                        "portfolio_quality": snapshot.data_quality.status,
                        "warnings": list(snapshot.data_quality.warnings),
                        "affected_symbols": list(dict.fromkeys(
                            item.underlying_symbol or item.exact_symbol for item in affected_positions
                        )),
                        "affected_details": [
                            {
                                "position_id": item.position_id,
                                "symbol": item.underlying_symbol or item.exact_symbol,
                                "warnings": list(item.data_quality.warnings),
                            }
                            for item in affected_positions
                        ],
                        "consequence": (
                            f"Review {len(affected_positions)} affected holding"
                            f"{'s' if len(affected_positions) != 1 else ''} listed in this alert. "
                            "Their position-level warnings identify the missing marks, Greeks, or "
                            "reference data that must be corrected before relying on affected totals."
                        ),
                    },
                )
            )
        return candidates, evaluated_rules


__all__ = ["AlertEngine"]
