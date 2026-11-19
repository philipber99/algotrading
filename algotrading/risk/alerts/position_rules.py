"""Configurable, daily adverse-direction alerts for held positions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from hashlib import sha256
import math
from typing import Any, Mapping, Sequence

from algotrading.risk.api.portfolio_models import PortfolioSnapshotResponse
from algotrading.risk.models import NormalizedPosition
from algotrading.refdata.stock_analysis.analytics.statistics import channel_position
from algotrading.refdata.stock_analysis.analytics.windows import period_boundary

from .models import AlertCandidate


_DURATION_DAYS = {"1W": 7, "1M": 31, "3M": 93, "6M": 186, "1Y": 366, "3Y": 1096, "5Y": 1827}
_FINRA_SESSION_COUNTS = {"1D": 1, "1W": 5, "1M": 21}
_FINRA_HALFLIVES = {"1W": 3.0, "1M": 11.0}

# These are the chart-history requests used by Stock Analysis Workbench.  A
# position alert must evaluate the same channel that the user sees there,
# rather than a calendar-month slice with a subtly different number of bars.
_STOCK_GUI_CHANNEL_LOOKBACK_DAYS = {
    "1W": 7,
    "1M": 35,
    "3M": 100,
    "6M": 190,
    "1Y": 370,
    "3Y": 1095,
    "5Y": 1830,
}


@dataclass(frozen=True)
class _PositionTarget:
    key: str
    side: str
    position_type: str
    symbol: str
    positions: tuple[NormalizedPosition, ...]


def evaluate_position_rules(
    snapshot: PortfolioSnapshotResponse,
    context: Mapping[str, Any],
    reference_client: Any,
    *,
    evaluated_at: datetime,
) -> list[AlertCandidate]:
    conditions = {
        str(item.get("name") or "").casefold(): item
        for item in context.get("risk.alerts.position.conditions", [])
        if isinstance(item, Mapping) and item.get("name") and item.get("predicates")
    }
    rules = [item for item in context.get("risk.alerts.position.rules", []) if isinstance(item, Mapping)]
    if not rules or not conditions or reference_client is None:
        return []

    prepared: list[tuple[Mapping[str, Any], Mapping[str, Any], _PositionTarget]] = []
    for rule in rules:
        condition = conditions.get(str(rule.get("condition_name") or "").casefold())
        if condition is None:
            continue
        for target in _targets(snapshot.data.positions, str(rule.get("position_type") or "")):
            if target.side == rule.get("side"):
                prepared.append((rule, condition, target))
    if not prepared:
        return []

    # Batch history for all eligible rules before evaluating individual targets.
    symbols = sorted({item[2].symbol for item in prepared if item[2].symbol})
    price_predicates = [
        predicate
        for _, condition, _ in prepared
        for predicate in condition.get("predicates", [])
        if isinstance(predicate, Mapping)
        and str(predicate.get("signal") or "") != "FINRA Short Volume %"
    ]
    finra_predicates = [
        predicate
        for _, condition, _ in prepared
        for predicate in condition.get("predicates", [])
        if isinstance(predicate, Mapping)
        and str(predicate.get("signal") or "") == "FINRA Short Volume %"
    ]
    duration_days = [
        _predicate_lookback_days(predicate)
        for predicate in price_predicates
    ]
    maximum_days = max(duration_days, default=31)
    groups: Sequence[Any] = []
    if price_predicates:
        try:
            groups = reference_client.bars(
                symbols,
                start=(evaluated_at - timedelta(days=maximum_days)).date(),
                end=evaluated_at.date(),
                include_current_quote=True,
                remote_query=False,
            )
        except (Exception,):
            groups = []
    closes_by_symbol = {
        str(group.get("symbol") or "").strip().upper(): _daily_closes(group.get("bars") or [])
        for group in groups
        if isinstance(group, Mapping)
    }
    finra_by_symbol: dict[str, list[tuple[date, float]]] = {}
    finra_sessions: list[date] = []
    if finra_predicates:
        session_limit = max(
            _FINRA_SESSION_COUNTS.get(str(item.get("duration") or ""), 0)
            for item in finra_predicates
        )
        try:
            payload = reference_client.finra_short_volume(
                symbols,
                as_of=evaluated_at.date(),
                session_limit=session_limit,
            )
        except (Exception,):
            payload = {}
        if isinstance(payload, Mapping):
            finra_sessions = _finra_dates(payload.get("sessions") or [])
            finra_rows = payload.get("observations") or []
        else:
            # Compatibility for injected clients written against the initial
            # list-returning contract during rolling upgrades.
            finra_rows = payload if isinstance(payload, Sequence) else []
        finra_by_symbol = _finra_observations(finra_rows)
        if not finra_sessions:
            finra_sessions = sorted({
                day for values in finra_by_symbol.values() for day, _ in values
            })

    matched_groups: dict[
        tuple[str, str],
        list[tuple[Mapping[str, Any], Mapping[str, Any], _PositionTarget, dict[str, Any]]],
    ] = {}
    for rule, condition, target in prepared:
        closes = closes_by_symbol.get(target.symbol.upper(), [])
        matched, evidence = _condition_matches(
            condition,
            closes,
            finra_by_symbol.get(target.symbol.upper(), []),
            finra_sessions,
            as_of=evaluated_at.date(),
        )
        if not matched:
            continue
        condition_name = str(condition.get("name") or "configured alert")
        # Combine matching rules for the same condition and underlying into one alert.
        key = (condition_name.casefold(), target.symbol.upper())
        matched_groups.setdefault(key, []).append((rule, condition, target, evidence))

    candidates: list[AlertCandidate] = []
    alert_day = evaluated_at.date().isoformat()
    for (condition_identity, symbol), matches in sorted(matched_groups.items()):
        condition_name = str(matches[0][1].get("name") or "configured alert")
        condition_key = sha256(condition_identity.encode("utf-8")).hexdigest()[:16]
        suppression_key = f"position_direction:{condition_key}:{symbol}"
        once = all(bool(rule.get("once")) for rule, _, _, _ in matches)
        dedup = suppression_key
        # Recurring rules get daily episodes while retaining a stable suppression key.
        if not once:
            dedup += f":{alert_day}"
        evidence = matches[0][3]
        predicate_evidence = evidence.get("predicates") or []
        last = predicate_evidence[-1] if predicate_evidence else {}
        positions = {
            position.position_id: position
            for _, _, target, _ in matches
            for position in target.positions
        }
        affected = sorted(positions)
        losses = [
            position.unrealized_pnl_usd
            for position in positions.values()
            if position.unrealized_pnl_usd is not None and position.unrealized_pnl_usd < 0
        ]
        sides = sorted({target.side for _, _, target, _ in matches})
        position_types = sorted({target.position_type for _, _, target, _ in matches})
        configured_rule_ids = sorted({
            str(rule.get("rule_id") or "configured") for rule, _, _, _ in matches
        })
        matched_targets = sorted({
            (target.key, target.side, target.position_type)
            for _, _, target, _ in matches
        })
        candidates.append(AlertCandidate(
            dedup_key=dedup,
            rule_id="position_direction",
            rule_version="position_direction_rules_v2",
            category="Position direction",
            title=f"{symbol} triggered {condition_name}",
            condition_state="active",
            severity="high" if losses else "medium",
            observed_value=_finite(last.get("observed")),
            threshold_value=_finite(last.get("target")),
            threshold_unit=str(last.get("signal") or "signal_value"),
            financial_impact_usd=sum(abs(value) for value in losses) or None,
            affected_position_ids=affected,
            once=once,
            suppression_key=suppression_key,
            evidence={
                **evidence,
                "configured_rule_ids": configured_rule_ids,
                "condition_name": condition_name,
                "position_side": sides[0] if len(sides) == 1 else "Multiple",
                "position_sides": sides,
                "position_type": position_types[0] if len(position_types) == 1 else "Multiple",
                "position_types": position_types,
                "matched_targets": [
                    {"target_key": key, "side": side, "position_type": position_type}
                    for key, side, position_type in matched_targets
                ],
                "symbol": symbol,
                "once": once,
                "consequence": "the configured market signal indicates movement against the listed positions on this underlying",
            },
        ))
    return candidates


def _targets(positions: Sequence[NormalizedPosition], position_type: str) -> list[_PositionTarget]:
    if position_type in {"Equity", "Future"}:
        asset_types = {"EQUITY", "ETF"} if position_type == "Equity" else {"FUTURE"}
        return [
            _PositionTarget(
                key=position.position_id,
                side="Long" if position.signed_quantity > 0 else "Short",
                position_type=position_type,
                symbol=(position.underlying_symbol or position.futures_root or position.exact_symbol).upper(),
                positions=(position,),
            )
            for position in positions
            if position.asset_type in asset_types and position.signed_quantity != 0
        ]
    put_call = "CALL" if position_type.startswith("Call") or position_type == "Call" else "PUT"
    options = [position for position in positions if position.asset_type == "OPTION" and position.put_call == put_call and position.signed_quantity != 0]
    if position_type in {"Call", "Put"}:
        return [_option_target(position, position_type) for position in options]
    calendar = position_type.endswith("Calendar")
    targets: list[_PositionTarget] = []
    # Pair only compatible opposite-side option positions within one account.
    for index, left in enumerate(options):
        for right in options[index + 1:]:
            if left.account_id != right.account_id or left.underlying_symbol != right.underlying_symbol or left.signed_quantity * right.signed_quantity >= 0:
                continue
            if calendar:
                if left.strike != right.strike or left.expiration_at == right.expiration_at or left.expiration_at is None or right.expiration_at is None:
                    continue
                far = max((left, right), key=lambda item: item.expiration_at)
                side = "Long" if far.signed_quantity > 0 else "Short"
            else:
                if left.expiration_at != right.expiration_at or left.strike == right.strike or left.strike is None or right.strike is None:
                    continue
                directional_leg = min((left, right), key=lambda item: item.strike) if put_call == "CALL" else max((left, right), key=lambda item: item.strike)
                side = "Long" if directional_leg.signed_quantity > 0 else "Short"
            pair = tuple(sorted((left, right), key=lambda item: item.position_id))
            targets.append(_PositionTarget(
                key="+".join(item.position_id for item in pair),
                side=side,
                position_type=position_type,
                symbol=(left.underlying_symbol or left.exact_symbol).upper(),
                positions=pair,
            ))
    return targets


def _option_target(position: NormalizedPosition, position_type: str) -> _PositionTarget:
    return _PositionTarget(
        key=position.position_id,
        side="Long" if position.signed_quantity > 0 else "Short",
        position_type=position_type,
        symbol=(position.underlying_symbol or position.exact_symbol).upper(),
        positions=(position,),
    )


def _daily_closes(rows: Sequence[Any]) -> list[tuple[date, float]]:
    values: list[tuple[date, float]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        try:
            timestamp = datetime.fromisoformat(str(row.get("timestamp") or "").replace("Z", "+00:00"))
            close = float(row.get("close"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(close):
            values.append((timestamp.date(), close))
    return sorted(values)


def _condition_matches(
    condition: Mapping[str, Any],
    closes: list[tuple[date, float]],
    finra_observations: list[tuple[date, float]],
    finra_sessions: list[date],
    *,
    as_of: date,
) -> tuple[bool, dict[str, Any]]:
    results: list[bool] = []
    evidence: list[dict[str, Any]] = []
    for predicate in condition.get("predicates", []):
        if not isinstance(predicate, Mapping):
            return False, {}
        duration = str(predicate.get("duration") or "1M")
        signal = str(predicate.get("signal") or "")
        if signal == "FINRA Short Volume %":
            count = _FINRA_SESSION_COUNTS.get(duration)
            if count is None or not finra_sessions:
                return False, {}
            # FINRA windows follow published sessions rather than calendar-day offsets.
            selected_sessions = set(finra_sessions[-count:])
            window = [
                (day, value)
                for day, value in finra_observations
                if day in selected_sessions
            ]
            observed = _finra_signal(window, duration)
        else:
            if not closes:
                return False, {}
            try:
                cutoff = _predicate_window_start(predicate, as_of)
            except ValueError:
                return False, {}
            window = [(day, value) for day, value in closes if day >= cutoff]
            observed = _signal(signal, window)
        target = _finite(predicate.get("target"))
        if observed is None or target is None:
            return False, {}
        comparator = str(predicate.get("comparator") or "<")
        if comparator == "<":
            result = observed < target
        elif comparator == ">":
            result = observed > target
        else:
            result = observed >= target
        results.append(result)
        evidence.append({"duration": duration, "signal": signal, "comparator": comparator, "target": target, "observed": observed, "join": predicate.get("join", "AND"), "observation_count": len(window), "data_as_of": window[-1][0].isoformat() if window else None})
    # Fold predicate joins in their configured order.
    matched = results[0]
    for index in range(1, len(results)):
        matched = (matched and results[index]) if evidence[index]["join"] == "AND" else (matched or results[index])
    signal_dates = [item.get("data_as_of") for item in evidence if item.get("data_as_of")]
    return matched, {"predicates": evidence, "signal_as_of": max(signal_dates) if signal_dates else None}


def _finra_dates(values: Sequence[Any]) -> list[date]:
    dates: set[date] = set()
    for value in values:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
        except ValueError:
            try:
                parsed = date.fromisoformat(str(value)[:10])
            except ValueError:
                continue
        dates.add(parsed)
    return sorted(dates)


def _finra_observations(rows: Sequence[Any]) -> dict[str, list[tuple[date, float]]]:
    result: dict[str, list[tuple[date, float]]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        days = _finra_dates([row.get("timestamp")])
        value = _finite(row.get("short_volume_percent"))
        if value is None:
            short = _finite(row.get("short_volume"))
            total = _finite(row.get("finra_total_volume"))
            value = short / total if short is not None and total is not None and total > 0 else None
        if symbol and days and value is not None and 0 <= value <= 1:
            result.setdefault(symbol, []).append((days[0], value))
    for values in result.values():
        values.sort()
    return result


def _finra_signal(observations: list[tuple[date, float]], duration: str) -> float | None:
    if not observations:
        return None
    values = [value for _, value in observations]
    if duration == "1D":
        return values[-1]
    halflife = _FINRA_HALFLIVES.get(duration)
    if halflife is None:
        return None
    alpha = 1.0 - math.exp(math.log(0.5) / halflife)
    result = values[0]
    for value in values[1:]:
        result = alpha * value + (1.0 - alpha) * result
    return result


def _signal(name: str, observations: list[tuple[date, float]]) -> float | None:
    if not observations:
        return None
    if name == "Channel Position":
        result = channel_position(observations, percentile_method="nearest")
        if result.get("status") != "available":
            return None
        return _finite(result.get("channel_position"))
    values = [value for _, value in observations]
    if name == "MACD":
        if len(values) < 2:
            return None
        return _ema(values, 12) - _ema(values, 26)
    if name == "RSI":
        if len(values) < 2:
            return None
        changes = [right - left for left, right in zip(values, values[1:])]
        gains = sum(max(change, 0) for change in changes) / len(changes)
        losses = sum(max(-change, 0) for change in changes) / len(changes)
        if losses == 0:
            return 100.0
        return 100.0 - 100.0 / (1.0 + gains / losses)
    return None


def _predicate_lookback_days(predicate: Mapping[str, Any]) -> int:
    duration = str(predicate.get("duration") or "1M")
    if str(predicate.get("signal") or "") == "Channel Position":
        return _STOCK_GUI_CHANNEL_LOOKBACK_DAYS.get(duration, 35)
    return _DURATION_DAYS.get(duration, 31)


def _predicate_window_start(predicate: Mapping[str, Any], as_of: date) -> date:
    if str(predicate.get("signal") or "") == "Channel Position":
        return as_of - timedelta(days=_predicate_lookback_days(predicate))
    duration = str(predicate.get("duration") or "1M")
    return period_boundary(as_of, duration.lower())


def _ema(values: list[float], period: int) -> float:
    alpha = 2.0 / (period + 1.0)
    result = values[0]
    for value in values[1:]:
        result = alpha * value + (1.0 - alpha) * result
    return result


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


__all__ = ["evaluate_position_rules"]
