"""Deterministic transaction normalization and lifecycle reconstruction.

Grouping evidence is applied in descending strength: a multi-item parent,
common order ID, explicit position effect, and finally a narrow same-account /
same-underlying time cluster. Inventory and lot accounting are always isolated
by account and exact security identifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, time, timedelta, timezone
from decimal import Decimal
import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence

from algotrading.refdata.api.option_symbols import OccSymbol, parse_occ_symbol
from algotrading.risk.instruments.terms import build_instrument_terms
from algotrading.risk.models import InstrumentTerms
from algotrading.risk.transactions.models import TransactionItemModel, TransactionModel


EPSILON = 1e-9
METHODOLOGY_VERSION = "pnl_lifecycle_instrument_terms_v2"


@dataclass(frozen=True)
class ExecutionLeg:
    account_id: str
    repository_transaction_id: int
    broker_transaction_id: str | None
    item_index: int
    timestamp: datetime
    order_id: str | None
    description: str | None
    symbol: str
    asset_type: str
    underlying_symbol: str
    instruction: str | None
    position_effect: str | None
    signed_quantity: float
    fill_price: float | None
    multiplier: float
    fees_usd: float
    cash_effect_usd: float | None
    parent_item_count: int
    minimum_tick: float | None = None
    tick_value: float | None = None
    currency: str = "USD"
    quote_convention: str | None = None
    expiration_at: datetime | None = None
    instrument_terms_source: str = "legacy_execution"
    instrument_terms_version: str = "pnl_lifecycle_v1"
    fee_components_usd: dict[str, float] = field(default_factory=dict)
    role: str = "trade"

    @property
    def evidence_id(self) -> str:
        return str(self.repository_transaction_id)

    @property
    def action(self) -> str:
        instruction = str(self.instruction or "").strip()
        has_side = "BUY" in instruction.upper() or "SELL" in instruction.upper()
        values = []
        if not has_side:
            values.append("Buy" if self.signed_quantity > 0 else "Sell")
        values.extend((instruction, self.position_effect or ""))
        text = " · ".join(dict.fromkeys(value for value in values if value))
        if text:
            return text.replace("_", " ").title()
        return "Buy" if self.signed_quantity > 0 else "Sell"


@dataclass
class Lifecycle:
    account_id: str
    underlying_symbol: str
    instrument_type: str
    executions: list[ExecutionLeg] = field(default_factory=list)
    inventory: dict[str, float] = field(default_factory=dict)
    peak_inventory: dict[str, float] = field(default_factory=dict)
    original_symbols: set[str] = field(default_factory=set)
    order_ids: set[str] = field(default_factory=set)
    confidence_percent: float = 75.0
    evidence: str = "time_cluster"
    closed_at: datetime | None = None

    @property
    def opened_at(self) -> datetime:
        return min(item.timestamp for item in self.executions)

    @property
    def earliest_repository_transaction_id(self) -> int:
        return min(item.repository_transaction_id for item in self.executions)

    @property
    def symbols(self) -> set[str]:
        return {item.symbol for item in self.executions}

    @property
    def is_closed(self) -> bool:
        return bool(self.executions) and all(
            abs(quantity) <= EPSILON for quantity in self.inventory.values()
        )

    @property
    def has_reduction(self) -> bool:
        for symbol in self.original_symbols:
            peak = abs(self.peak_inventory.get(symbol, 0.0))
            current = abs(self.inventory.get(symbol, 0.0))
            if peak > EPSILON and current + EPSILON < peak:
                return True
        return False

    @property
    def trade_id(self) -> str:
        opening_at = self.opened_at
        opening = [
            item
            for item in self.executions
            if abs((item.timestamp - opening_at).total_seconds()) <= 2
        ]
        fingerprint = "|".join(
            sorted(
                f"{item.symbol}:{item.signed_quantity:.10g}:{item.order_id or ''}"
                for item in opening
            )
        )
        evidence = (
            f"{METHODOLOGY_VERSION}\x1f{self.account_id}\x1f"
            f"{self.earliest_repository_transaction_id}\x1f{fingerprint}"
        )
        return "pnl:" + hashlib.sha256(evidence.encode("utf-8")).hexdigest()


def normalize_execution_legs(
    account_id: str,
    transactions: Sequence[TransactionModel],
    *,
    instrument_terms: Mapping[str, InstrumentTerms | Mapping[str, Any]] | None = None,
    context: Mapping[str, Any] | None = None,
) -> tuple[list[ExecutionLeg], list[str]]:
    """Normalize repository parents and items without double-counting cash."""
    executions: list[ExecutionLeg] = []
    warnings: list[str] = []
    for parent in sorted(transactions, key=lambda row: (row.transaction_at, row.id)):
        if parent.transaction_type != "TRADE":
            continue
        items = list(parent.items)
        if not items:
            warnings.append(f"transaction_{parent.id}_has_no_execution_items")
            continue
        # Resolve per-leg contract economics before allocating parent cash and fees.
        item_terms = [
            _execution_terms(item, parent, instrument_terms or {}, context or {})
            for item in items
        ]
        fee_component_allocations = allocate_parent_fee_components(
            parent, items, instrument_terms=item_terms
        )
        fee_allocations = [sum(item.values()) for item in fee_component_allocations]
        cash_allocations = _cash_allocations(parent, items, item_terms)
        for index, item in enumerate(items):
            quantity = _finite(item.quantity)
            if quantity is None:
                warnings.append(
                    f"transaction_{parent.id}_item_{item.item_index}_missing_quantity"
                )
                continue
            if abs(quantity) <= EPSILON:
                if _is_non_execution_item(item):
                    continue
                warnings.append(
                    f"transaction_{parent.id}_item_{item.item_index}_zero_quantity"
                )
                continue
            symbol = str(item.symbol or parent.symbol or "").strip().upper()
            if not symbol:
                warnings.append(
                    f"transaction_{parent.id}_item_{item.item_index}_missing_symbol"
                )
                continue
            asset_type, occ = _asset_and_occ(item.asset_type or parent.asset_type, symbol)
            if occ is not None:
                symbol = occ.canonical
            # Infer execution direction from instruction and cash evidence.
            signed_quantity = _signed_quantity(
                quantity,
                item.instruction,
                cash_effect_usd=cash_allocations[index],
            )
            terms = item_terms[index]
            multiplier = terms.price_multiplier
            if asset_type in {"OPTION", "FUTURE"} and multiplier is None:
                warnings.append(
                    f"transaction_{parent.id}_item_{item.item_index}_instrument_terms_unavailable"
                )
                continue
            if multiplier is None:
                multiplier = 1.0
            executions.append(
                ExecutionLeg(
                    account_id=account_id,
                    repository_transaction_id=parent.id,
                    broker_transaction_id=parent.broker_transaction_id,
                    item_index=item.item_index,
                    timestamp=parent.transaction_at,
                    order_id=(parent.order_id or "").strip() or None,
                    description=parent.description,
                    symbol=symbol,
                    asset_type=asset_type,
                    underlying_symbol=occ.root if occ else symbol,
                    instruction=(item.instruction or "").strip().upper() or None,
                    position_effect=(item.position_effect or "").strip().upper() or None,
                    signed_quantity=signed_quantity,
                    fill_price=_finite(item.price),
                    multiplier=multiplier,
                    minimum_tick=terms.minimum_tick,
                    tick_value=terms.tick_value,
                    currency=terms.currency,
                    quote_convention=terms.quote_convention,
                    expiration_at=terms.expiration_at,
                    instrument_terms_source=terms.source,
                    instrument_terms_version=terms.source_version,
                    fees_usd=fee_allocations[index],
                    cash_effect_usd=cash_allocations[index],
                    parent_item_count=len(items),
                    fee_components_usd=fee_component_allocations[index],
                )
            )
    executions.sort(
        key=lambda item: (
            item.timestamp,
            item.repository_transaction_id,
            item.item_index,
            item.symbol,
        )
    )
    return executions, list(dict.fromkeys(warnings))


def _is_non_execution_item(item: TransactionItemModel) -> bool:
    """Identify broker audit rows that do not represent position movement."""
    asset_type = str(item.asset_type or "").strip().upper()
    return (
        asset_type in {"CURRENCY", "CASH_EQUIVALENT"}
        and not str(item.instruction or "").strip()
        and not str(item.position_effect or "").strip()
    )


def allocate_parent_fees(
    parent: TransactionModel,
    items: Sequence[TransactionItemModel],
    *,
    instrument_terms: Sequence[InstrumentTerms] | None = None,
) -> list[float]:
    """Allocate the stored fee components by leg notional, conserving their total."""
    return [
        sum(item.values())
        for item in allocate_parent_fee_components(
            parent, items, instrument_terms=instrument_terms
        )
    ]


def allocate_parent_fee_components(
    parent: TransactionModel,
    items: Sequence[TransactionItemModel],
    *,
    instrument_terms: Sequence[InstrumentTerms] | None = None,
) -> list[dict[str, float]]:
    """Allocate each fee type by leg notional in the P&L signed-expense convention.

    The transaction fee table records broker charges as positive amounts.  P&L
    represents their cash effect as negative.  Parent ``fees`` remains a
    fallback for legacy rows without component records.
    """
    if not items:
        return []
    grouped: dict[str, float] = {}
    for fee in parent.fee_components:
        amount = _finite(fee.amount)
        if amount is None or abs(amount) <= EPSILON:
            continue
        fee_type = str(fee.fee_type or "").strip().upper() or "UNCLASSIFIED"
        grouped[fee_type] = grouped.get(fee_type, 0.0) - abs(amount)
    if not grouped:
        raw_fee = _finite(parent.fees) or 0.0
        if raw_fee:
            grouped["UNCLASSIFIED"] = -abs(raw_fee)
    allocations = [dict() for _ in items]
    # Allocate each fee category separately to preserve the broker breakdown.
    for fee_type, total in grouped.items():
        for index, value in enumerate(
            _allocate_by_notional(total, items, instrument_terms=instrument_terms)
        ):
            if value:
                allocations[index][fee_type] = value
    return allocations


def _allocate_by_notional(
    total: float,
    items: Sequence[TransactionItemModel],
    *,
    instrument_terms: Sequence[InstrumentTerms] | None = None,
) -> list[float]:
    if not items:
        return []
    weights = []
    for index, item in enumerate(items):
        quantity = abs(_finite(item.quantity) or 0.0)
        price = abs(_finite(item.price) or 0.0)
        terms = instrument_terms[index] if instrument_terms is not None else _execution_terms(
            item, None, {}, {}
        )
        multiplier = terms.price_multiplier or 0.0
        weights.append(quantity * price * multiplier or quantity or 1.0)
    denominator = sum(weights)
    allocations: list[float] = []
    allocated = 0.0
    for index, weight in enumerate(weights):
        # Assign rounding remainder to the final leg so allocations exactly reconcile.
        value = total - allocated if index == len(weights) - 1 else total * weight / denominator
        allocations.append(value)
        allocated += value
    return allocations


def reconstruct_lifecycles(
    executions: Sequence[ExecutionLeg],
    *,
    cluster_seconds: float = 2.0,
) -> list[Lifecycle]:
    """Reconstruct account-isolated lifecycles in deterministic FIFO order."""
    lifecycles: list[Lifecycle] = []
    active: list[Lifecycle] = []
    grouped = _parent_groups(executions)
    for batch in grouped:
        if not batch:
            continue
        account_id = batch[0].account_id
        underlying = batch[0].underlying_symbol
        # Match executions only to open lifecycles in the same account and underlying.
        relevant = [
            item
            for item in active
            if item.account_id == account_id
            and item.underlying_symbol == underlying
            and not item.is_closed
        ]
        candidate, reason, confidence = _select_candidate(batch, relevant, cluster_seconds)
        if candidate is None:
            candidate = _new_lifecycle(batch, reason, confidence)
            lifecycles.append(candidate)
            active.append(candidate)
        else:
            candidate.confidence_percent = min(candidate.confidence_percent, confidence)
            if _evidence_rank(reason) > _evidence_rank(candidate.evidence):
                candidate.evidence = reason

        # A trade through zero closes one lifecycle and opens the opposite position.
        overflows: list[ExecutionLeg] = []
        for execution in batch:
            overflow = _append_with_flip(candidate, execution)
            if overflow is not None:
                overflows.append(overflow)
        if candidate.is_closed:
            candidate.closed_at = max(item.timestamp for item in candidate.executions)
            if candidate in active:
                active.remove(candidate)
        if overflows:
            continuation = _new_lifecycle(overflows, "explicit_position_effect", 88.0)
            for execution in overflows:
                _append_without_split(continuation, execution)
            lifecycles.append(continuation)
            if not continuation.is_closed:
                active.append(continuation)

    return sorted(
        lifecycles,
        key=lambda item: (
            item.opened_at,
            item.account_id,
            item.underlying_symbol,
            item.trade_id,
        ),
    )


def lifecycle_status(
    lifecycle: Lifecycle,
    *,
    end_at: datetime,
    lookahead_end_at: datetime | None,
) -> str:
    inventory_at_end = inventory_at(lifecycle, end_at)
    closed_by_end = all(abs(value) <= EPSILON for value in inventory_at_end.values())
    if closed_by_end and lifecycle.closed_at is not None and lifecycle.closed_at < end_at:
        return "closed"
    if expiration_closed_at(lifecycle, end_at) is not None:
        return "closed"
    if (
        lifecycle.is_closed
        and lifecycle.closed_at is not None
        and lookahead_end_at is not None
        and lifecycle.closed_at <= lookahead_end_at
    ):
        return "closed_after_period"
    if _has_reduction_through(lifecycle, lookahead_end_at or end_at):
        return "partial"
    if any(_option_symbol_expired(symbol, end_at) for symbol in inventory_at_end):
        return "partial"
    return "open"


def expiration_closed_at(lifecycle: Lifecycle, at: datetime) -> datetime | None:
    """Return the effective close when every remaining leg expired by ``at``."""

    remaining = inventory_at(lifecycle, at)
    if not remaining:
        return lifecycle.closed_at
    asset_types = {item.symbol: item.asset_type for item in lifecycle.executions}
    expirations = []
    for symbol in remaining:
        if asset_types.get(symbol) != "OPTION":
            return None
        try:
            expiration = parse_occ_symbol(symbol).expiration
        except ValueError:
            return None
        if expiration >= at.astimezone(timezone.utc).date():
            return None
        expirations.append(expiration)
    return datetime.combine(max(expirations) + timedelta(days=1), time.min, tzinfo=timezone.utc)


def _option_symbol_expired(symbol: str, at: datetime) -> bool:
    try:
        return parse_occ_symbol(symbol).expiration < at.astimezone(timezone.utc).date()
    except ValueError:
        return False


def inventory_at(lifecycle: Lifecycle, at: datetime) -> dict[str, float]:
    inventory: dict[str, float] = {}
    for item in lifecycle.executions:
        # Inventory uses an exclusive upper boundary for consistent period accounting.
        if item.timestamp >= at:
            continue
        inventory[item.symbol] = inventory.get(item.symbol, 0.0) + item.signed_quantity
    return {key: value for key, value in inventory.items() if abs(value) > EPSILON}


def recognize_structure(lifecycle: Lifecycle) -> str:
    executions = lifecycle.executions
    asset_types = {item.asset_type for item in executions}
    option_symbols = sorted({item.symbol for item in executions if item.asset_type == "OPTION"})
    has_equity = bool(asset_types & {"EQUITY", "ETF"})
    parsed = [_safe_occ(symbol) for symbol in option_symbols]
    occs = [item for item in parsed if item is not None]
    opening_sign = _opening_signs(lifecycle)
    root = lifecycle.underlying_symbol
    def strategy(name: str) -> str:
        return name.title()
    def option_name(option: OccSymbol) -> str:
        strike = f"{option.strike.normalize():f}".rstrip("0").rstrip(".")
        return (
            f"{option.root} {option.expiration.strftime('%d %b %y').upper()} "
            f"{'C' if option.put_call == 'CALL' else 'P'} {strike}"
        )
    if not occs and "EQUITY" not in asset_types:
        return f"{root} {strategy(lifecycle.instrument_type)}"
    if has_equity and not occs:
        return root
    if len(occs) == 1 and not has_equity:
        return option_name(occs[0])
    if has_equity and occs:
        short_calls = [item for item in occs if item.put_call == "CALL" and opening_sign.get(item.canonical, 0) < 0]
        long_puts = [item for item in occs if item.put_call == "PUT" and opening_sign.get(item.canonical, 0) > 0]
        if short_calls and long_puts:
            return f"{root} Collar"
        if short_calls:
            return f"{root} Covered Call"
        if long_puts:
            return f"{root} Protective Put"
    # Recognize option structures from opening signs, strikes, and expirations.
    if len(occs) == 2:
        expiries = {item.expiration for item in occs}
        strikes = {item.strike for item in occs}
        sides = {item.put_call for item in occs}
        if len(expiries) == 1 and len(sides) == 1 and len(strikes) == 2:
            return f"{root} Vertical"
        if len(expiries) == 2 and len(sides) == 1 and len(strikes) == 1:
            return f"{root} Calendar"
        if len(expiries) == 2 and len(sides) == 1:
            return f"{root} Diagonal"
        if len(expiries) == 1 and sides == {"CALL", "PUT"}:
            return f"{root} " + ("Straddle" if len(strikes) == 1 else "Strangle")
    if len(occs) == 3 and len({item.expiration for item in occs}) == 1:
        if len({item.strike for item in occs}) == 3:
            return f"{root} Butterfly"
    if len(occs) == 4:
        expiries = {item.expiration for item in occs}
        sides = [item.put_call for item in occs]
        if len(expiries) == 1 and sides.count("CALL") == 2 and sides.count("PUT") == 2:
            return f"{root} Iron Condor"
        if len(expiries) == 2:
            return f"{root} Double Calendar"
    return f"{root} Option Package"


def _cash_allocations(
    parent: TransactionModel,
    items: Sequence[TransactionItemModel],
    instrument_terms: Sequence[InstrumentTerms],
) -> list[float | None]:
    explicit = [_finite(item.amount) for item in items]
    if len(items) == 1:
        parent_amount = _finite(parent.amount)
        if parent_amount is not None:
            return [parent_amount]
        net = _finite(parent.net_amount)
        fee = -abs(_finite(parent.fees) or 0.0)
        if net is not None:
            return [net - fee]
    output: list[float | None] = []
    for index, (item, item_amount) in enumerate(zip(items, explicit)):
        if item_amount is not None:
            output.append(item_amount)
            continue
        quantity = _finite(item.quantity)
        price = _finite(item.price)
        if quantity is None or price is None:
            output.append(None)
            continue
        signed = _signed_quantity(quantity, item.instruction)
        multiplier = instrument_terms[index].price_multiplier
        output.append(-signed * price * multiplier if multiplier is not None else None)
    return output


def _execution_terms(
    item: TransactionItemModel,
    parent: TransactionModel | None,
    supplied: Mapping[str, InstrumentTerms | Mapping[str, Any]],
    context: Mapping[str, Any],
) -> InstrumentTerms:
    symbol = str(item.symbol or (parent.symbol if parent is not None else "") or "").strip().upper()
    asset_type = str(item.asset_type or (parent.asset_type if parent is not None else "") or "")
    if symbol.startswith("CURRENCY_"):
        asset_type = "CURRENCY"
    try:
        canonical = parse_occ_symbol(symbol).canonical
    except ValueError:
        canonical = symbol.removeprefix("/") if "FUTURE" in asset_type.upper() else symbol
    raw = supplied.get(symbol) or supplied.get(canonical)
    if isinstance(raw, InstrumentTerms):
        return raw
    return build_instrument_terms(
        symbol,
        broker_type=asset_type,
        option_terms=raw if isinstance(raw, Mapping) and "OPTION" in asset_type.upper() else None,
        futures_spec=raw if isinstance(raw, Mapping) and "FUTURE" in asset_type.upper() else None,
        context=context,
    )


def _parent_groups(executions: Sequence[ExecutionLeg]) -> list[list[ExecutionLeg]]:
    groups: list[list[ExecutionLeg]] = []
    current_key: tuple[str, int, str] | None = None
    for item in sorted(
        executions,
        key=lambda row: (
            row.timestamp,
            row.account_id,
            row.repository_transaction_id,
            row.item_index,
        ),
    ):
        key = (item.account_id, item.repository_transaction_id, item.underlying_symbol)
        if key != current_key:
            groups.append([])
            current_key = key
        groups[-1].append(item)
    return groups


def _select_candidate(
    batch: Sequence[ExecutionLeg],
    candidates: Sequence[Lifecycle],
    cluster_seconds: float,
) -> tuple[Lifecycle | None, str, float]:
    batch_symbols = {item.symbol for item in batch}
    batch_orders = {item.order_id for item in batch if item.order_id}
    overlap = [
        item
        for item in candidates
        if any(abs(item.inventory.get(symbol, 0.0)) > EPSILON for symbol in batch_symbols)
    ]
    if overlap:
        selected = max(
            overlap,
            key=lambda item: (
                sum(abs(item.inventory.get(symbol, 0.0)) for symbol in batch_symbols),
                -item.opened_at.timestamp(),
            ),
        )
        return selected, "explicit_position_effect", 90.0
    common_order = [item for item in candidates if item.order_ids & batch_orders]
    if common_order:
        return min(common_order, key=lambda item: item.opened_at), "common_order_id", 96.0
    explicitly_closing = any(_is_closing(item.position_effect) for item in batch)
    if explicitly_closing:
        return None, "unmatched_explicit_close", 55.0
    if candidates and all(item.asset_type == "OPTION" for item in batch):
        nearest = min(
            candidates,
            key=lambda item: abs((batch[0].timestamp - item.executions[-1].timestamp).total_seconds()),
        )
        seconds = abs((batch[0].timestamp - nearest.executions[-1].timestamp).total_seconds())
        no_orders = not batch_orders and not nearest.order_ids
        if seconds <= cluster_seconds and no_orders:
            return nearest, "time_cluster", 70.0
    if len(batch) > 1:
        return None, "multi_item_parent", 99.0
    if batch_orders:
        return None, "common_order_id", 94.0
    if any(item.position_effect for item in batch):
        return None, "explicit_position_effect", 88.0
    confidence = 90.0 if batch[0].asset_type in {"EQUITY", "ETF"} else 75.0
    return None, "inventory_continuity", confidence


def _new_lifecycle(
    batch: Sequence[ExecutionLeg], reason: str, confidence: float
) -> Lifecycle:
    first = batch[0]
    if any(item.asset_type == "OPTION" for item in batch):
        instrument_type = "Options"
    else:
        instrument_type = {
            "EQUITY": "Equity",
            "ETF": "ETF",
            "MUTUAL_FUND": "Mutual Fund",
            "MUTUAL FUND": "Mutual Fund",
            "CURRENCY": "Currency",
            "CASH_EQUIVALENT": "Cash Equivalent",
        }.get(first.asset_type, first.asset_type.replace("_", " ").title() or "Other")
    return Lifecycle(
        account_id=first.account_id,
        underlying_symbol=first.underlying_symbol,
        instrument_type=instrument_type,
        confidence_percent=confidence,
        evidence=reason,
    )


def _append_with_flip(
    lifecycle: Lifecycle, execution: ExecutionLeg
) -> ExecutionLeg | None:
    current = lifecycle.inventory.get(execution.symbol, 0.0)
    quantity = execution.signed_quantity
    crosses = current * quantity < 0 and abs(quantity) > abs(current) + EPSILON
    if not crosses:
        role = "closing" if current * quantity < 0 else "opening"
        _append_without_split(lifecycle, replace(execution, role=role))
        return None
    closing_quantity = -current
    ratio = abs(closing_quantity / quantity)
    closing = replace(
        execution,
        signed_quantity=closing_quantity,
        fees_usd=execution.fees_usd * ratio,
        fee_components_usd={
            fee_type: value * ratio
            for fee_type, value in execution.fee_components_usd.items()
        },
        cash_effect_usd=(
            execution.cash_effect_usd * ratio
            if execution.cash_effect_usd is not None
            else None
        ),
        role="closing",
    )
    overflow = replace(
        execution,
        signed_quantity=quantity - closing_quantity,
        fees_usd=execution.fees_usd - closing.fees_usd,
        fee_components_usd={
            fee_type: value - closing.fee_components_usd.get(fee_type, 0.0)
            for fee_type, value in execution.fee_components_usd.items()
        },
        cash_effect_usd=(
            execution.cash_effect_usd - closing.cash_effect_usd
            if execution.cash_effect_usd is not None
            and closing.cash_effect_usd is not None
            else None
        ),
        role="opening",
    )
    _append_without_split(lifecycle, closing)
    return overflow


def _append_without_split(lifecycle: Lifecycle, execution: ExecutionLeg) -> None:
    before = lifecycle.inventory.get(execution.symbol, 0.0)
    after = before + execution.signed_quantity
    if abs(after) <= EPSILON:
        after = 0.0
    lifecycle.executions.append(execution)
    lifecycle.inventory[execution.symbol] = after
    lifecycle.peak_inventory[execution.symbol] = max(
        abs(after), lifecycle.peak_inventory.get(execution.symbol, 0.0)
    )
    if abs(before) <= EPSILON and abs(after) > EPSILON:
        lifecycle.original_symbols.add(execution.symbol)
    if execution.order_id:
        lifecycle.order_ids.add(execution.order_id)


def _has_reduction_through(lifecycle: Lifecycle, through: datetime) -> bool:
    inventory: dict[str, float] = {}
    peak: dict[str, float] = {}
    reduced = False
    for item in lifecycle.executions:
        if item.timestamp > through:
            break
        before = inventory.get(item.symbol, 0.0)
        after = before + item.signed_quantity
        if before * item.signed_quantity < 0 and abs(after) + EPSILON < abs(before):
            reduced = True
        inventory[item.symbol] = after
        peak[item.symbol] = max(peak.get(item.symbol, 0.0), abs(after))
    return reduced and any(abs(value) > EPSILON for value in inventory.values())


def _opening_signs(lifecycle: Lifecycle) -> dict[str, float]:
    output: dict[str, float] = {}
    for item in lifecycle.executions:
        if item.symbol not in output and item.role == "opening":
            output[item.symbol] = item.signed_quantity
    return output


def _asset_and_occ(asset_type: str | None, symbol: str) -> tuple[str, OccSymbol | None]:
    normalized_type = str(asset_type or "").strip().upper()
    if symbol.strip().upper().startswith("CURRENCY_"):
        return "CURRENCY", None
    occ = _safe_occ(symbol)
    if occ is not None or "OPTION" in normalized_type:
        return "OPTION", occ
    if normalized_type in {"ETF", "EXCHANGE_TRADED_FUND", "COLLECTIVE_INVESTMENT"}:
        return "ETF", None
    if normalized_type in {"EQUITY", "STOCK", "COMMON_STOCK", "ADR"}:
        return "EQUITY", None
    return normalized_type or "UNKNOWN", None


def _safe_occ(symbol: str) -> OccSymbol | None:
    try:
        return parse_occ_symbol(symbol)
    except ValueError:
        return None


def _signed_quantity(
    quantity: float,
    instruction: str | None,
    *,
    cash_effect_usd: float | None = None,
) -> float:
    instruction_text = str(instruction or "").strip().upper().replace(" ", "_")
    if "SELL" in instruction_text:
        return -abs(quantity)
    if "BUY" in instruction_text:
        return abs(quantity)
    if cash_effect_usd is not None and abs(cash_effect_usd) > EPSILON:
        return -abs(quantity) if cash_effect_usd > 0 else abs(quantity)
    return quantity


def _is_closing(position_effect: str | None) -> bool:
    return "CLOS" in str(position_effect or "").upper()


def _evidence_rank(value: str) -> int:
    return {
        "multi_item_parent": 4,
        "common_order_id": 3,
        "explicit_position_effect": 2,
        "inventory_continuity": 1,
        "time_cluster": 0,
    }.get(value, 0)


def _finite(value: Decimal | float | int | None) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


__all__ = [
    "EPSILON",
    "ExecutionLeg",
    "Lifecycle",
    "METHODOLOGY_VERSION",
    "allocate_parent_fee_components",
    "allocate_parent_fees",
    "inventory_at",
    "lifecycle_status",
    "normalize_execution_legs",
    "recognize_structure",
    "reconstruct_lifecycles",
]
