# P&L lifecycle methodology (`pnl_lifecycle_instrument_terms_v2`)

## Evidence and grouping

Executions are normalized from the MySQL parent transaction and every child
item. Account ID and exact security identifier are hard boundaries: positions
from different accounts are never merged, and close timestamps alone never
join different underlyings. Grouping evidence is considered in this order:

1. a multi-item parent/package;
2. a common nonblank broker order ID;
3. explicit opening/closing position effect and exact-symbol inventory;
4. a configured narrow same-account, same-underlying time cluster.

The lowest-quality evidence used by a lifecycle determines its displayed match
confidence. Low-confidence inferred packages produce a response warning. A
same-order close/open roll remains one lifecycle with a new exact-symbol leg
epoch. Stable trade IDs are SHA-256 fingerprints of the methodology version,
account, earliest immutable repository ID, and opening evidence.

## Inventory, lots, and fees

Buy quantities are positive and sell quantities are negative. Equities and ETFs
start a lifecycle when signed inventory leaves zero and close when it returns to
zero. An execution that flips through zero is split proportionally into the old
lifecycle's closing quantity and a new lifecycle's opening quantity. Option
legs are keyed by canonical OCC symbol and may reduce or close independently.
When a broker execution omits its Buy/Sell instruction, a nonzero signed cash
effect resolves the side: positive cash is a sell and negative cash is a buy.
Transaction-ledger quantities preserve this signed convention.

Lots are matched FIFO unless a future explicit broker lot identifier is added.
Parent fees use a signed expense convention and are allocated by absolute leg
notional (quantity when a price is absent); the final allocation receives the
floating remainder so the parent total is conserved exactly. Parent cash is
never added when child cash has already been used.

## Period and lifecycle P&L

For each trade and the aggregate, the period calculation is:

```text
ending signed market value
- starting signed market value
+ signed execution cash during [start_at, end_at)
+ signed fees/carry during [start_at, end_at)
```

The period decomposition seeds pre-period FIFO lots at the starting boundary
mark, so realized plus unrealized plus fees/carry reconciles to period P&L.
Lifecycle P&L is complete signed cash after closure. For an open lifecycle it is
signed cash plus the exact current value from the immutable requested snapshot.
Current RefData may only be used as an explicit fallback; a historical date is
never valued with a current mark.

Daily P&L applies each day's executions and fees to successive signed marked
values. Its final cumulative value is forced to the same rounded aggregate used
by the summary and must reconcile within one cent. Capital at risk is returned
for long equity/ETF positions, debit structures, and deterministically bounded
two-leg same-expiry spreads. It is null for uncovered short equity/options or
indeterminate structures.

Analyzer replay includes an intraday execution point only when the selection is
flat immediately afterward or every still-active exact contract has a fill at
that instant. This prevents a partial multi-leg close from combining its fresh
fill with carried prior-session marks on the remaining legs; the next daily
boundary incorporates the execution using a coherent marked portfolio.

## Market provenance and replay

Equity and ETF historical boundaries use published RefData daily bars. Exact
historical option closes use the Risk-GUI hybrid contract: nightly MySQL marks
captured while the contract was a broker position are authoritative, and the
typed RefData market-history contract fills missing sessions. P&L requests
include a ten-day lookback so the preceding completed session can value weekend
and holiday boundaries.
If a required boundary mark is genuinely missing, the trade query excludes and
reports only the affected lifecycle while calculating all fully priced trades.
Historical option Greeks remain unavailable. Analyzer requests retain valid
transaction chronology and cash-derived lifecycle P&L, set historical option
availability false, and omit unsupported paths/scenarios.

Replay points use only observations available at or before that timestamp.
Exit rules are evaluated sequentially from prior path state. Historical midpoint
fills are labeled non-executable; touch models cross the adverse side and the
touch-plus-tick model adds one configured valid tick.

## Modeled P&L attribution

The analyzer returns the same ordered factor set for every selection:
underlying movement, volatility, skew, term structure, time decay, rates,
dividends, commission and fees, and model residual. Equities, ETFs, and futures
put their exact signed marked price movement in underlying movement. Persisted
fees are kept as their exact signed execution total.

For option quantities held between successive exact daily marks, the analyzer
loads compact MySQL chain snapshots by OCC root and expiration. Snapshot choice
is causal: a capture after the boundary is never used. Valid noncrossed chain
prices are inverted with the installed Tian solver, fitted with bounded linear
interpolation in forward log-moneyness, and separated into parallel ATM level,
expiry term deviation, and strike skew. A one-expiry surface has zero term
structure by definition. Delayed, truncated, or partial captures are used only
when the target remains inside a valid local interpolation range and are
reported in data-quality warnings.

Each fully supported interval uses forward and reverse full-revaluation
waterfalls over time, spot, rates, dividends, volatility, term structure, and
skew. The two incremental results are averaged factor by factor. Historical
risk-free curves and trailing cash-dividend yields are resolved and cached at
calculation time; they are not stored in surface tables.

Fallback is local to each exact-contract interval and is independent of surface
availability. Consecutive exact observations are paired by causal session while
the signed account/lifecycle inventory is nonzero; each contract is calculated
separately and scaled by its multiplier once. The exact mark change remains the
accounting source of truth.

The canonical interval record and fallback decision are owned by
`historical_option_interval_resolver_v4`. P&L attribution supplies any available
surface or full contract-IV decomposition to that resolver; current-position
tail, factor, cluster, target, hedge-candidate, and proposal history consume the
same interval records through `current_portfolio_historical_revaluation_v4`.
The interval publishes endpoint marks, sources and causal observation times,
signed exact P&L, premium multiplier, terms version, method, missing inputs,
warnings, and a one-cent reconciliation check. A single exact mark can value
one point but is explicitly insufficient for an interval.

The no-surface tiers are:

1. **Contract-IV symmetric revaluation.** Invert each endpoint's exact mark with
   the installed Tian solver, using persisted endpoint spot, historical rate and
   dividend inputs, and captured IV only as a hint. Average forward and reverse
   waterfalls over time, spot, rates, dividends, and contract IV. Contract-IV
   change is displayed as Volatility; Skew and Term structure are zero.
2. **Captured-Greek bridge.** If inversion or fixed-IV repricing fails, use
   endpoint-average delta, theta, vega-per-1%-point, and rho-per-1%-point where
   available. When only the starting delta is usable, its starting gamma is
   included in Underlying movement. Decimal IV/rate changes are multiplied by
   100 before applying vega/rho, and theta is already per calendar day.
3. **Exact observed repricing.** If neither model tier has sufficient causal
   inputs, preserve that interval's exact mark change in
   `Option repricing residual` and emit a stable
   reason warning. It is not mislabeled as model error. Exact marks outside the
   contract's intrinsic and upper no-arbitrage bounds are identified explicitly
   and are not forced into a market factor without usable captured Greeks.

The stable interval method identifiers are `surface`, `contract_iv`,
`bounded_greek_bridge`, `observed_price_only`, `single_mark_only`, and
`insufficient`. For an exact-marks-only interval, the exact change is placed in
Observed option repricing, the generic residual remains zero, and driver-detail
missing fields remain visible.

If one historical rate or dividend endpoint is unavailable, the usable value is
held constant and the corresponding factor is zero. If neither is available,
only the server's explicitly configured model default may be held constant.
Missing persisted spot may be filled only from a causal RefData close for the
same completed session—never a future close. The bar's published timestamp is
used directly; a date label is never converted to a hardcoded close time.
For the portfolio-snapshot endpoint, the snapshot's own underlying quote is
used when present. Opening-fill to first-mark, last-mark to closing-fill, and
same-day execution edges use the exact option fill and the last completed
one-minute underlying bar at the execution. If that causal bar is unavailable,
only the unsupported edge stays in Model residual with an explicit warning;
intervening daily interval attribution is unaffected.
At and after contract expiration, revaluation uses the option's terminal
intrinsic value. This lets the final spot and time-to-expiry movements remain
attributable even though terminal implied volatility is undefined; any
difference between the exact broker mark and intrinsic value remains residual.

`data_quality.attribution_method` is `surface_revaluation` only when every
eligible option interval has two usable endpoints, `price_history_fallback`
when any interval degrades (including same-day option trades with no eligible
daily interval), and `mark_revaluation` when no option exposure needs a
surface. `surface_coverage_percent` counts eligible daily option intervals and
is null for mark-only selections. Exact lifecycle P&L remains authoritative:
the residual is computed from the unrounded factors, and the final rounding
difference is applied only to Model residual so displayed rows reconcile within
one cent.
