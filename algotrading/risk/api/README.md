# Algotrading Risk API

The Risk API is the sole application backend for the Portfolio Risk Workbench.
It preserves the original read-only account/position routes and adds coherent,
snapshot-bound portfolio analytics. It cannot place, cancel, or modify orders.

## Start

```bash
export ALGO_HOME=/path/to/algo_pkg
python -m algotrading.risk.api --environment risk
```

The default address is `127.0.0.1:8789`; OpenAPI documentation is at `/docs`.
The Reference Data API is expected at `risk.refdata.base_url` (default
`http://127.0.0.1:8787`).

## Portfolio endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/v1/portfolio/snapshot` | Coherent snapshot; `refresh=true` is the only portfolio refresh action |
| GET | `/api/v1/portfolio/dashboard` | Snapshot-bound dashboard bundle |
| GET | `/api/v1/portfolio/exposures` | Directional/sector exposure and factor availability |
| GET/POST | `/api/v1/portfolio/scenarios`, `/api/v1/portfolio/scenarios/evaluate` | Scenario library and anchored Tian evaluation |
| GET | `/api/v1/portfolio/options-risk` | Held-option aggregates in explicit Greek units |
| GET | `/api/v1/portfolio/risk-calendar` | Current option-expiration exposure |
| GET | `/api/v1/portfolio/tail-risk` | Weighted historical VaR/ES, or explicit insufficient-data response |
| GET | `/api/v1/portfolio/clusters` | Deterministic cluster result/availability |
| GET | `/api/v1/reference/companies/{symbol}` | Desktop-safe proxy to database-first RefData company information |
| GET | `/api/v1/portfolio/pnl-explain` | Refresh-anchor marked-value attribution with labeled broker-day P&L and coverage |
| POST | `/api/v1/portfolio/pnl/trades/query` | Snapshot-bound SQL trade lifecycle reconstruction and period P&L |
| POST | `/api/v1/portfolio/pnl/trades/analyze` | Read-only lifecycle replay and counterfactual analysis |
| POST | `/api/v1/risk-gui/options/market-history/query` | Risk-GUI-only SQL-first exact option history with RefData gap fill |
| POST | `/api/v1/risk-gui/options/contract-terms/query` | Risk-GUI-only exact option terms proxy; authoritative expiry/deliverable/multiplier |
| POST | `/api/v1/options/iv-greeks` | Direct vectorized Tian IV/Greek calculation |
| POST | `/api/v1/hedge-lab/options/chain` | Snapshot-bound, bounded, analysis-only chain |
| POST | `/api/v1/hedge-lab/targets/assess` | Selected-position scenario, historical tail, factor, Greek, and exact terminal-payoff assessment |
| POST | `/api/v1/hedge-lab/candidates/query` | Deterministic, bounded research candidate ranking with directly reusable marked legs |
| POST | `/api/v1/hedge-lab/proposals/evaluate` | Ephemeral target and portfolio before/after revaluation; hidden legs are audit-only |
| GET | `/api/v1/data-health` | Provenance, coverage, capability, cache, and transaction diagnostics |
| GET/POST/PATCH | `/api/v1/alerts...` | Persistent alert evaluation and lifecycle |
| GET/PUT/DELETE | `/api/v1/strategy-groups...` | Persistent user strategy groups |
| GET/PATCH | `/api/v1/risk-configuration` | Non-secret risk-method configuration |
| GET | `/api/v1/accounts/{account_id}/transactions[/status]` | MySQL-only reads; never calls the broker |

ETF sector labels describe fund exposure. Verified fund labels and declared
sector funds take precedence; other ETF positions use the RefData fund-exposure
endpoint for a conservative single-bucket classification. The endpoint uses
fund information and sector weights, with a 24-hour cache. Ambiguous or
unavailable exposure remains `Unclassified` with a position data-quality
warning; issuer-company sectors are not substituted. VaR continues to use
SQL-first daily bars, with bounded remote repair for configured symbols.

Every portfolio analysis carries a snapshot/configuration ID, timestamps,
methodology version, and data quality. Account scope accepts repeated or
comma-separated IDs, or `all`.
Exposure and cluster reads also accept `asset_scope=all|equities|options|futures`.

### Hedge Lab

All Hedge Lab requests bind both the account scope and
`portfolio_snapshot_id`. Target position identifiers must resolve completely in
that scope; stale or cross-scope snapshots return 409, while unknown positions
return 422. Requests and objective-specific parameter dictionaries are strict
and bounded. Responses are always marked `analysis_only: true` and never expose
an order payload.

Target assessment uses the same weighted historical return, ridge factor, Tian
scenario, and explicit-unit Greek methodologies as the portfolio routes, but
recomputes them on the selected position set rather than allocating portfolio
risk pro rata. Candidate marks and multipliers must be published by RefData;
option Greeks are normalized by Risk Tian. Stale, crossed, incomplete, or
unpriced instruments are excluded. Ranking version `hedge_candidate_full_revaluation_rank_v5`
publishes its weights in each response and is deterministic for fixed inputs.

Proposal evaluation builds immutable request-local positions and evaluates the
selected target and full resulting portfolio. It runs the primary objective,
historical VaR/ES, all factors and Greeks, the scenario library plus a joint
stress, and exact piecewise terminal payoff where defined. Hidden legs affect no
calculation and are echoed in `hidden_legs`. Margin and portfolio liquidity are
reported as structured `insufficient_data` until a reliable model is available.

### P&L lifecycle endpoints

Both P&L endpoints are read-only and snapshot-bound. Transaction evidence is
loaded directly from the Risk MySQL `risk_transactions` and
`risk_transaction_items` tables; these request paths never call the broker.
`end_at` is exclusive. With look-through disabled, the trade query removes
every lifecycle not fully closed before `end_at`. With look-through enabled,
it includes later closing evidence through `lookahead_end_at`, while keeping
period P&L separate from lifecycle P&L.

The methodology version is `pnl_lifecycle_instrument_terms_v2`. It uses account-isolated exact
security inventories, deterministic grouping evidence, FIFO lots, signed cash
flows, and exact parent-fee conservation. Historical equity/ETF boundaries use
RefData daily bars. Historical option boundaries first use nightly marks
captured while the exact contract was a broker position. Missing sessions use
RefData's exact-contract daily OHLC service, with a ten-day request lookback so
the last completed session can value a weekend or holiday boundary. A genuinely
missing required mark excludes only the affected lifecycle from the trade query;
the response reports it in `data.excluded_trades`, and no current or zero price is
substituted. A market-history service failure still returns
`PNL_MARK_HISTORY_UNAVAILABLE`. The analyzer retains valid transaction chronology
and fill-derived lifecycle results if historical data is unavailable.

Additional sanitized errors are `PNL_SELECTION_NOT_FOUND` (404) and
`PNL_RECONSTRUCTION_LIMIT_EXCEEDED` (422). Snapshot errors and request IDs use
the same envelopes as every existing Risk route. See
[`../pnl/METHODOLOGY.md`](../pnl/METHODOLOGY.md) for calculation details.

## Refresh, cache, and state

There is no scheduler or polling path. A portfolio provider attempt requires
`refresh=true`; attempts for the same account scope are separated by at least
60 seconds, including failures. Snapshots and bounded latest chain results use
SQLite WAL at `${XDG_CACHE_HOME:-~/.cache}/algotrading/risk-api-cache.sqlite3`
unless `risk.cache.path` is configured. Alerts and strategy groups use the
separate `risk.state.path` state database.

Before the regular session opens, and whenever that session is closed, current
portfolio risk is anchored to the latest calendar-confirmed completed session.
Equities use the completed daily close. Held options use the provider's exact
contract `closePrice`, timestamped at that completed session's close; premarket
bid/ask values, zero timestamps, underlying prices, IV, and Greeks are not mixed
into the EOD snapshot. Tian recalculates option IV and Greeks from the aligned
option and underlying closing marks.

| Key | Default |
| --- | --- |
| `risk.api.account_cache_ttl_seconds` | `60` |
| `risk.refdata.risk_free_curve_cache_seconds` | `1800`; the Risk API caches each as-of curve locally for 30 minutes and batch-loads missing historical dates in one RefData request |
| `risk.history.risk_free_curve_prefetch_workers` | `8`; compatibility fallback used only when RefData does not yet publish the historical-curve batch contract |
| `risk.option_history.adaptive_base_steps` / `adaptive_max_steps` | `128` / `256`; historical IV shock estimation uses a bounded Tian tree while current valuation retains the stricter global adaptive limits |
| `risk.refresh.remote_attempt_floor_seconds` | `60` (minimum enforced) |
| `risk.snapshot.ttl_seconds` / `risk.snapshot.hard_max_age_seconds` | `60` / `300` |
| `risk.instrument_terms.cache_age_seconds` / `maximum_option_contracts_per_request` | `86400` / `500` |
| `risk.instrument_terms.allow_standard_occ_assumption` / `standard_option_timezone` / `standard_option_expiration_local_time` | `true` / `America/New_York` / `16:00`; assumptions remain partial-quality and cannot certify unsupported contracts |
| `risk.instrument_terms.static_futures_specifications` | Versioned supported-root table; multiplier/tick consistency validated at startup |
| `risk.freshness.intraday_max_age_seconds` / `completed_session_eod_allowed` | `900` / `true` |
| `risk.freshness.low_delta_option_threshold` / `low_delta_option_intraday_max_age_seconds` | `0.10` / `14400`; options at or below 10 absolute delta use the four-hour limit |
| `risk.freshness.future_timestamp_tolerance_seconds` | `5`; permits only bounded clock/acquisition skew while assembling a snapshot |
| `risk.freshness.mutual_fund_nav_max_session_lag` | `1`; the provider's latest returned `mark` is accepted as NAV for at most one completed exchange-session lag |
| `risk.freshness.regular_session_open_local_time` / `close_local_time` | `09:30` / `16:00` |
| `risk.freshness.calendar_version` / `calendars` | Versioned RefData calendar facts; startup requires `XNYS` and `CME_GLOBEX` |
| `risk.pricing.adaptive_base_steps` / `adaptive_max_steps` | `256` / `2048` |
| `risk.pricing.price_absolute_tolerance` / `price_relative_tolerance` | `0.01` / `0.005` |
| `risk.pricing.delta_absolute_tolerance` / `greek_relative_tolerance` / `greek_absolute_floor` | `0.01` / `0.05` / `1e-6` |
| `risk.option_history.minimum_contract_iv_observations` | `126`; the fixed surface → contract-IV → bounded-Greek → observed-price hierarchy is not configurable |
| `risk.history.allow_configured_economic_proxies` / `economic_proxies` | `false` / empty; mappings require beta/provenance |
| `risk.history.allow_fixed_current_iv_fallback` / `fixed_current_iv_greek_bridge_maximum_spot_return` | `true` / `0.20`; when an exact historical IV shock is absent, one-day returns within the bound use a current-mark-anchored fixed-IV delta/gamma/theta bridge and warn; observed volatility shocks retain full option repricing |
| `risk.history.remote_daily_bar_symbols` | `SHY`; bounded RefData remote repair is permitted when this required factor series is absent from SQL history |
| `risk.scenarios.maximum_absolute_equity_shock` | `0.50` |
| `risk.scenarios.etf_response_minimum_observations` / `etf_response_ridge_alpha` / `etf_response_minimum_r_squared` | `126` / `0.001` / `0.20`; held ETFs are regressed on SPY, QQQ, and sector ETF histories, and inadequate fits remain explicitly unmodeled |
| `risk.calculations.maximum_excluded_marked_value_fraction` | `0.03`; calculations may omit bounded problem positions only when their aggregate gross marked value is strictly below 3% of NAV and must warn |
| `risk.scenarios.factor_mappings` / `economic_instrument_mappings` | Empty; explicit mappings are required for non-broad-equity economics |
| `risk.factors.ridge_alpha` / `minimum_observations` | `0.001` / `126` |
| `risk.clusters.source_weights` / `minimum_pairwise_overlap` | history `0.5`, factor `0.2`, scenario `0.3` / `60` |
| `risk.hedge_lab.maximum_overhedge_fraction` / `maximum_candidate_quantity` | `0.25` / `100` |
| `risk.options.greek_source` / `risk.options.requested_dtype` | `recalculated` / `float64` |
| `risk.options.cache_ttl_seconds` | `60` |
| `risk.options.tian.escrowed_dividend_mode_enabled` | `true`; known future ordinary cash dividends are deducted at present value from Tian's spot input and disclosed in solver diagnostics |
| `risk.rates.max_age_days` | `5` |
| `risk.hedge_lab.maximum_quote_age_seconds` | `900` |
| `risk.hedge_lab.maximum_mixed_quote_skew_seconds` | `60` |
| `risk.hedge_lab.maximum_equity_symbols` / `price_query_workers` | `12` / `4` |
| `risk.hedge_lab.futures_contracts` | Unset: derive bounded market/rates/energy/currency contracts from CME lifecycle dates |
| `risk.hedge_lab.allowed_futures_roots` | `ES,MES,NQ,MNQ,RTY,M2K`; roots must exist in the futures terms registry |
| `risk.hedge_lab.futures_roll_buffer_days` | `7` |
| `risk.cache.max_entries` | `5000` |
| `risk.pnl.max_accounts` / `risk.pnl.max_filters` | `20` / `100` |
| `risk.pnl.max_transaction_rows` | `50000` |
| `risk.pnl.max_period_days` / `risk.pnl.max_lookahead_days` | `366` / `366` |
| `risk.pnl.max_history_lookback_days` | `7305` |
| `risk.pnl.max_option_symbols` / `risk.pnl.max_analyzer_path_points` | `500` / `1000` |
| `risk.option_prices.max_surface_snapshots` / `max_surface_points` | `2000` / `100000` |
| `risk.option_prices.surface_point_batch_size` | `500` |
| `risk.option_prices.market_timezone` | `America/New_York` |
| `risk.option_prices.max_history_days` | `376` |
| `risk.option_prices.backfill.timeout_seconds` / `max_attempts` / `max_requests_per_minute` | `25` / `5` / `5` |

Secrets remain in server configuration. Responses and logs do not expose
credentials, raw provider payloads, or full account payloads.

The analyzer endpoint augments `data_quality` with an attribution method and
daily surface coverage. Compact MySQL surfaces are optional: missing, partial,
malformed, over-limit, or temporarily unavailable surface data downgrades the
affected option intervals to exact-price attribution and warnings. It does not
change the response envelope or make pre-surface trades fail. Surface fitting
never calls a broker or option-chain provider from the analyzer request path;
only the existing bounded RefData inputs for exact-mark gaps, rates, dividends,
and equity marks remain available.

## Semantic calculation contract

Normalized positions expose canonical `instrument_terms`, signed liquidation
value, signed notional exposure, and delta exposure as separate fields.
Futures notional is never included in NAV. A future without a resolved
multiplier/tick specification and an option without usable contract terms fail
closed for term-dependent calculations. The versioned static futures table is
an audited fallback only for its listed roots; it validates
`minimum_tick * price_multiplier == tick_value` during startup.

Option validity is field-specific. An exact authoritative premium multiplier
can support observed mark-to-mark history even when exercise style, settlement,
subtype, expiry session, or payoff deliverable remains partial. Those missing
fields still fail closed for current Greeks, model repricing, expiry lifecycle,
payoff, and deliverable-notional denominators; unknown exercise style is never
silently mapped to American.

Portfolio responses expose field-specific `risk_coverage` for instrument terms,
market value, delta, gamma, vega, theta, and rho. A missing derivative term makes
the affected dollar denominator and percentage null; it cannot disappear from
the denominator. Nested warnings are merged into the outer analysis quality.

The option-history resolver is SQL-first and causal. Its fallback sequence is
surface revaluation, exact-contract mark history with endpoint IV inversion,
captured-Greek bridge, exact observed repricing without driver detail, then
insufficient data. Retrospective mark-to-mark P&L always reconciles to exact
marks. Tail/factor/cluster history uses current-position dollar revaluation,
never automatic SPY substitution or zero-filled missing option observations.
Weighted ES consumes exactly `1 - confidence` mass, including a fractional
boundary observation; confidence must be less than one.

Historical interval resolution is versioned as
`historical_option_interval_resolver_v4` and its cross-consumer matrix as
`current_portfolio_historical_revaluation_v4`. A completed interval's
`exact_position_pnl_usd` is signed P&L (losses are negative). Tail responses
convert this into `loss_usd` and expose `var_usd` and
`expected_shortfall_usd` as nonnegative loss amounts; signed risk-unit ES
contributions reconcile to the positive portfolio ES under the contribution
contract returned by the endpoint.

Economic scenarios use typed equity, sector, exact-underlying, futures-root,
commodity, FX, rate, IV, and horizon shocks. Rate futures require validated
DV01 for rate scenarios. Custom scenario requests expose audited
`instantaneous_and_held`, `log_linear_to_horizon`, and causal explicit-dated
paths. An omitted nonlinear, unbounded, futures, or economically unknown
position makes the aggregate result insufficient regardless of position-count
coverage. Payoff output uses
`exact_payoff_certification_v3` statuses (`exact_bounded`, `unbounded`,
`bounded_stress_only`, `insufficient_data`). Hedge discovery enumerates every
bounded admissible integer quantity, fully re-evaluates it against the selected
objective, and retains only strictly positive, completely covered objective
improvements. Convex singles/spreads and covered collars participate when live,
fully supported option terms permit them. Output is Pareto-filtered research
only and always has `analysis_only=true` and `recommendation=false`.

Unsupported futures options, unsupported deliverable property, pre-expiry
European repricing without a European engine, dividend-sensitive American
calls that require discrete-dividend early-exercise handling, and
expiry-crossing scenarios without a determining path rule all fail closed with
null results, missing IDs, and stable warnings. Exact observed option marks may
still support historical P&L even when payoff or forward-scenario certification
is unavailable.

## Transaction collector

Initialize MySQL using `algotrading/risk/sql/mysql/README.md`, then run:

```bash
python -m algotrading.risk.transactions --account ENCRYPTED_ACCOUNT_ID
```

The collector uses `TDABrokerInterface.get_transactions`, validates and
normalizes each broker response, and persists it into the durable Risk tables.

## Nightly option-position prices

After applying `002_create_risk_option_prices.sql`, schedule this after the
market session:

```bash
python -m algotrading.risk.option_prices --environment refdata
```

The process constructs `TDABrokerInterface` directly (the Risk API need not be
running), calls `get_accounts()` and the same strict account normalizer used by
the server, and queries each selected account's `TRADE` transactions for the
snapshot's current market day. The quote universe is the union of open exact
OCC positions and option contracts found in transaction parent/items, so a
contract closed that day is still captured; transaction-only contracts are
stored with signed quantity zero. The union is batched through
`TDAReferenceData.get_option_quotes()` and written to MySQL directly. It returns
a non-zero status if any included contract lacks a usable mark. `--dry-run`
exercises the broker/provider path without database writes; repeat `--account`
to limit a manual run. After all processing finishes, the job reports exactly
one Nagios event: `MONITOR_OK` for a complete capture or `MONITOR_CRITICAL` when
the job fails or an included contract has no usable quote. Its configured
`nagios.service_id` is reported with ` - Option Position Prices` appended.

To fill historical sessions once from the durable transaction tables, first
preview the work and then run it without `--dry-run`:

```bash
python -m algotrading.risk.option_prices.backfill \
  --environment refdata --from-date 2024-01-01 --dry-run
python -m algotrading.risk.option_prices.backfill \
  --environment refdata --from-date 2024-01-01
```

The historical command reads all stored `TRADE` transactions before the cutoff
so positions already open on `--from-date` retain the right inventory. It
reconstructs inclusive opening/closing-day windows for each account and exact
OCC contract, requests unadjusted Massive daily aggregates, and stores OHLC,
close-as-mark, and volume. Existing session rows are preserved so a historical
bar cannot erase richer quote or Greek data from a nightly capture. The default
`--through-date` is yesterday in `risk.option_prices.market_timezone`; repeat
`--account` or `--symbol` to narrow a run. `massive_key` is read from the normal
configuration, with `MASSIVE_API_KEY` as an environment fallback. The CLI
enforces Massive's five-requests-per-minute plan limit across normal requests,
pagination, and retries; use `--max-requests-per-minute` only if the plan limit
changes. Before each provider request it checks the stored account/contract
sessions and requests only uncovered weekday windows. Each returned aggregate
batch is committed immediately, so a later invocation skips the completed
sessions after an interruption.

## Boundaries

The API calls only published Reference Data REST contracts. It has no
ClickHouse dependency and never imports or constructs the Options GUI API or
its data pipeline. Only the Tian numerical solver implementation is reused.
