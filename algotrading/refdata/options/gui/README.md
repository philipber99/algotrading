# Options GUI API backend

`algotrading.refdata.options.gui` contains two read-only backend facilities:

- `gui.cache` builds and queries immutable time-of-day Parquet generations.
- `gui.screener` screens Phase 1 verticals, calendars, and diagonals from one
  active, completed ClickHouse snapshot.

The screener never writes results to ClickHouse or the Parquet cache. The API
initializes the cache and screener independently, so historical cache routes
remain available when ClickHouse is unavailable.

## Commands

Install runtime dependencies with:

```bash
python -m pip install -r algotrading/refdata/options/gui/requirements.txt
```

Run the cache-management CLI:

```bash
python -m algotrading.refdata.options.gui.cache --help
python -m algotrading.refdata.options.gui.cache -e refdata build daily --dry-run
python -m algotrading.refdata.options.gui.cache -e refdata validate --active
python -m algotrading.refdata.options.gui.cache -e refdata status --json
python -m algotrading.refdata.options.gui.cache -e refdata cleanup --execute
```

Run the REST service (always one Uvicorn worker):

```bash
python -m algotrading.refdata.options.gui.api \
  --environment refdata --host 0.0.0.0 --port 8788 --log-level info

# Equivalent server entry point:
python -m algotrading.refdata.options.gui --environment refdata

# Atomically update the cache before starting its query engine:
python -m algotrading.refdata.options.gui.api --environment refdata --update-cache
```

The REST CLI retains `--environment`, `--host`, `--port`, `--log-level`,
`--no-access-log`, and `--update-cache`.

## Configuration

Existing deployments keep their working configuration keys unchanged:

- `options_chain.time_of_day_cache.*` controls cache generation and storage.
- `options_chain.time_of_day_api.*` controls the host, port, middleware,
  request limit, result cache, query concurrency, logging, and its independent
  pricing pool.

Screener settings use `options_chain.gui.screener.*`. Safe defaults include:

```ini
options_chain.gui.screener.max_results_per_strategy = 50
options_chain.gui.screener.max_prescore_candidates_per_strategy = 500
options_chain.gui.screener.max_generated_candidates_per_strategy = 5000
options_chain.gui.screener.max_full_scenario_candidates_per_strategy = 100
options_chain.gui.screener.max_concurrent_screens = 2
options_chain.gui.screener.screen_timeout_seconds = 30
options_chain.gui.screener.result_cache_mb = 256
options_chain.gui.screener.simulation_paths = 5000
options_chain.gui.screener.maximum_simulation_paths = 25000
options_chain.gui.screener.spot_grid_points = 81
options_chain.gui.screener.maximum_holding_sessions = 30
options_chain.gui.screener.iv_residual_half_life_sessions = 5
options_chain.gui.screener.maximum_short_itm_fraction = 0.20
options_chain.gui.screener.pricing_method = tian
options_chain.gui.screener.pricing_workers = 0
options_chain.gui.screener.pricing_batch_size = 4096
options_chain.gui.screener.max_pending_pricing_batches = 0
options_chain.gui.screener.pricing_worker_threads = 1
```

All bounds are validated at startup. Requests cannot raise configured result,
path, DTE, holding-horizon, candidate, or grid limits.

`pricing_method` accepts exactly `tian`, `crr`, `crank_nicolson`, or
`configured`. The default deliberately forces Tian trees for every
positive-tenor scenario-priced contract, including contracts marked European.
`configured` restores the exercise-aware global routing and short-tenor
override. This screener-only setting does not alter stored IV or Greeks,
calculation/collector routing, cache contents, or surface fitting. Scenario
model version 2 changes because future scenario prices can therefore differ
from version 1; Tian and Crank–Nicolson should not be assumed numerically
identical.

Zero workers selects a CPU-affinity-aware default of at most eight workers.
Zero pending batches selects twice the resolved worker count. Each spawned
worker owns one solver registry and uses the configured worker thread bound.
The process-pool batch size is an outer bound; Tian's existing `chunk_size`
continues to bound the inner vectorized kernel.

## REST endpoints

Existing paths and response contracts are retained:

```text
GET  /
GET  /health/live
GET  /health/ready
GET  /v1/cache/metadata
GET  /v1/cache/windows
GET  /v1/cache/symbols
GET  /v1/cache/coverage
POST /v1/time-of-day/normalize
POST /v1/time-of-day/profile
```

The cache routes retain request IDs, ETags, gzip, CORS, request-size limits,
diagnostic errors, and profile result-cache behavior. Existing cache manifests,
schemas, and generations remain readable because `CACHE_SCHEMA_VERSION` and
the on-disk contract did not change.

Screener routes are:

```text
GET  /v1/screener/health
GET  /v1/screener/capabilities
POST /v1/screener/spreads
```

When ClickHouse or screener initialization fails, screener health is `503` and
screen requests return `503 SCREENER_NOT_READY`; cache health and time-of-day
routes continue independently.

## Screening example

```bash
curl -sS http://localhost:8788/v1/screener/spreads \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: workbench-example-1' \
  -d '{
    "underlying_symbol": "QQQ",
    "enabled_strategies": ["vertical", "calendar", "diagonal"],
    "option_types": ["call", "put"],
    "outlook": "any",
    "dte": {"minimum": 7, "maximum": 90,
      "front_minimum": 7, "front_maximum": 35,
      "back_minimum": 21, "back_maximum": 90,
      "minimum_expiration_separation_days": 7},
    "minimum_execution_quality": "questionable",
    "maximum_quote_age_seconds": 120,
    "result_count_per_strategy": 20,
    "delta_hedge_mode": "both",
    "entry_basis": "natural",
    "probability": {"drift_mode": "zero", "simulation_paths": 5000}
  }'
```

An abbreviated response has this shape:

```json
{
  "request_id": "workbench-example-1",
  "model_version": "phase1-2.0.0",
  "snapshot_id": "...",
  "snapshot_timestamp": "2026-07-31T18:00:00Z",
  "underlying_bid": 551.10,
  "underlying_ask": 551.14,
  "underlying_mark": 551.12,
  "applied_assumptions": {
    "drift_mode": "zero",
    "scenario_pricing_method": "tian",
    "static_hedge_rebalanced": false,
    "probability_outputs": "conditional_model_estimates"
  },
  "candidates": [{
    "candidate_id": "stable-snapshot-and-leg-hash",
    "strategy": "vertical",
    "variant": "call_debit_vertical",
    "outlook": "bullish",
    "synthetic_bid": 2.10,
    "synthetic_midpoint": 2.25,
    "synthetic_ask": 2.40,
    "entry_value": 2.40,
    "opening_classification": "debit",
    "target_exit_value": 3.12,
    "target_profit": 0.72,
    "model_probability_of_profit": 0.57,
    "probability_of_reaching_target": 0.41,
    "expected_pnl_after_costs": 18.30,
    "static_hedge_shares": -23,
    "maximum_profit_loss_basis": "exact_expiration_payoff",
    "warnings": []
  }],
  "funnel": {"contracts_loaded": 640, "contracts_eligible": 188,
    "results_returned": 20},
  "rejections": {"by_reason": {"STALE_QUOTE": 12}, "examples": {},
    "dominant_reason": "STALE_QUOTE"},
  "cache_hit": false,
  "elapsed_ms": 824.2,
  "warnings": []
}
```

A valid screen with no qualifying candidates returns HTTP 200, an empty
`candidates` list, and the complete bounded rejection funnel.

## Pricing, probability, and execution conventions

The strategy value is always:

```text
sum(long option values) - sum(short option values)
```

Synthetic bid uses long bids and short asks; synthetic ask uses long asks and
short bids. Positive opening value is a debit and negative opening value is a
credit. Natural execution (`synthetic_ask`) is the default. Exit P&L is exit
strategy value minus entry value, with quantities and contract multipliers
included in dollar results.

Screening results are model estimates. “Probability of profit” is conditional
on the configured model, including its drift, fitted ATM volatility, residual
half-life, exit policy, execution costs, and deterministic path seed. It is not
a guarantee. The default path model uses daily geometric steps, zero drift,
and fitted ATM IV. Per-contract IV residuals decay toward zero with a five-session
half-life. Fitted surfaces are not silently extrapolated; conservative boundary
treatment is explicitly warned and reduces robustness.

By default every positive-tenor screener scenario option uses Tian. The
`configured` mode uses the canonical exercise-aware routing, including the
short-tenor override and vectorized European route. Scenario valuation creates
one deterministic standard-normal shock matrix per screen, uses common random
numbers for exact drift-keyed path sets, and shares horizon spot grids and
unique contract price grids across bounded candidates.

Numerical work is submitted in deterministic bounded batches to one persistent
spawn process pool. `delta_hedge_mode=both` derives unhedged and static-hedged
metrics from the same base option P&L matrix and does not expand the option
pricing universe. The explicit full-scenario candidate bound is applied per
strategy after the authoritative prescore.

Static delta hedging opens one rounded underlying-share position with the
spread, includes its opening/closing execution and P&L, and is never rebalanced.
`delta_hedge_mode=both` returns distinct stable IDs for hedged and unhedged
variants.

Vertical maximum profit, maximum loss, and breakeven are exact expiration
payoffs. Calendar and diagonal maximum profit/loss may be scenario-based rather
than mathematically exact; their response fields remain nullable and carry an
explicit modeled basis.

Execution quality is `good`, `questionable`, or `poor` and reports numeric,
per-leg, and spread-level reasons. It considers quote width and age, displayed
size, open interest, volume, calculation flags, source-only values, and the
synthetic spread width. Poor candidates are excluded unless explicitly allowed.

## Health and errors

Errors use a sanitized envelope:

```json
{"error": {"code": "SCREENER_NOT_READY", "message": "...",
  "details": null, "request_id": "..."}}
```

Screener codes include `SCREENER_NOT_READY`, `SCREENER_BUSY`,
`SCREENER_TIMEOUT`, `SNAPSHOT_NOT_FOUND`, `SNAPSHOT_NOT_TERMINAL`,
`SURFACE_NOT_AVAILABLE`, `NO_ELIGIBLE_CONTRACTS`, and `PRICING_UNAVAILABLE`.
Responses never expose raw SQL, credentials, connection strings, or tracebacks.
One monotonic screen deadline covers universe loading, eligibility, generation,
grid preparation, pool submission/waits, and metric evaluation. Timed-out work
cancels queued batches and returns the structured `SCREENER_TIMEOUT` error.
Responses and completion logs include bounded performance diagnostics such as
path sets, unique price grids, numerical jobs, pricing batches, and phase times.

## Upgrade from the old package

Replace the removed `algotrading.refdata.options.cache` module paths with:

- cache CLI: `algotrading.refdata.options.gui.cache`
- REST server: `algotrading.refdata.options.gui.api`
- Python cache imports: `algotrading.refdata.options.gui.cache.*`

There is no permanent compatibility shim. Configuration and existing Parquet
generations do not need to be renamed or rebuilt.

## Validation

```bash
python -m compileall algotrading/refdata/options/gui
python -m pytest algotrading/refdata/options/gui
python -m algotrading.refdata.options.gui.screener.benchmark
python -m algotrading.refdata.options.gui.screener.benchmark --real-tian-jobs 256
python -m algotrading.refdata.options.gui.api --help
python -m algotrading.refdata.options.gui.cache --help
rg "algotrading\.refdata\.options\.cache|refdata/options/cache"
```
