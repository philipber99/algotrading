# Algotrading Reference Data REST API

This package exposes selected `algotrading.refdata.MarketData` functions over
HTTP. It is a single Python application: Uvicorn runs inside the process, so no
Apache, nginx, Gunicorn, or other web server is required.

The API is read-oriented. Historical OHLCV completeness, gap detection, and
repairs are handled by the SA updater jobs and the scoped manual repair command. The bars
endpoint calls `prices_sql` with `raise_missing=False` and does not invoke
`update_symbol_prices`.

## Installation

The archive is rooted at `refdata/`. Extract it in the `algotrading` package
directory that already contains `refdata/`.

Install the two additional runtime dependencies:

```bash
python -m pip install -r algotrading/refdata/api/requirements.txt
```

Ensure the directory containing the `algotrading` package is on `PYTHONPATH`
or that the package has been installed in the active environment.

## Starting the service

`ALGO_HOME` is required for the SA updater commands.

```bash
export ALGO_HOME=/home/algo

python -m algotrading.refdata.api \
    --environment refdata \
    --host 0.0.0.0 \
    --port 8787
```

The compatibility launcher is also available:

```bash
python -m algotrading.refdata.RefDataApiMain \
    --environment refdata \
    --host 0.0.0.0 \
    --port 8787
```

The default listen address is `127.0.0.1`. Binding to `0.0.0.0` should only be
done on the intended protected network.

Interactive OpenAPI documentation is available at:

```text
http://HOST:8787/docs
```

## Endpoints

| Method | Route |
| --- | --- |
| `GET` | `/health/live` |
| `GET` | `/health/ready` |
| `GET` | `/api/v1/market/status` |
| `GET` | `/api/v1/calendars/trading` |
| `GET` | `/api/v1/quotes` |
| `POST` | `/api/v1/options/quotes` |
| `POST` | `/api/v1/options/contract-terms/query` — bounded exact-contract terms |
| `POST` | `/api/v1/options/expirations/query` — lightweight provider expiration calendar |
| `POST` | `/api/v1/options/chains/query` |
| `POST` | `/api/v1/options/market-history/query` — exact-contract daily OHLC history |
| `GET` | `/api/v1/rates/risk-free-curve` |
| `POST` | `/api/v1/rates/risk-free-curves/query` — bounded point-in-time curve history |
| `POST` | `/api/v1/futures/contract-specifications` |
| `GET` | `/api/v1/futures/{contract}/current-price` |
| `GET` | `/api/v1/bars` |
| `GET` | `/api/v1/symbols/{symbol}/bars` |
| `GET` | `/api/v1/symbols/{symbol}/intraday-bars` |
| `GET` | `/api/v1/symbols/{symbol}/current-price` |
| `GET` | `/api/v1/symbols/{symbol}/splits` |
| `GET` | `/api/v1/symbols/{symbol}/earnings` |
| `GET` | `/api/v1/symbols/{symbol}/dividends` |
| `GET` | `/api/v1/calendars/earnings` — all symbols in a date range |
| `GET` | `/api/v1/calendars/dividends` — all symbols in a date range |
| `GET` | `/api/v1/calendars/fomc` |
| `GET` | `/api/v1/calendars/economic` |
| `GET` | `/api/v1/economic-releases` |
| `GET` | `/api/v1/universe/symbols` |
| `GET` | `/api/v1/indexes` |
| `GET` | `/api/v1/indexes/{index_id}/constituents` |
| `GET` | `/api/v1/classifications` |
| `GET` | `/api/v1/classifications/{classification_id}/constituents` |
| `GET` | `/api/v1/symbols/{symbol}/profile` |
| `GET` | `/api/v1/symbols/{symbol}/company-information` (database-first, configured-provider fallback for symbols outside the durable universe) |
| `GET` | `/api/v1/symbols/{symbol}/fundamentals` |
| `GET` | `/api/v1/symbols/{symbol}/estimates` |
| `GET` | `/api/v1/symbols/{symbol}/guidance` |
| `GET` | `/api/v1/symbols/{symbol}/overview` |
| `GET` | `/api/v1/symbols/{symbol}/news` |
| `GET` | `/api/v1/data-status` |
| `GET` | `/api/v1/capabilities` |
| `GET` | `/api/v1/analytics/market-overview` |
| `GET` | `/api/v1/analytics/sectors` |
| `GET` | `/api/v1/analytics/sectors/{classification_id}/constituents` |
| `GET` | `/api/v1/analytics/sector-performance` |
| `GET` | `/api/v1/analytics/sector-relationships` |
| `GET` | `/api/v1/analytics/futures/positioning` |
| `GET` | `/api/v1/analytics/futures/participation` |
| `GET` | `/api/v1/analytics/futures/term-structure` |
| `GET` | `/api/v1/analytics/futures/crowding` |
| `GET` | `/api/v1/analytics/futures/market-health/overview` |
| `GET` | `/api/v1/symbols/{symbol}/analytics` |
| `GET` | `/api/v1/finra/short-volume` — bounded multi-symbol daily history |

### FINRA short-volume history

`GET /api/v1/finra/short-volume` accepts one or more comma-separated or
repeated `symbols`, an optional inclusive `as_of` date, and a `session_limit`
from 1 through 252. The response includes the exact recent FINRA session dates
plus only the observations present for each requested symbol. Each observation
contains adjusted short and total volume and `short_volume_percent` on a 0–1
scale. Missing symbol/date observations are intentionally not backfilled.

### Historical exact-option prices

`POST /api/v1/options/market-history/query` accepts 1 through 500 unique exact
OCC symbols, aware `start_at` and exclusive `end_at` timestamps, `interval:
"1d"`, and `include_greeks`. Symbols are normalized with the shared OCC parser;
duplicates after normalization, invalid symbols, inverted ranges, and ranges
beyond `refdata.api.option_market_history_max_days` (default `376`) are rejected.

The endpoint converts each canonical symbol to its six-character-root OCC
provider form and calls the configured option-history provider's `get_bars`
with `1day`. The provider's
daily OHLCV close is returned as both `mark` and `last`; historical bid/ask and
Greeks remain null and are identified by a response warning.

The repository-wide `refdata.providers.prices` selects the live-price provider for
the API and other price consumers. When
`refdata.providers.option_history` is empty, exact option history uses
that same primary provider; setting it to another fully-qualified Python class
routes only option-history calls to the separate provider. Both classes receive
the loaded `config.ini` context. The shipped configuration keeps
`TDAReferenceData` as the live-price provider. The category registry defaults to FMP.

`refdata.providers.historical_bars` separately selects stock/ETF historical
bars for scheduled ingestion, manual SQL repair and historical API fallbacks.
An empty value retains the primary provider. The `[refdata]` environment selects
`algotrading.refdata.providers.AlpacaReferenceData.AlpacaReferenceData` for these
bars. Live day quotes, today's live intraday endpoint, and option history retain
their existing selections. FX, crypto and indexes retain their own routing.

Daily option bars and explicit date coverage are stored in a durable SQLite
cache. A request fully inside active coverage is served without contacting the
provider. A partially covered request loads only its uncovered leading,
trailing, or interior date intervals and merges them into the existing
contract data. Empty covered dates are retained as coverage so weekends and
holidays are not fetched repeatedly. Each contract has a sliding seven-day TTL
which is reset whenever requested coverage is found in the cache. The default
cache is `${XDG_CACHE_HOME:-~/.cache}/algotrading/refdata-option-bars.sqlite3`.

### Exact option contract terms

`POST /api/v1/options/contract-terms/query` accepts 1 through 500 exact OCC
symbols and returns the canonical symbol, multiplier, deliverable, adjusted
state, exercise style, settlement style/type, authoritative expiration and
last-trade/settlement timestamps, exchange timezone, source version, quality,
and stable warnings. The service loads only one strike on one requested expiry
per underlying, reads `exerciseType` and `multiplier` from that chain sample,
and caches the result by underlying symbol for 24 hours. Consequently, contracts
on the same underlying do not trigger additional chain requests. When chain
metadata is
unavailable, an OCC-parseable equity option may use the configured audited
standard-contract assumption only when no evidence indicates adjustment; the
row is marked `usable_with_warnings` and includes
`option_terms_assumed_standard_occ`. Adjusted or futures-option contracts are
never converted to standard 100-share contracts.

### Risk session calendars

Risk freshness and SQL option-history gap detection share the versioned
`refdata_exchange_calendars_product_2020_2035_v2` registry. It is deliberately
bounded to 2020–2035 and fails closed outside that range. XNYS and distinct CME
equity-index, rates, energy, metals, and FX profiles publish holidays, early
closes, exchange timezone, overnight session labeling, maintenance breaks, and
supported product roots; futures terms select the product calendar by root.

Supported index identifiers are `NASDAQ100`, `SP100`, and `SP500`. S&P 100
membership comes from the official machine-readable holdings download for the
[iShares S&P 100 ETF](https://www.ishares.com/us/products/239723/ishares-sp-100-etf).
NASDAQ-100 and S&P 500 membership continue to come from FMP.

Examples:

```bash
curl 'http://localhost:8787/api/v1/market/status'

curl 'http://localhost:8787/api/v1/calendars/trading?start_date=2026-07-01&end_date=2026-07-31'

curl 'http://localhost:8787/api/v1/quotes?symbols=MSFT&symbols=AAPL,NVDA'

curl -X POST 'http://localhost:8787/api/v1/options/contract-terms/query' \
  -H 'Content-Type: application/json' \
  -d '{"symbols":["MSFT260918C00400000"]}'

curl 'http://localhost:8787/api/v1/bars?symbols=MSFT,AAPL,NVDA&start=2025-01-01&end=2026-08-12'

curl 'http://localhost:8787/api/v1/symbols/MSFT/bars?start=2026-07-24T06:30:00&end=2026-07-24T13:00:00&interval=1m&market_hours_only=true'

curl 'http://localhost:8787/api/v1/symbols/MSFT/intraday-bars?market_hours_only=true'

curl 'http://localhost:8787/api/v1/symbols/MSFT/current-price?view=bar'
curl 'http://localhost:8787/api/v1/futures/ESU26/current-price?view=price'

curl 'http://localhost:8787/api/v1/finra/short-volume?symbols=AAPL,MSFT&session_limit=21'

curl 'http://localhost:8787/api/v1/symbols/MSFT/splits?start_date=2015-01-01&end_date=2027-01-01'

curl 'http://localhost:8787/api/v1/symbols/MSFT/earnings?start_date=2026-01-01&end_date=2026-12-31'

curl 'http://localhost:8787/api/v1/symbols/MSFT/dividends?start_date=2026-01-01&end_date=2026-12-31&date_type=ex_dividend_date'

curl 'http://localhost:8787/api/v1/calendars/earnings?start_date=2026-08-17&end_date=2026-09-07'

curl 'http://localhost:8787/api/v1/calendars/dividends?start_date=2026-08-17&end_date=2026-09-07&date_type=ex_dividend_date'

curl 'http://localhost:8787/api/v1/calendars/fomc?start_date=2026-01-01&end_date=2026-12-31'

curl 'http://localhost:8787/api/v1/calendars/economic?start_date=2026-07-01&end_date=2026-07-31'

curl 'http://localhost:8787/api/v1/economic-releases?view=today&timezone=America/New_York&impact=High'
```

Dates and timestamps are inclusive. For a bar query, a date-only `end` value
means the end of that local day. Intraday timestamps without an offset are
interpreted in the configured API timezone.

Supported bar intervals:

- `1d`, mapped to `MarketData.prices_sql(..., frequency="1day")`
- `1m`, mapped to `MarketData.prices_sql(..., frequency="1min")`

An empty supported one-minute range returns
`INTRADAY_DATA_UNAVAILABLE`; an explicitly partial source range returns
`INTRADAY_DATA_INCOMPLETE` with bounded coverage details. Daily behavior is
unchanged and one-minute requests never substitute daily bars.

### Day quotes and detailed intraday bars

`GET /api/v1/quotes` accepts repeatable or comma-separated `symbols`. Every
request checks the shared in-process quote cache first. Cache misses on a
trading day make a batched `get_quotes` call on the provider selected by
`refdata.providers.prices`, returning `open`, `bid`, `ask`, and `mark`. Cache
misses on a non-trading day read each symbol's last trading-day SQL bar, map
`close` to `mark`, and omit `bid` and `ask`.

An independent startup worker preloads the deduplicated SP100, SP500, and NASDAQ100
constituents, plus configured supplemental symbols, from that same provider.
It refreshes the complete universe every 60 seconds by default during market
hours. Outside market hours it performs only the initial startup load and keeps
the last snapshot until the next session begins. Server-side operations that
need a latest scalar price use this cache before their legacy current-price
source. Quote-cache population and worker status are visible in `/health/ready`.

`GET /api/v1/bars` accepts comma-separated or repeatable symbols and returns
one `{symbol, bars}` group per symbol. It uses the bounded SQL batch repository,
so a 500-name screen does not create 500 SQL/API calls. The date-scoped daily
bar cache is persisted in SQLite across API restarts and is cleared when the
local calendar date changes. `include_current_quote=true` is opt-in; it obtains
all missing current quotes through the shared quote cache and at most one
provider batch per configured provider batch size. Ordinary historical and
screener requests do not call the remote quote provider.

`GET /api/v1/symbols/{symbol}/intraday-bars` is deliberately single-symbol.
On a trading day it calls `get_bars(..., "1min")` on the same configured
provider; otherwise it reads the latest trading session's one-minute SQL bars.
It never calls or consults the streaming-prices class. Use the general bars
endpoint for historical or bulk workflows.

### Economic releases

`GET /api/v1/economic-releases` is a typed, SQL-only view of the existing
`economic_calendar` table. It never refreshes from a provider in the request
path and returns `Cache-Control: no-store`. The older
`/api/v1/calendars/economic` contract remains unchanged.

Query parameters:

| Parameter | Default | Notes |
| --- | --- | --- |
| `view` | `upcoming` | `upcoming`, `today`, `prior`, or `range` |
| `start`, `end` | view-dependent | ISO date/timestamp; both required for `range` |
| `internal_name` | unset | Repeatable stable-name filter |
| `impact` | unset | Repeatable impact filter |
| `actual_status` | `all` | `all`, `available`, or `pending` |
| `limit` | `250` | 1 through 1,000 |
| `offset` | `0` | Non-negative pagination offset |
| `timezone` | API timezone | Valid IANA timezone for boundaries and output |

`upcoming` spans now through 21 calendar days and sorts ascending. `today`
uses the requested timezone's local day. `prior` defaults to a 90-day
lookback and sorts newest first. `range` is explicit, inclusive at its public
date/timestamp bounds, and capped at five years. Internally, date ranges are
converted to half-open intervals in the legacy SQL storage timezone.
Provider timestamps must be timezone-aware and are converted uniformly to
configured `LocalTZ` before SQL writes; a naïve provider timestamp is rejected
instead of being guessed. SQL `DATETIME` rows are consequently interpreted as
`LocalTZ`, never as a mixture of local and UTC values.

The response contains `items` with `timestamp`, `internal_name`, provider
display `report`, `period`, `previous`, `estimate`, `actual`, `impact`, and a
derived `status` (`released`, `due_pending`, or `scheduled`). Numeric zero is
a released actual. Metadata contains the page count, total, limit, offset,
as-of timestamp, timezone, view, resolved range, `source: "sql"`, and warnings.

```json
{
  "items": [{
    "timestamp": "2026-08-10T05:30:00-04:00",
    "internal_name": "monthly_cpi",
    "report": "Inflation Rate MoM",
    "period": "Jul",
    "previous": 0.003,
    "estimate": 0.002,
    "actual": null,
    "impact": "High",
    "status": "scheduled"
  }],
  "meta": {
    "count": 1,
    "total": 1,
    "limit": 250,
    "offset": 0,
    "as_of": "2026-08-08T12:00:00-04:00",
    "timezone": "America/New_York",
    "view": "upcoming",
    "range_start": "2026-08-08T12:00:00-04:00",
    "range_end": "2026-08-29T12:00:00-04:00",
    "source": "sql",
    "warnings": []
  }
}
```

API bar reads explicitly disable `prices_sql`'s legacy per-symbol current-price
append. Clients that need a provisional current daily bar use the bulk route's
`include_current_quote=true` option, which is backed by the batched quote cache.

### Stock Analysis analytics

Market, sector, and company analytics are calculated at request time from
split-adjusted SQL OHLCV, effective-dated index membership, normalized company
classifications, market caps, filings, estimates, events, and economic releases.
They are not persisted. Composite market/sector requests use one universe read
and one batched daily-price read; today's market overview adds one bounded
current-minute read and uses the shared batched quote cache for open/current
returns. Bulk analytics never fan out to per-symbol streaming-history calls.
Membership and daily histories are shared across analytics routes for the local
market date, so later period, constituent, and relationship requests normally
reuse those reads.

Market overview also returns a bounded FINRA short-volume overlay. The headline
uses the latest available S&P 500 observation and exposes both the equal-weighted
mean constituent ratio and a market-cap-weighted mean. `most_shorted` and
`least_shorted` rank the requested overview universe by aggregate short volume
divided by aggregate FINRA total volume over 1, 5, 21, or 63 completed sessions;
the `today` selection intentionally maps to the latest completed one-day FINRA
reading. These metrics describe reported off-exchange activity, not short
interest or directional positioning.

The regression channel is OLS close versus observation number with empirical
10th/90th-percentile residual bands. Beta uses pairwise-aligned close-to-close
log returns and requires four aligned returns. Channel statistics require four
prices. Daily realized volatility is annualized by `sqrt(252)`. Missing inputs
remain null and include typed reason codes; one-minute gaps are never replaced
with daily observations.

The analytics response cache is bounded and in-process. Keys include every
semantic query parameter. Defaults are 30 seconds for intraday and five
minutes for daily analytics, and concurrent identical misses are coalesced.
Full-table watermarks are disabled by default; day-scoped membership/history
caches naturally roll over on the next local date. A background startup worker
warms one year of shared daily rows for SP100, SP500, and NASDAQ100.

`GET /api/v1/capabilities` advertises supported universes, benchmarks, periods,
ranges, groupings, and weightings. It deliberately reports
`stock_screening: false`: screening is owned by the GUI and there is no Stock
Analysis screener route or server-side screening score.

## Configuration

The service calls `load_configuration` with the requested environment. Existing
configuration values override these defaults.

| Setting | Default | Purpose |
| --- | ---: | --- |
| `refdata.api.host` | `127.0.0.1` | Listen address |
| `refdata.api.port` | `8787` | Listen port |
| `refdata.api.timezone` | host/application timezone | Response and input timezone |
| `refdata.api.log_level` | `INFO` | Application log level |
| `refdata.api.log_file` | unset | Optional dedicated log file |
| `refdata.api.cors_origins` | empty | Comma-separated or list of allowed browser origins |
| `refdata.analytics.cache.max_entries` | `256` | Maximum ephemeral analytics responses |
| `refdata.analytics.cache.intraday_ttl_seconds` | `30` | Intraday analytics response TTL |
| `refdata.analytics.cache.daily_ttl_seconds` | `300` | Daily analytics TTL |
| `refdata.analytics.slow_log_threshold_ms` | `1000` | Emit per-stage market-overview timing after this duration |
| `refdata.analytics.cache.use_watermark` | `False` | Opt in to SQL watermark cache keys |
| `refdata.analytics.daily_cache.warmup_enabled` | `True` | Warm shared membership and daily bars in the background at startup |
| `refdata.analytics.daily_cache.warmup_universes` | `SP100,SP500,NASDAQ100` | Universes included in startup warmup |
| `refdata.analytics.intraday_rows_ttl_seconds` | `60` | Shared current-session minute-row TTL |
| `refdata.analytics.relationships.max_dimension` | `64` | Maximum relationship-matrix dimension |
| `refdata.stock_analysis.forward_pe.provider_preference` | empty | Comma-separated provider order for same-target estimates |
| `refdata.api.initialize_streaming_prices` | `True` | Initialize the configured pricing client |
| `refdata.api.require_streaming_prices` | `False` | Fail startup if that client is unavailable |
| `refdata.api.streaming_symbols` | `pricing.active_symbols` | Initial pricing subscriptions |
| `refdata.api.max_concurrent_upstream_calls` | `8` | Simultaneous SQL/provider operations |
| `refdata.api.upstream_slot_wait_seconds` | `5` | Queue time before returning service busy |
| `refdata.api.calendar_refresh_seconds` | `21600` | Refresh MarketData's in-process calendar |
| `refdata.api.gzip_minimum_size` | `1000` | Minimum response bytes for Gzip |
| `refdata.api.cache.enabled` | `True` | Enable the bounded API cache |
| `refdata.api.cache.max_entries` | `1024` | Maximum API cache entries |
| `refdata.api.cache.day_quotes_ttl_seconds` | `2` | Duplicate-request suppression TTL; per-symbol validity follows the quote refresh interval |
| `refdata.api.option_history_cache.path` | XDG cache path | Durable exact-option daily OHLC cache |
| `refdata.api.option_history_cache.ttl_seconds` | `604800` | Sliding per-contract option-history TTL (one week) |
| `refdata.api.option_market_history_max_days` | `376` | Maximum exact-option request span, including P&L boundary lookback |
| `refdata.api.option_market_history_max_rows` | `250000` | Maximum exact-option rows in one response |
| `refdata.providers.prices` | `TDAReferenceData` | Live-price and primary API provider class |
| `refdata.providers.historical_bars` | empty; Alpaca in `[refdata]` | Historical stock/ETF 1min and 1day provider |
| `refdata.bars.lookback_trading_days` | `1` | Reconcile 1–10 completed trading dates plus available subsequent sessions |
| `alpaca.bars.historical_delay_minutes` | `16` | Historical account cutoff; minimum 15 minutes |
| `alpaca.bars.requests_per_minute` | `190` | Account request budget, shared by feeds and local processes |
| `refdata.bars.symbol_batch_size` | `500` | Maximum symbols per bulk request, up to 500 |
| `refdata.bars.sql_batch_size` | `1000` | Rows per SQL insert, up to 10000 |
| `working_path` | must be configured | Absolute writable directory outside the source tree for historical-bar checkpoints and shared Alpaca request pacing |
| `refdata.bars.state_path` | `working_path/alpaca_historical_bars.sqlite3` | Scheduled paging and source coverage; outside the source tree |
| `refdata.providers.option_history` | empty (primary provider) | Optional fully-qualified provider class used only for exact option history |
| `refdata.api.quote_precache.enabled` | `True` | Start the independent universe quote worker |
| `refdata.api.quote_precache.poll_interval_seconds` | `60` | Quote refresh interval during market hours |
| `refdata.api.quote_precache.indexes` | `SP100,SP500,NASDAQ100` | Constituent universes included in each refresh |
| `refdata.api.quote_precache.additional_symbols` | `SPY,QQQ,VIX` | Supplemental symbols included in the shared cache |
| `refdata.api.quote_precache.batch_size` | `500` | Maximum symbols in each provider quote call |
| `refdata.api.quote_precache.max_symbols` | `1000` | Maximum deduplicated precache universe size |
| `refdata.api.quote_precache.max_entries` | `2000` | Maximum shared quote-cache entries |
| `refdata.api.quote_precache.shutdown_wait_seconds` | `10` | Graceful worker shutdown wait bound |
| `refdata.api.cache.option_contract_metadata_ttl_seconds` | `86400` | Symbol-level exercise-type and multiplier cache loaded from one minimal chain sample |
| `refdata.api.max_bulk_bar_symbols` | `1000` | Maximum symbols accepted by one bulk-bars request |
| `refdata.api.max_bulk_bar_rows` | `1000000` | Maximum total daily bars returned by one bulk request |
| `refdata.api.daily_bars_cache.enabled` | `True` | Persist bulk daily-bar results across restarts |
| `refdata.api.daily_bars_cache.path` | `tmp/refdata_api_daily_bars.sqlite3` | Cache path relative to `ALGO_HOME` |
| `refdata.api.cache.detailed_intraday_bars_ttl_seconds` | `2` | Server-side detailed current-session bar TTL; non-trading default is 300 seconds |
| `refdata.api.max_quote_symbols` | `1000` | Maximum symbols in a batched quote request |
| `refdata.api.max_scalar_quote_fallbacks` | `25` | Maximum completed-session scalar lookups after the shared quote cache and daily-bar SQL both miss |
| `refdata.api.max_detailed_intraday_rows` | `2000` | Maximum rows from the single-symbol detailed endpoint |
| `refdata.api.max_1m_days` | `31` | Maximum one-minute query span |
| `refdata.api.max_1d_days` | `7305` | Maximum daily query span |
| `refdata.api.max_event_days` | `3660` | Maximum event-calendar query span |
| `refdata.api.max_calendar_days` | `3660` | Maximum trading-calendar query span |
| `refdata.api.max_bar_rows` | `100000` | Maximum bars in one response |
| `refdata.providers.economic_calendar` | `refdata.providers.default` | Optional category-specific calendar provider |
| `refdata.providers.earnings_calendar` | FMP | Bulk earnings-calendar provider implementing the category contract |
| `refdata.providers.stock_splits` | FMP | Stock-splits provider implementing the category contract |
| `refdata.providers.earnings_calendar_fallback` | unset | Optional per-symbol earnings source when the bulk calendar is unentitled |
| `refdata.providers.dividends` | unset | Optional legacy dividend-history source |
| `refdata.providers.fx`, `refdata.providers.backup`, `refdata.providers.crypto` | unset | Price-bar routing for FX, backup, and crypto |
| `refdata.economic_calendar.auto_update.enabled` | `True` | Start the server-owned updater |
| `refdata.economic_calendar.auto_update.horizon_days` | `60` | Inclusive future provider horizon |
| `refdata.economic_calendar.auto_update.poll_interval_seconds` | `1800` | Recheck/poll bound; minimum 60 seconds |
| `refdata.economic_calendar.auto_update.coverage_tail_slack_days` | `3` | Allowed quiet tail at the horizon |
| `refdata.economic_calendar.auto_update.daily_overlap_days` | `1` | Adjacent days included in actual polling |
| `refdata.economic_calendar.auto_update.lock_wait_seconds` | `0` | Advisory-lock wait bound |
| `refdata.economic_calendar.weekly_update.lookback_days` | `10` | Prior days reconciled by `SAWeeklyUpdateMain`; the future horizon uses `auto_update.horizon_days` |
| `refdata.earnings_calendar.auto_update.enabled` | `True` | Start live earnings-result polling |
| `refdata.earnings_calendar.auto_update.poll_interval_seconds` | `1800` | Retry while today's earnings remain pending; minimum 60 seconds |
| `refdata.earnings_calendar.auto_update.lock_wait_seconds` | `0` | Earnings advisory-lock wait bound |
| `refdata.earnings_calendar.overlap_days` | `14` | Nightly earnings-calendar lookback from the last successful check |
| `refdata.earnings_calendar.horizon_days` | `90` | Nightly earnings-calendar future horizon |
| `refdata.earnings_calendar.chunk_days` | `30` | Maximum inclusive FMP earnings request window |
| `refdata.earnings_calendar.backfill_start` | `refdata.reload_start` | Inclusive history used by `SANightlyUpdateMain --force-refresh` |
| `refdata.earnings_calendar.legacy_fallback.enabled` | `True` | Use the configured `refdata.providers.earnings_calendar_fallback` when the bulk calendar is not entitled |
| `refdata.earnings_calendar.legacy_fallback.interval_days` | `7` | Minimum interval between normal per-symbol earnings fallback refreshes |
| `refdata.dividend_calendar.backfill_start` | `refdata.reload_start` | Inclusive forced-refresh start when a rolling history length is unset |
| `refdata.dividend_calendar.backfill_history_days` | `0` | Rolling forced-refresh history; takes precedence over `backfill_start` |
| `refdata.dividend_calendar.legacy_fallback.enabled` | `True` | Allow the configured `refdata.providers.dividends` as the durable source |
| `refdata.dividend_calendar.prefer_configured_provider` | `False` | Use the configured all-symbol dividend provider before the registry provider |

Browser CORS remains disabled when `refdata.api.cors_origins` is empty, and a
wildcard is not supported. For the deployed Stock Analysis Workbench, configure
the exact HTTPS origin:

```text
refdata.api.cors_origins = https://workbench.example.test
```

Startup logs state either that browser CORS is disabled or the number of
configured origins; origins and other configuration values are not logged.

The updater starts one supervised task per app instance. On every enabled
startup, including weekends, it checks future SQL coverage. It fetches the
inclusive `today` through `today + 21 days` provider window only when the SQL
window is empty or its latest release is earlier than the horizon minus the
three-day slack. Coverage is rechecked under a cross-process SQL advisory lock.
Shutdown sets a cooperative stop signal, wakes scheduler sleep, and joins the
single tracked worker before provider sessions close. A join timeout is logged
and raised as `WORKER_JOIN_TIMEOUT`; it is never reported as a successful stop.

On dates present in the existing trading calendar, the task queries due rows
whose scheduled time has passed and whose `actual IS NULL`. Due rows trigger a
provider request for yesterday through tomorrow, followed by one atomic,
non-destructive reconciliation. Moved and renamed releases are updated in
place when semantic identity is unambiguous; provider nulls never erase stored
facts; missing remote rows are retained. Ambiguities and target-key collisions
are reported and left for review. Provider and SQL failures retain existing
rows and retry no sooner than the configured scheduler interval.

A separate supervised earnings task checks the existing `earnings_calendar`
rows for the current New York date at server startup and each date boundary.
A row is pending while either `eps` or `revenue` is null. Pending rows trigger a
same-day FMP earnings-calendar request under a cross-process advisory lock; the
task rechecks SQL under the lock, updates only matching `(symbol, date)` rows,
and never inserts, deletes, or replaces scheduled rows. Provider nulls preserve
stored facts and numeric zero is a completed value. While any row remains
pending, provider failures, lock contention, and incomplete results retry after
the configured interval (30 minutes by default). Once all results are present,
or no earnings are scheduled that day, provider polling ceases until the next
New York date.

Cache TTL settings, in seconds:

```text
refdata.api.cache.calendar_ttl_seconds = 21600
refdata.api.cache.daily_bars_ttl_seconds = 900
refdata.api.cache.intraday_bars_ttl_seconds = 300
refdata.api.cache.current_bars_ttl_seconds = 2
refdata.api.cache.current_price_ttl_seconds = 1
refdata.api.cache.splits_ttl_seconds = 900
refdata.api.cache.earnings_ttl_seconds = 900
refdata.api.cache.dividends_ttl_seconds = 900
refdata.api.cache.fomc_ttl_seconds = 3600
refdata.api.cache.economic_ttl_seconds = 60
```

The API defaults `sql.enable_cache` to `False` for its context because it has a
bounded cache of its own. An explicit existing configuration value can override
that default.

## Stock Analysis data foundation

The additive implementation is split into guarded repositories under
`algotrading/refdata/repositories`, normalized providers under
`algotrading/refdata/providers/fmp` and `providers/sec`, shared refresh and batch
logic under `stock_analysis`, and small FastAPI routers under `api/routes`.
New issuer and classification queries use `securities`, `companies`, and
`company_classifications`; they do not use the legacy symbol map or the legacy
MarketData sector/industry helpers. The existing general reference-data updater
is unchanged and is not imported by either new job.

Install the SQL objects with a schema-owner account in this order:

1. `sql/stock_analysis/001_create_stock_analysis_tables.sql`
2. `sql/stock_analysis/002_create_stock_analysis_indexes.sql`
3. `sql/stock_analysis/003_seed_supported_indexes.sql`
4. Substitute the database and runtime principal, then run
   `sql/stock_analysis/004_grant_stock_analysis_runtime.sql`
5. `sql/stock_analysis/005_optimize_read_paths.sql`
6. Run `sql/stock_analysis/earnings_calendar_fit_assessment.sql` read-only.

The runtime principal receives only `SELECT`, `INSERT`, and `UPDATE`. Python
never executes the setup scripts. `reset_stock_analysis_data.sql` is destructive,
maintenance-only, and is not exposed by the API or jobs. See the SQL directory's
README for the deployed earnings-table assessment and reset implications.

The independent jobs are:

```bash
python -m algotrading.refdata.SAWeeklyUpdateMain --environment refdata
python -m algotrading.refdata.SAMorningUpdateMain --environment refdata
python -m algotrading.refdata.SANightlyUpdateMain --environment refdata

# Bounded repair/smoke examples
python -m algotrading.refdata.SAWeeklyUpdateMain \
  --symbols AAPL MSFT --force-refresh
# Add symbols and hydrate every symbol-scoped durable Stock Analysis dataset.
# This does not run benchmark, economic-calendar, dividend, or index updates.
python -m algotrading.refdata.SAWeeklyUpdateMain \
  --add-symbols BAESY TSM --force-refresh
python -m algotrading.refdata.SANightlyUpdateMain \
  --max-symbols 10 --dry-run
python -m algotrading.refdata.SANightlyUpdateMain \
  --skip-symbol-updates --force-refresh
# One-time operator reactivation; also includes this symbol in morning prices.
python -m algotrading.refdata.SAMorningUpdateMain --reactivate-symbol AAPL
# Scoped manual repair of one existing or missing 1min bar.
python -m algotrading.refdata.PriceBarsRepairMain \
  --symbols AAPL --granularities 1min --at-bar '2026-09-01 06:30:00'
# Revalidate and refresh a bounded history window without deleting rows.
python -m algotrading.refdata.PriceBarsRepairMain \
  --symbols AAPL --granularities 1min 1day --start-date 2026-09-01
```

Supported controls are `--symbols`, `--categories`, `--max-symbols`,
`--force-refresh`, and `--dry-run`. The weekly tool additionally supports
`--add-symbols SYMBOL [SYMBOL ...]`: it creates the requested durable security
identities and refreshes the complete nightly per-symbol dataset (profile,
classification, identifiers, filings, financials, market cap, estimates, and
guidance), without running any global weekly work. It cannot be combined with
`--symbols`, `--categories`, `--max-symbols`, or `--dividends-only`. The nightly
tool also supports `--skip-symbol-updates`, which bypasses symbol selection and
all per-symbol providers and the `1min` price-bar stage while retaining its global index and event refreshes.
Dry-run performs no domain or fetch-state writes. Both tools print their mode,
selected work, per-category validation or update progress, and an explicit
final `SUCCESS` or `FAILED` status with the exit code and summary counters.
News is absent from both jobs. The nightly job
also refreshes the existing `earnings_calendar` and `market_events` contracts
from FMP's all-symbol earnings and upcoming stock-splits calendars. If the
configured bulk earnings calendar is not entitled, it uses the configured
legacy per-symbol earnings provider; forced runs backfill from the configured
history start. Both event feeds are filtered to the distinct symbol universe in
`price_bars_1day`.
The morning job refreshes `trading_calendar` and updates `price_bars_1min`
and `price_bars_1day`; the nightly job refreshes the same calendar and updates
`price_bars_1min`. Both use the shared historical bar updater and report failed
symbol/granularity pairs. Alpaca minute requests combine SIP pre-market, regular
and post-market bars with BOATS overnight bars. Bulk pages contain up to 10000
bars across up to 500 symbols; the provider owns requests and the shared writer
owns SQL transactions. Normal reconciliation inserts absent keys and preserves
existing prices. Missing eligible extended-hours trades do not create synthetic
bars. Regular-session checks retain the configured time-gap window; completed
daily session dates are verified against SQL.

By default, reconcile the full latest completed trading date from 04:00 Eastern
and all available subsequent sessions. Checking that date's regular session
prevents newer overnight rows from concealing unfinished market-hours history.
Increase the bounded lookback using configuration or the morning/nightly
`--bar-lookback-trading-days 10` override. Each completed invocation rechecks the
lookback; older source completion never hides a missing SQL row. A gap between
scheduled runs is also caught up from the saved successful source cutoff.
Daily bars keep their New York date and are published only after that date's
20:00 Eastern post-market boundary has passed the delay cutoff. Current daily
bars remain provisional at Alpaca and are excluded from SQL ingestion.

Configure `working_path` in `[Default]` or the active environment before running
the Alpaca updater on each deployment. Use an absolute, persistent directory
writable by the job's OS user and outside the installed code, for example
`working_path = /home/algo/trading/work` for that service account. The small
SQLite files there store request checkpoints and pacing; price bars are written
directly to MySQL. Moving or deleting the checkpoint can cause requests to be
replayed, while duplicate-safe inserts preserve existing SQL bars.

Scheduled pages have independent feed/granularity cursors under `working_path`.
SQL commits precede cursor updates; interruption can replay a page safely.
Saved tokens retain their exact request bounds and symbol sets. A rejected
token restarts only that request. After resuming frozen requests, the updater
fetches the newly available suffix. This state is separate from historical
backfill checkpoints. Both downloaders share the account pacing database at
`working_path/alpaca_market_data_rate.sqlite3`; HTTP 429 defers all participating
processes for at least 60 seconds. Restart already-running downloaders after
upgrading so they use this shared limiter.

Overlap prices differing by more than 20% fail with a price-basis conflict.
Mixed TDA and already split-adjusted Alpaca history must not receive a guessed
blanket split multiplier. Use explicit symbol/date-scoped repair to retrieve
canonical replacement values when such a conflict is reported. Normal updates
do not rewrite existing history or apply automatic split rescaling.

A symbol is excluded from scheduled and manual
price downloads when its `securities` identities are all inactive. A complete
provider profile can deactivate a symbol, but
missing bars cannot. The existing `updated_at` records the last security-row
change, and `delisted_date` holds a provider delisting date when present; there
is no separate deactivation-reason field. There is no periodic inactive-symbol
probe. Morning
`--reactivate-symbol SYMBOL` is a one-time reset; a later successful provider
profile check may deactivate the symbol again. The manual `PriceBarsRepairMain`
appends, revalidates from `--start-date`, or refreshes one `--at-bar` timestamp
for requested symbols and granularities. Repairs upsert validated bars without
deleting history, and never process an inactive symbol. `--at-bar` uses the database's configured
local timestamp convention (Pacific time in the supplied configuration).
Apply the `016_grant_price_bar_runtime.sql` runtime grants before deploying
these commands. No new deactivation table or schema migration is required.

For production cutover, record `MAX(date)` from `trading_calendar` and the
per-symbol `MAX(timestamp)` from both `price_bars_1min` and `price_bars_1day`
before and after the next morning and nightly windows. Check the two SA jobs'
exit status and Nagios messages for failed symbol/granularity pairs, and verify
that the morning window advances both bar tables while the nightly window
advances `1min`. Run the replacement commands without overlapping the old
price writer. Once both windows pass, remove all four `RefDataUpdateMain` cron
entries. The former cleanup and FOMC cron invocations have no replacements:
profile refresh supplies deactivation, and the weekly/API economic-calendar
updaters supply FOMC decisions. FOMC-sensitive scheduler jobs are held and
alerted if the economic calendar query fails or its latest event precedes the
trading day.

The weekly job also force-reconciles the SQL `economic_calendar` over the prior
10 days and the configured future horizon. It uses the same provider registry,
normalizer, reconciler, parameterized repository, transaction handling, and
advisory-lock path as the RefData API server's startup updater. This refresh is
independent of `--force-refresh`, because the global calendar must be checked on
every scheduled weekly run; `--dry-run` only validates its configured provider.
Each batch invocation reports exactly one final Nagios event. Successful runs
and tolerated outcomes use `MONITOR_OK`; a failed exit or interrupted run uses
`MONITOR_CRITICAL` and identifies the job and final status in the event text.

Provider selection is category-specific with a configured default. FMP Starter
is the assumed tier; an endpoint that the configured credentials cannot access
is reported as `PROVIDER_ENTITLEMENT_REQUIRED`, with no plan upgrade or silent
provider fallback. SEC access requires a user agent identifying the application
and a contact email.

```ini
refdata.providers.default = algotrading.refdata.providers.fmp.FMPProvider
refdata.providers.company_filings = algotrading.refdata.providers.sec.SECProvider
refdata.providers.company_financials = algotrading.refdata.providers.sec.SECProvider
refdata.providers.company_guidance = algotrading.refdata.providers.sec.SECProvider
refdata.providers.index_constituents = algotrading.refdata.providers.index_constituents.IndexConstituentProvider
refdata.index_constituents.providers.default = algotrading.refdata.providers.fmp.FMPProvider
refdata.index_constituents.providers.SP100 = algotrading.refdata.providers.index_constituents.IsharesOEFProvider

refdata.index_constituents.ishares_oef.url = https://www.ishares.com/us/products/239723/ishares-s-p-100-etf/latest-holdings.csv
refdata.index_constituents.ishares_oef.connect_timeout_seconds = 5
refdata.index_constituents.ishares_oef.read_timeout_seconds = 15
refdata.index_constituents.ishares_oef.max_attempts = 3
refdata.index_constituents.ishares_oef.retry_backoff_seconds = 0.5
refdata.index_constituents.ishares_oef.max_response_bytes = 2097152
refdata.index_constituents.ishares_oef.max_snapshot_age_days = 7
refdata.index_constituents.ishares_oef.max_future_days = 1
refdata.index_constituents.ishares_oef.min_equity_count = 100
refdata.index_constituents.ishares_oef.max_equity_count = 102
refdata.index_constituents.ishares_oef.min_equity_weight = 0.98
refdata.index_constituents.ishares_oef.max_equity_weight = 1.01
refdata.index_constituents.ishares_oef.symbol_aliases = BRKB=BRK.B

financialmodelingprep.base_url = https://financialmodelingprep.com/stable
financialmodelingprep.timeout_sec = 10
financialmodelingprep.rate_limit_per_minute = 300
financialmodelingprep.rate_limit_retry_seconds = 60
financialmodelingprep.daily_request_limit = 0
financialmodelingprep.quota_timezone = UTC
financialmodelingprep.quota_boundary_hour = 0
financialmodelingprep.profile_cache_ttl_seconds = 60
financialmodelingprep.profile_cache_max_entries = 128
financialmodelingprep.index_authoritative_snapshot.NASDAQ100 = true
financialmodelingprep.index_authoritative_snapshot.SP500 = true

sec.user_agent = Stock Analysis Workbench operations@example.com
sec.base_url = https://data.sec.gov
sec.timeout_sec = 10
sec.requests_per_second = 8
```

The existing `financialmodelingprep_key` setting is reused; do not place a key
in documentation or source. The paid-plan default paces all threads through one
client-side limiter at 300 requests per minute. An FMP HTTP 429 or rate-limit
payload waits for `Retry-After` (or `rate_limit_retry_seconds`) and then retries;
it does not consume the ordinary transient-error retry count. Set
`daily_request_limit` to `1` through `240` only for an account with a daily cap.
The default value `0` disables the obsolete free-tier daily budget. When enabled,
every outbound attempt reserves from the cross-process budget before network
access; exhaustion waits until `next_eligible_at` and retries. Reservations use
a MySQL advisory lock and the reusable
`refdata_fetch_state(provider_request_budget, FMP:DAILY)` row. Profile,
classification, and identifier calls share a bounded in-process profile payload
for immediate duplicate-work reduction only. The configured FMP constituent
endpoints are treated as authoritative snapshots only when a response is
complete, non-empty, and contains no discarded rows. Set an index's
`index_authoritative_snapshot` value to `false` if an FMP endpoint does not carry
the same full-snapshot guarantee; partial or empty results never close an absent
membership. Other Stock Analysis settings are:

```text
refdata.stock_analysis.sql.lock_wait_seconds
refdata.stock_analysis.sql.connection_timeout_seconds
refdata.stock_analysis.sql.connection_attempts
refdata.stock_analysis.sql.connection_retry_backoff_seconds
refdata.stock_analysis.sql.port
refdata.stock_analysis.sql.pool.enabled
refdata.stock_analysis.sql.pool.size
refdata.stock_analysis.news.cache_ttl_seconds
refdata.stock_analysis.read_cache.max_entries
refdata.stock_analysis.read_cache.fundamentals_ttl_seconds
refdata.stock_analysis.read_cache.overview_ttl_seconds
refdata.stock_analysis.refresh.profile_seconds
refdata.stock_analysis.refresh.classification_seconds
refdata.stock_analysis.refresh.identifiers_seconds
refdata.stock_analysis.refresh.filings_seconds
refdata.stock_analysis.refresh.financials_seconds
refdata.stock_analysis.refresh.market_cap_seconds
refdata.stock_analysis.refresh.index_constituents_seconds
refdata.stock_analysis.refresh.estimates_seconds
refdata.stock_analysis.refresh.guidance_seconds
refdata.stock_analysis.batch.max_workers
refdata.stock_analysis.batch.chunk_size
refdata.stock_analysis.batch.failure_threshold
```

Durable routes are database-first. Fresh rows and fresh negative-cache state make
no provider call. Stale or missing rows are refreshed under a bounded MySQL named
lock, rechecked after lock acquisition, and committed with fetch state in one
transaction. Provider failures never replace good rows; stale data is returned
with `stale=true` and a warning when possible.

For an existing database, run the idempotent supported-index seed after selecting
the reference-data schema:

```bash
mysql --database="$REFDATA_DATABASE" < algotrading/refdata/sql/stock_analysis/003_seed_supported_indexes.sql
```

The seed adds or updates `SP100` with source provider `iShares OEF` and preserves
an already populated benchmark security because the duplicate-key clause does
not update `benchmark_security_id`. The comma-separated/INI alias representation
uses `PROVIDER_SYMBOL=CANONICAL_SYMBOL`; a JSON object is accepted as well.

`/health/ready` requires both the legacy/core price component and a bounded
`SELECT 1` against durable Stock Analysis SQL. Provider availability remains
deferred. `/health/live` remains process-only.

News is deliberately non-durable: `/news` has no repository or fetch-state
write. Provider results are held only in the API's bounded process cache for 30
minutes by default, and the GUI uses the same configurable 30-minute TTL.

The Company Overview GUI requests `/overview?refresh=false&include_current_price=false`:
scheduled refresh jobs own provider synchronization, and the GUI obtains the
current mark from the batched quote endpoint. Database-only overview and
fundamental responses are cached for 30 minutes by default.

Overview calculates current forward P/E at request time as current price divided
by the mean EPS estimate for the next fiscal-year annual target. Missing/stale
estimates, non-positive EPS, ambiguous targets, missing price, and entitlement
failures produce a typed status instead of a misleading value. Neither P/E nor
profit margin is stored.

## Current-price and streaming behavior

At startup the application looks for `PARAM_STREAMING_PRICES_CLASS`. When it is
configured, the corresponding client is initialized and stored under
`PARAM_STREAMING_PRICES`.

If initialization fails:

- The failure is logged.
- The service continues by default.
- `retrieve_current_price` can fall back to the configured reference-data
  provider.
- Current-day one-minute history will only contain what the remaining
  `MarketData` path can supply.

Set `refdata.api.require_streaming_prices = True` if the service must refuse to
start without that connection.

## Responses and errors

Collection responses use:

```json
{
  "data": [],
  "meta": {
    "count": 0,
    "as_of": "2026-07-25T07:30:00-07:00",
    "timezone": "America/Los_Angeles",
    "cache": "miss"
  }
}
```

Every response includes an `X-Request-ID`. Data responses also include
`X-Cache`, `X-Data-As-Of`, and `Cache-Control`.

Errors do not expose Python tracebacks:

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "bars end must be on or after its start",
    "request_id": "..."
  }
}
```

## Tests

The core cache, serialization, and service tests use `unittest`. The HTTP
contract test additionally uses FastAPI's test client.

```bash
python -m pip install -r algotrading/refdata/api/requirements-test.txt

python -m unittest discover \
    -s algotrading/refdata/api/tests \
    -p 'test_*.py'

python -m unittest discover -s algotrading/refdata/repositories/tests -p 'test_*.py'
python -m unittest discover -s algotrading/refdata/providers/fmp/tests -p 'test_*.py'
python -m unittest discover -s algotrading/refdata/providers/sec/tests -p 'test_*.py'
python -m unittest discover -s algotrading/refdata/stock_analysis/tests -p 'test_*.py'
```

The tests use an in-memory fake `MarketData` implementation and do not access
the SQL database or an external market-data provider.

## Market Flow analytics

The read-only Market Flow API is rooted at
`/api/v1/analytics/market-flows`. It estimates primary-market ETF creations and
redemptions from NAV multiplied by the split-adjusted change in shares
outstanding. Its scope is the ETF universe configured in
`stock_analysis/etf_universe.py`; it is not a measure of all money entering or
leaving an asset class.

All routes accept optional `as_of=YYYY-MM-DD`. Routes with a `window` accept
`1d`, `1w`, `1m`, `3m`, or `6m`, representing 1, 5, 21, 63, or 126 completed
XNYS sessions. Results disclose the actual interval and separate source
watermarks. Every metric is an object with `status`, nullable `value`, `unit`,
and supporting dates/counts/warnings. An unavailable value is `null` and has a
machine-readable `reason`; clients must not treat it as zero.

`refdata.market_flow.neutral_pressure_bps` controls the finite, non-negative
market-state neutral band and defaults to `5`. The bounded raw/response cache
uses `refdata.market_flow.cache_ttl_seconds` (default `120`, maximum `300`).

### `GET /overview`

Query parameters: `window` (default `1d`) and `as_of`. The response contains
the market-flow state, ten headline measures, equity/fixed-income instruments,
economic sector comparisons, chart series, availability, coverage, and
methodology:

```json
{"data":{"requested_window":"1w","actual_start_date":"2026-08-26","actual_end_date":"2026-09-01","state":{"code":"confirmed_inflow","label":"Confirmed tracked-ETF inflow"},"headline":{"tracked_equity_etf_net_flow":{"status":"available","value":125000000,"unit":"usd"}},"instruments":[],"sector_comparison":[],"series":[]},"meta":{"cache":"miss","sql_query_count":7}}
```

### `GET /categories`

Query parameters: `window`, `as_of`, `view=all|mandate|economic`, optional
`level`, optional `parent_id`, `include_constituents` (default `false`), and
`limit` from 1 through 500. Constituent expansion requires a specific parent.
The hierarchy is deliberately bounded and intended for lazy traversal:

```json
{"data":{"requested_window":"1w","view":"all","parent_id":"root:tracked","nodes":[{"node_id":"asset:equity","parent_id":"root:tracked","level":"asset_class","name":"Equities","has_children":true,"capital_allocation":{"net_flow":{"status":"available","value":125000000,"unit":"usd"}}}]}}
```

Mandate flow assigns an ETF's activity to its declared mandate. Economic flow
allocates that same activity using the latest eligible point-in-time stored
weights. These are parallel views of the same flow and must never be added.
Missing portfolio weight remains an explicit residual rather than being
renormalized into known exposures.

### `GET /rotations`

Query parameters: `window`, `as_of`,
`dimension=sector|industry|capitalization|fixed_income`, and `limit` from 1
through 100. Rankings are cross-sectional rotation evidence, not traced dollar
transfers:

```json
{"data":{"requested_window":"1w","dimension":"sector","points":[{"node_id":"economic:sector:information-technology","relative_flow_pressure":{"status":"available","value":8.2,"unit":"bps"},"relative_performance":{"status":"available","value":0.012,"unit":"decimal_return"}}],"destinations":[],"sources":[],"evidence_pairs":[]}}
```

### `GET /etfs`

Query parameters: `as_of`, optional `asset_class=EQUITY|FIXED_INCOME`, and
optional configured `exposure_type`. This catalog supplies identity, source
availability, and supported windows for bounded selectors:

```json
{"data":{"as_of":"2026-09-01","actual_start_date":"2026-09-01","actual_end_date":"2026-09-01","asset_class":"EQUITY","etfs":[{"symbol":"SPY","fund_name":"SPDR S&P 500 ETF Trust","latest_reference_date":"2026-09-01","has_nav":true,"has_shares":true,"available_windows":[]}]}}
```

### `GET /etfs/{symbol}`

Query parameters: `window` and `as_of`. `symbol` must belong to the configured
tracked universe. The response separates primary-market flow from
secondary-market price/volume behavior and returns equity composition or
fixed-income characteristics as appropriate:

```json
{"data":{"symbol":"SPY","requested_window":"1w","identity":{"asset_class":"EQUITY","exposure_type":"BROAD_MARKET"},"primary_market":{"net_flow":{"status":"available","value":125000000,"unit":"usd"}},"secondary_market":{},"exposures":{"type":"equity"},"fixed_income_characteristics":null}}
```

FINRA data is a contextual off-exchange short-volume overlay. It is not short
interest, bearish positioning, or a count of opening short positions; FINRA
availability does not reduce core Market Flow confidence. Multi-session ETF
observations remain attributed to their observation end date, and a `1d`
metric is unavailable when the latest usable interval spans multiple sessions.

## Futures positioning analytics

The read-only futures analytics API is rooted at
`/api/v1/analytics/futures`. It calculates responses at request time from
`futures_products`, `futures_cftc_positions`, `futures_product_daily`, and
`futures_contract_daily`; it does not persist derived market-health data and
does not depend on the Risk API.

| Route | Important query parameters |
| --- | --- |
| `/positioning` | repeatable or comma-separated `products` and `participants`; `change_window=1w|4w|13w`; `history_window=1y|3y|5y`; `as_of`; `include_series` |
| `/participation` | repeatable or comma-separated `products` and `windows=1d,5d,20d`; `as_of`; `include_series` |
| `/term-structure` | `product` (default `VX`); `as_of`; `months` (default 6); `compare_sessions` (default 5) |
| `/crowding` | optional `products` and `participants`; `history_window`; `change_window`; `as_of`; `limit` |
| `/market-health/overview` | `position_window`; `history_window`; `as_of` |

CFTC positioning uses the report row's own open interest:
`net = long_positions - short_positions` and `net_oi = net / open_interest`.
The 1W, 4W, and 13W changes compare report observations, not exact calendar
dates. Position and change percentiles use the selected 1Y, 3Y, or 5Y window;
crowding is `abs(position_percentile - 50) * 2`. The overview's Direction
score combines 40% position level and 60% selected-window change, weighted 35%
each for ES/NQ Asset Manager and 15% each for ES/NQ Leveraged Money. Available
weights are renormalized, but both Asset Manager components remain required.

Daily settlement returns are roll-safe: each session selects an eligible
contract, then computes its return only from the same contract on the prior
eligible session. Daily OI is keyed by `open_interest_as_of_date`, deduplicated
by preferring `FINAL` and then the latest valid observation. Participation
classifies price/OI quadrants with configurable 0.25% price and 0.50% OI
thresholds. The overview weights each product's 5D state 65% and 20D state 35%,
then ES and NQ equally. VX curves use non-expired `MONTHLY` contracts only;
`M1_M2 = M2 / M1 - 1` and `M1_M3 = M3 / M1 - 1`.

All routes accept `as_of=YYYY-MM-DD` using the API timezone. CFTC rows are
eligible only when `available_at_utc` is no later than the resolved as-of end;
daily trade and OI-effective dates are also bounded by that date. Responses
disclose the actual report, settlement, OI-effective, and curve dates used.

Every derived metric carries `status`, nullable finite `value`, `unit`,
observation counts, data dates, and a stable reason when unavailable. Existing
but short histories return HTTP 200 with `insufficient_history`; absent source
data uses `unavailable`, and neither state is converted to zero. Coverage lists
requested/available products, earliest and latest dates, observation counts,
horizon capabilities, VX maturities, and stable warning codes. The bounded
in-process cache is controlled by
`refdata.futures.analytics.cache_ttl_seconds` and
`refdata.futures.analytics.cache_max_entries`.

## Risk-workbench ingress

Exact option quotes, lightweight expiration calendars, and tight live chains use the configured primary provider
(`TDAReferenceData` by default) and process-local single-flight caches.
Expiration discovery uses Schwab's expiration-chain resource and does not load contracts.
Chain requests are single-symbol, date/strike bounded, and make exactly one
`get_options_chain` call per cache miss. There is no chain warmer.

The FRED curve is enabled only with `refdata.fred.api_key`; configured series
are queried with bounded recent observations and normalized from percentage
points to simple annual decimals. CME reference data is disabled by default
and is ready only when `enabled`, `api_id`, `app_password`, and the entitled
`base_url` are all configured. CME uses the OAuth 2.0 client-credentials flow;
`token_url` defaults to the production CME authorization endpoint. Both adapters use
the XDG/configurable `refdata.api.cache.reference_path` SQLite WAL latest cache;
credentials and raw unbounded provider responses are not returned.

### CME Basic Plus morning settlements

`SAMorningUpdateMain` supports `refdata.futures.cme_feed = BASIC_PLUS` through
its existing CME stage. `EOD` retains the Standard EOD adapter. The deployment
configuration selects Basic Plus; the library fallback remains EOD.

Basic Plus downloads each available final exchange settlement CSV ZIP once per
business date for configured CME products. COMEX forward-settlement files,
preliminary files, options, and non-monthly instruments are excluded. Payloads
are processed entirely in memory and discarded after normalization: no download
archive, file manifest, or ingestion-state table is created.

Apply `017_seed_basic_plus_products.sql` to register the additional policy-rate
and Treasury roots (reference-data inserts only). Settlements use existing
`futures_contract_daily` rows and canonical keys such as `ZNZ2026`. Previous-day
volume and open interest are not written as current-day observations. Missing
fields preserve richer existing data, absent contracts are not deleted, and a
final Standard EOD settlement takes precedence over Basic Plus. Changed Basic
Plus settlements replace earlier Basic Plus values; identical records are not
rewritten merely because retrieval time changed.

Normal runs revisit 14 calendar days and catch up from the oldest stored
settlement date among an exchange's configured products, with a seven-day
correction overlap. `--backfill-futures` enumerates all currently available files
within `refdata.futures.history_years`; it cannot recover history CME no longer
exposes. `--force-refresh` does not widen that date range: Basic Plus always
redownloads selected available files because no local copies are retained.
`--dry-run` validates and computes proposed SQL changes without writing them.
Each product/date is committed atomically; file validation happens before any
writes from that file. Logs report available date ranges, late files, missing
products, failures, and insert/update counts. Existing facts establish coverage,
but do not prove all contracts were published for a date.

### ETF fund exposure

`GET /api/v1/etfs/{symbol}/exposure` returns normalized fund information,
sector weights, and country weights from the configured ETF provider. It
reports fund exposure rather than the issuer company's classification and
caches a successful result for 24 hours. A missing fund returns structured
`unavailable` data; provider failures retain the API's standard error envelope.
