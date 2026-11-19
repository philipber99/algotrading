# Trading System Code Examples

This repository contains selected Python packages from a trading system I have developed. This code is published to showcase my UI design, software architecture, and coding skills. The code is part of a fully operational system. As such, there are additional software components required to run the system. Operational demonstration copies of the GUIs are available through the links below.

![System Architecture](<images/Trading System Architecture.png>)

The system was designed without AI assistance however much of the code was generated using ChatGPT Codex. I have included the Trading Engine package which pre-dates AI for generating code. 

## Examples

| Example | GUI demonstration | Backend source | Purpose |
| --- | --- | --- | --- |
| Portfolio risk | [Risk GUI](https://github.com/philipber99/risk-gui-demo) | [Risk API](algotrading/risk/api/) | Risk calculations and middleware services for backend data. |
| Stock analysis | [Stock Analysis GUI](https://github.com/philipber99/stock-gui-demo) | [Reference Data API](algotrading/refdata/api/) | Middleware services for backend data. |
| Options analytics | [Volatility GUI](https://github.com/philipber99/vol-gui-demo) | [Options](algotrading/refdata/options/) | Option-chain collection, pricing, volatility surfaces, historical profiles, and spread screening. |
| Trade execution | — | [Trading Engine](algotrading/trading/) | Automated order execution. |

The Trading Engine code is 'unvarnished': it has no AI clean-up passes other than a security sweep to ensure no sensitive information is present in the code.

## Technology summary

The wider trading system uses the following technologies:

| Area | Technologies |
| --- | --- |
| Languages | Python, TypeScript, JavaScript, SQL, Bash |
| Machine learning | PyTorch, CUDA, scikit-learn, LightGBM, XGBoost, Optuna, TensorBoard, Captum, SHAP |
| Data engineering | NumPy, pandas, Polars, SciPy, PyArrow, Parquet, Zarr, Dask, MySQL, ClickHouse, SQLite |
| APIs and applications | FastAPI, Pydantic, REST, WebSockets, asyncio, OAuth 2.0, React, Electron, Node.js, Vite, Tailwind CSS |
| Visualization and testing | Plotly, Matplotlib, pytest, Playwright |
| Quantitative finance | QuantLib |

## Backend components

The examples cover middleware components that facilitate GUI accces to
back-end services such as reference data and risk calculations. Each GUI has
its own dedicated process with a REST API for communication.

## What is included

| Package | Purpose |
| --- | --- |
| [`algotrading.risk`](algotrading/risk/) | Portfolio exposure, scenario analysis, historical risk, option valuation integration, P&L reconstruction, and hedge research. |
| [`algotrading.refdata.api`](algotrading/refdata/api/) | REST interfaces for market and reference data, with request validation, provider integration, caching, and managed background updates. |
| [`algotrading.refdata.options`](algotrading/refdata/options/) | Option-chain ingestion and history, implied volatility and Greeks, volatility-surface calibration, historical profile caches, and multi-leg spread screening. |
| [`algotrading.trading`](algotrading/trading/) | Strategy-driven trade instructions, order execution, broker integration, exposure checks, state reconciliation, and trading reports. |

These are source excerpts from a larger application. Shared infrastructure and
runtime configuration are outside the scope of this repository; see
[Runtime context](#runtime-context) below.

## Portfolio risk analysis

The risk package is responsible for real-time risk updates, scenario analysis, 
trade post-mortems, and factor hedging. It provides a REST API to the GUI while 
using SQL for database retrieval. Some information may be retrieved directly 
from the broker via a REST API.

The main areas include:

- **Position normalization and instrument terms.** Represent equities, ETFs,
  mutual funds, options, and futures in common economic terms such as
  liquidation value, notional exposure, and delta-equivalent exposure distinct.
- **Portfolio analytics.** Calculate exposures and option sensitivities, evaluate
  market scenarios, estimate weighted historical value at risk (VaR) and
  expected shortfall, fit factor exposures, and identify correlated risk groups.
- **Option valuation and history.** Integrate options by obtaining IV / greeks
- through options pricing calculations (Tian Trees and CRR).
- **P&L reconstruction and attribution.** Reconstruct trade lifecycles from
  transaction records and reconcile marked-value changes with cash flows and
  modeled drivers.
- **Hedge research.** Assess risk on selected positions, provide hedging
  recommendations, and evaluate the impact of candidate hedges.
- **Operational state.** Data quality and transaction history auditing.

The [Risk API documentation](algotrading/risk/api/README.md) describes its
contracts and calculation behavior. The
[P&L methodology](algotrading/risk/pnl/METHODOLOGY.md) explains lifecycle
reconstruction, valuation boundaries, and attribution in more detail.

## Reference data API

The reference data package provides FastAPI routes and service adapters over the
larger application's data providers and repositories. It includes:

- Market status, exchange calendars, quotes, historical bars, and company events.
- Canonical OCC option symbols, exact-contract quotes and history, contract
  terms, option chains, and expiration calendars.
- Risk-free curve inputs, historical curve queries, and futures contract
  specifications.
- API contracts for company, sector, market-flow, and futures analytics backed
  by shared services in the larger application.
- Bounded caches, duplicate-load suppression, persistent history coverage, and
  lifecycle-managed calendar update workers.

The [Reference Data API documentation](algotrading/refdata/api/README.md)
contains the endpoint descriptions and integration settings.

## Options data and analytics

The options package covers collection, numerical analysis, durable history, and
read-only services for options research. Its main areas include:

- **Collection and publication.** Capture Schwab option chains, preserve raw
  responses for recovery, normalize contracts and observations, and publish
  completed snapshots to ClickHouse. Reconciliation recovers work from durable
  records after interruptions.
- **Pricing and sensitivities.** Calculate implied volatility and Greeks using
  configurable American and European pricing routes. The numerical engines
  include Tian and Cox–Ross–Rubinstein (CRR) trees and Crank–Nicolson finite
  differences, with shared solver configuration and precision controls.
- **Volatility surfaces.** Fit per-expiration smiles with SVI or PCHIP, generate
  standardized surface nodes, and record butterfly and calendar arbitrage
  diagnostics. Calculation definitions, attempts, and completion states identify
  which results are available for downstream queries.
- **Historical storage and maintenance.** Archive and verify raw history,
  support resumable monthly compaction, and build immutable Parquet generations
  for time-of-day analysis. Incremental and full builds publish an active
  generation for readers.
- **Historical strategy profiles.** Query cached observations to normalize
  option legs and compare time-of-day behavior across selected sessions,
  including coverage diagnostics and bounded result caching.
- **Multi-leg spread screening.** Generate and evaluate verticals, calendars,
  diagonals, butterflies, iron condors, and double calendars from a completed
  snapshot. Compare execution quality, scenario P&L, conditional probability
  estimates, and optional static delta hedges within configured work limits.
- **Read-only API services.** Serve historical profiles and spread-screening
  requests through FastAPI. Cache and screener initialization are independent,
  allowing historical cache routes to remain available during a screener outage.

Screening probabilities depend on the scenario model and its assumptions.
Screening and profile services provide research results and do not submit orders.

The [Options GUI API documentation](algotrading/refdata/options/gui/README.md)
describes the service endpoints, configuration, and calculation conventions. The
[surface-node methodology](algotrading/refdata/options/SURFACE_NODE_HYBRID.md)
explains how proposed American-option surface nodes are numerically validated.

## Trading Engine

The trading package coordinates strategy evaluation and broker execution. It
models a trade instruction, describing a desired entry and exit, separately from
the broker orders used to carry it out. Its main areas include:

- **Strategy integration and trade sizing.** Convert strategy results into long
  and short trade instructions using configured position, gross, and net
  exposure limits, trading windows, and optional market-event and spread filters.
- **Order lifecycle management.** Create entry and exit orders, schedule
  activation and expiry, manage bracket and one-cancels-the-other (OCO) orders,
  and handle cancellations, partial fills, and bounded trade retries.
- **Order netting and fill allocation.** Group eligible market orders by symbol,
  offset opposing internal orders, submit the remaining net quantity to the
  broker, and allocate broker fills back to the original orders.
- **Broker coordination.** Use a common broker interface with an included
  Alpaca adapter, process streamed order notifications, and poll for order
  updates to reconcile execution state.
- **Persistence and reconciliation.** Maintain trade instructions, orders, and
  P&L records through cached MySQL-backed objects. Startup and shutdown checks
  compare stored orders and trade state with broker orders and positions.
- **Operational checks and reporting.** Check exposure thresholds, repeated
  cancellations or rejections, and excessive broker API calls. Generate
  post-trading reports covering P&L and open positions.

[AlgoTradingMain.py](algotrading/trading/AlgoTradingMain.py) coordinates
configuration, broker initialization, validation, processing, and reporting.
[Processing.py](algotrading/trading/Processing.py) contains the execution loop
that advances trade instructions, submits or expires orders, and consumes
broker updates. The included
[TestBrokerInterface](algotrading/trading/TestBrokerInterface.py) simulates broker
orders and fills for integrated testing, including partial-fill scenarios.

## Engineering choices

Several concerns recur throughout the implementation:

- **Consistent snapshots.** Portfolio analyses bind to a snapshot and its
  configuration so related calculations use the same position state.
- **Explicit units.** Quantities, multipliers, dollar exposures, volatility
  points, basis points, and time conventions are represented deliberately.
- **Causal history.** Historical calculations respect observation availability
  and exchange sessions, with checks against using future information.
- **Visible data quality.** Results carry coverage, provenance, warnings, and
  methodology identifiers. Missing inputs can produce an explicit
  `insufficient_data` result instead of an apparently complete estimate.
- **Bounded external work.** Provider requests, retries, concurrency, history
  windows, response sizes, and cache growth have defined limits.
- **Testable boundaries.** Calculation logic is separated from broker, provider,
  database, and HTTP concerns so those boundaries can be exercised independently.

## Suggested reading

| Start here | What it illustrates |
| --- | --- |
| [Risk models](algotrading/risk/models.py) | Immutable position and snapshot contracts, instrument terms, and data-quality metadata. |
| [Portfolio service](algotrading/risk/api/portfolio_service.py) | Assembling broker positions and reference inputs into a coherent portfolio snapshot. |
| [Historical revaluation](algotrading/risk/history/revaluation.py) | Applying historical shocks to current holdings with causal windows and per-position coverage. |
| [P&L methodology](algotrading/risk/pnl/METHODOLOGY.md) | The accounting and modeling decisions behind lifecycle P&L and attribution. |
| [Reference data service](algotrading/refdata/api/service.py) | Provider and repository boundaries, batching, session selection, and fallback behavior. |
| [Reference data caches](algotrading/refdata/api/cache.py) | Thread coordination, duplicate-load suppression, persistent coverage, and expiration. |
| [Options collector](algotrading/refdata/options/Collector.py) | Raw capture, normalized publication, and recovery through durable snapshot records. |
| [Options worker and repositories](algotrading/refdata/options/Worker.py) | Coordinating numerical work with durable calculation and surface publication state. |
| [Volatility surfaces](algotrading/refdata/options/Surface.py) | Implied-volatility inversion, smile fitting, arbitrage diagnostics, and standardized node generation. |
| [Spread screener](algotrading/refdata/options/gui/screener/service.py) | Bounded candidate selection, scenario evaluation, caching, and result diagnostics. |
| [Trade instruction management](algotrading/trading/TradeManagement.py) | Turning strategy results into sized trade instructions and managing entry, retry, and closeout decisions. |
| [Order management](algotrading/trading/OrderManagement.py) | Order construction, exposure checks, netting, broker submission, and fill processing. |
| [Execution validations](algotrading/trading/Validations.py) | Broker and database reconciliation, position checks, and runtime circuit breakers. |

## Tests

Tests accompany the implementation in
[`risk/tests`](algotrading/risk/tests/),
[`risk/api/tests`](algotrading/risk/api/tests/),
[`refdata/api/tests`](algotrading/refdata/api/tests/),
[`options/tests`](algotrading/refdata/options/tests/),
[`options/gui/cache/tests`](algotrading/refdata/options/gui/cache/tests/), and
[`options/gui/screener/tests`](algotrading/refdata/options/gui/screener/tests/).

They cover calculation invariants, financial units, historical availability,
P&L reconciliation, API contracts, cache behavior, options strategy evaluation,
and sensitive-data handling in errors.
Many use deterministic fixtures, fake providers, injected clocks, or temporary
databases to exercise behavior without live market connections.

## Runtime context

These examples are intended for source review. Running the complete services
requires additional components from the larger `algotrading` application,
including configuration loading, broker adapters, provider registries, SQL
repositories, and shared pricing infrastructure. A complete runtime also needs
the relevant databases and provider configuration.

The Trading Engine also depends on strategy definitions, trading database tables,
and market-data streams supplied by the full application.

The options pipeline also requires ClickHouse, configured chain and reference-rate
providers, and storage for archive and Parquet cache generations.

The implementation uses Python, FastAPI, Pydantic, NumPy, pandas, HTTPX, SQLite,
MySQL, ClickHouse, and PyArrow/Parquet, with adapters for option-pricing components. The package-level
requirements files describe service-specific dependencies; they do not describe
a standalone installation of the full application.

The package READMEs retain details of the original integrated environment and
provide additional context for reviewing the code.
