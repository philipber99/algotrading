from __future__ import annotations

from fastapi import APIRouter, Request, Response

from ..analytics_models import CapabilitiesResponse
from ..app_identity import SERVICE_NAME
from ..service import SERVICE_VERSION


router = APIRouter(tags=["Capabilities"])


@router.get("/api/v1/capabilities", response_model=CapabilitiesResponse)
def capabilities(request: Request, response: Response):
    active = request.app.state.analytics_service
    now = active.now()
    response.headers["X-Cache"] = "BYPASS"
    response.headers["X-Data-As-Of"] = now.isoformat()
    response.headers["Cache-Control"] = "private, max-age=300"
    context = active.context
    intraday = bool(
        context.get("refdata.api.intraday_data_configured", context.get("refdata.api.streaming_prices_available", False))
    )
    # Advertise optional ingress features from the configured service readiness.
    live_options = getattr(request.app.state, "live_option_service", None)
    fred = getattr(request.app.state, "fred_curve_service", None)
    cme = getattr(request.app.state, "cme_reference_service", None)
    series_tenors = fred.series_tenors() if fred is not None else ()
    return {
        "data": {
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "capability_schema_version": "1.0",
            "available": {
                "market_analytics": True, "sector_analytics": True,
                "company_analytics": True,
                "finra_short_volume": callable(
                    getattr(active.repository, "market_short_volume", None)
                ),
                "finra_short_interest": callable(
                    getattr(active.repository, "company_short_interest", None)
                ),
                "news": True, "calendars": True,
                "prices": True, "fundamentals": True,
                "stock_screening": False,
                "risk_free_curve": bool(fred is not None and fred.ready),
                "futures_contract_specifications": bool(cme is not None and cme.ready),
                "options_chain": bool(
                    live_options is not None and live_options.option_chain_ready
                ),
                "exact_option_quotes": bool(
                    live_options is not None and live_options.option_quotes_ready
                ),
                "exact_option_contract_terms": bool(
                    live_options is not None and live_options.option_quotes_ready
                ),
                "exact_option_market_history": bool(
                    live_options is not None
                    and callable(
                        getattr(
                            getattr(live_options, "provider", live_options),
                            "get_bars",
                            None,
                        )
                    )
                ),
            },
            "supported": {
                "universes": ["SP100", "SP500", "NASDAQ100"],
                "benchmarks": ["SPY", "QQQ"],
                "periods": ["today", "1d", "1w", "1m", "3m", "6m", "1y", "5y"],
                "ranges": ["intraday", "1d", "1m", "3m", "6m", "1y", "3y", "5y"],
                "weightings": ["cap_weighted", "equal_weighted"],
                "groupings": ["sector", "industry"],
                "finra_short_volume_max_symbols": 500,
                "finra_short_volume_max_sessions": 252,
                "finra_short_interest_max_reports": 130,
                "options_chain_max_strike_count": int(
                    context.get("refdata.api.options_chain.max_strike_count", 120)
                ),
                "options_chain_max_date_span_days": int(
                    context.get("refdata.api.options_chain.max_date_span_days", 95)
                ),
                "option_market_history_max_days": int(
                    context.get("refdata.api.option_market_history_max_days", 376)
                ),
                "futures_spec_max_contracts": int(
                    context.get("refdata.cme_reference.max_contracts_per_request", 100)
                ),
                "fred_series_ids": [series for series, _ in series_tenors],
            },
            "intraday_data_configured": intraday,
        },
        "meta": {"as_of": now, "timezone": active.timezone, "cache": "bypass", "warnings": [], "sql_query_count": 0},
    }
