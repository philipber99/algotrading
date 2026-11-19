"""Conservative, single-bucket ETF economic exposure classification."""

from __future__ import annotations

from collections import defaultdict
import math
import re
from typing import Any, Mapping

from algotrading.risk.positions.normalization import canonical_sector

CURATED_ETF_SECTORS = {
    "CIBR": "Technology",
    "SOXL": "Technology",
    "HEWJ": "International Equity",
    "TLH": "U.S. Treasuries",
    "TLT": "U.S. Treasuries",
}

_CANONICAL_SECTORS = frozenset({
    "Basic Materials", "Communication Services", "Consumer Cyclical",
    "Consumer Defensive", "Energy", "Financial Services", "Healthcare",
    "Industrials", "Real Estate", "Technology", "Utilities",
})
_INTERNATIONAL = re.compile(
    r"\b(?:international|non[ -]?u\.?s\.?|ex[ -]?u\.?s\.?|"
    r"developed markets?|emerging markets?|japan(?:ese)?|europe(?:an)?|"
    r"asia(?:n)?|pacific|latin america(?:n)?|china|chinese|india(?:n)?|"
    r"canada|canadian|united kingdom|british|australia(?:n)?|germany|german|"
    r"france|french|brazil|brazilian|south korea|korean)\b",
    re.IGNORECASE,
)
_BROAD = re.compile(r"\b(?:index|broad|total market|large[ -]?and[ -]?mid|large[ -]?cap)\b", re.IGNORECASE)
_LEVERAGED = re.compile(r"\b(?:leveraged|inverse|short|[23]x|[23]00%)\b", re.IGNORECASE)


def classify_etf_exposure(exposure: Mapping[str, Any] | None) -> tuple[str, str | None]:
    """Return an economic bucket and a stable reason when evidence is insufficient."""
    if not exposure or exposure.get("status") != "available":
        reason = str((exposure or {}).get("reason") or "provider_unavailable")
        return "Unclassified", reason if reason == "fund_info_missing" else "provider_unavailable"
    asset_class = str(exposure.get("asset_class") or "").casefold()
    objective = " ".join(str(exposure.get(key) or "") for key in ("fund_name", "objective"))
    # Classify the fund's holdings rather than the issuer's corporate sector.
    if "fixed income" in asset_class or "bond" in asset_class:
        if re.search(r"\b(?:u\.?s\.?|united states|us)\s+treasur(?:y|ies)\b", objective, re.IGNORECASE):
            return "U.S. Treasuries", None
        return "Fixed Income", None
    if "equity" not in asset_class:
        return "Unclassified", "asset_class_unsupported"
    weights: dict[str, float] = defaultdict(float)
    for row in exposure.get("sector_weights") or ():
        if not isinstance(row, Mapping):
            continue
        sector = canonical_sector(row.get("sector"))
        if sector not in _CANONICAL_SECTORS:
            continue
        try:
            weight = float(row.get("weight_percent"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(weight) and weight > 0:
            weights[sector] += weight
    total = sum(weights.values())
    # Require a near-complete sector breakdown before choosing a dominant exposure.
    if not 90 <= total <= 110:
        return "Unclassified", "sector_weights_incomplete"
    if weights:
        dominant, maximum = max(weights.items(), key=lambda item: item[1])
        if maximum >= 80:
            return dominant, None
        if (
            len(weights) >= 5 and maximum < 50
            and _INTERNATIONAL.search(objective)
            and _BROAD.search(objective)
            and not _LEVERAGED.search(objective)
        ):
            return "International Equity", None
    return "Unclassified", "mixed_or_unclear_exposure"
