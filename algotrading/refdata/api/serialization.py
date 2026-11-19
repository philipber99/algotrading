"""Conversion helpers for Pandas/MySQL values returned by MarketData."""

from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd


def resolve_timezone(value: Any) -> tzinfo:
    if isinstance(value, tzinfo):
        return value
    if isinstance(value, str) and value:
        try:
            return ZoneInfo(value)
        except ZoneInfoNotFoundError:
            return datetime.now().astimezone().tzinfo or timezone.utc
    return datetime.now().astimezone().tzinfo or timezone.utc


def timezone_name(value: tzinfo) -> str:
    return getattr(value, "key", None) or str(value)


def aware_datetime(value: Any, local_timezone: tzinfo) -> datetime:
    timestamp = pd.Timestamp(value)
    # Interpret naive database timestamps in the service's configured local timezone.
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(local_timezone)
    else:
        timestamp = timestamp.tz_convert(local_timezone)
    return timestamp.to_pydatetime()


def marketdata_datetime(value: datetime | date, local_timezone: tzinfo) -> datetime:
    """Convert a client value to the naive local time expected by MarketData."""
    if isinstance(value, date) and not isinstance(value, datetime):
        return datetime.combine(value, time.min)

    result = value
    if result.tzinfo is not None:
        result = result.astimezone(local_timezone).replace(tzinfo=None)
    return result


def json_value(value: Any, local_timezone: tzinfo) -> Any:
    if value is None or value is pd.NaT:
        return None

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return aware_datetime(value, local_timezone).isoformat()

    if isinstance(value, datetime):
        return aware_datetime(value, local_timezone).isoformat()

    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, (pd.Timedelta, timedelta)):
        return value.total_seconds()

    if isinstance(value, Decimal):
        return float(value)

    # Represent nonfinite numeric values as JSON null.
    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, dict):
        return {
            str(key): json_value(item, local_timezone) for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [json_value(item, local_timezone) for item in value]

    return value


def dataframe_records(
    dataframe: pd.DataFrame | None,
    local_timezone: tzinfo,
    index_field: str | None = None,
) -> list[dict[str, Any]]:
    if dataframe is None or len(dataframe) == 0:
        return []

    working = dataframe.copy()

    # Promote a meaningful DataFrame index into the requested public timestamp field.
    if (
        index_field is not None
        and index_field not in working.columns
        and not isinstance(working.index, pd.RangeIndex)
    ):
        original_name = working.index.name
        working = working.reset_index()
        index_column = original_name or "index"
        if index_column != index_field:
            working.rename(columns={index_column: index_field}, inplace=True)

    records: list[dict[str, Any]] = []
    for raw_record in working.to_dict(orient="records"):
        records.append(
            {
                str(key): json_value(value, local_timezone)
                for key, value in raw_record.items()
            }
        )
    return records


def scalar_price(value: Any) -> float | None:
    if value is None:
        return None

    if isinstance(value, pd.DataFrame):
        if len(value) == 0 or "close" not in value.columns:
            return None
        value = value.iloc[-1]["close"]
    elif isinstance(value, pd.Series):
        if len(value) == 0:
            return None
        value = value.iloc[-1]

    if isinstance(value, np.generic):
        value = value.item()

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    return result if math.isfinite(result) else None
