"""Optional cache refresh policy used before the REST engine starts."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

from .builder import BuildOptions, BuildResult, CacheBuilder
from .config import CacheConfig
from .store import CacheStore


UTC = timezone.utc


@dataclass(frozen=True)
class CacheUpdateDecision:
    mode: str | None
    reason: str
    last_updated_date: date | None
    current_date: date


@dataclass(frozen=True)
class CacheUpdateOutcome:
    decision: CacheUpdateDecision
    result: BuildResult | None = None


def choose_update_mode(
    store: CacheStore,
    *,
    now: datetime | None = None,
) -> CacheUpdateDecision:
    """Choose the requested startup action from the active generation age.

    Dates are evaluated in the cache's configured local timezone. A missing
    cache receives a weekly bootstrap. A generation updated yesterday receives
    a daily increment. Older generations receive a weekly rebuild only when a
    Saturday or Sunday has occurred after their update date; otherwise the
    existing cache is used unchanged.
    """

    zone = ZoneInfo(store.config.local_timezone)
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    current_date = current.astimezone(zone).date()
    manifest = store.load_active_manifest()
    if manifest is None:
        return CacheUpdateDecision(
            "weekly",
            "no_active_generation",
            None,
            current_date,
        )

    last_updated = _manifest_updated_date(store, manifest, zone)
    if last_updated >= current_date:
        return CacheUpdateDecision(
            None,
            "cache_updated_today_or_later",
            last_updated,
            current_date,
        )
    if last_updated == current_date - timedelta(days=1):
        return CacheUpdateDecision(
            "daily",
            "cache_updated_yesterday",
            last_updated,
            current_date,
        )
    if _contains_weekend_after(last_updated, current_date):
        return CacheUpdateDecision(
            "weekly",
            "cache_is_older_than_one_day_and_weekend_elapsed",
            last_updated,
            current_date,
        )
    return CacheUpdateDecision(
        None,
        "cache_is_older_than_one_day_without_elapsed_weekend",
        last_updated,
        current_date,
    )


def update_cache_if_needed(
    context: Mapping[str, Any],
    *,
    logger: logging.Logger,
    now: datetime | None = None,
    client_factory: Callable[[Mapping[str, Any]], Any] | None = None,
) -> CacheUpdateOutcome:
    """Apply the startup policy and invoke the builder directly when needed."""

    config = CacheConfig.from_context(context)
    store = CacheStore(config)
    decision = choose_update_mode(store, now=now)
    logger.info(
        "event=cache_api_update_check action=%s reason=%s last_updated=%s current_date=%s",
        decision.mode or "use_existing",
        decision.reason,
        decision.last_updated_date,
        decision.current_date,
    )
    if decision.mode is None:
        return CacheUpdateOutcome(decision)

    if client_factory is None:
        from algotrading.refdata.options.Common import create_clickhouse_client

        client_factory = create_clickhouse_client
    client = client_factory(context)
    try:
        result = CacheBuilder(
            context,
            client=client,
            config=config,
            store=store,
            logger=logger,
        ).build(BuildOptions(mode=decision.mode))
    finally:
        client.close()
    logger.info(
        "event=cache_api_update_completed mode=%s generation=%s data_through=%s "
        "no_op=%s files=%s rows=%s duration_seconds=%.3f",
        decision.mode,
        result.generation_id,
        result.data_through_session,
        result.no_op,
        result.file_count,
        result.row_count,
        result.duration_seconds,
    )
    return CacheUpdateOutcome(decision, result)


def _manifest_updated_date(
    store: CacheStore,
    manifest: Mapping[str, Any],
    zone: ZoneInfo,
) -> date:
    built_at = manifest.get("built_at_utc")
    if built_at:
        try:
            parsed = datetime.fromisoformat(str(built_at).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed.astimezone(zone).date()
        except ValueError:
            pass

    generation_id = str(manifest.get("generation_id", ""))
    manifest_path = store.manifests / f"{generation_id}.json"
    try:
        modified = datetime.fromtimestamp(manifest_path.stat().st_mtime, tz=UTC)
        return modified.astimezone(zone).date()
    except (OSError, ValueError, OverflowError):
        pass

    data_through = manifest.get("data_through_session")
    try:
        return date.fromisoformat(str(data_through))
    except ValueError as exc:
        raise ValueError(
            "Active cache manifest has no usable update timestamp"
        ) from exc


def _contains_weekend_after(last_updated: date, current: date) -> bool:
    cursor = last_updated + timedelta(days=1)
    while cursor <= current:
        if cursor.weekday() >= 5:
            return True
        cursor += timedelta(days=1)
    return False
