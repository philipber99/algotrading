"""Generate, validate, inspect, and clean the options Time-of-Day cache."""

from __future__ import annotations

import argparse
import json
import logging
import traceback
from datetime import date
from enum import IntEnum
from typing import Any, Sequence

from ..bootstrap import configure_logging, load_context
from .builder import BuildOptions, CacheBuilder
from .config import CacheConfig
from .errors import ConfigurationError, SourceBacklogError
from .store import CacheStore


class ExitCode(IntEnum):
    OK = 0
    WARNING = 1
    CRITICAL = 2
    UNKNOWN = 3


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-e", "--environment", required=True)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="Generate a new cache generation")
    build.add_argument("mode", choices=["daily", "weekly"])
    build.add_argument("--symbols", help="Comma-separated configured-symbol subset")
    build.add_argument(
        "--session-date",
        action="append",
        type=date.fromisoformat,
        default=[],
        help="Completed daily session override; repeatable",
    )
    build.add_argument("--history-start", type=date.fromisoformat)
    build.add_argument("--history-end", type=date.fromisoformat)
    build.add_argument("--dry-run", action="store_true")
    build.add_argument("--force", action="store_true")
    build.add_argument("--no-publish", action="store_true")
    build.add_argument("--no-nagios", action="store_true")

    validate = commands.add_parser("validate", help="Validate an immutable generation")
    target = validate.add_mutually_exclusive_group(required=True)
    target.add_argument("--active", action="store_true")
    target.add_argument("--generation")

    status = commands.add_parser("status", help="Print active cache status")
    status.add_argument("--json", action="store_true")

    cleanup = commands.add_parser("cleanup", help="Remove safely unreferenced files")
    cleanup_mode = cleanup.add_mutually_exclusive_group()
    cleanup_mode.add_argument(
        "--dry-run",
        action="store_true",
        help="List reclaimable files without deleting them (default)",
    )
    cleanup_mode.add_argument(
        "--execute",
        action="store_true",
        help="Perform deletion; otherwise cleanup is a dry-run",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(argv)
    logger = configure_logging(None, "algotrading.refdata.options.gui.cache.cli")
    monitor: Any | None = None
    monitor_failed = False
    try:
        context = load_context(
            options.environment,
            logger_name="algotrading.refdata.options.gui.cache.cli",
        )
        logger = next(
            (
                value
                for value in context.values()
                if isinstance(value, logging.Logger)
            ),
            logger,
        )
        config = CacheConfig.from_context(context)
        store = CacheStore(config)
        if options.command == "validate":
            manifest = (
                store.load_active_manifest()
                if options.active
                else store.load_manifest(options.generation)
            )
            if manifest is None:
                raise ConfigurationError("No active cache generation exists")
            store.validate_manifest_files(manifest)
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "generation_id": manifest["generation_id"],
                        "data_through_session": manifest["data_through_session"],
                        "files": len(manifest.get("files", ())),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return int(ExitCode.OK)
        if options.command == "status":
            manifest = store.load_active_manifest()
            payload = {
                "status": "ok" if manifest else "not_built",
                "generation_id": manifest.get("generation_id") if manifest else None,
                "data_through_session": (
                    manifest.get("data_through_session") if manifest else None
                ),
                "build_mode": manifest.get("build_mode") if manifest else None,
                "symbols": manifest.get("symbols", []) if manifest else [],
                "windows": manifest.get("windows", []) if manifest else [],
                "files": len(manifest.get("files", ())) if manifest else 0,
            }
            if options.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print(
                    "Time-of-Day cache "
                    f"status={payload['status']} "
                    f"generation={payload['generation_id']} "
                    f"data_through={payload['data_through_session']} "
                    f"files={payload['files']}"
                )
            return int(ExitCode.OK if manifest else ExitCode.WARNING)
        if options.command == "cleanup":
            payload = store.cleanup(dry_run=not options.execute)
            print(json.dumps(payload, indent=2, sort_keys=True))
            return int(ExitCode.OK)

        if not options.no_nagios:
            monitor_context = dict(context)
            if "nagios.service_id" in monitor_context:
                monitor_context["nagios.service_id"] = (
                    str(monitor_context["nagios.service_id"])
                    + " "
                    + config.monitor_service_suffix
                )
            try:
                from algotrading.util.NagiosSystemMonitor import NagiosSystemMonitor

                monitor = NagiosSystemMonitor(monitor_context)
            except Exception:
                logger.warning("cache_monitor_initialization_failed", exc_info=True)
                monitor_failed = True

        from algotrading.refdata.options.Common import create_clickhouse_client

        client = create_clickhouse_client(context)
        try:
            symbols = (
                tuple(
                    item.strip().upper()
                    for item in options.symbols.split(",")
                    if item.strip()
                )
                if options.symbols
                else None
            )
            result = CacheBuilder(
                context,
                client=client,
                config=config,
                store=store,
                logger=logger,
            ).build(
                BuildOptions(
                    mode=options.mode,
                    symbols=symbols,
                    session_dates=tuple(options.session_date),
                    history_start=options.history_start,
                    history_end=options.history_end,
                    dry_run=options.dry_run,
                    force=options.force,
                    publish=not options.no_publish,
                )
            )
        finally:
            client.close()
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
        message = (
            f"Time-of-Day cache {result.mode} build completed; "
            f"generation={result.generation_id} "
            f"data_through={result.data_through_session} "
            f"symbols={len(result.symbols)} rows={result.row_count} "
            f"files={result.file_count} duration_s={result.duration_seconds:.2f} "
            f"queries={result.metrics.get('queries', 0)} "
            f"source_state_rows={result.metrics.get('state_rows', 0)} "
            f"source_surface_rows={result.metrics.get('surface_rows', 0)} "
            f"arrow_bytes={result.metrics.get('approximate_bytes', 0)} "
            f"no_op={result.no_op}"
        )
        if monitor is not None:
            try:
                from algotrading.util.NagiosSystemMonitor import NagiosSystemMonitor

                monitor.report_event(NagiosSystemMonitor.MONITOR_OK, message)
            except Exception:
                logger.exception("cache_monitor_delivery_failed")
                return int(ExitCode.WARNING)
        if monitor_failed:
            return int(ExitCode.WARNING)
        return int(ExitCode.OK)
    except SourceBacklogError as exc:
        logger.warning(
            "event=cache_build status=blocked reason=%s details=%s",
            exc,
            exc.details,
        )
        if monitor is not None:
            try:
                from algotrading.util.NagiosSystemMonitor import NagiosSystemMonitor

                monitor.report_event(
                    NagiosSystemMonitor.MONITOR_WARNING,
                    f"Time-of-Day cache not advanced; {exc}; details={exc.details}",
                )
            except Exception:
                logger.exception("cache_monitor_delivery_failed")
        return int(ExitCode.WARNING)
    except ConfigurationError as exc:
        logger.error("event=cache_configuration status=failed error=%s", exc)
        return int(ExitCode.UNKNOWN)
    except Exception as exc:
        logger.error("event=cache_build status=failed\n%s", traceback.format_exc())
        if monitor is not None:
            try:
                from algotrading.util.NagiosSystemMonitor import NagiosSystemMonitor

                monitor.report_event(
                    NagiosSystemMonitor.MONITOR_CRITICAL,
                    "Time-of-Day cache build failed; active generation unchanged; "
                    f"error={type(exc).__name__}:{exc}",
                )
            except Exception:
                logger.exception("cache_monitor_delivery_failed")
        return int(ExitCode.CRITICAL)


if __name__ == "__main__":
    raise SystemExit(main())
