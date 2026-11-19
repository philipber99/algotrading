"""Command-line entry point: python -m algotrading.refdata.api."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .app import create_app
from .bootstrap import build_context


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the algotrading reference-data REST API"
    )
    parser.add_argument(
        "-e",
        "--environment",
        default="refdata",
        help="Configuration environment passed to load_configuration",
    )
    parser.add_argument("--host", help="Listen address; overrides configuration")
    parser.add_argument(
        "--port",
        type=int,
        help="Listen port; overrides configuration",
    )
    parser.add_argument(
        "--log-level",
        choices=["critical", "error", "warning", "info", "debug"],
        help="Uvicorn log level",
    )
    parser.add_argument(
        "--no-access-log",
        action="store_true",
        help="Disable Uvicorn's HTTP access log",
    )
    args = parser.parse_args(argv)

    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "uvicorn is not installed; install refdata/api/requirements.txt"
        ) from exc

    context = build_context(args.environment, quiet_configuration=True)
    host = args.host or str(context.get("refdata.api.host", "127.0.0.1"))
    port = args.port or int(context.get("refdata.api.port", 8787))
    log_level = (
        args.log_level
        or str(context.get("refdata.api.uvicorn_log_level", "info")).lower()
    )

    app = create_app(context=context, environment=args.environment)
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        access_log=not args.no_access_log,
        workers=1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
