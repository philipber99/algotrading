"""Command line entry point for ``python -m algotrading.risk.api``."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the read-only algotrading Risk REST API"
    )
    parser.add_argument(
        "-e",
        "--environment",
        default="refdata",
        help="Configuration environment passed to load_configuration",
    )
    parser.add_argument("--host", help="Listen address; overrides Risk API config")
    parser.add_argument(
        "--port", type=int, help="Listen port; overrides Risk API config"
    )
    parser.add_argument(
        "--log-level",
        choices=["critical", "error", "warning", "info", "debug"],
        help="Uvicorn log level; overrides Risk API config",
    )
    args = parser.parse_args(argv)

    from .bootstrap import build_context
    from .config import RiskApiConfig

    try:
        context = build_context(args.environment, quiet_configuration=True)
        settings = RiskApiConfig.from_context(context)
    except Exception as exc:
        raise SystemExit(
            f"Risk API startup failed: configuration unavailable ({type(exc).__name__})"
        ) from None

    try:
        import uvicorn
    except ImportError:
        raise SystemExit(
            "uvicorn is not installed; install risk/api/requirements.txt"
        ) from None

    from .app import create_app

    host = args.host or settings.host
    port = args.port or settings.port
    if not 1 <= port <= 65535:
        raise SystemExit("Risk API startup failed: port must be between 1 and 65535")
    log_level = args.log_level or settings.log_level.lower()
    app = create_app(context=context, environment=args.environment, config=settings)
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        # The app logs route templates; Uvicorn would log sensitive path values.
        access_log=False,
        workers=1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
