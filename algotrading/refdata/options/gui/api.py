"""REST entry point: ``python -m algotrading.refdata.options.gui.api``."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .bootstrap import load_context
from .config import ApiConfig
from .app import create_app


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the read-only Options GUI REST API"
    )
    parser.add_argument("-e", "--environment", default="refdata")
    parser.add_argument("--host", help="Listen address; overrides configuration")
    parser.add_argument("--port", type=int, help="Listen port; overrides configuration")
    parser.add_argument(
        "--log-level",
        choices=["critical", "error", "warning", "info", "debug"],
    )
    parser.add_argument("--no-access-log", action="store_true")
    parser.add_argument(
        "--update-cache",
        action="store_true",
        help=(
            "Before serving, run a daily increment when the active generation "
            "was updated yesterday, or a weekly rebuild when an older "
            "generation has crossed a weekend"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "uvicorn is not installed; install gui/requirements.txt"
        ) from exc

    context = load_context(
        args.environment,
        logger_name="algotrading.refdata.options.gui.api",
    )
    config = ApiConfig.from_context(context)
    uvicorn.run(
        create_app(
            context=context,
            environment=args.environment,
            update_cache=args.update_cache,
        ),
        host=args.host or config.host,
        port=args.port or config.port,
        log_level=args.log_level or config.log_level.lower(),
        access_log=not args.no_access_log,
        workers=1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
