"""Delegate ``python -m algotrading.refdata.options.gui`` to the REST API."""

from .api import main


if __name__ == "__main__":
    raise SystemExit(main())
