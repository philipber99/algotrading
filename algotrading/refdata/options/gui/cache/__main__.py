"""Default package entry point delegates to the cache-management CLI."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())

