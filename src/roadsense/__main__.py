"""Allow ``python -m roadsense`` to invoke the CLI."""

from roadsense.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
