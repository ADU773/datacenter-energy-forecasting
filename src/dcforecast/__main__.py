"""Allow ``python -m dcforecast``."""

from dcforecast.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
