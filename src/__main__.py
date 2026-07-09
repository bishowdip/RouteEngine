"""Entry point so the engine runs as ``python -m src``.

Delegates to the command-line interface in :mod:`src.cli`.
"""

from src.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
