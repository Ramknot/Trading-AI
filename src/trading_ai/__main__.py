"""Allow ``python -m trading_ai`` to invoke the diagnostic CLI."""

from trading_ai.cli import main

raise SystemExit(main())
