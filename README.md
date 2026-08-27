# Trading AI

Trading AI is a safety-first, modular Python project for reproducible market research, paper trading, and future controlled execution. Lot 1 adds a historical Data Engine to the foundations delivered in Lots 0 and 0.1. It does not add strategies, indicators, backtesting logic, broker integration, or any real-order path.

## Safety guarantees

- `DEV`, `TEST`, `PAPER`, and `LIVE` remain separate environments.
- `balanced` is the active research/paper profile; `aggressive` remains disabled and unconditionally code-locked.
- `LIVE` startup has no unlock mechanism.
- Every future execution path must pass through `RiskEngine`; `DenyAllRiskEngine` remains the default and rejects every request.
- Strategies, ML, portfolio, and data components cannot call a broker directly.
- Secrets, local datasets, logs, environments, and caches are excluded from version control.

These are architectural safeguards, not a claim that trading systems or market data are risk-free.

## Structure

```text
config/profiles/                  configurable profiles and market universes
src/trading_ai/core/              models, configuration, safety policy, health, logging
src/trading_ai/data/
  base.py                         provider-neutral DataProvider contract
  engine.py                       orchestration, cache modes, retries, 4h derivation
  calendar.py                     exchange-calendar boundary and expected-bar logic
  quality.py                      normalization and fail-closed OHLCV validation
  resampling.py                   deterministic session-anchored 1h -> 4h aggregation
  storage.py                      Parquet datasets, manifests, and SHA-256 integrity
  providers/fake.py               deterministic offline test provider
  providers/yahoo.py              development historical-data adapter
src/trading_ai/strategies/        Strategy contract only
src/trading_ai/ml/                MLScorer contract only
src/trading_ai/portfolio/         PortfolioEngine contract only
src/trading_ai/risk/              mandatory RiskEngine and DenyAllRiskEngine
src/trading_ai/execution/         sealed risk-gated ExecutionEngine
src/trading_ai/brokers/           BrokerAdapter contract
src/trading_ai/backtesting/       Backtester contract only; Lot 2 is not started
tests/                            offline unit, architecture, and safety tests
```

The historical-data dependency flow is deliberately one-way:

```text
TradingProfile -> DataEngine -> DataProvider -> YahooFinanceProvider
                                      `-----> FakeDataProvider (tests)
```

Provider-specific pandas/yfinance objects are converted inside the adapter and never exposed to another engine. No strategy, backtester, portfolio engine, or risk engine imports `yfinance`.

## Installation

Use Python 3.11 or newer; CI runs Python 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

Lot 1 runtime dependencies are deliberately limited to `pandas`, `pyarrow`, `yfinance`, and `pandas-market-calendars`. `pytest` is the development dependency. Yahoo Finance is a development/research source, not a guaranteed production or live-trading feed.

## Data Engine

### Configured universe

Symbols and supported timeframes are authorized from the selected TOML profile. They are not embedded in the Data Engine. Balanced V1 currently enables `1h`, `4h`, and `1d` for:

```text
SPY, QQQ, IWM,
AAPL, MSFT, NVDA, AMZN, GOOGL, META,
ASML, SAP, MC.PA, AIR.PA
```

`YahooFinanceProvider` retrieves native `1h` and `1d` bars with `auto_adjust=False`. Yahoo availability varies by interval, especially for older intraday history. Each manifest records both the requested and actual range; missing expected bars produce warnings, and the engine never invents unavailable data.

The configured V1 list is a present-day research universe, not a historical point-in-time constituent dataset. Future backtests can therefore still have survivorship and selection bias. The provider/profile boundary and dataset manifests can later accept dated universes, but Lot 1 does not claim to solve that problem.

### Raw, adjusted, and corporate-action data

Normalized `MarketBar` values retain raw `open`, `high`, `low`, and `close`. `adjusted_close` is a separate optional field and is never silently substituted for a raw price. Dividends and stock splits are represented by immutable `Dividend` and `StockSplit` events and stored separately. Lot 1 collects these facts; it does not apply them to transactions or portfolios.

### Timezones, calendars, and 4h bars

All normalized and persisted timestamps are timezone-aware UTC values. Instrument metadata retains the provider exchange, source timezone, currency when available, and a market-calendar identifier. NYSE-family exchanges and Euronext Paris (`XPAR`) use distinct exchange calendars, so weekends, holidays, and closed sessions are not automatically labeled as missing data. Unknown exchange calendars fail explicitly.

The `4h` dataset is derived deterministically from normalized `1h` data. Buckets start at each exchange session open, never at midnight UTC, and never cross a session close. Each bucket uses first open, maximum high, minimum low, last close, summed volume, and last available adjusted close. A partial final session bucket remains explicit. The derived manifest records the exact source 1h dataset ID.

### Quality policy

Normalization sorts by `symbol`, `timeframe`, and UTC timestamp. Within a series, timestamps are strictly increasing. Quality checks detect invalid/missing OHLCV values, impossible price relationships, negative volume, naive timestamps, duplicate keys, expected missing bars, missing spans, and off-session bars.

Invalid, duplicate, naive, inconsistent-symbol/timeframe, or empty datasets fail with `DataValidationError`; they are never silently repaired or filled. Gaps remain unfilled and are reported as warnings when the surrounding exchange calendar indicates that a bar was expected. `DataQualityReport` exposes row counts, duplicate and invalid counts, missing expected bars, gap spans, timestamp bounds, timezone/sort state, warnings, and a `PASS`, `WARNING`, or `FAIL` status.

### Parquet storage, manifests, and cache

The default untracked store is:

```text
data_local/
  market/<provider>/<symbol>/<timeframe>/*.parquet
  derived/<symbol>/4h/*.parquet
  corporate_actions/<provider>/<symbol>/*.parquet
  manifests/*.json
```

Every Parquet dataset has a JSON manifest containing its dataset ID, provider and version, symbol, timeframe, requested/actual range, UTC download time, row count, source timezone, exchange/calendar, data kind, schema version, relative path, warnings, and SHA-256 checksum. Derived datasets additionally record lineage. Reads verify integrity before returning data.

Cache behavior is exact-range and deterministic:

- `CACHE_ONLY` performs no provider call and fails on a true cache miss; a missing 4h file may still be derived from an exact cached 1h source.
- `CACHE_FIRST` reuses a matching local dataset, otherwise fetches and stores it.
- `REFRESH` forces a bounded provider request and replaces the deterministic dataset entry.

Transient provider failures use a small bounded exponential retry. Project exceptions (`DataProviderError`, `DataUnavailableError`, `DataValidationError`, `DataIntegrityError`, and related subclasses) prevent yfinance, pandas, requests, or pyarrow exceptions from leaking through application layers.

## CLI

Safety diagnostics remain available:

```powershell
trading-ai doctor
trading-ai doctor --environment PAPER --profile balanced --json
```

Historical-data commands are explicit; no command downloads the full universe unless `--all` is supplied:

```powershell
trading-ai data fetch --profile balanced --symbol AAPL --timeframe 1d --start 2024-01-01 --end 2025-01-01
trading-ai data fetch --profile balanced --symbol AAPL --timeframe 4h --start 2024-06-01 --end 2024-07-01 --cache-mode CACHE_FIRST --json
trading-ai data fetch --profile balanced --all --timeframe 1d --start 2024-01-01 --end 2024-02-01
trading-ai data validate --symbol AAPL --timeframe 1d --json
trading-ai data inspect --symbol AAPL --timeframe 1d --json
```

Date-only CLI values are UTC day boundaries. Datetime values must carry `Z` or an explicit offset. `validate` and `inspect` only read local storage.

## Tests and CI

The default suite is deterministic, fast, and independent of Yahoo Finance and the network. It uses `FakeDataProvider` plus small synthetic sessions, invalid bars, gaps, corporate actions, and partial datasets.

```powershell
.\.venv\Scripts\python -m compileall -q src
.\.venv\Scripts\python -m pytest -m "not integration"
```

GitHub Actions performs the same compile and offline test checks on pushes and pull requests to `main`. A real Yahoo smoke test is opt-in only:

```powershell
$env:TRADING_AI_RUN_NETWORK_TESTS = "1"
.\.venv\Scripts\python -m pytest -m integration tests/test_yahoo_provider.py
```

## Profiles and initial scope

Balanced V1 prioritizes liquid index ETFs, liquid US large caps, and liquid European large caps. This favors liquidity, reasonable spreads, abundant history, straightforward future backtesting, and compatibility with a traditional broker while avoiding 24/7 crypto-market concerns at the start.

Crypto assets, options, futures, CFDs, significant leverage, high-frequency data, real-time feeds, bid/ask, order books, and alternative data are outside V1 Lot 1. They may be studied only in dedicated future work without weakening the balanced profile, the aggressive lock, or the mandatory risk boundary.

Expected safety matrix:

| Environment | Profile | Result |
| --- | --- | --- |
| PAPER | balanced | Configuration accepted; every order still denied |
| PAPER | aggressive | Blocked: profile disabled and code-locked |
| LIVE | balanced | Blocked: LIVE has no unlock mechanism |

## Roadmap

Lots 0, 0.1, and 1 provide foundations, universe/CI alignment, and historical data. Lot 2 will be the Backtesting Engine and is intentionally not implemented here. Later lots cover baseline quant, a real risk engine, regime detection, ML, portfolio construction, dashboarding, and broker/paper integration. Aggressive Research remains locked. See `PROJECT_STATE.md` for the authoritative status.
