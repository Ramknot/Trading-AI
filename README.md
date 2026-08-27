# Trading AI

Trading AI is a safety-first, modular Python project for reproducible market research, paper trading, and future controlled execution. Lot 3 adds a shared Feature Engine and three explainable quantitative research baselines to the foundations, market-data pipeline, and deterministic Backtesting Engine delivered in Lots 0 through 2. It does not add optimization, a real Risk Engine, regime classification, machine learning, broker integration, or any real-order path.

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
src/trading_ai/features/          shared trend, momentum, volatility, volume, structure features
src/trading_ai/strategies/        contracts, configs, sizing, registry, and Lot 3 baselines
src/trading_ai/ml/                MLScorer contract only
src/trading_ai/portfolio/         PortfolioEngine contract only
src/trading_ai/risk/              mandatory RiskEngine and DenyAllRiskEngine
src/trading_ai/execution/         sealed risk-gated ExecutionEngine
src/trading_ai/brokers/           BrokerAdapter contract
src/trading_ai/backtesting/
  engine.py                       chronological event loop and safety validation
  strategy.py                     look-ahead-safe strategy API and technical CLI demo
  execution.py                    MARKET/LIMIT fills, spread, slippage, commissions
  portfolio.py                    cash-only long ledger, positions, PnL, equity
  trades.py                       FIFO closed-trade reconstruction
  metrics.py                      independent performance metrics
  benchmark.py                    dedicated Buy & Hold benchmark
  input.py                        in-memory and exact-cache offline dataset adapters
  storage.py                      JSON/Parquet result export and SHA-256 verification
tests/                            offline unit, architecture, and safety tests
```

The historical-data dependency flow is deliberately one-way:

```text
TradingProfile -> DataEngine -> DataProvider -> YahooFinanceProvider
                                      `-----> FakeDataProvider (tests)
```

Provider-specific pandas/yfinance objects are converted inside the adapter and never exposed to another engine. No strategy, backtester, portfolio engine, or risk engine imports `yfinance`.

The historical-simulation flow is separate from both providers and brokers:

```text
DataEngine -> validated BacktestDataset -> BacktestEngine
                                           |-> FeatureEngine (past + present only)
                                           |-> BacktestStrategy (feature snapshots/signals)
                                           |-> ExecutionModel
                                           |-> PortfolioLedger
                                           |-> MetricsEngine
                                           `-> BacktestResult / local export
```

The simulated `OrderIntent` boundary is not permission to transmit an order. It cannot reach `BrokerAdapter`, and the real execution architecture remains guarded by the mandatory `RiskEngine` with `DenyAllRiskEngine` as its default.

## Installation

Use Python 3.11 or newer; CI runs Python 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

Runtime dependencies remain deliberately limited to the Lot 1 set: `pandas`, `pyarrow`, `yfinance`, and `pandas-market-calendars`. Lots 2 and 3 add no dependency or external backtesting/indicator framework; features and numerical metrics use the Python standard library. `pytest` is the development dependency. Yahoo Finance is a development/research source, not a guaranteed production or live-trading feed.

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

## Backtesting Engine

### Chronology and strategy boundary

`BacktestEngine` accepts only already-normalized `BacktestDataset` objects, either assembled in memory for tests or loaded from an exact Lot 1 cache entry. It never calls a `DataProvider`, downloads data, or imports Yahoo/broker code. Bars from multiple symbols are merged deterministically by UTC timestamp, then symbol and timeframe. At each bar the strategy receives an immutable `StrategyContext` containing the current bar, an immutable portfolio snapshot, and history whose timestamps are all less than or equal to the current timestamp. No future bar, next close, or full future DataFrame exists in the normal strategy API.

Lot 2 runs one configured primary timeframe per simulation. Any of the Balanced `1h`, `4h`, or `1d` datasets can be the primary stream, and every event identifies its timeframe. Supplying secondary timeframes in the same run fails explicitly until a later, carefully specified multi-timeframe scheduler is added. Multi-asset runs on the same primary timeframe are supported.

### Backtest assumptions

Every result records these assumptions through immutable `BacktestConfig` and dataset references:

- A signal produced after processing bar `t` cannot fill on that bar. A `MARKET` order first becomes eligible on the next later bar for the same symbol/timeframe and fills completely at that bar's open before costs.
- A `BUY LIMIT` or `SELL LIMIT` stays `PENDING` until a later bar reaches it, or until an optional eligible-bar expiry. V1 applies a deterministic full fill at the limit, not a favorable intrabar price. When spread/slippage are non-zero, the raw touch must be sufficient for the submitted limit to remain the all-in execution-price bound.
- Spread and slippage are configurable in basis points and applied adversely to buys and sells inside `BarExecutionModel`. Commission supports a fixed amount, percentage basis points, and an optional minimum. Zero-cost reference runs remain possible. Costs are always recorded on fills and in metrics.
- Balanced V1 is cash-only and long-only. A buy that exceeds available cash is rejected, cash cannot become negative, and a sell larger than the held position is rejected. There is no margin, leverage, shorting, market depth, realistic partial fill, latency, market impact, or order-book model.
- `STOP` and `STOP_LIMIT` enum values are reserved for future execution models but are rejected by the Lot 2 order-intent model. V1 accepts only `MARKET` and `LIMIT`.
- Raw OHLC is used with explicit dividends and splits. `adjusted_close` is never substituted. A held long position receives an explicit dividend ledger credit; a split changes quantity and average entry price without economic PnL. Adjusted-only inputs are rejected so price adjustment and actions cannot be counted twice.
- Gaps are never filled or forward-filled. `DataQualityReport.FAIL` always blocks a run. `WARNING` blocks under the default `STRICT` policy and may continue, with warnings preserved in the result, only under `ALLOW_WARNINGS`.

These simplifying assumptions are conservative and deterministic, not a claim of achievable real-market execution. A backtest result is not evidence of future profitability.

### Ledger, fills, trades, and equity

`PortfolioLedger` is the only component that mutates simulated cash and positions. Each fill, dividend, and split creates an immutable ledger entry. It tracks cash, quantity, average entry price, net realized PnL, unrealized PnL, dividend income, and an equity point per market timestamp. FIFO reconstruction retains partial exits and produces closed `Trade` records with entry/exit times, effective prices, quantity, gross and net PnL, fees, spread/slippage attribution, return, and holding period. The V1 execution model fills eligible orders in full, while trade reconstruction safely supports partial position exits across multiple complete fills.

### Metrics and benchmark

The independent `MetricsEngine` calculates initial capital, final equity, total and annualized return, annualized volatility, Sharpe, Sortino, maximum drawdown and recovery metadata, Calmar, profit factor, win rate, average win/loss, expectancy, trade count, turnover, exposure, best/worst trade, average holding period, cost totals, and dividend income. Undefined cases such as no losing trade, no downside, zero variance, or too few observations return `None` rather than a misleading number.

Annualization is centralized: `252` observations for daily equity data, `504` for 4h, and `1638` for 1h, as documented equity-market research conventions. The risk-free rate is an explicit configuration value and defaults to zero; no external rate is fetched.

The optional `BuyAndHoldBenchmark` is a dedicated benchmark component, not an operational strategy. Its symbol is configurable and requires an explicitly supplied dataset on the same primary timeframe. It uses the simulation period's raw closes and explicit corporate actions, and reports return, maximum drawdown, and strategy excess return.

### Provenance, reproducibility, and exports

Every `BacktestResult` records strategy name/version/parameters, feature schema/engine lineage for Lot 3 baselines, exact dataset IDs and SHA-256 checksums, corporate-action lineage, simulation config, historical start/end, optional Git commit SHA, a SHA-256 of the current Python source tree, explainable signals, orders, fills, trades, ledger entries, warnings, benchmark, and a stable run/result hash. The source hash makes an uncommitted/dirty run distinguishable even when `HEAD` still names an older commit. Technical creation time is excluded from the deterministic hash, so identical strategy parameters, source/code version, config, and datasets produce identical features, signals, orders, fills, trades, equity, metrics, run ID, and result hash.

Exports stay below the Git-ignored local tree:

```text
data_local/backtests/<run_id>/
  summary.json
  equity.parquet
  orders.parquet
  fills.parquet
  trades.parquet
  signals.parquet
  ledger.parquet
  checksums.json
```

`BacktestResultStore` verifies each exported file against its SHA-256 checksum before inspection. No result or market dataset is versioned.

### Lot 2 limitations

Lot 2 deliberately has no strategy parameter optimization, grid search, walk-forward training, ML comparison, automatic end-of-run liquidation, or real liquidity model. Trade metrics describe closed FIFO trades; an open final position remains visible in equity and exposure but is not fabricated into a closing trade. Profile-level sizing rules such as maximum positions, maximum exposure, risk budget, and turnover budget remain responsibilities of the future Portfolio/Risk components; the simulator itself still enforces the hard Balanced cash and no-short boundaries. Multi-timeframe scheduling is reserved as described above. The ledger currently assumes every supplied price and cash amount uses one common currency; cross-currency US/European portfolios require an explicit future FX policy and must not be interpreted as currency-correct today. The current profile universe is still not point-in-time and future research may retain survivorship bias.

## Feature Engine and Quantitative Baselines

### Shared feature definitions

`FeatureEngine` is the only Lot 3 source of quantitative calculations. It accepts one immutable, normalized symbol/timeframe history and returns an immutable `FeatureSnapshot` for its latest observable bar. Stable feature names and `feature_schema_version = 1.0` make the definitions traceable for strategies and future regime/ML lots. The current definitions are:

- Trend: `sma_N`, `ema_N`, per-bar moving-average slope over a configured lookback, and decimal price-to-average distance.
- Momentum: one-bar simple return, `return_N` as a decimal rolling return, `roc_N` as the percentage equivalent, plus exact-timestamp cross-sectional rank and percentile.
- Volatility: true range, Wilder ATR seeded after a full window, and sample rolling volatility of simple returns.
- Volume: rolling mean volume and current volume divided by that mean. A zero mean produces unavailable data, never infinity.
- Structure: `previous_high_N`, `previous_low_N`, and price distance to each level. The reference range explicitly excludes the current bar.

All windows are counts of bars, not days: `20` means 20 bars for `1h`, `4h`, or `1d`. Warm-up values remain `None`; they are never backfilled, extrapolated, or replaced with partial-window estimates. Strategy-required features must all be available before an order can be proposed. The bounded Feature Engine cache reuses only exact immutable inputs.

### Anti-look-ahead rule

A feature at time `t` uses only bars with timestamps less than or equal to `t`. Supplying a later history with `as_of=t` filters every later bar before calculation, and tests assert that appending future data cannot change SMA/EMA, momentum, ATR, or prior-range levels at `t`. Strategies still receive the Lot 2 `StrategyContext`; no full future DataFrame, `next_close`, or negative-future shift is exposed.

Cross-sectional relative strength uses only assets that have a real bar at exactly the same UTC timestamp. Missing Europe/US observations are reported and the baseline skips that snapshot; it never forward-fills one market into another. This strict policy is safe but means mixed-calendar datasets may have few or no coherent ranking points until a future explicit decision-calendar policy is designed.

### Research baselines

The production research registry exposes three versioned long-only baselines. Their immutable defaults are reference values, not optimized parameters:

- `trend` (`1.0`): enter when EMA 20 is above EMA 50, close is above EMA 50, and the five-bar fast-EMA slope is positive; exit when EMA 20 is no longer above EMA 50. Default allocation fraction: 25%.
- `momentum` (`1.0`): rank exact-common-timestamp 20-bar returns, select up to the positive top 3, and reconsider the selection every five coherent bars. Default total allocation fraction: 60%, divided equally across new targets.
- `breakout` (`1.0`): enter when close exceeds the previous 20-bar high and exit when close falls below the previous 10-bar low. Current-bar highs/lows never define their own trigger. Default allocation fraction: 25%.

Every ENTER/EXIT signal stores strategy/version, UTC timestamp, reason, strength, and exact feature values; simulated orders reference the originating signal ID. No opaque BUY/SELL action is generated. `BaselineSizer` provides only temporary total-allocation/fractional-share sizing for these research runs, split across configured symbols or momentum targets and capped by the active profile exposure ceiling. It is deliberately not named or treated as `PortfolioEngine`. The existing ledger remains authoritative for cash, no leverage, and no short positions.

`StrategyReport` exposes strategy parameters, trades, return, drawdown, Sharpe, turnover, benchmark return, and excess return. Multi-report comparison preserves the requested order and never declares an automatic winner. Turnover remains an observed metric, not an optimization objective.

The Lot 3 strategies are research baselines, not claims of profitable trading systems. No parameter sweep, hidden winner selection, ML, regime detector, or real risk-budget logic is present. Mean Reversion is intentionally reserved for Lot 5 so it can later be conditioned on an explicit regime policy. A present-day configured universe is not point-in-time, so future results can retain survivorship bias.

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

Backtests never download missing data. Each requested symbol, timeframe, start, and end must exactly match an existing Lot 1 cache manifest. The technical demo remains available, and Lot 3 baselines can be selected from the shared registry:

```powershell
trading-ai backtest run --strategy buy-and-hold --symbol SPY --timeframe 1d --start 2024-01-01 --end 2025-01-01 --starting-cash 100000 --spread-bps 5 --slippage-bps 5 --commission-fixed 1 --json
trading-ai backtest run --strategy trend --symbol SPY --timeframe 1d --start 2024-01-01 --end 2025-01-01 --spread-bps 5 --slippage-bps 5 --commission-fixed 1 --json
trading-ai backtest run --strategy momentum --symbol SPY --symbol QQQ --symbol IWM --timeframe 1d --start 2024-01-01 --end 2025-01-01 --top-k 2 --rebalance-every 5 --spread-bps 5 --slippage-bps 5 --commission-fixed 1 --json
trading-ai backtest run --strategy breakout --symbol AAPL --timeframe 4h --start 2024-06-01 --end 2024-07-01 --entry-window 20 --exit-window 10 --spread-bps 5 --slippage-bps 5 --commission-fixed 1 --json
trading-ai backtest inspect --run-id bt-0123456789abcdef01234567 --json
trading-ai strategy list --json
```

The demo submits one buy intent. Every command uses the first selected symbol as its configurable Buy & Hold benchmark unless `--benchmark-symbol` names another cached profile symbol. Strategy-specific window/allocation flags override immutable defaults for that run, and the resolved values are stored in `BacktestResult`. `run` exports under `data_local/backtests/`; `inspect` verifies checksums before reading the summary. A cache miss fails clearly without falling back to Yahoo Finance.

## Tests and CI

The default suite is deterministic, fast, and independent of Yahoo Finance and the network. It uses `FakeDataProvider` plus small synthetic sessions, invalid bars, gaps, corporate actions, feature warm-ups, future-append invariance, relative-strength ties/missing assets, all three baselines, execution costs, partial exits, known metric curves, and look-ahead probes.

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

Crypto assets, options, futures, CFDs, significant leverage, high-frequency data, real-time feeds, bid/ask, order books, and alternative data are outside the current V1 lots. They may be studied only in dedicated future work without weakening the balanced profile, the aggressive lock, or the mandatory risk boundary.

Expected safety matrix:

| Environment | Profile | Result |
| --- | --- | --- |
| PAPER | balanced | Configuration accepted; real orders still denied; offline simulation allowed |
| PAPER | aggressive | Blocked: profile disabled and code-locked |
| LIVE | balanced | Blocked: LIVE has no unlock mechanism |

## Roadmap

Lots 0, 0.1, 1, 2, and 3 provide foundations, universe/CI alignment, historical data, deterministic simulation, shared features, and quantitative research baselines. Lot 4 is the next TODO and will address a real Risk Engine in its own reviewed scope; it has not started here. Later lots cover regime detection, ML, portfolio construction, dashboarding, and broker/paper integration. Aggressive Research remains locked. See `PROJECT_STATE.md` for the authoritative status.
