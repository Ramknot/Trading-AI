# Trading AI

Trading AI is a safety-first, modular Python project for reproducible market research, paper trading, and future controlled execution. Lots 8.1–8.3 add configuration-driven transaction economics, point-in-time cash reservations, an Economic Gate, frozen robustness governance, official-evidence comparison, and read-only Paper-readiness review to the local Dashboard and the engines delivered in Lots 0 through 8. They distinguish modeled gross performance, cost-aware net performance, historical proof quality, and forward Paper readiness without moving business logic into the UI. Portfolio construction still proposes targets; `BalancedRiskEngine` remains the final authority on every simulated order, and no validation or readiness result can unlock Paper or `LIVE` execution.

## Safety guarantees

- `DEV`, `TEST`, `PAPER`, and `LIVE` remain separate environments.
- `balanced` is the active research/paper profile; `aggressive` remains disabled and unconditionally code-locked.
- `LIVE` startup has no unlock mechanism.
- Every order path must pass through `RiskEngine`; `DenyAllRiskEngine` remains the generic default and rejects every request. `BalancedRiskEngine` must be injected explicitly for offline simulation.
- Strategies, ML, portfolio, and data components cannot call a broker directly.
- Secrets, local datasets, logs, environments, and caches are excluded from version control.

These are architectural safeguards, not a claim that trading systems or market data are risk-free.

## Structure

```text
config/profiles/                  configurable profiles and market universes
config/regimes/                   Balanced detector/policy TOML; Aggressive locked
config/portfolio/                 Balanced sleeves/caps/currencies; Aggressive locked
config/costs/                     dated tariffs, tax/instrument metadata, operating costs
config/validation/                fixed research-validation thresholds
src/trading_ai/core/              models, configuration, safety policy, health, logging
src/trading_ai/data/
  base.py                         provider-neutral DataProvider contract
  engine.py                       orchestration, cache modes, retries, 4h derivation
  calendar.py                     exchange-calendar boundary and expected-bar logic
  quality.py                      normalization and fail-closed OHLCV validation
  resampling.py                   deterministic session-anchored 1h -> 4h aggregation
  storage.py                      Parquet datasets, manifests, and SHA-256 integrity
  providers/fake.py               deterministic offline test provider
  providers/local.py              explicit mapped CSV/Parquet ingestion, no network
  providers/yahoo.py              development historical-data adapter
src/trading_ai/features/          shared trend, momentum, volatility, volume, structure features
src/trading_ai/regimes/           two-axis detector, confirmation, policy, reporting
src/trading_ai/strategies/        configs, sizing, registry, and four research baselines
src/trading_ai/ml/                training/inference contracts, tabular adapters, registry, scoring
src/trading_ai/portfolio/         deterministic sleeves, allocation, netting, FX contracts
src/trading_ai/costs/             estimates, actual costs, reconciliation, economics
src/trading_ai/risk/              mandatory gate, Balanced limits/state/guards/reporting
src/trading_ai/validation/        immutable OOS/economic research validation reports
src/trading_ai/execution/         sealed risk-gated ExecutionEngine
src/trading_ai/brokers/           BrokerAdapter contract
src/trading_ai/monitoring/        immutable events/snapshots, SQLite, API, local Dashboard
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
                                           |-> BalancedRegimeDetector
                                           |-> BacktestStrategy (feature snapshots/signals)
                                           |-> MLScorer (optional PASS/BLOCK entry filter)
                                           |-> StrategyActivationPolicy (<= 1.0)
                                           |-> BalancedPortfolioEngine (targets/proposals)
                                           |-> TransactionCostEngine (point-in-time estimate)
                                           |-> EconomicGate (entry eligibility only)
                                           |-> BalancedRiskEngine (explicit)
                                           |-> ExecutionModel (risk-approved orders only)
                                           |-> PortfolioLedger
                                           |-> MetricsEngine
                                           `-> BacktestResult / local export
```

The simulated `PortfolioPlan` and `OrderIntent` boundaries are proposals, not permission to transmit an order. In portfolio CLI runs, every executable proposal carries signal, ML, activation, opportunity, portfolio-plan, portfolio-decision, and risk-decision lineage before it can reach the simulated execution model. It cannot reach `BrokerAdapter`, and the generic execution architecture remains guarded by mandatory `RiskEngine` with `DenyAllRiskEngine` as its fail-closed default.

## Installation

Use Python 3.11 or newer; CI runs Python 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

Runtime dependencies remain deliberately limited to the data/research stack (`pandas`, `pyarrow`, `yfinance`, `pandas-market-calendars`, `scikit-learn`) plus the Lot 8 local web stack (`FastAPI`, `Jinja2`, and `Uvicorn`). `HTTPX` is used only by development tests. No external backtesting, indicator, risk, regime, neural-network, registry-server, cloud-monitoring, or streaming framework is used. Yahoo Finance is a development/research source, not a guaranteed production or live-trading feed.

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

Lot 2 runs one configured primary timeframe per simulation. Any of the Balanced `1h`, `4h`, or `1d` datasets can be the primary stream, and every event identifies its timeframe. Supplying secondary timeframes in the same run fails explicitly until a later, carefully specified multi-timeframe scheduler is added. Multi-asset and Lot 7 multi-strategy runs share one chronological event loop, ledger, cash balance, portfolio state, and `BalancedRiskEngine`.

### Backtest assumptions

Every result records these assumptions through immutable `BacktestConfig` and dataset references:

- A signal produced after processing bar `t` cannot fill on that bar. A `MARKET` order first becomes eligible on the next later bar for the same symbol/timeframe and fills completely at that bar's open before costs.
- A `BUY LIMIT` or `SELL LIMIT` stays `PENDING` until a later bar reaches it, or until an optional eligible-bar expiry. V1 applies a deterministic full fill at the limit, not a favorable intrabar price. When spread/slippage are non-zero, the raw touch must be sufficient for the submitted limit to remain the all-in execution-price bound.
- Spread and slippage remain configurable in basis points and are applied adversely inside `BarExecutionModel`. In cost-aware runs, `BalancedTransactionCostEngine` represents those same price impacts economically and receives their realized fill amounts; it never debits them a second time. Broker commission, exchange/pass-through fees, transaction tax, FX cost, financing, and other variable costs come only from dated configuration. A missing critical component remains `UNAVAILABLE`, making coverage incomplete rather than silently becoming zero. Legacy zero-cost assumptions remain available only to old/test paths; production-research CLI runs require an explicit cost profile and reject non-zero legacy commission flags.
- Balanced V1 is cash-only and long-only. A BUY reserves point-in-time notional plus estimated entry costs and a configured buffer before Risk evaluation; concurrent pending orders cannot reuse that cash. `BalancedRiskEngine` first reduces or rejects a request that exceeds free cash after costs, position/portfolio/group limits, or an available exit quantity; the ledger independently rejects any remaining inconsistency. The buffer is released rather than reported as a realized cost. Cash cannot become negative. There is no margin, leverage, shorting, market depth, realistic partial fill, latency, market impact, or order-book model.
- `STOP` and `STOP_LIMIT` enum values are reserved for future execution models but are rejected by the Lot 2 order-intent model. V1 accepts only `MARKET` and `LIMIT`.
- Raw OHLC is used with explicit dividends and splits. `adjusted_close` is never substituted. A held long position receives an explicit dividend ledger credit; a split changes quantity and average entry price without economic PnL. Adjusted-only inputs are rejected so price adjustment and actions cannot be counted twice.
- Gaps are never filled or forward-filled. `DataQualityReport.FAIL` always blocks a run. `WARNING` blocks under the default `STRICT` policy and may continue, with warnings preserved in the result, only under `ALLOW_WARNINGS`.
- Portfolio targets and quantities use the last close actually observed at `t`; they never use the next open. The order remains eligible only on a later bar, so an open gap can make realized exposure differ from the target. Execution and Risk remain the downstream safeguards.

These simplifying assumptions are conservative and deterministic, not a claim of achievable real-market execution. A backtest result is not evidence of future profitability.

### Ledger, fills, trades, and equity

`PortfolioLedger` is the only component that mutates simulated cash and positions. Each fill, dividend, and split creates an immutable ledger entry. It tracks cash, quantity, average entry price, net realized PnL, unrealized PnL, dividend income, and an equity point per market timestamp. FIFO reconstruction retains partial exits and produces closed `Trade` records with entry/exit times, effective prices, quantity, gross and net PnL, fees, spread/slippage attribution, return, and holding period. The V1 execution model fills eligible orders in full, while trade reconstruction safely supports partial position exits across multiple complete fills.

### Metrics and benchmark

The independent `MetricsEngine` calculates initial capital, final equity, total and annualized return, annualized volatility, Sharpe, Sortino, maximum drawdown and recovery metadata, Calmar, profit factor, win rate, average win/loss, expectancy, trade count, turnover, exposure, best/worst trade, average holding period, cost totals, and dividend income. Undefined cases such as no losing trade, no downside, zero variance, or too few observations return `None` rather than a misleading number.

Annualization is centralized: `252` observations for daily equity data, `504` for 4h, and `1638` for 1h, as documented equity-market research conventions. The risk-free rate is an explicit configuration value and defaults to zero; no external rate is fetched.

The optional `BuyAndHoldBenchmark` is a dedicated benchmark component, not an operational strategy. Its symbol is configurable and requires an explicitly supplied dataset on the same primary timeframe. It uses the simulation period's raw closes and explicit corporate actions, and reports return, maximum drawdown, and strategy excess return.

### Provenance, reproducibility, and exports

Every `BacktestResult` records strategy name/version/parameters, feature schema/engine lineage, exact dataset IDs and SHA-256 checksums, corporate-action lineage, simulation config, regime detector/version/config/hash, optional ML model/schema/config/prediction/decision lineage, policy/version/config/hash, portfolio engine/version/config/hash, opportunities/decisions/plans/targets/sleeves, risk engine/version/config/hash, every risk decision/state transition, historical start/end, optional Git commit SHA, a SHA-256 of the current Python source tree, explainable signals, orders, fills, trades, ledger entries, warnings, benchmark, and a stable run/result hash. The source hash makes an uncommitted/dirty run distinguishable even when `HEAD` still names an older commit. Technical creation time and inference latency are excluded from the deterministic hash, so identical data, features, regimes, model artifacts, policy, strategy, portfolio, risk, source, and simulation inputs produce identical decisions, fills, metrics, run ID, and result hash.

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
  risk_decisions.parquet
  risk_states.parquet
  regime_snapshots.parquet
  regime_transitions.parquet
  activation_decisions.parquet
  ml_predictions.parquet       # when ML is active
  ml_decisions.parquet         # when ML is active
  portfolio_opportunities.parquet
  portfolio_decisions.parquet
  portfolio_targets.parquet
  portfolio_sleeves.parquet
  cost_estimates.parquet
  economic_decisions.parquet
  cost_actuals.parquet
  cost_reconciliation.parquet
  validation_report.json
  checksums.json
```

Schema `1.6` adds the cost/economic/validation files while `BacktestResultStore` remains checksum-compatible with schemas `1.0` through `1.5`. It verifies every exported file against its SHA-256 checksum before inspection. No result or market dataset is versioned.

### Lot 2 limitations

The core simulator deliberately has no implicit model training, strategy/risk parameter optimization, grid search, automatic end-of-run liquidation, or real liquidity model. Lot 6 training and Quant-vs-ML comparison are explicit, separate offline research paths; inference cannot fit. Trade metrics describe closed FIFO trades; an open final position remains visible in equity and exposure but is not fabricated into a closing trade. Lot 7 uses fixed research sleeves and deterministic equal weights, not Markowitz, Kelly, risk parity, covariance optimization, performance chasing, or ML allocation. Multi-timeframe scheduling is reserved as described above. The Portfolio Engine has an explicit point-in-time FX contract, but no production FX dataset/provider yet: same-currency conversion is identity, new mixed-currency allocation fails closed without an injected rate book, and exits remain possible. Detailed realized PnL attribution to overlapping strategy sleeves is intentionally deferred because the physical ledger is aggregated by symbol. The current profile universe is still not point-in-time and future research may retain survivorship bias.

## Feature Engine and Quantitative Baselines

### Shared feature definitions

`FeatureEngine` is the shared source of quantitative calculations. It accepts one immutable, normalized symbol/timeframe history and returns an immutable `FeatureSnapshot` for its latest observable bar. Lot 5 extends the schema additively to `feature_schema_version = 1.1`; all Lot 3 definitions retain their meaning. Stable names make every value traceable for strategies, regimes, risk controls, and Lot 6 ML inputs. The definitions are:

- Trend: `sma_N`, `ema_N`, per-bar moving-average slope over a configured lookback, and decimal price-to-average distance.
- Momentum: one-bar simple return, `return_N` as a decimal rolling return, `roc_N` as the percentage equivalent, plus exact-timestamp cross-sectional rank and percentile.
- Volatility: true range, Wilder ATR seeded after a full window, and sample rolling volatility of simple returns.
- Volume: rolling mean volume and current volume divided by that mean. A zero mean produces unavailable data, never infinity.
- Structure: `previous_high_N`, `previous_low_N`, and price distance to each level. The reference range explicitly excludes the current bar.
- Lot 5 structure: `efficiency_ratio_N`, direction divided by total absolute price path over `N` changes. A flat denominator is unavailable.
- Lot 5 context: `volatility_percentile_V_P`, the midpoint empirical rank of current rolling volatility `V` within the latest `P` already-observed volatility values.
- Lot 5 Mean Reversion: `price_zscore_N`, using the current close, full-window mean, and population standard deviation. Zero standard deviation is unavailable.

All windows are counts of bars, not days: `20` means 20 bars for `1h`, `4h`, or `1d`. Warm-up values remain `None`; they are never backfilled, extrapolated, or replaced with partial-window estimates. Strategy-required features must all be available before an order can be proposed. The bounded Feature Engine cache reuses only exact immutable inputs.

### Anti-look-ahead rule

A feature at time `t` uses only bars with timestamps less than or equal to `t`. Supplying a later history with `as_of=t` filters every later bar before calculation, and tests assert that appending future data cannot change SMA/EMA, momentum, ATR, prior-range levels, Efficiency Ratio, volatility percentile, or price z-score at `t`. Strategies still receive the Lot 2 `StrategyContext`; no full future DataFrame, `next_close`, centered window, or negative-future shift is exposed.

Cross-sectional relative strength uses only assets that have a real bar at exactly the same UTC timestamp. Missing Europe/US observations are reported and the baseline skips that snapshot; it never forward-fills one market into another. This strict policy is safe but means mixed-calendar datasets may have few or no coherent ranking points until a future explicit decision-calendar policy is designed.

### Research baselines

The production research registry exposes four versioned long-only baselines. Their immutable defaults are reference values, not optimized parameters:

- `trend` (`1.0`): enter when EMA 20 is above EMA 50, close is above EMA 50, and the five-bar fast-EMA slope is positive; exit when EMA 20 is no longer above EMA 50. Default allocation fraction: 25%.
- `momentum` (`1.0`): rank exact-common-timestamp 20-bar returns, select up to the positive top 3, and reconsider the selection every five coherent bars. Default total allocation fraction: 60%, divided equally across new targets.
- `breakout` (`1.0`): enter when close exceeds the previous 20-bar high and exit when close falls below the previous 10-bar low. Current-bar highs/lows never define their own trigger. Default allocation fraction: 25%.
- `mean-reversion` (`1.0`): propose an entry when `price_zscore_20 <= -1.5`; the activation policy permits that proposal only in an eligible `RANGE` and outside `HIGH` volatility. Exit at `price_zscore_20 >= -0.25` or when structure leaves `RANGE`. Default allocation fraction: 20%. It is long-only, never averages down, and has no martingale path.

Every ENTER/EXIT signal stores strategy/version, UTC timestamp, reason, strength, and exact feature values; simulated orders reference the originating signal ID. No opaque BUY/SELL action is generated. `BaselineSizer` provides only temporary total-allocation/fractional-share proposals for these research runs. It is deliberately not named or treated as `PortfolioEngine`: **BaselineSizer proposes; RiskEngine disposes**. The Risk Engine may preserve, reduce, or reject that quantity and can never increase it; the ledger remains an independent cash/no-leverage/no-short backstop.

`StrategyReport` exposes strategy parameters, trades, return, drawdown, Sharpe, turnover, benchmark return, and excess return. Multi-report comparison preserves the requested order and never declares an automatic winner. Turnover remains an observed metric, not an optimization objective.

The strategies are research baselines, not claims of profitable trading systems. No parameter sweep, hidden winner selection, or performance-driven activation is present. Lot 6 may score their already-generated entry candidates, but it does not change their indicator definitions or turn ML into a strategy. A present-day configured universe is not point-in-time, so future results can retain survivorship bias.

## Regime Detector & Strategy Activation

### Two independent axes

`BalancedRegimeDetector` (`balanced-regime`, version `1.0`) describes context and never emits a signal or order. Structure and volatility are intentionally separate dimensions:

- `StructureRegime`: `TREND_UP`, `TREND_DOWN`, `RANGE`, or conservative `UNKNOWN`.
- `VolatilityRegime`: `LOW`, `NORMAL`, `HIGH`, or `UNKNOWN` during percentile warm-up.

Combinations such as `TREND_UP + HIGH` are valid. The structure candidate combines shared EMA 20/50 ordering, normalized EMA slope, price-to-slow-EMA distance, EMA separation, and `efficiency_ratio_20`. `RANGE` requires low efficiency, contained EMA separation, and contained price distance. Missing critical evidence, conflicting rules, or insufficient history produces `UNKNOWN`; it is never silently converted to `RANGE`.

Structure changes require three consecutive candidate bars by default. Until confirmation, `RegimeSnapshot` retains the current structure plus the candidate and confirmation progress. Confirmed changes create a single immutable `RegimeTransition`. Volatility classification independently uses `volatility_percentile_20_100`: at or below 0.20 is `LOW`, at or above 0.80 is `HIGH`, and the interval is `NORMAL`. Windows are bar counts and all thresholds are configurable, deterministic research defaults—not optimized values or probabilities.

### Balanced activation matrix

`BalancedStrategyActivationPolicy` (`balanced-strategy-policy`, version `1.0`) maps a strategy signal and current `RegimeSnapshot` to immutable `ALLOW`, `REDUCE`, or `BLOCK`. Its multiplier is always in `[0, 1]`; it cannot increase the strategy proposal.

| Structure | Trend | Momentum | Breakout | Mean Reversion |
| --- | --- | --- | --- | --- |
| `TREND_UP` | ALLOW 1.0 | ALLOW 1.0 | ALLOW 1.0 | BLOCK |
| `RANGE` | BLOCK | BLOCK | REDUCE 0.50 | ALLOW 1.0 |
| `TREND_DOWN` | BLOCK | BLOCK | BLOCK | BLOCK |
| `UNKNOWN` | BLOCK | BLOCK | BLOCK | BLOCK |

The sole V1 volatility overlay blocks new Mean Reversion entries in `HIGH` volatility. Trend, Momentum, and Breakout are not reduced a second time by this contextual label; the independent Lot 4 `VolatilityGuard` remains responsible for risk sizing. Every `EXIT_LONG` is allowed by policy in every regime, including `UNKNOWN`, `TREND_DOWN`, and `HIGH` volatility.

Quantity authority remains ordered and monotonic:

```text
BaselineSizer proposal / candidate signal
  -> optional ML PASS or BLOCK (never sizing up)
  -> activation multiplier <= 1
  -> BalancedRiskEngine approved quantity <= adjusted proposal
  -> next-bar simulated execution
```

A blocked activation never becomes a new-risk request. An allowed activation can still be rejected by a `HALTED` Risk Engine. Each result preserves `dataset -> features -> regime snapshot -> signal -> optional ML prediction/decision -> activation decision -> order -> risk decision -> fill` lineage, hashes both TOML configurations, and retains the regime/activation Parquet files in schema `1.4`. Older schema `1.0` through `1.3` runs remain inspectable, with unavailable layers reported explicitly.

Mixed US/European calendars are not forward-filled. Each symbol receives its own point-in-time regime, while Momentum continues to rank only exact-common UTC observations. This conservative rule can reduce the number of mixed-market ranking opportunities.

Regime classification is a research abstraction, not a prediction of future market behaviour. Strategy eligibility does not override the Risk Engine and does not guarantee profitability.

## ML Platform Foundation & Statistical Scoring

### Role and architecture

Lot 6 treats machine learning as an optional quality score for an existing quantitative `ENTER_LONG` candidate, never as a strategy or order generator. The guarded entry path is:

```text
validated data -> FeatureEngine -> RegimeDetector -> Strategy candidate
  -> MLScorer -> StrategyActivationPolicy -> BaselineSizer
  -> BalancedRiskEngine -> simulated execution
```

`EXIT_LONG` is `NOT_APPLICABLE` to the ML filter and always continues. ML cannot increase allocation: V1 is strictly `PASS` or `BLOCK`, policy remains capped at 1.0, and the Risk Engine remains the final authority. A high probability cannot override an ineligible regime, a `HALTED` Risk Engine, cash/exposure limits, `LIVE` lock, or the aggressive-profile lock.

Framework-neutral contracts separate `ModelTrainer`, inference-only `ModelAdapter`, `InferenceEngine`, and `ModelRegistry`. Scikit-learn classes are confined to `ml/adapters/sklearn.py`; backtesting, strategies, regimes, and risk depend only on the contracts. `InputKind` reserves `TABULAR`, `SEQUENCE`, `CROSS_SECTIONAL`, and `MULTIMODAL`, but only `TABULAR` is operational. `score_one` supports future event-driven inference and `score_batch` supports offline research; both produce the same probability for the same immutable input. Inference exposes no fitting or model-update operation.

### Inputs, labels, and leakage controls

`MLFeatureBuilder` consumes the shared `FeatureSnapshot`, point-in-time `RegimeSnapshot`, and signal metadata. It never recalculates EMA, ATR, volatility, momentum, or regime definitions. `ml_feature_schema_version = 1.0` has stable ordering and currently uses shared normalized values such as `return_20`, `rolling_vol_20`, `relative_volume_20`, `efficiency_ratio_20`, `volatility_percentile_20_100`, `price_zscore_20`, price-to-EMA distances, and one-hot structure/volatility regimes. Momentum may additionally include its point-in-time relative-strength percentile. The symbol remains metadata rather than a feature.

Models are scoped to one strategy and one timeframe, while multiple assets may share a model under common UTC split boundaries. Missing required features exclude a training example. During inference they produce an explicit unavailable result; `FILTER` fails closed for a new entry and `SCORE_ONLY` records a warning without changing trading.

The binary target is built only inside `LabelBuilder`: a candidate emitted at close `t` uses the next bar open as the hypothetical entry and the close after a configurable horizon (five bars by default) as the exit reference. The default positive class requires forward return above zero basis points. Observations without the required future are unlabeled and dropped. Future prices, returns, fills, outcomes, regimes, and volatility never enter `ModelInput` or inference.

`PurgedWalkForwardSplitter` preserves chronological `TRAIN -> VALIDATION -> FINAL TEST` regions, shared across symbols. It does not expose random shuffling. Label windows crossing a boundary are purged, a configurable one-bar embargo is applied by default, and expanding walk-forward folds evaluate validation. Learned preprocessing—including the Logistic Regression `StandardScaler`—is fitted only on each fold's TRAIN data. The final test never participates in fitting, scaling, threshold selection, feature selection, or model selection.

### Tabular baselines and evaluation

The fixed, non-optimized model families are:

- Logistic Regression (`1.0`), with TRAIN-only standardization and inspectable ordered coefficients;
- Random Forest (`1.0`), with 100 trees, bounded depth, and explicit `random_state = 42`;
- Gradient Boosting (`1.0`), with fixed learning rate/tree settings and `random_state = 42`.

There is no grid search, parameter sweep, automatic feature selection, calibration fitting, or winner selection. Training refuses insufficient samples, insufficient observations per class, or a one-class target. Reports contain sample/class counts, accuracy, balanced accuracy, precision, recall, F1, ROC AUC, PR AUC, log loss, Brier score, confusion matrix, calibration bins, and coefficients/importances when supported. Undefined metrics remain unavailable instead of being fabricated. Feature importance is descriptive, not causal.

### Registry, lifecycle, and inference modes

`TrainingPipeline` and `InferenceEngine` are separate paths. A new model is stored as `CANDIDATE` in the untracked `data_local/ml/` registry with its strategy/timeframe scope, dataset IDs/checksums, feature and label schemas, temporal ranges, git/source hashes, framework/runtime metadata, serialized payload checksum, evaluation report, and audit event. Loading is allowed only by validated `model_id`; path traversal, corruption, schema mismatch, family mismatch, and incompatible strategy/timeframe fail explicitly.

The lifecycle is `CANDIDATE -> VALIDATED -> APPROVED -> RETIRED`. Promotion is an explicit audited command requiring a reason; `CANDIDATE -> APPROVED` and automatic promotion based on returns or classification scores are forbidden. `FILTER` requires an explicitly `APPROVED` model. Approved aliases never select a silent `latest.pkl`, and rollback is explicit, audited, and does not retrain. There is no fallback to another or older model when an artifact is missing or corrupt.

Three modes are available:

- `DISABLED`: the Lot 5 quant/policy/risk path is unchanged.
- `SCORE_ONLY`: predictions are recorded but cannot block or resize any signal; a `CANDIDATE` or `VALIDATED` model may be observed.
- `FILTER`: an approved model passes a new entry when `P(positive) >= 0.55` by default and otherwise blocks it. The threshold is a fixed research baseline, not optimized.

Schema `1.4` adds checksum-protected `ml_predictions.parquet` and `ml_decisions.parquet` plus model provenance to the existing result export. The deterministic result hash includes the artifact, schema, configuration, prediction outputs, decisions, and threshold while excluding technical latency. Inspection remains backward-compatible with schemas `1.0` through `1.3`, reporting ML as unavailable/not used. A neutral research report compares Quant and Quant+ML only when datasets, dates, fees, strategy, regime, policy, risk, benchmark, and backtest assumptions match; it labels the period `IN_SAMPLE`, `VALIDATION`, or `OUT_OF_SAMPLE` and never declares a winner.

Machine-learning probabilities are statistical model outputs, not certainty about future market movements. Adding machine learning does not guarantee that a quantitative strategy will outperform its non-ML baseline. ML should be retained or promoted only when robust out-of-sample evidence is sufficiently convincing to justify its added complexity; this principle never triggers automatic promotion.

### Future ML Evolution — PLANNED / LOCKED

The contracts can later accept sequence and multimodal adapters without changing strategy, policy, risk, or backtest consumers:

```text
Tabular ML (Lot 6)
  -> sequence models
  -> Temporal CNN / LSTM / GRU / Temporal Transformer
  -> cross-sectional neural and multimodal market models
  -> controlled real-time inference
```

Future research may include news NLP/sentiment, macro variables, bid/ask, order books, multimodal fusion, justified GPU inference, and controlled online-learning studies. These capabilities are **PLANNED / LOCKED** until the Balanced platform is validated and existing components demonstrate value. Lot 6 implements no neural network, streaming transport, GPU/CUDA dependency, online learning, continuous fitting, or self-modifying model.

## Multi-Strategy Portfolio Engine

### Portfolio proposes; Risk disposes

`BalancedPortfolioEngine` (`balanced-portfolio`, version `1.0`) receives complete point-in-time batches of opportunities after ML filtering and regime eligibility. It resolves overlapping strategy proposals, builds target weights, nets them to at most one rebalance proposal per symbol and UTC cycle, then passes each proposal to `BalancedRiskEngine`. It never authorizes execution, contacts a provider/broker, creates leverage, or changes Risk state. Risk may still `REDUCE` or `REJECT` any proposed quantity; risk-reducing exits remain eligible while new risk is halted.

The multi-strategy chain is:

```text
Data -> Features -> Regime -> Strategy candidates -> ML filter
     -> Activation Policy -> Portfolio opportunities/targets
     -> BalancedRiskEngine -> next-bar simulated execution
```

`BaselineSizer` remains for backward compatibility and single-strategy tests. In new multi-strategy CLI runs it only creates an upstream technical proposal; fixed sleeve targets from `BalancedPortfolioEngine` determine the requested rebalance quantity, and Risk remains final.

### Fixed sleeves, aggregation, and netting

Balanced V1 assigns four non-optimized, configuration-driven sleeves: 15% each for Trend, Momentum, Breakout, and Mean Reversion, for a maximum target exposure of 60%. Unused sleeve capacity stays in cash and is never transferred to another strategy because of recent performance. Within a sleeve, active opportunities receive equal target weight. Ranking uses signal strength only within the same strategy, converted to a stable percentile with deterministic symbol/signal tie-breaking; cross-strategy raw strengths, past return, Sharpe, and ML probability are not allocation inputs.

Logical sleeve contributions are retained separately while the physical ledger owns one aggregated position per symbol. For example, Trend 8% plus Momentum 5% creates one 13% symbol target. Exiting one sleeve removes only its contribution; all contributions reaching zero creates a close target. Simultaneous entry/exit proposals are netted. Existing holdings persist until an explicit strategy exit or reduction and are not rotated merely because a new candidate ranks higher.

All opportunities sharing a UTC timestamp are allocated as one `PortfolioDecisionBatch`. Input order cannot change ranks, targets, orders, or the deterministic result hash. `SELECT`, `DEFER`, `REJECT`, `EXIT`, and `NO_CHANGE` decisions record stable reason codes and complete signal/ML/activation lineage.

### Limits, diversification, turnover, and pending orders

The TOML configuration is checked against profile and Risk hard ceilings before use: at most five unique targets, 15% per symbol, 60% total target exposure, and a 40% cash floor. Exceeding a profile/Risk ceiling is invalid configuration and fails closed.

Diversification is a soft selection preference only. Exact-common-timestamp return correlation and configuration-driven asset groups prefer less-correlated and less-represented candidates at otherwise comparable ranks. There is no forward/backfill. Unknown correlation remains `UNKNOWN` and is deprioritized by default, never treated as zero; `BalancedRiskEngine` independently applies its hard correlation and concentration rules.

A 1% no-trade band suppresses insignificant rebalances. Entry/increase turnover is capped at 20% of equity per cycle and excess opportunities are deterministically `DEFER`red; exits are exempt. Target quantities use close(`t`) and therefore retain open-gap risk before next-bar execution. Pending same-direction orders are not duplicated; an unresolved opposing pending order causes a conservative defer rather than crossed instructions.

### Currency policy and provenance

Asset quote currencies come from `config/portfolio/asset_currencies.toml`; no symbol or default currency is embedded in Portfolio Python code. `CurrencyConverter` is a provider-neutral point-in-time contract. The production-safe default performs same-currency identity only. Under `REJECT_WITHOUT_FX`, a new EUR allocation in a USD-base portfolio is rejected without an explicit rate known at `t`; unknown currency also fails closed. Tests use a small fixed rate book, while future validated FX series can implement the same contract. Existing exits do not require a fabricated FX rate.

Backtest schema `1.5` records the Portfolio Engine config/hash, opportunities, decisions, targets, plans, sleeve state, and portfolio metrics. Orders link `signal -> ML decision -> activation decision -> portfolio opportunity/decision/plan -> risk decision -> fill`. Inspection remains backward-compatible with schemas `1.0` through `1.4`, reporting legacy sizing when portfolio files are absent. Metrics include exposure/cash, maximum unique positions, planned/executed turnover, selected/deferred/rejected counts, sleeve capital, unused budget time, group exposure, and high/unknown-correlation cases. Detailed overlapping-sleeve PnL attribution is not claimed.

`compare_single_to_multi` is an offline research helper that first verifies identical datasets, periods, costs, source, regime/policy, Risk configuration, and per-strategy parameters, then reports mono-strategy and shared-portfolio metrics side by side. It never selects or labels a winner automatically.

Portfolio construction can diversify exposure but cannot eliminate market risk or guarantee profitability.

## Balanced Risk Engine

### Philosophy and mandatory gate

`BalancedRiskEngine` (`balanced-risk`, version `1.0`) answers only whether an already-proposed order may add risk and, if so, the maximum permitted quantity. It does not generate signals, select assets or strategies, optimize allocations, contact data providers/brokers, or try to improve returns. Each immutable decision is one of:

- `APPROVE`: the full requested quantity is accepted.
- `REDUCE`: a positive quantity strictly below the request is accepted.
- `REJECT`: approved quantity is zero.

The invariant `0 <= approved_quantity <= requested_quantity` is validated in the domain model and again at the execution boundary. The production `BacktestEngine` defaults to `DenyAllRiskEngine`; CLI baseline runs explicitly construct the validated Balanced engine. Tests that need old permissive Lot 2 mechanics inject a test-only engine from `tests/`, which is unavailable to production and CLI code. Pending buys reserve cash/exposure and pending sells reserve exit quantity, preventing multiple same-timestamp intents from reusing the same capacity.

The generic/future broker path remains fail closed. Giving it no engine uses `DenyAllRiskEngine`; giving `BalancedRiskEngine` only the older incomplete execution context also rejects. Neither risk configuration nor risk state can unlock `LIVE` or `aggressive`.

### Configuration and limits

Trading-profile ceilings remain in `config/profiles/balanced.toml`; finer immutable limits live in `config/risk/balanced.toml`. Startup rejects a risk configuration that exceeds the profile's maximum positions, portfolio exposure, or trade-risk budget. `config/risk/aggressive.toml` exists only for schema compatibility with `enabled = false`. `config/risk/asset_groups.toml` owns all symbol-to-group classification; no symbol is embedded in Risk Engine Python code.

Balanced research defaults are:

- maximum 5 open positions;
- maximum gross portfolio exposure 60%;
- maximum single-position exposure 15%;
- maximum configured-group exposure 30%;
- maximum explicit trade-risk budget 1% of current equity;
- daily loss protection 2%, soft drawdown 5%, and hard drawdown 10%;
- high-correlation threshold 0.85 and maximum highly-correlated exposure 30%;
- reduced-risk multiplier 50%, UTC risk-day boundary, and timeframe-specific volatility thresholds.

These values are research defaults, not optimized values or guarantees of safe trading. Risk limits reduce exposure; they do not eliminate trading risk or guarantee profitability.

New entries and position increases are capped deterministically by available cash (no margin/leverage), portfolio exposure, single-position exposure, configured concentration group, correlation, volatility, state multiplier, and—when explicitly supplied—trade risk. A sixth position is rejected, while an increase to an existing position does not consume an additional position slot. Balanced sells may only reduce/close held long quantity; an oversized sell is reduced to the unreserved holding and a sell without a position is rejected.

When `invalidation_price` or `risk_distance` exists, the engine computes `risk_per_share` and caps quantity to `equity * max_trade_risk_fraction / risk_per_share`. It never invents a stop. Without explicit invalidation it records `NO_EXPLICIT_RISK_DISTANCE` and continues under the other controls without claiming a precise maximum trade loss.

### Volatility, concentration, and correlation

The volatility guard consumes the shared Feature Engine's `rolling_vol_20`; it does not implement a competing ATR or volatility definition. Configured `NORMAL`, `ELEVATED`, and `EXTREME` bands apply multipliers of 1.0, 0.5, and 0.0 by default for new risk. Missing features follow an explicit policy; they are never interpreted as zero risk. Risk-reducing exits bypass this new-risk guard.

Concentration groups and limits are configuration-driven. An unknown group is explicit and rejects under the default Balanced policy; an optional warning policy is represented in the schema. Correlation uses only already-realized returns supplied by `FeatureEngine`, aligns exact common UTC timestamps, and never forward-fills/backfills. Fewer than the configured observations yields `UNKNOWN`; Balanced defaults to `ALLOW_WITH_WARNING`, while `REJECT` remains configurable. This is an exposure guard, not covariance optimization, risk parity, or a Regime Detector.

### Daily loss, drawdown, and circuit breaker

`RiskStateTracker` maintains current/peak/day-start equity and deterministic `NORMAL`, `REDUCED`, or `HALTED` state. Daily return uses the configured risk-day timezone (UTC by default). A daily loss at the threshold halts new risk until the next risk day. Soft drawdown reduces new quantities. Hard drawdown latches `HALTED` across days and recoveries until an explicit reasoned reset. Manual and invalid-market-data halts are also represented; future broker-desync/latency reason codes are schema-only.

`HALTED` means `STOP_NEW_RISK`: new positions and increases are rejected, but valid sells that reduce an existing long position remain possible. The engine never silently liquidates or emits `SELL EVERYTHING`; automated flattening requires a future execution/broker policy and is outside Lot 4.

### Provenance and reporting

Every decision records its stable ID, order ID, engine/version, status, requested/approved quantity, reason codes and explanations, state, configuration SHA-256, cash/equity, before/after exposures, loss/drawdown, and available volatility/correlation metrics. Orders retain both `signal_id` and `risk_decision_id`, producing `Signal -> Order -> RiskDecision -> Fill` lineage. The result summary includes approved/reduced/rejected counts, rejection reasons, maximum observed exposures/drawdown/daily loss, and time in protected states. Risk decisions/transitions introduced in schema `1.2` remain checksum-protected in current schema `1.5`; inspection remains compatible with all older schemas.

The risk layer remains offline and simulated. It has no FX conversion, real broker state reconciliation, order-book/liquidity impact, portfolio optimizer, automatic liquidation, or proof that its limits are sufficient for live markets. Regime classification remains a separate upstream context layer and never changes Risk Engine sovereignty.

## Dashboard & Observability

### Architecture and trust boundary

Lot 8 adds an observability layer without moving calculations or decisions into the web interface:

```text
Trading AI engines
      -> immutable MonitoringEvent / MonitoringSnapshot contracts
      -> BacktestMonitoringSource + local SQLiteMonitoringStore
      -> deterministic read models
      -> versioned local JSON API
      -> responsive read-only web Dashboard
```

`BacktestMonitoringSource` accepts only safe run IDs below `data_local/backtests/`, invokes the existing checksum verifier before every trusted read, and supports export schemas `1.0` through `1.6`. A changed or corrupt file is rejected as untrusted. Missing sections in older runs remain `UNAVAILABLE`; they are never synthesized as healthy observations. Parsed Parquet tables are cached in memory by checksum-manifest fingerprint, while integrity is still rechecked before reuse. The source contract is separate from `EventMonitoringSource`, which reserves the same event boundary for a future authenticated Paper/Live session without implementing one now.

Important engine facts are normalized into stable UTC event types covering data quality, features, regimes, signals, ML predictions/decisions, activation, portfolio/risk decisions, order intents, fills, positions, equity, health, and costs. Events retain the run/session, component/version, related IDs, and provenance. `SQLiteMonitoringStore` provides a local standard-library event/snapshot store below the Git-ignored `data_local/monitoring/` tree; there is no cloud database, Redis, Kafka, or monitoring SaaS.

### Pages, API, and local-only operation

The single responsive interface includes Overview, Equity/Drawdown, Portfolio, Strategies, Regimes, ML, Risk, Data Quality, Costs, System Health, searchable decision history, and an order-level Decision Trace. The trace follows available lineage across:

```text
Dataset -> Feature -> Regime -> Strategy -> Signal -> ML
        -> Activation -> Portfolio -> Cost Estimate -> Economic Gate
        -> Risk -> Order -> Fill -> Actual Cost -> Reconciliation
```

The frontend reads `/api/v1` JSON endpoints for runs, snapshots, overview, equity, portfolio, strategies, regimes, ML, risk, data quality, costs, decisions, events, traces, and health. It never opens Parquet files itself. Polling every five seconds is suitable for local monitoring and preserves a future event-driven source boundary; completed backtests remain static. Values are rendered with DOM `textContent`, templates auto-escape input, and no external CDN is required.

Start the interface with:

```powershell
trading-ai dashboard serve --host 127.0.0.1 --port 8080
trading-ai dashboard inspect --run-id bt-0123456789abcdef01234567 --json
trading-ai monitoring health --json
```

Lot 8 accepts only loopback hosts (`127.0.0.1`, `localhost`, or `::1`) and defaults to `127.0.0.1`. It has no remote authentication and therefore cannot be exposed automatically on `0.0.0.0` or the public Internet. The Dashboard is strictly read-only: it has no order-entry, configuration mutation, profile activation, or `LIVE` control. Secure remote access and permissions must be designed before any future Paper/Live monitoring deployment.

### Health and decision visibility

System Health reports Data, Feature, Regime, Strategies, ML, Portfolio, Risk, execution simulator, monitoring store, and trading-cost coverage as `HEALTHY`, `WARNING`, `ERROR`, or `UNAVAILABLE`. A disabled or absent component is never shown as healthy. Data provenance exposes dataset/provider/symbol/timeframe/ranges/checksums and warnings from the signed result; quality details not embedded in a historical schema remain explicitly unavailable. The Dashboard does not reopen another dataset and claim it is the one used by the run.

Decision and risk views expose machine-readable reason codes, human explanations, state transitions, circuit-breaker state, and available limits. ML probabilities are shown as statistical outputs, never as certainty. Portfolio views distinguish physical quantities from targets and logical sleeve contributions where exported; detailed sleeve PnL attribution remains unavailable because Lot 7 intentionally did not fabricate it.

### Cost coverage and gross/net display

`TradingCostBreakdown` keeps commission, spread, slippage, exchange fees, transaction tax, FX cost, financing cost, other variable cost, and total variable cost separate. `OperatingCostBreakdown` separately reserves market-data, server/VPS, software/subscription, and other fixed costs. Every component is `KNOWN`, `ESTIMATED`, `NOT_APPLICABLE`, or `UNAVAILABLE`; an unavailable component cannot carry a numeric zero.

Schema 1.6 Dashboard views reuse the immutable estimates, actual costs, reconciliation, and validation report produced by the engines. They show entry and round-trip estimates, commission, spread, slippage, exchange/pass-through fees, transaction tax, FX, financing, other variable costs, operating costs, gross P&L, net trading P&L before operating costs, and net economic P&L only when coverage permits it. `NET COMPLETE` and `NET INCOMPLETE` are explicit. Older exports retain the Lot 8 compatibility behavior: unmodeled categories remain `UNAVAILABLE`. The UI never recalculates tariffs, taxes, expected edge, or validation criteria.

Portfolio construction can diversify exposure but cannot eliminate market risk or guarantee profitability. Monitoring improves visibility; it does not validate profitability or make a backtest predictive.

## Transaction Cost Economics & Research Validation

### Cost engine, provenance, and coverage

`BalancedTransactionCostEngine` (`balanced-transaction-cost / 1.0`) is an offline, broker-neutral adapter boundary. It consumes dated TOML tariff, tax, instrument, FX, and operating-cost metadata; it never contacts a broker or provider. The repository contains source-provenanced current reference profiles for IBKR Pro Fixed and Tiered US stock/ETF pricing, verified on 2026-08-29 against the official [Interactive Brokers stock commission schedule](https://www.interactivebrokers.com/en/pricing/commissions-stocks.php) and [pricing overview](https://www.interactivebrokers.com/en/pricing/commissions-home.php). Tiered and fixed commission formulas support per-share, proportional, minimum, maximum, notional-cap, marginal monthly-volume tiers, and explicit exchange/pass-through fields. Fixed US pricing marks bundled exchange costs `NOT_APPLICABLE`; Tiered retains aggregate exchange/pass-through coverage as `UNAVAILABLE` because the currently configured NSCC/DTC amount is only one known component of venue-dependent total fees. These configurations are research assumptions, not runtime scraping and not a promise that an account will receive those prices.

Every variable component is independently `KNOWN`, `ESTIMATED`, `NOT_APPLICABLE`, or `UNAVAILABLE`. Commission is distinct from exchange/pass-through fees so the same fee cannot be counted twice. Taxes require explicit, dated instrument eligibility metadata and a dated rule; a ticker suffix alone is never enough. The configured 2026 French transaction-tax rate is sourced from [CGI article 235 ter ZD on Légifrance](https://www.legifrance.gouv.fr/codes/section_lc/LEGITEXT000006069577/LEGISCTA000006162552/2026-01-01), while eligibility for `MC.PA` and `AIR.PA` deliberately remains `UNVERIFIED` pending validation against the official dated annual instrument list. Financing is explicitly `NOT_APPLICABLE` for cash-only Balanced simulation. Market-data, VPS/server, and software costs are period-level operating costs and are never allocated arbitrarily to fills.

The configured tariff `effective_from`, optional `effective_to`, `verified_at`, source URL, lifecycle status, version, and SHA-256 are retained in every estimate. Applying a current tariff to older history produces `CURRENT_TARIFF_APPLIED_RETROSPECTIVELY`, an `ESTIMATED` component, and `tariff_period_covered = false`; strict validation cannot pass it as historically verified. No IBKR rate, tax rate, or subscription amount is hard-coded in Python.

### Estimates, actual costs, FX, and cash safety

At close `t`, `PreTradeCostEstimate` uses only the observed symbol, side, proposed quantity, current known price, timeframe, configured spread/slippage, dated tariff/tax metadata, and point-in-time FX book. It never uses the next open or a future fill. Entry costs and estimated round-trip costs are separate. If an exit leg cannot be estimated, round-trip coverage is incomplete instead of guessed.

For a BUY, `EstimatedCashRequirement` is:

```text
base-currency notional + estimated entry costs + configured cash buffer
```

The default buffer is the greater of 5 bps and USD 0.50. It reserves capacity but is not a realized fee. Pending orders reserve the same full requirement, and `BalancedRiskEngine` may reduce or reject quantity before simulated execution. Risk remains sovereign after both Cost Engine and Economic Gate. Risk-reducing exits are never blocked by Economic Gate, and the Risk Engine continues to permit valid reductions while new risk is halted.

Currency conversion and FX cost are different facts. Same-currency conversion is identity; no implicit USD/EUR rate exists. A mixed-currency entry without a point-in-time conversion rate and explicit cost treatment stays incomplete and fails closed. `FixedFxRateBook` exists only in tests; future real FX series must enter through validated Data Engine contracts. The current physical ledger is not presented as production multi-currency accounting.

After a simulated fill, `ActualTradingCost` recomputes notional-sensitive modeled costs from the realized fill and retains the execution model's realized spread/slippage attribution. The ledger debits commission, exchange fees, transaction tax, FX fees, financing, and other explicit cash fees; spread/slippage are already embedded in execution price and are not debited twice. `CostReconciliation` records estimate-versus-actual differences per component without rewriting the immutable decision made at `t`.

### Expected edge and Economic Gate

`EconomicGate` (`balanced-economic-gate / 1.0`) applies only to new risk. It compares an independently provenanced `ExpectedEdgeEstimate` with estimated round-trip variable cost. ML probability is never converted into expected return. The optional `HistoricalEdgeEstimator` accepts only observations strictly before the decision from explicit `TRAIN` or `VALIDATION` regions; it refuses final `TEST` observations and returns `UNAVAILABLE` below its minimum sample size.

With a valid edge, the gate calculates expected net edge and edge-to-cost ratio against fixed, non-optimized thresholds. It returns `PASS`, `BLOCK`, `INCOMPLETE`, or `NOT_APPLICABLE`. Missing edge defaults to `INCOMPLETE_ALLOW_RESEARCH`: the run may continue for instrumentation, but it cannot be presented as economically validated. Incomplete critical cost coverage blocks new risk. `EXIT_LONG` always receives `NOT_APPLICABLE` pass-through. Economic eligibility cannot authorize execution or override `BalancedRiskEngine`.

### Validation Gate and real-data campaign

`ResearchValidationGate` (`balanced-research-validation / 1.0`) emits an immutable, checksum-protected report. Its fixed predeclared criteria include dataset integrity and DataQuality, final out-of-sample status, no overlap with ML/edge calibration, complete variable-cost coverage, tariff verification for the tested period, at least 30 closed trades, positive net return and expectancy, profit factor above 1, drawdown within the 10% Balanced hard limit, coherent circuit-breaker states, and non-negative cash. Missing operating costs prevent complete deployment economics. Cost stress is reported at 1.0x, 1.25x, 1.5x, and 2.0x; chronological subperiod and symbol concentration reports expose fragility without automatically selecting a winner.

Statuses are `PASS`, `WARNING`, `FAIL`, and `BLOCKED_EXTERNAL_DATA`. `PASS` means only that the configured research checks were satisfied; it never means safe, profitable, approved for Paper, or approved for `LIVE`. A synthetic fixture cannot be relabeled as real data through the CLI. Every report retains `SURVIVORSHIP_BIAS_NOT_RESOLVED`, because the current universe is not historical point-in-time membership. Final TEST data must never tune strategy, ML, regimes, portfolio, Risk, expected-edge thresholds, buffers, tariffs, or cost assumptions.

Yahoo remains a replaceable research adapter. TLS certificate validation is mandatory; `verify=False` is forbidden. When a host requires additional trusted corporate roots, a CA bundle may be assembled from `certifi` plus roots already trusted by the operating system, without disabling verification. `LocalHistoricalFileProvider` is an explicit mapped CSV/Parquet fallback for real files supplied by the user; it rejects absolute/path-traversal mappings and still routes data through UTC normalization, validation, manifests, and SHA-256 storage.

The controlled Lot 8.1 campaign used checksum-verified Yahoo daily data for all 13 configured Balanced symbols from 2020-01-01 through 2025-01-01. TLS verification remained enabled, every dataset passed integrity and DataQuality checks, and the backtest retained raw-price/corporate-action provenance. The research Validation Gate returned `FAIL`, not a fabricated financial approval: 16 trades closed versus the predeclared minimum of 30, max drawdown was 11.23% versus the 10% Balanced limit, and the current IBKR reference tariff is not historically verified for that period. Operating costs are also unavailable, results were concentrated in one symbol, and `SURVIVORSHIP_BIAS_NOT_RESOLVED` remains explicit. The stress table stayed positive through 2x modeled variable costs, but this does not override any failed criterion or establish future profitability.

Transaction costs reduce measured performance and cash capacity; their estimates can still differ from real execution. Validation gates reduce research error; they do not eliminate market risk or guarantee profitability.

## Real-data robustness and holdout governance

Lot 8.2 freezes the Lot 8.1 research state before performing any remediation analysis. `ResearchBaselineManifest` records the exact commit/source hash, result hash, dataset and corporate-action checksums, Feature/Regime/Strategy/ML/Policy/Portfolio/Risk/Cost/Validation hashes, tariff evidence, period, and original `FAIL`. The 2020-01-01 through 2025-01-01 observation is permanently labelled `CONSUMED_DIAGNOSTIC_OOS_V1`: it can be reproduced and diagnosed, but it is not a fresh final test after any decision-changing modification. Its immutable causes remain the historically unverified tariff, 16 rather than 30 closed trades, and 11.23% rather than at most 10% drawdown. Roughly 79% of positive realized P&L came from AAPL, which is a fragility warning, not a reason to delete AAPL after the fact.

`RobustnessResearchPlan` V2.0.1 is frozen and SHA-256 hashed before the final holdout is evaluated. It retains the original thresholds and configurations and predeclares coverage, funnel, drawdown, concentration, leave-one-out, temporal, regime, cost-stress, uncertainty, survivorship, holdout-adequacy, and readiness analyses. Holdouts move only through `UNTOUCHED`, `CONSUMED`, and `INVALIDATED`. The first final evaluation consumes the holdout; an identical rerun is permitted only for reproducibility. A changed decision-core/config hash makes the consumed period ineligible as new final evidence. Training, expected-edge calibration, configuration selection, and ML promotion are forbidden consumers of an untouched or consumed final holdout.

The V2 holdout uses the first complete common local session after V1 through the last complete common session observed before evaluation. UTC boundaries account for XPAR daily bars stamped at local midnight; incomplete provider rows are rejected rather than repaired. `INSUFFICIENT_HOLDOUT_EVIDENCE` remains the only valid outcome if duration or the frozen 30-trade minimum is not met. A completed Lot 8.2 implementation does not imply a passing campaign.

The frozen V2 holdout was consumed once for 2025-01-02 through the last complete common session before 2026-08-28. Run `bt-db4b176497845f51c77f10c8` closed 38 trades, returned 10.88% net before operating costs, recorded 3.37% backtest max drawdown, profit factor 3.03, and net expectancy 202.94 in account currency. These figures are reported, not used for retuning. Report `robustness-32a8615bf46fc13c64fda98e` retains `FAIL` because the unchanged Validation Gate rejects the historically unverified broker tariff; operating economics and point-in-time universe membership also remain incomplete. The resulting Paper Readiness status is therefore `NOT_READY`, and no broker, Paper, or `LIVE` capability is enabled.

## Evidence Closure & Paper Readiness V2

Lot 8.3 adds a checksum-verified, offline `EvidenceRegistryV2` and immutable evidence reassessment. Evidence records carry provider, market, fee type and plan, effective dates, acquisition/archive dates, original and archive references, normalized rules, verification state, and stable SHA-256 hashes. Accepted source classes are `CURRENT_OFFICIAL_SOURCE`, `HISTORICAL_OFFICIAL_SOURCE`, `ARCHIVED_OFFICIAL_SOURCE`, `REGULATORY_SOURCE`, `BROKER_STATEMENT`, `ESTIMATE`, and `UNAVAILABLE`. A Wayback capture is historical evidence only when it identifies an archived official URL and timestamp; a current official page alone is never silently treated as proof for an earlier period. Runtime backtests do not scrape any source.

Tariff comparison reports `EXACT_MATCH`, `COMPATIBLE_CONSERVATIVE`, `NUMERICALLY_DIFFERENT`, `INSUFFICIENT_EVIDENCE`, or `NOT_APPLICABLE`. Conservative compatibility requires component-wise mathematical dominance over fixed, per-unit, proportional, minimum, and cap rules for every dated interval. A higher modeled cost is not enough if it could have changed Economic Gate, Risk, orders, or fills. Conflicting official records remain unresolved rather than selecting a convenient source.

The official archived Interactive Brokers commission pages covering the V2 holdout confirm the frozen US `IBKR Pro Fixed` commission rule: USD 0.005 per share, USD 1 minimum, capped at 1% of trade value. That broker-commission component is therefore `EXACT_MATCH`. The separate official IBKR fee-scope disclosure and SEC FY2025/FY2026 advisories also establish that Section 31 is charged separately on covered US sells: zero from 2025-05-14 through 2026-04-03, then USD 20.60 per million from 2026-04-04. The consumed run recorded zero exchange/pass-through fee on affected 2026 sells. Lot 8.3 therefore does **not** issue an evidence-only PASS: it reports `ECONOMIC_RECOMPUTATION_REQUIRED`, approximately USD 0.9965 of indicated omitted Section 31 charges, and preserves the holdout as `CONSUMED`. This diagnostic amount does not rewrite the ledger or the historical decision.

French FTT remains dated and issuer-specific: the official 30 bps schedule changes to 40 bps from 2025-04-01, and annual MC.PA/AIR.PA eligibility comes from the existing official BOFiP lists rather than ticker suffixes. No French instrument filled in the V2 holdout, so FTT is `NOT_APPLICABLE` to that run. All filled instruments use explicit USD metadata, so no FX conversion occurred; the current official IBKR FX schedule is retained as current evidence only, not historical proof. Ordinary exchange/clearing fees included by the Fixed plan are distinguished from the separately charged SEC fee to prevent double counting.

`PAPER_ESTIMATE_V1` is a read-only operating-cost scenario with low/central/high monthly ranges for current market-data entitlements, local-or-future VPS use, software, and other fixed costs. Values are `VERIFIED`, `ESTIMATED_CONSERVATIVE`, or `UNAVAILABLE`; no subscription is purchased and no deployment provider is selected. The retrospective report shows net before operating costs, net after each estimated range, variable/fixed break-even capacity, observed profit-to-cost ratio, and whether the pre-existing 2x cost stress remains positive. It is an analytical scenario, not a forecast or invoice.

Evidence reassessment compares stable hashes for Signals, ML, Activation, Portfolio, Economic, Risk, Orders, Fills, cost estimates, actual costs, and reconciliations. Only strict identity permits `EVIDENCE_ONLY_RECLASSIFICATION`. Numeric cost differences require `ECONOMIC_RECOMPUTATION_REQUIRED`; a changed decision core produces `DECISION_CORE_CHANGED`. None of these modes can relabel the consumed holdout as untouched or fresh.

`PaperReadinessReviewV2` returns `READY_FOR_REVIEW`, `NOT_READY`, or `INSUFFICIENT_EVIDENCE`. It preserves the frozen Validation hard checks, cost completeness, holdout governance, operating scenario, survivorship warning, and concentration warning. `READY_FOR_REVIEW` would mean only that a human may review whether to start a separate no-capital Paper-development lot; it is not Live readiness and unlocks nothing. The real V2 evidence assessment is `NOT_READY` because the SEC fee discrepancy requires economic recomputation. V1 remains historically `FAIL`, the 2012–2025 campaign remains diagnostic `WARNING`, and survivorship bias remains unresolved.

Evidence commands are explicit and offline:

```powershell
trading-ai evidence verify --json
trading-ai evidence compare --run-id bt-db4b176497845f51c77f10c8 --report-id robustness-32a8615bf46fc13c64fda98e --operating-scenario PAPER_ESTIMATE_V1 --json
trading-ai evidence inspect --reassessment-id evidence-reassessment-example --json
trading-ai validation reassess-evidence --run-id bt-db4b176497845f51c77f10c8 --report-id robustness-32a8615bf46fc13c64fda98e --json
trading-ai validation paper-readiness --version 2 --reassessment-id evidence-reassessment-example --json
```

Schema 1.8 evidence bundles store `evidence_registry`, `evidence_reassessment`, `economic_completeness`, `paper_operating_scenario`, and `paper_readiness_v2` with SHA-256 checksums without modifying schema 1.6 backtests or schema 1.7 robustness reports. The Dashboard exposes the same read-only facts through `/api/v1/paper-readiness`; older runs show `UNAVAILABLE` rather than invented evidence. There is no browser control, broker adapter, Paper session, credential, or `LIVE` switch.

## Economic Recomputation & Human Paper Readiness V3

Lot 8.4 leaves run `bt-db4b176497845f51c77f10c8`, its schema 1.6 export, the frozen Research Plan, the schema 1.7 robustness report, the schema 1.8 Evidence Registry bundle, and the historical `Final Holdout V2: FAIL / CONSUMED` status immutable. A separate `balanced-economic-recomputation / 1.1` analytical model reads the already verified `SEC_SECTION_31` rules from `EvidenceRegistryV2`; it does not alter `balanced-transaction-cost / 1.0` or any Strategy, Feature, Regime, ML, Policy, Portfolio, Risk, Expected Edge, Economic Gate, or Validation configuration. The decision-core hash must remain identical to the Lot 8.3 baseline before any recomputation is accepted.

Section 31 remains distinct from IBKR Fixed commission, ordinary exchange/clearing costs included by that plan, spread, slippage, French FTT, FX, financing, and operating costs. Only verified covered US `SELL` fills receive the point-in-time rule active at their timestamp. A `BUY`, an uncovered instrument, or a sale outside the verified scope is `NOT_APPLICABLE`; a missing or conflicting required rule yields `INSUFFICIENT_EVIDENCE`. The recomputation also evaluates the entry-time estimated exit leg using only the rule known at that time. A future regulatory rate cannot leak into an earlier Economic Gate decision.

`EconomicRecomputationReport` records original/new engine and config hashes, every affected fill and exact notional/rate/fee, original and recomputed ledger cash change, trade-level cost allocation, equity/cash curves, P&L/return/drawdown/profit-factor/expectancy deltas, stress costs, PAPER operating scenarios, evidence IDs, and economic completeness. It proves invariance separately for Feature evidence, Regimes, Signals, ML, Activation, Portfolio, Economic eligibility, Risk, Orders, and Fills. A changed eligibility status, Risk quantity, order, fill, or decision-core hash is surfaced as `ECONOMIC_DECISION_CHANGED`, `RISK_DECISION_CHANGED`, `ORDER_OR_FILL_CHANGED`, or `DECISION_CORE_CHANGED`; it is never hidden by the small nominal fee.

The recomputed period is always labelled `CONSUMED_HOLDOUT_ECONOMIC_RECOMPUTATION`. It is retrospective analysis, never fresh OOS. Schema 1.9 stores the recomputation, affected fills/trades, recomputed equity, decision-invariance proof, evidence completeness, unchanged `PAPER_ESTIMATE_V1`, `PaperReadinessReviewV3`, and the initial `AWAITING_HUMAN_REVIEW` record in a checksum-verified bundle beside—never inside—the original export.

`PaperReadinessReviewV3` uses the frozen minimum of 30 closed trades, positive net return/expectancy, profit factor above 1, maximum drawdown at most 10%, non-negative cash, complete-enough applicable transaction-cost evidence, a present PAPER operating scenario, and strict decision-chain invariance. `READY_FOR_REVIEW` means only that a person may assess whether to authorize development of Lot 9. It does not enable a broker, Paper session, order transmission, credentials, or `LIVE`. Human acceptance or rejection requires a separate local CLI action and a non-empty audited reason; generated results never approve themselves. This Lot deliberately ends with `AWAITING_HUMAN_REVIEW`.

The real consumed-holdout recomputation finds eight covered US sells after the FY2026 effective date. Applying each verified USD 20.60-per-million rule at fill precision adds USD 0.99650280 exactly in the analytical ledger. Variable costs move from USD 466.55817886 to USD 467.55468166; net P&L before operating costs moves from USD 10,881.80249268745 to USD 10,880.80598988745; net return moves from 10.88180249% to 10.88080599%; maximum drawdown moves from 3.37285977% to 3.37339833%; profit factor moves from 3.02823661 to 3.02775236; and net expectancy moves from USD 202.94168713 to USD 202.91546337. The minimum cash remains non-negative and the +100% variable-cost stress remains positive. All decision layers are `STRICTLY_INVARIANT`, applicable transaction evidence is `COMPLETE_ESTIMATED` once the separate Paper operating range is included, and the resulting status is `READY_FOR_REVIEW / AWAITING_HUMAN_REVIEW`. These are retrospective consumed-period facts, not fresh validation or a forecast.

Commands remain offline and explicit:

```powershell
trading-ai validation recompute-economics --run-id bt-db4b176497845f51c77f10c8 --report-id robustness-32a8615bf46fc13c64fda98e --reassessment-id evidence-reassessment-c5325de4c2cc2c75482b4697 --operating-scenario PAPER_ESTIMATE_V1 --json
trading-ai validation economic-recomputation --id economic-recomputation-example --json
trading-ai validation paper-readiness --version 3 --recomputation-id economic-recomputation-example --json
trading-ai validation human-review --readiness-id paper-readiness-v3-example --decision status --json
trading-ai validation human-review --readiness-id paper-readiness-v3-example --decision accept-for-lot9-development --reason "Explicit methodological review for Lot 9 development only"
```

The final acceptance command is documented for a future, separate user decision; it is not executed as part of Lot 8.4. The Dashboard displays original versus recomputed economics, Section 31 affected sells, decision invariance, operating ranges, survivorship and historical concentration warnings, Readiness V3, and human-review state. It remains read-only and contains no accept/reject or trading controls.

### Diagnostics, not retuning

The read-only robustness layer consumes checksum-verified exports and never participates in signal, sizing, economics, Risk, or execution. It produces:

- a monotone decision funnel from candidate entry through ML, Activation Policy, Portfolio, Economic Gate, Risk, fill, and closed trade, with exits reported separately;
- drawdown episodes with peak, trough, recovery, duration, held symbols, observed realized contributors, regimes, Risk states/decisions, exposure, and costs;
- per-symbol gains/losses/trades/exposure plus top-1/top-3 positive-P&L share and HHI;
- calendar-year and predeclared chronological subperiod reports, with empty periods explicitly unavailable;
- regime-conditioned entry/trade observations only where exported lineage permits them;
- baseline, +25%, +50%, and +100% variable-cost stress, cost/trade, cost/notional, cost/gross-profit, and estimate-versus-actual error;
- fixed-seed non-parametric uncertainty only when sample size is adequate; small samples remain `INSUFFICIENT_FOR_RELIABLE_CI`;
- leave-one-symbol-out and leave-one-strategy-out as `POST_HOC_ROBUSTNESS_DIAGNOSTIC`, never as universe/strategy selection or automatic sleeve reallocation;
- single-strategy comparisons on the same frozen data, costs, Regime Policy, and Risk assumptions; they preserve the explicit legacy one-strategy construction path and never reallocate the fixed multi-strategy sleeves.

For the consumed V1 run, report `robustness-62a4b3d15f52d5fa505d676d` reconstructs 2,329 candidate entry signals, 284 activation-eligible candidates, 41 Portfolio selections, nine Risk-approved/reduced entry orders, and nine filled entries, followed by 16 closed trades. The 11.23% drawdown crossed the unchanged hard 10% limit in September 2020, latched Risk to `HALTED` for new exposure, and explains why subsequent candidate activity did not create a larger independent trade sample. This is a causal audit of exported decisions, not a justification to relax the Risk threshold.

The `HistoricalCoverageMatrix` reports each symbol/timeframe’s actual first/last bar, row count, DataQuality, checksums, corporate-action coverage, common coverage, and provider limitations. `AVAILABLE_HISTORY_DIAGNOSTIC` and a coverage-derived common-history campaign are kept separate. Histories with inconsistent OHLCV remain rejected or explicitly partial; timeframes are never pooled to manufacture the trade minimum.

The controlled coverage probe requested daily history from 2000 without relaxing validation. Eight symbols (`AAPL`, `AMZN`, `ASML`, `MSFT`, `NVDA`, `QQQ`, `SAP`, `SPY`) had valid coverage from 2000-01-03, `IWM` from 2000-05-26, `GOOGL` from 2004-08-19, and `META` from 2012-05-18. The longer `MC.PA` response contained eight invalid 2000 OHLC bars and the longer `AIR.PA` response one invalid 2008 OHLC bar; those responses were rejected rather than silently repaired. The predeclared common-history diagnostic consequently begins with the latest required history in May 2012 and keeps provider/calendar warnings visible. Run `bt-8ed3bd4def50a1e903608c4f` closed 397 trades through 2024, returned 357.90% net before operating costs, and recorded 7.60% maximum drawdown. Report `robustness-f1bb985fb77718c3166886e7` is deliberately `WARNING`, not final evidence: `MC.PA`/`AIR.PA` retain calendar-quality warnings, historical tariff proof and operating costs are incomplete, and survivorship bias remains unresolved. Intraday histories are not pooled with daily results.

### Historical cost and universe evidence

Historical evidence distinguishes a current official source from an archived official source. The current official IBKR schedule remains `HISTORICAL_TARIFF_UNVERIFIED` for retrospective runs unless a dated official/archive artifact proves the applicable schedule; `CURRENT_TARIFF_APPLIED_RETROSPECTIVELY` is always disclosed. French FTT uses dated official rates and explicit annual official eligibility records rather than ticker suffixes. Unknown exchange/pass-through or FX evidence remains `UNAVAILABLE`, never zero. `LOCAL_RESEARCH`, `PAPER_ESTIMATE`, and `LIVE_ESTIMATE` operating-cost scenarios keep market data, server/VPS, software, and other fixed costs separate and source-provenanced. `VARIABLE_COST_COMPLETE`, `OPERATING_COST_COMPLETE`, `COMPLETE_ESTIMATED`, and `COMPLETE_VERIFIED` are distinct claims.

`PointInTimeUniverse` and dated `UniverseMembership` contracts prepare a future unbiased membership campaign, but the configured 13-symbol run remains `CURATED_CURRENT_UNIVERSE` with `SURVIVORSHIP_BIAS_UNRESOLVED`. Leave-one-out does not retroactively redefine it.

`PaperReadinessReport` is read-only and returns `READY_FOR_REVIEW`, `NOT_READY`, or `INSUFFICIENT_EVIDENCE`. It combines the unchanged ValidationGate status, holdout adequacy, historical cost evidence, operating-cost completeness, concentration, uncertainty, and survivorship warnings. It never enables a broker, Paper, or `LIVE`.

Robustness commands are explicit and offline once datasets/runs exist:

```powershell
trading-ai robustness plan --json
trading-ai robustness run --run-id bt-example --period-classification CONSUMED_DIAGNOSTIC --json
trading-ai robustness inspect --report-id robustness-example --json
trading-ai robustness compare --run-id bt-example --without-symbol AAPL=bt-without-aapl --json
trading-ai robustness compare --run-id bt-example --single-strategy trend=bt-trend-only --json
trading-ai validation holdout-status --json
trading-ai validation paper-readiness --report-id robustness-example --json
```

The local Dashboard adds a read-only Robustness view for the frozen baseline, consumed OOS, final-holdout lifecycle, decision funnel, drawdowns, temporal/symbol concentration, leave-one-out completeness, historical cost evidence, survivorship, and Paper readiness. Older schemas remain readable and show unavailable Lot 8.2 fields rather than invented values.

## CLI

Safety diagnostics remain available:

```powershell
trading-ai doctor
trading-ai doctor --environment PAPER --profile balanced --json
trading-ai risk inspect --profile balanced --json
trading-ai regime policy --profile balanced --json
trading-ai portfolio inspect --profile balanced --json
trading-ai costs inspect --profile balanced --cost-profile ibkr_pro_fixed --json
trading-ai costs verify-config --profile balanced --cost-profile ibkr_pro_tiered --json
trading-ai costs estimate --profile balanced --symbol AAPL --side BUY --quantity 10 --price 100 --timeframe 1d --timestamp 2026-08-29T16:00:00Z --spread-bps 5 --slippage-bps 2 --json
```

`risk inspect` validates and displays engine/version, limits, drawdown/daily-loss rules, volatility/correlation policies, concentration groups, and the deterministic config hash. Inspecting `aggressive` reports it as locked and disabled; it does not activate it.

Regime commands read local Parquet datasets only and never fetch missing data:

```powershell
trading-ai regime inspect --profile balanced --symbol SPY --timeframe 1d --start 2024-01-01 --end 2025-01-01 --json
trading-ai regime latest --profile balanced --symbol SPY --timeframe 1d --json
trading-ai regime policy --profile balanced --json
```

`inspect` reports the latest two-axis snapshot, evidence, confirmation progress, transitions, time by regime, exact dataset ID/checksum, and detector/config provenance. `latest` resolves the latest local manifest. `policy` displays the TOML-driven structure matrix and volatility overlays. Aggressive detector and policy schemas exist but remain disabled and cannot be activated.

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
trading-ai backtest run --strategy buy-and-hold --symbol SPY --timeframe 1d --start 2024-01-01 --end 2025-01-01 --starting-cash 100000 --spread-bps 5 --slippage-bps 5 --cost-profile ibkr_pro_fixed --json
trading-ai backtest run --strategy trend --symbol SPY --timeframe 1d --start 2024-01-01 --end 2025-01-01 --spread-bps 5 --slippage-bps 5 --cost-profile ibkr_pro_fixed --json
trading-ai backtest run --strategy momentum --symbol SPY --symbol QQQ --symbol IWM --timeframe 1d --start 2024-01-01 --end 2025-01-01 --top-k 2 --rebalance-every 5 --spread-bps 5 --slippage-bps 5 --cost-profile ibkr_pro_tiered --json
trading-ai backtest run --strategy breakout --symbol AAPL --timeframe 4h --start 2024-06-01 --end 2024-07-01 --entry-window 20 --exit-window 10 --spread-bps 5 --slippage-bps 5 --cost-profile ibkr_pro_fixed --json
trading-ai backtest run --strategy mean-reversion --symbol SPY --timeframe 1d --start 2024-01-01 --end 2025-01-01 --mean-reversion-lookback 20 --entry-zscore -1.5 --exit-zscore -0.25 --spread-bps 5 --slippage-bps 5 --cost-profile ibkr_pro_fixed --json
trading-ai backtest run --strategy trend --symbol SPY --timeframe 1d --start 2024-01-01 --end 2025-01-01 --ml-mode score-only --ml-model-id ml-example-id --json
trading-ai backtest run --strategy trend --symbol SPY --timeframe 1d --start 2024-01-01 --end 2025-01-01 --ml-mode filter --ml-model-id ml-approved-id --ml-threshold 0.55 --json
trading-ai backtest run --strategy trend --strategy momentum --strategy breakout --strategy mean-reversion --symbol SPY --symbol QQQ --symbol AAPL --timeframe 1d --start 2024-01-01 --end 2025-01-01 --spread-bps 5 --slippage-bps 5 --cost-profile ibkr_pro_fixed --json
trading-ai backtest run --strategy trend --strategy momentum --symbol SPY --symbol QQQ --timeframe 1d --start 2024-01-01 --end 2025-01-01 --ml-mode filter --ml-model-id trend=ml-trend-approved --ml-model-id momentum=ml-momentum-approved --json
trading-ai backtest inspect --run-id bt-0123456789abcdef01234567 --json
trading-ai strategy list --json
```

The demo submits one buy intent. Every CLI run explicitly injects the selected `BalancedTransactionCostEngine`, `EconomicGate`, and `BalancedRiskEngine`; quantitative baselines also inject `BalancedRegimeDetector` and `BalancedStrategyActivationPolicy`. There is no CLI flag to bypass economics, regime eligibility, or Risk. Non-zero legacy `--commission-*` flags are rejected so commission cannot be charged twice. Repeating `--strategy` creates one shared multi-strategy run and explicitly injects `BalancedPortfolioEngine`; it does not launch independent backtests. ML is `disabled` by default. Active scoring always requires an explicit model ID; in a multi-strategy run each model is mapped as `strategy=model-id`. `FILTER` additionally requires `APPROVED`, and there is no latest-model fallback. A missing/invalid configuration fails closed. Every command uses the first selected symbol as its configurable Buy & Hold benchmark unless `--benchmark-symbol` names another cached profile symbol. Strategy-specific window/allocation flags override immutable defaults for that run, and the resolved values are stored in `BacktestResult`. `run` exports under `data_local/backtests/`; `inspect` verifies checksums and includes ML, regime/policy, portfolio, cost, economic, validation, and risk summaries. A cache miss fails clearly without falling back to Yahoo Finance.

Validation is always explicit and offline:

```powershell
trading-ai validation run --run-id bt-0123456789abcdef01234567 --final-oos-confirmed --no-training-edge-overlap-confirmed --real-data --json
trading-ai validation inspect --validation-id validation-0123456789abcdef01234567 --json
```

`--real-data` is checked against dataset provider provenance and cannot convert an in-memory/synthetic export into a real campaign. A missing flag or unavailable external dataset yields `BLOCKED_EXTERNAL_DATA`; a historical run using a current tariff outside its effective dates fails the strict tariff criterion. Reports are stored below `data_local/validation/`, never activate a profile, and never authorize Paper or `LIVE`.

ML training and lifecycle commands also use exact local cache entries only:

```powershell
trading-ai ml train --strategy trend --timeframe 1d --model logistic --symbol SPY --train-start 2018-01-01 --train-end 2021-01-01 --validation-start 2021-01-01 --validation-end 2022-01-01 --test-start 2022-01-01 --test-end 2023-01-01 --json
trading-ai ml evaluate --model-id ml-example-id --json
trading-ai ml model list --json
trading-ai ml model inspect --model-id ml-example-id --json
trading-ai ml model promote --model-id ml-example-id --to VALIDATED --reason "temporal validation reviewed"
trading-ai ml model promote --model-id ml-example-id --to APPROVED --reason "explicit research approval"
trading-ai ml model rollback --strategy trend --timeframe 1d --reason "restore prior approved artifact"
```

Training replays the selected quant baseline without ML to build candidate examples, then stores a `CANDIDATE`; it never downloads, promotes, or chooses a model automatically. Promotion and rollback update an auditable local registry alias only.

## Tests and CI

The default suite is deterministic, fast, and independent of Yahoo Finance and the network. It uses `FakeDataProvider` plus synthetic sessions, invalid bars, gaps, corporate actions, feature/regime warm-ups, future-append invariance, exact-timestamp relative strength, all four baselines, two-axis classification, confirmation/transitions, activation matrices, Mean Reversion eligibility/exits/no-averaging, monotonic ML/policy/portfolio/risk sizing, chronological labels/splits/purge/embargo, all three tabular adapters, registry integrity/lifecycle, inference modes, multi-strategy batching/netting/diversification/turnover/FX, current-close sizing, fee-aware pending reservations, fixed/tiered/proportional/capped commissions, dated tax/FX/operating-cost states, point-in-time Section 31, Economic Gate behavior, OOS/stress/robustness validation, frozen-baseline/holdout governance, decision-funnel/drawdown/concentration/temporal/cost/uncertainty diagnostics, official evidence provenance/conflicts, exact/conservative/different tariff comparison, decision/cost invariance, economic recomputation, Paper operating ranges, human-review audit, consumed-holdout reassessment, point-in-time universe contracts, shared-ledger Risk integration, backward-compatible tamper-evident exports, immutable monitoring events, SQLite round trips, all Dashboard API/UI sections, schemas 1.0–1.9, path traversal/corruption refusal, full cost/decision lineage, local-only serving, and architecture audits. Synthetic patterns validate mechanics only; they are not evidence of market edge or profitability.

```powershell
.\.venv\Scripts\python -m compileall -q src
.\.venv\Scripts\python -m pytest -m "not integration"
.\.venv\Scripts\python -m pip check
```

GitHub Actions performs the same compile, offline test, and dependency-consistency checks on pushes and pull requests to `main`. It never downloads market data or models. A real Yahoo smoke test is opt-in only:

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

Lots 0 through 8.4 now provide foundations, universe/CI alignment, historical data, deterministic simulation, shared Feature Engine 1.1, four quantitative research baselines, the offline Balanced Risk Engine, two-axis rule-based regime classification, governed tabular ML scoring, deterministic multi-strategy portfolio construction, a local read-only Dashboard/observability boundary, configuration-driven transaction economics plus a research Validation Gate, frozen real-data robustness/holdout governance, official-evidence review, and immutable consumed-holdout economic recomputation with explicit human readiness governance. Lot 9 — Broker / Paper Trading remains a future manual decision; Lot 10 — Balanced Paper Validation and Lot 11 — Limited Live remain TODO. Neural, sequence, multimodal, real-time, and online-learning research remains PLANNED / LOCKED, and Aggressive Research remains LOCKED. See `PROJECT_STATE.md` for the authoritative implementation, campaign, and readiness statuses.
