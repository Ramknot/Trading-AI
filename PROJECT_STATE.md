# Project State

## Delivery lots

- Lot 0 — Foundations: DONE
- Lot 0.1 — Market universe alignment + CI: DONE
- Lot 1 — Data Engine: DONE
- Lot 2 — Backtesting Engine: DONE
- Lot 3 — Feature Engine & Quantitative Baselines: DONE
- Lot 4 — Balanced Risk Engine: DONE
- Lot 5 — Regime Detector + Mean Reversion + Strategy Activation: DONE
- Lot 6 — ML Platform Foundation & Statistical Scoring: DONE
- Lot 7 — Multi-Strategy Portfolio Engine: DONE
- Lot 8 — Dashboard & Observability: DONE
- Lot 8.1 — Validation Gate & Transaction Cost Economics: TODO
- Lot 9 — Broker / Paper Trading: TODO
- Lot 10 — Balanced Paper Validation: TODO
- Lot 11 — Limited Live: TODO

## Future branch: Aggressive Research

Status: LOCKED

The aggressive profile is represented in the configuration schema for forward compatibility, but Lot 0 rejects its activation unconditionally.

## Future ML Evolution

Status: PLANNED / LOCKED

Sequence models, Temporal CNN, LSTM/GRU, Temporal Transformers, cross-sectional neural models, news/sentiment and macro inputs, bid/ask and order-book data, multimodal fusion, real-time inference, justified GPU inference, and controlled online-learning research remain locked until the Balanced platform is validated and the current components demonstrate value.

## Lot 1 delivery summary

Delivered: provider-neutral historical data retrieval, deterministic offline fixtures, explicit OHLCV validation, UTC normalization, exchange-aware gap detection, session-anchored 4h derivation, corporate actions, Parquet storage, JSON manifests, SHA-256 integrity checks, exact-range cache modes, bounded retries, and data CLI diagnostics.

## Lot 2 delivery summary

Delivered: chronological look-ahead-safe simulation, next-bar MARKET and deterministic LIMIT execution, configurable spread/slippage/commissions, cash-only long portfolio ledger, explicit dividends and splits, FIFO trades, equity curve, independent performance metrics, configurable Buy & Hold benchmark, DataQuality policy, exact dataset provenance, stable source/result hashes, local JSON/Parquet exports, and offline backtest CLI commands.

## Lot 3 delivery summary

Delivered: versioned shared Feature Engine with explicit warm-up and anti-look-ahead semantics; stable trend, momentum, volatility, volume, and previous-range features; exact-timestamp multi-asset relative strength; explainable immutable signals; profile-capped temporary baseline sizing; versioned Trend, Momentum, and Breakout research baselines; registry and CLI selection; feature/signal lineage in deterministic backtest results; benchmark reporting; offline synthetic tests and smoke runs. Aggressive Research remains LOCKED.

## Lot 4 delivery summary

Delivered: explicitly injected `BalancedRiskEngine` 1.0 with immutable APPROVE/REDUCE/REJECT decisions; profile-bounded TOML limits and SHA-256 provenance; cash, long-only, position-count, portfolio/single-position, concentration, exact-timestamp correlation, shared-feature volatility, and explicit trade-risk sizing guards; deterministic daily-loss/drawdown states and circuit breaker; risk-reducing exits while HALTED; pending-order risk reservations; mandatory Backtester gate with `DenyAllRiskEngine` default; decision/state metrics and backward-compatible JSON/Parquet exports; risk CLI diagnostics; offline baseline integration, safety, determinism, and tamper-detection tests. Aggressive Research remains LOCKED.

## Lot 5 delivery summary

Delivered: Feature Engine schema 1.1 with Efficiency Ratio, historical volatility percentile, and price z-score; deterministic `BalancedRegimeDetector` 1.0 with separate structure/volatility axes, conservative UNKNOWN, confirmation and transitions; TOML-driven `BalancedStrategyActivationPolicy` 1.0 with monotonic ALLOW/REDUCE/BLOCK decisions; long-only Mean Reversion 1.0 restricted to eligible RANGE context with no averaging down; exact signal/regime/activation/order/risk/fill lineage; Momentum per-asset regime filtering; schema 1.3 JSON/Parquet exports and backward inspection through schema 1.0; offline regime/policy CLI diagnostics; append-future, determinism, safety, integration, and architecture tests. Aggressive Research remains LOCKED.

## Lot 6 delivery summary

Delivered: framework-neutral `ModelAdapter`, `ModelTrainer`, `ModelRegistry`, and inference contracts with TABULAR operational and future input kinds reserved; ML feature schema 1.0 built only from shared point-in-time features/regimes; explicit next-bar-open/horizon-close labels; purged and embargoed chronological TRAIN/VALIDATION/FINAL TEST with expanding walk-forward validation; fixed Logistic Regression, Random Forest, and Gradient Boosting adapters; deterministic training and classification reports; checksum-verified local model registry with audited CANDIDATE/VALIDATED/APPROVED/RETIRED promotion and rollback; DISABLED/SCORE_ONLY/FILTER modes with exits always permitted and approved-model fail-closed filtering; ML/policy/risk sovereignty and lineage; schema 1.4 exports and backward inspection through schema 1.3; offline ML CLI, integration, leakage, reproducibility, and architecture tests. No broker, neural network, streaming, or online-learning path was added. Future ML Evolution and Aggressive Research remain LOCKED.

## Lot 7 delivery summary

Delivered: `BalancedPortfolioEngine` 1.0 with profile/Risk-bounded TOML configuration and SHA-256 provenance; fixed 15% Trend, Momentum, Breakout, and Mean Reversion sleeves; complete UTC opportunity batching, deterministic intra-strategy ranking, soft exact-timestamp correlation/group diversification, same-symbol aggregation, sleeve attribution, target netting, five-position/15%-symbol/60%-exposure construction caps, no-trade band, entry-turnover deferral, pending-order conflict handling, and current-close sizing; fail-closed point-in-time currency contracts with configuration-driven asset metadata; shared multi-strategy Backtester integration across ML modes, activation policy, mandatory Balanced Risk, and next-bar execution; full signal-to-fill portfolio lineage, schema 1.5 tamper-evident exports, backward inspection through schema 1.4, Portfolio CLI diagnostics, neutral mono-versus-multi research reporting, deterministic smoke runs, and offline architecture/safety tests. No broker, optimizer, neural network, streaming, online learning, leverage, or Aggressive path was added. Future ML Evolution remains PLANNED / LOCKED and Aggressive Research remains LOCKED.

## Lot 8 delivery summary

Delivered: immutable UTC monitoring events/snapshots, health and decision-trace contracts; checksum-verified `BacktestMonitoringSource` with schemas 1.0–1.5 and fingerprinted Parquet cache; a local SQLite event/snapshot store; deterministic read models for Overview, Equity/Drawdown, Portfolio, Strategies, Regimes, ML, Risk, Data Quality, Costs, System Health, decision history, and complete order lineage; a responsive FastAPI/Jinja2/vanilla-JS Dashboard and versioned JSON API restricted to loopback hosts and read-only HTTP routes; defensive metadata escaping/redaction and path-traversal/integrity refusal; explicit `KNOWN`/`ESTIMATED`/`UNAVAILABLE` trading and operating cost coverage with no unknown-to-zero conversion; Dashboard/Monitoring CLI diagnostics; and offline API/UI/store/security/backward-compatibility tests. No broker, order transmission, remote exposure, cloud service, market stream, strategy mutation, Risk bypass, or `LIVE` control was added. Lot 8.1 is reserved for the real Transaction Cost Engine and economic validation gate. Future ML Evolution remains PLANNED / LOCKED and Aggressive Research remains LOCKED.
