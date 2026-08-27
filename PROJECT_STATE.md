# Project State

## Delivery lots

- Lot 0 — Foundations: DONE
- Lot 0.1 — Market universe alignment + CI: DONE
- Lot 1 — Data Engine: DONE
- Lot 2 — Backtesting Engine: DONE
- Lot 3 — Feature Engine & Quantitative Baselines: DONE
- Lot 4 — Balanced Risk Engine: DONE
- Lot 5 — Regime Detector + Mean Reversion + Strategy Activation: DONE
- Lot 6 — Machine Learning: TODO
- Lot 7 — Portfolio Engine: TODO
- Lot 8 — Dashboard: TODO
- Lot 9 — Broker / Paper Trading: TODO

## Future branch: Aggressive Research

Status: LOCKED

The aggressive profile is represented in the configuration schema for forward compatibility, but Lot 0 rejects its activation unconditionally.

## Lot 1 delivery summary

Delivered: provider-neutral historical data retrieval, deterministic offline fixtures, explicit OHLCV validation, UTC normalization, exchange-aware gap detection, session-anchored 4h derivation, corporate actions, Parquet storage, JSON manifests, SHA-256 integrity checks, exact-range cache modes, bounded retries, and data CLI diagnostics.

## Lot 2 delivery summary

Delivered: chronological look-ahead-safe simulation, next-bar MARKET and deterministic LIMIT execution, configurable spread/slippage/commissions, cash-only long portfolio ledger, explicit dividends and splits, FIFO trades, equity curve, independent performance metrics, configurable Buy & Hold benchmark, DataQuality policy, exact dataset provenance, stable source/result hashes, local JSON/Parquet exports, and offline backtest CLI commands.

## Lot 3 delivery summary

Delivered: versioned shared Feature Engine with explicit warm-up and anti-look-ahead semantics; stable trend, momentum, volatility, volume, and previous-range features; exact-timestamp multi-asset relative strength; explainable immutable signals; profile-capped temporary baseline sizing; versioned Trend, Momentum, and Breakout research baselines; registry and CLI selection; feature/signal lineage in deterministic backtest results; benchmark reporting; offline synthetic tests and smoke runs. Aggressive Research remains LOCKED.

## Lot 4 delivery summary

Delivered: explicitly injected `BalancedRiskEngine` 1.0 with immutable APPROVE/REDUCE/REJECT decisions; profile-bounded TOML limits and SHA-256 provenance; cash, long-only, position-count, portfolio/single-position, concentration, exact-timestamp correlation, shared-feature volatility, and explicit trade-risk sizing guards; deterministic daily-loss/drawdown states and circuit breaker; risk-reducing exits while HALTED; pending-order risk reservations; mandatory Backtester gate with `DenyAllRiskEngine` default; decision/state metrics and backward-compatible JSON/Parquet exports; risk CLI diagnostics; offline baseline integration, safety, determinism, and tamper-detection tests. Aggressive Research remains LOCKED.

## Lot 5 delivery summary

Delivered: Feature Engine schema 1.1 with Efficiency Ratio, historical volatility percentile, and price z-score; deterministic `BalancedRegimeDetector` 1.0 with separate structure/volatility axes, conservative UNKNOWN, confirmation and transitions; TOML-driven `BalancedStrategyActivationPolicy` 1.0 with monotonic ALLOW/REDUCE/BLOCK decisions; long-only Mean Reversion 1.0 restricted to eligible RANGE context with no averaging down; exact signal/regime/activation/order/risk/fill lineage; Momentum per-asset regime filtering; schema 1.3 JSON/Parquet exports and backward inspection through schema 1.0; offline regime/policy CLI diagnostics; append-future, determinism, safety, integration, and architecture tests. Lot 6 remains TODO and Aggressive Research remains LOCKED.
