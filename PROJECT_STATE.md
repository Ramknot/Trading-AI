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
- Lot 8.1 — Validation Gate & Transaction Cost Economics: DONE
- Lot 8.2 — Real Data Robustness & Validation Remediation: DONE
- Lot 8.3 — Evidence Closure & Paper Readiness: DONE
- Lot 8.4 — Economic Recomputation & Human Paper Readiness Review: DONE
- Real Data Validation Campaign V1: FAIL
- Extended Historical Campaign: WARNING
- Final Holdout V2: FAIL / CONSUMED
- Holdout V2 Evidence Reassessment: FAIL
- Paper Readiness Review V2: NOT_READY
- Holdout V2 Economic Recomputation: PASS
- Decision Invariance: STRICTLY_INVARIANT
- Paper Readiness Review V3: READY_FOR_REVIEW
- Human Review: AWAITING_HUMAN_REVIEW
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

## Lot 8.1 delivery summary

Delivered: `BalancedTransactionCostEngine` 1.0 with dated, versioned, source-provenanced and SHA-256-hashed tariff/tax/instrument configurations; explicit KNOWN/ESTIMATED/NOT_APPLICABLE/UNAVAILABLE commission, spread, slippage, exchange, tax, FX, financing, other-variable, and operating-cost states; point-in-time entry/round-trip estimates, cost-aware cash and pending reservations, realized-cost ledger debits, reconciliation, and anti-double-count controls; a separate expected-edge contract and fail-closed `EconomicGate`; schema 1.6 lineage/exports and Dashboard monitoring; local CSV/Parquet historical ingestion; immutable `ResearchValidationGate` reports with OOS, integrity, cost coverage, tariff-period, sample, net metric, drawdown, cash, circuit-breaker, stress, subperiod, and symbol-concentration checks; offline Cost/Validation CLI commands and regression/security tests. No broker, Paper/`LIVE` unlock, neural network, stream, online learning, or test-driven threshold optimization was added.

The controlled real-data campaign used all 13 configured Balanced symbols at `1d` from 2020-01-01 through 2025-01-01 with checksum-verified Yahoo datasets and strict TLS verification. Dataset integrity and DataQuality passed, but the immutable campaign result is `FAIL`: the current IBKR reference tariff is not historically verified for the tested dates, only 16 trades closed versus the predeclared minimum of 30, and observed drawdown reached 11.23% versus the 10% limit. Operating costs remain unavailable and the configured universe still carries `SURVIVORSHIP_BIAS_NOT_RESOLVED`. This result does not unlock Paper or `LIVE`.

## Lot 8.2 delivery summary

Delivered: an immutable manifest reproducing the consumed Lot 8.1 baseline and a pre-evaluation hashed V2 research plan; tamper-evident `UNTOUCHED`/`CONSUMED`/`INVALIDATED` holdout governance; read-only coverage, decision-funnel, drawdown, symbol-concentration, yearly/subperiod, regime, cost-stress, deterministic uncertainty, leave-one-symbol-out, leave-one-strategy-out, and single-strategy diagnostics; dated official French FTT evidence plus explicit historical-broker-tariff and operating-cost incompleteness; point-in-time universe contracts with unresolved-survivorship disclosure; schema 1.7 robustness exports, CLI, Dashboard, and Paper-readiness review. The consumed V1 remains `FAIL`. The 2012–2025 common-history diagnostic is `WARNING` (397 trades, 7.60% maximum drawdown) because it is diagnostic data with incomplete historical economics and survivorship evidence. The frozen 2025–2026 V2 holdout is `CONSUMED` and `FAIL`: its 38 trades, 10.88% net return before operating costs, and 3.37% drawdown satisfy the unchanged sample/performance/Risk checks, but the unchanged tariff-period criterion fails. Paper readiness remains `NOT_READY`; no threshold was relaxed and no broker, Paper, or `LIVE` path was added.

## Lot 8.3 delivery summary

Delivered: immutable Evidence Registry V2 contracts with official/current/historical/archived/regulatory source classes, dated scope and stable SHA-256 provenance; deterministic exact-match, mathematically conservative, numerically-different, conflict, market, plan, and date-gap tariff assessments; strict decision-core and numeric-cost invariance checks; evidence-only versus economic-recomputation classification; schema 1.8 evidence bundles with tamper-evident checksums; retrospective `PAPER_ESTIMATE_V1` operating-cost ranges and break-even diagnostics; `EconomicEvidenceCompleteness` and read-only `PaperReadinessReviewV2`; offline Evidence/Validation CLI and Dashboard views. Official archived IBKR evidence confirms the modeled US Fixed commission for the consumed holdout, but official IBKR/SEC evidence also identifies a separately applicable Section 31 sell fee that was not debited in the original run. The outcome is therefore `ECONOMIC_RECOMPUTATION_REQUIRED`, evidence reassessment `FAIL`, economic completeness `INCOMPLETE`, and Paper readiness `NOT_READY`. All trading decisions, quantities, fills, and original numeric costs remain unchanged and auditable; the V2 holdout remains `CONSUMED`, no threshold or decision parameter was modified, and no broker, Paper, or `LIVE` path was added.

## Lot 8.4 delivery summary

Delivered: a separately versioned `balanced-economic-recomputation / 1.1` model reading verified, dated SEC Section 31 rules exclusively from `EvidenceRegistryV2`; point-in-time entry/exit applicability and anti-double-count safeguards; immutable per-fill/trade/ledger/equity reconciliation; frozen Validation-config hash enforcement; independent Feature/Regime/Signal/ML/Activation/Portfolio/Economic/Risk/Order/Fill invariance hashes; schema 1.9 checksum-verified analytical bundles; recalculated net economics and unchanged `PAPER_ESTIMATE_V1` ranges; read-only `PaperReadinessReviewV3`; explicit reason-required human-review audit; CLI, monitoring events, and Dashboard visibility. The consumed holdout has eight covered post-effective-date US sells and USD 0.99650280 of Section 31 cost, reducing net P&L before operating costs from USD 10,881.80249268745 to USD 10,880.80598988745. Decision invariance is `STRICTLY_INVARIANT`, all unchanged hard checks remain passing, completeness is `COMPLETE_ESTIMATED`, and Readiness V3 is `READY_FOR_REVIEW`; however the original Final Holdout V2 remains historically `FAIL / CONSUMED`, the prior evidence/readiness statuses remain unchanged, and Human Review remains `AWAITING_HUMAN_REVIEW`. No parameter was retuned and no broker, Paper session, order transmission, credential, or `LIVE` path was added.
