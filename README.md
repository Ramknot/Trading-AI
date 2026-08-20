# Trading AI

Trading AI is a safety-first, modular Python foundation for research, paper trading, and future controlled execution. Lot 0 establishes the boundaries and contracts that later data, backtesting, strategy, risk, ML, portfolio, monitoring, and broker implementations must respect.

No financial strategy, broker integration, machine-learning model, or live order path is implemented in this lot.

## Lot 0 guarantees

- Python 3.11+ package with typed, mostly immutable business models.
- Separate `DEV`, `TEST`, `PAPER`, and `LIVE` environment policies.
- Active `balanced` configuration for research and paper workflows.
- Schema-compatible `aggressive` profile that is explicitly disabled and also locked in code.
- `LIVE` startup rejected with no Lot 0 override.
- Mandatory `RiskEngine` boundary in the execution chain.
- Default `DenyAllRiskEngine` that rejects every order request.
- Broker adapters accept only risk-approved order envelopes.
- Structured JSON logging fields suitable for a future monitoring stack.
- Diagnostic CLI and unit tests for the critical safety cases.

These controls are software architecture safeguards, not a claim that trading systems are risk-free.

## Structure

```text
config/profiles/          TOML trading profiles
src/trading_ai/core/      models, configuration, environment policy, health, logging
src/trading_ai/data/      DataEngine contract
src/trading_ai/strategies Strategy contract
src/trading_ai/regimes/   future regime detection boundary
src/trading_ai/ml/        MLScorer contract
src/trading_ai/portfolio/ PortfolioEngine contract
src/trading_ai/risk/      RiskEngine and fail-closed DenyAllRiskEngine
src/trading_ai/execution/ sealed risk-gated ExecutionEngine
src/trading_ai/brokers/   BrokerAdapter contract
src/trading_ai/backtesting/ Backtester contract
src/trading_ai/monitoring/ future monitoring boundary
tests/                    unit and architecture safety tests
```

Strategies and ML components only produce typed opinions. They do not receive a broker. An order proposal must enter `ExecutionEngine.submit_order`, which calls the configured `RiskEngine` before it can construct the envelope accepted by `BrokerAdapter`. The default risk engine rejects everything.

## Installation

From the repository root, using Python 3.11 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

Runtime code has no third-party dependency. `pytest` is the only Lot 0 development dependency.

Never put credentials in source files. Copy `.env.example` to an untracked `.env` only when a future lot needs local credentials, and inject production secrets through an approved external secret manager.

## Diagnostics

After installation:

```powershell
trading-ai doctor
trading-ai doctor --environment PAPER --profile balanced --json
```

Without installation, point Python at `src` or run the package from an editable environment.

Expected safety matrix:

| Environment | Profile | Result in Lot 0 |
| --- | --- | --- |
| PAPER | balanced | Configuration accepted; orders still denied |
| PAPER | aggressive | Blocked: aggressive profile disabled and code-locked |
| LIVE | balanced | Blocked: LIVE startup locked |

## Tests

```powershell
.\.venv\Scripts\python -m pytest
```

The suite covers profile loading, environment locks, immutable model validation, CLI health checks, structured logs, mandatory risk evaluation, live defense in depth, and simple architecture-bypass attempts.

## Profiles

`balanced.toml` is the default research/paper profile. Its fields prepare timeframes, asset universe, position limits, exposure, turnover, short policy, risk budget, and signal thresholds.

`aggressive.toml` uses the same schema and contains `enabled = false`. Lot 0 additionally rejects `aggressive` in Python regardless of the TOML flag, so editing configuration alone cannot activate it.

## Security boundaries

- Lot 0 exposes no setting, environment variable, or CLI flag that unlocks `LIVE`.
- The execution entry point cannot be overridden by subclasses and owns a mandatory `RiskEngine` instance.
- `DenyAllRiskEngine` is installed when no future risk engine is supplied.
- An order with a rejected or mismatched risk decision never reaches a broker adapter.
- Secrets, `.env` files, private keys, local data, logs, toolchains, and caches are ignored by Git.

## Roadmap

The planned sequence is Data Engine, Backtesting Engine, Baseline Quant, Risk Engine, Regime Detector, Machine Learning, Portfolio Engine, Dashboard, and Broker/Paper Trading. Aggressive research remains a locked future branch and must not be enabled until the balanced path is validated. See `PROJECT_STATE.md` for status.
