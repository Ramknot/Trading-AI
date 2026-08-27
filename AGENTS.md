# Trading AI Agent Rules

These rules apply to Codex and every future development agent working in this repository.

## Non-negotiable safety rules

- Never enable live trading without an explicit user request and a dedicated, reviewed implementation lot.
- Never bypass, weaken, make optional, or route around the `RiskEngine`.
- Never store secrets, broker credentials, tokens, private keys, or real account identifiers in the repository.
- Use environment variables or an approved external secret manager for all secrets.
- Never increase risk limits merely to improve reported or backtested performance.
- Keep aggressive research disabled until the balanced branch has been validated and activation is explicitly approved.
- A strategy, ML component, or portfolio component must never call a broker directly.
- All broker integration must implement `BrokerAdapter` and be reached through the guarded execution chain.

## Engineering rules

- Add or update tests for every important behavior change; all tests must pass before publication.
- Keep components independent, typed, documented, and testable through their public interfaces.
- Preserve compatibility with both the `balanced` and `aggressive` profile schemas, even while aggressive remains locked.
- Do not change the central architecture without documenting the reason and migration impact.
- Prefer the Python standard library and add dependencies only when they clearly reduce project risk or complexity.
- Keep generated files, caches, local datasets, logs, virtual environments, and credentials out of version control.
- Market universes must come from configuration and must not be hard-coded inside strategies, data engines, backtesters, portfolio engines, or risk engines.
- Every pull request and important modification must leave the continuous-integration workflow green.
- Never silently repair invalid market data; reject it or report it explicitly.
- Never mix raw and adjusted prices without explicit metadata.
- All normalized market-data timestamps must be timezone-aware.
- Network access must not be required by the default test suite.
- Provider-specific objects must not leak outside `DataProvider` adapters.
- Derived datasets must record the exact source-dataset lineage.
- Backtests must never access future market data through normal strategy APIs.
- Backtests must never download data implicitly.
- Backtest results must reference the exact datasets and checksums used.
- Transaction costs must never be silently omitted from a backtest configuration or result.
- Raw prices and corporate actions must not be double counted.
- Balanced backtests must not allow leverage or short selling.
- A backtest result is not evidence of future profitability.
- Update `PROJECT_STATE.md` when a lot changes state.
