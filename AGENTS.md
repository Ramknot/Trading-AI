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
- Feature calculations must never depend on future bars.
- A feature value at time `t` must remain unchanged if future data is appended.
- Baseline strategy parameters must not be tuned solely to maximize historical performance.
- Strategies must use shared `FeatureEngine` definitions instead of reimplementing indicators independently.
- Mean Reversion may open new positions only in an eligible `RANGE` regime.
- Machine learning must be introduced only through its dedicated, governed ML lot and contracts.
- Strategy defaults are research baselines, not optimized parameters.
- The Risk Engine must never increase a requested position size.
- Risk-reducing exits must remain possible when new risk is halted.
- A missing critical risk input must never be interpreted as zero risk.
- Risk limits must not be optimized solely to improve historical performance.
- The default execution and backtest paths must remain fail-closed with `DenyAllRiskEngine`; `BalancedRiskEngine` requires explicit validated injection.
- `BalancedRiskEngine` must not unlock `LIVE` or aggressive profiles.
- Risk engines must not generate trading signals.
- Risk engines must not contact brokers or data providers.
- Circuit breakers must stop new risk, not silently liquidate portfolios.
- Every approved simulated order must reference an immutable `RiskDecision`.
- Pending orders must reserve their risk capacity so concurrent intents cannot reuse cash, exposure, or exit quantity.
- Market structure and volatility must be modeled as separate regime dimensions.
- Regime classification at time `t` must never depend on future bars.
- `UNKNOWN` must remain a valid conservative regime and must not be silently mapped to `RANGE`.
- `StrategyActivationPolicy` must never increase allocation above the strategy proposal.
- Strategy exits must not be blocked by regime policy.
- `BalancedRiskEngine` remains sovereign after regime filtering.
- Regime policy must not optimize or choose strategies based on historical profitability.
- Mean Reversion must not average down or use martingale sizing.
- The Lot 5 Regime Detector must remain deterministic and must not introduce machine learning.
- Regime modules must not contact data providers or brokers.
- Machine-learning models must score existing quantitative opportunities; they must not bypass strategy, regime policy, or risk controls.
- ML must never increase a strategy-proposed position size.
- `EXIT_LONG` signals must never be blocked by ML filtering.
- Training and inference must remain separate execution paths.
- Inference code must never fit, retrain, partially fit, or update a model.
- Model inputs at time `t` must never contain information from `t+1` or later.
- Future-derived values are permitted only inside explicit label construction.
- Temporal validation must never use random shuffling.
- Final test data must not participate in model fitting, preprocessing, threshold selection, feature selection, or model selection.
- Overlapping labels must be purged across temporal split boundaries.
- Models must never be promoted automatically based on historical profitability or classification metrics.
- `FILTER` mode requires an explicitly `APPROVED` model.
- No silent model fallback is permitted; artifact or schema failures must fail closed for new entries.
- ML architecture must depend on `ModelAdapter` contracts rather than directly on scikit-learn classes.
- Sequence, Transformer, multimodal, real-time, and online-learning capabilities remain PLANNED/LOCKED until Balanced validation.
- Update `PROJECT_STATE.md` when a lot changes state.
