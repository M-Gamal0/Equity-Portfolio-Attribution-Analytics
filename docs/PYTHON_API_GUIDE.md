# Using the Python package with your own data

The same analytics package serves the private research case study, the public-data demo and your own DataFrames. No research notebook or private snapshot is required by the mathematical functions or domain APIs. Python 3.13+ is required; install from the repository with `python -m pip install .`. The optional `demo` extra adds Yahoo acquisition; the `app` extra adds Streamlit. The supported DuckDB range is `>=1.5.6,<2`, because DuckDB 2.0 fails the package's storage tests.

**CAGR eligibility:** the general-purpose `compute_absolute_performance` and relative-performance functions annualize only windows of at least 365 calendar days, using the exponent `365.25 / actual_days`. The complete-month pipeline additionally treats any window of 12 or more complete months as eligible, because trading-day endpoints can span 363 or 364 days for a full year. The exponent is unchanged.

## Smallest working example

From the checkout, run `python examples/custom_inputs/run_example.py`. The [example](../examples/custom_inputs/run_example.py) validates eight synthetic return rows for two series and compounds wealth starting at 100. It does not use Yahoo, private defaults, holdings, factors or persistence. Its final wealth values are approximately 101.400862 for the illustrative portfolio and 101.201686 for the benchmark. Four observations illustrate the API; they are not sufficient for the full risk/factor analysis.

Low-level functions such as `performance.compute_wealth_index` accept the data needed for that calculation. Domain-level `run(...)` functions orchestrate a larger set of related calculations and need the input bundles below. The [public demo Python example](../examples/public_demo/run_example.py) shows a complete Performance domain call and the six-domain saved-snapshot workflow.

## Six input contracts

The `_df` suffix is a caller naming convention; loader keys and keyword arguments use the names below. Contracts are implemented in `src/portfolio_analytics/data/contracts.py`.

| Input | Required fields | Unique observation key |
| --- | --- | --- |
| `portfolio_returns` | `portfolio_id`, `date`, `return_total` | `portfolio_id`, `date` |
| `portfolio_holdings` | `portfolio_id`, `date`, `security_id`, `weight_bop`, `weight_eop`, `return_price`, `return_total` | `portfolio_id`, `date`, `security_id` |
| `security_returns` | `security_id`, `date`, `return_price`, `return_total` | `security_id`, `date` |
| `rebalance_targets` | `portfolio_id`, `selection_date`, `security_id`, `weight_target` | `portfolio_id`, `selection_date`, `security_id` |
| `factor_returns` | `date`, `rf`, and each name supplied through `factor_columns` | `date` |
| `portfolio_metadata` | `portfolio_id`, `return_frequency`, `return_units` | `portfolio_id` |

- Returns, weights, contributions, factors and RF use decimal fractions: 1% is `0.01`. Returns are daily simple returns, not log returns. Metadata values are `daily` and `daily_simple_decimal`. Dates are daily interval end dates, without intraday/time-zone information.
- IDs are caller-defined stable strings; no provider ID is required. Keep separate share classes distinct and the same security identity consistent across holdings, returns and targets. Tickers are optional labels. `sector` is optional in the universal holdings schema but required by sector-dependent domain calculations; provide it for the full six-domain workflow.
- Holdings may also contain `ticker`, `sector`, `weight_eop_pre_liquidation`, `contribution_total` and `contribution_price`. Contributions use beginning weights. Price-only returns drive drift; total returns drive performance. Do not substitute total-return drift or renormalize missing modeled exposure.
- Optional portfolio `return_price` supports price-return reconciliation. When absent, that check is explicitly not assessed. Target `effective_rebalance_date` and `return_base_date` support timing/implementation checks; an absent effective date does not by itself imply every target is pending. Optional target labels are `ticker`/`sector`.
- Validate with the six `portfolio_analytics.data.validate_*` functions. `validate_factor_returns` also requires the factor-column tuple. For a full bundle, `validate_analytics_input_bundle(returns_df, holdings_df, security_df, targets_df, factors_df, metadata_df, factor_columns, ...)` validates cross-frame financial identities. A structural validator alone does not establish financial reconciliation.

## Domain calls and outputs

Use `from portfolio_analytics.config import AnalysisSettings` and supply your own distinct `portfolio_id`, `benchmark_id`, `as_of_date` and preceding opening-close `inception_date`. Do not use `research_settings()` for external data. Choose reporting windows explicitly; the default compatibility presets are not a promise of sufficient history. Reporting starts do not reset rolling estimation histories.

Import a domain using `from portfolio_analytics import performance` (or the other names below), then call `domain.run(settings=settings, **required_frames)`.

| Domain | Required input keyword arguments | Representative returned DataFrame |
| --- | --- | --- |
| `performance` | `portfolio_returns`, `portfolio_holdings`, `factor_returns` | `performance_window` |
| `attribution` | `portfolio_returns`, `portfolio_holdings` | `brinson_sector_period` |
| `risk` | `portfolio_returns`, `portfolio_holdings`, `security_returns` | `risk_drawdown_daily` |
| `exposures` | `portfolio_holdings` | `active_share` |
| `factors` | `portfolio_returns`, `factor_returns` | `direct_portfolio_factor_coefficients` |
| `turnover` | `portfolio_holdings`, `rebalance_targets` | `rebalance_turnover_events` |

Each call returns a dictionary of named DataFrames. Inspect `outputs.keys()` and each table's columns; tables differ in grain, scope and units. Contributions/effects returned as decimal returns become percentage points only when multiplied by 100 for presentation. Do not infer units from a rounded chart or combine rows from different estimators. Preserve status/coverage/diagnostic tables alongside numerical tables.

The factors domain accepts additional security/holdings histories for optional security modeling. Full three-year history and observation gates remain required; `include_security_factor_history=True` opts into the expensive path. A default direct portfolio regression does not imply security models were run. Risk covariance uses its separate 252-session lookback. Genuine insufficient-history, sample-size, undefined-ratio and partial-coverage results remain explicit, not zero-filled.

## Storage, sources and examples

Direct domain calls do not save files or start Streamlit. The runner/snapshot/result-writer APIs handle explicit persistence separately. For the complete public workflow, follow [the public-demo setup](../examples/public_demo/README.md): `build`, `run`, and optionally `prepare-app`. The latter prepares one saved fixed five-year view; app interactions only read it. Generated datasets and database files are not bundled for publication.

Current constituents/caps applied retrospectively are an accepted demo convention, not a point-in-time investment backtest. Exact replay requires the frozen source cache, implementation and recorded software versions; live refresh may differ. Market-data terms and bundled third-party notices are separate from the [MIT licence](../LICENSE) for this project's original code.
