# Using the Python package with your own data

## November 12 API and source-publication checkpoint

Current accepted private build is `months_d47dc24f10c516ee57e6` at `local/analytics_outputs/month_windows.duckdb`; 5,995 selections and 47,960 saved validations passed. The 28 missing-CAGR intervals are restored, other result-value fingerprints match, and v1 remains at `month_windows_v1_backup_20261109.duckdb`. No new population or repair is queued. MIT is selected; DuckDB 2.x work remains deferred. Numerical/app evidence below retains its recorded dates; no numerical test or browser review was rerun for November 12 documentation maintenance.

The contracts, domain APIs and examples below remain unchanged. Source distribution uses MIT while data-source terms remain separate. Inline synthetic fixtures are selected; real provider responses/snapshots/databases are not. The current commit plan's path/format guards protect staging and outgoing history, including unreviewed JSON/TXT extracts. No new public helper module, API change or dependency change was needed. Build the final corrected artifact from the selected committed tree; old wheels retain their build-specific scope.

## Complete-month adapter versus general-purpose CAGR API

The complete-month pipeline now applies an explicit >=12-month eligibility policy, relaxing only its effective kernel day gate when trading endpoints span fewer than 365 days. It does not change the annualization exponent: `365.25 / actual_days`. The original `compute_absolute_performance` and relative kernel defaults still require 365 days for arbitrary caller-supplied windows; genuinely short periods are not globally annualized.

The month outputs save policy metadata and the monthly validator checks metrics/provenance/formulas. The new tests pass, including 363/364-day years, shorter/invalid selections, unavailable states and unchanged 13/60-month metrics. The owner-controlled private refresh is separate from API use and the fixed public 60-month values remain unchanged. Older MIT wheel artifacts do not yet include this fix; use current source pending the final release build.

## Publication and validation checkpoint

API contracts and mathematical behavior below are unchanged. Original code uses MIT; provider datasets and bundled third-party notices remain separate. DuckDB 2.x work is deferred; retain the tested 1.x requirement.

[COMMIT_PLAN.md](COMMIT_PLAN.md) publishes the reusable modules and examples through 90 Git Bash section commits, with 12 visible historical/legacy holds and no generated/private data. Required validators, current financial tests and AST provenance remain included. The prior 91-test result included two held legacy tests; final committed-tree checks must record their actual count. No API/code, data, package environment or Git history was changed during this progress-record synchronization.

The same analytics package serves the private research case study, the Yahoo public demonstration and caller-supplied DataFrames. No research notebook or private snapshot is required by the mathematical functions or domain APIs. Python 3.13+ is required; install from this checkout with `python -m pip install .`. The optional `demo` extra adds Yahoo acquisition; the `app` extra adds Streamlit. The supported DuckDB dependency is `>=1.5.6,<2`: a clean installation with 2.0 failed existing storage tests on November 9, 2026.

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

Current constituents/caps applied retrospectively are an accepted demo convention, not a point-in-time investment backtest. Exact replay requires the frozen source cache, implementation and recorded software versions; live refresh may differ. Market-data terms and bundled third-party notices are separate from the [MIT licence](../LICENSE) for this project's original code. See [release readiness](RELEASE_READINESS.md) for remaining checks.
