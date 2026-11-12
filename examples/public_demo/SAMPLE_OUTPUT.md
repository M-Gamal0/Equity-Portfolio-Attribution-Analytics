# Verified public-data demo output — 20-stock active portfolio

## November 12 sample evidence and source-only publication scope

Current accepted private build is `months_d47dc24f10c516ee57e6` at `local/analytics_outputs/month_windows.duckdb`; 5,995 selections and 47,960 saved validations passed. The 28 missing-CAGR intervals are restored, other result-value fingerprints match, and v1 remains at `month_windows_v1_backup_20261109.duckdb`. No new population or repair is queued. MIT is selected; DuckDB 2.x work remains deferred. Numerical/app evidence below retains its recorded dates; no numerical test or browser review was rerun for November 12 documentation maintenance.

The public 20-stock sample tables below remain the original frozen-acquisition results, not newly generated November 12 outputs. The public five-year values were unaffected by the twelve-month eligibility issue. Preserve the original sample/reproduction snapshot distinction and recorded software assumptions. The 90-step source publication plan includes this aggregate report and recipe, not provider observations, downloads or databases. Guard checks and ignore rules protect the selected paths; no sample numeric value or dataset was changed here.

## Sample results versus the publication plan

The numerical tables below remain the original verified 20-stock sample, not newly generated results. MIT is selected for code, DuckDB 2.x migration is deferred and public source will be committed in 90 Git Bash sections under [COMMIT_PLAN.md](../../docs/COMMIT_PLAN.md). Held historical records and generated datasets remain local; the recipe and current examples are included.

Do not relabel the earlier 91-test suite as a result for the new held-out selection, which excludes two legacy tests. The isolated reproduction snapshot and MIT artifact hashes remain separate evidence, not replacements for this sample's manifest. This synchronization changes no data, sample values, construction, code or Git state.

Updated November 6, 2026 using the original frozen public-source acquisition. No new market-data downloads were required. This replaces the earlier ten-stock sample; older snapshots/results remain preserved.

Current validated snapshot: `4b366f532e109d99`. Benchmark: 50 stocks at 2%; active: 20 stocks at 5%; both reset quarterly. Opening close: September 30, 2021; first return: October 1, 2021; cutoff: September 30, 2026.

## Evidence and replay scope

These are saved results from one frozen acquisition, not output regenerated for this documentation update. Current implementation evidence is `local/public_demo_20_stock_completion.json` and `local/public_demo_20_stock_wheel_validation.json`; these local evidence records are not automatically included in a public release. Preserve the cache, implementation and recorded software versions for exact offline replay. A fresh live fetch may change membership, caps, adjustments, coverage and factors.

The private research profile and its original charts/results remain unchanged. A tested local implementation is separate from the pending [public release checks](../../docs/RELEASE_READINESS.md). Nothing was deleted or moved; earlier snapshots remain provenance, as recorded in the [cleanup review](../../docs/CLEANUP_CANDIDATES.md).

## Input coverage

| Dataset | Rows |
| --- | ---: |
| `factor_returns` | 2,029 |
| `portfolio_holdings` | 87,780 |
| `portfolio_metadata` | 2 |
| `portfolio_returns` | 2,508 |
| `rebalance_targets` | 1,470 |
| `security_returns` | 99,390 |

The frozen Nasdaq universe contains 101 separate tickers; nine candidates fail complete portfolio-period price coverage. The active basket is the first 20 eligible names from the same ranking; GOOG and GOOGL remain separate. Sector count remains **4**, the same as the earlier ten-stock basket. No sector-balancing rule was added. Benchmark input returns, holdings and targets were compared with the earlier snapshot and remain unchanged.

## Existing analytics execution

| Domain | Status | Output tables |
| --- | --- | ---: |
| performance | COMPLETED | 21 |
| attribution | COMPLETED | 19 |
| risk | COMPLETED | 21 |
| exposures | COMPLETED | 18 |
| factors | COMPLETED | 14 |
| turnover | COMPLETED | 25 |

All six default domains completed (118 output tables). The fixed-view preparation also completed the full security-factor/lagged-holdings calculations and existing monthly validations. All eight Streamlit sections passed the separate public-data integration check; 19 charts and 133 detailed choices remain.

Each owner retains 19 measured recurring rebalances, one initial establishment and one pending September 30, 2026 selection. There are 1,470 target rows: 1,400 implemented and 70 pending. Pending measurements remain unavailable.

## Reconciliation

| Check | Result |
| --- | --- |
| Portfolio sessions | 1,254 |
| Factor coverage | 100%; no missing leading/internal/trailing dates |
| `max_portfolio_price_error` | 1.387779e-17 |
| `max_portfolio_total_error` | 2.775558e-17 |
| `max_bop_weight_error` | 3.330669e-16 |
| `max_eop_weight_error` | 3.330669e-16 |
| `max_price_drift_error` | 4.163336e-17 |
| `max_target_weight_error` | 0.000000e+00 |

Weight drift uses price returns; total returns follow the existing index-level dividend convention. Current membership/caps/sectors are deliberately applied retrospectively for demonstration. No original private datasets, calculation kernels or dashboard were changed.

## Reproduce

Follow [README.md](README.md). Frozen-cache `build --offline` creates the 20-stock snapshot without downloads; `prepare-app` saves one fixed 60-month view. Current full suite: **91 passed, one optional public-data integration skipped**; that integration passed separately. Downloaded data/results remain Git-ignored and excluded from the wheel.
