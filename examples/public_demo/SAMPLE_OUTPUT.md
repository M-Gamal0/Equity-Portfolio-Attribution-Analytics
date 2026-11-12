# Verified public-data demo output — 20-stock active portfolio

These are aggregate results from one verified build of the public demo, using a frozen download of the public sources. They show what a successful run looks like; no market data is included. A fresh download can select a different basket, because index membership, market caps, price adjustments and factor releases change over time.

Snapshot: `4b366f532e109d99`. Benchmark: 50 stocks at 2%; active portfolio: 20 stocks at 5%; both reset to equal weights quarterly. Opening close: September 30, 2021; first return: October 1, 2021; cutoff: September 30, 2026.

## Input coverage

| Dataset | Rows |
| --- | ---: |
| `factor_returns` | 2,029 |
| `portfolio_holdings` | 87,780 |
| `portfolio_metadata` | 2 |
| `portfolio_returns` | 2,508 |
| `rebalance_targets` | 1,470 |
| `security_returns` | 99,390 |

The frozen Nasdaq universe contains 101 separate tickers; nine candidates fail complete portfolio-period price coverage. The active basket is the first 20 eligible names from the same ranking; GOOG and GOOGL remain separate. The active basket covers **4** sectors; no sector-balancing rule is applied.

## Analytics execution

| Domain | Status | Output tables |
| --- | --- | ---: |
| performance | COMPLETED | 21 |
| attribution | COMPLETED | 19 |
| risk | COMPLETED | 21 |
| exposures | COMPLETED | 18 |
| factors | COMPLETED | 14 |
| turnover | COMPLETED | 25 |

All six default domains completed (118 output tables). The dashboard preparation also completed the full security-factor and lagged-holdings calculations and the monthly validations. All eight dashboard sections, with 19 charts and 133 detailed result choices, passed the public-data integration check.

Each portfolio has 19 measured recurring rebalances, one initial establishment and one pending September 30, 2026 selection. There are 1,470 target rows: 1,400 implemented and 70 pending. Pending measurements remain unavailable.

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

Weight drift uses price returns; total returns use index-level dividend reinvestment. Current membership, market caps and sectors are deliberately applied retrospectively for demonstration.

## Reproduce

Follow [README.md](README.md). With a frozen source cache, `build --offline` recreates the snapshot without downloads; `prepare-app` saves one fixed 60-month view for the dashboard. Downloaded data and results stay in the Git-ignored `demo_workspace/` and are not part of the package.
