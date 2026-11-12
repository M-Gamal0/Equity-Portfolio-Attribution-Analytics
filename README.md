# Equity Portfolio Attribution & Analytics

[![Public package checks](https://github.com/M-Gamal0/Equity-Portfolio-Attribution-Analytics/actions/workflows/public-checks.yml/badge.svg)](https://github.com/M-Gamal0/Equity-Portfolio-Attribution-Analytics/actions/workflows/public-checks.yml)

An end-to-end equity portfolio analytics framework in Python. It explains **what drove a portfolio's return relative to its benchmark, what risks it carried, and how its positioning changed**, using institutional methods: Brinson-Fachler attribution with Carino and Menchero linking, ex-post and holdings-based risk decomposition, Fama-French five-factor plus momentum regressions, Active Share and turnover analysis.

The project is a personal showcase of analytics, calculation and validation. A rules-based 20-stock active portfolio and a custom equal-weight Top-100 benchmark serve as the case study; any outperformance is an observed historical outcome, not the project's objective or evidence of skill.

## What it does

The package has six independent analytics domains. Each consumes standardized pandas DataFrames and explicit settings, can run on its own, and never depends on another domain's output.

| Domain | Main analytics |
| --- | --- |
| **Performance** | Cumulative and period returns, CAGR, calendar-year and rolling returns, relative wealth, up/down capture, Sharpe, Sortino, Information Ratio, security and sector contributions |
| **Attribution** | Daily Brinson-Fachler allocation, selection and interaction effects by sector; multi-period linking with Carino and Menchero; security-level active contributions |
| **Risk** | Volatility, downside deviation, tracking error, drawdowns and recovery, historical VaR and Expected Shortfall, covariance-based security and sector risk contributions |
| **Holdings & exposures** | Security and sector active weights, Active Share, overlap, concentration (HHI, effective number of holdings), top positions and exposure history |
| **Factors** | Fama-French 5 + momentum regressions for portfolio, benchmark and active returns; rolling three-year security betas; lagged holdings-based exposures; factor return attribution and risk; classical and Newey-West inference |
| **Turnover & rebalancing** | Scheduled rebalance turnover, purchases and sales, price-driven weight drift, trade concentration and transaction-cost sensitivity |

Every calculation reconciles its results: daily contributions sum to portfolio returns, Brinson effects sum to active return, and linked effects reconcile to the compounded return difference. Insufficient history, partial factor coverage and undefined ratios are reported explicitly, never filled with zeros.

## Two ways to see it

1. **Private research dashboard** (`app/streamlit_app.py`): an eight-page Streamlit app over the original case study, built on licensed historical S&P 500 data. It reads results precomputed offline for every complete-month window of 12 to 120 months between October 2016 and September 2026 (5,995 windows). The data and databases are private and not included in this repository.
2. **Public demo** (`app/public_demo_app.py`): the same analytics and dashboard pages on freely available data from Nasdaq, Yahoo Finance and the Kenneth French Data Library, which you download yourself. This makes the project fully reproducible without private data.

## Quick start: public demo

Requires Python 3.13 or later. From a clone of this repository:

```bash
python -m pip install -e ".[demo,app]"
portfolio-analytics-demo build --workdir demo_workspace
portfolio-analytics-demo prepare-app --workdir demo_workspace
python -m streamlit run app/public_demo_app.py --server.port 8503
```

`build` downloads the public data, constructs both portfolios and validates the inputs. `prepare-app` calculates one fixed five-year view (October 2021 to September 2026). The dashboard then only reads saved results. To run the analytics without the dashboard, use `portfolio-analytics-demo run --workdir demo_workspace`.

**Demo construction:** a 50-stock benchmark (2% each) and a 20-stock active portfolio (5% each), both chosen from today's Nasdaq-100 members ranked by today's market capitalization and rebalanced to equal weights quarterly. Because current members and market caps are applied to past dates, the demo has survivorship and look-ahead bias. It is a reproducible software test case, not an investment backtest. See the [public demo guide](examples/public_demo/README.md) for construction rules, offline replay and troubleshooting.

## Use it with your own data

The analytics run on any portfolio supplied as six standardized DataFrames: portfolio returns, holdings, security returns, rebalance targets, factor returns and portfolio metadata. No private data, Yahoo dependency or dashboard is needed.

```bash
python -m pip install .
python examples/custom_inputs/run_example.py
```

The [Python API guide](docs/PYTHON_API_GUIDE.md) documents the input contracts, units, required fields, domain calls and outputs. The [minimal example](examples/custom_inputs/run_example.py) validates a small synthetic return series and computes wealth paths.

```python
from portfolio_analytics import performance

outputs = performance.run(
    portfolio_returns=returns_df,
    portfolio_holdings=holdings_df,
    factor_returns=factors_df,
    settings=settings,
)
summary = outputs["performance_window"]
```

Each domain's `run` returns a dictionary of named DataFrames.

## Command-line tools

| Command | Purpose |
| --- | --- |
| `portfolio-analytics list` | List the six domains and their required inputs |
| `portfolio-analytics run <domains>` | Run one domain, a subset or all, with explicit inputs |
| `portfolio-analytics precompute-months` | Precompute every complete-month window offline |
| `portfolio-analytics month-result` | Read one precomputed window without recalculating |
| `portfolio-analytics-demo build / run / prepare-app` | Build and run the public demo |

The `--research` option selects the private case study and requires the private input snapshot, which is not part of this repository. To analyse another portfolio, supply `--snapshot-root`, `--snapshot-id`, `--portfolio-id`, `--benchmark-id`, `--as-of` and `--inception-date` explicitly. See `portfolio-analytics run --help`.

## Methodology and conventions

- Returns and weights are decimal fractions. Daily total return is the primary convention; price returns are kept for dividend analysis.
- Contributions use beginning-of-period weights. Weights drift with price returns between rebalances, and dividends are reinvested at index level.
- Quarter-end rebalancing applies new weights to the next session's full close-to-close return. This is a **hypothetical, frictionless same-close implementation**, not an executable trading result.
- Multi-period attribution reconciles to the compounded portfolio return minus the compounded benchmark return.
- Security factor betas use a full three-year daily window and are lagged, so each day's attribution uses only earlier information.
- Complete-month windows open at the last close of the preceding month, so the first selected day's return is included. Windows of 12 or more complete months are eligible for CAGR.

**Research assumptions:** USD reporting without FX conversion; sector labels held fixed across history rather than point-in-time; delisted securities exit at their last recorded close with proceeds reinvested proportionally.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/portfolio_analytics/` | The package: six analytics domains, input contracts, offline pipeline, result storage and CLI |
| `src/portfolio_analytics/demo/` | Public-data acquisition, portfolio construction and snapshot tools |
| `app/` | Streamlit dashboards, page helpers, local chart renderer, bundled fonts and chart libraries |
| `examples/` | Public demo guide and a minimal own-data example |
| `tests/` | Unit, financial-identity, storage and dashboard tests on synthetic data |
| `scripts/` | Maintenance utilities used when building the private case study |
| `docs/` | Python API guide and calculation provenance |

## Development

```bash
python -m pip install -e ".[demo,app]" pytest ruff
python -m ruff check src app scripts tests examples
python -m pytest -q -m "not research and not public_demo"
```

Tests use synthetic data and need no downloads or private files. Tests marked `research` or `public_demo` are opt-in and need the private snapshot or a prepared demo workspace. GitHub Actions runs lint, tests, the own-data example and a package build on Windows and Linux.

## Licence and data

Original code is released under the [MIT License](LICENSE). Bundled fonts and chart libraries keep their own licences (`app/static/fonts/`, `app/vendor/`). Market data is not covered by this licence and is not included: the private case study's data and results stay private, and public-demo users download data themselves under each provider's terms.
