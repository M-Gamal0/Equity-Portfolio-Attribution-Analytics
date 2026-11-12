# Try the analytics package with public market data

This guide builds a fully reproducible example of the analytics package using free public data: the current Nasdaq-100 list from Nasdaq, prices and market caps from Yahoo Finance, and daily Fama-French factors from the Kenneth French Data Library. You download the data yourself; no private files, research notebooks or databases are needed.

The result is a **real-public-data-derived demonstration**, not a synthetic dataset and not an investment backtest. Small synthetic fixtures are used only in the automated tests. [SAMPLE_OUTPUT.md](SAMPLE_OUTPUT.md) shows the counts and reconciliation results from one verified build; a fresh download can select a different basket because constituents and market caps change.

## Install

Python **3.13+** is required. From a clone of the repository, create and activate a virtual environment, then:

```bash
python -m pip install -e ".[demo]"
```

The `demo` extra adds `yfinance`; the base package can still analyse your own standardized DataFrames without it. To build a wheel instead, run `python -m pip install build` and `python -m build`, then install the wheel from `dist/`. The supported DuckDB range is `>=1.5.6,<2`, because DuckDB 2.0 fails the package's storage tests. For the input contracts and an own-DataFrame example, see the [Python API guide](../../docs/PYTHON_API_GUIDE.md). Original code is covered by the [MIT licence](../../LICENSE); data-source terms and third-party notices remain separate.

## Build and run

From any working directory:

```bash
portfolio-analytics-demo build --workdir demo_workspace
portfolio-analytics-demo run --workdir demo_workspace
```

Equivalent module commands (useful if the environment's Scripts/bin directory is not on PATH):

```bash
python -m portfolio_analytics.demo.cli build --workdir demo_workspace
python -m portfolio_analytics.demo.cli run --workdir demo_workspace
```

The first command retrieves the current Nasdaq-100 security list, current Yahoo market caps/sectors, Yahoo histories and French factors; validates the six input contracts and financial identities; and saves six versioned Parquet files plus `manifest.json`. The second executes all six existing domains and writes named result tables and `run_summary.json` under the **demo workspace only**. A failed domain gives a nonzero exit code and a recorded explanation, not a successful-looking partial demo.

To run selected domains or the additional rolling security-factor/lagged-holdings calculations:

```bash
portfolio-analytics-demo run --workdir demo_workspace --domains performance attribution risk
portfolio-analytics-demo run --workdir demo_workspace --domains factors --security-factor-history
```

For Python DataFrame usage, see [run_example.py](run_example.py).

## Fixed five-year Streamlit viewer (optional)

The repository also includes a separate, simple viewer for **October 2021–September 2026**, anchored at the September 30, 2021 close. It has no date selector. It reuses the existing presentation helpers, month adapters, read-only readers and financial validators; it does not open the private research databases or change the original app.

From the source checkout, install the app extra and prepare the view once:

```bash
python -m pip install -e ".[demo,app]"
portfolio-analytics-demo build --workdir demo_workspace
portfolio-analytics-demo prepare-app --workdir demo_workspace
python -m streamlit run app/public_demo_app.py --server.port 8503
```

For the dashboard path, a separate `run` command is not required before `prepare-app`. Preparation uses the existing full-history runner (including security factors), `MonthCalculationContext`, `validate_month_outputs` and `MonthResultWriter` to save **one** validated 60-month interval. It does not enumerate or populate 5,995 selections. All generated files are under `demo_workspace/streamlit/`. Repeating preparation with the same source snapshot and package fingerprint reuses the completed view.

The UI offers Overview, the six analytics sections and About. It has 19 charts and retains all 133 detailed-result choices, with ticker identities, saved eligibility/reconciliation information, fullscreen and no exports. Data acquisition, models and financial calculations never run on Streamlit reruns. The original app remains `app/streamlit_app.py`. Port 8503 is only an example; if it is occupied, choose another free port such as 8504 rather than stopping an unrelated process.

The viewer and its shared visual assets are repository examples, not bundled in the runtime wheel. Wheel-only users can still prepare/run analytics through the installed command; download the source checkout to run this Streamlit entry point. To use another generated workspace, set `PORTFOLIO_DEMO_WORKDIR` to its absolute path before launching.

Optional saved-public-data UI integration test (after preparation): set `PORTFOLIO_DEMO_TEST_WORKDIR` to the workspace's absolute path and run `python -m pytest tests/test_streamlit_public_demo.py -m public_demo -q`. Portable tests do not require downloads or this environment variable.

## Construction rules

| Choice | Rule |
| --- | --- |
| Universe | Current Nasdaq-100 securities from Nasdaq's public list endpoint; separate share classes are preserved. |
| Ranking | Current positive USD Yahoo `marketCap`, descending, with ticker as tie-break. These are Yahoo-reported caps, not reconstructed share-class-specific capitalizations. |
| Eligibility | Available current cap/currency/sector and finite positive Close/Adj Close on the opening date and **every** reference portfolio-period trading session. No missing-price filling. |
| Benchmark | First 50 eligible securities; 2% target per security. |
| Active | First 20 from the same eligible ranking; 5% target per security. GOOG and GOOGL can both be selected. |
| Rebalancing | The selected baskets stay fixed; reset to equal weights quarterly. This is **not** historical quarterly market-cap re-ranking. |
| Opening valuation | September 30, 2021 close; first earned return October 1, 2021. |
| Cutoff | September 30, 2026. The selection on that date is pending beyond the return cutoff. |
| Warm-up download | September 1, 2018 onward, allowing three-year rolling security-model history where available. Young listings retain existing unavailable-model statuses. |
| Weight drift | Split-adjusted price returns between rebalances, using the existing package convention. |
| Total returns | Dividend-adjusted return contributions with index-level dividend reinvestment. This is not a separate cash ledger or per-security dividend-reinvestment account. |
| Timing | Quarter-end targets enter BOP holdings on the first following reference session and earn that session's full previous-close return. Hypothetical frictionless implementation. |
| Sectors/calendar | Fixed current Yahoo sectors; Yahoo SPY sessions as the reference calendar, checked against French factor coverage. |
| Factors | Daily FF5 plus momentum and RF from Kenneth French, converted from percentages to daily decimals once. |

The current universe/caps are intentionally used retrospectively, as a package demonstration. Results are not a point-in-time Nasdaq index, the original private research portfolio, a historical cap-ranking backtest or an investment recommendation. Data coverage, not performance, determines eligibility.

## Six standardized inputs

The existing loader uses names without `_df`; these correspond directly to the six standardized DataFrame contracts:

| DataFrame | Parquet / loader key |
| --- | --- |
| `portfolio_returns_df` | `portfolio_returns.parquet` / `portfolio_returns` |
| `portfolio_holdings_df` | `portfolio_holdings.parquet` / `portfolio_holdings` |
| `security_returns_df` | `security_returns.parquet` / `security_returns` |
| `rebalance_targets_df` | `rebalance_targets.parquet` / `rebalance_targets` |
| `factor_returns_df` | `factor_returns.parquet` / `factor_returns` |
| `portfolio_metadata_df` | `portfolio_metadata.parquet` / `portfolio_metadata` |

Demo identities are `demo_active20` and `demo_benchmark50`; security IDs are public ticker strings, **not Sharadar permatickers**. `weight_eop_pre_liquidation` equals `weight_eop` because selected securities require uninterrupted portfolio-period coverage and the demo performs no terminal liquidation. The optional field lets the existing turnover diagnostics assess that convention explicitly.

## Reproduce a frozen acquisition

```bash
portfolio-analytics-demo build --workdir demo_workspace --offline
```

This verifies and replays cached inputs without network access. An unchanged cache, implementation and software versions produce the same content-derived snapshot ID. Snapshots are immutable; `latest.json` moves only after exact verified Parquet readback. The original date-compatible acquisition cache is reusable when changing only basket size; expanding the active portfolio from 10 to 20 required no downloads. To obtain a fresh current universe/cap ranking, use `build --refresh`; this creates another source-cache directory and preserves earlier caches/snapshots. A fresh fetch can differ because constituents, caps, classifications, adjustments and factor releases change.

The manifest records construction, source retrieval times/hashes, dataset SHA-256/row counts/dtypes/primary keys, selection-audit hash, software versions and validation results. `selection_audit.json` records exclusions and selected ranks. The public recipe is reproducible from public sources; exact historical bytes require keeping the frozen cache and recorded software versions.

## Tests and sample output

Run the tests from the source checkout; test files are not included in the runtime wheel.

```bash
python -m pip install pytest
python -m pytest tests/test_public_demo.py -q
```

Tests use deterministic, clearly synthetic data and no network: share classes/rank ties, missing histories, quarterly BOP resets/drift, final pending targets, factor units/sentinels, corruption detection, immutable snapshot readback, offline caching and all-six-domain execution. The full rolling-model test fixture spans more than three years so the existing 700-observation/three-year rules remain unchanged.

See [SAMPLE_OUTPUT.md](SAMPLE_OUTPUT.md) for actual run counts and reconciliation evidence from the verified public acquisition. Shorter windows and young security histories retain the package's original eligibility restrictions; a successfully built input bundle does not make every estimator available.

## Source access and distribution

The public repository includes the original analytics and app code plus this separate public-data recipe. It does not include private Sharadar files, original saved results or prebundled Yahoo market data. Users run the loader to obtain example data on their own machines. The original research viewer needs its private saved results; `app/public_demo_app.py` uses only the generated public demo workspace.

- [Nasdaq current security list](https://api.nasdaq.com/api/quote/list-type/nasdaq100)
- [yfinance documentation](https://ranaroussi.github.io/yfinance/) and [project/data-use notice](https://pypi.org/project/yfinance/)
- [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)

The distributable package contains the recipe, not downloaded Yahoo prices, French archives or generated Parquet. Each user obtains data directly from the sources under their applicable terms. Free access and the yfinance software licence do not grant a separate licence to redistribute provider data. The default generated `demo_workspace/` is Git-ignored; source data are never relabelled as freely licensed synthetic observations.

## Troubleshooting and replay

| Situation | What to do |
| --- | --- |
| `--offline` reports a missing cache | Build online once, or restore your own complete verified source cache. Offline replay cannot invent missing responses. |
| Yahoo/Nasdaq is unavailable or rate-limited | Retry later. If a failed response was frozen as an exclusion, use `build --refresh` to acquire a separate cache. Keep old caches/snapshots rather than editing their contents. |
| Too few eligible names or incomplete price/calendar coverage | Read the exclusions and source failure messages. The fixed example requires 50 eligible securities and exact portfolio-period history. Do not fill missing prices or replace caps to force completion. |
| Hash/provenance mismatch | Restore a known verified cache or acquire a new one using `--refresh`. Do not edit hashes to make a changed file appear valid. |
| Factor coverage fails | Confirm the downloaded factor release covers every included return date. Missing factors/RF are not zeros; incomplete coverage is not a validated input bundle. |
| A refreshed run differs | Current constituents/caps/sectors, adjusted histories and factor releases can change. Keep the frozen cache plus implementation/software records for exact replay; a fresh live fetch is not a byte-identical guarantee. |
| New software gives a different snapshot ID | Software versions and Parquet serialization may change content hashes even when construction and reconciled observations agree. Preserve both manifests; do not overwrite earlier snapshots or require the example's saved ID as your own output. |
| Port 8503 is occupied | Choose another free port, e.g. `--server.port 8504`. No financial rerun is needed to change a server port. |

The default tests use synthetic fixtures and no market-data downloads. The optional prepared-public-data integration checks the real-data viewer separately and needs a prepared workspace; GitHub Actions runs the portable tests on every push.
