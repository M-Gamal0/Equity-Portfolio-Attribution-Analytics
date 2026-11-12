"""Construct a separate quarterly equal-weight public-data demonstration."""
from dataclasses import asdict, dataclass
import json

import numpy as np
import pandas as pd

from portfolio_analytics.data import contracts
from portfolio_analytics.data.integration import validate_analytics_input_bundle
from .sources import FACTOR_COLUMNS

ACTIVE_ID = "demo_active20"
BENCHMARK_ID = "demo_benchmark50"


@dataclass(frozen=True)
class DemoConfig:
    inception: str = "2021-09-30"
    cutoff: str = "2026-09-30"
    history_start: str = "2018-09-01"
    benchmark_size: int = 50
    active_size: int = 20

    def __post_init__(self):
        dates = [pd.Timestamp(getattr(self, key)) for key in ("history_start", "inception", "cutoff")]
        if any(pd.isna(d) or d.tz is not None or d != d.normalize() for d in dates) or not dates[0] < dates[1] < dates[2]:
            raise ValueError("Require timezone-free history_start < inception < cutoff dates")
        if not dates[1].is_quarter_end or not dates[2].is_quarter_end:
            raise ValueError("Demo inception/cutoff must be calendar quarter-ends")
        if any(type(n) is not int for n in (self.benchmark_size, self.active_size)) or not 0 < self.active_size <= self.benchmark_size:
            raise ValueError("Require integer 0 < active_size <= benchmark_size")


def eligible_ranking(quotes, histories, calendar, config):
    """Missing public responses are audited exclusions, never fabricated inputs."""
    if quotes.ticker.duplicated().any():
        raise ValueError("Universe tickers must be unique; share classes stay separate")
    dates = calendar.index[(calendar.index >= config.inception) & (calendar.index <= config.cutoff)]
    if not len(dates) or dates[0] != pd.Timestamp(config.inception):
        raise ValueError("Reference calendar must contain the opening valuation date")
    rows = []
    for row in quotes.to_dict("records"):
        cap = row.get("marketCap")
        error = row.get("source_error")
        reason = error if isinstance(error, str) else ""
        if not reason and (isinstance(cap, bool) or not isinstance(cap, (int, float)) or not np.isfinite(cap) or cap <= 0):
            reason = "MISSING_OR_INVALID_CURRENT_MARKET_CAP"
        if not reason and row.get("currency") != "USD":
            reason = "NON_USD_OR_MISSING_QUOTE_CURRENCY"
        if not reason and (not isinstance(row.get("sector"), str) or not row["sector"].strip()):
            reason = "MISSING_CURRENT_SECTOR"
        price = histories.get(row["ticker"])
        if not reason:
            if price is None or not {"Close", "Adj Close"}.issubset(price):
                reason = "MISSING_PRICE_HISTORY"
            else:
                values = price.reindex(dates)[["Close", "Adj Close"]].to_numpy(dtype=float)
                if not np.isfinite(values).all() or (values <= 0).any():
                    reason = "INCOMPLETE_OR_INVALID_PORTFOLIO_PERIOD_PRICES"
        rows.append({**row, "eligible": not bool(reason), "exclusion_reason": str(reason)})
    audit = pd.DataFrame(rows)
    # Numeric sorting, deterministic tie-break; never use Nasdaq's supplied caps.
    audit["marketCap"] = pd.to_numeric(audit.marketCap, errors="coerce")
    audit = audit.sort_values(["marketCap", "ticker"], ascending=[False, True], na_position="last", kind="stable").reset_index(drop=True)
    valid = audit.loc[audit.eligible].copy()
    valid["eligible_rank"] = np.arange(1, len(valid) + 1)
    audit = audit.merge(valid[["ticker", "eligible_rank"]], on="ticker", how="left", validate="one_to_one")
    audit["selected_benchmark"] = audit.eligible_rank.le(config.benchmark_size)
    audit["selected_active"] = audit.eligible_rank.le(config.active_size)
    if len(valid) < config.benchmark_size:
        raise ValueError(f"Only {len(valid)} eligible securities; need {config.benchmark_size}. Inspect acquisition/exclusion audit.")
    selected = valid.head(config.benchmark_size).reset_index(drop=True)
    return selected, audit


def construct_inputs(quotes, histories, calendar, factors, config=DemoConfig()):
    """Return the six standard frames plus acquisition/selection evidence."""
    selected, audit = eligible_ranking(quotes, histories, calendar, config)
    reference = calendar.index[(calendar.index >= config.inception) & (calendar.index <= config.cutoff)]
    if reference[-1] != pd.Timestamp(config.cutoff):
        raise ValueError("Public reference data do not reach the requested quarter-end cutoff")
    dates = reference[1:]
    price = np.column_stack([histories[t].reindex(reference).Close.to_numpy() for t in selected.ticker])
    adjusted = np.column_stack([histories[t].reindex(reference)["Adj Close"].to_numpy() for t in selected.ticker])
    price_returns = price[1:] / price[:-1] - 1
    total_returns = adjusted[1:] / adjusted[:-1] - 1
    quarter_ends = pd.date_range(config.inception, config.cutoff, freq="QE-DEC")
    events = []
    for anchor in quarter_ends:
        later = reference[reference > anchor]
        effective = later[0] if len(later) else pd.NaT
        events.append((anchor, effective, reference[reference <= anchor][-1]))
    effective_dates = {event[1] for event in events if pd.notna(event[1])}
    returns, holdings, targets = [], [], []
    for owner, size in ((BENCHMARK_ID, config.benchmark_size), (ACTIVE_ID, config.active_size)):
        names = selected.head(size)
        bop, eop = np.empty((len(dates), size)), np.empty((len(dates), size))
        state = np.full(size, 1.0 / size)
        for i, date in enumerate(dates):
            if date in effective_dates:
                state = np.full(size, 1.0 / size)
            bop[i] = state
            grown = state * (1.0 + price_returns[i, :size])
            state = grown / grown.sum()
            eop[i] = state
        contribution_price = bop * price_returns[:, :size]
        contribution_total = bop * total_returns[:, :size]
        returns.append(pd.DataFrame({"portfolio_id": owner, "date": dates,
                                     "return_price": contribution_price.sum(axis=1),
                                     "return_total": contribution_total.sum(axis=1)}))
        holdings.append(pd.DataFrame({
            "portfolio_id": owner, "date": np.repeat(dates.to_numpy(), size),
            "security_id": np.tile(names.ticker.to_numpy(), len(dates)),
            "ticker": np.tile(names.ticker.to_numpy(), len(dates)),
            "sector": np.tile(names.sector.to_numpy(), len(dates)),
            "weight_bop": bop.ravel(), "weight_eop": eop.ravel(),
            "weight_eop_pre_liquidation": eop.ravel(),
            "return_price": price_returns[:, :size].ravel(), "return_total": total_returns[:, :size].ravel(),
            "contribution_price": contribution_price.ravel(), "contribution_total": contribution_total.ravel(),
        }))
        for anchor, effective, base in events:
            targets.append(pd.DataFrame({"portfolio_id": owner, "selection_date": anchor,
                                         "effective_rebalance_date": effective, "return_base_date": base,
                                         "security_id": names.ticker, "ticker": names.ticker,
                                         "sector": names.sector, "weight_target": 1.0 / size}))
    securities = []
    for ticker in selected.ticker:
        history = histories[ticker].reindex(calendar.index).loc[config.history_start:config.cutoff]
        # Keep discontinuities as missing and omit invalid intervals; never bridge them.
        data = history.pct_change(fill_method=None).rename(columns={"Close": "return_price", "Adj Close": "return_total"})
        data = data.replace([np.inf, -np.inf], np.nan).dropna().reset_index()
        data["security_id"], data["ticker"] = ticker, ticker
        securities.append(data[["security_id", "date", "return_price", "return_total", "ticker"]])
    metadata = []
    for owner, role, name in ((ACTIVE_ID, "active", f"Public demo: quarterly equal-weight top {config.active_size}"),
                              (BENCHMARK_ID, "benchmark", f"Public demo: quarterly equal-weight top {config.benchmark_size}")):
        metadata.append({"portfolio_id": owner, "display_name": name, "portfolio_role": role,
                         "benchmark_portfolio_id": BENCHMARK_ID if role == "active" else None,
                         "return_frequency": "daily", "return_units": "daily_simple_decimal",
                         "reporting_cutoff": pd.Timestamp(config.cutoff), "base_currency": "USD",
                         "methodology_version": "public_yahoo_equal_weight_quarterly_v1",
                         "methodology_json": json.dumps({**asdict(config), "rebalance": "quarterly_fixed_basket",
                             "weight_drift": "price_returns", "dividends": "index_level_total_returns",
                             "selection": "current_universe_current_caps_applied_retrospectively",
                             "share_classes": "separate", "sectors": "fixed_current_yahoo"}, sort_keys=True),
                         "source_system": "Yahoo Finance via yfinance; Nasdaq; Kenneth French",
                         "run_status": "demo", "publication_approved": False})
    frames = {
        "portfolio_returns": contracts.validate_portfolio_returns(pd.concat(returns, ignore_index=True)),
        "portfolio_holdings": contracts.validate_portfolio_holdings(pd.concat(holdings, ignore_index=True)),
        "security_returns": contracts.validate_security_returns(pd.concat(securities, ignore_index=True)),
        "rebalance_targets": contracts.validate_rebalance_targets(pd.concat(targets, ignore_index=True)),
        "factor_returns": contracts.validate_factor_returns(factors, FACTOR_COLUMNS),
        "portfolio_metadata": contracts.validate_portfolio_metadata(pd.DataFrame(metadata)),
    }
    validate_demo(frames, config)
    return frames, audit


def validate_demo(frames, config):
    report = validate_analytics_input_bundle(
        frames["portfolio_returns"], frames["portfolio_holdings"], frames["security_returns"],
        frames["rebalance_targets"], frames["factor_returns"], frames["portfolio_metadata"],
        FACTOR_COLUMNS, fully_invested=True, require_aligned_calendars=True)
    returns = frames["portfolio_returns"]
    observed_dates = set(returns.date)
    factors = frames["factor_returns"]
    factor_dates = set(factors.loc[factors.date.gt(config.inception) & factors.date.le(config.cutoff), "date"])
    if observed_dates != factor_dates or report["factor_coverage_ratio"] != 1.0:
        raise ValueError("Exact public price/factor calendar coverage is required; do not truncate or fill the requested period")
    holdings = frames["portfolio_holdings"]
    price = returns.set_index(["portfolio_id", "date"]).return_price
    actual = holdings.set_index(["portfolio_id", "date"])
    expected = holdings.weight_bop.to_numpy() * (1 + holdings.return_price.to_numpy()) / (1 + price.reindex(actual.index).to_numpy())
    error = float(np.max(np.abs(expected - holdings.weight_eop.to_numpy())))
    if error > 1e-10:
        raise ValueError("Price-drift weights fail reconciliation")
    for name in ("contribution_price", "contribution_total"):
        target = "return_price" if name.endswith("price") else "return_total"
        values = holdings.groupby(["portfolio_id", "date"])[name].sum()
        error_contribution = float((values - returns.set_index(["portfolio_id", "date"])[target]).abs().max())
        if error_contribution > 1e-10:
            raise ValueError(f"{name} fails reconciliation")
    report["max_price_drift_error"] = error
    report["status"] = "PASSED"
    return report
