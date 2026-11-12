"""Period adapters over validated kernels and immutable shared daily histories.

Daily/rolling histories belong to the pinned source run. Only genuinely
interval-dependent tables are recalculated here; closing-risk tables are cached
once per ending month, using all prior observations rather than the report slice.
"""

from dataclasses import replace
import json
from pathlib import Path
import time

import duckdb
import numpy as np
import pandas as pd

from portfolio_analytics import performance, attribution, risk, exposures, factors, turnover
from portfolio_analytics.common.selection import bind
from portfolio_analytics.common.month_windows import MonthWindow
from portfolio_analytics.data import load_standardized_inputs, validate_analytics_input_bundle
from portfolio_analytics.database.results import qualified_result
from portfolio_analytics.factors.api import RESPONSE_MAPPING
from portfolio_analytics.factors.period_regressions import estimate_direct_periods
from portfolio_analytics.pipeline.runner import DOMAIN_INPUTS

SOURCE_RUN_ID = "0ddde9043b9b4189b235102471de3d8f"

# Large rolling/security histories stay in the source database. The catalog
# records references to ALL source outputs, not just these calculation inputs.
NEEDED = {
    "performance": ("performance_security_contributions_daily", "performance_rolling_relative"),
    "attribution": ("brinson_sector_daily", "security_active_contributions_daily"),
    "exposures": ("security_concentration", "sector_concentration", "top_k_concentration",
                  "active_share", "holdings_overlap", "active_sector_exposures"),
    "factors": ("portfolio_factor_attribution",),
    "turnover": ("rebalance_turnover_events", "daily_rebalancing_diagnostics",
                 "rebalance_security_changes", "rebalance_sector_turnover"),
}


class MonthCalculationContext:
    """Load/validate once; reuse source histories for independent period adapters."""

    def __init__(self, *, snapshot_root, snapshot_id, settings, source_database,
                 source_run_id=SOURCE_RUN_ID):
        self.settings = replace(settings, include_security_factor_history=True)
        self.snapshot_id = snapshot_id
        self.source_database = Path(source_database).resolve()
        self.source_run_id = source_run_id
        self.inputs = load_standardized_inputs(snapshot_root, snapshot_id=snapshot_id, verify=True)
        self.input_validation = validate_analytics_input_bundle(
            self.inputs["portfolio_returns"], self.inputs["portfolio_holdings"],
            self.inputs["security_returns"], self.inputs["rebalance_targets"],
            self.inputs["factor_returns"], self.inputs["portfolio_metadata"],
            settings.factor_columns, fully_invested=settings.fully_invested,
            require_aligned_calendars=True)
        import hashlib
        self.manifest_sha256 = hashlib.sha256(
            (Path(snapshot_root) / snapshot_id / "manifest.json").read_bytes()).hexdigest()
        self.shared = {}
        with duckdb.connect(str(self.source_database), read_only=True) as db:
            run = db.execute("SELECT * FROM control.analytics_runs WHERE run_id=?", [source_run_id]).df()
            if len(run) != 1 or run.iloc[0].status != "COMPLETED":
                raise ValueError("An explicit completed full-history source run is required.")
            row = run.iloc[0]
            if row.snapshot_id != snapshot_id or row.snapshot_manifest_sha256 != self.manifest_sha256:
                raise ValueError("Shared histories must use the exact verified input snapshot.")
            if json.loads(row.config_json) != json.loads(self.settings.to_json()):
                raise ValueError("Shared-history settings differ from the requested methodology.")
            self.source_fingerprint = row.source_fingerprint
            self.catalog = db.execute("SELECT * FROM control.analytics_outputs WHERE run_id=?",
                                      [source_run_id]).df()
            for domain, names in NEEDED.items():
                for name in names:
                    match = self.catalog.loc[(self.catalog.domain == domain) & (self.catalog.output_name == name)]
                    if len(match) != 1:
                        raise ValueError(f"Missing shared history: {domain}/{name}")
                    table = qualified_result(db, match.iloc[0].table_name)
                    frame = db.execute(f"SELECT * FROM {table}").df()
                    if len(frame) != match.iloc[0].row_count:
                        raise ValueError(f"Shared-history row count mismatch: {name}")
                    self.shared[domain, name] = frame
            lag = self.catalog.loc[self.catalog.output_name == "security_lagged_factor_exposures"]
            if len(lag) != 1:
                raise ValueError("Full lagged security history must be present in the source run.")
            table = qualified_result(db, lag.iloc[0].table_name)
            if db.execute(f"SELECT count(*) FROM {table} WHERE model_available AND estimation_date>=date").fetchone()[0]:
                raise ValueError("Shared factor models violate strict lagging.")
        ids = set(settings.portfolio_ids)
        for name in ("portfolio_returns", "portfolio_holdings", "rebalance_targets", "portfolio_metadata"):
            self.inputs[name] = self.inputs[name].loc[self.inputs[name].portfolio_id.isin(ids)].copy()
        self.returns = self.inputs["portfolio_returns"]
        self.holdings = self.inputs["portfolio_holdings"]
        self.pair = risk.align_portfolio_risk_returns(
            self.returns, settings.portfolio_id, settings.benchmark_id)
        self.calendar = pd.DatetimeIndex(self.pair.date).sort_values()
        if self.calendar.max() != pd.Timestamp(settings.as_of_date):
            raise ValueError("The approved full-history cutoff must be present.")
        self.rf_aligned = performance.align_performance_risk_free(self.pair, self.inputs["factor_returns"])
        self.factor_aligned = self.pair[["date", "portfolio_return_total", "benchmark_return_total"]].rename(
            columns={"portfolio_return_total": "active_return_total"}).merge(
                self.inputs["factor_returns"], on="date", how="left", validate="one_to_one", indicator=True)
        a = self.factor_aligned
        a["factor_available"] = a.pop("_merge").eq("both")
        a["active_portfolio_excess_return"] = a.active_return_total - a.rf
        a["benchmark_excess_return"] = a.benchmark_return_total - a.rf
        a["active_spread_regression_response"] = a.active_return_total - a.benchmark_return_total
        # Explicit full-calendar checks before trusting reused daily components.
        for (domain, name), frame in self.shared.items():
            if "date" in frame:
                if not pd.DatetimeIndex(frame.date.unique()).sort_values().equals(self.calendar):
                    raise ValueError(f"Shared daily calendar mismatch: {domain}/{name}")
        daily = self.shared["performance", "performance_security_contributions_daily"]
        check = daily.groupby(["portfolio_id", "date"]).contribution_total.sum()
        expected = self.returns.set_index(["portfolio_id", "date"]).return_total.reindex(check.index)
        np.testing.assert_allclose(check, expected, atol=1e-10, rtol=0)
        fa = self.shared["factors", "portfolio_factor_attribution"]
        expected = self.returns.set_index(["portfolio_id", "date"]).return_total.reindex(
            pd.MultiIndex.from_frame(fa[["portfolio_id", "date"]]))
        np.testing.assert_allclose(fa.realized_total_return, expected, atol=1e-10, rtol=0)
        self.indexed = {key: frame.set_index("date", drop=False).sort_index()
                        for key, frame in self.shared.items() if "date" in frame}
        self.fixed_cache = {}

    def slice(self, domain, name, w):
        return self.indexed[domain, name].loc[w["first_return_date"]:w["last_return_date"]].reset_index(drop=True)

    def calculate(self, w):
        out = {}
        timings = []
        for domain in DOMAIN_INPUTS:
            start = time.perf_counter()
            try:
                out[domain] = getattr(self, domain)(w)
            except Exception as exc:
                raise ValueError(f"{w['interval_id']} / {domain}: {type(exc).__name__}: {exc}") from exc
            timings.append(dict(domain=domain, elapsed_seconds=time.perf_counter() - start))
        return out, timings

    def performance(self, w):
        s = self.settings
        definitions = performance_definitions(w, s.portfolio_ids)
        out = {"performance_window_definitions": definitions}
        selected = self.returns.loc[self.returns.date.between(w["first_return_date"], w["last_return_date"])]
        pair = self.pair.iloc[w["start_index"]:w["stop_index"]]
        out["performance_period"] = performance.aggregate_period_returns(selected)
        out["performance_relative_period"] = performance.compute_relative_period_performance(pair)
        rolling = self.slice("performance", "performance_rolling_relative", w)
        out["performance_rolling_consistency"] = performance.summarize_rolling_outperformance(rolling)
        horizons = ("1M", "3M", "12M")
        available = tuple(h for h in horizons if (
            rolling.horizon_id.eq(h) & rolling.calculation_status.eq("AVAILABLE")).any())
        _, distribution = performance.prepare_rolling_relative_visuals(
            rolling, portfolio_id=s.portfolio_id, benchmark_portfolio_id=s.benchmark_id, horizons=available)
        distribution["availability_status"] = "AVAILABLE"
        records = distribution.to_dict("records")
        for horizon in set(horizons) - set(available):
            record = {c: pd.NaT if "date" in c else np.nan for c in distribution.columns}
            record.update(horizon_id=horizon, available_windows=0, outperforming_windows=0,
                underperforming_windows=0, tied_windows=0, availability_status="UNAVAILABLE_INSUFFICIENT_ROLLING_HISTORY")
            records.append(record)
        out["performance_rolling_distribution_summary"] = pd.DataFrame(records)
        out.update(compute_month_performance(
            self.returns, self.pair, w, s.portfolio_ids,
            arithmetic_trading_days=s.annualization_sessions))
        out["performance_market_capture"] = performance.compute_market_capture(
            self.pair, definitions, portfolio_id=s.portfolio_id, benchmark_portfolio_id=s.benchmark_id)
        out["performance_rf_coverage"] = performance.assess_performance_rf_coverage(self.rf_aligned, definitions)
        out["performance_risk_adjusted"] = performance.compute_risk_adjusted_performance(
            self.rf_aligned, definitions, sessions_per_year=s.annualization_sessions)
        bind(out, ("performance_security_contributions_period", "performance_security_contribution_reconciliation"),
             performance.link_security_contributions(
                 self.shared["performance", "performance_security_contributions_daily"], self.returns, definitions))
        linked = out["performance_security_contributions_period"]
        out["performance_sector_contributions_period"] = performance.aggregate_sector_contributions(linked)
        out["performance_security_contribution_rankings"] = performance.rank_linked_security_contributions(linked)
        out["performance_return_basis_window"] = performance.compute_return_basis_windows(self.returns, definitions)
        return out

    def attribution(self, w):
        s = self.settings
        definition = pd.DataFrame([dict(portfolio_id=s.portfolio_id, benchmark_id=s.benchmark_id,
            window_id=w["interval_id"], window_type="MONTH_RANGE", window_label=w["interval_id"],
            calendar_start=w["calendar_start"], calendar_end=w["calendar_end"],
            valuation_anchor_date=w["opening_valuation_date"], start_date=w["first_return_date"],
            end_date=w["last_return_date"], observations=w["observation_count"],
            coverage_status="COMPLETE", is_full_period=True)])
        out = {"attribution_reporting_windows": definition}
        bind(out, ("brinson_sector_period", "brinson_total_period"), attribution.build_period_attribution(
            self.slice("attribution", "brinson_sector_daily", w), definition,
            linkers={"carino": attribution.link_attribution_carino, "menchero": attribution.link_attribution_menchero}))
        bind(out, ("brinson_method_comparison_period", "brinson_method_comparison_sector"),
             attribution.compare_attribution_linking_methods(out["brinson_total_period"], out["brinson_sector_period"]))
        bind(out, ("security_contributions_period", "security_contributions_period_reconciliation"),
             attribution.link_security_contributions_by_period(
                 self.slice("attribution", "security_active_contributions_daily", w),
                 self.pair.rename(columns={"benchmark_portfolio_id": "benchmark_id"}),
                 out["brinson_total_period"]))
        bind(out, ("attribution_period_report", "attribution_sector_report", "attribution_security_report"),
             attribution.build_attribution_reporting_tables(out["brinson_total_period"],
                 out["brinson_sector_period"], out["security_contributions_period"]))
        return out

    def risk(self, w):
        s = self.settings
        panel = self.pair.iloc[w["start_index"]:w["stop_index"]]
        n, end = len(panel), w["last_return_date"]
        returns = self.returns.loc[self.returns.date.between(w["first_return_date"], end)]
        holdings = self.holdings.loc[self.holdings.date.between(w["first_return_date"], end)]
        absolute, tails, details, audits, episodes, paths = [], [], [], [], [], []
        for pid, col in ((s.portfolio_id, "portfolio_return_total"), (s.benchmark_id, "benchmark_return_total")):
            path = risk.calculate_drawdown_path(panel[col], panel.date)
            absolute.append(dict(portfolio_id=pid, sample_start=panel.date.min(), as_of_date=end, **{
                **risk.calculate_absolute_risk(panel[col], s.annualization_sessions, s.variance_ddof),
                **risk.summarize_drawdown_path(path)}))
            episodes.append(risk.extract_drawdown_episodes(path).assign(portfolio_id=pid))
            paths.append(path.assign(portfolio_id=pid))
            tails.append(risk.calculate_historical_tail_risk(panel[col], panel.date, end,
                s.tail_confidences, s.minimum_tail_mass).assign(portfolio_id=pid))
            detail, audit = risk.calculate_realized_risk_contributions(holdings, returns, pid, end, n,
                s.annualization_sessions, s.variance_ddof)
            details.append(detail.assign(portfolio_id=pid))
            audits.append(dict(portfolio_id=pid, **audit))
        spread = panel.portfolio_return_total - panel.benchmark_return_total
        tails.append(risk.calculate_historical_tail_risk(spread, panel.date, end,
            s.tail_confidences, s.minimum_tail_mass, return_kind="ARITHMETIC_ACTIVE").assign(
                portfolio_id=s.portfolio_id, benchmark_portfolio_id=s.benchmark_id))
        active_detail, active_audit = risk.calculate_active_risk_contributions(holdings, panel,
            s.portfolio_id, s.benchmark_id, end, n, s.annualization_sessions, s.variance_ddof)
        sectors, sector_audit = risk.calculate_sector_risk_contributions(holdings, panel,
            s.portfolio_id, s.benchmark_id, end, n, s.annualization_sessions, s.variance_ddof)
        relative = (1 + panel.portfolio_return_total) / (1 + panel.benchmark_return_total) - 1
        return dict(risk_reporting_windows=pd.DataFrame([dict(window_id=w["interval_id"],
                window_start=w["first_return_date"], as_of_date=end, observations=n, coverage_status="AVAILABLE")]),
            risk_absolute_summary=pd.DataFrame(absolute),
            risk_active_summary=pd.DataFrame([dict(portfolio_id=s.portfolio_id,
                benchmark_portfolio_id=s.benchmark_id, sample_start=panel.date.min(), as_of_date=end,
                **risk.calculate_active_risk(panel.portfolio_return_total, panel.benchmark_return_total,
                    s.annualization_sessions, s.variance_ddof))]),
            risk_historical_tail=pd.concat(tails, ignore_index=True),
            risk_security_contributions=pd.concat(details, ignore_index=True),
            risk_security_reconciliation=pd.DataFrame(audits), risk_active_contributions=active_detail,
            risk_active_reconciliation=pd.DataFrame([active_audit]), risk_sector_contributions=sectors,
            risk_sector_reconciliation=sector_audit, risk_drawdown_episodes=pd.concat(episodes, ignore_index=True),
            risk_drawdown_daily=pd.concat(paths, ignore_index=True),
            risk_relative_drawdown_daily=risk.calculate_drawdown_path(relative, panel.date))

    def closing_risk(self, w):
        """Calculated once per end month, not once per start/end combination."""
        end, s = w["last_return_date"], self.settings
        if end in self.fixed_cache:
            return self.fixed_cache[end]
        closing = risk.select_asof_closing_weights(self.holdings, s.portfolio_ids, end)
        ids = closing.security_id.unique()
        calendar = self.calendar[self.calendar <= end]
        dates = calendar[-s.covariance_sessions:]
        sec = self.inputs["security_returns"]
        observed = sec.loc[sec.security_id.isin(ids) & sec.date.isin(dates)]
        out = {"risk_fixed_weight_holdings": closing}
        if len(dates) != s.covariance_sessions or len(observed) != len(ids) * len(dates):
            out["risk_covariance_diagnostics"] = pd.DataFrame([dict(as_of_date=end,
                lookback_sessions=s.covariance_sessions, coverage_status="UNAVAILABLE_INCOMPLETE_COVARIANCE_HISTORY")])
        else:
            cov, corr, diagnostic = risk.estimate_security_covariance(sec, ids, calendar, end,
                s.covariance_sessions, s.annualization_sessions, s.variance_ddof)
            out["risk_covariance_diagnostics"] = pd.DataFrame([diagnostic])
            i, j = np.indices(cov.shape)
            out["risk_security_covariance"] = pd.DataFrame(dict(security_id=cov.index.to_numpy()[i.ravel()],
                other_security_id=cov.columns.to_numpy()[j.ravel()], covariance_daily=cov.to_numpy().ravel(),
                correlation=corr.to_numpy().ravel(), as_of_date=end))
            weights = closing.pivot(index="security_id", columns="portfolio_id", values="weight_eop").fillna(0)
            details, records = [], []
            for label, weight in [(pid, weights[pid]) for pid in s.portfolio_ids] + [
                    ("ACTIVE_DIFFERENCE", weights[s.portfolio_id] - weights[s.benchmark_id])]:
                detail, record = risk.calculate_fixed_weight_risk_components(cov,
                    weight.reindex(cov.index, fill_value=0), s.annualization_sessions)
                details.append(detail.assign(weight_vector=label, as_of_date=end))
                records.append(dict(**record, weight_vector=label, as_of_date=end))
            out["risk_fixed_weight_components"] = pd.concat(details, ignore_index=True)
            out["risk_fixed_weight_diagnostics"] = pd.DataFrame(records)
        self.fixed_cache[end] = out
        return out

    def exposures(self, w):
        definitions = pd.DataFrame([dict(report_window=w["interval_id"], years=w["inclusive_months"] / 12,
            as_of_date=w["last_return_date"], required_start_date=w["calendar_start"],
            window_start_date=w["first_return_date"], first_available_date=w["first_return_date"],
            available_observations=w["observation_count"], availability_status="AVAILABLE")])
        out = {"exposure_reporting_windows": definitions}
        bind(out, ("portfolio_window_summary", "active_window_summary", "sector_tilt_window_summary"),
             exposures.build_exposure_window_summaries(*[
                 self.slice("exposures", name, w) for name in NEEDED["exposures"]], definitions))
        return out

    def factors(self, w):
        s, fc = self.settings, self.settings.factor_columns
        aligned = self.factor_aligned.iloc[w["start_index"]:w["stop_index"]].copy()
        definitions = pd.DataFrame([dict(period_id=w["interval_id"], period_type="MONTH_RANGE",
            period_start_date=w["first_return_date"], period_end_date=w["last_return_date"])])
        out = {"direct_regression_periods": definitions,
            "factor_execution_scope": pd.DataFrame([dict(component="security_factor_history", status="REUSED_VERIFIED_HISTORY",
                factor_cutoff=aligned.loc[aligned.factor_available, "date"].max(),
                portfolio_cutoff=w["last_return_date"], missing_factor_dates=int((~aligned.factor_available).sum()))])}
        bind(out, ("direct_portfolio_factor_fit", "direct_portfolio_factor_coefficients", "direct_portfolio_factor_residuals"),
             estimate_direct_periods(aligned, definitions, factor_columns=fc, model_id=s.factor_model_id))
        inference, diagnostics, risks, components, covariances = [], [], [], [], []
        for fit in out["direct_portfolio_factor_fit"].itertuples(index=False):
            if fit.fit_status != "FIT_COMPLETED":
                continue
            xfactor = aligned[list(fc)].to_numpy(dtype=float)
            x = np.column_stack([np.ones(len(aligned)), xfactor])
            y = aligned[RESPONSE_MAPPING[fit.target_id]].to_numpy(dtype=float)
            coef = out["direct_portfolio_factor_coefficients"]
            beta = coef.loc[coef.target_id.eq(fit.target_id)].set_index("term").loc[["intercept", *fc], "coefficient"].to_numpy()
            residual = y - x @ beta
            tag = dict(target_id=fit.target_id, period_id=w["interval_id"], model_id=s.factor_model_id)
            frame, _, _ = factors.calculate_direct_ols_hac_inference(x, y, beta,
                terms=("intercept", *fc), hac_lags=factors.select_direct_hac_lag(len(y)))
            inference.append(frame.assign(**tag))
            diagnostics.append(dict(**tag, **factors.calculate_direct_regression_residual_diagnostics(x, residual)))
            record, frame, cov = factors.decompose_direct_regression_factor_risk(xfactor, y, beta, residual,
                factor_names=fc, annualization_days=s.annualization_sessions)
            risks.append(dict(**tag, **record))
            components.append(frame.assign(**tag))
            covariances.extend(dict(**tag, factor=f1, other_factor=f2, covariance_daily=cov[i, j])
                for i, f1 in enumerate(fc) for j, f2 in enumerate(fc))
        if inference:
            out.update(direct_portfolio_factor_inference=pd.concat(inference, ignore_index=True),
                direct_portfolio_factor_residual_diagnostics=pd.DataFrame(diagnostics),
                direct_factor_risk_summary=pd.DataFrame(risks),
                direct_factor_risk_contributions=pd.concat(components, ignore_index=True),
                direct_factor_covariance=pd.DataFrame(covariances))
        columns = ("risk_free_contribution", "estimated_intercept_contribution",
            *[f"factor_contribution_{f}" for f in fc], "modeled_residual_contribution", "unmodeled_excess_contribution")
        modeled = tuple(c for c in columns if c not in ("risk_free_contribution", "unmodeled_excess_contribution"))
        linked = factors.link_portfolio_factor_attribution_periods(
            self.slice("factors", "portfolio_factor_attribution", w), definitions,
            component_columns=columns, modeled_component_columns=modeled)
        out["portfolio_factor_linked"] = linked
        out["active_factor_linked"] = factors.calculate_linked_active_factor_attribution(linked,
            active_portfolio_id=s.portfolio_id, benchmark_portfolio_id=s.benchmark_id, component_columns=columns)
        return out

    def turnover(self, w):
        from .month_turnover import select_turnover_events
        start, end = w["first_return_date"], w["last_return_date"]
        selected = select_turnover_events(self.shared["turnover", "rebalance_turnover_events"], w)
        daily = self.slice("turnover", "daily_rebalancing_diagnostics", w)
        out = {}
        out["rebalance_turnover_events"] = selected
        bind(out, ("quarterly_turnover", "annual_turnover", "portfolio_turnover_summary"),
             turnover.summarize_rebalance_turnover_periods_portable(selected, daily))
        security = self.shared["turnover", "rebalance_security_changes"]
        security = security.loc[security.effective_rebalance_date.between(start, end)]
        sector = self.shared["turnover", "rebalance_sector_turnover"]
        sector = sector.merge(selected.loc[selected.turnover_status.eq("PASSED"), ["portfolio_id", "selection_date"]],
            on=["portfolio_id", "selection_date"], validate="many_to_one")
        bind(out, ("rebalance_security_concentration", "rebalance_sector_concentration", "rebalance_event_concentration",
                   "cumulative_security_turnover_concentration", "cumulative_sector_turnover_concentration", "portfolio_turnover_concentration"),
             turnover.analyze_scheduled_turnover_concentration(security, sector, selected))
        bind(out, ("rebalance_event_cost_sensitivity", "quarterly_cost_sensitivity", "annual_cost_sensitivity", "portfolio_cost_sensitivity"),
             turnover.calculate_turnover_unit_cost_sensitivity(selected, out["quarterly_turnover"],
                out["annual_turnover"], out["portfolio_turnover_summary"]))
        return out


def complete_month_cagr_policy(w):
    """Approve full-year month selections without changing the kernel's formula.

    Trading-session endpoints can be fewer than 365 days apart for 12 complete
    months. Only this validated monthly adapter relaxes that eligibility gate;
    the general-purpose kernels retain their 365-day defaults and actual-day
    annualization. The effective day gate and month rule are saved explicitly.
    """
    start, end = pd.Timestamp(w["calendar_start"]), pd.Timestamp(w["calendar_end"])
    window = MonthWindow(start.to_period("M"), end.to_period("M"))
    if window.months < 12:
        raise ValueError("CAGR requires at least 12 complete months.")
    months = w["inclusive_months"]
    if (isinstance(months, (bool, np.bool_))
            or not isinstance(months, (int, np.integer)) or months != window.months):
        raise ValueError("CAGR requires consistent inclusive complete-month metadata.")
    if start != window.start.start_time.normalize() or end != window.end.end_time.normalize():
        raise ValueError("CAGR month selections must cover full calendar months.")
    opening = pd.Timestamp(w["opening_valuation_date"])
    first, last = pd.Timestamp(w["first_return_date"]), pd.Timestamp(w["last_return_date"])
    if (opening.to_period("M") != window.start - 1
            or first.to_period("M") != window.start or last.to_period("M") != window.end
            or not opening < first <= last):
        raise ValueError("CAGR requires the preceding-month close and selected-month sessions.")
    days = (last - opening).days
    if days <= 0:
        raise ValueError("CAGR requires a positive actual-day duration.")
    return dict(cagr_eligibility_basis="AT_LEAST_12_COMPLETE_MONTHS",
                minimum_cagr_complete_months=12, inclusive_months=int(months),
                minimum_cagr_calendar_days=min(365, days))


def compute_month_performance(returns, aligned_returns, w, portfolio_ids, *, arithmetic_trading_days=252):
    """Calculate both CAGR tables under one explicit complete-month policy."""
    policy = complete_month_cagr_policy(w)
    definitions = performance_definitions(w, portfolio_ids)
    options = dict(arithmetic_trading_days=arithmetic_trading_days,
                   minimum_cagr_calendar_days=policy["minimum_cagr_calendar_days"])
    outputs = {
        "performance_window": performance.compute_absolute_performance(returns, definitions, **options),
        "performance_relative_window": performance.compute_relative_window_performance(
            aligned_returns, definitions, **options),
    }
    for frame in outputs.values():
        for column, value in policy.items():
            frame[column] = value
    return outputs


def performance_definitions(w, portfolio_ids):
    """Use a real prior-month valuation anchor, not the first in-month close."""
    return pd.DataFrame([dict(portfolio_id=pid, window_id=w["interval_id"],
        requested_as_of_date=w["calendar_end"], as_of_date=w["last_return_date"],
        requested_start_date=w["calendar_start"], resolved_start_date=w["opening_valuation_date"],
        first_return_date=w["first_return_date"], last_return_date=w["last_return_date"],
        observation_count=w["observation_count"],
        calendar_days=(w["last_return_date"] - w["opening_valuation_date"]).days,
        start_adjustment_days=(w["calendar_start"] - w["opening_valuation_date"]).days,
        start_anchor_source="EXPLICIT_INCEPTION_BASELINE" if w["start_index"] == 0 else "OBSERVED_RETURN_DATE",
        cutoff_status="EXACT" if w["calendar_end"] == w["last_return_date"] else "PREVIOUS_OBSERVED_SESSION",
        window_status="AVAILABLE", session_coverage_status="COMPLETE") for pid in portfolio_ids])
