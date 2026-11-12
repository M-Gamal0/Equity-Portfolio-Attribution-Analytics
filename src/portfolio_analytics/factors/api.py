"""Independent factor application, with expensive security history opt-in."""

import numpy as np
import pandas as pd

from portfolio_analytics.common.selection import bind, select_pair
from portfolio_analytics.config import AnalysisSettings
from . import (
    direct,
    risk,
    rolling,
    inputs,
    security,
    lagging,
    exposures,
    attribution,
    linking,
    sectors,
)
from .period_regressions import estimate_direct_periods

RESPONSE_MAPPING = {
    "ACTIVE_PORTFOLIO": "active_portfolio_excess_return",
    "BENCHMARK": "benchmark_excess_return",
    "ACTIVE_RETURN": "active_spread_regression_response",
}


def run(
    *,
    portfolio_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    settings: AnalysisSettings,
    security_returns: pd.DataFrame | None = None,
    portfolio_holdings: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute direct period regressions, inference and conditional factor risk.

    Enable include_security_factor_history for daily security betas, strictly
    lagged holdings attribution and linked factor effects. This is an offline job.
    """
    returns = select_pair(portfolio_returns, settings)
    fc, model = settings.factor_columns, settings.factor_model_id
    factor = factor_returns[["date", "rf", *fc]].copy()
    factor["date"] = pd.to_datetime(factor["date"], errors="raise")
    if factor["date"].isna().any() or factor["date"].duplicated().any():
        raise ValueError("Factor dates must be unique and nonmissing.")
    if not np.isfinite(factor[["rf", *fc]].to_numpy(dtype=float)).all():
        raise ValueError(
            "Factor observations must be finite; missing dates remain unavailable."
        )
    wide = returns.pivot(
        index="date", columns="portfolio_id", values="return_total"
    ).sort_index()
    if wide.isna().any().any():
        raise ValueError("The portfolio and benchmark calendars must align.")
    aligned = wide.reset_index().rename(
        columns={
            settings.portfolio_id: "active_return_total",
            settings.benchmark_id: "benchmark_return_total",
        }
    )
    aligned = aligned.merge(
        factor, on="date", how="left", validate="one_to_one", indicator=True
    )
    aligned["factor_available"] = aligned["_merge"].eq("both")
    aligned = aligned.drop(columns="_merge")
    aligned["active_portfolio_excess_return"] = (
        aligned["active_return_total"] - aligned["rf"]
    )
    aligned["benchmark_excess_return"] = (
        aligned["benchmark_return_total"] - aligned["rf"]
    )
    aligned["active_spread_regression_response"] = (
        aligned["active_return_total"] - aligned["benchmark_return_total"]
    )
    first, last = aligned["date"].min(), aligned["date"].max()
    cutoff = aligned.loc[aligned["factor_available"], "date"].max()
    out = {"factor_calendar_coverage": aligned[["date", "factor_available"]].copy()}
    if pd.isna(cutoff):
        out["factor_execution_scope"] = pd.DataFrame(
            [{"status": "UNAVAILABLE_NO_COMMON_FACTOR_DATES"}]
        )
        return out
    periods = pd.DataFrame(
        [
            dict(
                period_id=f"TRAILING_{y}Y",
                period_type="TRAILING",
                period_start_date=cutoff - pd.DateOffset(years=y),
                period_end_date=cutoff,
            )
            for y in settings.reporting_years
        ]
        + [
            dict(
                period_id="FULL_FACTOR_AVAILABLE_SAMPLE",
                period_type="FULL_FACTOR_SAMPLE",
                period_start_date=first,
                period_end_date=cutoff,
            ),
            dict(
                period_id="FULL_PORTFOLIO_SOURCE_SAMPLE",
                period_type="FULL_SOURCE_SAMPLE",
                period_start_date=first,
                period_end_date=last,
            ),
        ]
    )
    out["direct_regression_periods"] = periods
    bind(
        out,
        (
            "direct_portfolio_factor_fit",
            "direct_portfolio_factor_coefficients",
            "direct_portfolio_factor_residuals",
        ),
        estimate_direct_periods(aligned, periods, factor_columns=fc, model_id=model),
    )
    fits, coeffs = (
        out["direct_portfolio_factor_fit"],
        out["direct_portfolio_factor_coefficients"],
    )
    (
        inference_frames,
        diagnostic_records,
        risk_records,
        risk_components,
        covariance_records,
    ) = [], [], [], [], []
    for row in fits.loc[fits["fit_status"].eq("FIT_COMPLETED")].itertuples(index=False):
        sample = aligned.loc[
            aligned["date"].between(row.period_start_date, row.period_end_date)
        ]
        factors = sample[list(fc)].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(sample)), factors])
        y = sample[RESPONSE_MAPPING[row.target_id]].to_numpy(dtype=float)
        beta = (
            coeffs.loc[
                coeffs["period_id"].eq(row.period_id)
                & coeffs["target_id"].eq(row.target_id)
            ]
            .set_index("term")
            .loc[["intercept", *fc], "coefficient"]
            .to_numpy(dtype=float)
        )
        residual = y - X @ beta
        tag = dict(target_id=row.target_id, period_id=row.period_id, model_id=model)
        inference, _, _ = direct.calculate_direct_ols_hac_inference(
            X,
            y,
            beta,
            terms=("intercept", *fc),
            hac_lags=direct.select_direct_hac_lag(len(sample)),
        )
        inference_frames.append(inference.assign(**tag))
        diagnostic_records.append(
            {
                **tag,
                **direct.calculate_direct_regression_residual_diagnostics(X, residual),
            }
        )
        summary, components, cov = risk.decompose_direct_regression_factor_risk(
            factors,
            y,
            beta,
            residual,
            factor_names=fc,
            annualization_days=settings.annualization_sessions,
        )
        risk_records.append({**tag, **summary})
        risk_components.append(components.assign(**tag))
        for i, f1 in enumerate(fc):
            for j, f2 in enumerate(fc):
                covariance_records.append(
                    {
                        **tag,
                        "factor": f1,
                        "other_factor": f2,
                        "covariance_daily": cov[i, j],
                    }
                )
    if inference_frames:
        out.update(
            direct_portfolio_factor_inference=pd.concat(
                inference_frames, ignore_index=True
            ),
            direct_portfolio_factor_residual_diagnostics=pd.DataFrame(
                diagnostic_records
            ),
            direct_factor_risk_summary=pd.DataFrame(risk_records),
            direct_factor_risk_contributions=pd.concat(
                risk_components, ignore_index=True
            ),
            direct_factor_covariance=pd.DataFrame(covariance_records),
        )
    bind(
        out,
        (
            "rolling_portfolio_factor_fit",
            "rolling_portfolio_factor_coefficients",
            "rolling_portfolio_factor_fit_status",
        ),
        rolling.estimate_rolling_direct_portfolio_ols(
            aligned,
            factor_columns=fc,
            response_mapping=RESPONSE_MAPPING,
            model_id=model,
            lookback_years=settings.factor_lookback_years,
        ),
    )
    out["factor_execution_scope"] = pd.DataFrame(
        [
            dict(
                component="security_factor_history",
                status="REQUESTED"
                if settings.include_security_factor_history
                else "NOT_REQUESTED",
                factor_cutoff=cutoff,
                portfolio_cutoff=last,
                missing_factor_dates=int((~aligned["factor_available"]).sum()),
            )
        ]
    )
    if settings.include_security_factor_history:
        if security_returns is None or portfolio_holdings is None:
            raise ValueError(
                "Security factor history requires security_returns and portfolio_holdings."
            )
        out.update(
            run_security_history(
                security_returns=security_returns,
                portfolio_holdings=portfolio_holdings,
                factor_returns=factor,
                settings=settings,
            )
        )
        out["factor_execution_scope"]["status"] = "COMPLETED"
    return out


def run_security_history(
    *, security_returns, portfolio_holdings, factor_returns, settings
):
    """Compute rolling security history and lagged attribution, without other domains."""
    fc, model = settings.factor_columns, settings.factor_model_id
    holdings = select_pair(portfolio_holdings, settings)
    factor = factor_returns.loc[
        pd.to_datetime(factor_returns["date"]).le(pd.Timestamp(settings.as_of_date))
    ].copy()
    history = security_returns.loc[
        security_returns["security_id"].isin(holdings["security_id"].unique())
        & pd.to_datetime(security_returns["date"]).le(pd.Timestamp(settings.as_of_date))
    ].copy()
    observed = inputs.align_factor_regression_observations(
        history, factor, entity_columns=("security_id",), factor_columns=fc
    )
    eligibility = inputs.evaluate_factor_history_eligibility(
        observed,
        factor,
        entity_column="security_id",
        factor_columns=fc,
        model_id=model,
        lookback_years=settings.factor_lookback_years,
    )
    out = {"security_factor_eligibility": eligibility}
    bind(
        out,
        (
            "security_factor_fit",
            "security_factor_coefficients",
            "security_factor_coverage",
        ),
        security.estimate_full_universe_security_factors(
            observed, factor, eligibility, factor_columns=fc, model_id=model
        ),
    )
    coefficients = out["security_factor_coefficients"]
    wide = (
        coefficients.set_index(["security_id", "estimation_date", "term"])[
            "coefficient"
        ]
        .unstack("term")
        .reset_index()
    )
    wide = wide.rename(
        columns={"intercept": "estimated_alpha_daily", **{f: f"beta_{f}" for f in fc}}
    )
    columns = ("estimated_alpha_daily", *[f"beta_{f}" for f in fc])
    targets = holdings[["security_id", "date"]].drop_duplicates()
    lagged = lagging.align_prior_security_factor_coefficients(
        targets, wide, coefficient_columns=columns, model_id=model
    )
    reference_dates = (
        pd.DatetimeIndex(factor["date"]).sort_values().to_numpy(dtype="datetime64[ns]")
    )
    positions = (
        np.searchsorted(
            reference_dates,
            lagged["date"].to_numpy(dtype="datetime64[ns]"),
            side="left",
        )
        - 1
    )
    previous = np.full(len(lagged), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    previous[positions >= 0] = reference_dates[positions[positions >= 0]]
    lagged["preceding_factor_date"] = previous
    lagged["factor_available_on_return_date"] = lagged["date"].isin(factor["date"])
    available = lagged["model_available"]
    current = lagged["factor_available_on_return_date"]
    fresh = lagged["estimation_date"].eq(lagged["preceding_factor_date"])
    lagged["lagged_beta_status"] = np.select(
        [
            ~available,
            available & ~current,
            available & current & fresh,
            available & current & ~fresh,
        ],
        [
            "NO_PRIOR_ESTIMATE",
            "MODEL_AVAILABLE_NO_CURRENT_FACTOR",
            "MODELED_FRESH",
            "MODELED_STALE",
        ],
        default="INVALID_ALIGNMENT",
    )
    lagged["estimation_age_calendar_days"] = (
        lagged["date"] - lagged["estimation_date"]
    ).dt.days.astype("Int32")
    # The research factor kernels use string security IDs for the as-of join.
    held = holdings[["portfolio_id", "date", "security_id", "weight_bop"]].copy()
    held["security_id"] = held["security_id"].astype(str)
    held = held.merge(lagged, on=["security_id", "date"], validate="many_to_one")
    out["security_lagged_factor_exposures"] = lagged
    out["portfolio_factor_exposures"] = (
        exposures.aggregate_holdings_based_factor_exposures(
            held, factor_columns=fc, model_id=model
        )
    )
    out["active_factor_exposures"] = (
        exposures.calculate_active_holdings_factor_exposures(
            out["portfolio_factor_exposures"],
            active_portfolio_id=settings.portfolio_id,
            benchmark_portfolio_id=settings.benchmark_id,
            factor_columns=fc,
        )
    )
    out["security_factor_attribution"] = (
        attribution.calculate_holding_factor_return_decomposition(
            held, holdings, factor, factor_columns=fc, model_id=model
        )
    )
    portfolio = attribution.aggregate_daily_security_factor_attribution(
        out["security_factor_attribution"], factor_columns=fc, model_id=model
    )
    act = attribution.calculate_daily_active_factor_attribution(
        portfolio,
        active_portfolio_id=settings.portfolio_id,
        benchmark_portfolio_id=settings.benchmark_id,
        factor_columns=fc,
    )
    out.update(portfolio_factor_attribution=portfolio, active_factor_attribution=act)
    dates = pd.DatetimeIndex(act["date"])
    cutoff = act.loc[act["factor_available_on_return_date"], "date"].max()
    period_records = []
    for label, freq in (("MONTHLY", "M"), ("QUARTERLY", "Q"), ("ANNUAL", "Y")):
        for period in dates.to_period(freq).unique():
            period_records.append(
                dict(
                    period_id=f"{label}_{period}",
                    period_type=label,
                    period_start_date=period.start_time.normalize(),
                    period_end_date=period.end_time.normalize(),
                )
            )
    period_records.extend(
        dict(
            period_id=f"TRAILING_{y}Y_{cutoff:%Y%m%d}",
            period_type="TRAILING",
            period_start_date=cutoff - pd.DateOffset(years=y),
            period_end_date=cutoff,
        )
        for y in settings.reporting_years
    )
    period_records.extend(
        [
            dict(
                period_id="FULL_FACTOR_AVAILABLE_SAMPLE",
                period_type="FULL_FACTOR_SAMPLE",
                period_start_date=dates.min(),
                period_end_date=cutoff,
            ),
            dict(
                period_id="FULL_PORTFOLIO_SOURCE_SAMPLE",
                period_type="FULL_SOURCE_SAMPLE",
                period_start_date=dates.min(),
                period_end_date=dates.max(),
            ),
        ]
    )
    components = (
        "risk_free_contribution",
        "estimated_intercept_contribution",
        *[f"factor_contribution_{f}" for f in fc],
        "modeled_residual_contribution",
        "unmodeled_excess_contribution",
    )
    modeled = tuple(
        c
        for c in components
        if c not in ("risk_free_contribution", "unmodeled_excess_contribution")
    )
    linked = linking.link_portfolio_factor_attribution_periods(
        portfolio,
        pd.DataFrame(period_records),
        component_columns=components,
        modeled_component_columns=modeled,
    )
    out["portfolio_factor_linked"] = linked
    out["active_factor_linked"] = linking.calculate_linked_active_factor_attribution(
        linked,
        active_portfolio_id=settings.portfolio_id,
        benchmark_portfolio_id=settings.benchmark_id,
        component_columns=components,
    )
    classified = holdings[["portfolio_id", "date", "security_id", "sector"]].copy()
    classified["security_id"] = classified["security_id"].astype(str)
    out["sector_factor_exposures"] = sectors.aggregate_sector_modeled_factor_exposures(
        held.merge(
            classified,
            on=["portfolio_id", "date", "security_id"],
            validate="one_to_one",
        ),
        factor_columns=fc,
    )
    return out
