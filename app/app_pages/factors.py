"""Saved direct regressions and distinct lagged holdings-based factor analytics."""
import numpy as np
import pandas as pd
import streamlit as st

from research_ui import FACTOR_NAMES, context, pct, detailed, methodology, status_text
from factor_details import (TARGETS, RESPONSES, alpha_evidence, coefficient_chart, variance_chart,
                            linked_evidence, linked_chart, coverage_evidence, exposure_chart,
                            supporting_data, remembered_choice)

ctx = context()
ctx.heading("Factors", "Understand direct factor estimates, their uncertainty and lagged holdings-based return attribution.")
st.caption("Hypothetical 20-security active strategy versus a custom Top-100 eligible-issuer equal-weight benchmark; both rebalance quarterly. Total returns assume frictionless same-close implementation.")
factors = ctx.settings.get("factor_columns", ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"])
fits = ctx.load("factors", "direct_portfolio_factor_fit")
target = remembered_choice("Direct-regression target", fits.target_id.unique().tolist(), "factor_direct_target", lambda x: TARGETS.get(x, x))
fit_rows = fits.loc[fits.target_id.eq(target)]
fit = fit_rows.iloc[0] if len(fit_rows) == 1 else None
available = fit is not None and fit.fit_status == "FIT_COMPLETED"
coefficients = ctx.load("factors", "direct_portfolio_factor_coefficients")
coefficients = coefficients.loc[coefficients.target_id.eq(target)]
inference = ctx.load("factors", "direct_portfolio_factor_inference")
inference = inference.loc[inference.target_id.eq(target)]
alpha = alpha_evidence(inference)
alpha_row = alpha.iloc[0] if available and len(alpha) == 1 else None
st.markdown("**Direct fit: " + TARGETS.get(target, target) + " · selected period**")
st.caption(RESPONSES.get(target, target) + ". FF5 + momentum fitted across the applied interval. Direct target and holdings owner below are independent choices.")
def saved_coefficient(term):
    rows = coefficients.loc[coefficients.term.eq(term)]
    return rows.coefficient.iloc[0] if available and len(rows) == 1 else float("nan")
a, market = saved_coefficient("intercept"), saved_coefficient("mkt_rf")
with st.container(horizontal=True):
    for label, value in [("Daily alpha", f"{a * 10000:+.2f} bp" if np.isfinite(a) else "Unavailable"),
                         ("Market beta", f"{market:.3f}" if np.isfinite(market) else "Unavailable"),
                         ("R-squared", pct(fit.r_squared) if available else "Unavailable"),
                         ("Regression observations", f"{int(fit.regression_observations):,}" if fit is not None else "Unavailable")]:
        st.metric(label, value, border=True, width=190)
if not available:
    st.warning("Direct fit unavailable: " + (status_text(fit.fit_status) if fit is not None else "No unique saved fit") + ".")
if alpha_row is not None and np.isfinite([alpha_row.hac_ci_lower_daily_bp, alpha_row.hac_ci_upper_daily_bp, alpha_row.hac_p_value]).all():
    st.markdown(f"**Daily alpha · saved HAC 95% interval: {alpha_row.hac_ci_lower_daily_bp:+.2f} to {alpha_row.hac_ci_upper_daily_bp:+.2f} bp**")
    st.caption(f"HAC p-value {alpha_row.hac_p_value:.4g} · {int(alpha_row.n_observations):,} observations · {int(alpha_row.hac_lags)} saved HAC lags. Alpha and its interval are daily, not annualized; statistical estimates do not forecast investment outcomes.")
else:
    st.caption("Daily alpha uncertainty is unavailable under the saved fit/inference coverage; complete saved evidence remains below.")
if fit is not None:
    st.caption(f"Saved regression sample: {pd.Timestamp(fit.observed_start_date):%d %b %Y}–{pd.Timestamp(fit.observed_end_date):%d %b %Y} · {int(fit.factor_available_days):,} factor-available days · {int(fit.factor_missing_days):,} missing factor days. R-squared describes in-sample explanatory fit.")

st.subheader("Direct factor coefficients and uncertainty")
coefficient_chart(coefficients, inference, factors, available)
variance = ctx.load("factors", "direct_factor_risk_summary")
variance = variance.loc[variance.target_id.eq(target)]
st.subheader("Systematic and residual regression-response variance")
variance_chart(variance, available, target)

st.subheader("Holdings-based linked contributions to total return")
linked = ctx.load("factors", "portfolio_factor_linked")
owner = remembered_choice("Holdings-based return series", linked.portfolio_id.unique().tolist(), "factor_holdings_owner", segmented=True)
rows = linked.loc[linked.portfolio_id.eq(owner)]
evidence = pd.DataFrame()
if len(rows) == 1:
    row = rows.iloc[0]
    evidence = linked_evidence(row, factors)
    st.caption(f"{owner} · {status_text(row.link_status)} · Minimum modeled gross-weight coverage {row.minimum_modeled_gross_weight_coverage:.2%} · Factor-date coverage {row.factor_coverage_ratio:.2%} · Source: {status_text(row.period_source_status)}.")
    linked_chart(evidence)
else:
    st.info("No unique saved linked attribution for this owner.")
st.caption("Holdings-based attribution uses BOP weights and strictly prior security models. Risk-free, six factors, security-model intercept, modeled residual and unmodeled excess remain separate. It explains the owner's compounded total return; it is distinct from the direct regression above and sector attribution to the portfolio–benchmark return difference.")

st.subheader("Lagged holdings-based factor exposure through time")
factor = remembered_choice("Exposure factor", list(factors), "factor_exposure_term", lambda x: FACTOR_NAMES.get(x, x))
exposure = ctx.history("factors", "portfolio_factor_exposures")
coverage = coverage_evidence(exposure)
for item in coverage.itertuples():
    st.caption(f"{item.portfolio_id}: {item.fully_modeled_days:,} fully modeled days · {item.partially_modeled_days:,} partially modeled · {item.other_coverage_status_days:,} other/unavailable · Minimum modeled gross-weight coverage {item.minimum_modeled_gross_weight_coverage:.2%} · {item.factor_available_days:,}/{item.daily_observations:,} factor-available days.")
exposure_chart(exposure, "exposure_" + factor, FACTOR_NAMES.get(factor, factor) + " · modeled holdings exposure")
st.caption("BOP-weighted exposures include modeled holdings only and are not scaled up to full portfolio weight. Missing security models are not zero betas. The original three-calendar-year estimation history and strict prior-date lag remain unchanged by the reporting selection.")
exposure_chart(exposure, "modeled_gross_weight_coverage", "Modeled gross-weight coverage", coverage=True)
st.caption("Model coverage and factor-calendar availability are different. Partial model coverage remains explicit; unmodeled and stale weights, model counts and daily statuses remain in supporting data.")

risk_components = ctx.load("factors", "direct_factor_risk_contributions")
risk_components = risk_components.loc[risk_components.target_id.eq(target)]
covariance = ctx.load("factors", "direct_factor_covariance")
covariance = covariance.loc[covariance.target_id.eq(target)]
supporting_data({
    "Selected-target fit and sample metadata": fit_rows,
    "Selected-target coefficients · all terms": coefficients,
    "Selected-target inference · includes alpha": inference,
    "Daily alpha and HAC interval · bp units": alpha,
    "Selected-target variance summary and reconciliation": variance,
    "Selected-target factor risk contributions": risk_components,
    "Selected-target daily factor covariance": covariance,
    "Selected-owner linked return components · pp": evidence,
    "Saved linked totals, coverage and statuses · both owners": linked,
    "Daily holdings exposures, weights and statuses · both owners": exposure,
    "Selected-period model and factor-date coverage counts": coverage,
})
detailed(ctx, "factors")
methodology(ctx, "Direct targets use saved portfolio-minus-RF, benchmark-minus-RF or portfolio-minus-benchmark daily responses. Daily alpha and its HAC interval are in bp; other direct coefficients are dimensionless. Variance shares describe the in-sample response. Beginning-wealth-linked holdings contributions are in pp of compounded total return, retaining modeled residual and unmodeled excess. Strict lagging, three-year security estimation, fit eligibility and all 26 raw result choices remain unchanged. No regressions are run by the app.")
