"""Read-only display of saved factor fits, inference, linking and coverage."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from chart_renderer import render_chart
from research_ui import COLORS, FACTOR_NAMES, SERIES, date_encoding, display_table, styled

TARGETS = {"ACTIVE_PORTFOLIO": "Portfolio excess return", "BENCHMARK": "Benchmark excess return", "ACTIVE_RETURN": "Portfolio minus benchmark"}
RESPONSES = {"ACTIVE_PORTFOLIO": "Daily portfolio total return minus the saved risk-free return",
             "BENCHMARK": "Daily benchmark total return minus the saved risk-free return",
             "ACTIVE_RETURN": "Daily portfolio total return minus benchmark total return"}
LINKED_STATUSES = {"LINKED_FULL_MODEL_COVERAGE", "LINKED_PARTIAL_MODEL_COVERAGE"}


def alpha_evidence(inference):
    frame = inference.loc[inference.term.eq("intercept")].copy()
    for field in ["coefficient", "hac_ci_lower", "hac_ci_upper"]:
        frame[field + "_daily_bp"] = frame[field] * 10000
    columns = ["target_id", "term", "coefficient_daily_bp", "hac_ci_lower_daily_bp", "hac_ci_upper_daily_bp",
               "hac_p_value", "hac_z_statistic", "hac_lags", "n_observations", "n_parameters", "residual_degrees_of_freedom",
               "hac_small_sample_adjustment", "interval_id", "result_row", "period_id", "model_id"]
    return frame[[column for column in columns if column in frame]]


def coefficient_chart(coefficients, inference, factors, available):
    if not available:
        st.info("Coefficient estimates are unavailable under the saved fit status.")
        return
    rows = coefficients.loc[coefficients.term.isin(factors)].copy()
    intervals = inference.loc[inference.term.isin(factors)]
    if rows.term.duplicated().any() or intervals.term.duplicated().any():
        st.error("Saved factor terms are not unique; inspect the complete inference evidence.")
        return
    rows = rows.merge(intervals[["term", "hac_ci_lower", "hac_ci_upper", "hac_p_value"]], on="term", how="left", validate="one_to_one")
    finite = np.isfinite(rows.coefficient)
    if not finite.any():
        st.info("No finite saved factor coefficients are available.")
        return
    rows["factor_label"] = rows.term.map(FACTOR_NAMES).fillna(rows.term)
    order = [FACTOR_NAMES.get(f, f) for f in factors]
    base = alt.Chart(rows.loc[finite]).encode(y=alt.Y("factor_label:N", sort=order, title=None, axis=alt.Axis(labelLimit=190)))
    tips = [alt.Tooltip("factor_label:N", title="Factor"), alt.Tooltip("coefficient:Q", title="Coefficient", format=".3f"),
            alt.Tooltip("hac_ci_lower:Q", title="HAC 95% lower", format=".3f"), alt.Tooltip("hac_ci_upper:Q", title="HAC 95% upper", format=".3f"),
            alt.Tooltip("hac_p_value:Q", title="HAC p-value", format=".4g")]
    interval = base.mark_rule(strokeWidth=3, color="#94A3B8").encode(x=alt.X("hac_ci_lower:Q", title="Direct coefficient · HAC 95% interval"), x2="hac_ci_upper:Q", tooltip=tips)
    points = base.mark_point(filled=True, size=80, color="#0F2A47").encode(x="coefficient:Q", tooltip=tips)
    zero = alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(color="#8293A8").encode(x="zero:Q")
    render_chart(styled((zero + interval + points).properties(height=max(260, len(factors) * 42))), alt="Direct factor coefficients with saved HAC 95% intervals and zero reference")
    interval_count = int((finite & np.isfinite(rows.hac_ci_lower) & np.isfinite(rows.hac_ci_upper)).sum())
    st.caption(f"{int(finite.sum())} of {len(factors)} saved factor coefficients available; {interval_count} have finite HAC intervals. Coefficients are dimensionless. Crossing zero reflects uncertainty; sign does not indicate investment quality. Daily alpha uses its own bp units above.")


def variance_chart(frame, available, target):
    if not available or len(frame) != 1:
        st.info("No eligible saved variance decomposition for this target.")
        return
    row = frame.iloc[0]
    values = [row.systematic_variance_share, row.residual_variance_share]
    if not np.isfinite(values).all() or abs(sum(values) - 1) > 1e-8:
        st.warning("The saved variance shares are unavailable or do not reconcile; inspect the evidence below.")
        return
    data = pd.DataFrame({"component": ["Systematic factors", "Residual"], "variance_share": values})
    chart = alt.Chart(data).mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y("component:N", sort=data.component.tolist(), title=None),
        x=alt.X("variance_share:Q", title="Share of regression-response variance", axis=alt.Axis(format=".0%", tickCount=6)),
        color=alt.Color("component:N", scale=alt.Scale(domain=data.component.tolist(), range=["#0F2A47", "#94A3B8"]), legend=None),
        tooltip=["component:N", alt.Tooltip("variance_share:Q", title="Variance share", format=".2%")],
    ).properties(height=150)
    render_chart(styled(chart), alt="Systematic and residual shares of the selected regression response variance")
    basis = "daily portfolio-minus-benchmark returns; response volatility is tracking error" if target == "ACTIVE_RETURN" else "daily excess returns over RF; response volatility can differ from total-return volatility"
    st.caption(f"In-sample decomposition of {basis}. Saved variance reconciliation error: {row.variance_reconciliation_error:+.3e} in daily variance units. This describes the fitted sample, rather than predictive accuracy.")


def linked_evidence(row, factors):
    components = [("Risk-free", "linked_risk_free_contribution"), ("Security-model intercept", "linked_estimated_intercept_contribution")]
    components += [(FACTOR_NAMES.get(f, f), "linked_factor_contribution_" + f) for f in factors]
    components += [("Modeled residual", "linked_modeled_residual_contribution"), ("Unmodeled excess", "linked_unmodeled_excess_contribution")]
    return pd.DataFrame({"component": [name for name, _ in components], "linked_contribution_pp": [row.get(field, float("nan")) * 100 for _, field in components],
                         "portfolio_id": row.portfolio_id, "link_status": row.link_status,
                         "realized_total_return_pp": row.realized_compounded_total_return * 100,
                         "saved_linked_total_pp": row.linked_attributed_total_return * 100,
                         "saved_reconciliation_error_pp": row.period_reconciliation_error * 100})


def bridge_rows(evidence):
    if evidence.empty or evidence.component.duplicated().any() or not evidence.link_status.isin(LINKED_STATUSES).all():
        raise ValueError("Linked components are unavailable under their saved status.")
    values = evidence.linked_contribution_pp.to_numpy()
    targets = evidence.realized_total_return_pp.unique()
    if len(targets) != 1 or not np.isfinite(values).all() or not np.isfinite(targets).all():
        raise ValueError("A unique finite saved return target and complete linked components are required.")
    if abs(values.sum() - targets[0]) > 1e-6:
        raise ValueError("Linked components do not reconcile to the saved total-return target.")
    rows, running = [], 0.
    for component, value in zip(evidence.component, values):
        rows.append({"component": component, "start_pp": running, "end_pp": running + value, "contribution_pp": value, "kind": "Positive" if value >= 0 else "Negative"})
        running += value
    rows.append({"component": "Total return", "start_pp": 0., "end_pp": targets[0], "contribution_pp": targets[0], "kind": "Total"})
    return pd.DataFrame(rows)


def linked_chart(evidence):
    try:
        rows = bridge_rows(evidence)
    except ValueError as error:
        st.warning(str(error))
        return
    chart = alt.Chart(rows).mark_bar(cornerRadius=3).encode(
        y=alt.Y("component:N", sort=rows.component.tolist(), title=None, axis=alt.Axis(labelLimit=200)),
        x=alt.X("end_pp:Q", title="Linked contribution to total return (pp)"), x2="start_pp:Q",
        color=alt.Color("kind:N", scale=alt.Scale(domain=["Positive", "Negative", "Total"], range=["#0D9488", "#D97706", "#0F2A47"]), legend=None),
        tooltip=[alt.Tooltip("component:N", title="Component"), alt.Tooltip("contribution_pp:Q", title="Contribution (pp)", format="+.2f")],
    ).properties(height=len(rows) * 36)
    render_chart(styled(chart), alt="Beginning-wealth linked factor contributions in pp reconciling to compounded total return")
    row = evidence.iloc[0]
    st.caption(f"Beginning-wealth linking · Saved linked total {row.saved_linked_total_pp:+.6f} pp · Realized total-return target {row.realized_total_return_pp:+.6f} pp · Saved residual {row.saved_reconciliation_error_pp:+.3e} pp.")
    st.caption("Teal adds to return, orange subtracts, navy is the total. These colors identify contribution sign. The linked security-model intercept aggregates prior security estimates and is distinct from the selected-period direct daily alpha.")


def coverage_evidence(exposure):
    records = []
    for owner, rows in exposure.groupby("portfolio_id", sort=False):
        full = int(rows.exposure_coverage_status.eq("FULLY_MODELED").sum())
        partial = int(rows.exposure_coverage_status.eq("PARTIALLY_MODELED").sum())
        records.append({"portfolio_id": owner, "daily_observations": len(rows), "fully_modeled_days": full, "partially_modeled_days": partial,
                        "other_coverage_status_days": len(rows) - full - partial, "minimum_modeled_gross_weight_coverage": rows.modeled_gross_weight_coverage.min(),
                        "factor_available_days": int(rows.factor_available_on_return_date.eq(True).sum())})
    return pd.DataFrame(records)


def exposure_chart(source, field, title, coverage=False):
    frame = source.copy()
    eligible = np.isfinite(frame[field])
    if not coverage:
        eligible &= frame.exposure_coverage_status.isin(["FULLY_MODELED", "PARTIALLY_MODELED"])
    frame.loc[~eligible, field] = float("nan")
    if not eligible.any():
        st.info("No eligible saved observations for " + title.lower() + ".")
        return
    owners = [owner for owner in SERIES if frame.loc[eligible, "portfolio_id"].eq(owner).any()]
    colors = [COLORS[SERIES.index(owner)] for owner in owners]
    tips = [alt.Tooltip("date:T", title="Return date"), alt.Tooltip("portfolio_id:N", title="Portfolio"),
            alt.Tooltip(f"{field}:Q", title=title, format=".2%" if coverage else ".3f")]
    if not coverage:
        tips.append(alt.Tooltip("modeled_gross_weight_coverage:Q", title="Modeled gross-weight coverage", format=".2%"))
    tips.append(alt.Tooltip("exposure_coverage_status:N", title="Saved coverage status"))
    chart = alt.Chart(frame).mark_line(strokeWidth=2.3).encode(
        x=date_encoding(frame), y=alt.Y(f"{field}:Q", title=title, scale=alt.Scale(domain=[0, 1]) if coverage else alt.Scale(zero=False), axis=alt.Axis(format=".0%" if coverage else ".2f")),
        color=alt.Color("portfolio_id:N", scale=alt.Scale(domain=owners, range=colors), title=None),
        tooltip=tips,
    ).properties(height=230 if coverage else 300)
    render_chart(styled(chart), alt=title + " for portfolio and benchmark; saved coverage retained")
    if (~eligible).any():
        st.caption(f"{int((~eligible).sum()):,} ineligible/missing observations retain gaps; no betas were filled with zero.")


def supporting_data(choices):
    st.subheader("Supporting chart data")
    selected = st.selectbox("Factors supporting data", list(choices))
    display_table(choices[selected], alt="Complete saved evidence for " + selected)


def remembered_choice(label, options, state_key, format_func=str, segmented=False):
    previous = st.session_state.get(state_key, options[0])
    previous = previous if previous in options else options[0]
    if segmented:
        selected = st.segmented_control(label, options, default=previous, format_func=format_func, key=state_key + "_choice") or previous
    else:
        selected = st.selectbox(label, options, index=options.index(previous), format_func=format_func, key=state_key + "_choice")
    st.session_state[state_key] = selected
    return selected
