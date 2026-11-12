"""Overview evidence and risk context from immutable selected-period outputs."""
import altair as alt
import pandas as pd
import streamlit as st

from chart_renderer import render_chart
from research_ui import SERIES, display_table, pct, status_text, styled


def attribution_evidence(frame, *, period=False):
    """Present saved pp columns, falling back to unit conversion for old contracts."""
    result = frame.copy()
    fields = {"allocation_pp": "linked_allocation_effect", "selection_pp": "linked_selection_effect",
              "interaction_pp": "linked_interaction_effect",
              "total_attribution_pp" if period else "brinson_total_pp": "linked_total_effect"}
    if period:
        fields["active_return_pp"] = "compounded_active_return"
    for target, source in fields.items():
        if target not in result and source in result:
            result[target] = result[source] * 100
    if "reconciliation_error" in result:
        result["reconciliation_error_pp"] = result.reconciliation_error * 100
    columns = (["sector"] if not period else []) + list(fields)
    columns += ["reconciliation_error_pp", "coverage_status", "linking_method", "period_start", "period_end", "valuation_anchor_date"]
    return result[[c for c in columns if c in result]]


def active_risk_values(active_risk, ratios):
    """Do not expose a numeric estimate when its saved eligibility status fails."""
    row = active_risk.iloc[0]
    te = row.get("tracking_error_annualized") if row.get("tracking_error_status") == "PASS" else None
    matches = ratios.loc[ratios.metric_name.eq("information_ratio")]
    ir = matches.iloc[0] if not matches.empty else None
    value = ir.metric_value if ir is not None and ir.calculation_status == "CALCULATED" else None
    return te, value, row.get("tracking_error_status", "Unavailable"), ir.calculation_status if ir is not None else "Unavailable"


def risk_headlines(risk, active_risk, ratios):
    with st.container(horizontal=True, vertical_alignment="distribute"):
        for title, field in [("Annualized volatility", "annualized_volatility"), ("Maximum drawdown", "maximum_drawdown")]:
            with st.container(border=True):
                st.markdown(f"**{title} · selected period**")
                with st.container(horizontal=True):
                    for owner in SERIES:
                        st.metric(owner, pct(risk.loc[owner, field]))
    te, ir, te_status, ir_status = active_risk_values(active_risk, ratios)
    with st.container(horizontal=True):
        st.metric("Annualized tracking error", pct(te) if te is not None else "Unavailable", border=True,
                  help="Selected-period volatility of daily portfolio-minus-benchmark returns, annualized.")
        st.metric("Information ratio", f"{ir:.3f}" if ir is not None and pd.notna(ir) else "Unavailable", border=True,
                  help="Saved selected-period annualized mean active return divided by tracking error.")
    st.caption("Active risk versus the benchmark · Same applied period. Numerical differences are not automatically favorable or unfavorable.")
    if te is None or pd.isna(te):
        st.caption("Tracking error unavailable: " + status_text(te_status))
    if ir is None or pd.isna(ir):
        st.caption("Information ratio unavailable: " + status_text(ir_status))


def sector_chart(sectors):
    evidence = attribution_evidence(sectors)
    chart = alt.Chart(evidence).mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y("sector:N", sort="-x", title=None, axis=alt.Axis(labelLimit=240)),
        x=alt.X("brinson_total_pp:Q", title="Sector attribution to return difference (pp)",
                axis=alt.Axis(format=".1f", tickCount=6)),
        color=alt.condition(alt.datum.brinson_total_pp >= 0, alt.value("#0D9488"), alt.value("#D97706")),
        tooltip=[alt.Tooltip("sector:N", title="Sector"),
                 alt.Tooltip("brinson_total_pp:Q", title="Return difference (pp)", format="+.2f"),
                 alt.Tooltip("allocation_pp:Q", title="Allocation (pp)", format="+.2f"),
                 alt.Tooltip("selection_pp:Q", title="Selection (pp)", format="+.2f"),
                 alt.Tooltip("interaction_pp:Q", title="Interaction (pp)", format="+.2f"),
                 alt.Tooltip("linking_method:N", title="Linking method")],
    ).properties(height=max(340, len(evidence) * 30))
    render_chart(styled(chart), alt="Carino sector attribution to return difference in percentage points; all sectors")


def supporting_data(ctx, sectors, linked, paths, raw_paths):
    st.subheader("Supporting chart data")
    choice = st.selectbox("Supporting data", ["Sector attribution (pp)", "Linked attribution and reconciliation (pp)",
                                               "Growth and drawdown observations", "Saved drawdown observations",
                                               "Active-risk and ratio statuses"], key="overview_supporting_data")
    if choice == "Sector attribution (pp)":
        frame = attribution_evidence(sectors)
        st.caption("All sectors · Carino linking · Allocation + selection + interaction = sector total. Effects are percentage points (pp).")
    elif choice == "Linked attribution and reconciliation (pp)":
        frame = attribution_evidence(linked, period=True)
        st.caption("Carino linked components and saved target · All return differences, effects and reconciliation errors below are in pp.")
    elif choice == "Growth and drawdown observations":
        frame = paths
        st.caption("Every plotted observation, including the labeled preceding-month opening close: growth starts at 100 and opening drawdown at zero. Drawdown is shown as a percentage.")
    elif choice == "Saved drawdown observations":
        frame = raw_paths
        st.caption("All original saved drawdown fields and in-period observations, without the added opening display anchor.")
    else:
        st.markdown("**Saved active-risk summary**")
        display_table(ctx.load("risk", "risk_active_summary"), alt="Overview saved active-risk values and eligibility statuses")
        frame = ctx.load("performance", "performance_risk_adjusted")
        st.caption("Saved metric values, calculation statuses and sample requirements. Unavailable estimates are not replaced by zero.")
    display_table(frame, alt="Overview supporting data: " + choice)
    st.caption(f"{len(frame):,} {'row' if len(frame) == 1 else 'rows'} available · Full data retained; scroll within the table to inspect every row.")
