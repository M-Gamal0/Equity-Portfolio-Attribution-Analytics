"""Saved Brinson and security attribution with method-aligned evidence."""
import streamlit as st

from research_ui import context, pct, detailed, methodology, security_labels
from overview_details import attribution_evidence
from attribution_details import (waterfall_chart, signed_chart, component_matrix, bridge_chart,
                                 security_evidence, security_ranking, supporting_data)

ctx = context()
ctx.heading("Attribution", "Explain the allocation, selection and security contributions behind the return difference.")
st.caption("Hypothetical 20-security active strategy versus a custom Top-100 eligible-issuer equal-weight benchmark; both rebalance quarterly. Total returns · Hypothetical frictionless same-close implementation.")
previous = st.session_state.get("attribution_method", "carino")
method = st.segmented_control("Linking method", ["carino", "menchero"], default=previous,
                              format_func=str.title, key="attribution_method_choice") or previous
st.session_state.attribution_method = method
report = ctx.load("attribution", "attribution_period_report")
row = report.loc[report.linking_method.eq(method)].iloc[0]
sectors = ctx.load("attribution", "attribution_sector_report")
sectors = sectors.loc[sectors.linking_method.eq(method)]
securities = ctx.load("attribution", "security_contributions_period")
securities = security_evidence(security_labels(ctx, securities.loc[securities.linking_method.eq(method)]))
ranked, balance = security_ranking(securities)
reconciliation = ctx.load("attribution", "security_contributions_period_reconciliation")
with st.container(horizontal=True):
    st.metric("Portfolio return", pct(row.portfolio_compounded_return), border=True)
    st.metric("Benchmark return", pct(row.benchmark_compounded_return), border=True)
    st.metric("Return difference", f"{row.compounded_active_return * 100:+.2f} pp", border=True,
              help="Portfolio minus benchmark compounded return, not relative wealth.")
with st.container(border=True):
    st.subheader("Allocation, selection and interaction (pp)")
    waterfall_chart(row, method)
with st.container(border=True):
    st.subheader("Sector attribution to return difference (pp)")
    signed_chart(attribution_evidence(sectors), "sector", "brinson_total_pp", "Sector attribution to return difference (pp)")
    st.caption(f"{method.title()} linking · All sectors shown. Teal adds to the return difference; orange subtracts from it.")
with st.container(border=True):
    st.subheader("What drove each sector's effect?")
    component_matrix(sectors)
with st.container(border=True):
    st.subheader("Security contributors and detractors (pp)")
    signed_chart(ranked, "security_label", "active_contribution_pp", "Linked active security contribution (pp)", security=True)
    st.caption(f"{method.title()} linking · Up to five positive contributors and five detractors. Tickers are the latest observed within the selection; distinct security identities remain separate. Contributions compare portfolio and benchmark legs, rather than standalone stock returns. Hover for holding context, including benchmark-only securities.")
    benchmark_only = ranked.loc[ranked.holding_context.eq("Benchmark only during selection"), "security_label"].tolist()
    if benchmark_only:
        st.caption("Benchmark-only names in this ranking: " + ", ".join(benchmark_only) + ". Their active contributions reflect the benchmark leg for securities not held by the portfolio during this selection.")
    with st.container(horizontal=True):
        for item in balance.itertuples():
            st.metric(f"{item.group} ({item.security_count})", f"{item.active_contribution_pp:+.2f} pp", border=True)
    if abs(balance.active_contribution_pp.iloc[2] - row.compounded_active_return * 100) > 1e-6:
        st.error("The saved security total does not reconcile to the return-difference target. Inspect the saved checks below.")
    st.caption("Ranked securities + Other securities = All securities. Other securities includes every historical security omitted from the ranking; the full universe is available below. Teal/orange indicate contribution sign, not portfolio identity.")
with st.container(border=True):
    st.subheader("Why the two sector views differ")
    bridge_chart(sectors)

supporting_data(method, report, sectors, securities, balance, reconciliation)
st.caption("Detailed results below retain all original result choices. Their independent Linking method filter can explicitly show both methods; the supporting chart data above follows the selected method.")
detailed(ctx, "attribution")
methodology(ctx, "All chart effects are percentage points. Carino and Menchero link to compounded portfolio return minus compounded benchmark return; component allocations can differ. Security contributions equal sector Brinson effects plus their saved benchmark-reference bridge when aggregated by sector. Sector classifications are fixed for history; execution and terminal-value mechanics are hypothetical. The preceding-month opening close and first selected session remain included.")
