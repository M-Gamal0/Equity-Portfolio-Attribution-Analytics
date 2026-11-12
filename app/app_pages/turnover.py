"""Measured recurring events, pending decisions and illustrative unit-rate costs."""
import pandas as pd
import streamlit as st

from research_ui import SERIES, COLORS, context, pct, detailed, methodology, security_labels, status_text
from factor_details import remembered_choice
from turnover_details import (measured_events, event_scope, paired_headlines, event_chart, turnover_ranking,
                              concentration_chart, cost_scenarios, cost_chart, supporting_data)

ctx = context()
ctx.heading("Turnover & rebalancing", "Compare recurring portfolio activity, its sources and hypothetical cost sensitivity.")
st.caption("Hypothetical 20-security active strategy versus a custom Top-100 eligible-issuer equal-weight benchmark; both rebalance quarterly. Returns use frictionless same-close implementation. Turnover measures modeled NAV-weight changes, not actual fills or currency traded.")
summary = ctx.load("turnover", "portfolio_turnover_summary")
paired_headlines(summary)
st.caption("Selected-period sums of measured recurring activity. Cumulative turnover may exceed 100%. Gross two-way turnover is purchases plus sales; one-way turnover is half that amount. Initial establishment and pending decisions are separate.")
events = ctx.load("turnover", "rebalance_turnover_events")
measured = measured_events(events)
scope = event_scope(events)
for row in scope.itertuples():
    st.caption(f"{row.portfolio_id}: {row.measured_rebalances:,} eligible measured rebalances · {row.initial_establishment_events:,} initial establishment · {row.pending_events:,} pending · {row.other_or_ineligible_events:,} other/ineligible. Pending measurements remain unavailable.")
with st.container(border=True):
    st.subheader("Measured one-way turnover at each rebalance")
    event_chart(measured, "one_way_turnover", "portfolio_id", SERIES, COLORS, "One-way NAV turnover")
st.caption("Events belong by effective date. Prior-month selections effective inside the chosen interval are included; pending decisions at calendar month-end are not measured trades.")
owner = remembered_choice("Flow and concentration portfolio", SERIES, "turnover_owner", segmented=True)
st.subheader("Purchases and sales at measured rebalances")
flows = measured.loc[measured.portfolio_id.eq(owner)].melt(id_vars="effective_rebalance_date", value_vars=["total_purchases", "total_sales"], var_name="flow", value_name="nav_weight")
flows["flow"] = flows.flow.map({"total_purchases": "Purchases", "total_sales": "Sales"})
event_chart(flows, "nav_weight", "flow", ["Purchases", "Sales"], ["#0F2A47", "#94A3B8"], "Purchases / sales as NAV weight")
st.caption(f"{owner} · Navy identifies purchases, gray identifies sales. Both are positive traded amounts; these colors describe flows, rather than portfolio identity. No initial or pending trade amounts are filled with zero.")

st.subheader("Top ten sources of cumulative one-way turnover")
names = ctx.load("turnover", "cumulative_security_turnover_concentration")
names = security_labels(ctx, names)
selected_names = names.loc[names.portfolio_id.eq(owner)]
rows = summary.loc[summary.portfolio_id.eq(owner)]
balance = pd.DataFrame()
try:
    target = rows.iloc[0].cumulative_one_way_turnover if len(rows) == 1 else float("nan")
    ranked, balance = turnover_ranking(selected_names, target)
    concentration_chart(ranked, owner)
    with st.container(horizontal=True):
        for item in balance.itertuples():
            st.metric(f"{item.group} ({item.security_count})", pct(item.turnover_share), border=True, width=240)
    st.caption(f"Full security amount {selected_names.cumulative_one_way.sum():.6f} NAV units · Saved owner total {target:.6f} · Display residual {selected_names.cumulative_one_way.sum() - target:+.3e} NAV units. Top ten + Remaining = All securities.")
except ValueError as error:
    st.info(str(error))
st.caption("These are shares of cumulative turnover, not position weights or return contributions. Ticker labels retain separate stable security identities. Complete source amounts, historical names and remaining balances are available below.")

st.subheader("Hypothetical trading-cost sensitivity · both portfolios")
costs = ctx.load("turnover", "portfolio_cost_sensitivity")
for row in costs.itertuples():
    st.caption(f"{row.portfolio_id}: {status_text(row.cost_sensitivity_status)} · Saved 1 bp coefficient {row.nav_bp_per_1bp_traded:.6f} arithmetic NAV bp per 1 bp charged on purchased AND sold weight.")
scenarios = cost_scenarios(costs)
cost_chart(scenarios)
st.caption("Arithmetic scaling of saved gross two-way traded-weight sensitivity at 1, 5, 10 and 25 bp. Event NAV-normalized exposures are summed without compounding. This is neither realized expense nor a terminal-return deduction, net return or revised CAGR. Initial/pending events and passive price drift or terminal model adjustments are not added to measured trade costs.")
diagnostics = summary[[c for c in ["portfolio_id", "cumulative_price_drift_magnitude", "terminal_adjustment_days", "terminal_zeroed_securities", "cumulative_terminal_adjustment_magnitude"] if c in summary]]
supporting_data({
    "Selected-period turnover comparison · both portfolios": summary,
    "Measured, initial, pending and other event counts": scope,
    "All saved event rows · including pending and initial": events,
    "Eligible measured event observations · both portfolios": measured,
    "Selected-owner purchase and sale chart data": flows,
    "All security turnover concentrations · both portfolios": names,
    "Selected-owner top-ten, remaining and all balances": balance,
    "Saved unit-rate cost sensitivity · both portfolios": costs,
    "Hypothetical arithmetic cost scenarios · both portfolios": scenarios,
    "Separate passive drift and terminal model diagnostics": diagnostics,
})
detailed(ctx, "turnover")
methodology(ctx, "Measured recurring events require effective rebalance date, passed saved status and finite measurements. Initial establishment, pending calendar-end decisions and other unassessed events remain separately labeled, with blank measurements. Implemented membership uses observed effective dates; pending membership uses full calendar-month boundaries. One-way turnover is half gross purchases plus sales. Linear cost scenarios use both traded sides and never alter saved returns. All 27 detailed choices remain; passive drift and terminal adjustments remain distinct diagnostics.")
