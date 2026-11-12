"""Saved EOP holdings, benchmark comparisons and complete chart evidence."""
import pandas as pd
import streamlit as st

from research_ui import SERIES, context, pct, detailed, methodology, status_text
from exposure_details import (monthly_sectors, closing_positions, holdings_ranking, headline_comparison,
                              sector_history, tilt_chart, holdings_chart, active_share_chart, supporting_data)

ctx = context()
ctx.heading("Holdings & exposures", "Compare ownership, concentration and portfolio weight differences from the benchmark.")
st.caption("Hypothetical 20-security active strategy versus a custom Top-100 eligible-issuer equal-weight benchmark; both rebalance quarterly. Total returns assume frictionless same-close implementation. Sector classifications are fixed historical labels.")
summary = ctx.load("exposures", "portfolio_window_summary")
summary = summary.loc[summary.weight_basis.eq("eop")]
active = ctx.load("exposures", "active_window_summary")
active = active.loc[active.weight_basis.eq("eop")]
closing_date = pd.Timestamp(ctx.definition["last_return_date"])
st.markdown(f"**Closing snapshot: {closing_date:%d %B %Y} · end-of-period weights**")
headline_comparison(summary)
st.caption("These are endpoint observations: changing the starting month while keeping the same ending month leaves closing holdings unchanged. Effective holdings is inverse concentration, rather than the number of positions. Counts are observed invested holdings, not forced strategy targets. Selected-period medians remain separately labeled in supporting data.")
with st.container(horizontal=True):
    for title, field in [("Security Active Share", "latest_security_active_share"), ("Sector Active Share", "latest_sector_active_share"), ("Common-weight overlap", "latest_common_weight_overlap")]:
        row = active.iloc[0] if len(active) == 1 else None
        available = row is not None and row.availability_status == "AVAILABLE" and pd.notna(row[field])
        with st.container(border=True, width=220):
            st.metric(title, pct(row[field]) if available else "Unavailable")
            st.caption("Portfolio relative to benchmark · closing EOP")
            if not available:
                st.caption(status_text(row.availability_status) if row is not None else "No unique saved summary")

raw_sectors = ctx.history("exposures", "sector_exposures")
monthly_observed, monthly_plotted = pd.DataFrame(), pd.DataFrame()
with st.container(border=True):
    st.subheader("Portfolio and benchmark sector allocation over time")
    owner = st.segmented_control("Sector history", SERIES, default=SERIES[0]) or SERIES[0]
    try:
        monthly_observed, monthly_plotted = monthly_sectors(raw_sectors)
        sector_history(monthly_plotted, owner)
    except ValueError as error:
        st.warning(str(error))
    st.caption("Last saved EOP snapshot in each selected month; these are not monthly averages. Steps display discrete observations, rather than an inferred daily allocation path. Sector colors remain fixed. A sector absent from a fully invested, reconciled snapshot is displayed as zero; supporting data identifies these display rows separately.")

st.subheader("Closing sector overweights and underweights")
tilts = ctx.load("exposures", "sector_tilt_window_summary")
tilts = tilts.loc[tilts.weight_basis.eq("eop")].copy()
tilts["active_sector_weight_pp"] = tilts.latest_active_sector_weight * 100
tilts = tilts.drop(columns="latest_active_sector_weight")
if not tilts.empty:
    tilt_chart(tilts)
    st.caption(f"Portfolio minus benchmark · {closing_date:%d %B %Y} · Sum of sector differences {tilts.active_sector_weight_pp.sum():+.3e} pp. Navy means overweight; gray means underweight. The sign does not indicate investment performance or quality.")
else:
    st.info("No saved closing sector differences for this selection.")

st.subheader("Largest closing positions and remaining weight")
holdings_owner = st.segmented_control("Closing holdings portfolio", SERIES, default=SERIES[0]) or SERIES[0]
closing, balance = pd.DataFrame(), pd.DataFrame()
try:
    closing = closing_positions(ctx.history("exposures", "security_exposures"), closing_date)
    ranked, balance = holdings_ranking(closing, holdings_owner)
    holdings_chart(ranked, holdings_owner)
    with st.container(horizontal=True):
        for item in balance.itertuples():
            st.metric(f"{item.group} ({item.position_count})", pct(item.position_weight), border=True, width=240)
    st.caption(f"{holdings_owner} · {closing_date:%d %B %Y}. Top ten + Remaining = All invested positions. Weights are ownership shares, not return contributions. Tickers label separate stable security identities; missing tickers are labeled unavailable. Complete closing observations for both portfolios remain below, including saved invested flags.")
except ValueError as error:
    st.warning(str(error))

st.subheader("Portfolio weight differences from the benchmark")
share = ctx.history("exposures", "active_share")
share = share.loc[share.weight_basis.eq("eop")].copy()
share_basis = st.segmented_control("Active Share basis", ["Security", "Sector"], default="Security") or "Security"
field = "security_active_share" if share_basis == "Security" else "sector_active_share"
active_share_chart(share, field, share_basis + " Active Share (EOP)")
st.caption("One portfolio-versus-benchmark measure, not separate portfolio and benchmark return lines. Active Share measures weight differences, not performance or investment quality. The 0–100% scale is shared across security and sector views. Daily EOP observations start inside the selected period; no preceding-month holdings snapshot is invented.")

closing_sectors = raw_sectors.loc[raw_sectors.weight_basis.eq("eop") & pd.to_datetime(raw_sectors.date).eq(closing_date)]
supporting_data({
    "Closing and selected-period concentration summaries": summary,
    "Closing and selected-period Active Share summaries": active,
    "Monthly sector chart observations · both portfolios": monthly_plotted,
    "Monthly sector observations · saved rows only": monthly_observed,
    "Closing sector weights · both portfolios": closing_sectors,
    "Sector differences · closing pp and selected-period statistics": tilts,
    "Complete closing security observations · both portfolios": closing,
    "Selected top-ten, remaining and total weights": balance,
    "Daily EOP Active Share and overlap observations": share,
})
detailed(ctx, "exposures")
methodology(ctx, "All page charts use observed end-of-period (EOP) weights. Daily BOP and EOP histories retain their labels in all 17 detailed result choices. The opening-return-price rule does not imply an unobserved opening holdings snapshot. Concentration and Active Share summaries retain their saved availability; window medians are not endpoint values. Sector labels are fixed historical classifications. No analysis is run when choosing an owner or period.")
