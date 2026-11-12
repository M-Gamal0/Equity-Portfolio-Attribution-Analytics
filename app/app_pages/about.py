"""Project purpose and the documented constructions behind the saved results."""
import streamlit as st

from research_ui import context


ctx = context()
ctx.heading("About the project", "A rules-based case study in portfolio analytics, attribution and validation.")
st.caption("The dates above identify the applied reporting period. Portfolio construction rules remain the same across selections.")

with st.container(border=True):
    st.subheader("An analytics project, not a performance claim")
    st.write(
        "This project demonstrates an end-to-end equity portfolio analytics workflow: "
        "constructing a benchmark and an active portfolio, calculating returns and risk, "
        "explaining performance differences, and validating the results. "
        "The active portfolio serves as a systematic test case for these calculations, "
        "rather than as evidence of a superior investment strategy."
    )
    st.write(
        "The project presents a consistent rules-based case study, rather than a portfolio "
        "chosen to showcase the strongest historical returns. Any observed outperformance "
        "is an outcome of the specified rules over the period examined, not the objective "
        "of the project or evidence of persistent investment skill. Results vary with the "
        "selected period; the full research history and all eligible month selections "
        "remain available for inspection."
    )

st.subheader("The two portfolio constructions")
with st.container(horizontal=True):
    with st.container(border=True, width=400):
        st.subheader("Custom equal-weight benchmark")
        st.write(
            "At each quarterly selection date, the benchmark selects the 100 largest "
            "eligible issuers by market capitalization from the historical S&P 500 universe, "
            "after consolidating multiple share classes and applying the price-history "
            "eligibility screen. Each selected security receives a 1% target weight."
        )
        st.write(
            "Market capitalization determines selection, not portfolio weights. "
            "Weights drift between quarterly rebalances. This is a custom research "
            "benchmark, not the official S&P 100 index or the S&P 500 index."
        )
    with st.container(border=True, width=400):
        st.subheader("Rules-based active portfolio")
        st.write(
            "The active portfolio selects 20 securities from the benchmark universe "
            "at each quarterly selection date. Trailing 12-month relative sector momentum "
            "informs sector allocations, while trailing 12-month Sortino ratios guide "
            "security selection within the allocation constraints."
        )
        st.write(
            "Positions are equally weighted within each selected sector, with individual "
            "target weights bounded between 1% and 10%. Sector allocations are constrained "
            "to a maximum active deviation of 15 percentage points from the benchmark. "
            "The portfolio therefore creates both sector-allocation and security-selection "
            "differences for the attribution workflow to explain."
        )

st.subheader("What the project demonstrates")
st.markdown(
    "- **Performance:** compounded returns, growth, drawdowns, rolling comparisons and risk-adjusted measures.\n"
    "- **Attribution:** sector allocation, selection and interaction effects; linked security contributions; Carino and Menchero reconciliation.\n"
    "- **Risk:** absolute and active risk, contribution analysis, covariance estimates and historical tail losses.\n"
    "- **Holdings and exposures:** ownership, concentration, sector tilts, overlap and Active Share.\n"
    "- **Factors:** direct regressions, lagged holdings-based exposures, return attribution and model coverage.\n"
    "- **Turnover and rebalancing:** trading requirements, weight drift and hypothetical transaction-cost sensitivity."
)
st.write(
    "The emphasis is on transparent calculations, consistent definitions, saved validation "
    "checks and traceable supporting evidence. Reporting uses precalculated results; "
    "changing dates does not rerun the analysis or change the portfolio's construction rules."
)

with st.expander("Research assumptions and interpretation"):
    st.write(
        "The research history covers October 2016 through September 2026. Selections contain "
        "at least 12 complete months. Returns include the first selected trading day's "
        "close-to-close movement, using the preceding month's last available close as the opening valuation."
    )
    st.write(
        "Both constructions are hypothetical, long-only, quarterly rebalanced total-return "
        "portfolios. Calculations assume frictionless same-close implementation, with "
        "dividends reinvested at index level. Base returns exclude trading costs; separate "
        "cost sensitivities are illustrative. These are historical simulations, not live "
        "realized investment results."
    )
    st.write(
        "Sector classifications are held fixed through history. The active sector signal "
        "evaluates trailing returns using the constituent basket selected at that quarter-end. "
        "These conventions, hypothetical execution timing and simplified terminal-event "
        "treatment limit real-world interpretation. Calculation checks establish internal "
        "consistency; they do not independently certify the underlying source data."
    )
