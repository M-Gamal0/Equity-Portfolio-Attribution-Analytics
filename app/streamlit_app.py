"""Read-only portfolio research application; analysis is computed offline."""
import streamlit as st
from research_ui import select_context

st.set_page_config(page_title="Equity | Portfolio research", page_icon=":material/query_stats:", layout="wide")
st.set_option("client.disableDataExport", True)

page = st.navigation([
    st.Page("app_pages/overview.py", title="Overview", icon=":material/dashboard:"),
    st.Page("app_pages/performance.py", title="Performance", icon=":material/monitoring:", default=True),
    st.Page("app_pages/attribution.py", title="Attribution", icon=":material/account_tree:"),
    st.Page("app_pages/risk.py", title="Risk", icon=":material/shield:"),
    st.Page("app_pages/exposures.py", title="Holdings & exposures", icon=":material/donut_large:"),
    st.Page("app_pages/factors.py", title="Factors", icon=":material/hub:"),
    st.Page("app_pages/turnover.py", title="Turnover & rebalancing", icon=":material/swap_horiz:"),
    st.Page("app_pages/about.py", title="About the project", icon=":material/info:"),
])
st.session_state.research_context = select_context()
page.run()
