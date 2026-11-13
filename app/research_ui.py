"""Presentation helpers only. Never invoke an analytics runner from this module."""
import calendar
import json
import os
from dataclasses import dataclass
from pathlib import Path

import altair as alt
import duckdb
import pandas as pd
import streamlit as st
from chart_renderer import render_chart

from portfolio_analytics.common.month_windows import MonthWindow
from portfolio_analytics.database.month_bundles import BundleReader, bundle_source_from_environment
from portfolio_analytics.database.month_results import list_month_builds, read_month_result, read_month_history

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = ROOT / "local/analytics_outputs/month_windows.duckdb"
def _configured_bundle_source():
    """Hosted deployments read exported bundles (local folder or hf://datasets/owner/name).

    Streamlit copies root-level secrets into os.environ only once secrets are loaded, so
    load them first; this also exposes HF_TOKEN to the Hugging Face client. Without a
    secrets file this does nothing and the local DuckDB files are used.
    """
    st.secrets.load_if_toml_exists()
    return bundle_source_from_environment()


BUNDLE_SOURCE = _configured_bundle_source()
SERIES = ["Active portfolio", "Benchmark"]
COLORS = ["#0D9488", "#D97706"]
FACTOR_NAMES = {"mkt_rf": "Market", "smb": "Size", "hml": "Value", "rmw": "Profitability", "cma": "Investment", "mom": "Momentum", "intercept": "Daily alpha"}


def status_text(value):
    labels = {"LINKED_PARTIAL_MODEL_COVERAGE": "Linked with partial model coverage",
              "LOW_EFFECTIVE_TAIL_SAMPLE": "Too few effective tail observations",
              "UNIT_RATE_SENSITIVITY_ONLY": "Illustrative unit-rate sensitivity",
              "CALCULATED": "Available", "MEASURED": "Measured", "FIT_COMPLETED": "Fit available"}
    return labels.get(str(value), str(value).replace("_", " ").capitalize())


def date_axis(frame, column="date"):
    """Years only for long histories; month names and a separate year caption otherwise."""
    dates = pd.to_datetime(frame[column]).dropna()
    if dates.empty:
        return alt.Axis(format="%Y", title=None)
    first, last = dates.min(), dates.max()
    if (last - first).days > 730:
        ticks = pd.date_range(f"{first.year}-01-01", last, freq="YS")
        ticks = ticks[ticks >= first]
        return alt.Axis(values=[d.isoformat() for d in ticks], format="%Y", title=None, labelAngle=0)
    months = pd.date_range(first.normalize().replace(day=1), last, freq="MS")
    months = months[months >= first]
    if len(months) > 14:
        months = months[::2]
    # A preceding-month opening close may lie in December of the prior year.
    # The separate year label describes the displayed monthly tick sequence.
    first_year = months[0].year if len(months) else first.year
    years = str(first_year) if first_year == last.year else f"{first_year}–{last.year}"
    if first_year != last.year and len(months):
        # Guarantee a year marker even when two-month sampling would skip January.
        boundaries = pd.date_range(f"{first_year}-01-01", last, freq="YS")
        months = months.union(boundaries[(boundaries >= months[0]) & (boundaries <= last)])
        return alt.Axis(values=[d.isoformat() for d in months], labelAngle=0, title=None,
                        labelLineHeight=14,
                        labelExpr="[timeFormat(toDate(datum.value), '%b'), "
                        "month(toDate(datum.value)) === 0 "
                        "? timeFormat(toDate(datum.value), '%Y') : '']")
    return alt.Axis(values=[d.isoformat() for d in months], format="%b", labelAngle=0,
                    title=years, titlePadding=14, titleFontWeight=500)


def date_encoding(frame, column="date"):
    return alt.X(f"{column}:T", axis=date_axis(frame, column))


def display_table(frame, *, alt, height=420):
    """Retain saved observations and numbers; publish ticker identities only."""
    frame = public_security_frame(frame, st.session_state.get("research_context"))
    identity = [c for c in ("portfolio_id", "target_id", "ticker", "security_id", "sector", "term", "linking_method", "weight_basis", "date", "estimation_date") if c in frame]
    preferred = [c for c in ("period_return", "annualized_return_cagr", "annualized_volatility", "tracking_error_annualized", "maximum_drawdown", "coefficient", "r_squared", "adjusted_r_squared", "regression_observations", "fit_status", "metric_name", "metric_value", "metric_units", "confidence_level", "var_loss_daily", "expected_shortfall_loss_daily", "coverage_status", "contribution_to_annualized_volatility", "contribution_to_annualized_risk", "risk_share", "risk_share_of_volatility", "latest_invested_security_count", "latest_security_active_share", "latest_effective_holdings", "latest_top_five_weight", "measured_rebalances", "cumulative_one_way_turnover", "average_rebalance_turnover", "nav_bp_per_1bp_traded") if c in frame]
    metadata_columns = {"interval_id", "result_row", "window_id", "period_id", "report_window", "model_id"}
    order = list(dict.fromkeys(identity + preferred + [c for c in frame if c not in metadata_columns] + [c for c in frame if c in metadata_columns]))
    config = {}
    for col in frame:
        label = {"portfolio_id": "Portfolio", "target_id": "Target", "security_id": "Security ID", "ticker": "Ticker", "annualized_return_cagr": "CAGR", "r_squared": "R-squared", "adjusted_r_squared": "Adjusted R-squared"}.get(col, col.replace("_", " ").capitalize())
        if pd.api.types.is_datetime64_any_dtype(frame[col]):
            config[col] = st.column_config.DateColumn(label, format="YYYY-MM-DD")
        elif pd.api.types.is_numeric_dtype(frame[col]) and not pd.api.types.is_bool_dtype(frame[col]):
            if "error" in col or "variance" in col or col.endswith("p_value"):
                fmt = "%.6e"
            elif col.endswith("_pp"):
                fmt = "%.2f pp"
            elif (any(x in col for x in ("return", "drawdown", "volatility", "tracking_error", "weight", "capture", "turnover"))
                  or col.startswith("linked_") or col.endswith("_share")) and not any(x in col for x in ("observations", "days", "sessions", "count", "sum_daily", "pct", "variance", "bp", "duration")):
                fmt = "percent"
            elif pd.api.types.is_integer_dtype(frame[col]):
                fmt = "%d"
            else:
                fmt = "%.6g"
            config[col] = st.column_config.NumberColumn(label, format=fmt)
        else:
            config[col] = label
    st.dataframe(frame, hide_index=True, height=height, column_order=order, column_config=config, alt=alt)


def security_labels(ctx, frame):
    """Use the latest observed ticker within the selected period, keyed by stable ID."""
    frame = frame.copy()
    try:
        observed = ctx.history("exposures", "security_exposures")
        observed = observed.dropna(subset=["ticker"]).sort_values("date")
        latest = observed.drop_duplicates("security_id", keep="last").set_index("security_id").ticker
        tickers = frame.security_id.map(latest)
        frame["security_label"] = tickers.fillna("Ticker unavailable")
    except (OSError, KeyError, ValueError, duckdb.Error):
        frame["security_label"] = "Ticker unavailable"
    return frame


@st.cache_data(max_entries=32, show_spinner=False)
def security_ticker_lookup(path, build_id, iid):
    """Server-only identity map, shared by tables and filters for this selection."""
    observed = history(path, build_id, iid, "exposures", "security_exposures")
    observed = observed.dropna(subset=["ticker"]).sort_values("date", kind="stable")
    observed = observed.loc[observed.ticker.astype(str).str.strip().ne("")]
    return observed.drop_duplicates("security_id", keep="last").set_index("security_id").ticker.to_dict()


def security_display_labels(ctx, ids):
    """Disambiguate repeated/missing tickers without exposing provider IDs."""
    lookup = security_ticker_lookup(ctx.path, ctx.build_id, ctx.iid) if ctx is not None else {}
    lookup = {str(key): str(value) for key, value in lookup.items()}
    requested = ids.dropna().astype(str).unique()
    for key in requested:
        lookup.setdefault(key, "Ticker unavailable")
    grouped = {}
    for key, ticker in sorted(lookup.items()):
        grouped.setdefault(ticker, []).append(key)
    labels = {}
    for ticker, keys in grouped.items():
        for number, key in enumerate(keys, 1):
            labels[key] = ticker if len(keys) == 1 else f"{ticker} (security {number})"
    return labels


def public_security_frame(frame, ctx=None):
    """Create a display copy; never mutate the saved data or identity keys."""
    frame = frame.copy()
    id_columns = [c for c in ("security_id", "other_security_id") if c in frame]
    labels = security_display_labels(ctx, pd.concat([frame[c] for c in id_columns], ignore_index=True)) if id_columns else {}
    for source, target in (("security_id", "ticker"), ("other_security_id", "other_ticker")):
        if source not in frame:
            continue
        tickers = frame[source].astype(str).map(labels).fillna("Ticker unavailable")
        if target in frame:
            existing = frame[target].notna() & frame[target].astype(str).str.strip().ne("")
            # Preserve observed historical tickers when they are already saved.
            frame[target] = frame[target].where(existing, tickers)
        else:
            frame[target] = tickers
    return frame.drop(columns=[c for c in ("security_id", "other_security_id", "security_key", "category_key") if c in frame]).reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def bundle_reader(source):
    return BundleReader(source, revision=os.environ.get("PORTFOLIO_BUNDLE_REVISION") or None)


def _bundle(path):
    """The bundle reader when this path is the configured bundle source, else None."""
    return bundle_reader(path) if BUNDLE_SOURCE and path == BUNDLE_SOURCE else None


@st.cache_data(ttl=60, max_entries=4, show_spinner=False)
def builds(path):
    reader = _bundle(path)
    return reader.builds() if reader else list_month_builds(path)


@st.cache_data(max_entries=4, show_spinner=False)
def metadata(path, build_id):
    reader = _bundle(path)
    if reader:
        return reader.metadata(build_id)
    with duckdb.connect(path, read_only=True) as db:
        catalog = db.execute("SELECT * FROM control.month_outputs WHERE build_id=?", [build_id]).df()
        intervals = db.execute("SELECT interval_id,definition_json,validation_json FROM control.month_intervals WHERE build_id=? AND status='COMPLETED'", [build_id]).df()
    return catalog, intervals


@st.cache_data(max_entries=128, show_spinner=False)
def result(path, build_id, interval_id, domain, name):
    start, end = interval_id.split("_")
    window = MonthWindow(pd.Period(start, "M"), pd.Period(end, "M"))
    reader = _bundle(path)
    if reader:
        return reader.result(build_id, window, domain, name)
    return read_month_result(path, build_id, window, domain, name)


@st.cache_data(max_entries=16, show_spinner=False)
def history(path, build_id, interval_id, domain, name, date_column="date"):
    start, end = interval_id.split("_")
    window = MonthWindow(pd.Period(start, "M"), pd.Period(end, "M"))
    reader = _bundle(path)
    if reader:
        return reader.history(build_id, window, domain, name, date_column=date_column)
    return read_month_history(path, build_id, window, domain, name, date_column=date_column)


@dataclass
class ResearchContext:
    path: str
    build_id: str
    iid: str
    settings: dict
    catalog: pd.DataFrame
    definition: dict
    validations: list
    snapshot_id: str

    def load(self, domain, name):
        return label_frame(result(self.path, self.build_id, self.iid, domain, name), self.settings)

    def history(self, domain, name, date_column="date"):
        return label_frame(history(self.path, self.build_id, self.iid, domain, name, date_column), self.settings)

    def heading(self, title, subtitle):
        st.caption("EQUITY RESEARCH / " + title.upper())
        st.title(title)
        st.write(subtitle)
        start, end = (pd.Period(x, "M") for x in self.iid.split("_"))
        st.subheader(f"{start.strftime('%B %Y')} — {end.strftime('%B %Y')}")
        st.caption(f"{self.definition['inclusive_months']} complete months · {self.definition['observation_count']:,} trading sessions · Active portfolio versus benchmark")


def select_context():
    """Global widgets live in the entry point, so page navigation retains edits."""
    path = BUNDLE_SOURCE or os.environ.get("PORTFOLIO_MONTH_DATABASE", str(DEFAULT_DATABASE))
    try:
        available = builds(path)
        if available.empty:
            st.info("Your workspace is ready. Complete the offline month population to view performance here.")
            st.stop()
        record = available.iloc[0]
        catalog, intervals = metadata(path, record.build_id)
    except (OSError, ValueError, duckdb.Error) as exc:
        st.error(f"Saved month results could not be opened: {exc}")
        st.stop()
    if "month_window" not in st.session_state:
        st.session_state.month_window = "2016-10_2026-09"
    # Keep draft values independently of widget lifecycle and the applied range.
    start, end = [pd.Period(p, "M") for p in st.session_state.month_window.split("_")]
    defaults = {"start_month": calendar.month_name[start.month], "start_year": start.year,
                "end_month": calendar.month_name[end.month], "end_year": end.year}
    for name, default in defaults.items():
        st.session_state.setdefault("draft_" + name, default)
    with st.sidebar:
        st.caption("PORTFOLIO RESEARCH")
        st.subheader("Choose your period")
        st.caption("Active portfolio · Equal-weight benchmark")
        with st.form("month_selection"):
            c1, c2 = st.columns([3, 2])
            sm = c1.selectbox("Beginning month", list(calendar.month_name)[1:], index=list(calendar.month_name)[1:].index(st.session_state.draft_start_month), key="start_month")
            sy = c2.selectbox("Beginning year", list(range(2016, 2027)), index=st.session_state.draft_start_year - 2016, key="start_year")
            c1, c2 = st.columns([3, 2])
            em = c1.selectbox("Ending month", list(calendar.month_name)[1:], index=list(calendar.month_name)[1:].index(st.session_state.draft_end_month), key="end_month")
            ey = c2.selectbox("Ending year", list(range(2016, 2027)), index=st.session_state.draft_end_year - 2016, key="end_year")
            applied = st.form_submit_button("Apply period", type="primary", width="stretch")
        for name, value in zip(defaults, (sm, sy, em, ey)):
            st.session_state["draft_" + name] = value
        st.caption("October 2016–September 2026 · At least 12 complete months. January–December is 12 months.")
        if applied:
            try:
                candidate = MonthWindow.from_names(sm, sy, em, ey)
                if candidate.interval_id not in set(intervals.interval_id):
                    raise ValueError("This period is not available in the completed build.")
                st.session_state.month_window = candidate.interval_id
            except ValueError as exc:
                st.error(str(exc))
        st.divider()
        st.caption("Saved results only. Your selection does not start a new analysis.")
    iid = st.session_state.month_window
    selected = intervals.loc[intervals.interval_id.eq(iid)]
    if selected.empty:
        st.error("The applied period is unavailable in this saved build.")
        st.stop()
    row = selected.iloc[0]
    return ResearchContext(path, record.build_id, iid, json.loads(record.config_json), catalog,
                           json.loads(row.definition_json), json.loads(row.validation_json), record.snapshot_id)


def context():
    return st.session_state.research_context


def methodology(ctx, note=""):
    with st.expander("Period boundaries, methodology and validation"):
        st.write("Opening valuation uses the last available preceding-month close, including the first selected trading day's earned return. USD research assumption · Fixed historical sector labels · Hypothetical same-close implementation.")
        if note:
            st.write(note)
        st.json({"opening_close": ctx.definition["opening_valuation_date"], "first_return": ctx.definition["first_return_date"],
                 "last_return": ctx.definition["last_return_date"], "build_id": ctx.build_id, "snapshot_id": ctx.snapshot_id})
        st.dataframe(pd.DataFrame(ctx.validations), hide_index=True, alt="Saved selected-period validation results")


def detail_options(catalog, domain):
    """Map all new summaries/estimators and safe dated histories, never old summaries."""
    rows = []
    for r in catalog.loc[catalog.domain.eq(domain)].itertuples():
        fields = json.loads(getattr(r, "schema_json", "{}"))
        date = next((c for c in ("date", "effective_rebalance_date", "estimation_date") if c in fields), None)
        if r.scope != "SOURCE_REFERENCE" or (date and not {"period_id", "window_id", "report_window"}.intersection(fields)):
            rows.append((r.output_name, r.scope, date))
    return sorted(rows, key=lambda x: (x[1] == "SOURCE_REFERENCE", x[0], x[1]))


def detailed(ctx, domain):
    st.divider()
    st.subheader("Detailed results")
    st.caption("Complete saved results, grouped by reporting scope. Decimal returns and weights of 0.01 mean 1%. Availability statuses and original precision remain visible.")
    options = detail_options(ctx.catalog, domain)
    preferred = {"performance": "performance_window", "attribution": "attribution_period_report",
                 "risk": "risk_absolute_summary", "exposures": "portfolio_window_summary",
                 "factors": "direct_portfolio_factor_fit", "turnover": "portfolio_turnover_summary"}
    options.sort(key=lambda x: x[0] != preferred.get(domain) or x[1] != "PERIOD")
    scope_names = {"PERIOD": "Selected period", "CLOSING_MONTH": "Ending-month estimator", "SOURCE_REFERENCE": "Dated shared history"}
    option = st.selectbox("Result table", options, format_func=lambda x: f"{x[0].replace('_', ' ').capitalize()} · {scope_names[x[1]]}", key=f"detail_{domain}")
    if option is None:
        st.info("No saved tables are available for this page.")
        return
    name, scope, date = option
    if scope == "SOURCE_REFERENCE":
        st.caption("History endpoints are inside your selection. Rolling estimates can use observations before the selected beginning; these retain their original estimation lookback.")
        st.caption("Cumulative shared histories retain their original inception anchor. Filtering dates does not rebase them to the selected opening close.")
    elif scope == "CLOSING_MONTH":
        st.caption("Closing weights and covariance use the saved ending-month estimator and 252-session lookback, independently of the reporting start.")
    try:
        frame = ctx.history(domain, name, date) if scope == "SOURCE_REFERENCE" else ctx.load(domain, name)
    except (OSError, ValueError, KeyError, duckdb.Error) as exc:
        st.warning(f"This saved table could not be opened: {exc}")
        return
    for field in ("portfolio_id", "target_id", "weight_basis", "linking_method", "sector", "security_id"):
        if field in frame and frame[field].nunique() > 1:
            if field == "security_id":
                labels = security_display_labels(ctx, frame[field])
                options = sorted({labels[str(value)] for value in frame[field].dropna().unique()})
                choice = st.selectbox("Security (ticker)", ["All", *options], key=f"detail_{domain}_{name}_{scope}_{field}")
                if choice != "All":
                    frame = frame.loc[frame[field].astype(str).map(labels).eq(choice)]
                continue
            choice = st.selectbox({"portfolio_id": "Portfolio", "target_id": "Target", "security_id": "Security"}.get(field, field.replace("_", " ").capitalize()), ["All", *sorted(frame[field].dropna().astype(str).unique())], key=f"detail_{domain}_{name}_{scope}_{field}")
            if choice != "All":
                frame = frame.loc[frame[field].astype(str).eq(choice)]
    display_table(frame, alt=f"Saved {domain} {name.replace('_', ' ')} results")
    st.caption(f"{len(frame):,} rows shown · {len(options)} result tables available.")


def metrics(items):
    # Five metrics form a deliberate 3 + 2 grid instead of a stretched 4 + 1.
    sizes = [3, 2] if len(items) == 5 else [len(items)]
    offset = 0
    for count in sizes:
        for col, (label, value, help_text) in zip(st.columns(count), items[offset:offset + count]):
            col.metric(label, value, border=True, help=help_text)
        offset += count


def bars(frame, label, value, title, percent=True, height=340, series=None):
    """Readable signed bars, with original values available in tooltips/tables."""
    frame = frame.dropna(subset=[value]).copy()
    if frame.empty:
        st.info("No eligible saved observations for this chart.")
        return
    category = label
    label_axis = alt.Axis(labelLimit=200)
    if label == "security_label" and "security_id" in frame:
        # Distinct securities must remain distinct even when their tickers match.
        frame = frame.copy()
        category = "security_key"
        frame[category] = [f"item_{i}" for i in range(len(frame))]
        labels = dict(zip(frame[category], frame.security_label.astype(str)))
        label_axis = alt.Axis(labelLimit=200, labelExpr=f"({json.dumps(labels)})[datum.label]")
    enc = dict(y=alt.Y(f"{category}:N", sort="-x", title=None, axis=label_axis),
               x=alt.X(f"{value}:Q", title=title, axis=alt.Axis(format=".1%" if percent else ".2f", tickCount=6)),
               tooltip=[alt.Tooltip(f"{label}:N", title="Item"), alt.Tooltip(f"{value}:Q", title=title, format=".2%" if percent else ".3f")])
    if series:
        enc["color"] = alt.Color(f"{series}:N", title=None, scale=alt.Scale(domain=SERIES, range=COLORS))
        enc["yOffset"] = f"{series}:N"
    else:
        enc["color"] = alt.condition(alt.datum[value] >= 0, alt.value("#0D9488"), alt.value("#D97706"))
    chart = alt.Chart(frame).mark_bar(cornerRadiusEnd=3).encode(**enc).properties(height=height)
    render_chart(styled(chart), width="stretch", alt=title)


def grouped_date_bars(frame, date, value, group, title, *, groups=None, height=330):
    """Discrete event bands make offsets genuinely grouped rather than stacked."""
    frame = frame.dropna(subset=[date, value]).copy()
    if frame.empty:
        st.info("No measured events in this selection.")
        return
    dates = pd.to_datetime(frame[date])
    frame["event"] = dates.dt.strftime("%Y-%m-%d")
    events = sorted(frame.event.unique())
    years = str(dates.min().year) if dates.min().year == dates.max().year else f"{dates.min().year}–{dates.max().year}"
    if (dates.max() - dates.min()).days > 730:
        ticks = list(dict((e[:4], e) for e in reversed(events)).values())
        axis = alt.Axis(values=sorted(ticks), labelExpr="slice(datum.label, 0, 4)", labelAngle=0, title="Effective rebalance date")
    else:
        axis = alt.Axis(labelExpr="timeFormat(toDate(datum.label), '%b')", labelAngle=0, title=years, titlePadding=14)
    groups = groups or SERIES
    chart = alt.Chart(frame).mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        x=alt.X("event:O", sort=events, axis=axis), xOffset=alt.XOffset(f"{group}:N", sort=groups),
        y=alt.Y(f"{value}:Q", title=title, stack=None, axis=alt.Axis(format=".0%", tickCount=6)),
        color=alt.Color(f"{group}:N", title=None, scale=alt.Scale(domain=groups, range=COLORS), legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip(f"{date}:T", title="Effective date", format="%d %b %Y"), alt.Tooltip(f"{group}:N", title="Series"), alt.Tooltip(f"{value}:Q", title=title, format=".2%")],
    ).properties(height=height)
    render_chart(styled(chart), alt=title + " by effective rebalance date")


def waterfall(components, total, label="Return difference"):
    """Plot saved additive components; the sum must equal the stated target."""
    values = list(components.values())
    if any(pd.isna(v) for v in values) or pd.isna(total):
        st.info("The saved decomposition is unavailable for this selection.")
        return
    if abs(sum(values) - total) > 1e-8:
        st.error("The chart components do not reconcile to the saved target. Review the validation details.")
        return
    rows = []
    running = 0.0
    for name, value in components.items():
        rows.append({"component": name, "start": running, "end": running + value, "value": value, "kind": "Positive" if value >= 0 else "Negative"})
        running += value
    rows.append({"component": label, "start": 0.0, "end": total, "value": total, "kind": "Total"})
    chart = alt.Chart(pd.DataFrame(rows)).mark_bar(size=48, cornerRadius=3).encode(
        x=alt.X("component:N", sort=list(components) + [label], title=None, axis=alt.Axis(labelAngle=-20)),
        y=alt.Y("end:Q", title="Contribution / return", axis=alt.Axis(format=".0%")), y2="start:Q",
        color=alt.Color("kind:N", scale=alt.Scale(domain=["Positive", "Negative", "Total"], range=["#0D9488", "#D97706", "#0F2A47"]), legend=None),
        tooltip=["component", alt.Tooltip("value:Q", title="Contribution", format=".2%")],
    ).properties(height=360)
    render_chart(styled(chart), width="stretch", alt="Additive saved contributions reconciling to " + label)


def label_frame(frame, settings):
    frame = frame.copy()
    for col in ("portfolio_id", "benchmark_portfolio_id", "benchmark_id"):
        if col in frame:
            frame[col] = frame[col].replace({settings["portfolio_id"]: SERIES[0], settings["benchmark_id"]: SERIES[1]})
    return frame


def growth_frame(paths, definition):
    """Add the actual prior-month opening close; keep the first earned return."""
    opening = pd.DataFrame({"date": pd.Timestamp(definition["opening_valuation_date"]),
                            "portfolio_id": paths.portfolio_id.unique(), "wealth_index": 1.0, "drawdown": 0.0})
    frame = pd.concat([opening, paths], ignore_index=True)
    frame["growth"] = frame.wealth_index * 100
    return frame.sort_values(["portfolio_id", "date"])


def styled(chart):
    # A final configure() replaces earlier configure_axis/configure_view calls.
    # Set the complete style in one operation so labels and legends survive.
    return chart.configure(
        font="Roboto", view=alt.ViewConfig(stroke=None),
        axis=alt.AxisConfig(labelColor="#6B7280", titleColor="#6B7280", labelFont="Roboto", titleFont="Roboto", labelFontSize=11, titleFontSize=12, titleFontWeight=400, gridColor="#EDF1F5", domain=False, tickColor="#DFE6ED"),
        axisX=alt.AxisConfig(grid=False),
        legend=alt.LegendConfig(title=None, orient="top", labelColor="#0F2A47", labelFont="Roboto", labelFontSize=11),
    )


def line_chart(frame, value, title, percent=False, height=330):
    frame = frame[["date", "portfolio_id", value]].copy()
    chart = alt.Chart(frame).mark_line(strokeWidth=2.4).encode(
        x=date_encoding(frame),
        y=alt.Y(f"{value}:Q", title=title, scale=alt.Scale(zero=False), axis=alt.Axis(format=".1%" if percent else ".0f" if value in ("growth", "wealth_index", "wealth_eop") else ".2f", tickCount=6)),
        color=alt.Color("portfolio_id:N", title=None, scale=alt.Scale(domain=SERIES, range=COLORS), legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("date:T", title="Close", format="%d %b %Y"), alt.Tooltip("portfolio_id:N", title="Series"), alt.Tooltip(f"{value}:Q", title=title, format=".2%" if percent else ".2f")],
    ).properties(height=height)
    render_chart(styled(chart), width="stretch", alt=title + ": active portfolio compared with benchmark")


def calendar_chart(periods):
    annual = periods.loc[periods.period_type.eq("annual")].copy()
    annual["year"] = pd.to_datetime(annual.period_start).dt.year.astype(str)
    chart = alt.Chart(annual).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("year:O", title=None, axis=alt.Axis(labelAngle=0)),
        xOffset="portfolio_id:N", y=alt.Y("period_return:Q", title="Return", axis=alt.Axis(format=".0%")),
        color=alt.Color("portfolio_id:N", title=None, scale=alt.Scale(domain=SERIES, range=COLORS)),
        tooltip=["year", alt.Tooltip("portfolio_id:N", title="Series"), alt.Tooltip("period_return:Q", format=".2%", title="Return"), alt.Tooltip("observed_calendar_months:Q", title="Months included")],
    ).properties(height=290)
    render_chart(styled(chart), width="stretch", alt="Portfolio and benchmark calendar-year returns within the selected months")


def heatmap(periods, series):
    frame = periods.loc[periods.period_type.eq("monthly") & periods.portfolio_id.eq(series)].copy()
    if frame.empty:
        st.info("Monthly returns are unavailable for this selection.")
        return
    dates = pd.to_datetime(frame.period_start)
    frame["year"] = dates.dt.year.astype(str)
    frame["month"] = dates.dt.strftime("%b")
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    limit = max(float(frame.period_return.abs().max()), 0.01)
    chart = alt.Chart(frame).mark_rect(cornerRadius=3, stroke="white", strokeWidth=3).encode(
        x=alt.X("month:O", sort=months, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("year:O", title=None),
        color=alt.Color("period_return:Q", title="Monthly return", scale=alt.Scale(domain=[-limit, 0, limit], range=["#D97706", "#F7F9FC", "#0D9488"], interpolate="rgb", clamp=True), legend=alt.Legend(format=".0%")),
        tooltip=["year", "month", alt.Tooltip("period_return:Q", title="Return", format=".2%")],
    ).properties(height=max(120, frame.year.nunique() * 32))
    text = alt.Chart(frame).mark_text(fontSize=11, fontWeight=500).encode(
        x=alt.X("month:O", sort=months), y="year:O", text=alt.Text("period_return:Q", format=".1%"),
        color=alt.value("#000000"),
    )
    render_chart(styled(chart + text), width="stretch", alt=f"Monthly return heatmap for {series}; unselected months are absent")


def pct(value):
    return "Unavailable" if pd.isna(value) else f"{value:.2%}"
