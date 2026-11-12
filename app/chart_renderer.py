"""Local Vega renderer with tooltips, fitting fullscreen and no export toolbar.

Libraries are vendored, integrity checked, and served by Streamlit's component
asset transport. Financial data and chart code never go to a third-party CDN.
"""
from pathlib import Path
import hashlib

import altair
import streamlit as st

ROOT = Path(__file__).resolve().parent
LIBRARIES = "\n;\n".join((ROOT / "vendor" / name).read_text(encoding="utf-8")
                           for name in ("vega.min.js", "vega-lite.min.js", "vega-embed.min.js"))
_CHART = st.components.v2.component(
    "research_chart",
    html='<section class="chart"><div class="tools"><button type="button">Fullscreen</button></div><div class="plot"></div><p class="error" role="alert" hidden></p></section>',
    css="""
    .chart {font-family:var(--st-font),Roboto,sans-serif;color:#0F2A47;background:#fff;width:100%;}
    .tools {display:flex;justify-content:flex-end;height:28px;}
    button {font:inherit;font-size:12px;color:#52657A;background:transparent;border:1px solid #DFE6ED;border-radius:5px;padding:3px 9px;cursor:pointer;}
    button:focus-visible {outline:2px solid #0D9488;outline-offset:2px;}
    .plot {width:100%;min-width:0;}
    .plot svg {display:block;}
    .chart:fullscreen {box-sizing:border-box;padding:20px;width:100vw;height:100vh;overflow:auto;}
    .error {color:#9B3324;padding:12px;border:1px solid #DFE6ED;}
    """,
    js=LIBRARIES + "\n;\n" + (ROOT / "chart_renderer.js").read_text(encoding="utf-8"),
)


def render_chart(chart, *, alt, width="stretch", **kwargs):
    """Render trusted app-authored Altair specs; database values travel as data."""
    with altair.data_transformers.enable("default", max_rows=None):
        spec = chart.to_dict()
    # Source IDs stay server-side. Chart categories use local, disposable keys.
    for rows in spec.get("datasets", {}).values():
        for row in rows:
            row.pop("security_id", None)
            row.pop("other_security_id", None)
    spec["description"] = alt
    spec["width"] = "container"
    spec["autosize"] = {"type": "fit-x", "contains": "padding", "resize": True}
    key = "chart_" + hashlib.sha256(alt.encode()).hexdigest()[:16]
    return _CHART(data={"spec": spec, "description": alt}, key=key, width=width, height="content")
