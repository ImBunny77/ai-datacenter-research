"""
AI Data Center Research Dashboard
===================================
Master Streamlit dashboard — three analytical tools in one UI:

  1. Circular Financing Network  (networkx + plotly, live market data via yfinance)
  2. 1GW Data Center P&L Monte Carlo  (numpy + matplotlib, interactive sliders)
  3. Earnings Call CAPEX Conviction Analyzer  (rule-based NLP, instant scoring)

Run:  tool3_nlp/venv/Scripts/streamlit run dashboard/app.py
"""

import math
import re
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
DEALS_CSV   = ROOT / "tool1_network" / "deals_data.csv"
TRANSCRIPTS = ROOT / "tool3_nlp" / "transcripts"

# ── Ticker map for live market data ──────────────────────────────────────────
COMPANY_TICKERS = {
    "Microsoft":  "MSFT",
    "NVIDIA":     "NVDA",
    "Amazon":     "AMZN",
    "Oracle":     "ORCL",
    "CoreWeave":  "CRWV",
    "OpenAI":     None,   # private
    "xAI":        None,   # private
}

# ── Color palettes ─────────────────────────────────────────────────────────────
COMPANY_COLORS = {
    "Microsoft": "#00a4ef",
    "OpenAI":    "#10a37f",
    "NVIDIA":    "#76b900",
    "CoreWeave": "#6f42c1",
    "Oracle":    "#f80000",
    "xAI":       "#1da1f2",
    "Amazon":    "#ff9900",
}
FLOW_COLORS = {
    "Equity Investment":  "#f4d03f",
    "Compute Spend":      "#5dade2",
    "Hardware Purchase":  "#ec7063",
}

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Center Research",
    page_icon=":building_construction:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer { visibility: hidden; }

.page-title {
    font-size: 1.85rem;
    font-weight: 700;
    color: #e6edf3;
    letter-spacing: -0.4px;
    margin-bottom: 3px;
    line-height: 1.2;
}
.page-subtitle {
    font-size: 0.83rem;
    color: #7d8590;
    margin-bottom: 20px;
}
[data-testid="metric-container"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    padding: 14px 16px !important;
}
[data-testid="stSidebar"] > div:first-child {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
hr { border-color: #21262d !important; }
[data-testid="stDataFrame"] {
    border: 1px solid #30363d;
    border-radius: 6px;
}
[data-testid="baseButton-primary"] {
    background: #1f6feb !important;
    border-color: #1f6feb !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}
.ticker-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 8px;
    font-size: 0.82rem;
}
.ticker-name  { color: #7d8590; font-size: 0.75rem; }
.ticker-price { color: #e6edf3; font-weight: 700; font-size: 1.1rem; }
.ticker-up    { color: #2ecc71; }
.ticker-down  { color: #e74c3c; }
.ticker-na    { color: #7d8590; font-style: italic; }
.info-box {
    background: #0d1117;
    border-left: 3px solid #1f6feb;
    padding: 10px 14px;
    border-radius: 0 6px 6px 0;
    font-size: 0.82rem;
    color: #8b949e;
    margin-bottom: 16px;
}
.disclaimer {
    font-size: 0.72rem;
    color: #484f58;
    margin-top: 24px;
    border-top: 1px solid #21262d;
    padding-top: 10px;
}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=timedelta(hours=1), show_spinner=False)
def fetch_market_data(tickers: tuple[str, ...]) -> dict:
    """Pull price + 1-day change for a list of tickers. Refreshes every hour."""
    try:
        import yfinance as yf
        result = {}
        for tkr in tickers:
            try:
                info   = yf.Ticker(tkr).fast_info
                price  = getattr(info, "last_price", None)
                prev   = getattr(info, "previous_close", None)
                mktcap = getattr(info, "market_cap", None)
                if price and prev:
                    chg_pct = (price - prev) / prev * 100
                else:
                    chg_pct = None
                result[tkr] = dict(price=price, chg_pct=chg_pct, mktcap=mktcap)
            except Exception:
                result[tkr] = dict(price=None, chg_pct=None, mktcap=None)
        return result
    except ImportError:
        return {}


@st.cache_data(show_spinner=False)
def load_deals() -> pd.DataFrame:
    try:
        df = pd.read_csv(DEALS_CSV)
        df = df[df["amount_billions"] > 0].copy()
        return df
    except Exception as e:
        st.error(f"Could not load deal data: {e}")
        return pd.DataFrame()


def fmt_mktcap(val) -> str:
    if val is None:
        return "N/A"
    if val >= 1e12:
        return f"${val/1e12:.2f}T"
    if val >= 1e9:
        return f"${val/1e9:.1f}B"
    return f"${val/1e6:.0f}M"


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## AI Data Center Research")
    st.caption("Quantitative Research Dashboard")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Circular Financing Map", "P&L Monte Carlo", "Earnings Sentiment"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Global refresh button
    if st.button("Refresh All Data", use_container_width=True, help="Clears cached market data and reloads everything"):
        st.cache_data.clear()
        st.rerun()

    st.markdown(
        f"<div style='font-size:0.72rem;color:#484f58;margin-top:6px;text-align:center'>"
        f"Market data refreshes hourly<br>"
        f"Last loaded: {datetime.now().strftime('%H:%M:%S')}"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("**About**")
    st.caption(
        "Deal figures sourced from public filings, press releases, and analyst estimates. "
        "Market data via Yahoo Finance. For research and illustrative purposes only."
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — CIRCULAR FINANCING MAP
# ═════════════════════════════════════════════════════════════════════════════
if page == "Circular Financing Map":

    st.markdown('<div class="page-title">AI Circular Financing Network</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        'Directed cash-flow map between major AI data center players (2023–2025) &mdash; '
        'hover nodes and edges for detail, scroll to zoom, drag to pan'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="info-box">'
        '<b>How to read this chart:</b> Each arrow represents a real capital flow. '
        'Arrow thickness scales with dollar amount. '
        'Node size reflects total outbound spend. '
        'Colors: <span style="color:#f4d03f">Gold = Equity</span> &nbsp; '
        '<span style="color:#5dade2">Blue = Compute</span> &nbsp; '
        '<span style="color:#ec7063">Red = Hardware</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Map Controls")
        flow_filter = st.multiselect(
            "Flow Types",
            list(FLOW_COLORS.keys()),
            default=list(FLOW_COLORS.keys()),
            help="Show or hide specific types of capital flows",
        )
        min_deal = st.slider(
            "Min Deal Size ($B)", 0.0, 12.0, 0.0, 0.5,
            help="Filter out deals below this threshold to reduce clutter",
        )
        show_labels = st.toggle("Show $ Labels on Edges", value=False)
        node_scale  = st.slider("Node Size", 0.5, 2.0, 1.0, 0.1)

    # ── Load data ─────────────────────────────────────────────────────────────
    df_all = load_deals()
    if df_all.empty:
        st.stop()

    df = df_all[
        df_all["flow_type"].isin(flow_filter) &
        (df_all["amount_billions"] >= min_deal)
    ].copy()

    # ── Live market data ──────────────────────────────────────────────────────
    public_tickers = tuple(
        t for t in COMPANY_TICKERS.values() if t is not None
    )
    with st.spinner("Fetching live market data..."):
        mkt = fetch_market_data(public_tickers)

    # ── Summary metrics row ───────────────────────────────────────────────────
    total_flow  = df_all["amount_billions"].sum()
    nvidia_in   = df_all[df_all["target"] == "NVIDIA"]["amount_billions"].sum()
    msft_out    = df_all[df_all["source"] == "Microsoft"]["amount_billions"].sum()
    openai_out  = df_all[df_all["source"] == "OpenAI"]["amount_billions"].sum()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Total Tracked Flow", f"${total_flow:.0f}B",
        delta=f"{len(df_all)} disclosed deals",
        help="Sum of all capital flows in the dataset. Some flows are estimated from public announcements.",
    )
    m2.metric(
        "NVIDIA Hardware Inflow", f"${nvidia_in:.0f}B",
        delta="Largest single beneficiary",
        delta_color="normal",
        help="Total GPU hardware purchases directed to NVIDIA across all tracked deals.",
    )
    m3.metric(
        "Microsoft Deployed", f"${msft_out:.0f}B",
        delta="OpenAI + NVIDIA + CoreWeave",
        delta_color="off",
        help="Total capital deployed by Microsoft across equity investments and compute/hardware spend.",
    )
    m4.metric(
        "OpenAI Compute Spend", f"${openai_out:.0f}B",
        delta="Oracle + CoreWeave contracts",
        delta_color="off",
        help="OpenAI's external compute commitments. OpenAI does not own its own data centers.",
    )

    st.markdown("---")

    # ── Network graph ─────────────────────────────────────────────────────────
    if df.empty:
        st.warning("No deals match the current filters. Try widening the flow type selection or lowering the minimum deal size.")
    else:
        G   = nx.DiGraph()
        all_nodes = sorted(set(df["source"]) | set(df["target"]))
        G.add_nodes_from(all_nodes)

        agg = (
            df.groupby(["source", "target", "flow_type"])["amount_billions"]
            .sum()
            .reset_index()
        )
        for _, row in agg.iterrows():
            G.add_edge(
                row["source"], row["target"],
                weight=row["amount_billions"],
                flow_type=row["flow_type"],
            )

        n = G.number_of_nodes()
        pos = {}
        for i, nd in enumerate(all_nodes):
            a = 2 * math.pi * i / n - math.pi / 2
            pos[nd] = (math.cos(a), math.sin(a))

        max_w  = agg["amount_billions"].max() or 1
        traces = []
        annots = []

        for (src, tgt), edata in G.edges.items():
            x0, y0 = pos[src]
            x1, y1 = pos[tgt]
            w      = edata["weight"]
            lw     = 1.5 + 10 * (w / max_w)
            color  = FLOW_COLORS.get(edata["flow_type"], "#aaa")

            traces.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines",
                line=dict(width=lw, color=color),
                hoverinfo="text",
                text=(
                    f"<b>{src} \u2192 {tgt}</b><br>"
                    f"Type: {edata['flow_type']}<br>"
                    f"Amount: <b>${w:.1f}B</b>"
                ),
                showlegend=False,
            ))

            if show_labels:
                annots.append(dict(
                    x=(x0 + x1) / 2, y=(y0 + y1) / 2,
                    text=f"${w:.0f}B",
                    showarrow=False,
                    font=dict(size=9, color="#e6edf3"),
                    bgcolor="#21262d", borderpad=2,
                ))

            dx, dy = x1 - x0, y1 - y0
            length = math.sqrt(dx ** 2 + dy ** 2) or 1
            annots.append(dict(
                ax=x0 + dx * 0.2, ay=y0 + dy * 0.2,
                x=x1 - dx / length * 0.14,
                y=y1 - dy / length * 0.14,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=3, arrowsize=1.6, arrowwidth=1.6,
                arrowcolor=color,
            ))

        for ft, fc in FLOW_COLORS.items():
            if ft in flow_filter:
                traces.append(go.Scatter(
                    x=[None], y=[None], mode="lines",
                    line=dict(width=5, color=fc),
                    name=ft, showlegend=True,
                ))

        total_out = df.groupby("source")["amount_billions"].sum().to_dict()
        nx_x, nx_y, nx_txt, nx_hov, nx_col, nx_sz = [], [], [], [], [], []
        for nd in all_nodes:
            x, y = pos[nd]
            nx_x.append(x)
            nx_y.append(y)
            out  = total_out.get(nd, 0)
            tkr  = COMPANY_TICKERS.get(nd)
            # Build rich hover
            hover_lines = [f"<b>{nd}</b>"]
            if tkr and tkr in mkt and mkt[tkr]["price"]:
                d = mkt[tkr]
                chg_str = (
                    f"{'▲' if d['chg_pct'] >= 0 else '▼'} {abs(d['chg_pct']):.2f}%"
                    if d["chg_pct"] is not None else ""
                )
                hover_lines.append(f"Ticker: {tkr}  {chg_str}")
                hover_lines.append(f"Price: ${d['price']:,.2f}")
                hover_lines.append(f"Mkt Cap: {fmt_mktcap(d['mktcap'])}")
            else:
                hover_lines.append("(Private company)")
            hover_lines += [
                f"Outbound flow: ${out:.1f}B",
                f"Connections: {G.degree(nd)}",
            ]
            nx_hov.append("<br>".join(hover_lines))
            nx_txt.append(f"<b>{nd}</b>")
            nx_col.append(COMPANY_COLORS.get(nd, "#cccccc"))
            raw_sz = (16 + out * 1.3) * node_scale
            nx_sz.append(max(26, min(68, raw_sz)))

        traces.append(go.Scatter(
            x=nx_x, y=nx_y,
            mode="markers+text",
            marker=dict(
                size=nx_sz, color=nx_col,
                line=dict(width=2.5, color="#0d1117"),
            ),
            text=nx_txt,
            textposition="top center",
            textfont=dict(size=11, color="#e6edf3"),
            hovertext=nx_hov,
            hoverinfo="text",
            name="Companies",
        ))

        fig_net = go.Figure(data=traces)
        fig_net.update_layout(
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font=dict(color="#c9d1d9"),
            dragmode="pan",          # pan by default, not box-select
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.55, 1.55]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.55, 1.55]),
            annotations=annots,
            legend=dict(
                bgcolor="#161b22", bordercolor="#30363d",
                borderwidth=1, font=dict(color="#c9d1d9"), itemsizing="constant",
            ),
            margin=dict(l=10, r=10, t=10, b=10),
            height=620,
            hovermode="closest",
            hoverlabel=dict(bgcolor="#161b22", bordercolor="#30363d", font=dict(color="#e6edf3")),
        )
        st.plotly_chart(fig_net, use_container_width=True)

    # ── Live market ticker strip ──────────────────────────────────────────────
    st.markdown("### Live Market Snapshot")
    st.caption("Public companies only — OpenAI and xAI are private. Updates every hour or on manual refresh.")

    cols = st.columns(5)
    public_companies = [(name, tkr) for name, tkr in COMPANY_TICKERS.items() if tkr]
    for col, (name, tkr) in zip(cols, public_companies):
        d = mkt.get(tkr, {})
        with col:
            if d.get("price"):
                chg     = d["chg_pct"]
                chg_cls = "ticker-up" if chg and chg >= 0 else "ticker-down"
                chg_str = f"{'▲' if chg and chg >= 0 else '▼'} {abs(chg):.2f}%" if chg is not None else "—"
                st.markdown(
                    f'<div class="ticker-card">'
                    f'<div class="ticker-name">{name} ({tkr})</div>'
                    f'<div class="ticker-price">${d["price"]:,.2f}</div>'
                    f'<div class="{chg_cls}">{chg_str} today</div>'
                    f'<div class="ticker-name">{fmt_mktcap(d["mktcap"])} mkt cap</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="ticker-card">'
                    f'<div class="ticker-name">{name} ({tkr})</div>'
                    f'<div class="ticker-na">Data unavailable</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── Deal data table ───────────────────────────────────────────────────────
    with st.expander("Full Deal Data & Sources", expanded=False):
        st.caption("All figures sourced from public earnings disclosures, press releases, and analyst estimates.")
        st.dataframe(
            df_all[["source", "target", "flow_type", "amount_billions", "year", "notes"]]
            .sort_values("amount_billions", ascending=False),
            use_container_width=True,
            hide_index=True,
            column_config={
                "amount_billions": st.column_config.NumberColumn("Amount ($B)", format="$%.1f B"),
                "year": st.column_config.NumberColumn("Year", format="%d"),
                "notes": st.column_config.TextColumn("Notes / Source", width="large"),
            },
        )

    st.markdown(
        '<div class="disclaimer">'
        'Deal amounts are estimates based on publicly available information as of Q1 2025. '
        'Figures may not reflect final contracted values. Not financial advice.'
        '</div>',
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — P&L MONTE CARLO
# ═════════════════════════════════════════════════════════════════════════════
elif page == "P&L Monte Carlo":

    st.markdown('<div class="page-title">1GW AI Data Center P&L Modeler</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        'Monte Carlo stress-test: 10,000 scenarios over a 5-year depreciation horizon. '
        'Adjust the sliders to update all charts instantly.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="info-box">'
        '<b>How this works:</b> We model a hypothetical 1 gigawatt AI data center '
        '(200,000 GPUs across 25,000 racks). Each of 10,000 scenarios randomises '
        'power costs, GPU downtime, and annual rental price compression. '
        'The simulation calculates 5-year cumulative net income + a terminal value '
        'exit multiple, then divides by equity invested to get the equity return.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Sidebar sliders (grouped) ─────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Infrastructure")
        gpu_cost    = st.slider(
            "GPU Unit Cost ($k)", 20, 60, 30, 5,
            help="Per-GPU purchase cost. H100 trades ~$25-35k; Blackwell B200 ~$35-45k.",
        )
        pue         = st.slider(
            "PUE (Power Efficiency)", 1.10, 1.80, 1.35, 0.05,
            help="Power Usage Effectiveness — ratio of total facility power to IT power. "
                 "1.0 = perfect (impossible). Best hyperscalers reach ~1.15. Typical = 1.3-1.5.",
        )

        st.markdown("### Revenue")
        gpu_price   = st.slider(
            "GPU Rental ($/GPU/hr)", 1.0, 8.0, 2.80, 0.10,
            format="$%.2f",
            help="Base H100/Blackwell spot rate. H100 spot was ~$2-3/hr in 2024; "
                 "Blackwell commands a premium at ~$4-6/hr.",
        )
        compression = st.slider(
            "Annual Price Compression (%)", 0, 40, 12, 1,
            help="How fast GPU rental rates drop each year as supply grows. "
                 "Industry analysts estimate 10-20%/yr compression through 2026.",
        )
        utilization = st.slider(
            "GPU Utilization (%)", 50, 100, 85, 1,
            help="% of GPU fleet actively rented at any time. "
                 "Hyperscalers report 85-95%. Startups often run 60-75%.",
        )

        st.markdown("### Power")
        power_kwh   = st.slider(
            "Power Cost ($/kWh)", 0.02, 0.12, 0.045, 0.005,
            format="$%.3f",
            help="Wholesale electricity rate for large datacenters. "
                 "US average ~$0.04-0.06; Texas ~$0.03; EU ~$0.07-0.12.",
        )

        st.markdown("### Capital Structure")
        equity_pct  = st.slider(
            "Equity Fraction (%)", 10, 60, 30, 5,
            help="% of total CAPEX funded by equity. Remainder = debt at 6.5% interest. "
                 "PE-backed builds typically use 25-40% equity.",
        )

        st.markdown("---")
        n_sims      = st.select_slider(
            "Simulations", [1_000, 5_000, 10_000, 25_000], value=10_000,
            help="More simulations = more accurate distribution, slower compute.",
        )
        target_x    = st.number_input(
            "Target Equity Multiple (x)", min_value=2.0, max_value=50.0, value=15.0, step=1.0,
            help="Your underwriting threshold. 15x is aggressive for infrastructure. "
                 "Typical PE infra returns are 2-4x over 5 years.",
        )
        if st.button("Reset to Defaults", use_container_width=True):
            st.rerun()

    # ── Simulation ────────────────────────────────────────────────────────────
    @st.cache_data(show_spinner=False)
    def run_mc(
        gpu_k: int, p_kwh: float, pue_v: float, rental: float,
        compress_p: int, util_p: int, eq_frac: float,
        N: int, tgt: float, seed: int = 42,
    ) -> dict:
        np.random.seed(seed)
        YRS = 5
        GPUS_PER_RACK = 8
        RACK_KW       = 40
        POWER_MW      = 1_000

        total_gpus     = (POWER_MW * 1_000 / RACK_KW) * GPUS_PER_RACK   # 200,000
        gpu_capex      = total_gpus * gpu_k * 1_000
        cooling_capex  = POWER_MW * 2_500_000
        building_capex = POWER_MW * 1_500_000
        total_capex    = gpu_capex + cooling_capex + building_capex
        equity_inv     = total_capex * eq_frac
        debt           = total_capex * (1 - eq_frac)
        annual_int     = debt * 0.065
        annual_depr    = total_capex / YRS

        # Stochastic inputs
        power_costs = np.random.lognormal(math.log(p_kwh), 0.25, N)
        downtime    = np.clip(np.random.normal(0.05, 0.04, N), 0, 0.5)
        compress    = np.clip(np.random.normal(compress_p / 100, 0.05, N), 0, 0.5)

        cum_net = np.zeros(N)
        cum_rev = np.zeros(N)
        gm_yr5  = np.zeros(N)
        HOURS   = 8_760

        for yr in range(1, YRS + 1):
            price_yr = rental * ((1 - compress) ** yr)
            revenue  = total_gpus * price_yr * (util_p / 100) * (1 - downtime) * HOURS
            power_kw = total_gpus * 0.70 * pue_v        # ~700W TDP per H100
            pcost    = power_kw * HOURS * power_costs
            gp       = revenue - pcost
            ebitda   = gp - revenue * 0.05              # 5% OpEx (staffing, networking)
            ebit     = ebitda - annual_depr
            ebt      = ebit - annual_int
            tax      = np.maximum(ebt * 0.21, 0)        # 21% effective rate
            net      = ebt - tax
            cum_net += net
            cum_rev += revenue
            if yr == YRS:
                gm_yr5 = gp / np.maximum(revenue, 1)

        terminal_v = (cum_rev / YRS) * 0.35 * 5         # 5x EV/EBITDA exit
        multiple   = (cum_net + terminal_v) / equity_inv

        return dict(
            multiple=multiple,
            gm_yr5=gm_yr5,
            total_capex=total_capex,
            equity_inv=equity_inv,
            total_gpus=total_gpus,
        )

    with st.spinner("Running simulations..."):
        mc   = run_mc(gpu_cost, power_kwh, pue, gpu_price, compression,
                      utilization, equity_pct / 100, n_sims, target_x)
        mult = mc["multiple"]
        gm   = mc["gm_yr5"]

    p10, p50, p90 = np.percentile(mult, [10, 50, 90])
    hit_pct  = float((mult >= target_x).mean() * 100)
    bust_pct = float((mult < 0).mean()  * 100)
    be_pct   = float((mult >= 1.0).mean() * 100)
    med_gm   = float(np.median(gm) * 100)

    # ── Metric row ────────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(
        "Total CAPEX", f"${mc['total_capex']/1e9:.1f}B",
        delta=f"{mc['total_gpus']:,.0f} GPUs",
        help="Total capital expenditure = GPU cost + cooling ($2.5M/MW) + building ($1.5M/MW). "
             "Fixed at 1GW / 40kW per rack / 8 GPUs per rack.",
    )
    m2.metric(
        "P50 Equity Multiple", f"{p50:.1f}x",
        delta=f"P10: {p10:.1f}x  |  P90: {p90:.1f}x",
        delta_color="off",
        help="P50 = median scenario. Half of all simulations returned more than this, half less. "
             "P10 is the bear case (bottom 10%), P90 is the bull case (top 10%).",
    )
    m3.metric(
        f"Hit {target_x:.0f}x Target", f"{hit_pct:.1f}%",
        delta=f"of {n_sims:,} scenarios",
        delta_color="normal" if hit_pct >= 20 else "inverse",
        help=f"Fraction of scenarios where the equity return exceeded {target_x:.0f}x. "
             "PE infrastructure deals typically target 2-4x. 15x is extremely aggressive.",
    )
    m4.metric(
        "Bankruptcy Risk", f"{bust_pct:.1f}%",
        delta="returns below 0x",
        delta_color="inverse" if bust_pct > 5 else "normal",
        help="Fraction of scenarios where cumulative net income + terminal value "
             "is negative — i.e., equity is wiped out.",
    )
    m5.metric(
        "Median Yr-5 Gross Margin", f"{med_gm:.1f}%",
        delta=f"Break-even rate: {be_pct:.0f}%",
        delta_color="normal",
        help="Gross Profit Margin in year 5 = (Revenue - Power Cost) / Revenue. "
             "High because power is the only direct COGS; OpEx and debt service come below.",
    )

    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")

    mult_c = np.clip(mult, -5, 60)
    bins   = np.linspace(mult_c.min(), mult_c.max(), 70)

    ax1.set_facecolor("#161b22")
    ax1.hist(mult_c[mult < 0],
             bins=bins, color="#e74c3c", alpha=0.9, label="Bankruptcy (<0x)")
    ax1.hist(mult_c[(mult >= 0) & (mult < target_x)],
             bins=bins, color="#f39c12", alpha=0.9, label=f"0x\u2013{target_x:.0f}x")
    ax1.hist(mult_c[mult >= target_x],
             bins=bins, color="#2ecc71", alpha=0.9, label=f"Hit {target_x:.0f}x")
    ax1.axvline(target_x, color="#fff", ls="--", lw=1.5, alpha=0.6,
                label=f"{target_x:.0f}x threshold")
    ax1.axvline(p50, color="#a29bfe", ls=":", lw=1.8,
                label=f"Median {p50:.1f}x")
    ax1.set_xlabel("Equity Return Multiple (5-yr)", color="#c9d1d9", fontsize=10)
    ax1.set_ylabel("Number of Scenarios", color="#c9d1d9", fontsize=10)
    ax1.set_title("Equity Multiple Distribution", color="#e6edf3", fontsize=11, pad=8)
    ax1.tick_params(colors="#c9d1d9")
    for sp in ax1.spines.values():
        sp.set_color("#30363d")
    ax1.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8.5)
    ax1.text(
        0.97, 0.97,
        f"P10: {p10:.1f}x\nP50: {p50:.1f}x\nP90: {p90:.1f}x\n\n"
        f"Hit {target_x:.0f}x: {hit_pct:.1f}%\nBankruptcy: {bust_pct:.1f}%",
        transform=ax1.transAxes, fontsize=8.5, va="top", ha="right", color="#c9d1d9",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#21262d", edgecolor="#30363d"),
    )

    gm_pct = np.clip(gm, -0.2, 1.0) * 100
    ax2.set_facecolor("#161b22")
    ax2.hist(gm_pct, bins=60, color="#5dade2", alpha=0.9, edgecolor="#0d1117", lw=0.3)
    ax2.axvline(med_gm, color="#f1c40f", ls="--", lw=1.8,
                label=f"Median {med_gm:.1f}%")
    ax2.axvline(0, color="#e74c3c", ls=":", lw=1.2, alpha=0.7, label="Zero margin")
    ax2.set_xlabel("Year-5 Gross Profit Margin (%)", color="#c9d1d9", fontsize=10)
    ax2.set_ylabel("Number of Scenarios", color="#c9d1d9", fontsize=10)
    ax2.set_title(
        "Year-5 Gross Margin Distribution\n(Power cost + rental compression stress)",
        color="#e6edf3", fontsize=11, pad=8,
    )
    ax2.tick_params(colors="#c9d1d9")
    for sp in ax2.spines.values():
        sp.set_color("#30363d")
    ax2.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8.5)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Interpretation ────────────────────────────────────────────────────────
    with st.expander("Scenario Interpretation & Key Assumptions", expanded=True):
        ia, ib = st.columns(2)
        with ia:
            st.markdown("#### Bear Case (P10)")
            st.markdown(f"- Equity Multiple: **{p10:.2f}x**")
            st.markdown(
                f"- Driven by above-average power costs and "
                f"{compression}%/yr rental price compression"
            )
            if p10 < 1.0:
                st.error(
                    f"P10 scenario returns **{p10:.1f}x** — equity loses money. "
                    "This indicates structural downside at current assumptions."
                )
            else:
                st.success(f"P10 breaks even at {p10:.1f}x. Equity is preserved even in bad scenarios.")

        with ib:
            st.markdown("#### Bull Case (P90)")
            st.markdown(f"- Equity Multiple: **{p90:.2f}x**")
            st.markdown("- Requires low power costs, minimal downtime, and stable rental rates")
            if p90 < target_x:
                st.warning(
                    f"Even the P90 scenario ({p90:.1f}x) misses the {target_x:.0f}x target. "
                    "Try raising the GPU rental price, cutting GPU cost, or increasing utilization."
                )
            else:
                st.success(f"P90 hits {p90:.1f}x — exceeds the {target_x:.0f}x target in bull scenarios.")

        st.markdown("---")
        st.markdown("#### Fixed Assumptions (not slider-adjustable)")
        fa1, fa2, fa3 = st.columns(3)
        fa1.markdown(
            "**Infrastructure**\n"
            "- 1 GW total power\n"
            "- 40 kW per rack\n"
            "- 8 GPUs per rack\n"
            "- 200,000 total GPUs\n"
            "- 5-year depreciation"
        )
        fa2.markdown(
            "**Cost Structure**\n"
            "- Cooling: $2.5M/MW\n"
            "- Building: $1.5M/MW\n"
            "- 5% OpEx on revenue\n"
            "- 6.5% debt interest rate\n"
            "- 21% effective tax rate"
        )
        fa3.markdown(
            "**Exit Assumptions**\n"
            "- Terminal value = 5x EV/EBITDA\n"
            "- EBITDA margin ~35%\n"
            "- GPU TDP: 700W (H100)\n"
            "- Downtime: avg 5%, std 4%\n"
            "- Compression: lognormal"
        )

    st.markdown(
        '<div class="disclaimer">'
        'This model is for illustrative purposes only. Actual returns will vary. '
        'Real datacenter economics depend on contract structures, hyperscaler relationships, '
        'power agreements, and GPU supply conditions not captured here.'
        '</div>',
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — EARNINGS SENTIMENT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Earnings Sentiment":

    st.markdown('<div class="page-title">Earnings Call CAPEX Conviction Analyzer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        'Paste any earnings call excerpt to get an instant hedge vs. conviction score. '
        'Load a sample from the sidebar to see how different companies compare.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="info-box">'
        '<b>What this detects:</b> CEOs hedge on CAPEX with language like '
        '"<i>evaluating</i>", "<i>right-sizing</i>", "<i>pacing</i>", '
        '"<i>long-term horizon</i>". Committed CEOs use '
        '"<i>deployed</i>", "<i>contracted</i>", "<i>free cash flow</i>", '
        '"<i>take-or-pay</i>". The score (0\u2013100) reflects which dominates.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Lexicons ──────────────────────────────────────────────────────────────
    HEDGE_WORDS: dict[str, float] = {
        "evaluating": 2.0, "re-assessing": 3.0, "reassessing": 3.0,
        "right-sizing": 3.0, "rightsizing": 3.0, "pacing": 1.5,
        "carefully": 1.0, "monitoring": 1.5, "watching": 1.0,
        "remain flexible": 2.0, "remaining flexible": 2.0,
        "long-term horizon": 2.0, "longer-term horizon": 2.0,
        "multi-year": 0.5, "early stages": 2.0, "too early": 2.5,
        "may need to": 2.0, "could accelerate": 1.5, "could decelerate": 2.5,
        "thoughtful": 1.0, "disciplined": 0.5, "deliberately": 1.0,
        "weighing on": 1.5, "weigh on": 1.5, "still in the": 1.5,
        "over-building": 1.5, "structural challenges": 2.0,
        "not locked into": 2.0,
    }
    CONVICTION_WORDS: dict[str, float] = {
        "free cash flow": 2.5, "operating income": 2.0, "gross margin": 2.0,
        "accretive": 3.0, "roi": 3.0, "return on investment": 3.0,
        "deployed": 2.5, "commissioned": 2.5, "contracted": 3.5,
        "committed": 3.0, "take-or-pay": 4.0, "pre-leased": 3.5,
        "booked": 2.5, "shovels": 4.0, "under construction": 3.0,
        "broke ground": 3.0, "fully funded": 3.5, "board approved": 3.0,
        "purchase orders": 3.0, "record": 1.5, "backlog": 2.5,
        "year-over-year": 1.0, "year over year": 1.0, "utilization": 2.0,
        "in production": 2.5, "fully deployed": 3.0, "fully on track": 2.5,
        "on track": 1.5, "clear contractual": 3.5, "zero speculative": 4.0,
        "funded": 2.0, "accelerating": 2.0,
    }

    def score_text(text: str) -> dict:
        tl   = text.lower()
        wc   = max(len(re.findall(r"[\w\-]+", tl)), 1)
        norm = 1_000 / wc
        hs, cs = 0.0, 0.0
        hm: dict = defaultdict(int)
        cm: dict = defaultdict(int)
        for phrase, w in HEDGE_WORDS.items():
            cnt = tl.count(phrase)
            if cnt:
                hs += w * cnt
                hm[phrase] += cnt
        for phrase, w in CONVICTION_WORDS.items():
            cnt = tl.count(phrase)
            if cnt:
                cs += w * cnt
                cm[phrase] += cnt
        for phrase in HEDGE_WORDS:
            neg = len(re.findall(r"\bnot\s+" + re.escape(phrase), tl))
            hs -= neg * 3.0
            cs += neg * 2.0
        raw_diff = (cs - hs) * norm
        score    = max(0.0, min(100.0, 50.0 + (raw_diff / 60.0) * 50.0))
        return dict(
            score=round(score, 1),
            hedge_norm=round(hs * norm, 2),
            conviction_norm=round(cs * norm, 2),
            hedge_matches=dict(hm),
            conviction_matches=dict(cm),
            word_count=wc,
        )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Sentiment Controls")

        sample_opts = ["(paste your own)"]
        if TRANSCRIPTS.exists():
            sample_opts += [p.stem.split("_")[0] for p in sorted(TRANSCRIPTS.glob("*.txt"))]

        load_ex = st.selectbox(
            "Load Sample Transcript",
            sample_opts,
            help="Pre-loaded illustrative transcripts for MSFT, NVDA, ORCL, AMZN",
        )

        uploaded = st.file_uploader(
            "Or upload a .txt transcript",
            type=["txt"],
            help="Upload any plain-text earnings call transcript",
        )

        st.markdown("---")
        st.markdown("**Scoring Methodology**")
        st.caption(
            "Rule-based lexicon: 28 hedge phrases (weighted 0.5\u20134.0) and "
            "32 conviction phrases (weighted 1.0\u20134.0). "
            "Scores normalised per 1,000 words. "
            "Negation detection: 'not evaluating' flips a hedge into conviction. "
            "Score formula: `50 + (conviction \u2212 hedge) / 60 \u00d7 50`, clipped 0\u2013100."
        )

    # ── Text input ────────────────────────────────────────────────────────────
    preload = ""
    if uploaded is not None:
        preload = uploaded.read().decode("utf-8", errors="replace")
    elif load_ex != "(paste your own)" and TRANSCRIPTS.exists():
        matches = list(TRANSCRIPTS.glob(f"{load_ex}*.txt"))
        if matches:
            preload = matches[0].read_text(encoding="utf-8", errors="replace")

    transcript_input = st.text_area(
        "Earnings call transcript",
        value=preload,
        height=220,
        placeholder=(
            "Paste earnings call text here, or load a sample from the sidebar...\n\n"
            "HIGH HEDGE example:\n"
            "  'We are carefully evaluating our CAPEX commitments, right-sizing our build "
            "cadence on a longer-term horizon, and re-assessing demand signals.'\n\n"
            "HIGH CONVICTION example:\n"
            "  'We deployed $14B in contracted, pre-leased infrastructure. "
            "Every rack is fully committed. Free cash flow is accretive.'"
        ),
        label_visibility="collapsed",
    )

    st.caption("Paste text above and click Analyze, or select a sample from the sidebar.")
    btn_col, _ = st.columns([1, 7])
    analyze = btn_col.button("Analyze", type="primary")

    should_run = analyze or (load_ex != "(paste your own)") or (uploaded is not None)

    # ── Results ───────────────────────────────────────────────────────────────
    if should_run and transcript_input.strip():
        result = score_text(transcript_input)
        score  = result["score"]

        if score >= 70:
            verdict, bar_color, d_color = "COMMITTED", "#2ecc71", "normal"
        elif score >= 40:
            verdict, bar_color, d_color = "NEUTRAL",   "#f39c12", "off"
        else:
            verdict, bar_color, d_color = "HEDGING",   "#e74c3c", "inverse"

        st.markdown("---")
        st.markdown("### Results")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric(
            "CAPEX Conviction Score", f"{score:.0f} / 100",
            delta=verdict, delta_color=d_color,
            help="0 = pure hedging language, 100 = maximum commitment language. "
                 "Scores above 70 indicate a CEO making concrete, measurable CAPEX commitments. "
                 "Scores below 40 indicate deliberate softening of prior guidance.",
        )
        r2.metric(
            "Conviction Signal", f"{result['conviction_norm']:.1f}",
            delta=f"{len(result['conviction_matches'])} unique phrases",
            help="Normalized conviction score per 1,000 words. "
                 "Reflects density of concrete financial commitment language.",
        )
        r3.metric(
            "Hedge Signal", f"{result['hedge_norm']:.1f}",
            delta=f"{len(result['hedge_matches'])} unique phrases",
            delta_color="inverse" if result["hedge_norm"] > 20 else "normal",
            help="Normalized hedge score per 1,000 words. "
                 "High values suggest the CEO is walking back or softening prior commitments.",
        )
        r4.metric(
            "Word Count", f"{result['word_count']:,}",
            delta="tokens analyzed", delta_color="off",
            help="Total words analyzed. Scores are normalized per 1,000 words "
                 "so short and long transcripts are comparable.",
        )

        st.markdown("")
        st.markdown(
            f"""
            <div style="margin:4px 0 6px; font-size:0.78rem; color:#7d8590;
                        font-weight:600; letter-spacing:0.06em; text-transform:uppercase;">
                Conviction Meter &mdash; <span style="color:{bar_color}">{verdict}</span>
            </div>
            <div style="background:#21262d; border-radius:6px; height:20px; overflow:hidden;">
                <div style="width:{score}%; height:20px; border-radius:6px;
                            background:linear-gradient(90deg, {bar_color}55, {bar_color});">
                </div>
            </div>
            <div style="display:flex; justify-content:space-between;
                        font-size:0.72rem; color:#7d8590; margin-top:5px;">
                <span>0 &mdash; Hedging</span>
                <span>50 &mdash; Neutral</span>
                <span>100 &mdash; Committed</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        ph_col, pc_col = st.columns(2)

        with ph_col:
            st.markdown("#### Hedge Phrases Detected")
            st.caption("Higher impact = more weight toward a lower conviction score")
            if result["hedge_matches"]:
                hdf = pd.DataFrame(
                    sorted(result["hedge_matches"].items(), key=lambda x: -x[1]),
                    columns=["Phrase", "Count"],
                )
                hdf["Weight"] = hdf["Phrase"].map(HEDGE_WORDS).fillna(1.0)
                hdf["Impact"] = (hdf["Count"] * hdf["Weight"]).round(1)
                st.dataframe(
                    hdf, use_container_width=True, hide_index=True,
                    column_config={
                        "Impact": st.column_config.ProgressColumn(
                            "Impact", min_value=0, max_value=20, format="%.1f"
                        )
                    },
                )
            else:
                st.success("No hedge phrases detected.")

        with pc_col:
            st.markdown("#### Conviction Phrases Detected")
            st.caption("Higher impact = more weight toward a higher conviction score")
            if result["conviction_matches"]:
                cdf = pd.DataFrame(
                    sorted(result["conviction_matches"].items(), key=lambda x: -x[1]),
                    columns=["Phrase", "Count"],
                )
                cdf["Weight"] = cdf["Phrase"].map(CONVICTION_WORDS).fillna(1.0)
                cdf["Impact"] = (cdf["Count"] * cdf["Weight"]).round(1)
                st.dataframe(
                    cdf, use_container_width=True, hide_index=True,
                    column_config={
                        "Impact": st.column_config.ProgressColumn(
                            "Impact", min_value=0, max_value=20, format="%.1f"
                        )
                    },
                )
            else:
                st.warning("No conviction phrases detected.")

        with st.expander("Raw transcript text"):
            st.text(transcript_input[:4_000])
            if len(transcript_input) > 4_000:
                st.caption(f"Showing first 4,000 of {len(transcript_input):,} characters.")

    else:
        # Default: pre-scored comparison table
        if TRANSCRIPTS.exists():
            rows = []
            for fp in sorted(TRANSCRIPTS.glob("*.txt")):
                ticker = fp.stem.split("_")[0]
                try:
                    text = fp.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                r = score_text(text)
                verdict = (
                    "COMMITTED" if r["score"] >= 70 else
                    ("NEUTRAL"  if r["score"] >= 40 else "HEDGING")
                )
                rows.append({
                    "Ticker":              ticker,
                    "Conviction Score":    r["score"],
                    "Conviction Signal":   r["conviction_norm"],
                    "Hedge Signal":        r["hedge_norm"],
                    "Net Diff":            round(r["conviction_norm"] - r["hedge_norm"], 1),
                    "Verdict":             verdict,
                })

            if rows:
                st.markdown("### Sample Transcript Comparison")
                st.caption(
                    "Illustrative transcripts only — not actual earnings calls. "
                    "AMZN transcript was written with deliberate hedging language to demonstrate detection."
                )
                comp = pd.DataFrame(rows).sort_values("Conviction Score", ascending=False)
                st.dataframe(comp, use_container_width=True, hide_index=True)

        st.info(
            "Select a sample from the sidebar, upload a .txt file, or paste text above "
            "then click **Analyze**."
        )

    st.markdown(
        '<div class="disclaimer">'
        'Conviction scores are algorithmic estimates based on lexical analysis only. '
        'They do not constitute investment advice and should not be used as the sole basis '
        'for any financial decision. All sample transcripts are illustrative and fictional.'
        '</div>',
        unsafe_allow_html=True,
    )
