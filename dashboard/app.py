"""
AI Data Center Research Dashboard
===================================
Master Streamlit dashboard wrapping all three analytical tools:
  1. Circular Financing Network  (networkx + plotly)
  2. 1GW Data Center P&L Monte Carlo  (numpy + matplotlib)
  3. Earnings Call CAPEX Conviction Analyzer  (rule-based NLP + FinBERT optional)

Run:
    streamlit run app.py
"""

import math
import re
import warnings
from collections import defaultdict
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

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="AI Data Center Research",
    page_icon=":building_construction:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }

/* Page title / subtitle */
.page-title {
    font-size: 1.9rem;
    font-weight: 700;
    color: #e6edf3;
    letter-spacing: -0.4px;
    margin-bottom: 2px;
    line-height: 1.2;
}
.page-subtitle {
    font-size: 0.83rem;
    color: #7d8590;
    margin-bottom: 18px;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    padding: 14px 16px !important;
}

/* Sidebar */
[data-testid="stSidebar"] > div:first-child {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] .stRadio > label {
    color: #c9d1d9 !important;
}

/* Divider */
hr { border-color: #21262d !important; }

/* DataFrames */
[data-testid="stDataFrame"] {
    border: 1px solid #30363d;
    border-radius: 6px;
}

/* Buttons */
[data-testid="baseButton-primary"] {
    background: #1f6feb !important;
    border-color: #1f6feb !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


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
    st.markdown("**Data Sources**")
    st.caption("All figures sourced from public filings, press releases, and analyst estimates. Sample data for illustrative purposes.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — CIRCULAR FINANCING MAP
# ═════════════════════════════════════════════════════════════════════════════
if page == "Circular Financing Map":

    st.markdown('<div class="page-title">AI Circular Financing Network</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        'Directed cash-flow map between major AI data center players (2023–2025) — '
        'hover nodes and edges for detail, scroll to zoom, drag to pan.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Financing Map Controls")

        flow_filter = st.multiselect(
            "Flow Types",
            list(FLOW_COLORS.keys()),
            default=list(FLOW_COLORS.keys()),
        )
        min_deal = st.slider("Min Deal Size ($B)", 0.0, 15.0, 0.0, 0.5)
        show_labels = st.toggle("Show $ Labels on Edges", value=False)
        node_scale  = st.slider("Node Size Scale", 0.5, 2.0, 1.0, 0.1)

    # ── Load & filter data ────────────────────────────────────────────────────
    @st.cache_data
    def load_deals() -> pd.DataFrame:
        return pd.read_csv(DEALS_CSV)

    df_all = load_deals()
    df = df_all[
        df_all["flow_type"].isin(flow_filter) &
        (df_all["amount_billions"] >= min_deal)
    ].copy()

    # ── Top-line metrics ──────────────────────────────────────────────────────
    total_flow    = df_all["amount_billions"].sum()
    nvidia_in     = df_all[df_all["target"] == "NVIDIA"]["amount_billions"].sum()
    msft_out      = df_all[df_all["source"] == "Microsoft"]["amount_billions"].sum()
    openai_out    = df_all[df_all["source"] == "OpenAI"]["amount_billions"].sum()
    hw_total      = df_all[df_all["flow_type"] == "Hardware Purchase"]["amount_billions"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Tracked Flow",    f"${total_flow:.0f}B",   delta=f"{len(df_all)} deals")
    c2.metric("NVIDIA Hardware Inflow",f"${nvidia_in:.0f}B",    delta="Largest single beneficiary", delta_color="normal")
    c3.metric("Microsoft Deployed",    f"${msft_out:.0f}B",     delta="OpenAI + NVIDIA + CoreWeave", delta_color="off")
    c4.metric("OpenAI Compute Spend",  f"${openai_out:.0f}B",   delta="Oracle + CoreWeave contracts", delta_color="off")

    st.markdown("---")

    if df.empty:
        st.warning("No deals match the current filters. Adjust the sidebar controls.")
    else:
        # ── Build graph ───────────────────────────────────────────────────────
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

        # Circular layout (top-most = first node alphabetically)
        n = G.number_of_nodes()
        pos = {}
        for i, nd in enumerate(all_nodes):
            a = 2 * math.pi * i / n - math.pi / 2
            pos[nd] = (math.cos(a), math.sin(a))

        max_w  = agg["amount_billions"].max() or 1
        traces = []
        annots = []

        # Edge traces + arrowheads
        for (src, tgt), edata in G.edges.items():
            x0, y0 = pos[src]
            x1, y1 = pos[tgt]
            w      = edata["weight"]
            lw     = 1.5 + 10 * (w / max_w)
            color  = FLOW_COLORS.get(edata["flow_type"], "#aaaaaa")

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
                    bgcolor="#21262d",
                    borderpad=2,
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

        # Legend dummy traces
        for ft, fc in FLOW_COLORS.items():
            if ft in flow_filter:
                traces.append(go.Scatter(
                    x=[None], y=[None], mode="lines",
                    line=dict(width=5, color=fc),
                    name=ft, showlegend=True,
                ))

        # Node trace
        total_out = df.groupby("source")["amount_billions"].sum().to_dict()
        nx_x, nx_y, nx_txt, nx_hov, nx_col, nx_sz = [], [], [], [], [], []
        for nd in all_nodes:
            x, y = pos[nd]
            nx_x.append(x)
            nx_y.append(y)
            nx_txt.append(f"<b>{nd}</b>")
            out = total_out.get(nd, 0)
            nx_hov.append(
                f"<b>{nd}</b><br>"
                f"Outbound flow: ${out:.1f}B<br>"
                f"Total connections: {G.degree(nd)}"
            )
            nx_col.append(COMPANY_COLORS.get(nd, "#cccccc"))
            raw_sz = (16 + out * 1.3) * node_scale
            nx_sz.append(max(26, min(68, raw_sz)))

        traces.append(go.Scatter(
            x=nx_x, y=nx_y,
            mode="markers+text",
            marker=dict(
                size=nx_sz,
                color=nx_col,
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
            xaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False,
                range=[-1.55, 1.55],
            ),
            yaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False,
                range=[-1.55, 1.55],
            ),
            annotations=annots,
            legend=dict(
                bgcolor="#161b22",
                bordercolor="#30363d",
                borderwidth=1,
                font=dict(color="#c9d1d9"),
                itemsizing="constant",
            ),
            margin=dict(l=10, r=10, t=10, b=10),
            height=640,
            hovermode="closest",
            hoverlabel=dict(
                bgcolor="#161b22",
                bordercolor="#30363d",
                font=dict(color="#e6edf3"),
            ),
        )

        st.plotly_chart(fig_net, use_container_width=True)

    # Deal data table
    with st.expander("Raw Deal Data", expanded=False):
        st.dataframe(
            df_all[["source", "target", "flow_type", "amount_billions", "year", "notes"]]
            .sort_values("amount_billions", ascending=False),
            use_container_width=True,
            hide_index=True,
            column_config={
                "amount_billions": st.column_config.NumberColumn(
                    "Amount ($B)", format="$%.1f B"
                ),
                "year": st.column_config.NumberColumn("Year", format="%d"),
            },
        )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — P&L MONTE CARLO
# ═════════════════════════════════════════════════════════════════════════════
elif page == "P&L Monte Carlo":

    st.markdown('<div class="page-title">1GW AI Data Center P&L Modeler</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        'Monte Carlo stress-test: 10,000 scenarios, 5-year depreciation horizon. '
        'Adjust the sidebar sliders to instantly update all charts and metrics.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Sidebar sliders ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Monte Carlo Inputs")

        gpu_cost    = st.slider("GPU Unit Cost ($k)",        20, 60,    30,    5,
                                help="Per-GPU cost: H100 ~$30k, Blackwell B200 ~$40k")
        power_kwh   = st.slider("Power Cost ($/kWh)",        0.02, 0.12, 0.045, 0.005,
                                format="$%.3f",
                                help="Wholesale datacenter power rate")
        pue         = st.slider("PUE",                       1.10, 1.80, 1.35,  0.05,
                                help="Power Usage Effectiveness: 1.0=perfect, 1.5=poor")
        gpu_price   = st.slider("GPU Rental ($/GPU/hr)",     1.0,  8.0,  2.80,  0.10,
                                format="$%.2f",
                                help="Base H100/Blackwell spot-market rate")
        compression = st.slider("Annual Price Compression (%)", 0, 40, 12, 1,
                                help="Rental price decay per year as supply scales")
        utilization = st.slider("GPU Utilization (%)",       50, 100,  85,    1)
        equity_pct  = st.slider("Equity Fraction (%)",       10,  60,  30,    5,
                                help="% of CAPEX funded by equity; remainder = 6.5% debt")
        st.markdown("---")
        n_sims      = st.select_slider("Simulations", [1_000, 5_000, 10_000, 25_000], value=10_000)
        target_x    = st.number_input("Target Equity Multiple (x)",
                                      min_value=2.0, max_value=50.0, value=15.0, step=1.0)

    # ── Simulation (cached on parameter hash) ─────────────────────────────────
    @st.cache_data(show_spinner=False)
    def run_mc(
        gpu_k: int, p_kwh: float, pue_v: float, rental: float,
        compress_p: int, util_p: int, eq_frac: float,
        N: int, tgt: float, seed: int = 42,
    ) -> dict:
        np.random.seed(seed)
        YRS = 5

        # Sizing (fixed 1GW @ 40kW/rack, 8 GPUs/rack)
        total_gpus     = (1_000 * 1_000 / 40) * 8          # 200,000
        gpu_capex      = total_gpus * gpu_k * 1_000
        cooling_capex  = 1_000 * 2_500_000
        building_capex = 1_000 * 1_500_000
        total_capex    = gpu_capex + cooling_capex + building_capex
        equity_inv     = total_capex * eq_frac
        debt           = total_capex * (1 - eq_frac)
        annual_int     = debt * 0.065
        annual_depr    = total_capex / YRS

        # Stochastic draws
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
            power_kw = total_gpus * 0.70 * pue_v          # ~700W TDP per GPU
            pcost    = power_kw * HOURS * power_costs
            gp       = revenue - pcost
            ebitda   = gp - revenue * 0.05                # 5% OpEx
            ebit     = ebitda - annual_depr
            ebt      = ebit - annual_int
            tax      = np.maximum(ebt * 0.21, 0)
            net      = ebt - tax
            cum_net += net
            cum_rev += revenue
            if yr == YRS:
                gm_yr5 = gp / np.maximum(revenue, 1)

        terminal_v = (cum_rev / YRS) * 0.35 * 5           # 5x EV/EBITDA exit
        multiple   = (cum_net + terminal_v) / equity_inv

        return dict(
            multiple=multiple,
            gm_yr5=gm_yr5,
            total_capex=total_capex,
            equity_inv=equity_inv,
            total_gpus=total_gpus,
        )

    with st.spinner("Running Monte Carlo simulation..."):
        mc   = run_mc(gpu_cost, power_kwh, pue, gpu_price, compression,
                      utilization, equity_pct / 100, n_sims, target_x)
        mult = mc["multiple"]
        gm   = mc["gm_yr5"]

    p10, p50, p90 = np.percentile(mult, [10, 50, 90])
    hit_pct    = float((mult >= target_x).mean() * 100)
    bust_pct   = float((mult < 0).mean()  * 100)
    be_pct     = float((mult >= 1.0).mean() * 100)
    median_gm  = float(np.median(gm) * 100)

    # ── Metric row ────────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(
        "Total CAPEX",
        f"${mc['total_capex'] / 1e9:.1f}B",
        delta=f"{mc['total_gpus']:,.0f} GPUs",
    )
    m2.metric(
        "P50 Equity Multiple",
        f"{p50:.1f}x",
        delta=f"Bear {p10:.1f}x  /  Bull {p90:.1f}x",
        delta_color="off",
    )
    m3.metric(
        f"Hit {target_x:.0f}x Target",
        f"{hit_pct:.1f}%",
        delta=f"of {n_sims:,} scenarios",
        delta_color="normal" if hit_pct >= 20 else "inverse",
    )
    m4.metric(
        "Bankruptcy Risk",
        f"{bust_pct:.1f}%",
        delta="scenarios below 0x",
        delta_color="inverse" if bust_pct > 5 else "normal",
    )
    m5.metric(
        "Median Yr-5 Gross Margin",
        f"{median_gm:.1f}%",
        delta=f"Break-even rate: {be_pct:.0f}%",
        delta_color="normal",
    )

    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")

    mult_c = np.clip(mult, -5, 60)
    bins   = np.linspace(mult_c.min(), mult_c.max(), 70)

    # Left: equity multiple histogram
    ax1.set_facecolor("#161b22")
    ax1.hist(mult_c[mult < 0],
             bins=bins, color="#e74c3c", alpha=0.9, label="Bankruptcy (<0x)")
    ax1.hist(mult_c[(mult >= 0) & (mult < target_x)],
             bins=bins, color="#f39c12", alpha=0.9, label=f"Middle (0–{target_x:.0f}x)")
    ax1.hist(mult_c[mult >= target_x],
             bins=bins, color="#2ecc71", alpha=0.9, label=f"Hit {target_x:.0f}x target")
    ax1.axvline(target_x, color="#ffffff", ls="--", lw=1.5, alpha=0.7)
    ax1.axvline(p50, color="#a29bfe", ls=":", lw=1.8, label=f"Median {p50:.1f}x")
    ax1.set_xlabel("Equity Return Multiple (5-yr)", color="#c9d1d9", fontsize=10)
    ax1.set_ylabel("Number of Scenarios",           color="#c9d1d9", fontsize=10)
    ax1.set_title("Equity Multiple Distribution",   color="#e6edf3", fontsize=11, pad=8)
    ax1.tick_params(colors="#c9d1d9")
    for sp in ax1.spines.values():
        sp.set_color("#30363d")
    ax1.legend(facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#c9d1d9", fontsize=8.5)

    # Probability annotations
    ax1.text(0.97, 0.97,
             f"P10: {p10:.1f}x\nP50: {p50:.1f}x\nP90: {p90:.1f}x\n\n"
             f"Hit {target_x:.0f}x: {hit_pct:.1f}%\nBankruptcy: {bust_pct:.1f}%",
             transform=ax1.transAxes, fontsize=8.5,
             va="top", ha="right", color="#c9d1d9",
             bbox=dict(boxstyle="round,pad=0.4",
                       facecolor="#21262d", edgecolor="#30363d"))

    # Right: year-5 gross margin distribution
    gm_pct = np.clip(gm, -0.2, 1.0) * 100
    ax2.set_facecolor("#161b22")
    ax2.hist(gm_pct, bins=60, color="#5dade2", alpha=0.9,
             edgecolor="#0d1117", linewidth=0.3)
    ax2.axvline(median_gm, color="#f1c40f", ls="--", lw=1.8,
                label=f"Median {median_gm:.1f}%")
    ax2.axvline(0, color="#e74c3c", ls=":", lw=1.2,
                alpha=0.7, label="Zero margin")
    ax2.set_xlabel("Year-5 Gross Profit Margin (%)", color="#c9d1d9", fontsize=10)
    ax2.set_ylabel("Number of Scenarios",            color="#c9d1d9", fontsize=10)
    ax2.set_title("Year-5 Gross Margin Distribution (Power + Compression Stress)",
                  color="#e6edf3", fontsize=11, pad=8)
    ax2.tick_params(colors="#c9d1d9")
    for sp in ax2.spines.values():
        sp.set_color("#30363d")
    ax2.legend(facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#c9d1d9", fontsize=8.5)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Scenario insight cards ────────────────────────────────────────────────
    with st.expander("Scenario Interpretation", expanded=True):
        ia, ib = st.columns(2)
        with ia:
            st.markdown("#### Bear Case (P10)")
            st.markdown(f"- Equity Multiple: **{p10:.2f}x**")
            st.markdown(f"- Driven by high power cost variance and aggressive {compression}%/yr rental compression")
            if p10 < 1.0:
                st.error(
                    f"P10 scenario returns only **{p10:.1f}x** — less than invested equity. "
                    "The bear case destroys equity value."
                )
            else:
                st.success(f"P10 breaks even at {p10:.1f}x. Equity is preserved in bear scenarios.")

        with ib:
            st.markdown("#### Bull Case (P90)")
            st.markdown(f"- Equity Multiple: **{p90:.2f}x**")
            st.markdown("- Requires low power costs, minimal downtime, and stable rental pricing")
            if p90 < target_x:
                st.warning(
                    f"Even the P90 scenario ({p90:.1f}x) misses the {target_x:.0f}x target. "
                    "Consider lowering GPU costs, increasing utilization, or raising rental pricing."
                )
            else:
                st.success(f"P90 hits **{p90:.1f}x** — exceeds the {target_x:.0f}x target in bull scenarios.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — EARNINGS SENTIMENT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Earnings Sentiment":

    st.markdown('<div class="page-title">Earnings Call CAPEX Conviction Analyzer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        'NLP-powered hedge detection for hyperscaler earnings calls. '
        'Paste any transcript snippet to get an instant conviction score — '
        'or load a sample from the sidebar.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Lexicons ──────────────────────────────────────────────────────────────
    HEDGE_WORDS: dict[str, float] = {
        "evaluating":          2.0,
        "re-assessing":        3.0,
        "reassessing":         3.0,
        "right-sizing":        3.0,
        "rightsizing":         3.0,
        "pacing":              1.5,
        "carefully":           1.0,
        "monitoring":          1.5,
        "watching":            1.0,
        "remain flexible":     2.0,
        "remaining flexible":  2.0,
        "long-term horizon":   2.0,
        "longer-term horizon": 2.0,
        "multi-year":          0.5,
        "early stages":        2.0,
        "too early":           2.5,
        "may need to":         2.0,
        "could accelerate":    1.5,
        "could decelerate":    2.5,
        "thoughtful":          1.0,
        "disciplined":         0.5,
        "deliberately":        1.0,
        "weighing on":         1.5,
        "weigh on":            1.5,
        "still in the":        1.5,
        "over-building":       1.5,
        "structural challenges": 2.0,
        "not locked into":     2.0,
    }

    CONVICTION_WORDS: dict[str, float] = {
        "free cash flow":      2.5,
        "operating income":    2.0,
        "gross margin":        2.0,
        "accretive":           3.0,
        "roi":                 3.0,
        "return on investment": 3.0,
        "deployed":            2.5,
        "commissioned":        2.5,
        "contracted":          3.5,
        "committed":           3.0,
        "take-or-pay":         4.0,
        "pre-leased":          3.5,
        "booked":              2.5,
        "shovels":             4.0,
        "under construction":  3.0,
        "broke ground":        3.0,
        "fully funded":        3.5,
        "board approved":      3.0,
        "purchase orders":     3.0,
        "record":              1.5,
        "backlog":             2.5,
        "year-over-year":      1.0,
        "year over year":      1.0,
        "utilization":         2.0,
        "in production":       2.5,
        "fully deployed":      3.0,
        "fully on track":      2.5,
        "on track":            1.5,
        "clear contractual":   3.5,
        "zero speculative":    4.0,
        "funded":              2.0,
        "accelerating":        2.0,
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

        # Negation inversion: "not evaluating" flips a hedge into a conviction signal
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

    # ── Sidebar: sample loader ────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Sentiment Controls")

        if TRANSCRIPTS.exists():
            ticker_opts = ["(paste your own)"] + [
                p.stem.split("_")[0]
                for p in sorted(TRANSCRIPTS.glob("*.txt"))
            ]
        else:
            ticker_opts = ["(paste your own)"]

        load_ex = st.selectbox("Load Sample Transcript", ticker_opts)

        st.markdown("---")
        st.caption(
            "The rule-based scorer runs instantly in-browser. "
            "Full FinBERT analysis is available via the CLI tool in `tool3_nlp/`."
        )

    preload = ""
    if load_ex != "(paste your own)" and TRANSCRIPTS.exists():
        matches = list(TRANSCRIPTS.glob(f"{load_ex}*.txt"))
        if matches:
            preload = matches[0].read_text(encoding="utf-8", errors="replace")

    # ── Text input ────────────────────────────────────────────────────────────
    transcript_input = st.text_area(
        "Paste earnings call transcript (or load a sample from the sidebar above)",
        value=preload,
        height=240,
        placeholder=(
            "Paste transcript text here...\n\n"
            "Example of HIGH hedge language:\n"
            "  'We are carefully evaluating our CAPEX commitments and right-sizing our build "
            "cadence on a longer-term horizon as we re-assess demand signals...'\n\n"
            "Example of HIGH conviction language:\n"
            "  'We have deployed $14B in committed, contracted infrastructure. "
            "Every GPU rack is pre-leased. Free cash flow is fully accretive.'"
        ),
    )

    btn_col, _ = st.columns([1, 7])
    analyze = btn_col.button("Analyze", type="primary")

    should_run = analyze or (load_ex != "(paste your own)" and bool(transcript_input.strip()))

    # ── Scoring results ───────────────────────────────────────────────────────
    if should_run and transcript_input.strip():
        result = score_text(transcript_input)
        score  = result["score"]

        if score >= 70:
            verdict, bar_color, d_color = "COMMITTED", "#2ecc71", "normal"
            verdict_emoji = "COMMITTED"
        elif score >= 40:
            verdict, bar_color, d_color = "NEUTRAL",   "#f39c12", "off"
            verdict_emoji = "NEUTRAL"
        else:
            verdict, bar_color, d_color = "HEDGING",   "#e74c3c", "inverse"
            verdict_emoji = "HEDGING"

        st.markdown("---")
        st.markdown("### Scoring Results")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("CAPEX Conviction Score",  f"{score:.0f} / 100",
                  delta=verdict_emoji, delta_color=d_color)
        r2.metric("Conviction Signal",       f"{result['conviction_norm']:.1f}",
                  delta=f"{len(result['conviction_matches'])} unique phrases")
        r3.metric("Hedge Signal",            f"{result['hedge_norm']:.1f}",
                  delta=f"{len(result['hedge_matches'])} unique phrases",
                  delta_color="inverse" if result["hedge_norm"] > 20 else "normal")
        r4.metric("Word Count",              f"{result['word_count']:,}",
                  delta="tokens analyzed", delta_color="off")

        # Conviction meter bar
        st.markdown("")
        st.markdown(
            f"""
            <div style="margin:4px 0 6px; font-size:0.78rem; color:#7d8590;
                        font-weight:600; letter-spacing:0.06em; text-transform:uppercase;">
                Conviction Meter
            </div>
            <div style="background:#21262d; border-radius:6px; height:20px; overflow:hidden;">
                <div style="width:{score}%; height:20px; border-radius:6px;
                            background:linear-gradient(90deg, {bar_color}55, {bar_color});
                            transition: width 0.4s ease;">
                </div>
            </div>
            <div style="display:flex; justify-content:space-between;
                        font-size:0.72rem; color:#7d8590; margin-top:5px;">
                <span>0 &mdash; Pure Hedge</span>
                <span>50 &mdash; Neutral</span>
                <span>100 &mdash; Fully Committed</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Phrase tables
        ph_col, pc_col = st.columns(2)

        with ph_col:
            st.markdown("#### Hedge Phrases Detected")
            if result["hedge_matches"]:
                hdf = pd.DataFrame(
                    sorted(result["hedge_matches"].items(), key=lambda x: -x[1]),
                    columns=["Phrase", "Count"],
                )
                hdf["Weight"] = hdf["Phrase"].map(HEDGE_WORDS).fillna(1.0)
                hdf["Impact"] = (hdf["Count"] * hdf["Weight"]).round(1)
                st.dataframe(
                    hdf,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Impact": st.column_config.ProgressColumn(
                            "Impact", min_value=0, max_value=20, format="%.1f"
                        )
                    },
                )
            else:
                st.success("No hedge phrases detected in this transcript.")

        with pc_col:
            st.markdown("#### Conviction Phrases Detected")
            if result["conviction_matches"]:
                cdf = pd.DataFrame(
                    sorted(result["conviction_matches"].items(), key=lambda x: -x[1]),
                    columns=["Phrase", "Count"],
                )
                cdf["Weight"] = cdf["Phrase"].map(CONVICTION_WORDS).fillna(1.0)
                cdf["Impact"] = (cdf["Count"] * cdf["Weight"]).round(1)
                st.dataframe(
                    cdf,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Impact": st.column_config.ProgressColumn(
                            "Impact", min_value=0, max_value=20, format="%.1f"
                        )
                    },
                )
            else:
                st.warning("No conviction phrases detected in this transcript.")

        with st.expander("View Raw Text (first 3000 chars)"):
            st.text(transcript_input[:3_000])

    else:
        # ── Default: comparison of all sample transcripts ─────────────────────
        if TRANSCRIPTS.exists():
            rows = []
            for fp in sorted(TRANSCRIPTS.glob("*.txt")):
                ticker = fp.stem.split("_")[0]
                text   = fp.read_text(encoding="utf-8", errors="replace")
                r      = score_text(text)
                verdict = (
                    "COMMITTED" if r["score"] >= 70 else
                    ("NEUTRAL"  if r["score"] >= 40 else "HEDGING")
                )
                rows.append({
                    "Ticker":            ticker,
                    "Conviction Score":  r["score"],
                    "Conviction Signal": r["conviction_norm"],
                    "Hedge Signal":      r["hedge_norm"],
                    "Net Diff":          round(r["conviction_norm"] - r["hedge_norm"], 1),
                    "Verdict":           verdict,
                })

            if rows:
                comp = (
                    pd.DataFrame(rows)
                    .sort_values("Conviction Score", ascending=False)
                )
                st.markdown("### Sample Transcript Comparison")
                st.dataframe(comp, use_container_width=True, hide_index=True)

        st.info(
            "Select a sample from the sidebar dropdown or paste transcript text above, "
            "then click **Analyze** to score it."
        )
