"""
AI Data Center Circular Financing Network Visualizer
=====================================================
Reads deals_data.csv and outputs an interactive financing_map.html
using networkx for graph construction and plotly for rendering.
"""

import os
import math
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

# ── Config ──────────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), "deals_data.csv")
OUTPUT_HTML = os.path.join(os.path.dirname(__file__), "financing_map.html")

# Company → brand color
COMPANY_COLORS = {
    "Microsoft": "#00a4ef",
    "OpenAI":    "#10a37f",
    "NVIDIA":    "#76b900",
    "CoreWeave": "#6f42c1",
    "Oracle":    "#f80000",
    "xAI":       "#1da1f2",
    "Amazon":    "#ff9900",
}

# Flow type → color
FLOW_COLORS = {
    "Equity Investment":  "#f4d03f",
    "Compute Spend":      "#5dade2",
    "Hardware Purchase":  "#ec7063",
}

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Drop rows with 0 amount (purely informational)
    df = df[df["amount_billions"] > 0].copy()
    return df


def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    companies = set(df["source"]) | set(df["target"])
    for c in companies:
        G.add_node(c)
    for _, row in df.iterrows():
        key = (row["source"], row["target"], row["flow_type"])
        if G.has_edge(row["source"], row["target"]):
            # Accumulate parallel flows into combined weight
            G[row["source"]][row["target"]]["weight"] += row["amount_billions"]
        else:
            G.add_edge(
                row["source"], row["target"],
                weight=row["amount_billions"],
                flow_type=row["flow_type"],
                notes=row["notes"],
            )
    return G


def compute_layout(G: nx.DiGraph) -> dict:
    """Circular layout — keeps companies evenly spaced."""
    nodes = list(G.nodes())
    n = len(nodes)
    pos = {}
    for i, node in enumerate(nodes):
        angle = 2 * math.pi * i / n
        pos[node] = (math.cos(angle), math.sin(angle))
    return pos


def edge_traces(G: nx.DiGraph, pos: dict, df: pd.DataFrame) -> list:
    traces = []
    max_weight = df["amount_billions"].max()

    for (src, tgt), data in G.edges.items():
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]

        # Line width scaled 1-12px
        weight = data["weight"]
        lw = 1 + 11 * (weight / max_weight)

        flow = data["flow_type"]
        color = FLOW_COLORS.get(flow, "#aaaaaa")

        hover = (
            f"<b>{src} → {tgt}</b><br>"
            f"Type: {flow}<br>"
            f"Amount: ${weight:.1f}B"
        )

        # Draw an invisible thick line for hover area
        traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(width=lw, color=color),
            hoverinfo="text",
            text=hover,
            name=flow,
            showlegend=False,
        ))

        # Arrowhead via annotation — stored separately, returned alongside
        traces.append({"arrow": True, "src": (x0, y0), "tgt": (x1, y1), "color": color})

    return traces


def node_trace(G: nx.DiGraph, pos: dict, df: pd.DataFrame) -> go.Scatter:
    x_vals, y_vals, texts, hovers, colors, sizes = [], [], [], [], [], []
    total_flows = df.groupby("source")["amount_billions"].sum().to_dict()

    for node in G.nodes():
        x, y = pos[node]
        x_vals.append(x)
        y_vals.append(y)
        texts.append(f"<b>{node}</b>")
        out = total_flows.get(node, 0)
        hovers.append(
            f"<b>{node}</b><br>Total outbound: ${out:.1f}B<br>"
            f"Degree: {G.degree(node)} connections"
        )
        colors.append(COMPANY_COLORS.get(node, "#cccccc"))
        # Node size scales with total spend
        sizes.append(max(30, min(70, 20 + out * 1.2)))

    return go.Scatter(
        x=x_vals,
        y=y_vals,
        mode="markers+text",
        marker=dict(size=sizes, color=colors, line=dict(width=2, color="#ffffff")),
        text=texts,
        textposition="top center",
        hovertext=hovers,
        hoverinfo="text",
        name="Companies",
    )


def legend_traces() -> list:
    """Dummy traces to render a legend for flow types."""
    traces = []
    for flow_type, color in FLOW_COLORS.items():
        traces.append(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(width=4, color=color),
            name=flow_type,
            showlegend=True,
        ))
    return traces


def build_figure(G: nx.DiGraph, pos: dict, df: pd.DataFrame) -> go.Figure:
    edge_data = edge_traces(G, pos, df)

    # Split real traces from arrow dicts
    real_edge_traces = [t for t in edge_data if isinstance(t, go.Scatter)]
    arrows = [t for t in edge_data if isinstance(t, dict) and t.get("arrow")]

    n_trace = node_trace(G, pos, df)
    leg = legend_traces()

    fig = go.Figure(data=real_edge_traces + [n_trace] + leg)

    # Build plotly annotations for arrowheads
    annotations = []
    for a in arrows:
        x0, y0 = a["src"]
        x1, y1 = a["tgt"]
        # Shorten to avoid overlapping node circle
        dx, dy = x1 - x0, y1 - y0
        length = math.sqrt(dx**2 + dy**2)
        shrink = 0.18
        ax = x0 + dx * shrink
        ay = y0 + dy * shrink
        annotations.append(dict(
            ax=ax, ay=ay,
            x=x1 - dx / length * 0.15,
            y=y1 - dy / length * 0.15,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.5,
            arrowwidth=1.5,
            arrowcolor=a["color"],
        ))

    fig.update_layout(
        title=dict(
            text="AI Data Center Circular Financing Network (2023-2025)",
            font=dict(size=22, color="#ffffff"),
            x=0.5,
        ),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#c9d1d9"),
        showlegend=True,
        legend=dict(
            bgcolor="#161b22",
            bordercolor="#30363d",
            borderwidth=1,
            font=dict(color="#c9d1d9"),
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        annotations=annotations,
        margin=dict(l=20, r=20, t=60, b=20),
        height=720,
        hovermode="closest",
    )
    return fig


def main():
    print("Loading data from:", CSV_PATH)
    df = load_data(CSV_PATH)
    print(f"  {len(df)} deal rows loaded")

    G = build_graph(df)
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    pos = compute_layout(G)
    fig = build_figure(G, pos, df)

    fig.write_html(OUTPUT_HTML, include_plotlyjs="cdn")
    print(f"\nOutput written -> {OUTPUT_HTML}")

    # Print summary table
    summary = (
        df.groupby(["source", "target", "flow_type"])["amount_billions"]
        .sum()
        .reset_index()
        .sort_values("amount_billions", ascending=False)
    )
    print("\n-- Deal Summary ------------------------------------------")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
