"""
AI Data Center Research Dashboard  —  dashboard/app.py
=======================================================
Simplified research dashboard covering:
  - Overview: market snapshot, key metrics
  - Deals & Capital Flows: network chart + sourced deal table
  - Active Projects: data centers under construction (what is NOT canceled)
  - Profitability & Margins: cloud P&L, GPU economics, break-even math
  - Underwriting & Growth: valuation multiples, bull/bear case, growth drivers

Run:  tool3_nlp/venv/Scripts/streamlit run dashboard/app.py
"""

import math
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
DEALS_CSV = ROOT / "tool1_network" / "deals_data.csv"

# ── Colors ────────────────────────────────────────────────────────────────────
COMPANY_COLORS = {
    "Microsoft": "#00a4ef",
    "OpenAI":    "#10a37f",
    "NVIDIA":    "#76b900",
    "CoreWeave": "#6f42c1",
    "Oracle":    "#f80000",
    "xAI":       "#1da1f2",
    "Amazon":    "#ff9900",
    "Meta":      "#0866ff",
    "Broadcom":  "#d94f0f",
    "Anthropic": "#c97b4b",
    "G42":       "#4a90d9",
}
FLOW_COLORS = {
    "Equity Investment": "#f4d03f",
    "Compute Spend":     "#5dade2",
    "Hardware Purchase": "#ec7063",
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Center Research",
    page_icon=":building_construction:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer { visibility: hidden; }

[data-testid="metric-container"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    padding: 14px 16px !important;
}
.ticker-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 8px 14px;
    text-align: center;
}
.ticker-sym   { color: #7d8590; font-size: 0.72rem; font-weight: 600; }
.ticker-price { color: #e6edf3; font-weight: 700; font-size: 1.05rem; }
.ticker-up    { color: #3fb950; font-size: 0.78rem; }
.ticker-down  { color: #f85149; font-size: 0.78rem; }
.ticker-flat  { color: #7d8590; font-size: 0.78rem; }
.src-link     { font-size: 0.73rem; color: #7d8590; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# EMBEDDED RESEARCH DATA
# ════════════════════════════════════════════════════════════════════════════

PROJECTS_ACTIVE = [
    dict(
        company="Meta", project="Hyperion", location="Richland Parish, Louisiana",
        scale="2 GW initial → 5 GW; 3,650+ acres (expanding)", capex_b=27.0,
        completion="2027–2029",
        terms=(
            "Joint venture with Blue Owl Capital. Meta controls operations; Blue Owl finances. "
            "Powered by gas turbines + solar (2.26 GW of turbines by 2029 + 1.5 GW solar). "
            "5,000 construction workers at peak (mid-2026). Phase 2: Meta quietly purchased "
            "1,400 additional adjacent acres in early 2026 — nearly double the original site."
        ),
        source="Fortune Feb 2026",
        url="https://fortune.com/2026/02/04/meta-hyperion-ai-data-center-louisiana-expansion/",
    ),
    dict(
        company="Microsoft", project="Global FY2025–2026 Build-Out", location="Virginia, Wisconsin, Arizona + 60+ countries",
        scale="$80B FY2025 → ~$120B/yr going forward", capex_b=80.0,
        completion="Ongoing",
        terms=(
            "Owned + leased capacity at hyperscale. Nuclear PPAs signed (Three Mile Island restart "
            "in Pennsylvania). Anchor off-taker for BlackRock GAIA fund ($30B equity raise, $100B+ "
            "with leverage). Post-DeepSeek pivot toward owned vs. leased capacity — "
            "canceled ~200–300 MW of leases Feb 2025 (see Stalled/Canceled below)."
        ),
        source="SemiAnalysis; Microsoft Investor Relations",
        url="https://newsletter.semianalysis.com/p/microsofts-ai-strategy-deconstructed",
    ),
    dict(
        company="xAI", project="Colossus 1 + 2 (Memphis Supercluster)", location="Memphis, Tennessee",
        scale="200k GPUs now; targeting 1M GPUs; 1 GW+ power", capex_b=7.0,
        completion="1M GPU target: 2026",
        terms=(
            "Full vertical integration: xAI owns chips, power generation, racks, and land. "
            "Powered by Tennessee Valley Authority (TVA) and Memphis Light Gas & Water (MLGW). "
            "Phase 1 (100k H100 GPUs) built in just 122 days — 10x faster than comparable "
            "Microsoft or Meta projects. Colossus 2 adds 100-acre adjacent sites, 200 MW already "
            "running. $20B capital raise uses SPV structure: $12.5B debt with GPUs as collateral."
        ),
        source="xAI blog; SemiAnalysis; HPCwire May 2025",
        url="https://www.hpcwire.com/2025/05/13/colossus-ai-hits-200000-gpus-as-musk-ramps-up-ai-ambitions/",
    ),
    dict(
        company="Oracle", project="Stargate Texas Campus", location="Abilene + Fort Worth, Texas (20+ sites)",
        scale="131k-GPU zettascale cluster online; 1 GW+ planned", capex_b=10.0,
        completion="Phase 1: active; full buildout 2025–2027",
        terms=(
            "Oracle operates as builder and operator for Stargate sites despite JV governance disputes. "
            "Oracle's 131k-GPU zettascale cluster is already the world's largest GPU supercomputer "
            "in cloud infrastructure. OpenAI signed a separate $300B, 5-yr bilateral deal directly "
            "with Oracle (starting 2027) — this proceeds independently of the stalled Stargate JV. "
            "Phase 1 = $10B of the original $500B Stargate pledge."
        ),
        source="TechCrunch Feb 2026; Oracle SEC filing Jun 2025",
        url="https://techcrunch.com/2026/02/28/billion-dollar-infrastructure-deals-ai-boom-data-centers-openai-oracle-nvidia-microsoft-google-meta/",
    ),
    dict(
        company="Amazon", project="AWS Global Infrastructure Expansion", location="US, Europe, Asia-Pacific",
        scale="$131B capex 2025 → $200B capex 2026 (largest absolute spend)", capex_b=200.0,
        completion="Ongoing — $200B guided for 2026",
        terms=(
            "Largest absolute capex of any company globally in 2026. OpenAI committed $38B over "
            "7 years for NVIDIA GB200/GB300 GPU access on EC2 UltraServers (announced 2025). "
            "Anthropic is AWS's exclusive AI lab partner — Anthropic models available via Amazon "
            "Bedrock and custom Amazon Trainium/Inferentia chips handle ~30% of inference workloads. "
            "AWS operating margin: 34.6% on $107B annualized revenue."
        ),
        source="Amazon Q4 2025 earnings; CNBC Feb 2026",
        url="https://www.cnbc.com/2026/02/06/google-microsoft-meta-amazon-ai-cash.html",
    ),
    dict(
        company="Google", project="Global Data Center Expansion", location="US, Europe, Asia-Pacific",
        scale="$91B capex 2025 → $175–185B capex 2026 (+98% YoY)", capex_b=180.0,
        completion="Ongoing — fastest % capex growth of all hyperscalers",
        terms=(
            "Fastest year-over-year capex growth rate of all hyperscalers (+98%). Google Cloud "
            "operating margin expanded from 17% to 24% in a single year. TPU v5/v6 handles "
            "internal training workloads, reducing NVIDIA dependency. OpenAI signed Google Cloud "
            "as a secondary cloud provider, diversifying away from Azure + CoreWeave. "
            "Google also invested ~$300M–$2B in Anthropic separately."
        ),
        source="Alphabet Q4 2025 earnings; CNBC Feb 2026",
        url="https://www.cnbc.com/2026/02/06/google-microsoft-meta-amazon-ai-cash.html",
    ),
    dict(
        company="CoreWeave (CRWV)", project="Multi-Site GPU Cloud Network", location="New Jersey, Illinois, Virginia, Texas, UK",
        scale="$55B contracted backlog; ~$1.4B quarterly revenue run-rate", capex_b=23.0,
        completion="IPO'd March 2025; ongoing expansion",
        terms=(
            "Neocloud model: builds and rents GPU clusters to AI labs. OpenAI ($22.4B) and Meta "
            "($14.2B) are anchor tenants. NVIDIA holds ~7% equity stake and agreed to buy $6.3B of "
            "CoreWeave cloud services. SPV financing structure: CoreWeave borrowed $2.6B against "
            "the committed $11.9B OpenAI contract. IPO'd at $40/share (March 2025) → stock up "
            "+200–300% by February 2026. Revenue grew 700%+ in FY2024 to $1.92B."
        ),
        source="CoreWeave S-1; Motley Fool Feb 2026; CNBC",
        url="https://www.fool.com/investing/2026/02/25/ai-stock-soared-since-ipo-still-but-coreweave/",
    ),
    dict(
        company="Consortium (MSFT + NVDA + xAI + BlackRock)", project="Aligned Data Centers Acquisition",
        location="Multiple US sites",
        scale="$40B acquisition", capex_b=40.0,
        completion="Closed October 2025",
        terms=(
            "Microsoft, NVIDIA, xAI, and BlackRock consortium acquired Aligned Data Centers for $40B. "
            "Signals that AI infrastructure is now treated as critical infrastructure comparable "
            "to utilities — attracting sovereign wealth funds, pension capital, and infrastructure "
            "investors alongside tech companies."
        ),
        source="CNBC October 2025",
        url="https://www.cnbc.com/amp/2025/10/15/nvidia-microsoft-blackrock-aligned-data-centers.html",
    ),
]

PROJECTS_STALLED = [
    dict(
        company="Stargate JV (SoftBank / OpenAI / Oracle / MGX)",
        status="Stalled — no staff hired, no active builds",
        issue=(
            "Despite the $500B pledge announced with President Trump in January 2025, the Stargate "
            "joint venture has hired no staff and is not actively developing any data centers more "
            "than a year after the announcement. OpenAI, Oracle, and SoftBank spent months "
            "fighting over who controls the data centers — disagreements over ownership structure, "
            "governance, and responsibility. Partners have pivoted to bilateral deals instead: "
            "OpenAI signed a direct $300B deal with Oracle and a $38B deal with AWS, bypassing "
            "the JV structure entirely. SoftBank's ability to fulfill its $100B commitment was also "
            "questioned by analysts given its balance sheet."
        ),
        source="The Decoder; Tom's Hardware; TechPortal Feb 2026",
        url="https://the-decoder.com/stargates-500-billion-ai-infrastructure-project-reportedly-stalls-over-unresolved-disputes-between-openai-oracle-and-softbank/",
    ),
    dict(
        company="Microsoft",
        status="Partial cancellation — ~2 GW of lease negotiations walked away from",
        issue=(
            "TD Cowen analysts (February 2025) reported Microsoft canceled agreements for "
            "~200–300 MW of U.S. leased data center capacity and walked away from negotiations "
            "to lease ~2 GW of additional capacity in the US and Europe. Microsoft cited "
            "'capacity optimization' and power/facility delays. The moves followed DeepSeek's "
            "January 2025 release, which raised questions about whether the AI buildout was "
            "overextended. Microsoft simultaneously reaffirmed its $80B FY2025 owned capex "
            "commitment, framing the lease cancellations as a strategic shift toward owned "
            "rather than leased infrastructure."
        ),
        source="Bloomberg; Fortune; DataCenterFrontier — February 2025",
        url="https://fortune.com/2025/02/24/microsoft-cancels-leases-for-ai-data-centers-analyst-says/",
    ),
]

FINANCIALS = [
    dict(
        company="AWS (Amazon)", rev_b=107.0, op_inc_b=37.0, margin_pct=34.6,
        yoy_growth=20, capex_b_2026=200.0,
        notes="Q3 2025 annualized. AWS generates ~35% of all Amazon operating income on ~17% of revenue.",
        source="Amazon Q3 2025 earnings",
        url="https://windowsforum.com/threads/q4-2025-cloud-ai-push-aws-azure-google-cloud-scale-and-margin.401419/",
    ),
    dict(
        company="Microsoft Intelligent Cloud", rev_b=160.0, op_inc_b=69.0, margin_pct=43.0,
        yoy_growth=20, capex_b_2026=120.0,
        notes="Q1 FY2026 segment. Azure grew 40% YoY. Includes Azure, GitHub, SQL Server.",
        source="Microsoft Q1 FY2026 earnings",
        url="https://siliconangle.com/2025/08/09/cloud-quarterly-azure-ai-pop-aws-supply-pinch-google-execution/",
    ),
    dict(
        company="Google Cloud", rev_b=55.0, op_inc_b=13.0, margin_pct=23.7,
        yoy_growth=32, capex_b_2026=180.0,
        notes="Q3 2025 annualized. Margin expanded 17% → 24% in one year — fastest improvement of the Big 3.",
        source="Alphabet Q3 2025 earnings",
        url="https://siliconangle.com/2025/08/09/cloud-quarterly-azure-ai-pop-aws-supply-pinch-google-execution/",
    ),
    dict(
        company="NVIDIA (Data Center segment)", rev_b=47.5, op_inc_b=28.5, margin_pct=60.0,
        yoy_growth=142, capex_b_2026=3.0,
        notes="FY2025 full year (ending Jan 2025). Single-quarter Q4 FY2025 data center revenue: ~$35.6B.",
        source="NVIDIA FY2025 earnings",
        url="https://techcrunch.com/2026/02/28/billion-dollar-infrastructure-deals-ai-boom-data-centers-openai-oracle-nvidia-microsoft-google-meta/",
    ),
    dict(
        company="CoreWeave", rev_b=1.92, op_inc_b=-1.2, margin_pct=-63.0,
        yoy_growth=700, capex_b_2026=23.0,
        notes="FY2024. Negative margin due to GPU debt service. $55B contracted backlog provides revenue visibility.",
        source="CoreWeave S-1 / IPO prospectus (2025)",
        url="https://www.fool.com/investing/2026/02/25/ai-stock-soared-since-ipo-still-but-coreweave/",
    ),
    dict(
        company="OpenAI", rev_b=3.4, op_inc_b=-5.0, margin_pct=-147.0,
        yoy_growth=100, capex_b_2026=0.0,
        notes="Annualized late 2024. Revenue from API + ChatGPT subscriptions. Does NOT own data centers — $1.15T committed to external compute over 10 years.",
        source="The Information; Bloomberg",
        url="https://tomtunguz.com/openai-hardware-spending-2025-2035",
    ),
]

GPU_DATA = [
    dict(gpu="H100 (SXM5 80GB)", purchase_k=32.5, ondemand=2.00, reserved=1.60, peak=9.0, peak_yr=2023),
    dict(gpu="H200 (SXM5 141GB)", purchase_k=37.5, ondemand=4.50, reserved=3.20, peak=10.5, peak_yr=2024),
    dict(gpu="B200 (Blackwell)",  purchase_k=40.0, ondemand=7.00, reserved=5.00, peak=7.0,  peak_yr=2025),
]

VALUATIONS = [
    dict(
        company="NVIDIA", mktcap_b=2800, rev_b=130.0, ev_rev=21.5, fwd_pe=35,
        verdict="Partially Justified",
        verdict_color="#f0a500",
        thesis=(
            "**Strengths:** ~60% operating margins in data center. Near-monopoly on AI training "
            "GPU supply (H100/H200/Blackwell). Every major hyperscaler and AI lab depends on "
            "NVIDIA hardware. The CUDA software ecosystem creates massive switching costs.\n\n"
            "**Risks:** Broadcom and AMD custom ASICs (Google TPU, Amazon Trainium, OpenAI's "
            "Broadcom deal) could erode 10–20% of addressable market by 2027. DeepSeek-style "
            "efficiency gains reduce training compute per model.\n\n"
            "**Math:** At $2.8T market cap and ~$130B FY2026E revenue (21.5x EV/Rev), NVIDIA "
            "requires ~25% revenue CAGR to justify on a 10-year DCF. Consensus: ~30% — so "
            "current price has limited margin of safety but is not wildly speculative."
        ),
        source="Bloomberg; SemiAnalysis",
        url="https://newsletter.semianalysis.com/p/microsofts-ai-strategy-deconstructed",
    ),
    dict(
        company="Microsoft", mktcap_b=3100, rev_b=265.0, ev_rev=11.7, fwd_pe=32,
        verdict="Reasonable",
        verdict_color="#3fb950",
        thesis=(
            "**Strengths:** Most defensible valuation in the group. Azure Intelligent Cloud at "
            "43% operating margins, growing 20% YoY. OpenAI exclusivity via Azure is a durable "
            "AI moat. GitHub Copilot + Microsoft 365 Copilot are highest-margin enterprise AI "
            "products at scale.\n\n"
            "**Risks:** $120B/yr capex is enormous — ROI depends on Azure AI revenue growing "
            ">30% YoY. Post-DeepSeek lease cancellations signal some uncertainty even internally.\n\n"
            "**Math:** At 11.7x EV/Revenue with 43% segment margins and 20% growth, valuation "
            "is the most grounded in the group. The stock is not cheap, but the cash flows justify it."
        ),
        source="Microsoft IR; SemiAnalysis",
        url="https://newsletter.semianalysis.com/p/microsofts-ai-strategy-deconstructed",
    ),
    dict(
        company="Amazon", mktcap_b=2400, rev_b=640.0, ev_rev=3.75, fwd_pe=38,
        verdict="Reasonable",
        verdict_color="#3fb950",
        thesis=(
            "**Strengths:** AWS at 34.6% margins generates disproportionate profit — ~35% of "
            "Amazon's total operating income from ~17% of revenue. $200B 2026 capex is the "
            "largest absolute spend but AWS is already a proven $107B annualized revenue business. "
            "Anthropic partnership provides frontier AI access; Bedrock is a defensible enterprise platform.\n\n"
            "**Risks:** AWS revenue growth (20% YoY) is slower than Azure (40%) and Google Cloud "
            "(32%). $200B capex is a massive bet — supply constraint (GPU allocation) may limit "
            "short-term revenue.\n\n"
            "**Math:** AWS segment alone (~$107B rev, 35% margin) is worth ~$1.5T at 20x EBIT. "
            "Remaining Amazon is essentially free at the current market cap."
        ),
        source="Amazon IR; CNBC",
        url="https://www.cnbc.com/2026/02/06/google-microsoft-meta-amazon-ai-cash.html",
    ),
    dict(
        company="CoreWeave (CRWV)", mktcap_b=46, rev_b=1.92, ev_rev=24.0, fwd_pe=None,
        verdict="Speculative",
        verdict_color="#d29922",
        thesis=(
            "**Strengths:** $55B contracted revenue backlog through 2029 provides unusual "
            "visibility. Revenue grew 700%+ in FY2024. Stock up 200–300% since March 2025 IPO. "
            "NVIDIA's equity ownership and CoreWeave's GPU expertise make it hard to replicate quickly.\n\n"
            "**Risks:** 60%+ of revenue from two customers (Microsoft + OpenAI). GPU rental spot "
            "rates compressed 60–70% from 2023 peaks — a 100k H100 cluster now earns at or "
            "below its break-even rate. SPV debt structure means cash flows are heavily committed. "
            "New capacity adds supply, which depresses pricing.\n\n"
            "**Math:** At $46B market cap vs $1.92B FY2024 revenue = 24x EV/Rev. Revenue must "
            "grow 5–8x to justify current multiple. Backlog supports this path, but execution "
            "risk and customer concentration make this a high-risk bet."
        ),
        source="CoreWeave filings; Motley Fool Feb 2026",
        url="https://www.fool.com/investing/2026/02/25/ai-stock-soared-since-ipo-still-but-coreweave/",
    ),
    dict(
        company="OpenAI (private)", mktcap_b=157, rev_b=3.4, ev_rev=46.2, fwd_pe=None,
        verdict="Aggressive",
        verdict_color="#d29922",
        thesis=(
            "**Strengths:** ChatGPT has 500M+ weekly active users. Revenue doubling annually. "
            "Azure exclusivity means Microsoft has strong incentive to help OpenAI succeed. "
            "GPT-5 and o3-series models maintain frontier leadership.\n\n"
            "**Risks:** Losing ~$5B per year at $3.4B revenue — the unit economics don't yet work. "
            "Committed $1.15T in infrastructure over 10 years while not profitable. Restructuring "
            "to for-profit entity is necessary for IPO but creates governance complexity. "
            "DeepSeek showed frontier-quality results at a fraction of the cost.\n\n"
            "**Math:** $157B valuation at 46x revenue. Needs to reach ~$15B revenue at ~30% "
            "operating margins to justify on DCF. Revenue is growing fast (doubling YoY) but "
            "compute costs scale proportionally. A 2027 IPO target makes the math tractable only "
            "if revenue reaches $10–15B and losses narrow substantially."
        ),
        source="WSJ; Bloomberg; The Information",
        url="https://tomtunguz.com/openai-hardware-spending-2025-2035",
    ),
    dict(
        company="xAI (private)", mktcap_b=50, rev_b=0.5, ev_rev=100.0, fwd_pe=None,
        verdict="Highly Speculative",
        verdict_color="#f85149",
        thesis=(
            "**Strengths:** Elon Musk's execution is demonstrably fast — 122 days from ground "
            "to 100k H100 GPUs. Vertical integration (owns chips, power, racks) is a genuine "
            "competitive moat if GPU rental rates stay compressed. X platform provides a "
            "distribution channel for Grok that no other AI lab has.\n\n"
            "**Risks:** Revenue is largely undisclosed — estimated under $1B. $50B+ valuation "
            "implies >50x revenue multiple with no clear path to the scale needed. The capital "
            "raise ($20B with $12.5B debt, GPUs as collateral) is financially fragile if GPU "
            "rental rates stay at or below break-even.\n\n"
            "**Math:** At $50B valuation and ~$0.5B estimated revenue = ~100x EV/Revenue. "
            "Requires Grok to achieve $5B+ revenue at strong margins — unproven as of March 2026. "
            "This is effectively a venture bet, not an infrastructure investment."
        ),
        source="Bloomberg; Reuters; xAI blog",
        url="https://www.hpcwire.com/2025/05/13/colossus-ai-hits-200000-gpus-as-musk-ramps-up-ai-ambitions/",
    ),
]

GROWTH_DRIVERS = [
    dict(
        driver="Inference Demand Surge",
        magnitude="Very High",
        color="#3fb950",
        detail=(
            "DeepSeek proved AI training is getting cheaper — but this makes AI more accessible, "
            "driving exponentially more inference usage. Every ChatGPT query, Copilot suggestion, "
            "code completion, and AI agent step requires GPU cycles. Inference now drives more "
            "GPU-hours than training. Lower per-query costs expand the addressable market faster "
            "than they compress revenue per GPU."
        ),
        source="SemiAnalysis; TechCrunch Feb 2026",
        url="https://techcrunch.com/2026/02/28/billion-dollar-infrastructure-deals-ai-boom-data-centers-openai-oracle-nvidia-microsoft-google-meta/",
    ),
    dict(
        driver="Agentic AI — 5–10x Compute Multiplier",
        magnitude="Very High",
        color="#3fb950",
        detail=(
            "Autonomous AI agents completing multi-step tasks require 10–50 model calls per task "
            "vs. one for a single chat response. Microsoft Copilot agents, OpenAI o3-series "
            "reasoning models, and enterprise process automation all drive this multiplier. "
            "Early deployments suggest 5–10x more inference compute per active user vs. chat-only "
            "usage — a structural demand multiplier even as per-token costs fall."
        ),
        source="Microsoft earnings calls; SemiAnalysis",
        url="https://newsletter.semianalysis.com/p/microsofts-ai-strategy-deconstructed",
    ),
    dict(
        driver="Enterprise AI Adoption at Scale",
        magnitude="High",
        color="#2ecc71",
        detail=(
            "Fortune 500 companies are deploying AI copilots, coding assistants, and process "
            "automation. Azure AI, AWS Bedrock, and Google Vertex are driving commercial cloud "
            "revenue growth of 20–40% YoY. Enterprise contracts are the highest-margin, most "
            "predictable demand source — and adoption is still early. Microsoft Copilot has tens "
            "of millions of seats; GitHub Copilot is used by millions of developers."
        ),
        source="Azure/AWS/GCP quarterly earnings",
        url="https://siliconangle.com/2025/08/09/cloud-quarterly-azure-ai-pop-aws-supply-pinch-google-execution/",
    ),
    dict(
        driver="Sovereign AI Programs",
        magnitude="High",
        color="#2ecc71",
        detail=(
            "Nations are building national AI infrastructure to reduce dependency on US providers. "
            "UAE committed $100B (HUMAIN/MGX), Saudi Arabia is building NEOM AI cities, France "
            "announced a $6B+ national AI plan, India and Japan are each investing billions. "
            "Microsoft's G42 deal ($1.5B) and Oracle's OCI international expansion are designed "
            "to capture this demand. Sovereign AI creates a parallel buildout entirely outside "
            "the hyperscaler ecosystem."
        ),
        source="Reuters; Microsoft IR; Oracle IR",
        url="https://newsletter.semianalysis.com/p/microsofts-ai-strategy-deconstructed",
    ),
    dict(
        driver="Frontier Model Arms Race",
        magnitude="High",
        color="#2ecc71",
        detail=(
            "Each generation of frontier models (GPT-5, Gemini Ultra 2, Grok 3+, Claude 4+) "
            "requires 5–10x more compute than the prior generation. Training clusters are now "
            "100k–1M GPU scale. Even if per-token inference costs fall 90%, total training "
            "compute demand grows with model capability. xAI's target of 1M GPUs and OpenAI's "
            "$100B NVIDIA alliance are bets on continued exponential scaling."
        ),
        source="OpenAI; xAI; Google DeepMind",
        url="https://tomtunguz.com/openai-hardware-spending-2025-2035",
    ),
    dict(
        driver="Power Scarcity = Structural Pricing Moat",
        magnitude="Medium-High",
        color="#f0a500",
        detail=(
            "Power grid limitations are now the binding constraint — not chips or capital. "
            "Electrical transformer lead times: 18–24 months. Grid interconnect queue wait: "
            "3–5 years in some US markets. Permitting for new generation: 2–4 years. "
            "Operators who secured long-term power purchase agreements (PPAs) can charge "
            "premium rates for years before new supply comes online. Microsoft's Three Mile "
            "Island nuclear PPA provides a 20-year hedge. This moat is structural, not cyclical."
        ),
        source="DataCenterFrontier; DataCenterDynamics",
        url="https://www.datacenterfrontier.com/hyperscale/article/55310441/ownership-and-power-challenges-in-metas-hyperion-and-prometheus-data-centers",
    ),
]


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=timedelta(hours=1), show_spinner=False)
def fetch_market_data(tickers: tuple) -> dict:
    try:
        import yfinance as yf
        result = {}
        for tkr in tickers:
            try:
                info  = yf.Ticker(tkr).fast_info
                price = getattr(info, "last_price", None)
                prev  = getattr(info, "previous_close", None)
                mktcap = getattr(info, "market_cap", None)
                chg   = (price - prev) / prev * 100 if price and prev else None
                result[tkr] = dict(price=price, chg=chg, mktcap=mktcap)
            except Exception:
                result[tkr] = dict(price=None, chg=None, mktcap=None)
        return result
    except ImportError:
        return {}


@st.cache_data(show_spinner=False)
def load_deals() -> pd.DataFrame:
    try:
        return pd.read_csv(DEALS_CSV)
    except Exception as e:
        st.error(f"Could not load deals data: {e}")
        return pd.DataFrame()


def build_network_fig(df: pd.DataFrame, show_labels: bool = False) -> go.Figure:
    G = nx.DiGraph()
    all_nodes = sorted(set(df["source"]) | set(df["target"]))
    G.add_nodes_from(all_nodes)

    agg = (
        df.groupby(["source", "target", "flow_type"])["amount_billions"]
        .sum().reset_index()
    )
    for _, row in agg.iterrows():
        G.add_edge(row["source"], row["target"],
                   weight=row["amount_billions"], flow_type=row["flow_type"])

    n = max(len(all_nodes), 1)
    pos = {}
    for i, nd in enumerate(all_nodes):
        a = 2 * math.pi * i / n - math.pi / 2
        pos[nd] = (math.cos(a), math.sin(a))

    max_w = agg["amount_billions"].max() or 1
    traces, annots = [], []

    for (src, tgt), edata in G.edges.items():
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]
        w      = edata["weight"]
        # Log scale so massive deals ($350B) don't drown out smaller ones
        lw     = 1.5 + 7 * (math.log1p(w) / math.log1p(max_w))
        color  = FLOW_COLORS.get(edata["flow_type"], "#aaa")

        traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines", line=dict(width=lw, color=color),
            hoverinfo="text",
            text=f"<b>{src} → {tgt}</b><br>Type: {edata['flow_type']}<br>Amount: <b>${w:.1f}B</b>",
            showlegend=False,
        ))
        if show_labels:
            annots.append(dict(
                x=(x0 + x1) / 2, y=(y0 + y1) / 2,
                text=f"${w:.0f}B", showarrow=False,
                font=dict(size=8, color="#e6edf3"),
                bgcolor="#21262d", borderpad=2,
            ))
        dx, dy = x1 - x0, y1 - y0
        length = math.sqrt(dx**2 + dy**2) or 1
        annots.append(dict(
            ax=x0 + dx * 0.2, ay=y0 + dy * 0.2,
            x=x1 - dx / length * 0.14, y=y1 - dy / length * 0.14,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.6, arrowwidth=1.6,
            arrowcolor=color,
        ))

    for ft, fc in FLOW_COLORS.items():
        traces.append(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(width=5, color=fc),
            name=ft, showlegend=True,
        ))

    total_out = df.groupby("source")["amount_billions"].sum().to_dict()
    max_out   = max(total_out.values()) if total_out else 1
    node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
    for nd in all_nodes:
        x, y = pos[nd]
        node_x.append(x); node_y.append(y)
        out   = total_out.get(nd, 0)
        in_   = df[df["target"] == nd]["amount_billions"].sum()
        node_color.append(COMPANY_COLORS.get(nd, "#8b949e"))
        node_size.append(16 + 24 * math.sqrt(out / max_out))
        node_text.append(f"<b>{nd}</b><br>Outbound: ${out:.1f}B<br>Inbound: ${in_:.1f}B")

    traces.append(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=all_nodes,
        textposition="top center",
        marker=dict(size=node_size, color=node_color,
                    line=dict(width=2, color="#0d1117")),
        hovertext=node_text, hoverinfo="text",
        showlegend=False,
    ))

    return go.Figure(data=traces, layout=go.Layout(
        annotations=annots,
        showlegend=True,
        legend=dict(x=1.01, y=0.5, font=dict(size=11, color="#e6edf3")),
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        xaxis=dict(visible=False, range=[-1.5, 1.5]),
        yaxis=dict(visible=False, range=[-1.5, 1.5]),
        margin=dict(l=10, r=140, t=10, b=10),
        dragmode="pan", height=600,
    ))


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

TICKERS = ("MSFT", "NVDA", "AMZN", "ORCL", "CRWV", "META", "GOOGL")

with st.sidebar:
    st.markdown("### AI Data Center Research")
    st.caption("Sourced quantitative research — March 2026")
    st.markdown("---")
    if st.button("Refresh Live Data", use_container_width=True,
                 help="Clears cached market prices and reloads"):
        st.cache_data.clear()
        st.rerun()
    st.markdown(
        f"<div style='font-size:0.72rem;color:#484f58;text-align:center;margin-top:6px'>"
        f"Prices refresh hourly &nbsp;|&nbsp; Loaded {datetime.now().strftime('%H:%M')}"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption(
        "All figures sourced from public filings, press releases, and analyst reports. "
        "For research purposes only. Not financial advice."
    )


# ════════════════════════════════════════════════════════════════════════════
# HEADER + LIVE TICKER STRIP
# ════════════════════════════════════════════════════════════════════════════

st.markdown("# AI Data Center Research Dashboard")
st.caption(
    "Capital flows, active projects, profitability, and underwriting analysis — "
    "sourced from public filings and analyst reports, updated March 2026"
)

with st.spinner("Loading live prices..."):
    mkt = fetch_market_data(TICKERS)

ticker_cols = st.columns(len(TICKERS))
for i, tkr in enumerate(TICKERS):
    d     = mkt.get(tkr, {})
    price = d.get("price")
    chg   = d.get("chg")
    with ticker_cols[i]:
        if price:
            chg_str = f"{chg:+.2f}%" if chg is not None else "—"
            cls     = "ticker-up" if (chg or 0) >= 0 else "ticker-down"
            st.markdown(
                f'<div class="ticker-card">'
                f'<div class="ticker-sym">{tkr}</div>'
                f'<div class="ticker-price">${price:,.2f}</div>'
                f'<div class="{cls}">{chg_str}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="ticker-card">'
                f'<div class="ticker-sym">{tkr}</div>'
                f'<div class="ticker-flat">N/A</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

st.markdown("---")


# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Deals & Capital Flows",
    "Active Projects",
    "Profitability & Margins",
    "Underwriting & Growth",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### The Largest Coordinated Capital Deployment in Corporate History")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "2026 Hyperscaler Capex", "$700B", "+36% vs 2025",
        help="Combined 2026 capex guidance: Amazon $200B + Google $180B + Microsoft $120B + Meta $125B + Oracle. Source: CNBC Feb 2026.",
    )
    m2.metric(
        "NVIDIA DC Revenue (FY2025)", "$47.5B", "+142% YoY",
        help="NVIDIA data center segment FY2025 (ending Jan 2025). ~60% operating margins. Q4 FY2025 alone: ~$35.6B.",
    )
    m3.metric(
        "OpenAI Infra Commitments", "$1.15T", "Over 10 years",
        help="Total committed hardware and cloud spending: Broadcom $350B, Oracle $300B, Microsoft $250B, NVIDIA $100B, AMD $90B, AWS $38B, CoreWeave $22B+. Source: Tunguz analysis.",
    )
    m4.metric(
        "Combined Capex Growth", "+36% YoY", "$515B (2025) → $700B (2026)",
        help="Year-over-year capex growth across Amazon, Google, Microsoft, Meta. Source: IEEE ComSoc; CNBC Feb 2026.",
    )

    st.markdown("---")

    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown("#### What Is Happening")
        st.markdown("""
The five hyperscalers — Amazon, Google, Microsoft, Meta, and Oracle — are spending **nearly
$700 billion in 2026 alone** on AI data centers, up 36% from 2025. This rivals the GDP of Sweden
and is among the largest coordinated capital deployments in modern economic history.

**The circular economy of AI capital:**
Microsoft invested $13B in OpenAI → OpenAI spends that on Azure compute → Azure uses that revenue
to buy NVIDIA GPUs → NVIDIA invests $100B back into OpenAI → repeat. Bloomberg's analysis of these
circular flows shows the same dollars cycling through the ecosystem multiple times, inflating
headline figures.

**The binding constraint is power, not capital or chips.**
Electrical transformer lead times are 18–24 months. Grid interconnect queues run 3–5 years in some
US markets. Operators with secured long-term power purchase agreements hold the real moat. This is
why Meta is building a 5 GW gas turbine plant in Louisiana and Microsoft signed a 20-year nuclear PPA.

**Key tension to watch:**
DeepSeek's January 2025 release showed frontier AI results at ~$6M training cost vs. hundreds of
millions for US peers. If efficiency gains outpace demand growth, the economics of this buildout
deteriorate significantly. Microsoft canceled 2 GW of leases; the $500B Stargate JV has stalled.
        """)
    with col_r:
        st.markdown("#### Capex Summary by Company")
        capex_df = pd.DataFrame({
            "Company": ["Amazon (AWS)", "Google (Alphabet)", "Microsoft", "Meta", "Oracle (est.)", "xAI (est.)"],
            "2025 CapEx": ["$131B", "$91B", "$80B", "$72B", "$15B+", "$7B+"],
            "2026 CapEx": ["$200B", "$175–185B", "~$120B", "$115–135B", "$20B+", "$20B+"],
            "YoY Change": ["+56%", "+98%", "+50%", "+74%", "—", "—"],
        })
        st.dataframe(capex_df, use_container_width=True, hide_index=True)

        st.markdown("#### Sources")
        st.markdown(
            '<span class="src-link">'
            '[CNBC Feb 2026](https://www.cnbc.com/2026/02/06/google-microsoft-meta-amazon-ai-cash.html) · '
            '[TechCrunch Feb 2026](https://techcrunch.com/2026/02/28/billion-dollar-infrastructure-deals-ai-boom-data-centers-openai-oracle-nvidia-microsoft-google-meta/) · '
            '[Tunguz OpenAI Analysis](https://tomtunguz.com/openai-hardware-spending-2025-2035) · '
            '[Bloomberg Circular Deals](https://www.bloomberg.com/graphics/2026-ai-circular-deals/)'
            '</span>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: DEALS
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Deals & Capital Flows")
    st.caption(
        "All major disclosed capital flows between AI infrastructure companies. "
        "Amounts are total contract value — multi-year commitments are shown at full face value, not present value."
    )

    df_all = load_deals()

    if not df_all.empty:
        total_flow = df_all["amount_billions"].sum()
        openai_out = df_all[df_all["source"] == "OpenAI"]["amount_billions"].sum()
        nvidia_in  = df_all[df_all["target"] == "NVIDIA"]["amount_billions"].sum()
        msft_out   = df_all[df_all["source"] == "Microsoft"]["amount_billions"].sum()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Tracked Capital", f"${total_flow:.0f}B", f"{len(df_all)} deals",
                  help="Sum of all disclosed deal face values. Note: circular flows mean the same capital may be counted more than once.")
        m2.metric("OpenAI Compute Commitments", f"${openai_out:.0f}B",
                  help="OpenAI's total compute and chip commitments. OpenAI owns no data centers.")
        m3.metric("NVIDIA Hardware Revenue", f"${nvidia_in:.0f}B",
                  help="Total GPU hardware purchase commitments flowing to NVIDIA.")
        m4.metric("Microsoft Total Deployed", f"${msft_out:.0f}B",
                  help="Microsoft's total outbound capital: equity + hardware + compute.")

        st.markdown("---")

        # Chart controls (inline, not sidebar)
        ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])
        with ctrl1:
            flow_filter = st.multiselect(
                "Show flow types", list(FLOW_COLORS.keys()),
                default=list(FLOW_COLORS.keys()),
            )
        with ctrl2:
            min_deal = st.slider("Min deal size ($B)", 0.0, 20.0, 0.0, 1.0)
        with ctrl3:
            show_labels = st.toggle("Show $ labels on edges", value=False)

        df_chart = df_all[
            df_all["flow_type"].isin(flow_filter) &
            (df_all["amount_billions"] >= min_deal)
        ]

        if df_chart.empty:
            st.warning("No deals match current filters.")
        else:
            st.plotly_chart(build_network_fig(df_chart, show_labels), use_container_width=True)

        st.markdown("---")
        st.markdown("#### Complete Deal Table — All Sources")

        # Build display table
        show_cols = [c for c in ["source", "target", "flow_type", "amount_billions", "year", "notes"] if c in df_all.columns]
        df_show = df_all[show_cols].copy()
        df_show.columns = ["Source", "Target", "Type", "Amount ($B)", "Year", "Notes"][:len(show_cols)]
        df_show["Amount ($B)"] = df_show["Amount ($B)"].map(lambda x: f"${x:.1f}B")
        st.dataframe(df_show, use_container_width=True, hide_index=True)

        # Source links
        if "source_url" in df_all.columns:
            with st.expander("Source Links for Every Deal"):
                for _, row in df_all.iterrows():
                    url = str(row.get("source_url", ""))
                    if url.startswith("http"):
                        st.markdown(
                            f"- **{row['source']} → {row['target']}** "
                            f"(${row['amount_billions']:.1f}B, {int(row['year'])}): "
                            f"[source]({url})"
                        )
    else:
        st.error("Failed to load deals data. Check that tool1_network/deals_data.csv exists.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: ACTIVE PROJECTS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Active Data Center Projects")
    st.caption(
        "Projects confirmed under construction or in active development as of March 2026. "
        "These have not been canceled. Each entry includes deal terms and a primary source."
    )

    for p in PROJECTS_ACTIVE:
        label = f"**{p['company']}** — {p['project']}   |   ${p['capex_b']:.0f}B   |   {p['location']}"
        with st.expander(label):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Status", "Active")
            c2.metric("Total CapEx", f"${p['capex_b']:.0f}B")
            c3.metric("Scale", p["scale"])
            c4.metric("Est. Completion", p["completion"])
            st.markdown(f"**Terms & Details:** {p['terms']}")
            st.markdown(
                f'<span class="src-link">Source: <a href="{p["url"]}" target="_blank">{p["source"]}</a></span>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("### Stalled / Canceled Projects")
    st.caption("Projects announced but not proceeding as described — including the biggest AI headline deal of 2025.")

    for s in PROJECTS_STALLED:
        with st.expander(f"**{s['company']}** — {s['status']}"):
            st.warning(s["issue"])
            st.markdown(
                f'<span class="src-link">Source: <a href="{s["url"]}" target="_blank">{s["source"]}</a></span>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: PROFITABILITY & MARGINS
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### Profitability & Margins")

    # ── Cloud segment P&L table ──────────────────────────────────────────────
    st.markdown("#### Cloud & AI Segment P&L")
    fin_rows = []
    for f in FINANCIALS:
        fin_rows.append({
            "Company":         f["company"],
            "Revenue (ann.)":  f"${f['rev_b']:.1f}B",
            "Op. Income":      f"${f['op_inc_b']:.1f}B",
            "Op. Margin":      f"{f['margin_pct']:.1f}%",
            "YoY Growth":      f"+{f['yoy_growth']}%" if f["yoy_growth"] > 0 else f"{f['yoy_growth']}%",
            "2026 CapEx":      f"${f['capex_b_2026']:.0f}B" if f["capex_b_2026"] > 0 else "—",
            "Notes":           f["notes"],
        })
    fin_df = pd.DataFrame(fin_rows)
    st.dataframe(fin_df, use_container_width=True, hide_index=True)

    with st.expander("P&L Sources"):
        for f in FINANCIALS:
            st.markdown(f"- **{f['company']}**: [{f['source']}]({f['url']})")

    st.markdown("---")

    # ── GPU economics table ──────────────────────────────────────────────────
    st.markdown("#### GPU Unit Economics — Purchase vs. Rental Rates")
    st.caption(
        "H100 rental rates have fallen 60–70% from 2023 peak. "
        "New clusters financed at full CapEx cost are at or below break-even at current spot rates."
    )
    gpu_rows = []
    for g in GPU_DATA:
        drop = (1 - g["ondemand"] / g["peak"]) * 100
        gpu_rows.append({
            "GPU":                   g["gpu"],
            "Purchase Price":        f"~${g['purchase_k']:.0f}K",
            "Cloud On-Demand /hr":   f"${g['ondemand']:.2f}",
            "Cloud Reserved /hr":    f"${g['reserved']:.2f}",
            f"Peak Rate ({g['peak_yr']})": f"${g['peak']:.1f}/hr",
            "Drop from Peak":        f"-{drop:.0f}%",
        })
    gpu_df = pd.DataFrame(gpu_rows)
    st.dataframe(gpu_df, use_container_width=True, hide_index=True)

    st.markdown(
        '<span class="src-link">Sources: '
        '[ThunderCompute Dec 2025](https://www.thundercompute.com/blog/ai-gpu-rental-market-trends) · '
        '[IntuitionLabs H100 Pricing 2026](https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison) · '
        '[GMI Cloud 2025 Analysis](https://www.gmicloud.ai/blog/nvidia-h100-gpu-pricing-2025-rent-vs-buy-cost-analysis)'
        '</span>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Break-even math ──────────────────────────────────────────────────────
    st.markdown("#### Break-Even Analysis: 100,000 H100 GPU Cluster")
    st.caption("At today's rental rates, new clusters are at or below break-even — the core financial stress in the neocloud sector.")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("""
**CapEx (total build)**

| Item | Estimated Cost |
|------|----------------|
| 100,000 H100 GPUs @ ~$32.5K each | $3.25B |
| Servers, networking, cooling infra | ~$2B |
| Data center construction + power | ~$0.5–1B |
| **Total CapEx** | **~$5.5–6.5B** |

**Annual OpEx**

| Item | Cost |
|------|------|
| Power: 70 MW GPU load, PUE 1.2, $60/MWh | ~$45M |
| Staff, maintenance, networking | ~$75M |
| **Total Annual OpEx** | **~$120M** |
""")
    with col_r:
        st.markdown("""
**Revenue Required to Break Even**

| Item | Amount |
|------|--------|
| CapEx annualized (5-yr straight-line) | ~$1.2B/yr |
| Annual OpEx | ~$120M |
| **Total revenue needed** | **~$1.3–1.4B/yr** |

**Required Rental Rate**

| Utilization | Break-Even Rate |
|-------------|-----------------|
| 70% utilization | **~$2.10–2.30/hr** |
| 55% utilization | **~$2.70–2.90/hr** |

**Current H100 spot rates: $1.50–$2.00/hr**

At today's rates, a cluster financed at full purchase cost is at or **below break-even** — which
explains emerging financing stress at neocloud operators like CoreWeave and why SPV/debt structures
are required to fund new GPU deployments.
""")

    st.markdown("---")

    # ── Capex vs. AI services revenue gap ───────────────────────────────────
    st.markdown("#### The Capex vs. Revenue Gap")
    st.markdown(
        "The Big Three cloud providers are spending ~$240B/year on data centers and servers, "
        "yet 2025 AI services revenue from all three is estimated at only ~$25B. "
        "The buildout is a multi-year bet on demand growth outpacing capital deployment."
    )
    gap_df = pd.DataFrame({
        "Company":       ["Amazon (AWS)", "Google Cloud", "Microsoft Intelligent Cloud"],
        "2025 CapEx":    ["$131B", "$91B", "$80B"],
        "2026 CapEx":    ["$200B", "$175–185B", "~$120B"],
        "Cloud Rev (ann.)": ["$107B", "$55B", "$160B"],
        "Cloud Op. Margin": ["34.6%", "23.7%", "43.0%"],
        "Cloud Rev Growth":  ["+20%", "+32%", "+20%"],
    })
    st.dataframe(gap_df, use_container_width=True, hide_index=True)
    st.markdown(
        '<span class="src-link">Sources: '
        '[Cloud Quarterly Analysis — SiliconAngle Aug 2025](https://siliconangle.com/2025/08/09/cloud-quarterly-azure-ai-pop-aws-supply-pinch-google-execution/) · '
        '[CNBC Hyperscaler Capex Feb 2026](https://www.cnbc.com/2026/02/06/google-microsoft-meta-amazon-ai-cash.html)'
        '</span>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: UNDERWRITING & GROWTH
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("### Underwriting Analysis")
    st.caption(
        "Are current valuations justified by the underlying business economics? "
        "Do the revenue multiples make sense given what the technology is actually delivering?"
    )

    # ── Valuation summary table ──────────────────────────────────────────────
    st.markdown("#### Valuation Multiples at a Glance")
    val_rows = []
    for v in VALUATIONS:
        val_rows.append({
            "Company":          v["company"],
            "Market Cap / Val": f"${v['mktcap_b']:,}B",
            "Revenue (ann.)":   f"${v['rev_b']:.1f}B",
            "EV / Revenue":     f"{v['ev_rev']:.1f}x",
            "Fwd P/E":          f"{v['fwd_pe']}x" if v["fwd_pe"] else "Not meaningful (losing $)",
            "Verdict":          v["verdict"],
        })
    val_df = pd.DataFrame(val_rows)
    st.dataframe(val_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Per-company detailed analysis ───────────────────────────────────────
    st.markdown("#### Detailed Analysis by Company")
    for v in VALUATIONS:
        with st.expander(
            f"**{v['company']}** — {v['verdict']}   |   "
            f"{v['ev_rev']:.1f}x EV/Revenue   |   "
            f"${v['mktcap_b']:,}B valuation"
        ):
            st.markdown(v["thesis"])
            st.markdown(
                f'<span class="src-link">Source: <a href="{v["url"]}" target="_blank">{v["source"]}</a></span>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Bull / Bear ──────────────────────────────────────────────────────────
    st.markdown("### Is the Tech Justifying the Multiples?")
    st.caption(
        "The central question: does the actual revenue growth from AI justify $700B/year in capital deployment "
        "and valuations at 20–100x revenue?"
    )

    col_bull, col_bear = st.columns(2)
    with col_bull:
        st.markdown("#### Bull Case — Yes, the multiples are justified")
        st.markdown("""
**1. Inference demand is not falling post-DeepSeek.**
Cheaper training expands the user base. Lower per-token costs mean more queries, not fewer GPUs.
The total compute demand keeps growing even as per-unit cost falls.

**2. Agentic AI multiplies compute structurally.**
A single enterprise AI agent executing a business process may make 20–50 model calls. Scale this
across millions of enterprise users and compute demand compounds at a rate that dwarfs chat usage.

**3. The contracted backlogs are real.**
CoreWeave's $55B backlog and OpenAI's $300B Oracle deal are legally committed obligations, not
speculative projections. This provides multi-year revenue visibility that justifies premium multiples.

**4. Power scarcity creates durable pricing power.**
Operators with locked-in power can't be easily undercut. New entrants face 3–5 year grid queues.
This structural bottleneck supports premium pricing for years.

**5. Sovereign AI creates demand outside the hyperscaler ecosystem.**
$100B+ in national AI programs (UAE, Saudi, France, Japan, India) adds incremental demand that
doesn't compete with — it supplements — hyperscaler buildout.
        """)
    with col_bear:
        st.markdown("#### Bear Case — The multiples are not justified")
        st.markdown("""
**1. GPU rental rates fell 60–70% from peak.**
A 100k H100 cluster financed at purchase cost earns at or below its break-even rate at current
spot prices. New supply continuously enters the market, maintaining downward pressure.

**2. Stargate's $500B pledge ≠ $500B spent.**
The biggest AI infrastructure announcement of 2025 has hired no staff and built nothing under
the JV entity. $500B in pledges are not $500B in committed capital.

**3. Microsoft canceled 2 GW of leases.**
The largest infrastructure spender in the world (in absolute dollars) walked away from
significant lease commitments after DeepSeek. If even Microsoft is reassessing at the margin,
the demand assumptions baked into 20–100x multiples deserve scrutiny.

**4. OpenAI loses $5B/year at $3.4B revenue.**
The company funding the largest compute commitments in history is not yet self-sustaining.
If OpenAI's growth slows or restructuring fails, CoreWeave's $22B contract is at risk.

**5. Customer concentration is existential for neoclouds.**
CoreWeave earns 60%+ of revenue from two customers. xAI's revenue is largely undisclosed.
If either major AI lab renegotiates pricing or builds in-house, the neocloud model breaks.
        """)

    st.markdown("---")

    # ── Growth drivers ───────────────────────────────────────────────────────
    st.markdown("### Where Is Growth Coming From?")
    st.caption("The demand drivers sustaining or threatening the current buildout economics.")

    for g in GROWTH_DRIVERS:
        with st.expander(f"**{g['driver']}** — Magnitude: {g['magnitude']}"):
            st.markdown(g["detail"])
            st.markdown(
                f'<span class="src-link">Sources: <a href="{g["url"]}" target="_blank">{g["source"]}</a></span>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown(
        '<span class="src-link">'
        "All analysis is for research and informational purposes only. Not financial advice. "
        "Data sourced from public company filings, press releases, and analyst reports as of March 2026. "
        "Multi-year contract values are shown at face value and are not discounted."
        '</span>',
        unsafe_allow_html=True,
    )
