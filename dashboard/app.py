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

import json
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

ALL_PROJECTS = [
    # ── Microsoft ──────────────────────────────────────────────────────────────
    dict(
        company="Microsoft", project="Wisconsin Campus (Mount Pleasant)",
        location="Mount Pleasant, WI", capex_b=3.3, status="Under Construction",
        scale="Large-scale campus; multiple buildings; Microsoft's largest WI investment",
        notes=(
            "Part of Microsoft's $3.3B Wisconsin investment announced 2024. Site in the "
            "former Foxconn development zone. Construction began 2024; hyperscale Azure cluster."
        ),
        source="WI Economic Development Corporation (2024)", year=2024,
        url="https://news.microsoft.com/2024/05/08/microsoft-to-invest-3-3-billion-in-wisconsin/",
    ),
    dict(
        company="Microsoft", project="Virginia Data Center Expansion (Loudoun County)",
        location="Loudoun County (NoVA), VA", capex_b=3.5, status="Active/Ongoing",
        scale="World's largest data center hub; Microsoft's anchor presence",
        notes=(
            "Part of Microsoft's ongoing Northern Virginia cluster expansion — the world's single "
            "largest data center market. Azure East US primary region. 40+ existing buildings with "
            "continued permitting for new facilities. Estimated $3.5B+ in active construction phases."
        ),
        source="DataCenterFrontier; Loudoun County gov (2025)", year=2025,
        url="https://www.datacenterfrontier.com/hyperscale/article/55243285/loudoun-county-data-center-hub",
    ),
    dict(
        company="Microsoft", project="North Carolina AI Campus (Goose Creek)",
        location="Catawba County, NC", capex_b=2.5, status="Active (Resumed 2025)",
        scale="~2M sq ft campus; multiple phases; resumed after brief 2025 review",
        notes=(
            "Originally paused in early 2025 alongside other lease reviews post-DeepSeek. "
            "Microsoft confirmed North Carolina build-out resumed by Q1 2025 after owned-vs-leased "
            "strategy was clarified. $2.5B estimated investment over multiple phases."
        ),
        source="Charlotte Observer; DataCenterDynamics (2025)", year=2025,
        url="https://www.datacenterdynamics.com/en/news/microsoft-reportedly-resumes-north-carolina-data-center-build-out/",
    ),
    dict(
        company="Microsoft", project="Phoenix / Goodyear (Arizona) Campus",
        location="Goodyear, AZ", capex_b=3.3, status="Under Construction",
        scale="Azure West US anchor; multi-building campus",
        notes=(
            "Arizona is a major Azure region anchor. Multiple facilities in Goodyear/Surprise area. "
            "Part of Microsoft's Southwest expansion. Nuclear PPA discussions with APS. "
            "Estimated $3B+ in active permits and construction across AZ as of 2025."
        ),
        source="Arizona Republic; Microsoft IR (2025)", year=2025,
        url="https://news.microsoft.com/source/features/ai/microsoft-data-center-expansion/",
    ),
    dict(
        company="Microsoft", project="Texas (San Antonio + Other Sites)",
        location="San Antonio, TX + DFW", capex_b=1.5, status="Active",
        scale="Azure South Central US region; multi-site",
        notes=(
            "San Antonio hosts one of Azure's core US cloud regions. Microsoft has $1B+ in "
            "active construction across Texas including DFW edge nodes and San Antonio primary cluster. "
            "Part of overall $120B/yr global capex plan."
        ),
        source="San Antonio Business Journal (2025)", year=2025,
        url="https://news.microsoft.com/source/features/ai/microsoft-data-center-expansion/",
    ),
    # ── Amazon / AWS ───────────────────────────────────────────────────────────
    dict(
        company="Amazon (AWS)", project="Indiana Data Center Campus",
        location="Whitestown / Boone County, IN", capex_b=15.5, status="Under Construction",
        scale="Largest single-state US data center investment by Amazon",
        notes=(
            "Amazon announced $15.5B investment in Indiana data centers across multiple "
            "Boone County sites. Includes 1,000+ construction jobs and 800 permanent jobs. "
            "Primarily for AWS US-East overflow capacity and AI inference. Construction 2024-2027."
        ),
        source="Amazon; Indiana Economic Development Corporation (2024)", year=2024,
        url="https://www.aboutamazon.com/news/aws/amazon-indiana-data-centers",
    ),
    dict(
        company="Amazon (AWS)", project="North Carolina Data Center Expansion",
        location="Eastern NC (multiple sites)", capex_b=10.0, status="Under Construction",
        scale="Multi-site AWS cluster; billions committed through 2030",
        notes=(
            "Amazon announced $10B+ commitment to North Carolina data center expansion. "
            "Sites in Columbus County and surrounding areas. Part of AWS US-East-1 overflow "
            "strategy. Includes renewable energy commitments (solar PPAs)."
        ),
        source="NC Commerce; Amazon announcement (2024)", year=2024,
        url="https://www.aboutamazon.com/news/aws/amazon-data-centers-north-carolina",
    ),
    dict(
        company="Amazon (AWS)", project="Louisiana Data Center Project",
        location="Bossier City, LA", capex_b=12.0, status="Announced/Early Construction",
        scale="Expected multi-GW campus; major Louisiana economic investment",
        notes=(
            "Amazon announced $12B investment in Louisiana data centers in Caddo and Bossier "
            "parishes. One of Amazon's largest announced state-level investments. Includes "
            "renewable energy partnerships. Long-term build through 2030."
        ),
        source="Governor's office; Amazon (2025)", year=2025,
        url="https://www.aboutamazon.com/news/aws/amazon-data-centers-louisiana",
    ),
    dict(
        company="Amazon (AWS)", project="Federal / Intelligence Community Cloud",
        location="US (GovCloud East + West)", capex_b=4.5, status="Active/Ongoing",
        scale="Classified facilities; separate GovCloud infrastructure",
        notes=(
            "AWS GovCloud regions serve DoD, NSA, CIA (C2E contract), and other intelligence "
            "agencies. Estimated $4.5B+ in active classified infrastructure build. CIA JEDI/C2E "
            "contract alone is a 10-year engagement. AWS is sole-sourced provider for many agencies."
        ),
        source="FedTech Magazine; AWS GovCloud filings (2025)", year=2025,
        url="https://aws.amazon.com/govcloud-us/",
    ),
    dict(
        company="Amazon (AWS)", project="Germany Data Center Expansion",
        location="Frankfurt region, Germany", capex_b=8.8, status="Under Construction",
        scale="EU-West expansion; compliance with EU data sovereignty rules",
        notes=(
            "Amazon announced EUR 8.2B (~$8.8B) investment in German data centers. "
            "Frankfurt AWS region serves EU financial services and sovereign data mandates. "
            "Part of Amazon's 2025-2027 European expansion plan."
        ),
        source="AWS; German government (2025)", year=2025,
        url="https://www.aboutamazon.com/news/aws/amazon-germany-data-centers",
    ),
    # ── Google (Alphabet) ──────────────────────────────────────────────────────
    dict(
        company="Google", project="Texas Statewide AI Infrastructure",
        location="Texas (multiple cities)", capex_b=10.0, status="Active",
        scale="Statewide: existing sites in Council Bluffs (IA), new TX sites",
        notes=(
            "Google announced $10B in Texas investments in 2025 for AI data center expansion. "
            "Specific sites include Midlothian, Garland, and other DFW-area campuses. "
            "Powers Google Cloud's South-Central US region and AI training clusters."
        ),
        source="Google; Texas Economic Development (2025)", year=2025,
        url="https://blog.google/inside-google/infrastructure/google-data-centers-united-states/",
    ),
    dict(
        company="Google", project="Virginia Data Center Expansion",
        location="Loudoun / Prince William County, VA", capex_b=9.0, status="Active/Ongoing",
        scale="Google's anchor presence in the NoVA hub; 10+ facilities",
        notes=(
            "Google has committed $9B+ to its Northern Virginia data center campus. "
            "Among the earliest hyperscaler presences in Loudoun County. Serves GCP US-East "
            "regions and YouTube. Active building permits for multiple new facilities."
        ),
        source="Google IR; Loudoun County (2025)", year=2025,
        url="https://blog.google/inside-google/infrastructure/google-data-centers-united-states/",
    ),
    dict(
        company="Google", project="South Carolina Data Center Campus",
        location="Berkeley County, SC", capex_b=9.0, status="Under Construction",
        scale="Multi-building campus; one of Google's largest single-site investments",
        notes=(
            "Google's Berkeley County, SC campus is one of its largest US data center investments. "
            "$9B committed with 300+ permanent jobs. Nuclear energy discussions with Duke Energy. "
            "Serves GCP US-East-1 overflow and AI inference."
        ),
        source="SC Commerce; Google (2025)", year=2025,
        url="https://blog.google/inside-google/infrastructure/google-south-carolina/",
    ),
    dict(
        company="Google", project="Project Mica — Kansas City",
        location="Kansas City, MO / KS", capex_b=10.0, status="Announced",
        scale="~1,000 acres; new Google data center campus",
        notes=(
            "'Project Mica' is Google's code name for a massive Kansas City area data center "
            "campus announced in early 2026. $10B estimated investment. Site selection driven "
            "by central US geography, power availability, and fiber connectivity."
        ),
        source="KC Business Journal; TechCrunch (Feb 2026)", year=2026,
        url="https://techcrunch.com/2026/02/28/billion-dollar-infrastructure-deals-ai-boom-data-centers-openai-oracle-nvidia-microsoft-google-meta/",
    ),
    dict(
        company="Google", project="PJM Grid Interconnect Expansion (Midwest)",
        location="Midwest US (PJM footprint)", capex_b=25.0, status="Active/Planned",
        scale="Largest grid interconnect request in PJM history; multiple sites",
        notes=(
            "Google filed for 25 GW of new data center power interconnections in the PJM "
            "electricity market — the single largest grid request in PJM history. Covers Ohio, "
            "Indiana, Illinois, and Virginia sites. Represents Google's full 2026-2030 "
            "US Midwest expansion plan."
        ),
        source="PJM grid filings; Bloomberg (2025)", year=2025,
        url="https://blog.google/inside-google/infrastructure/google-data-centers-united-states/",
    ),
    dict(
        company="Google", project="Germany Data Center & Cloud Expansion",
        location="Frankfurt + Hamburg, Germany", capex_b=5.9, status="Under Construction",
        scale="EUR 5.5B (~$5.9B); EU sovereignty compliance cluster",
        notes=(
            "Google committed EUR 5.5B to German data center expansion and cloud infrastructure. "
            "Includes a new sovereign cloud product for German government/enterprise customers "
            "under T-Systems partnership. Key for EU AI Act compliance hosting."
        ),
        source="Google; German government (2025)", year=2025,
        url="https://blog.google/inside-google/infrastructure/google-germany-investment/",
    ),
    dict(
        company="Google", project="Intersect Power Renewable Energy Data Center JV",
        location="US (Texas + other states)", capex_b=4.75, status="Active",
        scale="Solar + battery + data center colocation; first-of-kind at scale",
        notes=(
            "Google partnered with Intersect Power and TPG Rise Climate in a $4.75B deal to "
            "co-locate data centers directly with renewable energy generation. The hyperscale "
            "facilities sit on-site with solar farms and battery storage, eliminating grid "
            "transmission costs."
        ),
        source="Intersect Power; Bloomberg (2025)", year=2025,
        url="https://intersectpower.com/",
    ),
    # ── Meta ───────────────────────────────────────────────────────────────────
    dict(
        company="Meta", project="Hyperion AI Supercluster (Louisiana)",
        location="Richland Parish, LA", capex_b=27.0, status="Under Construction",
        scale="2 GW initial -> 5 GW; 3,650+ acres (largest private US land purchase for AI)",
        notes=(
            "Joint venture with Blue Owl Capital. Powered by gas turbines + solar (2.26 GW by 2029). "
            "5,000 construction workers at peak. Phase 2: Meta purchased 1,400 additional adjacent "
            "acres in early 2026 — nearly doubling the original site. Expected online 2027-2029."
        ),
        source="Fortune Feb 2026; Meta IR", year=2025,
        url="https://fortune.com/2026/02/04/meta-hyperion-ai-data-center-louisiana-expansion/",
    ),
    dict(
        company="Meta", project="New Albany Data Center Campus (Indiana)",
        location="New Albany, IN (Greater Louisville area)", capex_b=10.0, status="Under Construction",
        scale="$10B+ investment; one of Meta's largest US campuses",
        notes=(
            "Meta's New Albany, Indiana campus is one of its flagship data center sites. "
            "$10B committed over multiple build phases. Powers Meta AI (Llama) training and "
            "Facebook/Instagram inference. 800+ permanent jobs. Active construction as of 2025."
        ),
        source="Meta; Indiana Economic Dev (2025)", year=2025,
        url="https://sustainability.fb.com/innovation/data-centers/",
    ),
    dict(
        company="Meta", project="El Paso, Texas Data Center",
        location="El Paso, TX", capex_b=1.5, status="Active",
        scale="Regional inference cluster; Southwest coverage",
        notes=(
            "Meta's El Paso campus provides Southwest US inference capacity for Meta AI and "
            "WhatsApp. $1.5B investment across multiple phases. Part of Meta's distributed "
            "inference strategy to reduce latency for US users."
        ),
        source="El Paso Times; Meta IR (2025)", year=2025,
        url="https://sustainability.fb.com/innovation/data-centers/",
    ),
    dict(
        company="Meta", project="DeKalb, Illinois Campus",
        location="DeKalb, IL", capex_b=1.0, status="Active",
        scale="Midwest inference cluster; renewable-powered",
        notes=(
            "Meta's DeKalb, IL facility is 100% renewable-powered (wind PPAs with ComEd). "
            "$1B+ investment. Provides Midwest US capacity for Facebook, Instagram, and Threads. "
            "Part of Meta's announced $125B FY2025 capital plan."
        ),
        source="Meta sustainability report (2025)", year=2025,
        url="https://sustainability.fb.com/innovation/data-centers/",
    ),
    # ── Oracle ─────────────────────────────────────────────────────────────────
    dict(
        company="Oracle", project="Abilene / Fort Worth Stargate Campus (Texas)",
        location="Abilene + Fort Worth, TX", capex_b=10.0, status="Active — Phase 1 Online",
        scale="131k-GPU zettascale cluster LIVE; 1 GW+ planned",
        notes=(
            "Oracle's Phase 1 Stargate cluster in Abilene is the world's largest GPU supercomputer "
            "in cloud infrastructure — 131,000 GPUs online. OpenAI runs GPT-4o and other production "
            "models here. Separate $300B bilateral deal with OpenAI (starting 2027) proceeds "
            "independently of the stalled Stargate JV."
        ),
        source="TechCrunch Feb 2026; Oracle SEC filing Jun 2025", year=2025,
        url="https://techcrunch.com/2026/02/28/billion-dollar-infrastructure-deals-ai-boom-data-centers-openai-oracle-nvidia-microsoft-google-meta/",
    ),
    dict(
        company="Oracle", project="Port Washington, Wisconsin Campus",
        location="Port Washington, WI", capex_b=15.0, status="Under Construction",
        scale="Major new OCI region; largest WI data center project",
        notes=(
            "Oracle announced a $15B+ data center campus in Port Washington, WI — one of the "
            "largest single-site announcements in Midwest history. Part of Oracle's aggressive "
            "OCI (Oracle Cloud Infrastructure) expansion funded by $38B in debt raised in 2025. "
            "Powered in part by local utilities + renewable PPAs."
        ),
        source="Wisconsin State Journal; Oracle (2025)", year=2025,
        url="https://www.oracle.com/news/",
    ),
    dict(
        company="Oracle", project="Michigan Data Center",
        location="Michigan (Grand Rapids area)", capex_b=1.5, status="Announced/Permitting",
        scale="New OCI region; Great Lakes market",
        notes=(
            "Oracle announced a $1.5B data center investment in Michigan as part of its push "
            "to expand OCI regional coverage. Serves Midwest enterprise customers and government "
            "cloud. Part of Oracle's 162-data-center global expansion plan (from 80 current)."
        ),
        source="Crain's Detroit Business; Oracle (2025)", year=2025,
        url="https://www.oracle.com/news/",
    ),
    dict(
        company="Oracle", project="India Data Center Expansion",
        location="Mumbai + Hyderabad, India", capex_b=6.5, status="Active",
        scale="Largest cloud investment in India by Oracle; 3 new OCI regions",
        notes=(
            "Oracle committed $6.5B to Indian data center expansion — one of its largest "
            "single-country investments globally. Includes three new OCI regions to serve "
            "Indian government mandates and enterprise customers. Part of Reliance JV discussions."
        ),
        source="Economic Times India; Oracle (2025)", year=2025,
        url="https://www.oracle.com/news/",
    ),
    # ── xAI ────────────────────────────────────────────────────────────────────
    dict(
        company="xAI", project="Colossus Phase 1+2 (Memphis Supercluster)",
        location="Memphis, TN", capex_b=7.0, status="Active — 200k GPUs Online",
        scale="200k H100 GPUs now; targeting 1M GPUs; 1 GW+ power",
        notes=(
            "Full vertical integration: xAI owns chips, power generation, racks, and land. "
            "Phase 1 (100k H100) built in just 122 days. Phase 2 adds 100-acre adjacent sites. "
            "$20B capital raise uses SPV structure with GPUs as collateral. "
            "Powered by TVA + MLGW; targeting 1M GPUs total by end of 2026."
        ),
        source="xAI blog; HPCwire May 2025", year=2025,
        url="https://www.hpcwire.com/2025/05/13/colossus-ai-hits-200000-gpus-as-musk-ramps-up-ai-ambitions/",
    ),
    # ── CoreWeave ──────────────────────────────────────────────────────────────
    dict(
        company="CoreWeave", project="Pennsylvania Data Center Expansion",
        location="Multiple PA sites (Bethlehem + Pittsburgh area)", capex_b=6.0, status="Active",
        scale="Largest CoreWeave expansion outside NJ; NVIDIA-anchored GPU clusters",
        notes=(
            "CoreWeave expanded aggressively into Pennsylvania with $6B+ committed across "
            "multiple sites. Bethlehem (former Bethlehem Steel site) and Pittsburgh area campuses. "
            "Powered by PJM grid with nuclear PPAs. Serves Meta ($14.2B contract) and "
            "other AI lab tenants."
        ),
        source="PA Department of Community & Economic Development; CoreWeave (2025)", year=2025,
        url="https://www.fool.com/investing/2026/02/25/ai-stock-soared-since-ipo-still-but-coreweave/",
    ),
    dict(
        company="CoreWeave", project="North Dakota Expansion",
        location="Minot / Bismarck, ND", capex_b=2.0, status="Under Construction",
        scale="Power-advantaged location; renewable-heavy grid; cold climate cooling",
        notes=(
            "CoreWeave selected North Dakota for a $2B expansion due to low electricity costs, "
            "cold climate (free air cooling for GPUs), and available land. ND's grid is "
            "~70% renewable. Targets Microsoft Azure overflow workloads and AI training runs."
        ),
        source="ND Commerce; CoreWeave (2025)", year=2025,
        url="https://www.fool.com/investing/2026/02/25/ai-stock-soared-since-ipo-still-but-coreweave/",
    ),
    dict(
        company="CoreWeave", project="New Jersey HQ Campus Expansion",
        location="Parsippany / Roseland, NJ", capex_b=1.8, status="Active",
        scale="Original CoreWeave HQ cluster; NVIDIA proximity advantage",
        notes=(
            "CoreWeave's home base in New Jersey houses its original GPU clusters and HQ. "
            "$1.8B in active expansion to support growing customer contracts. "
            "Proximity to NYSE and financial services clients. NVIDIA's East Coast "
            "relationship hub. Powers OpenAI's early CoreWeave contract deliverables."
        ),
        source="CoreWeave S-1; NJ Business (2025)", year=2025,
        url="https://www.fool.com/investing/2026/02/25/ai-stock-soared-since-ipo-still-but-coreweave/",
    ),
    dict(
        company="CoreWeave", project="Multi-State GPU Cloud JV",
        location="US (TX, OH, VA additional sites)", capex_b=5.0, status="Active",
        scale="New sites in TX, OH, VA to diversify beyond NJ/PA",
        notes=(
            "CoreWeave has committed $5B+ to new data center sites across Texas (Austin), "
            "Ohio (Columbus), and Virginia (Ashburn area) — diversifying away from its "
            "northeastern concentration. Powers the Meta $14.2B contract geographic distribution. "
            "Funded through post-IPO equity and SPV debt structures."
        ),
        source="CoreWeave filings; DataCenterDynamics (2025)", year=2025,
        url="https://www.fool.com/investing/2026/02/25/ai-stock-soared-since-ipo-still-but-coreweave/",
    ),
    dict(
        company="CoreWeave", project="Europe Expansion (UK + Germany)",
        location="London, UK + Frankfurt, Germany", capex_b=2.2, status="Active",
        scale="EU and UK GPU cloud presence; serves European AI labs",
        notes=(
            "CoreWeave established European operations with $2.2B committed across UK (London "
            "Docklands area) and Germany (Frankfurt). Serves European AI startups and Microsoft "
            "Azure European overflow. UK site benefits from proximity to DeepMind, Stability AI. "
            "EU data sovereignty compliance built in."
        ),
        source="CoreWeave European expansion announcement (2025)", year=2025,
        url="https://www.fool.com/investing/2026/02/25/ai-stock-soared-since-ipo-still-but-coreweave/",
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


NEWS_JSON = ROOT / "dashboard" / "data" / "news_feed.json"


@st.cache_data(ttl=timedelta(minutes=30), show_spinner=False)
def load_news_feed() -> dict:
    """Load news articles from the pre-built JSON (updated by GitHub Actions every 6 hours)."""
    try:
        return json.loads(NEWS_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {"last_updated": None, "article_count": 0, "deal_alert_count": 0, "articles": []}


@st.cache_data(ttl=timedelta(hours=2), show_spinner=False)
def fetch_live_news() -> list[dict]:
    """Live RSS fetch — supplements the cached JSON. Runs in-browser, cached 2 hrs."""
    try:
        import feedparser as fp
        LIVE_FEEDS = [
            ("Google News: AI deals",       "https://news.google.com/rss/search?q=AI+data+center+deal+billion+GPU&hl=en-US&gl=US&ceid=US:en"),
            ("Google News: hyperscaler capex", "https://news.google.com/rss/search?q=Microsoft+Amazon+Google+Meta+Oracle+AI+capex+infrastructure&hl=en-US&gl=US&ceid=US:en"),
            ("CNBC Tech",                   "https://www.cnbc.com/id/19854910/device/rss/rss.html"),
            ("DataCenterDynamics",          "https://www.datacenterdynamics.com/en/rss/"),
            ("TechCrunch",                  "https://techcrunch.com/feed/"),
            ("SemiAnalysis",                "https://newsletter.semianalysis.com/feed"),
        ]
        KEYWORDS = ["data center", "gpu", "nvidia", "openai", "microsoft", "xai", "oracle",
                    "amazon", "coreweave", "meta", "ai infrastructure", "capex", "hyperscale",
                    "colossus", "stargate", "h100", "blackwell", "compute"]

        import re, hashlib
        articles, seen = [], set()
        for src, url in LIVE_FEEDS:
            try:
                feed = fp.parse(url)
                for e in feed.entries[:25]:
                    link  = e.get("link", "")
                    title = re.sub(r"<[^>]+>", "", e.get("title", "")).strip()
                    body  = re.sub(r"<[^>]+>", "", e.get("summary", ""))[:500]
                    if not link or not title:
                        continue
                    aid = hashlib.md5(link.encode()).hexdigest()[:14]
                    if aid in seen:
                        continue
                    seen.add(aid)
                    text = (title + " " + body).lower()
                    if not any(kw in text for kw in KEYWORDS):
                        continue
                    amounts = re.findall(r"\$[\d,]+\.?\d*\s*(?:billion|million|B|M|T)\b", text, re.I)
                    ds = sum(2 for sig in ["billion","deal","contract","invest","acqui","announce","signed","awarded"]
                             if sig in text)
                    articles.append({
                        "id": aid, "title": title, "url": link, "source": src,
                        "source_cat": "Live", "date": datetime.now().strftime("%Y-%m-%d"),
                        "summary": body, "deal_score": min(ds, 10),
                        "amounts": list(set(amounts))[:4], "is_deal_alert": ds >= 4,
                    })
            except Exception:
                pass
        return sorted(articles, key=lambda x: x["deal_score"], reverse=True)
    except ImportError:
        return []


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

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Overview",
    "Deals & Capital Flows",
    "Active Projects",
    "Largest Projects",
    "Profitability & Margins",
    "Underwriting & Growth",
    "Who's Bullshitting",
    "Live Intel",
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
# TAB 3: ACTIVE PROJECTS (comprehensive $1B+ table)
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Active Projects — Every $1B+ Data Center Investment")
    st.caption(
        "Every confirmed project over $1 billion for Microsoft, Amazon, Google, Meta, Oracle, xAI, and CoreWeave "
        "as of March 3, 2026. Sourced from company announcements, SEC filings, state economic development offices, "
        "and investigative reporting. Excludes the stalled Stargate JV (see Largest Projects tab)."
    )

    # Build DataFrame
    proj_rows = []
    for p in ALL_PROJECTS:
        proj_rows.append({
            "Company":    p["company"],
            "Project":    p["project"],
            "Location":   p["location"],
            "CapEx ($B)": p["capex_b"],
            "Status":     p["status"],
            "Year":       p["year"],
        })
    all_proj_df = pd.DataFrame(proj_rows)

    # ── Summary stats row ────────────────────────────────────────────────────
    total_capex = all_proj_df["CapEx ($B)"].sum()
    n_projects  = len(all_proj_df)
    n_companies = all_proj_df["Company"].nunique()
    active_ct   = all_proj_df[all_proj_df["Status"].str.contains("Active|Construction", na=False)].shape[0]

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Total Projects Tracked", f"{n_projects}")
    sc2.metric("Companies Covered", f"{n_companies}")
    sc3.metric("Total CapEx (tracked)", f"${total_capex:.0f}B")
    sc4.metric("Active / Under Construction", f"{active_ct}")

    st.markdown("---")

    # ── Filters ──────────────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns([2, 2, 1])
    with fc1:
        company_options = ["All"] + sorted(all_proj_df["Company"].unique().tolist())
        sel_company = st.selectbox("Filter by Company", company_options, key="ap_company")
    with fc2:
        status_options = ["All"] + sorted(all_proj_df["Status"].unique().tolist())
        sel_status = st.selectbox("Filter by Status", status_options, key="ap_status")
    with fc3:
        min_capex = st.number_input("Min CapEx ($B)", min_value=0.0, max_value=50.0,
                                     value=0.0, step=0.5, key="ap_mincapex")

    filtered_df = all_proj_df.copy()
    if sel_company != "All":
        filtered_df = filtered_df[filtered_df["Company"] == sel_company]
    if sel_status != "All":
        filtered_df = filtered_df[filtered_df["Status"] == sel_status]
    if min_capex > 0:
        filtered_df = filtered_df[filtered_df["CapEx ($B)"] >= min_capex]

    filtered_df = filtered_df.sort_values("CapEx ($B)", ascending=False).reset_index(drop=True)

    # Color map for company column
    def color_company(val):
        colors = {
            "Microsoft":      "color: #00a4ef",
            "Amazon (AWS)":   "color: #ff9900",
            "Google":         "color: #34a853",
            "Meta":           "color: #0866ff",
            "Oracle":         "color: #f80000",
            "xAI":            "color: #1da1f2",
            "CoreWeave":      "color: #6f42c1",
        }
        return colors.get(val, "")

    styled = filtered_df.style.applymap(color_company, subset=["Company"])

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        column_config={
            "CapEx ($B)": st.column_config.NumberColumn(format="$%.1fB"),
            "Year":       st.column_config.NumberColumn(format="%d"),
        },
    )

    st.caption(f"Showing {len(filtered_df)} of {n_projects} projects. Sorted by CapEx descending.")

    st.markdown("---")

    # ── Per-company totals bar chart ─────────────────────────────────────────
    st.markdown("#### Total Tracked CapEx by Company")
    co_totals = (
        all_proj_df.groupby("Company")["CapEx ($B)"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    co_colors_map = {
        "Microsoft":    "#00a4ef",
        "Amazon (AWS)": "#ff9900",
        "Google":       "#34a853",
        "Meta":         "#0866ff",
        "Oracle":       "#f80000",
        "xAI":          "#1da1f2",
        "CoreWeave":    "#6f42c1",
    }
    bar_colors = [co_colors_map.get(c, "#7d8590") for c in co_totals["Company"]]
    bar_fig = go.Figure(go.Bar(
        x=co_totals["Company"],
        y=co_totals["CapEx ($B)"],
        marker_color=bar_colors,
        text=[f"${v:.0f}B" for v in co_totals["CapEx ($B)"]],
        textposition="outside",
        hovertemplate="%{x}: $%{y:.1f}B<extra></extra>",
    ))
    bar_fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#c9d1d9"),
        yaxis=dict(title="$B", gridcolor="#21262d"),
        xaxis=dict(gridcolor="#21262d"),
        margin=dict(t=30, b=40, l=40, r=20),
        height=340,
    )
    st.plotly_chart(bar_fig, use_container_width=True)

    # ── Detailed project cards ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Project Detail Cards")
    st.caption("Click to expand full terms, notes, and source for each project.")

    display_projects = ALL_PROJECTS if sel_company == "All" else [
        p for p in ALL_PROJECTS if p["company"] == sel_company
    ]
    display_projects = sorted(display_projects, key=lambda x: x["capex_b"], reverse=True)
    if min_capex > 0:
        display_projects = [p for p in display_projects if p["capex_b"] >= min_capex]

    for p in display_projects:
        label = (
            f"**{p['company']}** — {p['project']}   |   "
            f"${p['capex_b']:.1f}B   |   {p['location']}   |   {p['status']}"
        )
        with st.expander(label):
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("CapEx", f"${p['capex_b']:.1f}B")
            d2.metric("Status", p["status"].split("—")[0].split("(")[0].strip())
            d3.metric("Location", p["location"].split(",")[0])
            d4.metric("Year", str(p["year"]))
            st.markdown(f"**Scale:** {p['scale']}")
            st.markdown(f"**Details:** {p['notes']}")
            st.markdown(
                f'<span class="src-link">Source: <a href="{p["url"]}" target="_blank">{p["source"]}</a></span>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: LARGEST PROJECTS
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### Largest Data Center Projects")
    st.caption(
        "The highest-profile, highest-value confirmed projects under construction or active development as of March 2026. "
        "These have not been canceled. Each entry includes deal terms and a primary source. "
        "See the Active Projects tab for every $1B+ project across all companies."
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
# TAB 5: PROFITABILITY & MARGINS
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
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
# TAB 6: UNDERWRITING & GROWTH
# ─────────────────────────────────────────────────────────────────────────────
with tab6:
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


# ─────────────────────────────────────────────────────────────────────────────
# TAB 7: WHO'S BULLSHITTING
# ─────────────────────────────────────────────────────────────────────────────
with tab7:
    st.markdown("### Who's Bullshitting?")
    st.caption(
        "An honest, sourced breakdown of where the gap between the narrative and the evidence is largest. "
        "Rated on a BS scale from 1 (mild spin) to 5 (detached from reality). "
        "This is editorial analysis, not financial advice."
    )

    BS_ENTRIES = [
        dict(
            name="The $500B Stargate JV",
            players="SoftBank + OpenAI + Oracle + MGX",
            score=5,
            label="Full Fiction",
            color="#f85149",
            claim=(
                "On January 21, 2025, President Trump stood flanked by Masayoshi Son, Sam Altman, "
                "and Larry Ellison to announce a $500B AI infrastructure joint venture that would "
                "create 100,000 American jobs. SoftBank committed $100B immediately. Wall Street "
                "called it the most ambitious infrastructure announcement in decades."
            ),
            reality=(
                "**More than a year later: zero staff hired. Zero data centers built under the JV entity.**\n\n"
                "OpenAI, Oracle, and SoftBank spent the intervening months in governance disputes — "
                "fighting over who controls the data centers, who owns the assets, and how profits get split. "
                "The partners eventually gave up on the JV and pivoted to bilateral deals: OpenAI signed "
                "a direct $300B contract with Oracle and a $38B contract with AWS, both bypassing the "
                "Stargate JV structure entirely.\n\n"
                "SoftBank's ability to deploy $100B was questioned by analysts from the start — the company "
                "had just completed a painful Vision Fund writedown cycle and was not sitting on $100B in "
                "liquid capital.\n\n"
                "**The $500B number was chosen for headline impact, not because it represented a funded, "
                "organized plan.** The underlying construction is happening — Oracle is building TX campuses, "
                "OpenAI is buying compute — but under ordinary bilateral contracts, not a historic joint venture."
            ),
            sources="The Decoder Feb 2026; Tom's Hardware; TechPortal",
            url="https://the-decoder.com/stargates-500-billion-ai-infrastructure-project-reportedly-stalls-over-unresolved-disputes-between-openai-oracle-and-softbank/",
        ),
        dict(
            name="OpenAI's '$1.15 Trillion Infrastructure Commitment'",
            players="OpenAI / Sam Altman",
            score=4,
            label="Aggressive Storytelling",
            color="#d29922",
            claim=(
                "OpenAI has 'committed' $1.15 trillion in infrastructure spending over 10 years "
                "across Broadcom ($350B), Oracle ($300B), Microsoft ($250B), NVIDIA ($100B), AMD ($90B), "
                "AWS ($38B), and CoreWeave ($22B+). Headlines called it the largest infrastructure "
                "commitment by any company in history."
            ),
            reality=(
                "**OpenAI is currently losing $5 billion per year on $3.4 billion in revenue.** "
                "It is a company that does not own a single data center and is currently dependent on "
                "Microsoft's balance sheet.\n\n"
                "The $1.15T 'commitment' is a 10-year figure aggregating deals that:\n"
                "- Start in 2027 (the Oracle $300B deal);\n"
                "- Have not been signed yet at the time of some announcements;\n"
                "- Are contingent on OpenAI generating revenues it does not yet have.\n\n"
                "The Broadcom $350B figure in particular is a 10-year chip development roadmap, not "
                "a funded purchase order. OpenAI's current runway is dependent on equity raises "
                "at $157B valuation, not on $350B in committed chip revenue.\n\n"
                "**None of this means OpenAI is failing** — revenue is doubling annually and ChatGPT "
                "has 500M+ users. But framing $1.15T in future obligations as a 'commitment' by a "
                "company losing $5B/yr conflates ambition with funded capital."
            ),
            sources="Tunguz OpenAI Infrastructure Analysis; Bloomberg; The Information",
            url="https://tomtunguz.com/openai-hardware-spending-2025-2035",
        ),
        dict(
            name="The Circular AI Economy (Everyone Paying Each Other)",
            players="Microsoft + OpenAI + NVIDIA + CoreWeave",
            score=4,
            label="Systemic Inflation",
            color="#d29922",
            claim=(
                "Microsoft has deployed $13B+ into OpenAI. OpenAI has committed $22B to CoreWeave. "
                "NVIDIA has invested $100B into OpenAI. The total 'disclosed capital' in this dataset "
                "exceeds $900B. Bloomberg, CNBC, and every financial outlet has covered the staggering "
                "size of AI infrastructure investment."
            ),
            reality=(
                "**Bloomberg documented in February 2026 that the same dollars are cycling through "
                "the AI ecosystem multiple times**, inflating the headline numbers.\n\n"
                "The loop: Microsoft invests $13B in OpenAI → OpenAI spends that money on Azure compute "
                "(back to Microsoft) → Azure uses that revenue to buy NVIDIA GPUs → NVIDIA invests "
                "$100B back into OpenAI → OpenAI spends that on more compute. **The same pool of capital "
                "generates multiple headline 'deals' as it cycles.**\n\n"
                "Add NVIDIA agreeing to buy $6.3B of CoreWeave cloud services (CoreWeave, which itself "
                "buys NVIDIA GPUs), and you have a system where every player can simultaneously claim "
                "to be a massive capital deployer.\n\n"
                "The infrastructure being built is real. The power being consumed is real. The GPUs are real. "
                "**But the aggregate dollar figures — $700B capex, $1.15T OpenAI commitments — "
                "substantially overstate the net new capital entering the system** because the circular "
                "flows aren't netted out in any headline."
            ),
            sources="Bloomberg 'AI Circular Deals' Feb 2026; SemiAnalysis",
            url="https://www.bloomberg.com/graphics/2026-ai-circular-deals/",
        ),
        dict(
            name="CoreWeave's GPU Economics at Current Rental Rates",
            players="CoreWeave (CRWV)",
            score=3,
            label="Optimistic Accounting",
            color="#d29922",
            claim=(
                "CoreWeave is the dominant GPU cloud provider with $55B in contracted revenue backlog, "
                "700%+ revenue growth, and a stock up 200–300% from its March 2025 IPO. "
                "It is positioned to be the backbone of AI computing infrastructure."
            ),
            reality=(
                "**The GPU rental economics are at or below break-even at current market rates.**\n\n"
                "H100 on-demand rates have fallen from ~$9/hr in 2023 to $1.50–2.00/hr by early 2026 "
                "— a 75%+ decline. A 100k H100 cluster financed at today's purchase prices needs roughly "
                "$2.10–2.30/hr at 70% utilization to service its debt. At $1.50–2.00/hr spot rates, "
                "the economics are marginal at best.\n\n"
                "CoreWeave's $55B backlog is real, but those contracts were signed when rates were higher. "
                "As contracts roll off and new capacity prices at spot, the revenue per GPU-hour compresses.\n\n"
                "The customer concentration risk is existential: **60%+ of revenue comes from two customers "
                "(Microsoft + OpenAI)**. If either renegotiates pricing at renewal — or if OpenAI builds "
                "more in-house capacity — CoreWeave's model is structurally exposed.\n\n"
                "The stock trading at ~24x revenue requires CoreWeave to grow into a $10B+ revenue "
                "company. The backlog supports the path, but the compressed unit economics make this "
                "a high-risk bet, not a safe infrastructure stock."
            ),
            sources="ThunderCompute Dec 2025; CoreWeave S-1; Motley Fool Feb 2026",
            url="https://www.thundercompute.com/blog/ai-gpu-rental-market-trends",
        ),
        dict(
            name="xAI's Valuation vs. Revenue",
            players="xAI / Elon Musk",
            score=3,
            label="Vision Premium",
            color="#d29922",
            claim=(
                "xAI raised at a $50B+ valuation and is executing the fastest data center build in "
                "history — 100,000 GPUs in 122 days. The Colossus supercluster in Memphis is vertically "
                "integrated (owns chips, power, racks, land), making xAI the only AI lab that controls "
                "its entire compute stack."
            ),
            reality=(
                "**The build execution is genuinely impressive and should not be dismissed.** "
                "122 days to 100k GPUs is a real engineering achievement that Microsoft, Meta, and "
                "Google have not matched at comparable speed. The vertical integration thesis is real: "
                "owning your own power and hardware provides long-term cost advantages.\n\n"
                "**But:** xAI's disclosed revenue is estimated at under $1 billion, against a "
                "$50B+ valuation. That is ~100x EV/Revenue — one of the highest multiples on any "
                "AI infrastructure company.\n\n"
                "The $20B capital raise ($12.5B in debt, GPUs as collateral) is financially fragile "
                "if H100/H200 rental rates remain at or below break-even. Using GPU hardware as loan "
                "collateral works when rates are $8/hr. At $1.50/hr, the collateral is worth much less "
                "than when the debt was issued.\n\n"
                "Grok is a good model. X platform is a distribution channel. But there is no publicly "
                "verified path from current revenue to a valuation that justifies $50B."
            ),
            sources="HPCwire May 2025; Bloomberg; Reuters; SemiAnalysis",
            url="https://www.hpcwire.com/2025/05/13/colossus-ai-hits-200000-gpus-as-musk-ramps-up-ai-ambitions/",
        ),
        dict(
            name="Oracle's Stock Reaction to the $300B Deal",
            players="Oracle / Larry Ellison",
            score=2,
            label="Mild Spin",
            color="#f0a500",
            claim=(
                "Oracle stock surged in June 2025 when the company disclosed a $300B, 5-year compute "
                "deal with OpenAI, briefly making Larry Ellison the richest person in the world. "
                "Oracle positioned itself as the primary infrastructure backbone for the AI era."
            ),
            reality=(
                "**The deal is real — but the market's reaction priced in revenue that doesn't "
                "start until 2027.**\n\n"
                "The $300B contract begins in 2027 and spreads over 5 years, meaning Oracle receives "
                "~$60B/year starting two years from the announcement date. The stock jumped immediately "
                "as if this revenue was imminent.\n\n"
                "Oracle's actual Q3 FY2025 cloud revenue was ~$7.7B — a $300B deal starting in 2027 "
                "would eventually be transformative, but pricing it into today's stock as if it were "
                "current revenue overstates the near-term fundamental picture.\n\n"
                "**That said, Oracle's Stargate execution is legitimate.** The 131k-GPU zettascale "
                "cluster is the world's largest in cloud infrastructure and is operational. Oracle "
                "is genuinely building — it's the announcement timing and market reaction that "
                "outpaced the underlying reality, not Oracle's actual construction."
            ),
            sources="Oracle SEC filing Jun 2025; TechCrunch Feb 2026; Oracle IR",
            url="https://techcrunch.com/2026/02/28/billion-dollar-infrastructure-deals-ai-boom-data-centers-openai-oracle-nvidia-microsoft-google-meta/",
        ),
        dict(
            name="Microsoft's '2 GW Cancellations' vs. '$80B Commitment'",
            players="Microsoft",
            score=1,
            label="Contradictory Messaging",
            color="#8b949e",
            claim=(
                "Microsoft committed $80B to AI data centers in FY2025 — the largest single-year "
                "capital commitment in the company's history. Satya Nadella repeatedly emphasized "
                "Microsoft's AI infrastructure leadership."
            ),
            reality=(
                "**Microsoft simultaneously canceled significant infrastructure commitments post-DeepSeek.**\n\n"
                "In February 2025 — the same quarter as the $80B FY2025 commitment — TD Cowen analysts "
                "reported that Microsoft had canceled ~200–300 MW of U.S. lease agreements and walked "
                "away from negotiations to lease ~2 GW of additional capacity in the US and Europe.\n\n"
                "Microsoft's official explanation was 'capacity optimization.' The reality is that "
                "DeepSeek's January 2025 release — which demonstrated GPT-4-class results at a "
                "fraction of the training cost — prompted a reassessment of how much capacity "
                "was actually needed.\n\n"
                "**This isn't the worst kind of BS — companies do change plans.** But the contrast "
                "between the public '$80B commitment' narrative and the simultaneous quiet lease "
                "cancellations illustrates that even the most committed hyperscaler had material "
                "uncertainty about whether the AI demand assumptions were right.\n\n"
                "The pivot from leased to owned capacity (the stated rationale) is also convenient "
                "framing for what may have simply been demand uncertainty."
            ),
            sources="Fortune Feb 2025; Bloomberg; DataCenterFrontier",
            url="https://fortune.com/2025/02/24/microsoft-cancels-leases-for-ai-data-centers-analyst-says/",
        ),
    ]

    # ── BS-o-meter legend ─────────────────────────────────────────────────────
    st.markdown(
        "**BS Scale:** "
        '<span style="color:#8b949e">1 = Mild Spin</span> &nbsp;|&nbsp; '
        '<span style="color:#f0a500">2 = Contradictory Messaging</span> &nbsp;|&nbsp; '
        '<span style="color:#d29922">3 = Vision Premium / Optimistic Accounting</span> &nbsp;|&nbsp; '
        '<span style="color:#d29922">4 = Aggressive Storytelling</span> &nbsp;|&nbsp; '
        '<span style="color:#f85149">5 = Full Fiction</span>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    for entry in BS_ENTRIES:
        score_dots = "●" * entry["score"] + "○" * (5 - entry["score"])
        with st.expander(
            f'**{entry["name"]}** — {entry["label"]}   '
            f'[{score_dots}]   ({entry["players"]})'
        ):
            st.markdown(
                f'<span style="color:{entry["color"]};font-size:1.1rem;font-weight:700;">'
                f'BS Score: {entry["score"]}/5 — {entry["label"]}'
                f'</span>',
                unsafe_allow_html=True,
            )

            col_claim, col_real = st.columns([1, 2])
            with col_claim:
                st.markdown("**The Claim / Narrative**")
                st.info(entry["claim"])
            with col_real:
                st.markdown("**What the Evidence Actually Shows**")
                st.markdown(entry["reality"])

            st.markdown(
                f'<span class="src-link">Sources: '
                f'<a href="{entry["url"]}" target="_blank">{entry["sources"]}</a>'
                f'</span>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    st.markdown("### The Bottom Line")
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("""
**What is genuinely real:**
- NVIDIA's GPU revenue and margins (~60% operating margin, $47.5B FY2025 data center revenue)
- Meta's Hyperion site is under active construction (boots on ground, 5,000 workers)
- xAI's 200k GPU Colossus cluster is operational and running Grok today
- Oracle's 131k-GPU zettascale cluster in Texas is operational
- CoreWeave's $55B contracted backlog is legally committed
- AWS, Azure, and Google Cloud are growing 20–40% YoY with expanding margins
- The demand for AI inference is real and growing
        """)
    with col_r:
        st.markdown("""
**What is narrative, not funded reality:**
- The "$500B Stargate JV" — stalled, no staff, no builds under the JV entity
- "$1.15T OpenAI infrastructure commitment" — 10-year aspiration by a company losing $5B/yr
- Any individual headline figure for circular AI deals (the capital cycles, it doesn't compound)
- GPU rental economics at current spot rates for newly financed clusters (at or below break-even)
- The idea that $50B+ xAI valuation is grounded in current revenues
- Any capex commitment announcement made the day of a Presidential photo op
        """)

    st.markdown("---")
    st.markdown(
        '<span class="src-link">'
        "Editorial analysis based on publicly sourced data. The author has no positions in any securities mentioned. "
        "Not financial advice. Sources linked in each section above."
        "</span>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 8: LIVE INTEL
# ─────────────────────────────────────────────────────────────────────────────
with tab8:
    st.markdown("### Live Intel — AI Infrastructure News Feed")
    st.caption(
        "Automatically scans 14 trusted sources every 6 hours via GitHub Actions. "
        "Articles with deal signals (company names + dollar amounts) are flagged automatically. "
        "Click any headline to read the full article."
    )

    # ── Load cached feed ──────────────────────────────────────────────────────
    feed_data   = load_news_feed()
    all_articles = feed_data.get("articles", [])
    last_updated = feed_data.get("last_updated", "Unknown")

    # ── Live supplement button ────────────────────────────────────────────────
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    col_stat1.metric("Tracked Articles", feed_data.get("article_count", 0))
    col_stat2.metric("Deal Alerts", feed_data.get("deal_alert_count", 0),
                     help="Articles whose title contains a company name + dollar amount signal")
    col_stat3.metric("Trusted Sources", "14",
                     help="Google News, TechCrunch, CNBC, DataCenterDynamics, Reuters, Ars Technica, The Register, SemiAnalysis, The Verge + keyword-targeted Google News feeds")
    col_stat4.metric("Update Frequency", "Every 6 hrs",
                     help="GitHub Actions workflow runs at 00:00, 06:00, 12:00, 18:00 UTC and commits fresh articles")

    if last_updated and last_updated != "Unknown":
        try:
            ts = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
            st.markdown(
                f'<span class="src-link">Last scanned: {ts.strftime("%B %d, %Y at %H:%M UTC")}</span>',
                unsafe_allow_html=True,
            )
        except Exception:
            pass

    live_col, _ = st.columns([1, 4])
    with live_col:
        if st.button("Fetch Live Now", help="Pull the latest headlines directly from RSS feeds (cached 2 hrs)"):
            st.cache_data.clear()
            st.rerun()

    st.markdown("---")

    # ── Merge live articles if available ─────────────────────────────────────
    with st.spinner("Loading live feed..."):
        live_articles = fetch_live_news()

    if live_articles:
        live_ids  = {a["id"] for a in all_articles}
        new_live  = [a for a in live_articles if a["id"] not in live_ids]
        if new_live:
            all_articles = new_live + all_articles
            st.success(f"{len(new_live)} new articles pulled live (not yet in the 6-hr cache)")

    # ── Filters ───────────────────────────────────────────────────────────────
    fcol1, fcol2, fcol3 = st.columns([2, 2, 1])
    with fcol1:
        search_q = st.text_input("Search headlines", placeholder="e.g. CoreWeave, $5B, Oracle...")
    with fcol2:
        sources_available = sorted(set(a["source"] for a in all_articles))
        source_filter = st.multiselect("Filter by source", sources_available, default=[])
    with fcol3:
        deal_only = st.toggle("Deal alerts only", value=False,
                              help="Show only articles flagged as potential new deals")

    # ── Apply filters ─────────────────────────────────────────────────────────
    filtered = all_articles
    if deal_only:
        filtered = [a for a in filtered if a.get("is_deal_alert")]
    if source_filter:
        filtered = [a for a in filtered if a["source"] in source_filter]
    if search_q:
        q = search_q.lower()
        filtered = [a for a in filtered if q in a["title"].lower() or q in a.get("summary", "").lower()]

    st.markdown(f"**{len(filtered)} articles** matching current filters")
    st.markdown("---")

    # ── Deal alerts section ───────────────────────────────────────────────────
    deal_alerts = [a for a in filtered if a.get("is_deal_alert")]
    if deal_alerts and not search_q and not source_filter:
        st.markdown("#### Deal Alerts — Potential New Capital Flows")
        st.caption(
            "These headlines contain company names + dollar amount signals. "
            "Not all are confirmed new deals — read the article to verify."
        )
        for a in deal_alerts[:20]:
            amt_str = "  ·  " + "  ".join(a["amounts"]) if a.get("amounts") else ""
            score_bar = "●" * min(a["deal_score"] // 2, 5)
            st.markdown(
                f'**[{a["title"]}]({a["url"]})**{amt_str}  \n'
                f'<span class="src-link">{a["source"]} &nbsp;·&nbsp; {a["date"]} '
                f'&nbsp;·&nbsp; Signal strength: {score_bar}</span>',
                unsafe_allow_html=True,
            )
        st.markdown("---")
        st.markdown("#### All Articles")

    # ── All articles ──────────────────────────────────────────────────────────
    if not filtered:
        st.info("No articles match the current filters.")
    else:
        # Group by date for readability
        from itertools import groupby
        by_date = {}
        for a in filtered[:150]:
            by_date.setdefault(a["date"], []).append(a)

        for date, arts in sorted(by_date.items(), reverse=True):
            st.markdown(f"##### {date}")
            for a in arts:
                flag  = " 🔔" if a.get("is_deal_alert") else ""
                amts  = ("  ·  *" + "  ".join(a["amounts"]) + "*") if a.get("amounts") else ""
                st.markdown(
                    f'**[{a["title"]}]({a["url"]})**{flag}{amts}  \n'
                    f'<span class="src-link">{a["source"]} ({a["source_cat"]})</span>',
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.markdown("#### Tracked Sources")
    sources_df = pd.DataFrame([
        {"Source": "Google News (AI deals)",         "Type": "Keyword search", "Update": "Every 6 hrs", "URL": "https://news.google.com"},
        {"Source": "Google News (hyperscaler capex)", "Type": "Keyword search", "Update": "Every 6 hrs", "URL": "https://news.google.com"},
        {"Source": "Google News (CoreWeave/xAI/NVIDIA)","Type": "Keyword search","Update": "Every 6 hrs","URL": "https://news.google.com"},
        {"Source": "Google News (Stargate/OpenAI)",   "Type": "Keyword search", "Update": "Every 6 hrs", "URL": "https://news.google.com"},
        {"Source": "Google News (cancellations)",     "Type": "Keyword search", "Update": "Every 6 hrs", "URL": "https://news.google.com"},
        {"Source": "TechCrunch",                      "Type": "Publication RSS", "Update": "Every 6 hrs", "URL": "https://techcrunch.com"},
        {"Source": "CNBC Technology",                 "Type": "Publication RSS", "Update": "Every 6 hrs", "URL": "https://cnbc.com"},
        {"Source": "DataCenterFrontier",              "Type": "DC Specialist",   "Update": "Every 6 hrs", "URL": "https://datacenterfrontier.com"},
        {"Source": "DataCenterDynamics",              "Type": "DC Specialist",   "Update": "Every 6 hrs", "URL": "https://datacenterdynamics.com"},
        {"Source": "Reuters Technology",              "Type": "Publication RSS", "Update": "Every 6 hrs", "URL": "https://reuters.com"},
        {"Source": "Ars Technica",                    "Type": "Publication RSS", "Update": "Every 6 hrs", "URL": "https://arstechnica.com"},
        {"Source": "The Register (Data Centre)",      "Type": "DC Specialist",   "Update": "Every 6 hrs", "URL": "https://theregister.com"},
        {"Source": "SemiAnalysis",                    "Type": "Deep Analysis",   "Update": "Every 6 hrs", "URL": "https://semianalysis.com"},
        {"Source": "The Verge",                       "Type": "Publication RSS", "Update": "Every 6 hrs", "URL": "https://theverge.com"},
    ])
    st.dataframe(sources_df, use_container_width=True, hide_index=True)
    st.markdown(
        '<span class="src-link">'
        "News articles are automatically fetched, keyword-filtered, and scored for deal signals. "
        "Articles do not represent endorsements. Always verify deal details from original sources. "
        "GitHub Actions workflow: .github/workflows/update_news.yml"
        "</span>",
        unsafe_allow_html=True,
    )
