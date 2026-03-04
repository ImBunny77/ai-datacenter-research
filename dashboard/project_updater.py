"""
dashboard/project_updater.py
============================
Runs every 2 hours via GitHub Actions. Actively monitors public sources for
new AI data center project announcements, expansions, cancellations, earnings
guidance changes, and official company statements.

Sources scanned:
  - SEC EDGAR full-text search API (8-K and 10-K filings — free, no key)
  - Google News RSS (12 targeted keyword queries)
  - DataCenterDynamics, DataCenterFrontier, SiliconAngle, The Register,
    SemiAnalysis, Ars Technica RSS feeds
  - Company investor relations RSS feeds (where available)
  - State economic development authority news

Output: dashboard/data/projects_feed.json

Run locally:
    pip install feedparser requests python-dateutil
    python dashboard/project_updater.py
"""

import hashlib
import json
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import feedparser
import requests

# ── Companies we track ────────────────────────────────────────────────────────
TRACKED_COMPANIES = [
    "microsoft", "amazon", "aws", "google", "alphabet", "meta", "oracle",
    "coreweave", "xai", "nvidia", "openai", "anthropic", "softbank",
    "broadcom", "apple", "tesla",
]

# ── SEC EDGAR CIKs for tracked public companies ───────────────────────────────
# CIK = Central Index Key; used by EDGAR to identify filers
SEC_CIKS = {
    "Microsoft":  "0000789019",
    "Amazon":     "0001018724",
    "Alphabet":   "0001652044",
    "Meta":       "0001326801",
    "Oracle":     "0001341439",
    "NVIDIA":     "0001045810",
    "CoreWeave":  "0002019662",
    "Broadcom":   "0001730168",
}

# ── Finding type keyword signals ──────────────────────────────────────────────
NEW_PROJECT_SIGNALS = [
    "announce", "commit", "invest", "build", "construct", "break ground",
    "groundbreaking", "new campus", "new data center", "billion", "gigawatt",
    "megawatt", "new region", "cloud region", "ai campus", "breaks ground",
    "new facility", "awarded", "signed", "partnership", "joint venture",
    "purchase site", "acquired land",
]

CANCELLATION_SIGNALS = [
    "cancel", "pause", "halt", "delay", "abandon", "suspend", "scrap",
    "withdraw", "pull back", "walk away", "canceled lease", "freeze",
    "pulled back", "scaled back", "putting on hold", "exit", "pulled out",
    "no longer proceeding", "reversed", "renegotiat", "terminated",
]

EXPANSION_SIGNALS = [
    "expand", "double", "increase", "additional", "phase 2", "phase 3",
    "next phase", "growing capacity", "add capacity", "upgrade", "more gpus",
    "raising investment", "upping", "larger", "extended", "extended through",
]

EARNINGS_SIGNALS = [
    "capex guidance", "capital expenditure guidance", "infrastructure spend",
    "quarterly earnings", "q1 earnings", "q2 earnings", "q3 earnings",
    "q4 earnings", "investor day", "raised guidance", "lowered guidance",
    "capex increase", "capex decrease", "spending outlook",
]

SEC_SIGNALS = [
    "material definitive agreement", "8-k", "significant construction",
    "capital commitment", "definitive agreement", "strategic investment",
    "significant agreement",
]

RELEVANCE_KEYWORDS = [
    "data center", "datacenter", "gpu cluster", "ai infrastructure",
    "cloud region", "hyperscale", "ai campus", "compute facility",
    "colocation", "ai factory", "power plant", "gigawatt", "megawatt",
    "nvidia", "gpu", "blackwell", "h100", "b200",
] + TRACKED_COMPANIES

# ── RSS feeds to scan ─────────────────────────────────────────────────────────
PROJECT_FEEDS = [
    # Google News — targeted project announcement queries
    dict(name="GNews: data center announce invest",
         url="https://news.google.com/rss/search?q=data+center+billion+invest+announce&hl=en-US&gl=US&ceid=US:en",
         category="Announcement"),
    dict(name="GNews: data center cancel pause halt",
         url="https://news.google.com/rss/search?q=data+center+cancel+pause+halt+AI+infrastructure&hl=en-US&gl=US&ceid=US:en",
         category="Cancellation"),
    dict(name="GNews: Microsoft data center build",
         url="https://news.google.com/rss/search?q=Microsoft+data+center+billion+invest+build+Azure&hl=en-US&gl=US&ceid=US:en",
         category="Company"),
    dict(name="GNews: Amazon AWS data center",
         url="https://news.google.com/rss/search?q=Amazon+AWS+data+center+billion+invest+build&hl=en-US&gl=US&ceid=US:en",
         category="Company"),
    dict(name="GNews: Google Alphabet data center",
         url="https://news.google.com/rss/search?q=Google+Alphabet+data+center+billion+invest+build&hl=en-US&gl=US&ceid=US:en",
         category="Company"),
    dict(name="GNews: Meta data center",
         url="https://news.google.com/rss/search?q=Meta+data+center+billion+invest+Hyperion+Llama&hl=en-US&gl=US&ceid=US:en",
         category="Company"),
    dict(name="GNews: Oracle OCI Stargate",
         url="https://news.google.com/rss/search?q=Oracle+OCI+data+center+Stargate+billion+invest&hl=en-US&gl=US&ceid=US:en",
         category="Company"),
    dict(name="GNews: CoreWeave xAI GPU",
         url="https://news.google.com/rss/search?q=CoreWeave+xAI+Colossus+data+center+billion+GPU&hl=en-US&gl=US&ceid=US:en",
         category="Company"),
    dict(name="GNews: hyperscaler capex earnings guidance",
         url="https://news.google.com/rss/search?q=hyperscaler+capex+guidance+data+center+spend+earnings&hl=en-US&gl=US&ceid=US:en",
         category="Earnings"),
    dict(name="GNews: data center gigawatt nuclear power",
         url="https://news.google.com/rss/search?q=AI+data+center+gigawatt+nuclear+renewable+power+2025+2026&hl=en-US&gl=US&ceid=US:en",
         category="Infrastructure"),
    dict(name="GNews: breaks ground data center",
         url="https://news.google.com/rss/search?q=%22breaks+ground%22+OR+%22groundbreaking%22+data+center+billion&hl=en-US&gl=US&ceid=US:en",
         category="Announcement"),
    dict(name="GNews: state economic development data center",
         url="https://news.google.com/rss/search?q=%22economic+development%22+%22data+center%22+billion+announce&hl=en-US&gl=US&ceid=US:en",
         category="State/Gov"),
    dict(name="GNews: SEC 8-K data center capital",
         url="https://news.google.com/rss/search?q=SEC+8-K+%22data+center%22+%22capital+commitment%22+billion&hl=en-US&gl=US&ceid=US:en",
         category="SEC Filing"),
    # Specialist data center publications
    dict(name="DataCenterDynamics",
         url="https://www.datacenterdynamics.com/en/rss/",
         category="DC Specialist"),
    dict(name="DataCenterFrontier",
         url="https://www.datacenterfrontier.com/rss",
         category="DC Specialist"),
    dict(name="SiliconAngle",
         url="https://siliconangle.com/feed/",
         category="Tech Analysis"),
    dict(name="The Register (Data Centre)",
         url="https://www.theregister.com/data_centre/rss",
         category="DC Specialist"),
    dict(name="SemiAnalysis",
         url="https://newsletter.semianalysis.com/feed",
         category="Deep Analysis"),
    dict(name="Ars Technica",
         url="https://feeds.arstechnica.com/arstechnica/index",
         category="Tech News"),
    # Company investor relations (RSS where available)
    dict(name="Microsoft IR Blog",
         url="https://blogs.microsoft.com/feed/",
         category="Company IR"),
    dict(name="Amazon AWS News",
         url="https://aws.amazon.com/blogs/aws/feed/",
         category="Company IR"),
    dict(name="Google Cloud Blog",
         url="https://cloud.google.com/blog/rss.xml",
         category="Company IR"),
    dict(name="Google Blog",
         url="https://blog.google/rss/",
         category="Company IR"),
    dict(name="Meta Newsroom",
         url="https://about.fb.com/news/feed/",
         category="Company IR"),
    dict(name="Oracle News",
         url="https://www.oracle.com/corporate/pressrelease/rss/rss.xml",
         category="Company IR"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text or "").strip()


def finding_id(url: str, title: str) -> str:
    return hashlib.md5(f"{url}|{title}".encode()).hexdigest()[:14]


def extract_amounts(text: str) -> list[str]:
    patterns = [
        r"\$[\d,]+\.?\d*\s*(?:billion|million|trillion|B|M|T)\b",
        r"[\d,]+\.?\d*\s*(?:billion|million)\s*dollar",
        r"EUR\s*[\d,]+\.?\d*\s*(?:billion|million|B|M)\b",
        r"[\d,]+\.?\d*\s*(?:gigawatt|megawatt|GW|MW)\b",
    ]
    found = []
    for pat in patterns:
        found.extend(re.findall(pat, text, re.IGNORECASE))
    return list(set(found))[:6]


def extract_companies(text: str) -> list[str]:
    found = []
    text_l = text.lower()
    mapping = {
        "microsoft": "Microsoft", "azure": "Microsoft",
        "amazon": "Amazon", "aws": "Amazon",
        "google": "Google", "alphabet": "Google",
        "meta": "Meta", "facebook": "Meta",
        "oracle": "Oracle",
        "coreweave": "CoreWeave",
        "xai": "xAI", "grok": "xAI",
        "nvidia": "NVIDIA",
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "softbank": "SoftBank",
    }
    for kw, co in mapping.items():
        if kw in text_l and co not in found:
            found.append(co)
    return found


def extract_locations(text: str) -> list[str]:
    patterns = [
        # US states
        r"\b(Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|"
        r"Delaware|Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|"
        r"Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|"
        r"Mississippi|Missouri|Montana|Nebraska|Nevada|New Hampshire|New Jersey|"
        r"New Mexico|New York|North Carolina|North Dakota|Ohio|Oklahoma|Oregon|"
        r"Pennsylvania|Rhode Island|South Carolina|South Dakota|Tennessee|Texas|"
        r"Utah|Vermont|Virginia|Washington|West Virginia|Wisconsin|Wyoming|"
        r"Washington D\.?C\.?)\b",
        # Countries (international)
        r"\b(Germany|France|Japan|Australia|UK|United Kingdom|India|Belgium|"
        r"Netherlands|Singapore|Malaysia|Indonesia|Brazil|UAE|Saudi Arabia|"
        r"Canada|Ireland|Spain|Poland|Sweden|Norway|Finland|Denmark)\b",
        # Major cities relevant to data centers
        r"\b(Loudoun|Ashburn|Chicago|Dallas|Phoenix|Atlanta|Seattle|Boston|"
        r"Frankfurt|London|Tokyo|Sydney|Singapore|Mumbai|Bangalore|Paris|"
        r"Memphis|Abilene|Montgomery|Lebanon|Kenilworth|Lancaster|Midlothian)\b",
    ]
    found = []
    for pat in patterns:
        for m in re.findall(pat, text, re.IGNORECASE):
            if m not in found:
                found.append(m)
    return found[:8]


def classify_finding(title: str, summary: str) -> str:
    text = (title + " " + summary).lower()
    cancel_score = sum(1 for s in CANCELLATION_SIGNALS if s in text)
    new_score    = sum(1 for s in NEW_PROJECT_SIGNALS if s in text)
    expand_score = sum(1 for s in EXPANSION_SIGNALS if s in text)
    earn_score   = sum(1 for s in EARNINGS_SIGNALS if s in text)
    sec_score    = sum(1 for s in SEC_SIGNALS if s in text)

    if cancel_score >= 1:
        return "Cancellation / Pause"
    if sec_score >= 1:
        return "SEC Filing"
    if earn_score >= 2:
        return "Earnings / Guidance"
    if expand_score >= 2:
        return "Expansion"
    if new_score >= 2:
        return "New Project"
    return "Statement / Update"


def confidence_score(title: str, summary: str, amounts: list, companies: list) -> int:
    """0-10: how confident are we this is a real capital event?"""
    score = 0
    text = (title + " " + summary).lower()
    if amounts:
        score += 3
    if companies:
        score += 2
    if any(s in text for s in ["data center", "datacenter", "cloud region"]):
        score += 2
    if any(s in text for s in ["billion", "gigawatt", "megawatt"]):
        score += 1
    if any(s in text for s in ["announce", "commit", "invest", "8-k", "press release"]):
        score += 1
    if any(s in text for s in CANCELLATION_SIGNALS):
        score += 1  # cancellations are also high value
    return min(score, 10)


def is_relevant(title: str, summary: str) -> bool:
    text = (title + " " + summary).lower()
    return any(kw in text for kw in RELEVANCE_KEYWORDS)


def parse_date(entry) -> str:
    for field in ("published", "updated", "created"):
        val = entry.get(field, "")
        if not val:
            continue
        try:
            from email.utils import parsedate_to_datetime
            return parsedate_to_datetime(val).strftime("%Y-%m-%d")
        except Exception:
            pass
        try:
            import dateutil.parser
            return dateutil.parser.parse(val).strftime("%Y-%m-%d")
        except Exception:
            pass
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ── SEC EDGAR scanner ─────────────────────────────────────────────────────────

def fetch_edgar_filings(days_back: int = 3) -> list[dict]:
    """
    Query SEC EDGAR full-text search for recent 8-K/8-K/A filings mentioning
    'data center' and 'billion'. EDGAR API is public, no auth required.
    """
    findings = []
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")

    queries = [
        '"data center" "billion"',
        '"AI infrastructure" "capital"',
        '"cloud infrastructure" "invest"',
        '"gigawatt" "data center"',
    ]

    for q in queries:
        try:
            url = (
                "https://efts.sec.gov/LATEST/search-index"
                f"?q={requests.utils.quote(q)}"
                "&forms=8-K,8-K/A"
                f"&dateRange=custom&startdt={cutoff}"
            )
            resp = requests.get(url, timeout=15,
                                headers={"User-Agent": "ai-datacenter-research/1.0 research@example.com"})
            if resp.status_code != 200:
                continue

            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])
            for hit in hits[:20]:
                src = hit.get("_source", {})
                title    = src.get("file_date", "?") + " — " + src.get("entity_name", "Unknown") + " 8-K"
                filer    = src.get("entity_name", "Unknown filer")
                date_str = src.get("file_date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
                form     = src.get("form_type", "8-K")
                acc      = src.get("accession_no", "").replace("-", "")
                cik      = src.get("entity_id", "")
                if acc:
                    filing_url = f"https://www.sec.gov/Archives/edgar/full-index/{date_str[:4]}/{date_str[5:7]}/{acc}/{acc}-index.htm"
                else:
                    filing_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=8-K&dateb=&owner=include&count=10"

                summary_text = src.get("period_of_report", "") + " " + form
                amounts   = extract_amounts(title + " " + filer)
                companies = extract_companies(filer)

                fid = finding_id(filing_url, title)
                findings.append(dict(
                    id=fid,
                    title=f"SEC {form}: {filer}",
                    url=filing_url,
                    source="SEC EDGAR (official filing)",
                    source_cat="SEC Filing",
                    date=date_str,
                    summary=f"Official {form} filing by {filer}. Period: {src.get('period_of_report','')}.",
                    finding_type="SEC Filing",
                    companies=companies,
                    locations=[],
                    amounts=amounts,
                    confidence=7,
                    is_high_confidence=True,
                ))
            print(f"  EDGAR '{q}': {len(hits)} hits")
        except Exception as e:
            print(f"  [WARN] EDGAR query '{q}': {e}", file=sys.stderr)

    return findings


# ── RSS scanner ───────────────────────────────────────────────────────────────

def fetch_all_feeds() -> list[dict]:
    findings = []
    seen_ids: set[str] = set()

    for feed_info in PROJECT_FEEDS:
        try:
            feed  = feedparser.parse(feed_info["url"])
            count = 0
            for entry in feed.entries[:30]:
                url   = entry.get("link", "").strip()
                title = strip_html(entry.get("title", "")).strip()
                raw   = entry.get("summary", entry.get("description", ""))
                summary = strip_html(raw)[:700]

                if not url or not title:
                    continue
                if not is_relevant(title, summary):
                    continue

                fid = finding_id(url, title)
                if fid in seen_ids:
                    continue
                seen_ids.add(fid)

                amounts   = extract_amounts(title + " " + summary)
                companies = extract_companies(title + " " + summary)
                locations = extract_locations(title + " " + summary)
                ftype     = classify_finding(title, summary)
                conf      = confidence_score(title, summary, amounts, companies)
                date      = parse_date(entry)

                findings.append(dict(
                    id=fid,
                    title=title,
                    url=url,
                    source=feed_info["name"],
                    source_cat=feed_info["category"],
                    date=date,
                    summary=summary[:500],
                    finding_type=ftype,
                    companies=companies,
                    locations=locations,
                    amounts=amounts,
                    confidence=conf,
                    is_high_confidence=conf >= 5,
                ))
                count += 1

            print(f"  {feed_info['name']}: {count} relevant findings")
        except Exception as e:
            print(f"  [WARN] {feed_info['name']}: {e}", file=sys.stderr)

    return findings


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    out_dir  = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "projects_feed.json"

    print(f"Scanning {len(PROJECT_FEEDS)} RSS feeds for project intelligence...")
    rss_findings = fetch_all_feeds()
    print(f"Found {len(rss_findings)} relevant RSS findings")

    print("Querying SEC EDGAR for recent 8-K filings...")
    sec_findings = fetch_edgar_filings(days_back=4)
    print(f"Found {len(sec_findings)} relevant SEC filings")

    all_fresh = rss_findings + sec_findings

    # Merge with existing to preserve history (keep 60 days)
    existing: dict[str, dict] = {}
    if out_file.exists():
        try:
            old_data = json.loads(out_file.read_text(encoding="utf-8"))
            cutoff_date = (datetime.now(timezone.utc) - timedelta(days=60)).strftime("%Y-%m-%d")
            for f in old_data.get("findings", []):
                if f.get("date", "0000-00-00") >= cutoff_date:
                    existing[f["id"]] = f
            print(f"Loaded {len(existing)} existing findings (last 60 days)")
        except Exception as e:
            print(f"[WARN] Could not load existing findings: {e}", file=sys.stderr)

    merged = {f["id"]: f for f in all_fresh}
    for fid, finding in existing.items():
        if fid not in merged:
            merged[fid] = finding

    all_findings = sorted(
        merged.values(),
        key=lambda x: (x["date"], x["confidence"]),
        reverse=True
    )[:600]

    cancellations = [f for f in all_findings if "Cancellation" in f.get("finding_type", "")]
    new_projects  = [f for f in all_findings if f.get("finding_type") == "New Project"]
    sec_filings   = [f for f in all_findings if f.get("finding_type") == "SEC Filing"]
    high_conf     = [f for f in all_findings if f.get("is_high_confidence")]

    output = dict(
        last_updated=datetime.now(timezone.utc).isoformat(),
        finding_count=len(all_findings),
        new_project_count=len(new_projects),
        cancellation_count=len(cancellations),
        sec_filing_count=len(sec_filings),
        high_confidence_count=len(high_conf),
        findings=all_findings,
    )
    out_file.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Saved {len(all_findings)} findings "
        f"({len(new_projects)} new projects, {len(cancellations)} cancellations, "
        f"{len(sec_filings)} SEC filings) -> {out_file}"
    )


if __name__ == "__main__":
    run()
