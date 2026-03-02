"""
dashboard/news_updater.py
=========================
Fetches AI data center news from trusted RSS feeds and saves structured
articles to dashboard/data/news_feed.json.

Run by GitHub Actions every 6 hours (see .github/workflows/update_news.yml).
Can also be run locally:
    pip install feedparser requests python-dateutil
    python dashboard/news_updater.py

Trusted sources:
  - Google News (keyword-targeted — broadest coverage)
  - TechCrunch
  - CNBC Technology
  - DataCenterFrontier
  - DataCenterDynamics
  - Reuters Technology
  - Ars Technica
  - The Register (Data Centre section)
  - SemiAnalysis (Substack RSS)
"""

import json
import hashlib
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import requests

# ── Trusted RSS feeds ─────────────────────────────────────────────────────────
FEEDS = [
    # Google News keyword searches (no API key needed)
    {
        "name": "Google News: AI data center deals",
        "url":  "https://news.google.com/rss/search?q=AI+data+center+deal+billion+GPU+infrastructure&hl=en-US&gl=US&ceid=US:en",
        "category": "Deal",
    },
    {
        "name": "Google News: hyperscaler capex",
        "url":  "https://news.google.com/rss/search?q=Microsoft+Amazon+Google+Meta+Oracle+AI+infrastructure+capex+2025+2026&hl=en-US&gl=US&ceid=US:en",
        "category": "Capex",
    },
    {
        "name": "Google News: CoreWeave NVIDIA xAI",
        "url":  "https://news.google.com/rss/search?q=CoreWeave+xAI+NVIDIA+Colossus+GPU+cluster+data+center&hl=en-US&gl=US&ceid=US:en",
        "category": "Infrastructure",
    },
    {
        "name": "Google News: Stargate OpenAI data center",
        "url":  "https://news.google.com/rss/search?q=Stargate+OpenAI+data+center+Oracle+SoftBank&hl=en-US&gl=US&ceid=US:en",
        "category": "Deal",
    },
    {
        "name": "Google News: AI data center canceled paused",
        "url":  "https://news.google.com/rss/search?q=AI+data+center+canceled+paused+lease+writedown&hl=en-US&gl=US&ceid=US:en",
        "category": "Cancel/Pause",
    },
    # Publication feeds
    {
        "name": "TechCrunch",
        "url":  "https://techcrunch.com/feed/",
        "category": "Tech News",
    },
    {
        "name": "CNBC Technology",
        "url":  "https://www.cnbc.com/id/19854910/device/rss/rss.html",
        "category": "Finance/Tech",
    },
    {
        "name": "DataCenterFrontier",
        "url":  "https://www.datacenterfrontier.com/rss",
        "category": "DC Specialist",
    },
    {
        "name": "DataCenterDynamics",
        "url":  "https://www.datacenterdynamics.com/en/rss/",
        "category": "DC Specialist",
    },
    {
        "name": "Reuters Technology",
        "url":  "https://feeds.reuters.com/reuters/technologyNews",
        "category": "Finance/Tech",
    },
    {
        "name": "Ars Technica",
        "url":  "https://feeds.arstechnica.com/arstechnica/index",
        "category": "Tech News",
    },
    {
        "name": "The Register (Data Centre)",
        "url":  "https://www.theregister.com/data_centre/rss",
        "category": "DC Specialist",
    },
    {
        "name": "SemiAnalysis",
        "url":  "https://newsletter.semianalysis.com/feed",
        "category": "Deep Analysis",
    },
    {
        "name": "The Verge (Tech)",
        "url":  "https://www.theverge.com/rss/index.xml",
        "category": "Tech News",
    },
]

# ── Keyword filters ───────────────────────────────────────────────────────────
# Article must contain at least one RELEVANCE keyword
RELEVANCE_KEYWORDS = [
    "data center", "datacenter", "gpu", "nvidia", "openai", "microsoft",
    "xai", "grok", "oracle", "amazon", "aws", "coreweave", "meta", "google",
    "hyperscale", "ai infrastructure", "colossus", "stargate", "capex",
    "compute", "inference", "training cluster", "blackwell", "h100", "h200",
    "b200", "power plant", "gigawatt", "megawatt", "anthropic", "softbank",
    "broadcom", "llm", "foundational model", "large language",
]

# If title contains any DEAL_SIGNAL, the article is flagged as a potential new deal
DEAL_SIGNALS = [
    "billion", "million", "deal", "contract", "invest", "acqui", "lease",
    "purchase", "fund", "raise", "ipo", "commit", "partner", "agreement",
    "announce", "sign", "awarded", "selected",
]

# Companies we track — for extraction hints
TRACKED_COMPANIES = [
    "microsoft", "openai", "nvidia", "coreweave", "oracle", "xai", "amazon",
    "aws", "meta", "google", "alphabet", "anthropic", "softbank", "broadcom",
    "blackrock", "g42", "mgx", "humain",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text or "").strip()


def article_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:14]


def is_relevant(title: str, summary: str) -> bool:
    text = (title + " " + summary).lower()
    return any(kw in text for kw in RELEVANCE_KEYWORDS)


def deal_score(title: str) -> int:
    """Return 0-10 score for how likely this article describes a new deal."""
    text = title.lower()
    score = 0
    for sig in DEAL_SIGNALS:
        if sig in text:
            score += 2
    for co in TRACKED_COMPANIES:
        if co in text:
            score += 1
    return min(score, 10)


def extract_amounts(text: str) -> list[str]:
    """Extract dollar amounts mentioned in text (e.g. '$5B', '$200 billion')."""
    patterns = [
        r"\$[\d,]+\.?\d*\s*(?:billion|million|B|M|T|trillion)\b",
        r"[\d,]+\.?\d*\s*(?:billion|million)\s*dollar",
    ]
    found = []
    for pat in patterns:
        found.extend(re.findall(pat, text, re.IGNORECASE))
    return list(set(found))[:5]


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


# ── Main fetch ────────────────────────────────────────────────────────────────

def fetch_all_feeds() -> list[dict]:
    articles = []
    seen_ids: set[str] = set()

    for feed_info in FEEDS:
        try:
            feed = feedparser.parse(feed_info["url"])
            count = 0
            for entry in feed.entries[:40]:
                url   = entry.get("link", "").strip()
                title = strip_html(entry.get("title", "")).strip()
                raw_summary = entry.get("summary", entry.get("description", ""))
                summary = strip_html(raw_summary)[:600]

                if not url or not title:
                    continue

                aid = article_id(url)
                if aid in seen_ids:
                    continue
                seen_ids.add(aid)

                if not is_relevant(title, summary):
                    continue

                ds     = deal_score(title + " " + summary[:200])
                amts   = extract_amounts(title + " " + summary[:400])
                date   = parse_date(entry)

                articles.append({
                    "id":           aid,
                    "title":        title,
                    "url":          url,
                    "source":       feed_info["name"],
                    "source_cat":   feed_info["category"],
                    "date":         date,
                    "summary":      summary,
                    "deal_score":   ds,
                    "amounts":      amts,
                    "is_deal_alert": ds >= 4,
                })
                count += 1

            print(f"  {feed_info['name']}: {count} relevant articles")

        except Exception as e:
            print(f"  [WARN] {feed_info['name']}: {e}", file=sys.stderr)

    # Sort by date desc, then deal_score desc
    articles.sort(key=lambda x: (x["date"], x["deal_score"]), reverse=True)
    return articles


def run():
    out_dir  = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "news_feed.json"

    print(f"Fetching from {len(FEEDS)} trusted RSS feeds...")
    fresh = fetch_all_feeds()
    print(f"Fetched {len(fresh)} relevant articles")

    # Merge with existing to preserve history
    existing: dict[str, dict] = {}
    if out_file.exists():
        try:
            old_data = json.loads(out_file.read_text(encoding="utf-8"))
            existing = {a["id"]: a for a in old_data.get("articles", [])}
            print(f"Loaded {len(existing)} existing articles")
        except Exception as e:
            print(f"[WARN] Could not load existing feed: {e}", file=sys.stderr)

    # Fresh entries take priority; keep up to 400 total
    merged = {a["id"]: a for a in fresh}
    for aid, art in existing.items():
        if aid not in merged:
            merged[aid] = art

    all_articles = sorted(merged.values(), key=lambda x: (x["date"], x["deal_score"]), reverse=True)[:400]
    deal_alerts  = [a for a in all_articles if a.get("is_deal_alert")]

    output = {
        "last_updated":   datetime.now(timezone.utc).isoformat(),
        "article_count":  len(all_articles),
        "deal_alert_count": len(deal_alerts),
        "articles":       all_articles,
    }
    out_file.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(all_articles)} articles ({len(deal_alerts)} deal alerts) -> {out_file}")


if __name__ == "__main__":
    run()
