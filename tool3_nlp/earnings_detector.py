"""
Earnings Call CAPEX Conviction Analyzer
=========================================
Analyzes earnings call transcripts for AI hyperscalers using:
  1. Custom rule-based scoring (fast, no GPU required)
  2. FinBERT sentiment analysis (transformer-based, optional)

Outputs:
  - Terminal ranking table
  - capex_conviction_scores.csv

Usage:
  python earnings_detector.py                   # reads ./transcripts/*.txt
  python earnings_detector.py --no-finbert      # rule-based only (faster)
"""

import os
import re
import sys
import glob
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ── Directories ───────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
TRANSCRIPTS   = BASE_DIR / "transcripts"
OUTPUT_CSV    = BASE_DIR / "capex_conviction_scores.csv"

# ── Custom Scoring Lexicon ────────────────────────────────────────────────────

# Negative weight: hedging, waffling, uncertainty language
HEDGE_WORDS = {
    # Explicit soft commitments
    "evaluating":           2.0,
    "re-assessing":         3.0,
    "reassessing":          3.0,
    "right-sizing":         3.0,
    "rightsizing":          3.0,
    "pacing":               1.5,
    "carefully":            1.0,
    "monitoring":           1.5,
    "watching":             1.0,
    "remain flexible":      2.0,
    "remaining flexible":   2.0,
    "long-term horizon":    2.0,
    "longer-term horizon":  2.0,
    "multi-year":           0.5,
    "early stages":         2.0,
    "too early":            2.5,
    "may need to":          2.0,
    "could accelerate":     1.5,
    "could decelerate":     2.5,
    "thoughtful":           1.0,
    "disciplined":          0.5,
    "deliberate":           1.0,
    "deliberately":         1.0,
    "appropriate pacing":   2.0,
    "demand curve":         1.0,
    "weighing on":          1.5,
    "weigh on":             1.5,
    "still in the":         1.5,
    "draw firm":            2.0,
    "firm conclusions":     2.0,
    "fair question":        1.0,
    "continuously":         0.5,
    "over-building":        1.5,
    "structural challenges":2.0,
    "not locked":           1.5,
    "not a concern":        0.5,   # slight dismissal
    "not locked into":      2.0,
}

# Positive weight: conviction, concrete metrics, deployed capital
CONVICTION_WORDS = {
    # Concrete financial metrics
    "free cash flow":       2.5,
    "operating income":     2.0,
    "gross margin":         2.0,
    "accretive":            3.0,
    "roi":                  3.0,
    "return on investment": 3.0,
    "deployed":             2.5,
    "commissioned":         2.5,
    "contracted":           3.5,
    "committed":            3.0,
    "take-or-pay":          4.0,
    "pre-leased":           3.5,
    "booked":               2.5,
    "shovels":              4.0,
    "under construction":   3.0,
    "active construction":  3.0,
    "breaking ground":      3.0,
    "broke ground":         3.0,
    "fully funded":         3.5,
    "board approved":       3.0,
    "board-approved":       3.0,
    "purchase order":       3.0,
    "purchase orders":      3.0,
    "record":               1.5,
    "backlog":              2.5,
    "remaining performance":2.0,
    "rpo":                  2.0,
    "year-over-year":       1.0,
    "year over year":       1.0,
    "utilization":          2.0,
    "utilization rate":     2.5,
    "percent utilization":  2.5,
    "sampled":              1.5,
    "production":           1.5,
    "in production":        2.5,
    "fully deployed":       3.0,
    "fully on track":       2.5,
    "on track":             1.5,
    "clear contractual":    3.5,
    "zero speculative":     4.0,
    "no speculative":       4.0,
    "funded":               2.0,
    "accelerating":         2.0,
}

# Amplifiers — multiply adjacent hedge/conviction score if found nearby
AMPLIFIERS = {
    "not": -1.2,     # "not evaluating" reverses a hedge into conviction
    "never": -1.5,
    "zero": 1.5,
    "absolutely": 1.3,
    "completely": 1.2,
    "fully": 1.2,
    "deeply": 1.2,
}


def extract_ticker(filename: str) -> str:
    """Parse ticker from filename like MSFT_Q2_FY2025_fake.txt"""
    parts = Path(filename).stem.split("_")
    return parts[0] if parts else filename


def score_text_rule_based(text: str) -> dict:
    """
    Scan text for hedge and conviction signals.
    Returns raw scores and matched phrases.
    """
    text_lower = text.lower()
    words = re.findall(r"[\w\-]+", text_lower)
    word_set = set(words)

    hedge_score     = 0.0
    conviction_score = 0.0
    hedge_matches   = defaultdict(int)
    conviction_matches = defaultdict(int)

    # Multi-word phrase matching (up to 4 words)
    for phrase, weight in HEDGE_WORDS.items():
        count = text_lower.count(phrase)
        if count:
            hedge_score += weight * count
            hedge_matches[phrase] += count

    for phrase, weight in CONVICTION_WORDS.items():
        count = text_lower.count(phrase)
        if count:
            conviction_score += weight * count
            conviction_matches[phrase] += count

    # Simple amplifier: "not" before a hedge word reduces hedge score
    not_before_hedge = 0
    for phrase in HEDGE_WORDS:
        pattern = r"\bnot\s+" + re.escape(phrase)
        matches = re.findall(pattern, text_lower)
        not_before_hedge += len(matches)
    # Each "not <hedge>" deducts double from hedge and adds to conviction
    hedge_score     -= not_before_hedge * 3.0
    conviction_score += not_before_hedge * 2.0

    # Normalize by word count (per 1000 words)
    word_count = max(len(words), 1)
    norm = 1000 / word_count

    return {
        "hedge_raw":         hedge_score,
        "conviction_raw":    conviction_score,
        "hedge_norm":        round(hedge_score * norm, 2),
        "conviction_norm":   round(conviction_score * norm, 2),
        "word_count":        word_count,
        "hedge_matches":     dict(hedge_matches),
        "conviction_matches":dict(conviction_matches),
    }


_FINBERT_PIPE = None   # module-level cache — loaded once


def finbert_sentiment(texts: list[str], batch_size: int = 8) -> list[dict]:
    """
    Run FinBERT (ProsusAI/finbert) over sentence chunks.
    Pipeline is cached after first load to avoid reloading per-transcript.
    Returns list of {positive, negative, neutral} score dicts.
    """
    global _FINBERT_PIPE
    try:
        from transformers import pipeline, logging as hf_logging
        hf_logging.set_verbosity_error()   # suppress loading noise
        if _FINBERT_PIPE is None:
            print("  Loading FinBERT model (ProsusAI/finbert)...", flush=True)
            _FINBERT_PIPE = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                top_k=None,       # return all 3 labels
                device=-1,        # CPU
                truncation=True,
                max_length=512,
            )
    except ImportError:
        print("  [!] transformers not installed - skipping FinBERT")
        return [{"positive": 0, "negative": 0, "neutral": 1}] * len(texts)

    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        batch_out = _FINBERT_PIPE(batch)
        for item_labels in batch_out:
            d = {lbl["label"].lower(): lbl["score"] for lbl in item_labels}
            results.append(d)
    return results


def chunk_text(text: str, max_chars: int = 400) -> list[str]:
    """Split transcript into sentence-level chunks for FinBERT."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) < max_chars:
            current += " " + s
        else:
            if current.strip():
                chunks.append(current.strip())
            current = s
    if current.strip():
        chunks.append(current.strip())
    return chunks


def analyze_transcripts(transcript_dir: Path, use_finbert: bool = True) -> pd.DataFrame:
    files = sorted(glob.glob(str(transcript_dir / "*.txt")))
    if not files:
        print(f"No .txt files found in {transcript_dir}")
        sys.exit(1)

    print(f"\nFound {len(files)} transcript(s):")
    for f in files:
        print(f"  {Path(f).name}")

    rows = []
    for fpath in files:
        ticker = extract_ticker(fpath)
        text = Path(fpath).read_text(encoding="utf-8", errors="replace")
        print(f"\n  Scoring {ticker} ({len(text):,} chars)...")

        rule = score_text_rule_based(text)

        # --- FinBERT ---
        fb_positive = fb_negative = None
        if use_finbert:
            chunks = chunk_text(text)
            fb_results = finbert_sentiment(chunks)
            fb_positive = float(np.mean([r.get("positive", 0) for r in fb_results]))
            fb_negative = float(np.mean([r.get("negative", 0) for r in fb_results]))

        # --- Composite CAPEX Conviction Score ---
        # Higher = more concrete / less hedging
        # Scale: 0 (pure hedge) to 100 (maximum conviction)
        raw_diff = rule["conviction_norm"] - rule["hedge_norm"]
        max_possible = 60.0   # calibrated to transcript scale
        capex_conviction = max(0, min(100, 50 + (raw_diff / max_possible) * 50))

        rows.append({
            "ticker":               ticker,
            "file":                 Path(fpath).name,
            "word_count":           rule["word_count"],
            "conviction_norm":      rule["conviction_norm"],
            "hedge_norm":           rule["hedge_norm"],
            "hedge_conviction_diff":round(raw_diff, 2),
            "capex_conviction_score":round(capex_conviction, 1),
            "finbert_positive":     round(fb_positive, 4) if fb_positive is not None else "N/A",
            "finbert_negative":     round(fb_negative, 4) if fb_negative is not None else "N/A",
            "top_hedge_phrases":    sorted(rule["hedge_matches"].items(),
                                          key=lambda x: -x[1])[:5],
            "top_conviction_phrases": sorted(rule["conviction_matches"].items(),
                                             key=lambda x: -x[1])[:5],
        })

    df = pd.DataFrame(rows).sort_values("capex_conviction_score", ascending=False)
    return df


def print_report(df: pd.DataFrame):
    print("\n" + "="*70)
    print("  CAPEX CONVICTION SCORE  -  EARNINGS CALL ANALYSIS")
    print("  (100 = Maximum conviction | 0 = Maximum hedging)")
    print("="*70)

    display_cols = [
        "ticker", "capex_conviction_score", "conviction_norm",
        "hedge_norm", "hedge_conviction_diff",
    ]
    if "finbert_positive" in df.columns and df["finbert_positive"].iloc[0] != "N/A":
        display_cols += ["finbert_positive", "finbert_negative"]

    print(df[display_cols].to_string(index=False))
    print()

    print("  -- Top Hedge Phrases --")
    for _, row in df.iterrows():
        print(f"  [{row['ticker']}]  {row['top_hedge_phrases']}")

    print()
    print("  -- Top Conviction Phrases --")
    for _, row in df.iterrows():
        print(f"  [{row['ticker']}]  {row['top_conviction_phrases']}")

    print()
    print("  RANKING  (High conviction = less hedging on CAPEX commitments)")
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        bar = "#" * int(row["capex_conviction_score"] / 5)
        verdict = ""
        if row["capex_conviction_score"] >= 70:
            verdict = "  [COMMITTED]"
        elif row["capex_conviction_score"] >= 50:
            verdict = "  [NEUTRAL]"
        else:
            verdict = "  [HEDGING]"
        print(f"  {rank}. {row['ticker']:<8} {row['capex_conviction_score']:5.1f}/100  "
              f"{bar:<20}{verdict}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Earnings Call CAPEX Conviction Analyzer")
    parser.add_argument("--transcripts", default=str(TRANSCRIPTS),
                        help="Path to folder of .txt transcript files")
    parser.add_argument("--no-finbert", action="store_true",
                        help="Skip FinBERT (rule-based scoring only, faster)")
    args = parser.parse_args()

    transcript_path = Path(args.transcripts)
    use_finbert = not args.no_finbert

    if use_finbert:
        try:
            import transformers  # noqa
            print("  [+] transformers available - will run FinBERT")
        except ImportError:
            print("  [!] transformers not installed - running rule-based only")
            use_finbert = False

    df = analyze_transcripts(transcript_path, use_finbert=use_finbert)
    print_report(df)

    # Save CSV (drop list columns which don't serialize cleanly)
    csv_df = df.drop(columns=["top_hedge_phrases", "top_conviction_phrases"], errors="ignore")
    csv_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Results saved -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
