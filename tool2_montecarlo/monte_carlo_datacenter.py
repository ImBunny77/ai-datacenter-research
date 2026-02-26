"""
1GW AI Data Center - Monte Carlo P&L & ROI Modeler
====================================================
Reads config.env and runs N simulations stress-testing:
  - Power cost volatility
  - GPU downtime
  - Rental price compression over time

Outputs:
  - Terminal summary table
  - Histogram: probability distribution of 15x multiple vs bankruptcy
  - Saves: datacenter_mc_results.png
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless – no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Load config ──────────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).parent / "config.env"
OUTPUT_IMG  = Path(__file__).parent / "datacenter_mc_results.png"


def load_config(path: Path) -> dict:
    cfg = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                val = val.split("#")[0].strip()   # strip inline comments
                try:
                    cfg[key.strip()] = float(val)
                except ValueError:
                    cfg[key.strip()] = val
    return cfg


# ── Core simulation ──────────────────────────────────────────────────────────

def run_simulations(cfg: dict) -> pd.DataFrame:
    np.random.seed(int(cfg["MC_RANDOM_SEED"]))
    N    = int(cfg["MC_SIMULATIONS"])
    YRS  = int(cfg["SIMULATION_YEARS"])

    # --- Datacenter sizing ---
    power_mw      = cfg["DATACENTER_POWER_MW"]
    rack_kw       = cfg["RACK_DENSITY_KW"]
    gpus_per_rack = int(cfg["GPUS_PER_RACK"])
    gpu_cost      = cfg["GPU_UNIT_COST_USD"]

    total_racks = (power_mw * 1000) / rack_kw          # 25,000 racks @ 1GW / 40kW
    total_gpus  = total_racks * gpus_per_rack            # 200,000 GPUs

    # --- CAPEX ---
    gpu_capex      = total_gpus * gpu_cost
    cooling_capex  = power_mw * cfg["COOLING_CAPEX_PER_MW"]
    building_capex = power_mw * cfg["BUILDING_CAPEX_PER_MW"]
    total_capex    = gpu_capex + cooling_capex + building_capex

    equity_invested = total_capex * cfg["EQUITY_FRACTION"]
    debt            = total_capex * (1 - cfg["EQUITY_FRACTION"])
    annual_interest = debt * cfg["DEBT_INTEREST_RATE"]
    annual_depr     = total_capex / YRS                 # straight-line

    # --- Stochastic draws (shape: N scenarios) ---
    # Power cost: log-normal around base, std = base * pct
    base_power = cfg["POWER_COST_PER_KWH"]
    power_costs = np.random.lognormal(
        mean=math.log(base_power),
        sigma=cfg["POWER_COST_STD_PCT"],
        size=N,
    )

    # Downtime fraction (clipped 0-0.5)
    downtime = np.clip(
        np.random.normal(cfg["DOWNTIME_MEAN_PCT"], cfg["DOWNTIME_STD_PCT"], size=N),
        0, 0.5,
    )

    # GPU price compression rate per year (clipped 0-0.5)
    compression = np.clip(
        np.random.normal(
            cfg["GPU_PRICE_COMPRESSION_PCT_PER_YR"],
            cfg["GPU_PRICE_COMPRESSION_STD"],
            size=N,
        ),
        0, 0.5,
    )

    # --- Multi-year P&L accumulation ---
    base_price     = cfg["GPU_RENTAL_PRICE_PER_HOUR"]
    utilization    = cfg["UTILIZATION_RATE"]
    pue            = cfg["PUE"]
    hours_per_year = 8_760

    cumulative_net_income = np.zeros(N)
    cumulative_revenue    = np.zeros(N)
    gross_margins         = np.zeros(N)   # year-5 margin

    for yr in range(1, YRS + 1):
        # Revenue: price decays by compression each year
        gpu_price_yr = base_price * ((1 - compression) ** yr)
        revenue_yr = (
            total_gpus * gpu_price_yr * utilization * (1 - downtime) * hours_per_year
        )

        # COGS: power consumed by GPUs (+ PUE overhead)
        gpu_power_kw    = total_gpus * 0.7          # ~700W TDP per H100
        total_power_kw  = gpu_power_kw * pue
        power_cost_yr   = total_power_kw * hours_per_year * power_costs

        gross_profit_yr = revenue_yr - power_cost_yr

        # OpEx: staffing, network, maintenance (~5% revenue)
        opex_yr = revenue_yr * 0.05

        ebitda_yr = gross_profit_yr - opex_yr
        ebit_yr   = ebitda_yr - annual_depr
        ebt_yr    = ebit_yr - annual_interest

        # 21% effective tax (0 if negative – no tax shield modeled simply)
        tax_yr = np.maximum(ebt_yr * 0.21, 0)
        net_income_yr = ebt_yr - tax_yr

        cumulative_net_income += net_income_yr
        cumulative_revenue    += revenue_yr

        if yr == YRS:
            gross_margins = gross_profit_yr / np.maximum(revenue_yr, 1)

    # Terminal value (5x EBITDA multiple) + cumulative net income
    terminal_ebitda = (cumulative_revenue / YRS) * 0.35    # rough 35% EBITDA margin
    terminal_value  = terminal_ebitda * 5

    total_equity_return = cumulative_net_income + terminal_value
    equity_multiple     = total_equity_return / equity_invested

    # --- Build results DataFrame ---
    df = pd.DataFrame({
        "equity_multiple":      equity_multiple,
        "gross_margin_yr5":     gross_margins,
        "cumulative_revenue_B": cumulative_revenue / 1e9,
        "total_capex_B":        total_capex / 1e9,
        "equity_invested_B":    equity_invested / 1e9,
        "power_cost_kwh":       power_costs,
        "downtime_pct":         downtime,
        "compression_pct":      compression,
    })

    # Classify scenarios
    TARGET = cfg["TARGET_MULTIPLE"]
    df["outcome"] = "Middle"
    df.loc[df["equity_multiple"] >= TARGET, "outcome"] = "Hit 15x"
    df.loc[df["equity_multiple"] < 0,       "outcome"] = "Bankruptcy"

    return df, total_capex, equity_invested, total_gpus


def print_summary(df: pd.DataFrame, total_capex: float,
                  equity_invested: float, total_gpus: float, cfg: dict):
    TARGET = cfg["TARGET_MULTIPLE"]
    N = len(df)

    hit_pct  = (df["equity_multiple"] >= TARGET).mean() * 100
    bust_pct = (df["equity_multiple"] < 0).mean() * 100
    p10, p50, p90 = np.percentile(df["equity_multiple"], [10, 50, 90])

    print("\n" + "="*62)
    print("  1GW AI DATA CENTER  -  MONTE CARLO RESULTS")
    print(f"  {N:,} simulations  |  {int(cfg['SIMULATION_YEARS'])}-year horizon")
    print("="*62)
    print(f"\n  CAPEX:           ${total_capex/1e9:.1f}B total")
    print(f"  Equity invested: ${equity_invested/1e9:.1f}B  "
          f"({cfg['EQUITY_FRACTION']*100:.0f}% of CAPEX)")
    print(f"  Total GPUs:      {total_gpus:,.0f}")
    print()
    print(f"  --- Equity Multiple Distribution ---")
    print(f"  P10 (bear):     {p10:.1f}x")
    print(f"  P50 (base):     {p50:.1f}x")
    print(f"  P90 (bull):     {p90:.1f}x")
    print()
    print(f"  --- Scenario Outcomes ---")
    print(f"  Hit {TARGET:.0f}x target:  {hit_pct:.1f}% of scenarios")
    print(f"  Bankruptcy:     {bust_pct:.1f}% of scenarios")
    print(f"  'Middle':       {100 - hit_pct - bust_pct:.1f}% of scenarios")
    print()

    gm = df["gross_margin_yr5"]
    print(f"  --- Year-5 Gross Margin Distribution ---")
    print(f"  P10: {gm.quantile(0.10)*100:.1f}%   "
          f"P50: {gm.quantile(0.50)*100:.1f}%   "
          f"P90: {gm.quantile(0.90)*100:.1f}%")

    # Break-even analysis
    breakeven = df[df["equity_multiple"] >= 1.0]
    print(f"\n  Break-even (>=1x): {len(breakeven)/N*100:.1f}% of scenarios")
    print("="*62)

    # Scenario summary table
    summary = (
        df.groupby("outcome")
          .agg(
              count=("equity_multiple", "count"),
              pct=("equity_multiple", lambda x: f"{len(x)/N*100:.1f}%"),
              median_multiple=("equity_multiple", "median"),
              median_gm_yr5=("gross_margin_yr5", lambda x: f"{x.median()*100:.1f}%"),
          )
          .reset_index()
    )
    print("\n  Outcome Breakdown:")
    print(summary.to_string(index=False))
    print()


def plot_histogram(df: pd.DataFrame, cfg: dict, output_path: Path):
    TARGET = cfg["TARGET_MULTIPLE"]
    multiples = df["equity_multiple"].clip(-5, 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0d1117")

    # --- Left: Equity Multiple Distribution ---
    ax1 = axes[0]
    ax1.set_facecolor("#161b22")

    bins = np.linspace(multiples.min(), multiples.max(), 80)
    hit  = multiples[df["outcome"] == "Hit 15x"]
    mid  = multiples[df["outcome"] == "Middle"]
    bust = multiples[df["outcome"] == "Bankruptcy"]

    ax1.hist(bust, bins=bins, color="#e74c3c", alpha=0.85, label="Bankruptcy (<0x)")
    ax1.hist(mid,  bins=bins, color="#f39c12", alpha=0.85, label="Middle (0x-15x)")
    ax1.hist(hit,  bins=bins, color="#2ecc71", alpha=0.85, label=f"Hit {TARGET:.0f}x target")

    ax1.axvline(TARGET, color="#ffffff", linestyle="--", lw=1.5,
                label=f"{TARGET:.0f}x threshold")
    ax1.axvline(df["equity_multiple"].median(), color="#a29bfe", linestyle=":",
                lw=1.5, label=f"Median: {df['equity_multiple'].median():.1f}x")

    ax1.set_xlabel("Equity Return Multiple (5-yr)", color="#c9d1d9", fontsize=11)
    ax1.set_ylabel("Number of Scenarios", color="#c9d1d9", fontsize=11)
    ax1.set_title("Equity Multiple Distribution\n(10,000 Monte Carlo Scenarios)",
                  color="#ffffff", fontsize=12, pad=10)
    ax1.tick_params(colors="#c9d1d9")
    ax1.spines[:].set_color("#30363d")
    ax1.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9",
               fontsize=9)

    # Stats annotation
    p10, p50, p90 = np.percentile(df["equity_multiple"], [10, 50, 90])
    hit_pct  = (df["equity_multiple"] >= TARGET).mean() * 100
    bust_pct = (df["equity_multiple"] < 0).mean() * 100
    stats_txt = (
        f"P10: {p10:.1f}x\n"
        f"P50: {p50:.1f}x\n"
        f"P90: {p90:.1f}x\n\n"
        f"Hit {TARGET:.0f}x: {hit_pct:.1f}%\n"
        f"Bankruptcy: {bust_pct:.1f}%"
    )
    ax1.text(0.97, 0.97, stats_txt, transform=ax1.transAxes, fontsize=9,
             verticalalignment="top", horizontalalignment="right",
             color="#c9d1d9", bbox=dict(boxstyle="round,pad=0.4",
             facecolor="#21262d", edgecolor="#30363d"))

    # --- Right: Year-5 Gross Margin Distribution ---
    ax2 = axes[1]
    ax2.set_facecolor("#161b22")

    gm_vals = df["gross_margin_yr5"].clip(-0.2, 0.9) * 100
    ax2.hist(gm_vals, bins=60, color="#5dade2", alpha=0.85,
             edgecolor="#0d1117", linewidth=0.3)
    ax2.axvline(gm_vals.median(), color="#f1c40f", linestyle="--", lw=1.5,
                label=f"Median: {gm_vals.median():.1f}%")
    ax2.axvline(0, color="#e74c3c", linestyle=":", lw=1.5, label="Zero margin")

    ax2.set_xlabel("Year-5 Gross Profit Margin (%)", color="#c9d1d9", fontsize=11)
    ax2.set_ylabel("Number of Scenarios", color="#c9d1d9", fontsize=11)
    ax2.set_title("Year-5 Gross Margin Distribution\n(Power cost + compression stress)",
                  color="#ffffff", fontsize=12, pad=10)
    ax2.tick_params(colors="#c9d1d9")
    ax2.spines[:].set_color("#30363d")
    ax2.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9",
               fontsize=9)

    plt.suptitle("1GW AI Data Center  |  5-Year Monte Carlo P&L Stress Test",
                 color="#ffffff", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="#0d1117")
    plt.close()
    print(f"  Chart saved -> {output_path}")


def main():
    print(f"Loading config from: {CONFIG_PATH}")
    cfg = load_config(CONFIG_PATH)

    print(f"Running {int(cfg['MC_SIMULATIONS']):,} simulations...")
    df, total_capex, equity_invested, total_gpus = run_simulations(cfg)

    print_summary(df, total_capex, equity_invested, total_gpus, cfg)
    plot_histogram(df, cfg, OUTPUT_IMG)

    # Save raw results sample
    sample_out = Path(__file__).parent / "mc_results_sample.csv"
    df.head(500).round(4).to_csv(sample_out, index=False)
    print(f"  Sample results (500 rows) -> {sample_out}")


if __name__ == "__main__":
    main()
