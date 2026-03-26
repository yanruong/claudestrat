"""
FCPO / Soybean Oil Spread Backtest — Session 1
================================================
Goals:
  1. Load and clean FCPO (intraday → daily), SBO, USD/MYR, WTI
  2. Normalize FCPO and SBO to USD/MT
  3. Compute the spread (FCPO_usd - SBO_usd)
  4. Plot spread over time, flag mean-reverting vs breakdown windows
  5. Overlay 2022–2024 biodiesel-divergence period

Data files expected in ./data/:
  fcpo.csv   — intraday OHLCV, MYR/MT
  sbo.csv    — daily OHLCV, USc/lb  (CBOT ZL)
  myrusd.csv — daily OHLCV, MYR per 1 USD
  wti.csv    — daily OHLCV, USD/bbl  (optional — needed for regime filter)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)

# ── unit conversion ──────────────────────────────────────────────────────────
LBS_PER_MT = 2_204.623          # 1 metric ton = 2204.623 lbs
USC_TO_USD  = 1 / 100           # USc → USD

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD & CLEAN
# ─────────────────────────────────────────────────────────────────────────────

# --- FCPO (intraday → daily settlement) --------------------------------------
print("Loading FCPO …")
fcpo_raw = pd.read_csv(
    os.path.join(DATA_DIR, "fcpo.csv"),
    parse_dates=["datetime"],
)
fcpo_raw = fcpo_raw.rename(columns={"datetime": "date"})
fcpo_raw = fcpo_raw.sort_values("date")

# Resample to daily: use last bar's close as settlement; rebuild OHLCV
fcpo_daily = (
    fcpo_raw.set_index("date")
    .resample("D")
    .agg(
        open=("open",  "first"),
        high=("high",  "max"),
        low=("low",   "min"),
        close=("close", "last"),
    )
    .dropna(subset=["close"])           # drop calendar gaps with no trades
)
fcpo_daily.index = fcpo_daily.index.normalize()   # strip time component
print(f"  FCPO daily: {fcpo_daily.index[0].date()} → {fcpo_daily.index[-1].date()}  "
      f"({len(fcpo_daily)} bars)")

# --- Soybean Oil (CBOT ZL, USc/lb) ------------------------------------------
print("Loading Soybean Oil …")
sbo = pd.read_csv(
    os.path.join(DATA_DIR, "sbo.csv"),
    parse_dates=["date"],
    index_col="date",
).sort_index()
print(f"  SBO daily:  {sbo.index[0].date()} → {sbo.index[-1].date()}  "
      f"({len(sbo)} bars)")

# --- USD/MYR (MYR per 1 USD) -------------------------------------------------
print("Loading USD/MYR …")
fx = pd.read_csv(
    os.path.join(DATA_DIR, "myrusd.csv"),
    parse_dates=["date"],
    index_col="date",
).sort_index()
print(f"  FX daily:   {fx.index[0].date()} → {fx.index[-1].date()}  "
      f"({len(fx)} bars)")

# --- WTI Crude (optional) ----------------------------------------------------
wti_path = os.path.join(DATA_DIR, "wti.csv")
has_wti  = os.path.exists(wti_path)
if has_wti:
    print("Loading WTI …")
    wti = pd.read_csv(wti_path, parse_dates=["date"], index_col="date").sort_index()
    print(f"  WTI daily:  {wti.index[0].date()} → {wti.index[-1].date()}  "
          f"({len(wti)} bars)")
else:
    print("WTI data not found — regime-filter panel will be skipped.")
    print("  → place wti.csv (date,open,high,low,close) in ./data/ to enable it.")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  CURRENCY NORMALISATION → USD/MT
# ─────────────────────────────────────────────────────────────────────────────

# Align all series on a common date range using inner join
base = fcpo_daily[["close"]].rename(columns={"close": "fcpo_myr"})
base = base.join(sbo[["close"]].rename(columns={"close": "sbo_usc_lb"}), how="inner")
base = base.join(fx[["close"]].rename(columns={"close": "myr_per_usd"}), how="inner")
if has_wti:
    base = base.join(wti[["close"]].rename(columns={"close": "wti"}), how="left")

print(f"\nCommon date range: {base.index[0].date()} → {base.index[-1].date()}  "
      f"({len(base)} days)")

# Forward-fill FX for any missing days (weekends already dropped by inner join,
# but public holidays may leave gaps)
base["myr_per_usd"] = base["myr_per_usd"].ffill()

# Convert FCPO: MYR/MT  →  USD/MT
base["fcpo_usd"] = base["fcpo_myr"] / base["myr_per_usd"]

# Convert SBO: USc/lb  →  USD/MT
base["sbo_usd"] = base["sbo_usc_lb"] * USC_TO_USD * LBS_PER_MT

print("\nSample normalised prices (USD/MT):")
print(base[["fcpo_usd", "sbo_usd"]].tail(5).round(2))

# ─────────────────────────────────────────────────────────────────────────────
# 3.  SPREAD CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

# Absolute spread: FCPO − SBO  (positive → FCPO premium over SBO)
base["spread"]     = base["fcpo_usd"] - base["sbo_usd"]

# Rolling z-score (252-day window, ~1 trading year)
ZSCORE_WINDOW = 252
base["spread_mean"] = base["spread"].rolling(ZSCORE_WINDOW, min_periods=60).mean()
base["spread_std"]  = base["spread"].rolling(ZSCORE_WINDOW, min_periods=60).std()
base["spread_z"]    = (base["spread"] - base["spread_mean"]) / base["spread_std"]

# Ratio (log scale useful for long-run view)
base["ratio"]  = np.log(base["fcpo_usd"] / base["sbo_usd"])

# WTI–SBO 30-day rolling correlation (regime filter)
if has_wti:
    sbo_ret = base["sbo_usd"].pct_change()
    wti_ret = base["wti"].pct_change()
    base["corr_sbo_wti_30d"] = sbo_ret.rolling(30).corr(wti_ret)

print("\nSpread summary (USD/MT):")
print(base["spread"].describe().round(2))

# ─────────────────────────────────────────────────────────────────────────────
# 4 & 5.  PLOTS
# ─────────────────────────────────────────────────────────────────────────────

BIODIESEL_START = pd.Timestamp("2022-01-01")
BIODIESEL_END   = pd.Timestamp("2024-12-31")

# Count panels
n_panels = 4 if has_wti else 3
fig = plt.figure(figsize=(16, 4 * n_panels))
gs  = GridSpec(n_panels, 1, figure=fig, hspace=0.45)

def shade_biodiesel(ax):
    """Shade the 2022-2024 biodiesel-divergence window."""
    ax.axvspan(BIODIESEL_START, BIODIESEL_END,
               color="tomato", alpha=0.12, label="2022–2024 biodiesel regime")

def add_band(ax, series, lo, hi, color, alpha=0.15):
    ax.axhline(lo, color=color, linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axhline(hi, color=color, linewidth=0.8, linestyle="--", alpha=0.7)
    ax.fill_between(series.index, lo, hi, color=color, alpha=alpha)

# ── Panel 1: FCPO and SBO USD/MT prices ──────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.plot(base.index, base["fcpo_usd"], color="#1f77b4", linewidth=1.0, label="FCPO (USD/MT)")
ax1.plot(base.index, base["sbo_usd"],  color="#ff7f0e", linewidth=1.0, label="SBO (USD/MT)")
shade_biodiesel(ax1)
ax1.set_title("FCPO vs Soybean Oil — Normalised to USD/MT", fontsize=11, fontweight="bold")
ax1.set_ylabel("USD / MT")
ax1.legend(loc="upper left", fontsize=8)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.grid(alpha=0.3)

# ── Panel 2: Absolute spread ──────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.plot(base.index, base["spread"], color="#2ca02c", linewidth=0.9, label="Spread (FCPO − SBO)")
ax2.plot(base.index, base["spread_mean"], color="black", linewidth=1.2,
         linestyle="--", alpha=0.6, label=f"{ZSCORE_WINDOW}d rolling mean")
ax2.fill_between(base.index,
                 base["spread_mean"] - base["spread_std"],
                 base["spread_mean"] + base["spread_std"],
                 color="#2ca02c", alpha=0.12, label="±1σ band")
ax2.fill_between(base.index,
                 base["spread_mean"] - 2 * base["spread_std"],
                 base["spread_mean"] + 2 * base["spread_std"],
                 color="#2ca02c", alpha=0.06, label="±2σ band")
shade_biodiesel(ax2)
ax2.set_title("Absolute Spread  (FCPO − SBO, USD/MT)", fontsize=11, fontweight="bold")
ax2.set_ylabel("USD / MT")
ax2.legend(loc="upper left", fontsize=8, ncol=2)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.grid(alpha=0.3)

# ── Panel 3: Rolling z-score ──────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2])
# Colour bars by sign for easy reading
pos_mask = base["spread_z"] >= 0
ax3.bar(base.index[pos_mask],  base["spread_z"][pos_mask],  color="#1f77b4", width=1, alpha=0.7)
ax3.bar(base.index[~pos_mask], base["spread_z"][~pos_mask], color="#d62728", width=1, alpha=0.7)
ax3.axhline( 2, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
ax3.axhline(-2, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
ax3.axhline( 1, color="grey",  linewidth=0.6, linestyle=":",  alpha=0.5)
ax3.axhline(-1, color="grey",  linewidth=0.6, linestyle=":",  alpha=0.5)
ax3.axhline( 0, color="black", linewidth=0.5, alpha=0.4)
shade_biodiesel(ax3)
ax3.set_title(f"Spread Z-Score  ({ZSCORE_WINDOW}d rolling window)", fontsize=11, fontweight="bold")
ax3.set_ylabel("Z-Score")
ax3.set_ylim(-5, 5)
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax3.grid(alpha=0.3)
# Annotate signal thresholds
ax3.text(base.index[10],  2.1, "+2σ entry zone", fontsize=7, color="black", alpha=0.7)
ax3.text(base.index[10], -2.4, "−2σ entry zone", fontsize=7, color="black", alpha=0.7)

# ── Panel 4 (optional): WTI–SBO 30d correlation ──────────────────────────────
if has_wti:
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(base.index, base["corr_sbo_wti_30d"], color="#9467bd", linewidth=0.9)
    ax4.axhline(0.5,  color="tomato", linewidth=1.0, linestyle="--",
                label="Regime threshold (0.5)")
    ax4.fill_between(base.index, 0.5, base["corr_sbo_wti_30d"],
                     where=base["corr_sbo_wti_30d"] >= 0.5,
                     color="tomato", alpha=0.25, label="Biodiesel regime active")
    shade_biodiesel(ax4)
    ax4.set_title("30-Day Rolling Correlation: SBO vs WTI  (Regime Filter)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Correlation")
    ax4.set_ylim(-1, 1)
    ax4.legend(loc="upper left", fontsize=8)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax4.grid(alpha=0.3)

# ── Shared legend patch ───────────────────────────────────────────────────────
bio_patch = mpatches.Patch(color="tomato", alpha=0.3, label="2022–2024 biodiesel regime")
fig.legend(handles=[bio_patch], loc="upper right", bbox_to_anchor=(0.99, 0.99), fontsize=9)

fig.suptitle("FCPO / Soybean Oil Spread Analysis — Session 1", fontsize=13, fontweight="bold", y=1.01)

out_path = os.path.join(OUT_DIR, "spread_session1.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved → {out_path}")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY STATS: mean-reversion characterisation
# ─────────────────────────────────────────────────────────────────────────────

pre_bio  = base.loc[base.index < BIODIESEL_START, "spread"]
bio      = base.loc[(base.index >= BIODIESEL_START) & (base.index <= BIODIESEL_END), "spread"]
post_bio = base.loc[base.index > BIODIESEL_END, "spread"]

print("\n── Spread regime summary (USD/MT) ─────────────────────────────────")
for label, s in [("Pre-2022  (mean-rev expected)", pre_bio),
                 ("2022–2024 (biodiesel divergence)", bio),
                 ("Post-2024", post_bio)]:
    if len(s) == 0:
        continue
    print(f"  {label:<35}  mean={s.mean():+.0f}  std={s.std():.0f}  "
          f"min={s.min():+.0f}  max={s.max():+.0f}  n={len(s)}")

# Z-score exceedance counts (|z| > 2) — how often does spread hit extremes?
z = base["spread_z"].dropna()
print(f"\n  Z-score > +2σ:  {(z >  2).sum()} days  ({(z >  2).mean()*100:.1f}%)")
print(f"  Z-score < −2σ:  {(z < -2).sum()} days  ({(z < -2).mean()*100:.1f}%)")

print("\nSession 1 complete.")
