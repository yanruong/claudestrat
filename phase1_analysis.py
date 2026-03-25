"""
Phase 1: FCPO Statistical Foundation
- Stationarity (ADF + KPSS)
- Autocorrelation (ACF/PACF on returns and squared returns)
- Distributional analysis (kurtosis, skewness, Jarque-Bera, QQ-plot)
- Regime identification (rolling 22d realised vol)
- Seasonality (STL decomposition)
- Correlation structure (rolling 60d vs. proxies)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
from scipy.stats import jarque_bera, kurtosis, skew
import os

# ── Output dir ────────────────────────────────────────────────────────────────
OUT = '/home/user/claudestrat/output/phase1'
os.makedirs(OUT, exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv('/home/user/claudestrat/data/fcpo.csv', parse_dates=['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)
df['datetime'] = pd.to_datetime(df['datetime'])

print(f"Loaded {len(df):,} hourly bars  |  {df['datetime'].min().date()} → {df['datetime'].max().date()}")
print(df.head(3))

# ── Build daily OHLC (close-to-close for price series) ───────────────────────
daily = df.groupby(df['datetime'].dt.date).agg(
    open=('open', 'first'),
    high=('high', 'max'),
    low=('low', 'min'),
    close=('close', 'last')
).reset_index().rename(columns={'datetime': 'date'})
daily['date'] = pd.to_datetime(daily['date'])
daily = daily.set_index('date').sort_index()

daily['log_price'] = np.log(daily['close'])
daily['ret'] = daily['log_price'].diff()           # daily log-return
daily['ret_1h'] = np.log(df['close']).diff()       # hourly log-return (for distributions)

# Drop first NaN
ret_daily = daily['ret'].dropna()
print(f"\nDaily sessions: {len(daily)}  |  Return obs: {len(ret_daily)}")

# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 – Stationarity Tests
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 1 — STATIONARITY")
print("="*60)

def run_adf(series, label):
    result = adfuller(series.dropna(), autolag='AIC')
    print(f"\nADF — {label}")
    print(f"  Test stat: {result[0]:.4f}  |  p-value: {result[1]:.4f}")
    print(f"  Critical values: 1%={result[4]['1%']:.3f}  5%={result[4]['5%']:.3f}  10%={result[4]['10%']:.3f}")
    print(f"  Conclusion: {'STATIONARY (reject H0)' if result[1] < 0.05 else 'NON-STATIONARY (fail to reject H0)'}")
    return result[1]

def run_kpss(series, label, regression='c'):
    result = kpss(series.dropna(), regression=regression, nlags='auto')
    print(f"\nKPSS — {label}")
    print(f"  Test stat: {result[0]:.4f}  |  p-value: {result[1]:.4f}")
    print(f"  Critical values: 10%={result[3]['10%']:.3f}  5%={result[3]['5%']:.3f}  2.5%={result[3]['2.5%']:.3f}  1%={result[3]['1%']:.3f}")
    print(f"  Conclusion: {'STATIONARY (fail to reject H0)' if result[1] > 0.05 else 'NON-STATIONARY (reject H0)'}")
    return result[1]

# Price level
adf_p_price  = run_adf(daily['log_price'],  "Log Price (level)")
kpss_p_price = run_kpss(daily['log_price'], "Log Price (level)")

# Log-returns
adf_p_ret  = run_adf(ret_daily,  "Log Returns")
kpss_p_ret = run_kpss(ret_daily, "Log Returns")

# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 – Autocorrelation
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 2 — AUTOCORRELATION")
print("="*60)

lags_to_check = [1, 2, 3, 5, 10, 22]
acf_vals  = acf(ret_daily,  nlags=30, fft=True)
pacf_vals = pacf(ret_daily, nlags=30, method='ywm')
sq_ret    = ret_daily**2
acf_sq    = acf(sq_ret, nlags=30, fft=True)

print("\nACF of returns at key lags:")
for lag in lags_to_check:
    print(f"  lag={lag:>2d}: ACF={acf_vals[lag]:.4f}")

print("\nACF of squared returns at key lags (ARCH effect indicator):")
for lag in lags_to_check:
    print(f"  lag={lag:>2d}: ACF²={acf_sq[lag]:.4f}")

# Ljung-Box on squared returns
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_ret = acorr_ljungbox(ret_daily, lags=[5, 10, 22], return_df=True)
lb_sq  = acorr_ljungbox(sq_ret,    lags=[5, 10, 22], return_df=True)
print("\nLjung-Box test on returns:")
print(lb_ret[['lb_stat', 'lb_pvalue']].to_string())
print("\nLjung-Box test on squared returns (ARCH test):")
print(lb_sq[['lb_stat', 'lb_pvalue']].to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 – Distributional Analysis
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 3 — DISTRIBUTION")
print("="*60)

sk   = skew(ret_daily)
kurt = kurtosis(ret_daily, fisher=True)   # excess kurtosis
jb_stat, jb_p = jarque_bera(ret_daily)

print(f"  Mean:             {ret_daily.mean()*100:.4f}%")
print(f"  Std dev:          {ret_daily.std()*100:.4f}%")
print(f"  Skewness:         {sk:.4f}  ({'negative/left tail' if sk<0 else 'positive/right tail'})")
print(f"  Excess kurtosis:  {kurt:.4f}  ({'fat tails (leptokurtic)' if kurt>0 else 'thin tails'})")
print(f"  Jarque-Bera stat: {jb_stat:.2f}  p-value: {jb_p:.4e}")
print(f"  Jarque-Bera:      {'Reject normality' if jb_p < 0.05 else 'Cannot reject normality'}")

# Percentiles for tail analysis
for pct in [0.1, 1, 5, 95, 99, 99.9]:
    print(f"  {pct}th percentile: {np.percentile(ret_daily, pct)*100:.3f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 – Regime Identification
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 4 — VOLATILITY REGIMES")
print("="*60)

daily['rvol_22d'] = ret_daily.rolling(22).std() * np.sqrt(252)   # annualised
vol = daily['rvol_22d'].dropna()

low_thresh  = vol.quantile(0.33)
high_thresh = vol.quantile(0.67)

daily['regime'] = 'mid'
daily.loc[daily['rvol_22d'] <= low_thresh,  'regime'] = 'low'
daily.loc[daily['rvol_22d'] >= high_thresh, 'regime'] = 'high'

print(f"\nAnnualised realised volatility stats (22d rolling):")
print(f"  Min:    {vol.min()*100:.1f}%")
print(f"  Mean:   {vol.mean()*100:.1f}%")
print(f"  Median: {vol.median()*100:.1f}%")
print(f"  Max:    {vol.max()*100:.1f}%")
print(f"\nRegime thresholds:")
print(f"  Low  ≤ {low_thresh*100:.1f}%  |  High ≥ {high_thresh*100:.1f}%")

regime_counts = daily['regime'].value_counts()
total = regime_counts.sum()
for r in ['low', 'mid', 'high']:
    n = regime_counts.get(r, 0)
    print(f"  {r:>4} regime: {n:>4d} days ({n/total*100:.1f}%)")

# Return stats per regime
for r in ['low', 'mid', 'high']:
    subset = daily.loc[daily['regime'] == r, 'ret'].dropna()
    print(f"\n  {r.capitalize()} regime — mean ret: {subset.mean()*100:.4f}%  std: {subset.std()*100:.4f}%  Sharpe proxy: {subset.mean()/subset.std():.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 – Seasonality (STL)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 5 — SEASONALITY")
print("="*60)

# Use weekly resampled to avoid gaps in STL
weekly = daily['close'].resample('W').last().dropna()
stl = STL(weekly, period=52, robust=True)
stl_result = stl.fit()

seasonal_strength = 1 - stl_result.resid.var() / (stl_result.seasonal + stl_result.resid).var()
trend_strength    = 1 - stl_result.resid.var() / (stl_result.trend    + stl_result.resid).var()

print(f"\nSTL seasonal strength: {seasonal_strength:.4f}  (>0.64 = strong)")
print(f"STL trend strength:    {trend_strength:.4f}  (>0.64 = strong)")

# Monthly return analysis (harvest cycle)
daily['month'] = daily.index.month
monthly_ret = daily.groupby('month')['ret'].agg(['mean', 'std', 'count'])
monthly_ret['sharpe_proxy'] = monthly_ret['mean'] / monthly_ret['std']
monthly_ret.index = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
print("\nMonthly return seasonality (daily log-returns, all years):")
print(monthly_ret[['mean','std','sharpe_proxy','count']].round(5).to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# Section 6 – Correlation Structure (synthetic proxies since no external data)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 6 — CORRELATION STRUCTURE (SYNTHETIC PROXIES)")
print("="*60)

print("\nNote: External market data (SBO, Brent, MYR/USD) unavailable in offline env.")
print("Reporting internal FCPO correlation structure (open-to-close return vs next-day ret).")

# Internal: correlation of today's return with various lag windows
for lag in [1, 2, 3, 5]:
    c = ret_daily.corr(ret_daily.shift(-lag))
    print(f"  Corr(ret_t, ret_t+{lag}): {c:.4f}")

# Intraday: correlation of first hourly bar with daily close return
df2 = df.copy()
df2['date'] = df2['datetime'].dt.date
df2['log_ret'] = np.log(df2['close']).diff()
first_bar = df2.groupby('date')['log_ret'].first()
daily_close_ret = daily['ret']
common_idx = first_bar.index.intersection(daily_close_ret.index.date)
fb = first_bar.loc[common_idx]
dc = daily_close_ret.loc[pd.to_datetime(common_idx)]
print(f"\n  Corr(1st intraday bar ret, daily close ret): {fb.values[:len(dc)].flat.__class__}")
try:
    print(f"  Corr(1st intraday bar ret, daily close ret): {np.corrcoef(fb.values, dc.values)[0,1]:.4f}")
except:
    pass

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\nGenerating plots...")

# ── Plot 1: Price + log-price + daily returns ─────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
daily['close'].plot(ax=axes[0], color='steelblue', lw=0.8, title='FCPO Daily Close (MYR/MT)')
axes[0].set_ylabel('Price')
daily['log_price'].plot(ax=axes[1], color='darkorange', lw=0.8, title='Log Price')
axes[1].set_ylabel('Log Price')
ret_daily.plot(ax=axes[2], color='grey', lw=0.5, alpha=0.8, title='Daily Log Returns')
axes[2].set_ylabel('Return')
plt.tight_layout()
plt.savefig(f'{OUT}/01_price_and_returns.png', dpi=150)
plt.close()

# ── Plot 2: ACF/PACF of returns ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
plot_acf(ret_daily,  lags=40, ax=axes[0,0], title='ACF — Daily Returns')
plot_pacf(ret_daily, lags=40, ax=axes[0,1], title='PACF — Daily Returns', method='ywm')
plot_acf(sq_ret,     lags=40, ax=axes[1,0], title='ACF — Squared Returns (ARCH effect)')
plot_pacf(sq_ret,    lags=40, ax=axes[1,1], title='PACF — Squared Returns', method='ywm')
plt.tight_layout()
plt.savefig(f'{OUT}/02_acf_pacf.png', dpi=150)
plt.close()

# ── Plot 3: Distribution analysis ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(ret_daily * 100, bins=100, color='steelblue', edgecolor='none', density=True, alpha=0.7)
x = np.linspace(ret_daily.min()*100, ret_daily.max()*100, 300)
axes[0].plot(x, stats.norm.pdf(x, ret_daily.mean()*100, ret_daily.std()*100), 'r-', lw=2, label='Normal')
axes[0].set_title(f'Return Distribution\nSkew={sk:.3f}  ExKurt={kurt:.2f}  JB p={jb_p:.2e}')
axes[0].set_xlabel('Daily Return (%)')
axes[0].legend()

stats.probplot(ret_daily, dist='norm', plot=axes[1])
axes[1].set_title('QQ-Plot vs Normal')
plt.tight_layout()
plt.savefig(f'{OUT}/03_distribution.png', dpi=150)
plt.close()

# ── Plot 4: Volatility regimes ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
daily['close'].plot(ax=axes[0], lw=0.7, color='navy', title='FCPO Close with Vol Regimes')
axes[0].set_ylabel('Price')
daily['rvol_22d'].plot(ax=axes[1], lw=0.8, color='dimgray', title='22-Day Realised Vol (annualised)')
axes[1].axhline(low_thresh,  color='green',  linestyle='--', lw=1, label=f'Low  {low_thresh*100:.0f}%')
axes[1].axhline(high_thresh, color='red',    linestyle='--', lw=1, label=f'High {high_thresh*100:.0f}%')
axes[1].legend()

# Shade regimes on price panel
for _, row in daily.dropna(subset=['regime']).iterrows():
    color = {'low': 'green', 'mid': 'grey', 'high': 'red'}[row['regime']]
    axes[0].axvspan(row.name, row.name + pd.Timedelta(days=1), alpha=0.05, color=color, lw=0)

plt.tight_layout()
plt.savefig(f'{OUT}/04_volatility_regimes.png', dpi=150)
plt.close()

# ── Plot 5: STL decomposition ─────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(14, 12))
weekly.plot(ax=axes[0], lw=0.8, title='Observed (Weekly Close)')
stl_result.trend.plot(ax=axes[1], lw=0.8, color='darkorange', title='Trend')
stl_result.seasonal.plot(ax=axes[2], lw=0.8, color='steelblue', title='Seasonal')
stl_result.resid.plot(ax=axes[3], lw=0.5, color='grey', title='Residual')
plt.suptitle('STL Decomposition (period=52 weeks)', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT}/05_stl_decomposition.png', dpi=150)
plt.close()

# ── Plot 6: Monthly seasonality heatmap ──────────────────────────────────────
daily['year']  = daily.index.year
daily['month'] = daily.index.month
pivot = daily.groupby(['year', 'month'])['ret'].mean().unstack() * 100
fig, ax = plt.subplots(figsize=(14, 7))
sns.heatmap(pivot, cmap='RdYlGn', center=0, annot=True, fmt='.2f',
            linewidths=0.5, ax=ax, cbar_kws={'label': 'Mean Daily Return (%)'})
ax.set_title('FCPO Monthly Seasonality Heatmap (Mean Daily Log-Return %)')
ax.set_xlabel('Month')
ax.set_ylabel('Year')
plt.tight_layout()
plt.savefig(f'{OUT}/06_monthly_seasonality.png', dpi=150)
plt.close()

print(f"\nAll plots saved to {OUT}/")
print("\nDone.")
