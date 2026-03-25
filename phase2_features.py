"""
Phase 2: Feature Engineering
- Lagged FCPO returns (1d, 3d, 5d, 10d, 22d)
- Intraday momentum / open-gap features
- Realised vol ratios (short/long) — regime proxy
- Rolling return z-score
- Calendar features (month, day-of-week)
- CPO spread proxies (internal: high-low range, close-to-open gap)
- VIF < 5 check
- Strict look-ahead prevention
- Target: binary direction (daily return > transaction-cost threshold)
- Output: features.parquet ready for Phase 3
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os

OUT = '/home/user/claudestrat/output/phase2'
os.makedirs(OUT, exist_ok=True)

# ── 1. Load raw hourly data ───────────────────────────────────────────────────
df = pd.read_csv('/home/user/claudestrat/data/fcpo.csv', parse_dates=['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

# ── 2. Build daily bars ───────────────────────────────────────────────────────
daily = df.groupby(df['datetime'].dt.date).agg(
    open=('open',   'first'),
    high=('high',   'max'),
    low=('low',     'min'),
    close=('close', 'last'),
    n_bars=('close', 'count')
).reset_index().rename(columns={'datetime': 'date'})
daily['date'] = pd.to_datetime(daily['date'])
daily = daily.set_index('date').sort_index()

print(f"Daily bars: {len(daily)}  |  {daily.index[0].date()} → {daily.index[-1].date()}")

# ── 3. Core return series (all look-ahead free — only use t-1 close) ──────────
daily['ret_1d']  = np.log(daily['close']).diff(1)
daily['ret_3d']  = np.log(daily['close']).diff(3)
daily['ret_5d']  = np.log(daily['close']).diff(5)
daily['ret_10d'] = np.log(daily['close']).diff(10)
daily['ret_22d'] = np.log(daily['close']).diff(22)

# Overnight gap: today's open vs yesterday's close
daily['gap']     = np.log(daily['open']) - np.log(daily['close'].shift(1))

# Intraday range (normalised by close) — proxy for intraday liquidity/activity
daily['range_pct'] = (daily['high'] - daily['low']) / daily['close'].shift(1)

# ── 4. Volatility features ────────────────────────────────────────────────────
daily['rvol_5d']  = daily['ret_1d'].rolling(5).std()  * np.sqrt(252)
daily['rvol_22d'] = daily['ret_1d'].rolling(22).std() * np.sqrt(252)
daily['rvol_66d'] = daily['ret_1d'].rolling(66).std() * np.sqrt(252)

# Vol ratio: short-term vs long-term (>1 = rising vol environment)
daily['vol_ratio_5_22']  = daily['rvol_5d']  / daily['rvol_22d']
daily['vol_ratio_22_66'] = daily['rvol_22d'] / daily['rvol_66d']

# ── 5. Momentum z-scores (return / rolling vol) ───────────────────────────────
daily['mom_z_5']  = daily['ret_5d']  / (daily['rvol_22d'] / np.sqrt(252) * np.sqrt(5)  + 1e-9)
daily['mom_z_22'] = daily['ret_22d'] / (daily['rvol_22d'] / np.sqrt(252) * np.sqrt(22) + 1e-9)

# Rolling return percentile (60d lookback) — position within recent distribution
daily['ret_pct_60d'] = daily['ret_1d'].rolling(60).apply(
    lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
)

# ── 6. Price-level features (relative to rolling mean) ───────────────────────
daily['price_z_22']  = (daily['close'] - daily['close'].rolling(22).mean())  / (daily['close'].rolling(22).std()  + 1e-9)
daily['price_z_66']  = (daily['close'] - daily['close'].rolling(66).mean())  / (daily['close'].rolling(66).std()  + 1e-9)

# High-low ratio (internal spread proxy, analogous to CPO premium)
daily['hl_ratio'] = np.log(daily['high'] / daily['low'])

# ── 7. Calendar features ──────────────────────────────────────────────────────
daily['month']  = daily.index.month
daily['dow']    = daily.index.dayofweek   # 0=Mon … 4=Fri
daily['qtr']    = daily.index.quarter

# Harvest cycle binary: Oct–Feb = peak season (1), else 0
daily['harvest'] = daily['month'].isin([10, 11, 12, 1, 2]).astype(int)

# ── 8. Lagged feature matrix (shift by 1 so features are known at t-1 close) ─
# All raw features are calculated using data up to and including day t.
# The ML model predicts t+1 direction, so we shift features by 1.
feature_cols_raw = [
    'ret_1d', 'ret_3d', 'ret_5d', 'ret_10d', 'ret_22d',
    'gap', 'range_pct',
    'rvol_5d', 'rvol_22d', 'rvol_66d',
    'vol_ratio_5_22', 'vol_ratio_22_66',
    'mom_z_5', 'mom_z_22',
    'ret_pct_60d',
    'price_z_22', 'price_z_66',
    'hl_ratio',
    'month', 'dow', 'qtr', 'harvest',
]

feat = pd.DataFrame(index=daily.index)
for col in feature_cols_raw:
    feat[col] = daily[col].shift(1)   # ← look-ahead prevention: use t-1 value

# ── 9. Target variable ────────────────────────────────────────────────────────
# Transaction cost threshold: RM 35 round-trip on a 25MT lot at ~RM 4000 close
# = 35 / (4000 * 25) ≈ 0.035% in price terms → use 0.04% as conservative threshold
TRANSACTION_COST_PCT = np.log(1 + 35 / (daily['close'].mean() * 25))
print(f"\nTransaction cost threshold: {TRANSACTION_COST_PCT*100:.4f}% log-return")

feat['fwd_ret_1d']   = daily['ret_1d']      # raw forward return (for backtest P&L)
feat['target']       = (daily['ret_1d'] > TRANSACTION_COST_PCT).astype(int)
feat['target_long']  = feat['target']
feat['target_short'] = (daily['ret_1d'] < -TRANSACTION_COST_PCT).astype(int)

# ── 10. Drop rows where any feature is NaN (warmup period) ────────────────────
n_before = len(feat)
feat = feat.dropna()
n_after = len(feat)
print(f"\nRows before NaN drop: {n_before}  |  After: {n_after}  |  Warmup dropped: {n_before - n_after}")

# ── 11. VIF check ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("VIF CHECK (drop features with VIF ≥ 5)")
print("="*60)

numeric_feat_cols = [c for c in feature_cols_raw if c not in ['month', 'dow', 'qtr', 'harvest']]
X_vif = feat[numeric_feat_cols].copy()

# Replace any remaining inf
X_vif = X_vif.replace([np.inf, -np.inf], np.nan).dropna()

vif_data = pd.DataFrame()
vif_data['feature'] = X_vif.columns
vif_data['VIF']     = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
vif_data = vif_data.sort_values('VIF', ascending=False)
print(vif_data.to_string(index=False))

print(f"\nIterative VIF removal (drop highest VIF feature one at a time until all < 5):")

def iterative_vif(cols, data, threshold=5.0):
    remaining = list(cols)
    dropped = []
    iteration = 0
    while True:
        X = data[remaining].replace([np.inf, -np.inf], np.nan).dropna()
        vif = pd.Series(
            [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
            index=remaining
        )
        max_vif = vif.max()
        if max_vif < threshold:
            break
        worst = vif.idxmax()
        print(f"  Iter {iteration+1}: drop '{worst}' (VIF={max_vif:.2f})")
        remaining.remove(worst)
        dropped.append(worst)
        iteration += 1
    print(f"\nDropped {len(dropped)} features: {dropped}")
    print("Remaining features and VIF:")
    X_final = data[remaining].replace([np.inf, -np.inf], np.nan).dropna()
    vif_final = pd.DataFrame({'feature': remaining,
                               'VIF': [variance_inflation_factor(X_final.values, i) for i in range(len(remaining))]})
    vif_final = vif_final.sort_values('VIF', ascending=False)
    print(vif_final.to_string(index=False))
    return remaining, vif_final

final_numeric_cols, vif_final = iterative_vif(numeric_feat_cols, feat)

# Final feature set = clean numeric + calendar
calendar_cols = ['month', 'dow', 'qtr', 'harvest']
final_feature_cols = final_numeric_cols + calendar_cols
print(f"\nFinal feature count: {len(final_feature_cols)}")
print("Features:", final_feature_cols)

# ── 12. Class balance ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TARGET CLASS BALANCE")
print("="*60)
vc = feat['target'].value_counts()
print(f"  Long  (1): {vc.get(1,0):>5d} ({vc.get(1,0)/len(feat)*100:.1f}%)")
print(f"  Short (0): {vc.get(0,0):>5d} ({vc.get(0,0)/len(feat)*100:.1f}%)")

# ── 13. Save feature set ──────────────────────────────────────────────────────
cols_to_save = final_feature_cols + ['fwd_ret_1d', 'target']
feat_clean = feat[cols_to_save].replace([np.inf, -np.inf], np.nan).dropna()

feat_clean.to_csv('/home/user/claudestrat/data/features.csv')
print(f"\nSaved {len(feat_clean)} rows × {len(final_feature_cols)} features to data/features.csv")

# Also save metadata
import json
meta = {
    'feature_cols': final_feature_cols,
    'target_col': 'target',
    'fwd_ret_col': 'fwd_ret_1d',
    'transaction_cost_pct': float(TRANSACTION_COST_PCT),
    'n_rows': len(feat_clean),
    'date_range': [str(feat_clean.index[0].date()), str(feat_clean.index[-1].date())],
}
with open('/home/user/claudestrat/data/feature_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)
print("Saved metadata to data/feature_meta.json")

# ── 14. Plots ─────────────────────────────────────────────────────────────────
print("\nGenerating plots...")

# Feature correlation heatmap
fig, ax = plt.subplots(figsize=(14, 12))
corr = feat_clean[final_numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            annot=True, fmt='.2f', linewidths=0.3, ax=ax,
            cbar_kws={'label': 'Pearson r'})
ax.set_title('Feature Correlation Matrix (final numeric features)')
plt.tight_layout()
plt.savefig(f'{OUT}/01_feature_correlation.png', dpi=150)
plt.close()

# VIF bar chart (final retained features)
fig, ax = plt.subplots(figsize=(10, 6))
vif_plot = vif_final.set_index('feature')['VIF'].sort_values()
colors = ['steelblue'] * len(vif_plot)
vif_plot.plot(kind='barh', color=colors, ax=ax)
ax.axvline(5, color='red', linestyle='--', lw=1.5, label='VIF=5 threshold')
ax.set_title('VIF — Retained Features (all < 5)')
ax.set_xlabel('VIF')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT}/02_vif.png', dpi=150)
plt.close()

# Feature distributions
n_feats = len(final_numeric_cols)
ncols = 4
nrows = (n_feats + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
axes = axes.flatten()
for i, col in enumerate(final_numeric_cols):
    axes[i].hist(feat_clean[col].dropna(), bins=50, color='steelblue', edgecolor='none', density=True)
    axes[i].set_title(col, fontsize=9)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle('Feature Distributions (post-NaN drop)', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT}/03_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()

# Feature vs forward return (box plots for key features)
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
key_feats = [f for f in ['ret_1d', 'mom_z_5', 'ret_22d', 'vol_ratio_22_66', 'price_z_66', 'hl_ratio']
             if f in feat_clean.columns]
for i, col in enumerate(key_feats):
    feat_clean['_q'] = pd.qcut(feat_clean[col], q=5, labels=['Q1','Q2','Q3','Q4','Q5'], duplicates='drop')
    feat_clean.groupby('_q')['fwd_ret_1d'].mean().plot(kind='bar', ax=axes[i], color='steelblue', edgecolor='none')
    axes[i].axhline(0, color='black', lw=0.8)
    axes[i].set_title(f'Fwd Return by {col} quintile')
    axes[i].set_xlabel('')
    axes[i].tick_params(axis='x', rotation=0)
feat_clean.drop(columns=['_q'], inplace=True)
plt.suptitle('Mean Forward Return by Feature Quintile')
plt.tight_layout()
plt.savefig(f'{OUT}/04_feature_vs_fwdret.png', dpi=150)
plt.close()

print(f"\nAll plots saved to {OUT}/")
print("\nPhase 2 complete. Ready for Phase 3.")
