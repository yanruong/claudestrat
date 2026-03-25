"""
Phase 2b (Combined): Rebuild features adding MYR/USD + SBO (CBOT Soybean Oil) cross-asset signals.

New features:
  MYR/USD:
    myr_ret_1d      : MYR/USD daily log-return (higher = weaker MYR)
    myr_ret_5d      : 5-day MYR/USD log-return
    myr_z_22        : MYR/USD z-score vs 22d mean (currency deviation)
    myr_vol_ratio   : MYR/USD 5d vol / 22d vol (currency stress)
    myr_fcpo_corr   : rolling 22d correlation between MYR ret and FCPO ret

  SBO (ZL USD cents/lb):
    sbo_ret_1d      : SBO daily log-return
    sbo_ret_5d      : 5-day SBO log-return
    sbo_z_22        : SBO z-score vs 22d mean
    sbo_vol_ratio   : SBO 5d vol / 22d vol
    cpo_sbo_spread  : log(FCPO_MYR / (SBO_USD * MYR_per_USD * conversion)) = CPO premium proxy
    cpo_sbo_spread_z22: CPO-SBO spread z-score vs 22d mean

All features shifted by 1 day (look-ahead free).
VIF < 5 enforced via iterative removal.
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
import json, os

OUT = '/home/user/claudestrat/output/phase2b'
os.makedirs(OUT, exist_ok=True)

# ── 1. Load FCPO daily bars ───────────────────────────────────────────────────
df_h = pd.read_csv('/home/user/claudestrat/data/fcpo.csv', parse_dates=['datetime'])
df_h = df_h.sort_values('datetime').reset_index(drop=True)

daily = df_h.groupby(df_h['datetime'].dt.date).agg(
    open=('open',   'first'),
    high=('high',   'max'),
    low=('low',     'min'),
    close=('close', 'last'),
).reset_index().rename(columns={'datetime': 'date'})
daily['date'] = pd.to_datetime(daily['date'])
daily = daily.set_index('date').sort_index()
print(f"FCPO daily bars: {len(daily)}  |  {daily.index[0].date()} → {daily.index[-1].date()}")

# ── 2. Load MYR/USD ───────────────────────────────────────────────────────────
myr = pd.read_csv('/home/user/claudestrat/data/myrusd.csv', parse_dates=['date'])
myr = myr.set_index('date').sort_index()
myr_close = myr['close'].rename('myr_close')
myr_aligned = myr_close.reindex(daily.index, method='ffill')
print(f"MYR/USD aligned: {myr_aligned.notna().sum()} / {len(daily)} days  "
      f"range: {myr_aligned.min():.4f} – {myr_aligned.max():.4f}")

# ── 3. Load SBO (CBOT Soybean Oil, USD cents/lb) ─────────────────────────────
sbo = pd.read_csv('/home/user/claudestrat/data/sbo.csv', parse_dates=['date'])
sbo = sbo.set_index('date').sort_index()
sbo_close = sbo['close'].rename('sbo_close')
sbo_aligned = sbo_close.reindex(daily.index, method='ffill')
print(f"SBO aligned: {sbo_aligned.notna().sum()} / {len(daily)} days  "
      f"range: {sbo_aligned.min():.2f} – {sbo_aligned.max():.2f} USC/lb")

# ── 4. FCPO feature series ────────────────────────────────────────────────────
daily['ret_1d']  = np.log(daily['close']).diff(1)
daily['ret_3d']  = np.log(daily['close']).diff(3)
daily['ret_10d'] = np.log(daily['close']).diff(10)
daily['ret_22d'] = np.log(daily['close']).diff(22)
daily['gap']     = np.log(daily['open']) - np.log(daily['close'].shift(1))
daily['hl_ratio'] = np.log(daily['high'] / daily['low'])

rvol_5d  = daily['ret_1d'].rolling(5).std()  * np.sqrt(252)
rvol_22d = daily['ret_1d'].rolling(22).std() * np.sqrt(252)
rvol_66d = daily['ret_1d'].rolling(66).std() * np.sqrt(252)
daily['vol_ratio_22_66'] = rvol_22d / rvol_66d
daily['mom_z_5']  = daily['ret_1d'].rolling(5).sum() / (rvol_22d / np.sqrt(252) * np.sqrt(5)  + 1e-9)
daily['price_z_66'] = (daily['close'] - daily['close'].rolling(66).mean()) / (daily['close'].rolling(66).std() + 1e-9)

# Open-to-close return — actual tradeable return when entering at open
daily['ret_oc'] = np.log(daily['close']) - np.log(daily['open'])

# Calendar
daily['month']   = daily.index.month
daily['dow']     = daily.index.dayofweek
daily['qtr']     = daily.index.quarter
daily['harvest'] = daily['month'].isin([10, 11, 12, 1, 2]).astype(int)

# ── 5. MYR/USD feature series ─────────────────────────────────────────────────
myr_log    = np.log(myr_aligned)
myr_ret_1d = myr_log.diff(1).rename('myr_ret_1d')
myr_ret_5d = myr_log.diff(5).rename('myr_ret_5d')

myr_rvol_5d  = myr_ret_1d.rolling(5).std()
myr_rvol_22d = myr_ret_1d.rolling(22).std()
myr_vol_ratio = (myr_rvol_5d / (myr_rvol_22d + 1e-9)).rename('myr_vol_ratio')

myr_z_22 = ((myr_aligned - myr_aligned.rolling(22).mean()) /
             (myr_aligned.rolling(22).std() + 1e-9)).rename('myr_z_22')

myr_fcpo_corr = myr_ret_1d.rolling(22).corr(daily['ret_1d']).rename('myr_fcpo_corr')

# ── 6. SBO feature series ─────────────────────────────────────────────────────
sbo_log    = np.log(sbo_aligned)
sbo_ret_1d = sbo_log.diff(1).rename('sbo_ret_1d')
sbo_ret_5d = sbo_log.diff(5).rename('sbo_ret_5d')

sbo_rvol_5d  = sbo_ret_1d.rolling(5).std()
sbo_rvol_22d = sbo_ret_1d.rolling(22).std()
sbo_vol_ratio = (sbo_rvol_5d / (sbo_rvol_22d + 1e-9)).rename('sbo_vol_ratio')

sbo_z_22 = ((sbo_aligned - sbo_aligned.rolling(22).mean()) /
             (sbo_aligned.rolling(22).std() + 1e-9)).rename('sbo_z_22')

# CPO-SBO spread:
# FCPO is MYR/MT; SBO is USC/lb
# Convert SBO to MYR/MT: SBO_USC/lb * 2204.62 lb/MT / 100 USC/USD * MYR/USD
# spread = log(FCPO_MYR) - log(SBO_MYR_per_MT)
sbo_myr_per_mt = sbo_aligned * 2204.62 / 100 * myr_aligned   # MYR per MT
cpo_sbo_spread = (np.log(daily['close']) - np.log(sbo_myr_per_mt)).rename('cpo_sbo_spread')
cpo_sbo_spread_z22 = ((cpo_sbo_spread - cpo_sbo_spread.rolling(22).mean()) /
                       (cpo_sbo_spread.rolling(22).std() + 1e-9)).rename('cpo_sbo_spread_z22')

# SBO momentum z-score (like FCPO mom_z_5)
sbo_mom_z5 = (sbo_ret_1d.rolling(5).sum() / (sbo_rvol_22d * np.sqrt(5) + 1e-9)).rename('sbo_mom_z5')

# ── 7. Build feature matrix (shift all by 1 — look-ahead free) ───────────────
feat = pd.DataFrame(index=daily.index)

fcpo_numeric = ['ret_1d','ret_3d','ret_10d','ret_22d','gap','hl_ratio',
                'vol_ratio_22_66','mom_z_5','price_z_66']
myr_numeric  = ['myr_ret_1d','myr_ret_5d','myr_vol_ratio','myr_z_22','myr_fcpo_corr']
sbo_numeric  = ['sbo_ret_1d','sbo_ret_5d','sbo_vol_ratio','sbo_z_22',
                'cpo_sbo_spread','cpo_sbo_spread_z22','sbo_mom_z5']
calendar     = ['month','dow','qtr','harvest']

for col in fcpo_numeric:
    feat[col] = daily[col].shift(1)

myr_series = pd.DataFrame({
    'myr_ret_1d':    myr_ret_1d,
    'myr_ret_5d':    myr_ret_5d,
    'myr_vol_ratio': myr_vol_ratio,
    'myr_z_22':      myr_z_22,
    'myr_fcpo_corr': myr_fcpo_corr,
})
for col in myr_numeric:
    feat[col] = myr_series[col].shift(1)

sbo_series = pd.DataFrame({
    'sbo_ret_1d':       sbo_ret_1d,
    'sbo_ret_5d':       sbo_ret_5d,
    'sbo_vol_ratio':    sbo_vol_ratio,
    'sbo_z_22':         sbo_z_22,
    'cpo_sbo_spread':   cpo_sbo_spread,
    'cpo_sbo_spread_z22': cpo_sbo_spread_z22,
    'sbo_mom_z5':       sbo_mom_z5,
})
for col in sbo_numeric:
    feat[col] = sbo_series[col].shift(1)

for col in calendar:
    feat[col] = daily[col].shift(1)

# Per-day transaction cost threshold (today's close ≈ tomorrow's open, avoids look-ahead)
tc_series = np.log(1 + 35 / (daily['close'] * 25))
# fwd_ret: next day's open-to-close return (enter at tomorrow open, exit tomorrow close)
feat['fwd_ret_1d'] = daily['ret_oc'].shift(-1)
feat['target']     = (feat['fwd_ret_1d'] > tc_series).astype(int)
# next_open: actual entry price for P&L calculation
feat['next_open']  = daily['open'].shift(-1)

n_before = len(feat)
feat = feat.dropna()
print(f"\nRows before drop: {n_before}  |  After: {len(feat)}")
print(f"Date range: {feat.index[0].date()} → {feat.index[-1].date()}")

# ── 8. VIF iterative removal ──────────────────────────────────────────────────
print("\n" + "="*60)
print("VIF — ALL NUMERIC FEATURES (pre-filter)")
print("="*60)
all_numeric = fcpo_numeric + myr_numeric + sbo_numeric
X_vif = feat[all_numeric].replace([np.inf, -np.inf], np.nan).dropna()

vif_init = pd.DataFrame({
    'feature': X_vif.columns,
    'VIF': [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
}).sort_values('VIF', ascending=False)
print(vif_init.to_string(index=False))

def iterative_vif(cols, data, threshold=5.0):
    remaining = list(cols)
    dropped   = []
    iteration = 0
    while True:
        X = data[remaining].replace([np.inf, -np.inf], np.nan).dropna()
        vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
                        index=remaining)
        if vif.max() < threshold:
            break
        worst = vif.idxmax()
        print(f"  Iter {iteration+1}: drop '{worst}' (VIF={vif.max():.2f})")
        remaining.remove(worst)
        dropped.append(worst)
        iteration += 1
    X_final  = data[remaining].replace([np.inf, -np.inf], np.nan).dropna()
    vif_final = pd.DataFrame({'feature': remaining,
                               'VIF': [variance_inflation_factor(X_final.values, i)
                                       for i in range(len(remaining))]}).sort_values('VIF', ascending=False)
    return remaining, vif_final, dropped

print("\nIterative VIF removal:")
final_numeric, vif_final, dropped = iterative_vif(all_numeric, feat)
print(f"\nDropped: {dropped}")
print("\nFinal retained features (VIF < 5):")
print(vif_final.to_string(index=False))

final_features = final_numeric + calendar
print(f"\nTotal features: {len(final_features)}")
print("Feature list:", final_features)

# Class balance
vc = feat['target'].value_counts()
print(f"\nTarget balance: Long={vc.get(1,0)} ({vc.get(1,0)/len(feat)*100:.1f}%)  "
      f"Flat={vc.get(0,0)} ({vc.get(0,0)/len(feat)*100:.1f}%)")

# ── 9. Save ───────────────────────────────────────────────────────────────────
cols_save = final_features + ['fwd_ret_1d', 'target', 'next_open']
feat_out  = feat[cols_save].replace([np.inf, -np.inf], np.nan).dropna()
feat_out.to_csv('/home/user/claudestrat/data/features_v2.csv')

meta_v2 = {
    'feature_cols':   final_features,
    'target_col':     'target',
    'fwd_ret_col':    'fwd_ret_1d',
    'next_open_col':  'next_open',
    'transaction_cost_pct': float(tc_series.mean()),
    'n_rows':         len(feat_out),
    'date_range':     [str(feat_out.index[0].date()), str(feat_out.index[-1].date())],
    'external_data':  ['MYR/USD', 'SBO (CBOT ZL)'],
    'new_myr_features': [f for f in myr_numeric if f in final_features],
    'new_sbo_features': [f for f in sbo_numeric if f in final_features],
}
with open('/home/user/claudestrat/data/feature_meta_v2.json', 'w') as f:
    json.dump(meta_v2, f, indent=2)

print(f"\nSaved {len(feat_out)} rows × {len(final_features)} features to features_v2.csv")

# ── 10. Plots ──────────────────────────────────────────────────────────────────
print("\nGenerating plots...")

# MYR/USD + SBO + FCPO overview
fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
daily['close'].reindex(feat_out.index).plot(ax=axes[0], color='steelblue', lw=0.8)
axes[0].set_title('FCPO Close (MYR/MT)')
axes[0].set_ylabel('MYR/MT')

myr_aligned.reindex(feat_out.index).plot(ax=axes[1], color='darkorange', lw=0.8)
axes[1].set_title('MYR/USD Rate')
axes[1].set_ylabel('MYR per USD')

sbo_aligned.reindex(feat_out.index).plot(ax=axes[2], color='forestgreen', lw=0.8)
axes[2].set_title('SBO Close (USC/lb)')
axes[2].set_ylabel('USC/lb')

cpo_sbo_spread.reindex(feat_out.index).plot(ax=axes[3], color='crimson', lw=0.8)
axes[3].axhline(cpo_sbo_spread.reindex(feat_out.index).mean(), color='black', lw=0.8, ls='--')
axes[3].set_title('CPO-SBO Spread (log ratio, MYR-adjusted)')
axes[3].set_ylabel('log ratio')

plt.tight_layout()
plt.savefig(f'{OUT}/01_price_overview.png', dpi=150)
plt.close()

# Feature correlation heatmap (final set)
fig, ax = plt.subplots(figsize=(16, 14))
corr = feat_out[final_numeric].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            annot=True, fmt='.2f', linewidths=0.3, ax=ax)
ax.set_title('Feature Correlation Matrix v2 (FCPO + MYR/USD + SBO)')
plt.tight_layout()
plt.savefig(f'{OUT}/02_correlation_matrix.png', dpi=150)
plt.close()

# Forward return by quintile for key cross-asset features
cross_feats = [f for f in ['cpo_sbo_spread','cpo_sbo_spread_z22','myr_z_22',
                             'sbo_ret_1d','myr_fcpo_corr','sbo_vol_ratio']
               if f in feat_out.columns]
n_plot = min(len(cross_feats), 6)
if n_plot > 0:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for i, col in enumerate(cross_feats[:n_plot]):
        feat_out['_q'] = pd.qcut(feat_out[col], q=5, labels=['Q1','Q2','Q3','Q4','Q5'], duplicates='drop')
        feat_out.groupby('_q', observed=True)['fwd_ret_1d'].mean().plot(
            kind='bar', ax=axes[i], color='steelblue', edgecolor='none')
        axes[i].axhline(0, color='black', lw=0.8)
        axes[i].set_title(f'Fwd Return by {col} quintile')
        axes[i].tick_params(axis='x', rotation=0)
    for j in range(n_plot, 6):
        axes[j].set_visible(False)
    feat_out.drop(columns=['_q'], inplace=True)
    plt.suptitle('Mean Forward Return by Cross-Asset Feature Quintile')
    plt.tight_layout()
    plt.savefig(f'{OUT}/03_crossasset_vs_fwdret.png', dpi=150)
    plt.close()

# VIF bar chart
fig, ax = plt.subplots(figsize=(10, 7))
vif_plot = vif_final.set_index('feature')['VIF'].sort_values()
vif_plot.plot(kind='barh', color='steelblue', ax=ax)
ax.axvline(5, color='red', linestyle='--', lw=1.5, label='VIF=5 threshold')
ax.set_title('VIF — Retained Features (all < 5)')
ax.set_xlabel('VIF')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT}/04_vif.png', dpi=150)
plt.close()

print(f"\nAll plots saved to {OUT}/")
print("\nPhase 2b complete. Ready to re-run Phase 3 with MYR/USD + SBO features.")
