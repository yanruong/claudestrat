"""
Phase 2c (Intraday): Build per-day features for the afternoon+evening FCPO strategy.

Entry:  15:00 (afternoon session open)
Exit:   23:30 (evening session close)
Target: ret_afeve = log(evening_close / afternoon_open) > TC_pct

Signal is generated just before 15:00 using only morning session data (available by 12:30/13:00).

No look-ahead: morning features use today's morning session (complete by 12:30pm).
Previous-day FCPO features use prev_day_close (yesterday's evening close), so no extra shift needed.
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

OUT = '/home/user/claudestrat/output/phase2c'
os.makedirs(OUT, exist_ok=True)

# ── 1. Load FCPO intraday bars ────────────────────────────────────────────────
df_h = pd.read_csv('/home/user/claudestrat/data/fcpo.csv', parse_dates=['datetime'])
df_h = df_h.sort_values('datetime').reset_index(drop=True)

times = df_h['datetime'].dt.strftime('%H:%M')
dates = df_h['datetime'].dt.date.astype(str)

print(f"FCPO intraday bars: {len(df_h)}  |  {df_h['datetime'].min()} → {df_h['datetime'].max()}")
print(f"Time slots present: {sorted(times.unique())}")

# ── 2. Build per-day session aggregates ───────────────────────────────────────
# Morning open = 11:00 open
# Morning close = 12:30 close (or 13:00 if 12:30 not available — exchange changed in Feb 2025)
# Morning high/low = max/min of 11:00, 12:00, 12:30 (or 13:00) bars
# Afternoon open = 15:00 open (entry price)
# Evening close = 23:30 close (exit price)

morning_bars = ['11:00', '12:00', '12:30', '13:00']
df_morning = df_h[times.isin(morning_bars)].copy()
df_morning['date'] = df_morning['datetime'].dt.date.astype(str)

# Morning open: open of 11:00 bar
morning_open = (df_h[times == '11:00']
                .assign(date=lambda d: d['datetime'].dt.date.astype(str))
                .set_index('date')['open']
                .rename('morning_open'))

# Morning close: close of 12:30 bar; fall back to 13:00 if 12:30 not present
mc_1230 = (df_h[times == '12:30']
            .assign(date=lambda d: d['datetime'].dt.date.astype(str))
            .set_index('date')['close']
            .rename('morning_close'))
mc_1300 = (df_h[times == '13:00']
            .assign(date=lambda d: d['datetime'].dt.date.astype(str))
            .set_index('date')['close']
            .rename('morning_close'))
morning_close = mc_1230.combine_first(mc_1300)

# Morning high / low across all morning bars
morning_hl = df_morning.groupby('date').agg(
    morning_high=('high', 'max'),
    morning_low=('low',  'min'),
)

# Afternoon open (entry)
afternoon_open = (df_h[times == '15:00']
                  .assign(date=lambda d: d['datetime'].dt.date.astype(str))
                  .set_index('date')['open']
                  .rename('afternoon_open'))

# Evening close (exit)
evening_close = (df_h[times == '23:30']
                 .assign(date=lambda d: d['datetime'].dt.date.astype(str))
                 .set_index('date')['close']
                 .rename('evening_close'))

# Assemble daily intraday frame
daily = pd.DataFrame({
    'morning_open':   morning_open,
    'morning_close':  morning_close,
    'afternoon_open': afternoon_open,
    'evening_close':  evening_close,
}).join(morning_hl)

daily.index = pd.to_datetime(daily.index)
daily = daily.sort_index()

# Keep only days that have all required bars
n_before = len(daily)
daily = daily.dropna(subset=['morning_open', 'morning_close', 'morning_high', 'morning_low',
                              'afternoon_open', 'evening_close'])
print(f"\nPer-day rows (all sessions present): {n_before} → {len(daily)}")
print(f"Date range: {daily.index[0].date()} → {daily.index[-1].date()}")

# Previous day's evening_close (for gap and prev-day features)
daily['prev_day_close'] = daily['evening_close'].shift(1)

# ── 3. Intraday FCPO features (morning session — complete by 12:30pm) ─────────
daily['morning_ret']      = np.log(daily['morning_close'] / daily['morning_open'])
daily['morning_gap']      = np.log(daily['morning_open']  / daily['prev_day_close'])
daily['morning_hl']       = np.log(daily['morning_high']  / daily['morning_low'])
daily['morning_gap_ret']  = daily['morning_gap'] + daily['morning_ret']

# ── 4. Previous-day FCPO features ─────────────────────────────────────────────
# These are built from the daily series of evening_close values.
# ret_1d  = log(prev_day_close / day_before_close)  — yesterday's full-day return
ev = daily['evening_close']
daily['ret_1d']     = np.log(daily['prev_day_close'] / daily['prev_day_close'].shift(1))
daily['ret_3d']     = np.log(daily['prev_day_close'] / daily['prev_day_close'].shift(3))
daily['ret_10d']    = np.log(daily['prev_day_close'] / daily['prev_day_close'].shift(10))
daily['price_z_66'] = ((daily['prev_day_close'] - daily['prev_day_close'].rolling(66).mean()) /
                        (daily['prev_day_close'].rolling(66).std() + 1e-9))
rvol_22d = daily['ret_1d'].rolling(22).std()
daily['mom_z_5'] = (daily['ret_1d'].rolling(5).sum() /
                    (rvol_22d * np.sqrt(5) + 1e-9))

# ── 5. Load and align SBO (CBOT Soybean Oil, USD cents/lb) ───────────────────
sbo = pd.read_csv('/home/user/claudestrat/data/sbo.csv', parse_dates=['date'])
sbo = sbo.set_index('date').sort_index()
sbo_close = sbo['close'].rename('sbo_close')
sbo_aligned = sbo_close.reindex(daily.index, method='ffill')
print(f"\nSBO aligned: {sbo_aligned.notna().sum()} / {len(daily)} days  "
      f"range: {sbo_aligned.min():.2f} – {sbo_aligned.max():.2f} USC/lb")

# ── 6. Load and align MYR/USD ────────────────────────────────────────────────
myr = pd.read_csv('/home/user/claudestrat/data/myrusd.csv', parse_dates=['date'])
myr = myr.set_index('date').sort_index()
myr_close = myr['close'].rename('myr_close')
myr_aligned = myr_close.reindex(daily.index, method='ffill')
print(f"MYR/USD aligned: {myr_aligned.notna().sum()} / {len(daily)} days  "
      f"range: {myr_aligned.min():.4f} – {myr_aligned.max():.4f}")

# ── 7. SBO features (yesterday's SBO close — US session closed ~3:20am MYT) ──
# All SBO features already use yesterday's data when aligned to trading date.
# We shift by 1 to ensure we're using yesterday's value (not today's).
sbo_log = np.log(sbo_aligned)
sbo_ret_1d_raw = sbo_log.diff(1)
sbo_ret_5d_raw = sbo_log.diff(5)
sbo_z_22_raw   = ((sbo_aligned - sbo_aligned.rolling(22).mean()) /
                   (sbo_aligned.rolling(22).std() + 1e-9))

# CPO-SBO spread: log(FCPO MYR/MT) - log(SBO MYR/MT)
# SBO USC/lb → MYR/MT: SBO * 2204.62 / 100 * MYR/USD
sbo_myr_per_mt = sbo_aligned * 2204.62 / 100 * myr_aligned
cpo_sbo_spread_raw = (np.log(daily['prev_day_close']) - np.log(sbo_myr_per_mt))
cpo_sbo_spread_z22_raw = ((cpo_sbo_spread_raw - cpo_sbo_spread_raw.rolling(22).mean()) /
                           (cpo_sbo_spread_raw.rolling(22).std() + 1e-9))

# Shift by 1 to use yesterday's SBO (US market closed before MYT morning)
daily['sbo_ret_1d']        = sbo_ret_1d_raw.shift(1)
daily['sbo_ret_5d']        = sbo_ret_5d_raw.shift(1)
daily['sbo_z_22']          = sbo_z_22_raw.shift(1)
daily['cpo_sbo_spread_z22'] = cpo_sbo_spread_z22_raw.shift(1)

# ── 8. MYR/USD features ───────────────────────────────────────────────────────
myr_log = np.log(myr_aligned)
myr_ret_1d_raw = myr_log.diff(1)
myr_z_22_raw   = ((myr_aligned - myr_aligned.rolling(22).mean()) /
                   (myr_aligned.rolling(22).std() + 1e-9))

daily['myr_ret_1d'] = myr_ret_1d_raw.shift(1)
daily['myr_z_22']   = myr_z_22_raw.shift(1)

# ── 9. Calendar features ─────────────────────────────────────────────────────
daily['month']   = daily.index.month
daily['dow']     = daily.index.dayofweek
daily['qtr']     = daily.index.quarter
daily['harvest'] = daily['month'].isin([10, 11, 12, 1, 2]).astype(int)

# ── 10. Target ───────────────────────────────────────────────────────────────
daily['ret_afeve'] = np.log(daily['evening_close']) - np.log(daily['afternoon_open'])
tc_pct = 35 / (daily['afternoon_open'] * 25)
daily['target'] = (daily['ret_afeve'] > tc_pct).astype(int)

# ── 11. Drop NaN rows ────────────────────────────────────────────────────────
all_numeric = ['morning_ret', 'morning_gap', 'morning_hl', 'morning_gap_ret',
               'ret_1d', 'ret_3d', 'ret_10d', 'price_z_66', 'mom_z_5',
               'sbo_ret_1d', 'sbo_ret_5d', 'sbo_z_22', 'cpo_sbo_spread_z22',
               'myr_ret_1d', 'myr_z_22']
calendar = ['month', 'dow', 'qtr', 'harvest']

n_before = len(daily)
daily = daily.dropna(subset=all_numeric + ['ret_afeve', 'target', 'afternoon_open'])
print(f"\nRows before drop NaN: {n_before}  |  After: {len(daily)}")
print(f"Date range: {daily.index[0].date()} → {daily.index[-1].date()}")

# Class balance
vc = daily['target'].value_counts()
print(f"\nTarget balance: Long={vc.get(1,0)} ({vc.get(1,0)/len(daily)*100:.1f}%)  "
      f"Flat={vc.get(0,0)} ({vc.get(0,0)/len(daily)*100:.1f}%)")

# ── 12. VIF iterative removal ────────────────────────────────────────────────
print("\n" + "="*60)
print("VIF — ALL NUMERIC FEATURES (pre-filter)")
print("="*60)
X_vif = daily[all_numeric].replace([np.inf, -np.inf], np.nan).dropna()

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
    X_final   = data[remaining].replace([np.inf, -np.inf], np.nan).dropna()
    vif_final = pd.DataFrame({'feature': remaining,
                               'VIF': [variance_inflation_factor(X_final.values, i)
                                       for i in range(len(remaining))]}).sort_values('VIF', ascending=False)
    return remaining, vif_final, dropped

print("\nIterative VIF removal (threshold = 5.0):")
final_numeric, vif_final, dropped = iterative_vif(all_numeric, daily)
print(f"\nDropped: {dropped}")
print("\nFinal retained features (VIF < 5):")
print(vif_final.to_string(index=False))

final_features = final_numeric + calendar
print(f"\nTotal features retained: {len(final_features)}")
print("Feature list:", final_features)

# ── 13. Save ─────────────────────────────────────────────────────────────────
cols_save = final_features + ['ret_afeve', 'target', 'afternoon_open']
feat_out  = daily[cols_save].replace([np.inf, -np.inf], np.nan).dropna()
feat_out.to_csv('/home/user/claudestrat/data/features_v3.csv')

meta_v3 = {
    'feature_cols':        final_features,
    'target_col':          'target',
    'fwd_ret_col':         'ret_afeve',
    'entry_price_col':     'afternoon_open',
    'transaction_cost_rm': 35,
    'contract_size_mt':    25,
    'entry_time':          '15:00',
    'exit_time':           '23:30',
    'n_rows':              len(feat_out),
    'date_range':          [str(feat_out.index[0].date()), str(feat_out.index[-1].date())],
    'intraday_features':   [f for f in ['morning_ret','morning_gap','morning_hl','morning_gap_ret'] if f in final_features],
    'prev_day_fcpo':       [f for f in ['ret_1d','ret_3d','ret_10d','price_z_66','mom_z_5'] if f in final_features],
    'sbo_features':        [f for f in ['sbo_ret_1d','sbo_ret_5d','sbo_z_22','cpo_sbo_spread_z22'] if f in final_features],
    'myr_features':        [f for f in ['myr_ret_1d','myr_z_22'] if f in final_features],
    'calendar_features':   calendar,
}
with open('/home/user/claudestrat/data/feature_meta_v3.json', 'w') as f:
    json.dump(meta_v3, f, indent=2)

print(f"\nSaved {len(feat_out)} rows × {len(final_features)} features → features_v3.csv")
print(f"Meta saved → feature_meta_v3.json")

# ── 14. Plots ────────────────────────────────────────────────────────────────
print("\nGenerating plots...")

# Plot 1: Afternoon open price and ret_afeve distribution
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

feat_out['afternoon_open'].plot(ax=axes[0, 0], color='steelblue', lw=0.8)
axes[0, 0].set_title('FCPO Afternoon Open (MYR/MT)')
axes[0, 0].set_ylabel('MYR/MT')

feat_out['ret_afeve'].hist(bins=60, ax=axes[0, 1], color='steelblue', edgecolor='none')
axes[0, 1].axvline(0, color='black', lw=0.8)
axes[0, 1].set_title('Distribution of ret_afeve (afternoon+evening return)')
axes[0, 1].set_xlabel('Log Return')

morning_plot_col = 'morning_gap_ret' if 'morning_gap_ret' in feat_out.columns else ('morning_gap' if 'morning_gap' in feat_out.columns else feat_out.columns[0])
feat_out[morning_plot_col].plot(ax=axes[1, 0], color='darkorange', lw=0.7, alpha=0.8)
axes[1, 0].set_title(f'{morning_plot_col} (morning session signal)')
axes[1, 0].set_ylabel('Log Return')

# Target over time (rolling 22d mean)
feat_out['target'].rolling(22).mean().plot(ax=axes[1, 1], color='forestgreen', lw=1.0)
axes[1, 1].axhline(feat_out['target'].mean(), color='grey', lw=0.8, linestyle='--',
                    label=f"Mean={feat_out['target'].mean():.2f}")
axes[1, 1].set_title('Rolling 22d Signal Rate (target=1 = profitable long)')
axes[1, 1].set_ylabel('Fraction Long')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(f'{OUT}/01_overview.png', dpi=150)
plt.close()

# Plot 2: Feature correlation heatmap
fig, ax = plt.subplots(figsize=(16, 14))
corr = feat_out[final_numeric].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            annot=True, fmt='.2f', linewidths=0.3, ax=ax)
ax.set_title('Feature Correlation Matrix v3 (Intraday FCPO Strategy)')
plt.tight_layout()
plt.savefig(f'{OUT}/02_correlation_matrix.png', dpi=150)
plt.close()

# Plot 3: VIF bar chart
fig, ax = plt.subplots(figsize=(10, 7))
vif_plot = vif_final.set_index('feature')['VIF'].sort_values()
vif_plot.plot(kind='barh', color='steelblue', ax=ax)
ax.axvline(5, color='red', linestyle='--', lw=1.5, label='VIF=5 threshold')
ax.set_title('VIF — Retained Features v3 (all < 5)')
ax.set_xlabel('VIF')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT}/03_vif.png', dpi=150)
plt.close()

# Plot 4: ret_afeve by morning_ret quintile (signal preview)
key_feats = [f for f in ['morning_gap_ret', 'morning_gap', 'morning_hl', 'sbo_ret_1d',
                          'myr_ret_1d', 'ret_1d'] if f in feat_out.columns]
n_plot = min(len(key_feats), 6)
if n_plot > 0:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for i, col in enumerate(key_feats[:n_plot]):
        feat_out['_q'] = pd.qcut(feat_out[col], q=5, labels=['Q1','Q2','Q3','Q4','Q5'], duplicates='drop')
        feat_out.groupby('_q', observed=True)['ret_afeve'].mean().plot(
            kind='bar', ax=axes[i], color='steelblue', edgecolor='none')
        axes[i].axhline(0, color='black', lw=0.8)
        axes[i].set_title(f'Afeve Return by {col} quintile')
        axes[i].tick_params(axis='x', rotation=0)
    for j in range(n_plot, 6):
        axes[j].set_visible(False)
    feat_out.drop(columns=['_q'], inplace=True)
    plt.suptitle('Mean Afternoon+Evening Return by Feature Quintile')
    plt.tight_layout()
    plt.savefig(f'{OUT}/04_feature_vs_return.png', dpi=150)
    plt.close()

print(f"\nAll plots saved to {OUT}/")
print("\nPhase 2c complete. Ready to run phase3_v3_model.py")
