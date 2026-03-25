"""
Plot trade entry/exit points on FCPO price chart.
Shows the nature of the strategy visually across the OOF period.
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

OUT = '/home/user/claudestrat/output/trade_plot'
os.makedirs(OUT, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
oof = pd.read_csv('/home/user/claudestrat/data/oof_signals.csv', parse_dates=['date'])
oof = oof.sort_values('date').reset_index(drop=True)

# Load FCPO daily price
df_h = pd.read_csv('/home/user/claudestrat/data/fcpo.csv', parse_dates=['datetime'])
df_h = df_h.sort_values('datetime').reset_index(drop=True)
daily = df_h.groupby(df_h['datetime'].dt.date).agg(
    close=('close', 'last')
).reset_index().rename(columns={'datetime': 'date'})
daily['date'] = pd.to_datetime(daily['date'])
daily = daily.set_index('date').sort_index()

# Align price to OOF dates
price = daily['close'].reindex(oof['date']).values
dates = oof['date'].values
signals = oof['signal'].values.astype(int)
equity = oof['equity'].values
prob = oof['prob_long'].values

# ── Identify entries and exits ────────────────────────────────────────────────
prev_sig = np.concatenate([[0], signals[:-1]])
entries = np.where((signals == 1) & (prev_sig == 0))[0]   # 0 → 1
exits   = np.where((signals == 0) & (prev_sig == 1))[0]   # 1 → 0

print(f"OOF period: {oof['date'].iloc[0].date()} → {oof['date'].iloc[-1].date()}")
print(f"Total bars: {len(oof)}")
print(f"Entries: {len(entries)}  |  Exits: {len(exits)}")
print(f"Avg hold per trade: {signals.mean() * len(signals) / max(len(entries), 1):.1f} days")

# Trade durations
trade_lengths = []
for i, entry_i in enumerate(entries):
    # Find the next exit after this entry
    next_exits = exits[exits > entry_i]
    if len(next_exits) > 0:
        trade_lengths.append(next_exits[0] - entry_i)
if trade_lengths:
    print(f"Median trade length: {np.median(trade_lengths):.0f} days  |  "
          f"Max: {np.max(trade_lengths):.0f}  |  Min: {np.min(trade_lengths):.0f}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1: Full period overview — price + shaded longs + entries/exits
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(18, 13),
                          gridspec_kw={'height_ratios': [4, 1.5, 1.5]},
                          sharex=True)
fig.patch.set_facecolor('#0f1117')
for ax in axes:
    ax.set_facecolor('#0f1117')
    ax.tick_params(colors='#cccccc')
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.label.set_color('#cccccc')
    ax.title.set_color('#ffffff')

dt = pd.to_datetime(dates)

# ── Panel 1: Price + shaded long zones + entry/exit markers ──────────────────
ax = axes[0]
ax.plot(dt, price, color='#4a9eff', lw=0.9, zorder=2, label='FCPO Close')

# Shade long periods
in_trade = False
trade_start = None
for i in range(len(signals)):
    if signals[i] == 1 and not in_trade:
        trade_start = dt[i]
        in_trade = True
    elif signals[i] == 0 and in_trade:
        ax.axvspan(trade_start, dt[i], alpha=0.18, color='#00cc66', zorder=1)
        in_trade = False
if in_trade:
    ax.axvspan(trade_start, dt[-1], alpha=0.18, color='#00cc66', zorder=1)

# Entry markers (green up-triangles)
ax.scatter(dt[entries], price[entries], marker='^', color='#00ff88', s=45,
           zorder=5, label=f'Entry ({len(entries)})', linewidths=0)

# Exit markers (red down-triangles)
ax.scatter(dt[exits], price[exits], marker='v', color='#ff4444', s=45,
           zorder=5, label=f'Exit ({len(exits)})', linewidths=0)

ax.set_ylabel('FCPO Close (MYR/MT)', color='#cccccc', fontsize=10)
ax.set_title('FCPO ML Strategy — Trade Entry/Exit Points (OOF 2016–2025)', fontsize=13, pad=10)
ax.legend(loc='upper left', fontsize=8, facecolor='#1a1a2e', labelcolor='white',
          edgecolor='#333')
ax.yaxis.set_tick_params(labelsize=8)

# ── Panel 2: Probability signal ───────────────────────────────────────────────
ax2 = axes[1]
ax2.plot(dt, prob, color='#aaaaaa', lw=0.6, alpha=0.7)
ax2.fill_between(dt, prob, 0.5, where=(prob > 0.5), color='#00cc66', alpha=0.5)
ax2.fill_between(dt, prob, 0.5, where=(prob <= 0.5), color='#cc3333', alpha=0.3)
ax2.axhline(0.5, color='white', lw=0.8, linestyle='--', alpha=0.5)
ax2.set_ylabel('P(Long)', color='#cccccc', fontsize=9)
ax2.set_ylim(0, 1)
ax2.yaxis.set_tick_params(labelsize=8)
ax2.set_title('Model Confidence — P(Long)', fontsize=10, pad=4)

# ── Panel 3: Equity curve ─────────────────────────────────────────────────────
ax3 = axes[2]
ax3.plot(dt, equity, color='#ffd700', lw=1.2)
ax3.axhline(8000, color='#888', lw=0.7, linestyle=':', alpha=0.8)
ax3.fill_between(dt, equity, 8000, where=(equity >= 8000), color='#00cc66', alpha=0.25)
ax3.fill_between(dt, equity, 8000, where=(equity < 8000), color='#cc3333', alpha=0.25)
ax3.set_ylabel('Equity (RM)', color='#cccccc', fontsize=9)
ax3.set_xlabel('Date', color='#cccccc', fontsize=9)
ax3.yaxis.set_tick_params(labelsize=8)
ax3.set_title('Strategy Equity', fontsize=10, pad=4)
ax3.xaxis.label.set_color('#cccccc')

plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig(f'{OUT}/01_full_period_trades.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("Saved full period chart")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2: Zoom into 4 representative 18-month windows
# ══════════════════════════════════════════════════════════════════════════════
windows = [
    ('2017-01-01', '2018-06-30', '2017 H1 — Sideways/choppy'),
    ('2019-01-01', '2020-06-30', '2019–2020 — COVID crash + recovery'),
    ('2021-01-01', '2022-06-30', '2021–2022 — Bull run + top'),
    ('2023-06-01', '2025-01-31', '2023–2024 — Recent period'),
]

fig, axes = plt.subplots(4, 1, figsize=(18, 20))
fig.patch.set_facecolor('#0f1117')

for idx, (start, end, title) in enumerate(windows):
    mask = (dt >= start) & (dt <= end)
    if mask.sum() < 10:
        continue

    dt_w   = dt[mask]
    price_w = price[mask]
    sig_w   = signals[mask]
    prob_w  = prob[mask]
    eq_w    = equity[mask]

    prev_w = np.concatenate([[0], sig_w[:-1]])
    entries_w = np.where((sig_w == 1) & (prev_w == 0))[0]
    exits_w   = np.where((sig_w == 0) & (prev_w == 1))[0]

    ax = axes[idx]
    ax.set_facecolor('#0f1117')
    ax.tick_params(colors='#cccccc')
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Price line
    ax.plot(dt_w, price_w, color='#4a9eff', lw=1.2, zorder=2)

    # Shade long zones
    in_trade = False
    trade_start = None
    for i in range(len(sig_w)):
        if sig_w[i] == 1 and not in_trade:
            trade_start = dt_w.iloc[i] if hasattr(dt_w, 'iloc') else dt_w[i]
            in_trade = True
        elif sig_w[i] == 0 and in_trade:
            ax.axvspan(trade_start, dt_w.iloc[i] if hasattr(dt_w, 'iloc') else dt_w[i],
                       alpha=0.20, color='#00cc66', zorder=1)
            in_trade = False
    if in_trade:
        ax.axvspan(trade_start, dt_w.iloc[-1] if hasattr(dt_w, 'iloc') else dt_w[-1],
                   alpha=0.20, color='#00cc66', zorder=1)

    # Entry/exit markers with price labels
    if len(entries_w) > 0:
        ax.scatter(dt_w.iloc[entries_w] if hasattr(dt_w, 'iloc') else dt_w[entries_w],
                   price_w[entries_w], marker='^', color='#00ff88', s=80, zorder=5,
                   linewidths=0, label=f'Entry ({len(entries_w)})')
    if len(exits_w) > 0:
        ax.scatter(dt_w.iloc[exits_w] if hasattr(dt_w, 'iloc') else dt_w[exits_w],
                   price_w[exits_w], marker='v', color='#ff4444', s=80, zorder=5,
                   linewidths=0, label=f'Exit ({len(exits_w)})')

    ax.set_title(title, fontsize=11, color='white', pad=5)
    ax.set_ylabel('FCPO (MYR/MT)', color='#aaaaaa', fontsize=9)
    ax.yaxis.set_tick_params(labelsize=8, colors='#aaaaaa')
    ax.xaxis.set_tick_params(labelsize=8, colors='#aaaaaa')
    ax.legend(loc='upper left', fontsize=8, facecolor='#1a1a2e', labelcolor='white',
              edgecolor='#333')

plt.suptitle('FCPO ML Strategy — Trade Entries/Exits by Period', fontsize=13,
             color='white', y=1.002)
plt.tight_layout()
plt.savefig(f'{OUT}/02_zoomed_periods.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("Saved zoomed periods chart")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3: Trade statistics breakdown
# ══════════════════════════════════════════════════════════════════════════════
# Compute per-trade P&L
trade_pnl = []
for i, ei in enumerate(entries):
    # Find next exit
    next_exits_i = exits[exits > ei]
    if len(next_exits_i) == 0:
        exit_i = len(signals) - 1
    else:
        exit_i = next_exits_i[0]

    entry_price = price[ei]
    exit_price  = price[min(exit_i, len(price)-1)]
    n_days      = exit_i - ei
    ret_pct     = (exit_price / entry_price - 1) * 100
    # approximate RM pnl: 1 lot, 25MT, entry-to-exit price diff
    pnl_rm      = (exit_price - entry_price) * 25 - 35  # minus commission

    trade_pnl.append({
        'entry_date': dates[ei],
        'exit_date':  dates[min(exit_i, len(dates)-1)],
        'entry_price': entry_price,
        'exit_price':  exit_price,
        'n_days':      n_days,
        'ret_pct':     ret_pct,
        'pnl_rm':      pnl_rm,
    })

trades_df = pd.DataFrame(trade_pnl)
wins = (trades_df['pnl_rm'] > 0).sum()
losses = (trades_df['pnl_rm'] <= 0).sum()
avg_win  = trades_df[trades_df['pnl_rm'] > 0]['pnl_rm'].mean()
avg_loss = trades_df[trades_df['pnl_rm'] <= 0]['pnl_rm'].mean()

print(f"\nTrade stats:")
print(f"  Total trades:  {len(trades_df)}")
print(f"  Win rate:      {wins/len(trades_df)*100:.1f}%")
print(f"  Avg win:       RM {avg_win:,.0f}")
print(f"  Avg loss:      RM {avg_loss:,.0f}")
print(f"  Profit factor: {abs(trades_df[trades_df['pnl_rm']>0]['pnl_rm'].sum() / trades_df[trades_df['pnl_rm']<=0]['pnl_rm'].sum()):.2f}")
print(f"  Median hold:   {trades_df['n_days'].median():.0f} days")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.patch.set_facecolor('#0f1117')
for ax in axes.flat:
    ax.set_facecolor('#0f1117')
    ax.tick_params(colors='#cccccc')
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Hold duration histogram
ax = axes[0, 0]
ax.hist(trades_df['n_days'], bins=40, color='#4a9eff', edgecolor='none', alpha=0.8)
ax.axvline(trades_df['n_days'].median(), color='yellow', lw=1.5, ls='--',
           label=f"Median={trades_df['n_days'].median():.0f}d")
ax.set_title('Trade Hold Duration (days)', color='white')
ax.set_xlabel('Days held', color='#aaa')
ax.set_ylabel('# Trades', color='#aaa')
ax.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white', edgecolor='#333')

# P&L per trade
ax = axes[0, 1]
colors_pnl = ['#00cc66' if p > 0 else '#cc3333' for p in trades_df['pnl_rm']]
ax.bar(range(len(trades_df)), trades_df['pnl_rm'].sort_values(), color=colors_pnl,
       edgecolor='none', alpha=0.85)
ax.axhline(0, color='white', lw=0.8)
ax.set_title(f'P&L per Trade (sorted)  |  Win rate {wins/len(trades_df)*100:.0f}%', color='white')
ax.set_xlabel('Trade rank', color='#aaa')
ax.set_ylabel('RM P&L (1 lot)', color='#aaa')

# P&L distribution
ax = axes[1, 0]
ax.hist(trades_df['pnl_rm'], bins=50, color='#4a9eff', edgecolor='none', alpha=0.8)
ax.axvline(0, color='white', lw=1, ls='--')
ax.axvline(trades_df['pnl_rm'].mean(), color='yellow', lw=1.5, ls='-',
           label=f"Mean=RM{trades_df['pnl_rm'].mean():,.0f}")
ax.set_title('Trade P&L Distribution (RM)', color='white')
ax.set_xlabel('RM P&L', color='#aaa')
ax.set_ylabel('# Trades', color='#aaa')
ax.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white', edgecolor='#333')

# Cumulative equity vs # trades
ax = axes[1, 1]
cum_pnl = trades_df['pnl_rm'].cumsum() + 8000
ax.plot(range(len(cum_pnl)), cum_pnl, color='#ffd700', lw=1.5)
ax.axhline(8000, color='#888', lw=0.8, ls=':')
ax.fill_between(range(len(cum_pnl)), cum_pnl, 8000,
                where=(cum_pnl >= 8000), color='#00cc66', alpha=0.2)
ax.fill_between(range(len(cum_pnl)), cum_pnl, 8000,
                where=(cum_pnl < 8000), color='#cc3333', alpha=0.2)
ax.set_title('Cumulative Trade-by-Trade Equity', color='white')
ax.set_xlabel('Trade #', color='#aaa')
ax.set_ylabel('Equity (RM)', color='#aaa')

plt.suptitle('FCPO ML Strategy — Trade Statistics', fontsize=13, color='white')
plt.tight_layout()
plt.savefig(f'{OUT}/03_trade_statistics.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("Saved trade statistics chart")

print(f"\nAll charts saved to {OUT}/")
