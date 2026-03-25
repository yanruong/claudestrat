"""
Phase 3 v3: LightGBM + SHAP with Purged Walk-Forward CV + Optuna
Strategy: FCPO afternoon+evening session (15:00 entry → 23:30 exit)
- Purged walk-forward CV (5-fold, 5-day embargo)
- Optuna hyperparameter tuning (60 trials)
- SHAP feature importance
- Custom Score = Final Equity / Years / (MaxDrawdown + 8000)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import shap
import optuna
import json, os, pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

optuna.logging.set_verbosity(optuna.logging.WARNING)

OUT = '/home/user/claudestrat/output/phase3_v3'
os.makedirs(OUT, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
INITIAL_CAPITAL  = 8_000      # RM
CONTRACT_SIZE    = 25         # MT per lot
COMMISSION_RT    = 35         # RM round-trip per lot
N_FOLDS          = 5
EMBARGO_DAYS     = 5
N_OPTUNA_TRIALS  = 60
RANDOM_STATE     = 42

# ── Load features ─────────────────────────────────────────────────────────────
meta = json.load(open('/home/user/claudestrat/data/feature_meta_v3.json'))
feat_cols       = meta['feature_cols']
target_col      = meta['target_col']
fwd_ret_col     = meta['fwd_ret_col']
entry_price_col = meta['entry_price_col']

df = pd.read_csv('/home/user/claudestrat/data/features_v3.csv', index_col=0, parse_dates=True)
df = df.sort_index()

X         = df[feat_cols].copy()
y         = df[target_col].copy()
fwd_ret   = df[fwd_ret_col].copy()
entry_price = df[entry_price_col].copy()
entry_price_vals = entry_price.values

print(f"Dataset: {len(df)} rows × {len(feat_cols)} features")
print(f"Date range: {df.index[0].date()} → {df.index[-1].date()}")
print(f"Features: {feat_cols}")
vc = y.value_counts()
print(f"Target balance: Long={vc.get(1,0)} ({vc.get(1,0)/len(y)*100:.1f}%)  "
      f"Flat={vc.get(0,0)} ({vc.get(0,0)/len(y)*100:.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
# Purged Walk-Forward CV splits
# ═══════════════════════════════════════════════════════════════════════════════
def purged_walkforward_splits(n, n_folds=5, embargo=5):
    """
    Expanding-window walk-forward CV with embargo.
    Splits data into (n_folds+1) segments. Fold k trains on [0, test_start-embargo),
    tests on segment k+1. NO min_train_frac — test must always be strictly after train.
    """
    fold_size = n // (n_folds + 1)
    splits = []
    for k in range(n_folds):
        test_start = (k + 1) * fold_size
        test_end   = test_start + fold_size if k < n_folds - 1 else n
        train_end  = test_start - embargo
        if train_end <= 0 or test_start >= n:
            continue
        train_idx = np.arange(0, train_end)
        test_idx  = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))
    return splits

splits = purged_walkforward_splits(len(df), n_folds=N_FOLDS, embargo=EMBARGO_DAYS)
print(f"\nWalk-forward CV: {len(splits)} folds")
for i, (tr, te) in enumerate(splits):
    print(f"  Fold {i+1}: train [{df.index[tr[0]].date()} → {df.index[tr[-1]].date()}] "
          f"({len(tr)}d)  |  test [{df.index[te[0]].date()} → {df.index[te[-1]].date()}] ({len(te)}d)")


# ═══════════════════════════════════════════════════════════════════════════════
# Custom Score metric
# ═══════════════════════════════════════════════════════════════════════════════
def compute_equity_curve(signals: np.ndarray, daily_rets: np.ndarray,
                         entry_prices: np.ndarray,
                         initial_capital=INITIAL_CAPITAL,
                         contract_size=CONTRACT_SIZE,
                         commission=COMMISSION_RT):
    """
    signals:      +1 (long) or 0 (flat) per bar
    daily_rets:   log-returns from afternoon open to evening close
    entry_prices: FCPO afternoon open price per bar (MYR/MT)
    Returns equity curve array.
    """
    equity = np.empty(len(signals) + 1)
    equity[0] = initial_capital
    prev_signal = 0
    for i, (sig, ret, ep) in enumerate(zip(signals, daily_rets, entry_prices)):
        cap  = equity[i]
        lots = 1 if (sig == 1 and cap >= 5_000) else 0
        # P&L: enter at afternoon_open, exit at evening_close = open * exp(ret)
        pnl = lots * contract_size * ep * (np.exp(ret) - 1)
        txn_cost = commission if lots != prev_signal else 0
        equity[i+1] = cap + pnl - txn_cost
        prev_signal = lots
    if prev_signal > 0:
        equity[-1] -= commission
    return equity

def max_drawdown(equity):
    peak = np.maximum.accumulate(equity)
    dd   = (peak - equity) / peak
    return dd.max() * equity[0]   # in RM terms from initial capital

def custom_score(equity, years):
    if years <= 0:
        return -1e9
    md = max_drawdown(equity)
    score = equity[-1] / years / (md + INITIAL_CAPITAL)
    return score


# ═══════════════════════════════════════════════════════════════════════════════
# Optuna objective — evaluated via OOF across all folds
# ═══════════════════════════════════════════════════════════════════════════════
X_vals   = X.values
y_vals   = y.values
fwd_vals = fwd_ret.values

def objective(trial):
    params = {
        'objective':         'binary',
        'metric':            'binary_logloss',
        'verbosity':         -1,
        'n_jobs':            -1,
        'random_state':      RANDOM_STATE,
        'num_leaves':        trial.suggest_int('num_leaves',          16,  256),
        'max_depth':         trial.suggest_int('max_depth',            3,   12),
        'learning_rate':     trial.suggest_float('learning_rate',    0.005, 0.2, log=True),
        'n_estimators':      trial.suggest_int('n_estimators',        50,  600),
        'min_child_samples': trial.suggest_int('min_child_samples',   10,  100),
        'subsample':         trial.suggest_float('subsample',         0.5,  1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree',  0.5,  1.0),
        'reg_alpha':         trial.suggest_float('reg_alpha',        1e-4, 10.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda',       1e-4, 10.0, log=True),
    }

    oof_signals  = []
    oof_fwd_rets = []
    total_days   = 0

    for tr_idx, te_idx in splits:
        X_tr, y_tr = X_vals[tr_idx], y_vals[tr_idx]
        X_te        = X_vals[te_idx]
        r_te        = fwd_vals[te_idx]

        val_size = max(int(len(tr_idx) * 0.2), 30)
        tr_fit_idx = tr_idx[:-val_size]
        tr_val_idx = tr_idx[-val_size:]

        scaler = StandardScaler()
        X_tr_fit_s = scaler.fit_transform(X_vals[tr_fit_idx])
        X_tr_val_s = scaler.transform(X_vals[tr_val_idx])
        X_te_s     = scaler.transform(X_te)

        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr_fit_s, y_vals[tr_fit_idx],
                  eval_set=[(X_tr_val_s, y_vals[tr_val_idx])],
                  callbacks=[lgb.early_stopping(30, verbose=False),
                              lgb.log_evaluation(-1)])

        prob = model.predict_proba(X_te_s)[:, 1]
        sig  = (prob > 0.5).astype(int)

        oof_signals.append(sig)
        oof_fwd_rets.append(r_te)
        total_days += len(te_idx)

    # Concatenate OOF and compute Score
    signals_all  = np.concatenate(oof_signals)
    fwd_rets_all = np.concatenate(oof_fwd_rets)
    ep_all = np.concatenate([entry_price_vals[te_idx] for _, te_idx in splits[:len(oof_signals)]])
    equity = compute_equity_curve(signals_all, fwd_rets_all, ep_all)
    years  = total_days / 252
    score  = custom_score(equity, years)
    return score


# ═══════════════════════════════════════════════════════════════════════════════
# Run Optuna
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\nRunning Optuna ({N_OPTUNA_TRIALS} trials)...")
study = optuna.create_study(direction='maximize',
                             sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)

best_params = study.best_params
best_score  = study.best_value
print(f"\nBest Score:  {best_score:.6f}")
print(f"Best params: {json.dumps(best_params, indent=2)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Final OOF evaluation with best params
# ═══════════════════════════════════════════════════════════════════════════════
print("\nFinal OOF evaluation with best hyperparameters...")

final_params = {
    'objective':    'binary',
    'metric':       'binary_logloss',
    'verbosity':    -1,
    'n_jobs':       -1,
    'random_state': RANDOM_STATE,
    **best_params,
}

oof_probs    = np.full(len(df), np.nan)
oof_signals  = np.full(len(df), np.nan)
fold_models  = []
fold_scalers = []
shap_values_list = []
shap_X_list      = []

for fold_i, (tr_idx, te_idx) in enumerate(splits):
    X_tr, y_tr = X_vals[tr_idx], y_vals[tr_idx]
    X_te, y_te = X_vals[te_idx], y_vals[te_idx]

    val_size = max(int(len(tr_idx) * 0.2), 30)
    tr_fit_idx = tr_idx[:-val_size]
    tr_val_idx = tr_idx[-val_size:]

    scaler = StandardScaler()
    X_tr_fit_s = scaler.fit_transform(X_vals[tr_fit_idx])
    X_tr_val_s = scaler.transform(X_vals[tr_val_idx])
    X_te_s     = scaler.transform(X_te)

    model = lgb.LGBMClassifier(**final_params)
    model.fit(X_tr_fit_s, y_vals[tr_fit_idx],
              eval_set=[(X_tr_val_s, y_vals[tr_val_idx])],
              callbacks=[lgb.early_stopping(30, verbose=False),
                          lgb.log_evaluation(-1)])

    prob = model.predict_proba(X_te_s)[:, 1]
    oof_probs[te_idx]   = prob
    oof_signals[te_idx] = (prob > 0.5).astype(float)

    fold_models.append(model)
    fold_scalers.append(scaler)

    # SHAP on test fold
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_te_s)
    if isinstance(sv, list):
        sv = sv[1]   # class 1 (long)
    shap_values_list.append(sv)
    shap_X_list.append(X_te_s)

    acc = accuracy_score(y_te, (prob > 0.5).astype(int))
    auc = roc_auc_score(y_te, prob)
    ll  = log_loss(y_te, prob)
    print(f"  Fold {fold_i+1}: Acc={acc:.4f}  AUC={auc:.4f}  LogLoss={ll:.4f}")

# ── Overall OOF metrics ───────────────────────────────────────────────────────
mask = ~np.isnan(oof_probs)
oof_acc = accuracy_score(y_vals[mask], (oof_probs[mask] > 0.5).astype(int))
oof_auc = roc_auc_score(y_vals[mask], oof_probs[mask])
oof_ll  = log_loss(y_vals[mask], oof_probs[mask])
print(f"\nOOF Overall: Acc={oof_acc:.4f}  AUC={oof_auc:.4f}  LogLoss={oof_ll:.4f}")

# ── OOF Custom Score ──────────────────────────────────────────────────────────
te_idx_all       = np.where(mask)[0]
sig_all          = oof_signals[te_idx_all]
ret_all          = fwd_vals[te_idx_all]
ep_all_final     = entry_price_vals[te_idx_all]
equity_all       = compute_equity_curve(sig_all, ret_all, ep_all_final)
years_all        = mask.sum() / 252
oof_score        = custom_score(equity_all, years_all)
oof_md           = max_drawdown(equity_all)

print(f"\nOOF Backtest Summary:")
print(f"  Initial Capital:  RM {INITIAL_CAPITAL:,.0f}")
print(f"  Final Equity:     RM {equity_all[-1]:,.0f}")
print(f"  Total Return:     {(equity_all[-1]/equity_all[0] - 1)*100:.1f}%")
print(f"  Max Drawdown:     RM {oof_md:,.0f}")
print(f"  Years:            {years_all:.2f}")
print(f"  Custom Score:     {oof_score:.6f}")

# Save results
results = {
    'best_params':             best_params,
    'best_optuna_score':       best_score,
    'oof_accuracy':            oof_acc,
    'oof_auc':                 oof_auc,
    'oof_logloss':             oof_ll,
    'oof_score':               oof_score,
    'oof_max_drawdown_rm':     float(oof_md),
    'oof_final_equity_rm':     float(equity_all[-1]),
    'oof_years':               years_all,
}
with open('/home/user/claudestrat/data/phase3_v3_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save last fold model + scaler for Phase 4/5
with open('/home/user/claudestrat/data/model_last_fold_v3.pkl', 'wb') as f:
    pickle.dump({'model': fold_models[-1], 'scaler': fold_scalers[-1],
                 'feature_cols': feat_cols}, f)

# Save OOF signals and equity for Phase 4/5
oof_df = pd.DataFrame({
    'date':          df.index[te_idx_all],
    'ret_afeve':     ret_all,
    'signal':        sig_all,
    'prob_long':     oof_probs[te_idx_all],
    'equity':        equity_all[1:],
    'afternoon_open': entry_price_vals[te_idx_all],
})
oof_df.to_csv('/home/user/claudestrat/data/oof_signals_v3.csv', index=False)
print("\nSaved model, results, OOF signals.")


# ═══════════════════════════════════════════════════════════════════════════════
# SHAP Analysis
# ═══════════════════════════════════════════════════════════════════════════════
shap_all = np.vstack(shap_values_list)
X_shap   = np.vstack(shap_X_list)

print("\nGenerating SHAP and diagnostic plots...")

mean_abs_shap = pd.Series(np.abs(shap_all).mean(axis=0), index=feat_cols).sort_values(ascending=False)
print("\nMean |SHAP| by feature:")
print(mean_abs_shap.to_string())
print("\nTop 5 features by SHAP:")
for rank, (feat, val) in enumerate(mean_abs_shap.head(5).items(), 1):
    print(f"  {rank}. {feat}: {val:.4f}")

# ── Plot 1: SHAP bar chart ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
mean_abs_shap.sort_values().plot(kind='barh', color='steelblue', ax=ax)
ax.set_title('Mean |SHAP| — Feature Importance v3 (OOF, Intraday Strategy)')
ax.set_xlabel('Mean |SHAP value|')
plt.tight_layout()
plt.savefig(f'{OUT}/01_shap_bar.png', dpi=150)
plt.close()

# ── Plot 2: SHAP beeswarm ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_all, X_shap, feature_names=feat_cols, show=False, plot_size=None)
plt.title('SHAP Summary v3 (OOF test sets — Intraday Strategy)')
plt.tight_layout()
plt.savefig(f'{OUT}/02_shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Plot 3: OOF equity curve ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

oof_dates = pd.to_datetime(oof_df['date'])
axes[0].plot(oof_dates, equity_all[1:], lw=1.5, color='steelblue', label='Strategy')
axes[0].axhline(INITIAL_CAPITAL, color='grey', lw=0.8, linestyle='--', label='Initial capital')
axes[0].set_ylabel('Equity (RM)')
axes[0].set_title(f'OOF Equity Curve v3 (15:00→23:30)  |  Score={oof_score:.4f}  |  MaxDD=RM{oof_md:,.0f}')
axes[0].legend()

running_max = np.maximum.accumulate(equity_all[1:])
dd_pct = (running_max - equity_all[1:]) / running_max * 100
axes[1].fill_between(oof_dates, -dd_pct, 0, color='red', alpha=0.4, label='Drawdown')
axes[1].set_ylabel('Drawdown (%)')
axes[1].set_xlabel('Date')
axes[1].legend()

plt.tight_layout()
plt.savefig(f'{OUT}/03_oof_equity.png', dpi=150)
plt.close()

# ── Plot 4: Optuna optimisation history ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
trial_scores = [t.value for t in study.trials if t.value is not None]
axes[0].plot(trial_scores, lw=0.8, color='dimgray', alpha=0.7)
running_best = pd.Series(trial_scores).cummax()
axes[0].plot(running_best, lw=1.5, color='steelblue', label='Best so far')
axes[0].set_title('Optuna: Score per Trial')
axes[0].set_xlabel('Trial')
axes[0].set_ylabel('Score')
axes[0].legend()

try:
    param_imp = optuna.importance.get_param_importances(study)
    pd.Series(param_imp).sort_values().plot(kind='barh', ax=axes[1], color='steelblue')
    axes[1].set_title('Optuna: Hyperparameter Importance')
    axes[1].set_xlabel('Importance')
except Exception:
    axes[1].text(0.3, 0.5, 'Param importance\nnot available', transform=axes[1].transAxes)

plt.tight_layout()
plt.savefig(f'{OUT}/04_optuna_history.png', dpi=150)
plt.close()

# ── Plot 5: Signal quality — return distribution by signal ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
long_rets  = ret_all[sig_all == 1] * 100
short_rets = ret_all[sig_all == 0] * 100
bins = np.linspace(-8, 8, 60)
axes[0].hist(long_rets,  bins=bins, alpha=0.6, color='green',  label=f'Long  n={len(long_rets)}',  density=True)
axes[0].hist(short_rets, bins=bins, alpha=0.6, color='salmon', label=f'Flat  n={len(short_rets)}', density=True)
axes[0].axvline(0, color='black', lw=0.8)
axes[0].set_title('Afeve Return Distribution by Signal')
axes[0].set_xlabel('Return (%)')
axes[0].legend()

prob_bins = np.linspace(0, 1, 11)
oof_df['prob_bin'] = pd.cut(oof_df['prob_long'], bins=prob_bins)
cal = oof_df.groupby('prob_bin', observed=True)['ret_afeve'].mean() * 100
cal.plot(kind='bar', ax=axes[1], color='steelblue', edgecolor='none')
axes[1].axhline(0, color='black', lw=0.8)
axes[1].set_title('Mean Afeve Return by Predicted Probability Decile')
axes[1].set_xlabel('P(Long) bin')
axes[1].set_ylabel('Mean Return (%)')
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(f'{OUT}/05_signal_quality.png', dpi=150)
plt.close()

print(f"\nAll plots saved to {OUT}/")
print("\nPhase 3 v3 complete. Ready for Phase 4/5.")
