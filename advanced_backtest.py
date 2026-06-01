#!/usr/bin/env python3
"""
Advanced Backtest: Japanese stocks — MACD + SMA strategy
Features: stop loss, grid-search parameter optimisation, train/test split
"""

import argparse
import os
import sys
from itertools import product

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Constants ──────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 1_000_000   # 100万円
LOT             = 100          # 単元株
COMMISSION      = 0.0005       # 片道0.05%

MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9

# Grid search candidates
SMA_PERIODS     = [20, 25, 30]
STOP_LOSS_RATES = [-0.02, -0.03, -0.05]   # -2%, -3%, -5%
TEST_DAYS       = 252                      # ≈最後の1年分の営業日

# ── CLI ────────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser(description="Advanced Backtest for Japanese stocks")
ap.add_argument("--code", default="7203", help="銘柄コード（例: 7203）")
args = ap.parse_args()
CODE = args.code

# ── Load data ──────────────────────────────────────────────────────────────────
data_path = f"data/{CODE}.csv"
if not os.path.exists(data_path):
    sys.exit(f"Error: {data_path} が見つかりません。先に fetch_japan_data.py を実行してください。")

df_raw = pd.read_csv(data_path, index_col=0, parse_dates=True).sort_index()
df_raw = df_raw[df_raw["Volume"] > 0].copy()   # 実際の取引日のみ

if len(df_raw) < TEST_DAYS + 60:
    sys.exit("Error: データが不足しています（最低でも約1.3年分必要です）。")

# ── Train / Test split ─────────────────────────────────────────────────────────
split_idx = len(df_raw) - TEST_DAYS
df_train  = df_raw.iloc[:split_idx]
df_test   = df_raw.iloc[split_idx:]


# ── Indicator builder (compute on full df → slice later) ───────────────────────
def build_indicators(df_full: pd.DataFrame, sma_period: int) -> pd.DataFrame:
    c     = df_full["Close"]
    ema_f = c.ewm(span=MACD_FAST, adjust=False).mean()
    ema_s = c.ewm(span=MACD_SLOW, adjust=False).mean()
    macd  = ema_f - ema_s
    sig   = macd.ewm(span=MACD_SIG, adjust=False).mean()
    sma   = c.rolling(sma_period).mean()

    gc = (macd > sig) & (macd.shift(1) <= sig.shift(1))
    dc = (macd < sig) & (macd.shift(1) >= sig.shift(1))

    return pd.DataFrame({
        "close":        c,
        "open":         df_full["Open"],
        "sma":          sma,
        "golden_cross": gc,
        "dead_cross":   dc,
        "above_sma":    c > sma,
        "below_sma":    c < sma,
    })


# ── Backtest engine ────────────────────────────────────────────────────────────
def run_backtest(ind: pd.DataFrame, stop_loss_pct: float) -> dict:
    """
    ind           : indicator DataFrame for the target period
    stop_loss_pct : negative float  e.g. -0.03 → -3%
    Returns a dict with metrics and asset_series (pd.Series).
    """
    n     = len(ind)
    close = ind["close"].values.astype(float)
    open_ = ind["open"].values.astype(float)
    dates = ind.index
    gc    = ind["golden_cross"].values
    dc    = ind["dead_cross"].values
    abv   = ind["above_sma"].values
    blw   = ind["below_sma"].values
    sma_v = ind["sma"].values

    cash        = float(INITIAL_CAPITAL)
    shares      = 0
    entry_price = 0.0
    asset_arr   = np.empty(n)
    trades      = []
    pending     = None    # 'buy' | 'sell' | None
    cur_trade   = None

    for i in range(n):

        # ── execute yesterday's pending order at today's open ──────────────────
        if pending == "sell" and shares > 0:
            sp         = open_[i]
            exit_proc  = sp * shares * (1.0 - COMMISSION)
            entry_cost = entry_price * shares * (1.0 + COMMISSION)
            cash      += exit_proc
            cur_trade.update(exit_date=dates[i], exit_price=sp,
                             profit=exit_proc - entry_cost)
            trades.append(cur_trade)
            cur_trade, shares, entry_price = None, 0, 0.0

        elif pending == "buy" and shares == 0:
            bp = open_[i]
            if bp > 0 and not np.isnan(bp):
                lots = int(cash / (bp * LOT * (1.0 + COMMISSION)))
                qty  = lots * LOT
                if qty > 0:
                    cash       -= bp * qty * (1.0 + COMMISSION)
                    shares      = qty
                    entry_price = bp
                    cur_trade   = dict(entry_date=dates[i], entry_price=bp, shares=qty)

        pending = None

        # ── mark-to-market at close ────────────────────────────────────────────
        asset_arr[i] = cash + shares * close[i]

        # ── signal check ──────────────────────────────────────────────────────
        if shares > 0:
            if close[i] <= entry_price * (1.0 + stop_loss_pct):  # ストップロス
                pending = "sell"
            elif dc[i] or blw[i]:                                 # デッドクロス or SMA割れ
                pending = "sell"
        else:
            if not np.isnan(sma_v[i]) and abv[i] and gc[i]:      # ゴールデンクロス + SMA上
                pending = "buy"

    # ── force-liquidate remaining position at final close ─────────────────────
    if shares > 0 and cur_trade is not None:
        sp         = close[-1]
        exit_proc  = sp * shares * (1.0 - COMMISSION)
        entry_cost = entry_price * shares * (1.0 + COMMISSION)
        cash      += exit_proc
        cur_trade.update(exit_date=dates[-1], exit_price=sp,
                         profit=exit_proc - entry_cost)
        trades.append(cur_trade)
        asset_arr[-1] = cash

    asset_s = pd.Series(asset_arr, index=dates, name="asset")

    # ── metrics ───────────────────────────────────────────────────────────────
    n_tr = len(trades)
    if n_tr > 0:
        profits      = [t["profit"] for t in trades]
        wins         = [p for p in profits if p > 0]
        losses       = [p for p in profits if p <= 0]
        win_rate     = len(wins) / n_tr * 100.0
        gross_profit = sum(wins) if wins else 0.0
        gross_loss   = abs(sum(losses)) if losses else 0.0
        pf           = (gross_profit / gross_loss
                        if gross_loss > 0
                        else (float("inf") if gross_profit > 0 else 0.0))
    else:
        win_rate = pf = 0.0

    rm     = asset_s.cummax()
    max_dd = float(((asset_s - rm) / rm * 100).min())
    final  = float(asset_s.iloc[-1])
    ret    = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100.0

    return dict(final_asset=final, total_return=ret, total_trades=n_tr,
                win_rate=win_rate, max_drawdown=max_dd, profit_factor=pf,
                asset_series=asset_s)


# ── Header ────────────────────────────────────────────────────────────────────
print(f"\n{'═'*64}")
print(f"  Advanced Backtest: {CODE}  (MACD{MACD_FAST}/{MACD_SLOW}/{MACD_SIG} + SMA + StopLoss)")
print(f"{'═'*64}")
print(f"  データ期間 : {df_raw.index[0].date()} ～ {df_raw.index[-1].date()}  ({len(df_raw)} 営業日)")
print(f"  訓練期間   : {df_train.index[0].date()} ～ {df_train.index[-1].date()}  ({len(df_train)} 営業日)")
print(f"  テスト期間 : {df_test.index[0].date()} ～ {df_test.index[-1].date()}  ({len(df_test)} 営業日)")

# ── Grid search on training data ───────────────────────────────────────────────
print(f"\n{'─'*64}")
print(f"  グリッドサーチ（訓練期間）")
print(f"{'─'*64}")
print(f"  {'SMA':>5} {'StopLoss':>10} {'最終資産(円)':>17} {'リターン':>9} {'取引数':>7} {'勝率':>7}")
print(f"  {'─'*5} {'─'*10} {'─'*17} {'─'*9} {'─'*7} {'─'*7}")

best_asset = -float("inf")
best_params, best_train_result = None, None

for sma, sl in product(SMA_PERIODS, STOP_LOSS_RATES):
    ind_full  = build_indicators(df_raw, sma)
    ind_train = ind_full.iloc[:split_idx]
    r         = run_backtest(ind_train, sl)

    is_best = r["final_asset"] > best_asset
    if is_best:
        best_asset, best_params, best_train_result = r["final_asset"], (sma, sl), r
    mark = " ◀" if is_best else ""

    print(f"  {sma:>5} {sl*100:>9.0f}% {r['final_asset']:>17,.0f} "
          f"{r['total_return']:>+8.1f}% {r['total_trades']:>7} "
          f"{r['win_rate']:>6.1f}%{mark}")

best_sma, best_sl = best_params
print(f"\n  ★ 最適パラメータ : SMA = {best_sma} 日,  ストップロス = {best_sl*100:.0f}%")
print(f"  　訓練期間 最終資産 : {best_asset:,.0f} 円")

# ── Out-of-sample test ────────────────────────────────────────────────────────
ind_full = build_indicators(df_raw, best_sma)
ind_test = ind_full.iloc[split_idx:]
result   = run_backtest(ind_test, best_sl)

pf_str = f"{result['profit_factor']:.2f}" if result["profit_factor"] != float("inf") else "∞"

print(f"\n{'═'*64}")
print(f"  テスト期間 結果  (SMA={best_sma}, StopLoss={best_sl*100:.0f}%)")
print(f"{'─'*64}")
print(f"  初期資金                   : {INITIAL_CAPITAL:>14,.0f} 円")
print(f"  最終資産額                 : {result['final_asset']:>14,.0f} 円")
print(f"  総利益率                   : {result['total_return']:>+13.2f} %")
print(f"  総トレード回数             : {result['total_trades']:>13} 回")
print(f"  勝率                       : {result['win_rate']:>13.1f} %")
print(f"  最大ドローダウン           : {result['max_drawdown']:>+13.2f} %")
print(f"  プロフィットファクター     : {pf_str:>13}")
print(f"{'═'*64}\n")

# ── Plot asset curve ───────────────────────────────────────────────────────────
os.makedirs("output", exist_ok=True)

asset_s    = result["asset_series"]
plot_dates = (asset_s.index.tz_convert(None)
              if asset_s.index.tz is not None
              else asset_s.index)

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(plot_dates, asset_s.values, color="steelblue", linewidth=1.4,
        label="Asset Value")
ax.axhline(INITIAL_CAPITAL, color="dimgray", linestyle="--", linewidth=0.9,
           label=f"Initial Capital  ¥{INITIAL_CAPITAL:,}")

ax.fill_between(plot_dates, asset_s.values, INITIAL_CAPITAL,
                where=(asset_s.values >= INITIAL_CAPITAL),
                alpha=0.20, color="green", label="Profit zone")
ax.fill_between(plot_dates, asset_s.values, INITIAL_CAPITAL,
                where=(asset_s.values <  INITIAL_CAPITAL),
                alpha=0.20, color="red",   label="Loss zone")

title = (f"Asset Curve — {CODE}  "
         f"[Test: {plot_dates[0].date()} ～ {plot_dates[-1].date()}]\n"
         f"SMA={best_sma}d, StopLoss={best_sl*100:.0f}%  |  "
         f"Return={result['total_return']:+.2f}%  |  "
         f"MaxDD={result['max_drawdown']:.2f}%  |  "
         f"Trades={result['total_trades']}  |  "
         f"WinRate={result['win_rate']:.1f}%")
ax.set_title(title, fontsize=10)
ax.set_xlabel("Date")
ax.set_ylabel("Asset (JPY)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"¥{x:,.0f}"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=45)
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.3)
plt.tight_layout()

out_path = f"output/asset_curve_{CODE}.png"
plt.savefig(out_path, dpi=150)
print(f"  グラフ保存 → {out_path}")
