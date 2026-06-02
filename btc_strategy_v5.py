"""
BTC-USD Strategy v5 — True Alpha Validation

これまでの全バージョンの根本問題:
  「2024-2026の強気相場だけでテスト」→ 本当に機能するか不明

v5で解決すること:
  1. 5年日足（2021 ATH → 2022 暴落 -70% → 2023-26 回復）
  2. バイ&ホールド / DCA との公正比較（真の付加価値）
  3. Kelly基準ポジションサイジング（数学的最適ベット額）
  4. 年次ブレークダウン（相場環境別の勝ち負け）
  5. 現在のシグナル（今日から使える判断基準）
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime

INITIAL_CAPITAL = 500_000
CSV_DAILY       = "/Users/hiroseren/btc_usd_daily_5y.csv"

# ━━ データ取得 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fetch_daily():
    if os.path.exists(CSV_DAILY):
        print(f"[DATA] キャッシュ読込: {CSV_DAILY}")
        df = pd.read_csv(CSV_DAILY, index_col=0, parse_dates=True)
    else:
        print("[DATA] BTC-USD 日足 5年分を取得中...")
        df = yf.download("BTC-USD", period="5y", interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            raise RuntimeError("データ取得失敗")
        df.to_csv(CSV_DAILY)
        print(f"[DATA] 保存完了: {len(df):,}行")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.dropna(subset=["Close"])
    print(f"[DATA] 期間: {df.index[0].date()} ～ {df.index[-1].date()}  ({len(df):,}バー)")
    return df

# ━━ テクニカル指標 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def add_indicators(df):
    c = df["Close"]
    h, l = df["High"], df["Low"]

    df["ema50"]  = c.ewm(span=50,  adjust=False).mean()
    df["ema100"] = c.ewm(span=100, adjust=False).mean()
    df["ema200"] = c.ewm(span=200, adjust=False).mean()

    # RSI
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    # ATR
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # ボラティリティ（20日実現ボラ、年率）
    df["rv20"] = c.pct_change().rolling(20).std() * np.sqrt(365)

    # EMA200の傾き（トレンド方向）
    df["ema200_slope"] = df["ema200"] - df["ema200"].shift(10)

    return df

# ━━ 戦略シグナル生成 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_signals(df):
    """
    エントリー条件（すべて満たす）:
      - 終値 > EMA200（上昇トレンド確認）
      - EMA200の傾き > 0（トレンド加速中）
      - RSIが56を下から上抜け（モメンタム発生）
      - 前日の実現ボラ < 0.8（異常ボラ回避）

    エグジット条件（いずれか）:
      - RSIが40を上から下抜け（モメンタム消滅）
      - 終値がEMA100を下抜け（中期トレンド割れ）
    """
    c      = df["Close"]
    rsi    = df["rsi"]
    e100   = df["ema100"]
    e200   = df["ema200"]
    slope  = df["ema200_slope"]
    rv     = df["rv20"]

    trend_ok   = (c > e200) & (slope > 0)
    vol_ok     = rv < 0.9
    rsi_cross_up   = (rsi.shift(1) < 56) & (rsi >= 56)
    rsi_cross_down = (rsi.shift(1) > 40) & (rsi <= 40)
    price_exit = c < e100

    buy  = rsi_cross_up  & trend_ok & vol_ok
    sell = rsi_cross_down | price_exit

    return buy, sell

# ━━ Kelly基準ポジションサイジング ━━━━━━━━━━━━━━━━━━━━━━━

class KellySizer:
    """
    運用中にトレード結果を蓄積し、Kelly基準で最適賭け率を動的計算。
    最初の15トレードは保守的な固定2%リスクルール。
    以降はHalf-Kellyに移行（上限10%/トレード）。
    """
    def __init__(self):
        self.wins  = []
        self.losses= []

    def record(self, pnl_pct):
        if pnl_pct > 0:
            self.wins.append(pnl_pct)
        else:
            self.losses.append(abs(pnl_pct))

    def get_fraction(self):
        n = len(self.wins) + len(self.losses)
        if n < 15:
            return 0.02  # 序盤: 固定2%リスク

        W = len(self.wins) / n
        R = np.mean(self.wins) / np.mean(self.losses) if self.losses else 3.0
        kelly = W - (1 - W) / R         # フル・Kelly
        half_kelly = max(0, kelly / 2)   # Half-Kellyで安全マージン
        return min(half_kelly, 0.10)     # 上限10%

    def stats(self):
        n = len(self.wins) + len(self.losses)
        if n == 0:
            return {}
        W = len(self.wins) / n
        avg_w = np.mean(self.wins)  if self.wins   else 0
        avg_l = np.mean(self.losses)if self.losses else 0
        R     = avg_w / avg_l if avg_l > 0 else 99
        kelly = W - (1 - W) / R if R > 0 else 0
        return {"win_rate": W, "R": R, "kelly": kelly, "n": n}

# ━━ バックテストエンジン ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def backtest(df, buy_sig, sell_sig):
    atr_    = df["atr"].values
    close   = df["Close"].values
    dates   = df.index

    capital   = float(INITIAL_CAPITAL)
    pos       = 0.0
    entry_px  = 0.0
    stop_px   = 0.0
    trail_hi  = 0.0
    tp_done   = False
    equity    = []
    trades    = []   # (date, type, pnl, pnl_pct)
    sizer     = KellySizer()

    for i in range(len(df)):
        px    = float(close[i])
        atr_v = float(atr_[i]) if not np.isnan(atr_[i]) else px * 0.02
        eq    = capital + pos * px
        equity.append(eq)

        # ── トレーリングストップ更新
        if pos > 0:
            trail_hi = max(trail_hi, px)
            new_sl   = trail_hi - atr_v * 3.0
            stop_px  = max(stop_px, new_sl)

        # ── ストップ判定
        if pos > 0 and px <= stop_px:
            pnl     = (px - entry_px) * pos
            pnl_pct = (px - entry_px) / entry_px
            capital += pos * px
            sizer.record(pnl_pct)
            trades.append((dates[i], "stop", round(pnl, 0), round(pnl_pct*100, 2)))
            pos = 0.0; tp_done = False

        # ── 部分利食い（ATR×2.5 到達で 40% 確定）
        if pos > 0 and not tp_done and px >= entry_px + atr_v * 2.5:
            cut     = pos * 0.40
            capital += cut * px
            pos    -= cut
            tp_done = True

        # ── 買いシグナル
        if pos == 0 and bool(buy_sig.iloc[i]):
            sl   = px - atr_v * 3.0
            risk = px - sl
            if risk <= 0:
                continue
            frac = sizer.get_fraction()
            # Kelly分を直接リスク額に変換
            size = min(
                (capital * frac) / risk,   # Kelly リスクベース
                capital / px * 0.98        # 資金上限
            )
            if size > 0:
                pos       = size
                entry_px  = px
                stop_px   = sl
                trail_hi  = px
                tp_done   = False
                capital  -= size * px

        # ── 売りシグナル
        elif pos > 0 and bool(sell_sig.iloc[i]):
            pnl     = (px - entry_px) * pos
            pnl_pct = (px - entry_px) / entry_px
            capital += pos * px
            sizer.record(pnl_pct)
            trades.append((dates[i], "signal", round(pnl, 0), round(pnl_pct*100, 2)))
            pos = 0.0; tp_done = False

    # 最終決済
    if pos > 0:
        pnl     = (float(close[-1]) - entry_px) * pos
        pnl_pct = (float(close[-1]) - entry_px) / entry_px
        capital += pos * float(close[-1])
        trades.append((dates[-1], "final", round(pnl, 0), round(pnl_pct*100, 2)))

    eq_arr = np.array(equity)
    peak   = np.maximum.accumulate(eq_arr)
    dd_arr = (eq_arr - peak) / peak * 100
    max_dd = float(dd_arr.min())

    bar_ret = np.diff(eq_arr) / eq_arr[:-1]
    sharpe  = float(bar_ret.mean() / bar_ret.std() * np.sqrt(365)) if bar_ret.std() > 0 else 0

    ret_pct = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    pnls    = [t[2] for t in trades]
    wins_n  = sum(1 for p in pnls if p > 0)
    wr      = wins_n / len(pnls) * 100 if pnls else 0

    wins_v  = [p for p in pnls if p > 0]
    loss_v  = [p for p in pnls if p < 0]
    pf = sum(wins_v) / abs(sum(loss_v)) if loss_v else 99

    return {
        "final": round(capital, 0),
        "ret":   round(ret_pct, 2),
        "wr":    round(wr, 1),
        "dd":    round(max_dd, 2),
        "sharpe":round(sharpe, 2),
        "pf":    round(pf, 2),
        "n":     len(pnls),
        "equity": eq_arr,
        "trades": trades,
        "sizer":  sizer,
    }

# ━━ ベンチマーク ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def buy_and_hold(df):
    c      = df["Close"].values
    units  = INITIAL_CAPITAL / float(c[0])
    equity = units * c
    eq_arr = np.array(equity)
    peak   = np.maximum.accumulate(eq_arr)
    max_dd = float(((eq_arr - peak) / peak).min() * 100)
    ret    = (float(c[-1]) - float(c[0])) / float(c[0]) * 100
    bar_ret= np.diff(eq_arr) / eq_arr[:-1]
    sharpe = float(bar_ret.mean() / bar_ret.std() * np.sqrt(365)) if bar_ret.std()>0 else 0
    return {"final": round(units*float(c[-1]), 0), "ret": round(ret, 2),
            "dd": round(max_dd, 2), "sharpe": round(sharpe, 2), "equity": eq_arr}

def dca_monthly(df):
    """毎月一定額を買い続ける（ドルコスト平均）"""
    monthly = df.resample("ME").last()
    monthly_budget = INITIAL_CAPITAL / len(monthly)
    units, capital_used = 0.0, 0.0
    for _, row in monthly.iterrows():
        units         += monthly_budget / float(row["Close"])
        capital_used  += monthly_budget
    final  = units * float(df["Close"].iloc[-1])
    ret    = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    # ドローダウン概算（月次）
    px = monthly["Close"].values
    cum = np.array([INITIAL_CAPITAL/len(monthly) * i / px[min(i, len(px)-1)]
                    * px[-1] for i in range(1, len(px)+1)])
    return {"final": round(final, 0), "ret": round(ret, 2),
            "n_buys": len(monthly)}

# ━━ 年次ブレークダウン ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def yearly_breakdown(df, strategy_equity, bah_equity):
    rows = []
    for year in range(df.index[0].year, df.index[-1].year + 1):
        mask = df.index.year == year
        if mask.sum() < 20:
            continue
        idx  = np.where(mask)[0]
        s_eq = strategy_equity[mask]
        b_eq = bah_equity[mask]

        s_ret = (s_eq[-1] - s_eq[0]) / s_eq[0] * 100
        b_ret = (b_eq[-1] - b_eq[0]) / b_eq[0] * 100

        # 年内最大ドローダウン
        pk    = np.maximum.accumulate(s_eq)
        s_dd  = float(((s_eq - pk) / pk).min() * 100)

        # 年内取引数
        n_t   = sum(1 for t in strategy_equity if True)  # 簡略

        rows.append({
            "year": year,
            "s_ret": round(s_ret, 1),
            "b_ret": round(b_ret, 1),
            "s_dd":  round(s_dd, 1),
            "alpha": round(s_ret - b_ret, 1),
        })
    return rows

# ━━ 現在のシグナル（今日） ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def current_signal(df, buy_sig, sell_sig):
    last   = df.iloc[-1]
    px     = float(last["Close"])
    rsi_v  = float(last["rsi"]) if not np.isnan(last["rsi"]) else 0
    e200   = float(last["ema200"])
    e100   = float(last["ema100"])
    e50    = float(last["ema50"])
    slope  = float(last["ema200_slope"]) if not np.isnan(last["ema200_slope"]) else 0
    rv     = float(last["rv20"]) if not np.isnan(last["rv20"]) else 0
    atr_v  = float(last["atr"]) if not np.isnan(last["atr"]) else 0

    # 買いシグナル条件を個別チェック
    cond_trend = px > e200 and slope > 0
    cond_vol   = rv < 0.9
    cond_rsi   = bool(buy_sig.iloc[-1])

    # 現在のポジション推奨
    is_buy_zone = cond_trend and cond_vol
    is_rsi_trig = cond_rsi

    if is_buy_zone and is_rsi_trig:
        action = "🟢 買いシグナル発生"
    elif is_buy_zone:
        action = "🟡 待機（トレンド良好、RSIエントリー待ち）"
    else:
        action = "🔴 ノーポジション推奨（トレンド条件未達）"

    # RSIが買い閾値まであとどれくらいか
    rsi_to_buy = 56 - rsi_v

    return {
        "date":      df.index[-1].date(),
        "price":     px,
        "ema200":    e200,
        "ema100":    e100,
        "ema50":     e50,
        "rsi":       rsi_v,
        "rv":        rv,
        "atr":       atr_v,
        "cond_trend":cond_trend,
        "cond_vol":  cond_vol,
        "action":    action,
        "rsi_to_buy":rsi_to_buy,
        "sl_level":  px - atr_v * 3.0,
    }

# ━━ レポート出力 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_report(df, result, bah, dca_res, yearly, csig):
    target30 = INITIAL_CAPITAL * 1.30

    print("\n" + "═"*65)
    print("  BTC-USD v5 真のアルファ検証（日足 5年）")
    print("═"*65)

    # ── 1. 5年間比較
    print("\n【1】5年間パフォーマンス比較（初期資金 50万円）")
    print(f"  {'戦略':<20} {'最終資産':>12} {'収益率':>8} {'最大DD':>8} {'Sharpe':>7}")
    print("  " + "─"*58)

    rows = [
        ("バイ&ホールド",  bah["final"],    bah["ret"],    bah["dd"],    bah["sharpe"]),
        ("v5戦略(RSI+Kelly)", result["final"], result["ret"], result["dd"], result["sharpe"]),
    ]
    for name, fin, ret, dd, sh in rows:
        arrow = "▲" if ret >= 0 else "▼"
        print(f"  {name:<20} {fin:>12,.0f}円 "
              f"{arrow}{abs(ret):>6.1f}%  {dd:>6.1f}%  {sh:>6.2f}")

    print(f"\n  DCA（月次積立）      最終資産: {dca_res['final']:>12,.0f}円  "
          f"({dca_res['ret']:+.1f}%,  {dca_res['n_buys']}回積立)")

    # リスク調整後の優位性
    print("\n  ─ リスク調整後の評価 ───────────────────────────")
    if result["sharpe"] > bah["sharpe"]:
        diff = result["sharpe"] - bah["sharpe"]
        print(f"  → 戦略はバイ&ホールドより Sharpe が {diff:.2f} 高い")
        print(f"    同じリターンを目指すならドローダウンを大幅に抑えられる")
    else:
        print(f"  → バイ&ホールドの Sharpe が戦略を上回る（強気相場での課題）")

    dd_improve = abs(bah["dd"]) - abs(result["dd"])
    print(f"  → 最大ドローダウン削減: {bah['dd']:.1f}% → {result['dd']:.1f}%  "
          f"（{dd_improve:.1f}pt 改善）")

    # ── 2. 年次ブレークダウン
    print("\n【2】年次ブレークダウン（相場環境別）")
    print(f"  {'年':>4}  {'戦略':>7}  {'B&H':>7}  {'最大DD':>7}  {'αAlpha':>7}  評価")
    print("  " + "─"*52)
    for r in yearly:
        note = ""
        if r["b_ret"] < -30:
            note = "← 暴落年：防御力テスト"
        elif r["alpha"] > 5:
            note = "← 戦略優位"
        elif r["alpha"] < -20:
            note = "← B&H優位（強気相場）"
        arrow_s = "▲" if r["s_ret"] >= 0 else "▼"
        arrow_b = "▲" if r["b_ret"] >= 0 else "▼"
        print(f"  {r['year']:>4}  "
              f"{arrow_s}{abs(r['s_ret']):>5.1f}%  "
              f"{arrow_b}{abs(r['b_ret']):>5.1f}%  "
              f"{r['s_dd']:>6.1f}%  "
              f"{r['alpha']:>+6.1f}%  {note}")

    # ── 3. Kelly基準
    ks = result["sizer"].stats()
    k_frac = result["sizer"].get_fraction()
    print("\n【3】Kelly基準ポジションサイジング")
    if ks:
        print(f"  勝率              : {ks['win_rate']*100:.1f}%")
        print(f"  平均利益/損失比 R : {ks['R']:.2f}")
        print(f"  フル・Kelly 分率  : {ks['kelly']*100:.1f}%")
        print(f"  採用 Half-Kelly   : {k_frac*100:.1f}%  （安全マージン50%）")
        rec_jpy = INITIAL_CAPITAL * k_frac
        print(f"  50万円での推奨リスク額/トレード: {rec_jpy:,.0f}円")
    else:
        print("  （取引データ不足）")

    # ── 4. 取引統計
    pnls   = [t[2] for t in result["trades"]]
    wins_  = [p for p in pnls if p > 0]
    loss_  = [p for p in pnls if p < 0]
    print("\n【4】取引統計")
    print(f"  総取引数          : {result['n']:>6}回")
    print(f"  勝率              : {result['wr']:>5.1f}%")
    print(f"  プロフィットF     : {result['pf']:>5.2f}  （目安 >1.5）")
    if wins_:
        print(f"  平均利益/取引     : {np.mean(wins_):>+8,.0f}円")
    if loss_:
        print(f"  平均損失/取引     : {np.mean(loss_):>+8,.0f}円")
    if pnls:
        consec_loss = _max_consec_losses(pnls)
        print(f"  最大連続負け数    : {consec_loss:>5}回")

    # ── 5. 現在のシグナル
    print("\n【5】現在のシグナル（" + str(csig["date"]) + "）")
    print(f"  BTC価格    : ${csig['price']:>10,.0f}")
    print(f"  EMA200     : ${csig['ema200']:>10,.0f}  "
          f"{'↑ 価格>EMA200' if csig['cond_trend'] else '↓ 価格<EMA200'}")
    print(f"  EMA100     : ${csig['ema100']:>10,.0f}")
    print(f"  RSI(14)    : {csig['rsi']:>10.1f}  "
          f"（買い閾値56まで {'あと{:.1f}pt'.format(csig['rsi_to_buy']) if csig['rsi_to_buy']>0 else '✓ 超過中'}）")
    print(f"  実現ボラ   : {csig['rv']*100:>9.1f}%  "
          f"{'正常' if csig['cond_vol'] else '高ボラ注意'}")
    print(f"  ATR(14)    : ${csig['atr']:>10,.0f}")
    print(f"\n  ▶ 判断: {csig['action']}")
    if csig["cond_trend"]:
        print(f"  ▶ エントリー時の推奨 SL: ${csig['sl_level']:>10,.0f}  "
              f"（ATR×3 = ${csig['atr']*3:,.0f}）")

    # ── 6. バージョン進化まとめ
    print("\n【6】v1 → v5 完全進化サマリー")
    evol = [
        ("v1 RSIモメンタム",         "+13.6%", "-34.8%", "1時間足 2年"),
        ("v2 +200EMAフィルター",      "+49.2%", "-10.3%", "4時間足 2年"),
        ("v3 グリッド最適化",         "+51.8%", " -7.6%", "4時間足 2年"),
        ("v4 安定領域+適応型",        "+45.5%", " -5.3%", "4時間足 2年"),
        (f"v5 真のアルファ検証",
         f"{result['ret']:+.1f}%", f"{result['dd']:.1f}%", "日足  5年"),
    ]
    print(f"  {'バージョン':<22} {'収益率':>8} {'MaxDD':>7}  カバー期間")
    print("  " + "─"*55)
    for v in evol:
        print(f"  {v[0]:<22} {v[1]:>8} {v[2]:>7}  {v[3]}")

    print("\n  ─ v5の意義 ──────────────────────────────────")
    print("  ・2022年の -70% 暴落を含む5年間で戦略を検証")
    print("  ・バイ&ホールドと比較し「真の付加価値」を測定")
    print("  ・Kelly基準で数学的に最適なベット額を算出")
    print("  ・今日から使える具体的シグナルを出力")
    print("═"*65)

def _max_consec_losses(pnls):
    max_cl, cur_cl = 0, 0
    for p in pnls:
        if p < 0:
            cur_cl += 1
            max_cl  = max(max_cl, cur_cl)
        else:
            cur_cl = 0
    return max_cl

# ━━ main ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    df  = fetch_daily()
    df  = add_indicators(df)
    buy_s, sell_s = generate_signals(df)

    print("\n[BACKTEST] 実行中...")
    result = backtest(df, buy_s, sell_s)
    bah    = buy_and_hold(df)
    dca_r  = dca_monthly(df)

    # 年次ブレークダウン（戦略とB&Hの equity を揃える）
    strat_eq = result["equity"]
    bah_eq   = bah["equity"]

    # データ長が一致しない場合の保護
    min_len = min(len(strat_eq), len(bah_eq), len(df))
    df_trim = df.iloc[:min_len]
    yearly  = yearly_breakdown(df_trim,
                                strat_eq[:min_len],
                                bah_eq[:min_len])

    csig = current_signal(df, buy_s, sell_s)

    print_report(df, result, bah, dca_r, yearly, csig)
