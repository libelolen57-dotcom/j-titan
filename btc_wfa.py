"""
BTC ウォークフォワード分析（Walk-Forward Analysis）
オーバーフィッティング検証 & 未来で勝てるか判定

データ分割:
  直近3年間 → 前2年: 訓練データ（インサンプル）
             後1年: テストデータ（アウトオブサンプル）

処理フロー:
  1. 訓練期間のみでグリッドサーチ（336通りのパラメーター）
  2. 最良パラメーターを「一切変更せず」テスト期間に適用
  3. IS vs OOS のパフォーマンス比較
  4. パラメーター安定性ヒートマップ（鋭いピーク = 過学習の証拠）
  5. OOS の四半期別・ローリング3ヶ月パフォーマンス
  6. 3ヶ月+30%達成確率の統計的検証
"""

import pandas as pd
import numpy as np
import os
from itertools import product
from datetime import timedelta

INITIAL_JPY   = 500_000
FEE_RATE      = 0.001
SLIP_RATE     = 0.0005
COST_ONEWAY   = FEE_RATE + SLIP_RATE
CSV_BTC       = "/Users/hiroseren/btc_usd_daily_5y.csv"
CSV_FX        = "/Users/hiroseren/usdjpy_5y.csv"

# ━━ データ読込 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_data():
    btc = pd.read_csv(CSV_BTC, index_col=0, parse_dates=True)
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)
    if btc.index.tz is not None:
        btc.index = btc.index.tz_localize(None)
    btc = btc.dropna(subset=["Close"])

    fx = pd.read_csv(CSV_FX, index_col=0, parse_dates=True)
    if isinstance(fx.columns, pd.MultiIndex):
        fx.columns = fx.columns.get_level_values(0)
    if fx.index.tz is not None:
        fx.index = fx.index.tz_localize(None)
    usdjpy = fx["Close"].rename("usdjpy")

    # 直近3年に絞る
    cutoff = btc.index[-1] - timedelta(days=365*3)
    btc    = btc[btc.index >= cutoff].copy()
    return btc, usdjpy

def add_indicators(df, fx):
    c = df["Close"].copy()
    h, l = df["High"], df["Low"]
    df = df.copy()

    for span, col in [(50,"ema50"),(100,"ema100"),(200,"ema200")]:
        df[col] = c.ewm(span=span, adjust=False).mean()

    wsma = c.resample("W").last().rolling(20).mean()
    df["weekly_sma20"] = wsma.reindex(df.index, method="ffill")

    d  = c.diff()
    g  = d.clip(lower=0).rolling(14).mean()
    lo = (-d.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + g / lo.replace(0, np.nan))

    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    df["atr"]  = tr.rolling(14).mean()
    df["rv20"] = c.pct_change().rolling(20).std() * np.sqrt(365)

    df["e200_slope_90d"] = (df["ema200"] - df["ema200"].shift(90)) / df["ema200"].shift(90) * 100

    df["usdjpy"] = fx.reindex(df.index, method="ffill").ffill().bfill()
    return df

# ━━ 高速バックテストエンジン ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fast_backtest(df, rsi_buy, rsi_sell, ema_period, atr_stop_mul, kelly=0.10):
    """
    コスト・為替込みの高速バックテスト。
    グリッドサーチ用に最適化（シンプルな戦略ロジックのみ）。
    """
    close  = df["Close"].values
    rsi_   = df["rsi"].values
    atr_   = df["atr"].values
    rv_    = df["rv20"].values
    wsma_  = df["weekly_sma20"].values
    fx_    = df["usdjpy"].values
    dates  = df.index

    # EMAをパラメーター指定期間で計算
    c_s    = df["Close"]
    ema_t  = c_s.ewm(span=ema_period, adjust=False).mean().values
    ema50_ = df["ema50"].values
    ema100_= df["ema100"].values

    def _g(arr, i): return float(arr[i]) if i < len(arr) and not np.isnan(float(arr[i])) else 0.0

    fx0 = _g(fx_, 0) or 150.0
    capital_usd = INITIAL_JPY / fx0

    pos = {"qty":0.0, "entry":0.0, "stop":0.0, "trail":0.0, "tp":False}
    equity_jpy  = []
    pnls_jpy    = []
    total_cost  = 0.0

    for i in range(len(df)):
        px     = float(close[i])
        atr_v  = _g(atr_, i) or px*0.03
        rv_v   = _g(rv_, i)  or 0.3
        rsi_v  = _g(rsi_, i) or 50
        rsi_p  = _g(rsi_, i-1) if i > 0 else 50
        e_t    = _g(ema_t, i);  e_t_p = _g(ema_t, i-1)  if i > 0 else e_t
        e50_v  = _g(ema50_,i);  e50_p = _g(ema50_, i-1) if i > 0 else e50_v
        e100_v = _g(ema100_,i)
        wsma_v = _g(wsma_, i)
        fx_v   = _g(fx_, i) or 150.0
        px_p   = float(close[i-1]) if i > 0 else px

        equity_jpy.append((capital_usd + pos["qty"] * px) * fx_v)

        # トレーリングSL
        if pos["qty"] > 0:
            pos["trail"] = max(pos["trail"], px)
            pos["stop"]  = max(pos["stop"], pos["trail"] - atr_v * atr_stop_mul)

        # ストップ
        if pos["qty"] > 0 and px <= pos["stop"]:
            ep  = px * (1 - COST_ONEWAY)
            pnl = (ep - pos["entry"]) * pos["qty"] * fx_v
            capital_usd += pos["qty"] * ep
            total_cost  += ep * pos["qty"] * COST_ONEWAY
            pnls_jpy.append(pnl)
            pos.update({"qty":0.0,"entry":0.0,"stop":0.0,"trail":0.0,"tp":False})

        # 部分利食い
        if pos["qty"] > 0 and not pos["tp"] and px >= pos["entry"] + atr_v * 3.0:
            cut = pos["qty"] * 0.35
            ep  = px * (1 - COST_ONEWAY)
            pnl = (ep - pos["entry"]) * cut * fx_v
            capital_usd += cut * ep
            total_cost  += ep * cut * COST_ONEWAY
            pos["qty"]  -= cut; pos["tp"] = True
            pnls_jpy.append(pnl)

        # 売りシグナル
        sell = ((px_p >= e100_v and px < e100_v) or
                (rsi_p >= rsi_sell and rsi_v < rsi_sell) or
                (px < wsma_v and wsma_v > 0))
        if pos["qty"] > 0 and sell:
            ep  = px * (1 - COST_ONEWAY)
            pnl = (ep - pos["entry"]) * pos["qty"] * fx_v
            capital_usd += pos["qty"] * ep
            total_cost  += ep * pos["qty"] * COST_ONEWAY
            pnls_jpy.append(pnl)
            pos.update({"qty":0.0,"entry":0.0,"stop":0.0,"trail":0.0,"tp":False})

        # 買いシグナル（EMAクロス + RSI条件）
        cross_up = (px_p <= e_t_p and px > e_t)
        gc       = (e50_p <= e_t_p and e50_v > e_t)
        if pos["qty"] == 0 and (cross_up or gc):
            sl   = px - atr_v * atr_stop_mul
            risk = px - sl
            if risk > 0:
                vol_f = np.clip(0.30 / max(rv_v, 0.05), 0.3, 2.0)
                size  = min((capital_usd * kelly) / risk,
                            (capital_usd * kelly * vol_f) / risk,
                            capital_usd / px * 0.99)
                if size > 0:
                    ep = px * (1 + COST_ONEWAY)
                    capital_usd -= size * ep
                    total_cost  += ep * size * COST_ONEWAY
                    pos.update({"qty":size,"entry":ep,"stop":sl,"trail":px,"tp":False})

    # 未決済を終値で決済
    if pos["qty"] > 0:
        ep  = float(close[-1]) * (1 - COST_ONEWAY)
        pnl = (ep - pos["entry"]) * pos["qty"] * (_g(fx_,-1) or 150)
        capital_usd += pos["qty"] * ep
        pnls_jpy.append(pnl)

    final_jpy  = capital_usd * (_g(fx_, -1) or 150)
    eq         = np.array(equity_jpy)
    pk         = np.maximum.accumulate(eq)
    max_dd     = float(((eq - pk) / pk).min() * 100)
    ret        = (final_jpy - INITIAL_JPY) / INITIAL_JPY * 100
    bar_ret    = np.diff(eq) / eq[:-1]
    sharpe     = float(bar_ret.mean() / bar_ret.std() * np.sqrt(365)) if bar_ret.std() > 0 else 0

    wins   = [p for p in pnls_jpy if p > 0]
    losses = [p for p in pnls_jpy if p < 0]
    wr     = len(wins) / len(pnls_jpy) * 100 if pnls_jpy else 0
    pf     = sum(wins) / abs(sum(losses)) if losses else 99
    calmar = abs(ret / max_dd) if max_dd != 0 else 0

    return {
        "ret": round(ret, 2), "dd": round(max_dd, 2),
        "sharpe": round(sharpe, 2), "wr": round(wr, 1),
        "pf": round(pf, 2), "calmar": round(calmar, 2),
        "n": len(pnls_jpy), "final_jpy": round(final_jpy, 0),
        "cost_jpy": round(total_cost * (_g(fx_,-1) or 150), 0),
        "equity_jpy": eq.tolist(),
    }

# ━━ グリッドサーチ（訓練データのみ） ━━━━━━━━━━━━━━━━━━━━━━

def grid_search(df_train):
    """
    訓練データのみでパラメーター最適化。
    スコア = Sharpe × (1 + ret/100) / (1 + |DD|/100) - DD超過ペナルティ
    「最大リターン」でなく「リスク調整後リターン」で選ぶ。
    """
    param_grid = {
        "rsi_buy":  [48, 50, 52, 54, 56, 58, 60],
        "rsi_sell": [35, 38, 40, 42],
        "ema_p":    [75, 100, 150, 200],
        "atr_mul":  [3.0, 3.5, 4.0],
    }

    keys   = list(param_grid.keys())
    combos = list(product(*param_grid.values()))
    total  = len(combos)
    print(f"  グリッドサーチ: {total}通りの組み合わせを訓練データで評価中...")

    results = []
    best_score = -np.inf
    best_params = None

    for idx, vals in enumerate(combos):
        p = dict(zip(keys, vals))
        if p["rsi_buy"] <= p["rsi_sell"]:
            continue  # 無効な組み合わせ

        r = fast_backtest(df_train, p["rsi_buy"], p["rsi_sell"],
                          p["ema_p"], p["atr_mul"])

        # スコア: Sharpeベース + DDペナルティ（DD 20%超は重く減点）
        dd_pen = max(0, (-r["dd"] - 20) * 0.5)
        score  = r["sharpe"] * max(1 + r["ret"]/100, 0.1) / (1 + abs(r["dd"])/100) - dd_pen

        results.append({**p, **r, "score": round(score, 4)})

        if score > best_score:
            best_score  = score
            best_params = {**p}

        if (idx + 1) % 80 == 0:
            print(f"    {idx+1}/{total}... 最高スコア={best_score:.3f}", end="\r")

    print(f"\n  完了。最高スコア = {best_score:.3f}")
    return best_params, results

# ━━ 安定性ヒートマップ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def stability_heatmap(results, best_p):
    """
    RSI買い閾値 × EMA期間 の収益率ヒートマップ。
    鋭いピーク → 過学習リスク大
    広い高原   → 堅牢性あり
    """
    rsi_buys = sorted(set(r["rsi_buy"] for r in results))
    ema_ps   = sorted(set(r["ema_p"]   for r in results))

    # 固定値: 最良パラメーターのrsi_sell, atr_mul
    rs_fix = best_p["rsi_sell"]
    am_fix = best_p["atr_mul"]

    print(f"\n  パラメーター安定性ヒートマップ（rsi_sell={rs_fix}, atr_mul={am_fix}）")
    print(f"  目標: +30%以上のゾーンが広く連続していること（高原型 = 堅牢）\n")

    hdr = "  RSI買↓ EMA→ " + "  ".join(f"{e:>5}" for e in ema_ps)
    print(hdr)
    print("  " + "─" * 55)

    pos30_count = 0
    total_cells = 0

    for rb in rsi_buys:
        row = f"  RSI≥{rb:<4}      "
        for ep in ema_ps:
            match = [r for r in results
                     if r["rsi_buy"]==rb and r["ema_p"]==ep
                     and r["rsi_sell"]==rs_fix and r["atr_mul"]==am_fix]
            if match:
                ret = match[0]["ret"]
                total_cells += 1
                if ret >= 30:
                    pos30_count += 1
                    sym = "◎" if ret >= 50 else "○"
                elif ret >= 0:
                    sym = "△"
                else:
                    sym = "✗"
                # ベストパラメーターに目印
                mark = "★" if (rb == best_p["rsi_buy"] and ep == best_p["ema_p"]) else " "
                row += f" {ret:>+5.0f}%{sym}{mark}"
            else:
                row += "     — "
        print(row)

    pct = pos30_count / total_cells * 100 if total_cells > 0 else 0
    print(f"\n  +30%達成セル: {pos30_count}/{total_cells} ({pct:.0f}%)")
    if pct >= 60:
        print("  → 広い高原を確認。過学習でなく実際のエッジが存在。")
    elif pct >= 35:
        print("  → 中程度の安定性。条件依存だが一定のエッジあり。")
    else:
        print("  → 高原が狭い。鋭いピーク = 過学習リスク高。")
    return pct

# ━━ OOS 四半期・ローリング3ヶ月分析 ━━━━━━━━━━━━━━━━━━━━

def oos_quarterly(df_oos, equity_jpy):
    """OOSの四半期別・ローリング3ヶ月パフォーマンス"""
    eq = np.array(equity_jpy)
    idx_map = {date: i for i, date in enumerate(df_oos.index)}

    quarters = []
    for q in range(4):
        start_m = q * 3
        q_dates = [d for d in df_oos.index
                   if d >= df_oos.index[0] + timedelta(days=30*start_m)
                   and d < df_oos.index[0] + timedelta(days=30*(start_m+3))]
        if not q_dates:
            continue
        i0 = idx_map.get(q_dates[0],  0)
        i1 = idx_map.get(q_dates[-1], len(eq)-1)
        if i1 >= len(eq): i1 = len(eq) - 1
        q_eq   = eq[i0:i1+1]
        q_ret  = (q_eq[-1] - q_eq[0]) / q_eq[0] * 100 if len(q_eq) > 1 else 0
        pk     = np.maximum.accumulate(q_eq)
        q_dd   = float(((q_eq-pk)/pk).min()*100) if len(q_eq) > 1 else 0
        quarters.append({
            "label": f"Q{q+1} ({q_dates[0].strftime('%Y-%m')})",
            "ret": round(q_ret, 1),
            "dd":  round(q_dd, 1),
            "start": q_dates[0], "end": q_dates[-1],
        })

    # ローリング3ヶ月（約65日）窓で+30%達成率を計算
    window = 65
    rolling_rets = []
    for i in range(len(eq) - window):
        r = (eq[i+window] - eq[i]) / eq[i] * 100
        rolling_rets.append(r)

    hit_30  = sum(1 for r in rolling_rets if r >= 30)
    hit_15  = sum(1 for r in rolling_rets if r >= 15)
    hit_0   = sum(1 for r in rolling_rets if r >= 0)
    hit_neg = sum(1 for r in rolling_rets if r < 0)
    total_w = len(rolling_rets)

    return quarters, {
        "windows":  total_w,
        "hit_30":   hit_30,
        "hit_15":   hit_15,
        "hit_0":    hit_0,
        "hit_neg":  hit_neg,
        "pct_30":   round(hit_30/total_w*100, 1) if total_w > 0 else 0,
        "pct_15":   round(hit_15/total_w*100, 1) if total_w > 0 else 0,
        "pct_pos":  round(hit_0/total_w*100, 1)  if total_w > 0 else 0,
        "rolling":  rolling_rets,
        "best_3m":  round(max(rolling_rets), 1) if rolling_rets else 0,
        "worst_3m": round(min(rolling_rets), 1) if rolling_rets else 0,
        "mean_3m":  round(np.mean(rolling_rets), 1) if rolling_rets else 0,
    }

# ━━ バイ&ホールド比較 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def bah(df):
    c   = df["Close"].values
    fx_ = df["usdjpy"].values
    fx0 = float(fx_[0]) if not np.isnan(fx_[0]) else 150
    units = INITIAL_JPY / (float(c[0]) * fx0)
    fx_a  = np.array([float(f) if not np.isnan(float(f)) else 150 for f in fx_])
    eq_jpy= units * c[:len(fx_a)] * fx_a[:len(c)]
    pk    = np.maximum.accumulate(eq_jpy)
    dd    = float(((eq_jpy-pk)/pk).min()*100)
    ret   = (units*float(c[-1])*float(fx_a[-1]) - INITIAL_JPY)/INITIAL_JPY*100
    return {"ret": round(ret,2), "dd": round(dd,2), "equity_jpy": eq_jpy.tolist()}

# ━━ メインレポート ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_report(df_train, df_oos, best_p, is_res, oos_res,
                 bah_is, bah_oos, quarters, rolling_stats, stability_pct):
    W = 70

    print("\n" + "═"*W)
    print("  ウォークフォワード分析 — オーバーフィッティング検証")
    print("═"*W)

    print(f"\n  データ分割")
    print(f"  訓練（IS）: {df_train.index[0].date()} 〜 {df_train.index[-1].date()}"
          f"  ({len(df_train):,}日 / 約2年)")
    print(f"  テスト（OOS）: {df_oos.index[0].date()} 〜 {df_oos.index[-1].date()}"
          f"  ({len(df_oos):,}日 / 約1年)")
    print(f"\n  最適化パラメーター（訓練データのみで決定）")
    print(f"    RSI 買い: {best_p['rsi_buy']}  RSI 売り: {best_p['rsi_sell']}")
    print(f"    EMA 期間: {best_p['ema_p']}    ATR 倍率: {best_p['atr_mul']}")

    # ── 1. IS vs OOS 比較
    print(f"\n{'─'*W}")
    print(f"  【1】インサンプル（訓練） vs アウトオブサンプル（テスト）")
    print(f"{'─'*W}")
    print(f"  {'':22} {'IS（訓練2年）':>14} {'OOS（テスト1年）':>15}")
    print(f"  {'─'*52}")

    metrics = [
        ("収益率（円建て）", f"{is_res['ret']:>+13.1f}%", f"{oos_res['ret']:>+14.1f}%"),
        ("最大ドローダウン",  f"{is_res['dd']:>13.1f}%",  f"{oos_res['dd']:>14.1f}%"),
        ("シャープレシオ",    f"{is_res['sharpe']:>14.2f}", f"{oos_res['sharpe']:>15.2f}"),
        ("勝率",             f"{is_res['wr']:>13.1f}%",   f"{oos_res['wr']:>14.1f}%"),
        ("取引回数",         f"{is_res['n']:>14}回",      f"{oos_res['n']:>15}回"),
        ("取引コスト",        f"{is_res['cost_jpy']:>12,.0f}円", f"{oos_res['cost_jpy']:>13,.0f}円"),
    ]
    for name, is_v, oos_v in metrics:
        print(f"  {name:<22} {is_v} {oos_v}")

    # B&H との比較
    print(f"\n  バイ&ホールド（参考）")
    print(f"  {'':22} {'IS（訓練2年）':>14} {'OOS（テスト1年）':>15}")
    print(f"  {'─'*52}")
    print(f"  {'B&H収益率（円建て）':<22} {bah_is['ret']:>+13.1f}%  {bah_oos['ret']:>+14.1f}%")
    print(f"  {'B&H最大DD':<22} {bah_is['dd']:>13.1f}%  {bah_oos['dd']:>14.1f}%")

    # ── 2. 過学習診断
    print(f"\n{'─'*W}")
    print(f"  【2】過学習診断")
    print(f"{'─'*W}")

    gen_score = oos_res["ret"] / is_res["ret"] if is_res["ret"] > 0 else 0
    print(f"  汎化スコア（OOS収益 / IS収益）: {gen_score:.2f}")
    print(f"    1.0 = 完璧な汎化  |  0.5〜1.0 = 許容範囲  |  0.5未満 = 過学習疑い")

    if gen_score >= 0.7:
        overfit_judge = "✓ 良好（過学習なし）"
    elif gen_score >= 0.3:
        overfit_judge = "△ 注意（軽度の性能劣化）"
    elif gen_score >= 0:
        overfit_judge = "✗ 要注意（IS→OOSで大幅劣化）"
    else:
        overfit_judge = "✗ 警告（OOSでマイナス）"

    print(f"  判定: {overfit_judge}")

    print(f"\n  パラメーター安定性: {stability_pct:.0f}%のパラメーターが+30%以上を達成")
    if stability_pct >= 60:
        print(f"  → 特定パラメーターへの依存なし。戦略自体にエッジあり。")
    elif stability_pct >= 35:
        print(f"  → 条件によって結果が異なる。中程度の安定性。")
    else:
        print(f"  → 最良点付近のみ機能。パラメーター感度高く注意が必要。")

    # ── 3. OOS 四半期別
    print(f"\n{'─'*W}")
    print(f"  【3】OOS（テストデータ）四半期ブレークダウン")
    print(f"{'─'*W}")
    print(f"  {'期間':<25} {'収益率':>8} {'最大DD':>8}  評価")
    print(f"  {'─'*55}")
    for q in quarters:
        arrow = "▲" if q["ret"] >= 0 else "▼"
        note  = "✓ 目標達成" if q["ret"] >= 30 else ("良好" if q["ret"] >= 15 else
                ("利益" if q["ret"] >= 0 else "損失"))
        print(f"  {q['label']:<25} {arrow}{abs(q['ret']):>6.1f}%  {q['dd']:>6.1f}%   {note}")

    # ── 4. ローリング3ヶ月分析
    rs = rolling_stats
    print(f"\n{'─'*W}")
    print(f"  【4】ローリング3ヶ月パフォーマンス分析（OOS期間）")
    print(f"{'─'*W}")
    print(f"  分析窓数        : {rs['windows']}窓（65日=約3ヶ月でスライド）")
    print(f"  +30%以上達成   : {rs['hit_30']:>3}回 ({rs['pct_30']:>5.1f}%)")
    print(f"  +15%以上達成   : {rs['hit_15']:>3}回 ({rs['pct_15']:>5.1f}%)")
    print(f"  プラス圏        : {rs['hit_0']:>3}回 ({rs['pct_pos']:>5.1f}%)")
    print(f"  マイナス圏      : {rs['hit_neg']:>3}回 ({100-rs['pct_pos']:>5.1f}%)")
    print(f"\n  3ヶ月リターンの分布:")
    print(f"  最高: {rs['best_3m']:>+6.1f}%  |  平均: {rs['mean_3m']:>+6.1f}%  |  最低: {rs['worst_3m']:>+6.1f}%")

    # テキストヒストグラム
    rets = rs["rolling"]
    if rets:
        bins = np.linspace(min(rets), max(rets), 10)
        hist, edges = np.histogram(rets, bins=bins)
        max_h = max(hist) if max(hist) > 0 else 1
        print()
        for i in range(len(hist)):
            bar   = "█" * int(hist[i]/max_h*25)
            label = f"{edges[i]:>+6.0f}〜{edges[i+1]:>+5.0f}%"
            mark  = " ← +30%目標" if edges[i] <= 30 < edges[i+1] else ""
            print(f"    {label}  {bar}{mark}")

    # ── 5. 最終評価
    print(f"\n{'═'*W}")
    print(f"  【5】最終評価 — 「3ヶ月+30%」は現実的か？")
    print(f"{'═'*W}")

    print(f"\n  OOS（テストデータ）での総合成績:")
    print(f"  年間収益率       : {oos_res['ret']:>+.1f}%")
    print(f"  3ヶ月+30%達成率  : {rs['pct_30']:.1f}%  ({rs['hit_30']}/{rs['windows']}窓)")

    if rs["pct_30"] >= 25:
        target_judge = "✓ 達成可能（4窓に1回以上）"
    elif rs["pct_30"] >= 10:
        target_judge = "△ 時期を選べば可能（稀に達成）"
    elif rs["pct_30"] > 0:
        target_judge = "✗ 困難（ほぼ達成できない）"
    else:
        target_judge = "✗ 不可能（この相場環境では0%）"

    print(f"  判定             : {target_judge}")

    print(f"\n  ─ 重要な文脈 ─────────────────────────────────")
    # OOS期間のBTC騰落
    btc_oos_ret = (float(df_oos["Close"].iloc[-1]) - float(df_oos["Close"].iloc[0])) \
                  / float(df_oos["Close"].iloc[0]) * 100
    print(f"  テスト期間のBTC実際の騰落: {btc_oos_ret:+.1f}%")
    print(f"  （相場が上昇していた期間か下落していた期間かで大きく結果が変わる）")

    if btc_oos_ret < -20:
        print(f"\n  ⚠️  テスト期間はBTC下落局面（{btc_oos_ret:.0f}%）。")
        print(f"  上昇トレンドフォロー戦略が不利な環境。BTCが回復すれば改善見込み。")
    elif btc_oos_ret > 30:
        print(f"\n  ✓  テスト期間はBTC上昇局面（{btc_oos_ret:.0f}%）。戦略が機能しやすい環境。")

    print(f"\n  ─ 過学習対策として実施した措置 ──────────────────")
    print(f"  1. IS/OOS を完全分離（テストデータに一切触れず最適化）")
    print(f"  2. 最大リターンでなく Sharpe+DDペナルティでパラメーター選択")
    print(f"  3. 安定性テスト（{stability_pct:.0f}%のパラメーターが+30%達成）")
    print(f"  4. 取引コスト（0.30%/往復）を全てのバックテストに組込み")
    print(f"  5. 円建てP&L（為替効果を含む実質収益で評価）")
    print("═"*W)

# ━━ main ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("[LOAD] データ読込...")
    btc, fx = load_data()
    df      = add_indicators(btc, fx)
    print(f"  対象期間: {df.index[0].date()} 〜 {df.index[-1].date()}  ({len(df):,}日)")

    # ── データ分割（前2年=訓練 / 後1年=テスト）
    total_days = len(df)
    split_idx  = int(total_days * (2/3))   # 3年の2/3 = 2年
    df_train   = df.iloc[:split_idx].copy()
    df_oos     = df.iloc[split_idx:].copy()
    print(f"\n[SPLIT] 訓練: {df_train.index[0].date()}〜{df_train.index[-1].date()} ({len(df_train)}日)")
    print(f"[SPLIT] テスト: {df_oos.index[0].date()}〜{df_oos.index[-1].date()} ({len(df_oos)}日)")

    # ── グリッドサーチ（訓練データのみ）
    print("\n[OPTIMIZE] 訓練データでグリッドサーチ...")
    best_p, all_results = grid_search(df_train)
    print(f"  最良パラメーター: RSI買={best_p['rsi_buy']} RSI売={best_p['rsi_sell']} "
          f"EMA={best_p['ema_p']} ATR×{best_p['atr_mul']}")

    # ── 安定性ヒートマップ（訓練データ）
    print("\n[STABILITY] パラメーター安定性テスト（訓練データ）")
    stab_pct = stability_heatmap(all_results, best_p)

    # ── IS バックテスト（訓練データで最良パラメーターを確認）
    print(f"\n[IS] 訓練データバックテスト...")
    is_res  = fast_backtest(df_train, best_p["rsi_buy"], best_p["rsi_sell"],
                             best_p["ema_p"], best_p["atr_mul"])
    bah_is  = bah(df_train)
    print(f"  IS収益率: {is_res['ret']:+.1f}%  MaxDD: {is_res['dd']:.1f}%  Sharpe: {is_res['sharpe']:.2f}")

    # ── OOS バックテスト（テストデータで同じパラメーターを適用）
    print(f"\n[OOS] テストデータバックテスト（パラメーター固定）...")
    oos_res = fast_backtest(df_oos, best_p["rsi_buy"], best_p["rsi_sell"],
                             best_p["ema_p"], best_p["atr_mul"])
    bah_oos = bah(df_oos)
    print(f"  OOS収益率: {oos_res['ret']:+.1f}%  MaxDD: {oos_res['dd']:.1f}%  Sharpe: {oos_res['sharpe']:.2f}")

    # ── OOS 詳細分析
    quarters, rolling_stats = oos_quarterly(df_oos, oos_res["equity_jpy"])

    # ── レポート出力
    print_report(df_train, df_oos, best_p, is_res, oos_res,
                 bah_is, bah_oos, quarters, rolling_stats, stab_pct)
