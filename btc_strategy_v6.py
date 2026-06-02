"""
BTC-USD Strategy v6 — トレンドライド戦略

v5で判明した根本問題:
  「RSIクロス条件 = 強気相場で RSI が高止まりするとシグナルが出ない」
  → 2021, 2023, 2024の大きな上昇をほぼ全て見逃した

v6の設計哲学の転換:
  × 「エントリーポイントを探す」（RSIクロス）
  ○ 「相場が上昇トレンドにある間はずっと乗り続ける」

v6の仕組み:
  1. マクロレジーム（週足 20SMA）で大きな方向を判定
  2. 「乗る」条件: 価格が EMA200 を上抜け OR EMA50 が EMA200 をゴールデンクロス
  3. 「降りる」条件: 価格が EMA100 を割れ OR RSI < 35（売られすぎ崩壊）
  4. ストップ: ATR×4（日足の通常変動に狩られない幅）
  5. ポジションサイズ: ボラティリティターゲティング + Half-Kelly
  6. 追加: 強トレンド時（ADX>30）に 1.5倍サイズ

比較対象（全て同期間・同資金）:
  A. バイ&ホールド
  B. DCA（月次積立）
  C. v5戦略（RSIクロス、失敗版）
  D. v6戦略（トレンドライド、新版）
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os

INITIAL_CAPITAL   = 500_000
CSV_DAILY         = "/Users/hiroseren/btc_usd_daily_5y.csv"
TARGET_VOL        = 0.30    # 年率30%ボラターゲット
MAX_KELLY         = 0.12    # Half-Kelly 上限
BULL_SCALE        = 1.5     # 強トレンド時の倍率
ATR_STOP_MUL      = 4.0
ATR_TP_MUL        = 3.0     # 部分利食いATR倍率

# ━━ データ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load():
    df = pd.read_csv(CSV_DAILY, index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.dropna(subset=["Close"])
    print(f"[DATA] {df.index[0].date()} ～ {df.index[-1].date()}  ({len(df):,}日)")
    return df

# ━━ 指標 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def indicators(df):
    c = df["Close"]
    h, l = df["High"], df["Low"]
    df = df.copy()

    df["ema50"]  = c.ewm(span=50,  adjust=False).mean()
    df["ema100"] = c.ewm(span=100, adjust=False).mean()
    df["ema200"] = c.ewm(span=200, adjust=False).mean()
    df["sma200"] = c.rolling(200).mean()

    # RSI
    d  = c.diff()
    g  = d.clip(lower=0).rolling(14).mean()
    lo = (-d.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + g / lo.replace(0, np.nan))

    # ATR
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # ADX
    up = h.diff(); dn = -l.diff()
    pdm = up.where((up>dn)&(up>0), 0).rolling(14).mean()
    ndm = dn.where((dn>up)&(dn>0), 0).rolling(14).mean()
    pdi = 100 * pdm / df["atr"].replace(0, np.nan)
    ndi = 100 * ndm / df["atr"].replace(0, np.nan)
    df["adx"] = (100*(pdi-ndi).abs()/(pdi+ndi).replace(0,np.nan)).rolling(14).mean()

    # 実現ボラ（20日、年率）
    df["rv20"] = c.pct_change().rolling(20).std() * np.sqrt(365)

    # EMA200 傾き
    df["e200_slope"] = df["ema200"] - df["ema200"].shift(15)

    # 週足 20SMA（マクロレジーム用：日足データを週次リサンプルして戻す）
    weekly = c.resample("W").last().rolling(20).mean()
    df["weekly_sma20"] = weekly.reindex(df.index, method="ffill")

    return df

# ━━ v6 シグナル（トレンドライド） ━━━━━━━━━━━━━━━━━━━━━━

def signals_v6(df):
    """
    エントリー（いずれか）:
      (a) 価格がEMA200を下から上に抜け、かつEMA200傾き>0
      (b) EMA50がEMA200をゴールデンクロス

    継続保有（以下が全て真の間は持ち続ける）:
      価格 > EMA100

    エグジット（いずれか）:
      価格が EMA100 を上から下に抜ける
      OR RSI が 35 を上から下に抜ける（崩壊シグナル）
      OR 週足 SMA20 を価格が割れる（マクロ転換）
    """
    c      = df["Close"]
    e50    = df["ema50"]
    e100   = df["ema100"]
    e200   = df["ema200"]
    slope  = df["e200_slope"]
    rsi    = df["rsi"]
    w_sma  = df["weekly_sma20"]

    # (a) 価格が EMA200 を上抜け
    cross_above_e200 = (c.shift(1) <= e200.shift(1)) & (c > e200) & (slope > 0)

    # (b) ゴールデンクロス
    golden_cross = (e50.shift(1) <= e200.shift(1)) & (e50 > e200)

    buy = cross_above_e200 | golden_cross

    # エグジット
    cross_below_e100 = (c.shift(1) >= e100.shift(1)) & (c < e100)
    rsi_collapse     = (rsi.shift(1) >= 35) & (rsi < 35)
    macro_break      = c < w_sma

    sell = cross_below_e100 | rsi_collapse | macro_break

    return buy, sell

# ━━ v5 シグナル（比較用） ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def signals_v5(df):
    c     = df["Close"]
    e200  = df["ema200"]
    slope = df["e200_slope"]
    rsi   = df["rsi"]
    e100  = df["ema100"]
    rv    = df["rv20"]

    trend   = (c > e200) & (slope > 0)
    vol_ok  = rv < 0.9
    buy     = (rsi.shift(1) < 56) & (rsi >= 56) & trend & vol_ok
    sell    = ((rsi.shift(1) > 40) & (rsi <= 40)) | (c < e100)
    return buy, sell

# ━━ ポジションサイジング ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AdaptiveSizer:
    def __init__(self):
        self.pnls = []   # 勝ちのpct, 負けのpct

    def record(self, pnl_pct):
        self.pnls.append(pnl_pct)

    def kelly_fraction(self):
        n = len(self.pnls)
        if n < 15:
            return 0.03
        wins = [p for p in self.pnls if p > 0]
        loss = [abs(p) for p in self.pnls if p < 0]
        if not loss:
            return MAX_KELLY
        W = len(wins) / n
        R = np.mean(wins) / np.mean(loss)
        k = max(0, W - (1-W) / R) / 2   # Half-Kelly
        return min(k, MAX_KELLY)

    def position_size(self, capital, px, atr_v, adx_v, rv_v):
        sl   = px - atr_v * ATR_STOP_MUL
        risk = px - sl
        if risk <= 0:
            return 0.0

        frac = self.kelly_fraction()

        # ボラティリティターゲティング
        rv   = max(rv_v, 0.05)
        vt   = TARGET_VOL / rv          # 高ボラ時はサイズ縮小
        vt   = np.clip(vt, 0.3, 2.0)

        # 強トレンドボーナス
        trend_scale = BULL_SCALE if (not np.isnan(adx_v) and adx_v > 30) else 1.0

        size_risk = (capital * frac) / risk
        size_vol  = (capital * frac * vt * trend_scale) / risk
        size = min(size_risk, size_vol, capital / px * 0.98)
        return max(size, 0.0)

    def stats(self):
        n = len(self.pnls)
        if n == 0: return {}
        wins = [p for p in self.pnls if p > 0]
        loss = [abs(p) for p in self.pnls if p < 0]
        W = len(wins)/n
        R = np.mean(wins)/np.mean(loss) if loss else 99
        k = max(0, W-(1-W)/R)
        return {"n":n,"W":round(W*100,1),"R":round(R,2),
                "kelly_full":round(k*100,1),"kelly_half":round(k/2*100,1)}

# ━━ バックテストエンジン ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def backtest(df, buy_s, sell_s, label=""):
    close  = df["Close"].values
    atr_   = df["atr"].values
    adx_   = df["adx"].values
    rv_    = df["rv20"].values
    dates  = df.index

    capital  = float(INITIAL_CAPITAL)
    pos      = 0.0
    entry_px = 0.0
    stop_px  = 0.0
    trail_hi = 0.0
    tp_done  = False
    equity   = []
    trades   = []
    sizer    = AdaptiveSizer()

    for i in range(len(df)):
        px    = float(close[i])
        atr_v = float(atr_[i])  if not np.isnan(atr_[i])  else px*0.03
        adx_v = float(adx_[i])  if not np.isnan(adx_[i])  else 0
        rv_v  = float(rv_[i])   if not np.isnan(rv_[i])   else 0.5

        equity.append(capital + pos * px)

        # トレーリングストップ
        if pos > 0:
            trail_hi = max(trail_hi, px)
            new_sl   = trail_hi - atr_v * ATR_STOP_MUL
            stop_px  = max(stop_px, new_sl)

        # ストップ判定
        if pos > 0 and px <= stop_px:
            pnl_pct = (px - entry_px) / entry_px
            pnl_jpy = (px - entry_px) * pos
            capital += pos * px
            sizer.record(pnl_pct)
            trades.append({"date": dates[i], "pnl": pnl_jpy, "pct": pnl_pct*100, "type": "SL"})
            pos = 0.0; tp_done = False

        # 部分利食い
        if pos > 0 and not tp_done and px >= entry_px + atr_v * ATR_TP_MUL:
            cut = pos * 0.35
            capital += cut * px
            pos -= cut
            tp_done = True

        # 買いシグナル
        if pos == 0 and bool(buy_s.iloc[i]):
            size = sizer.position_size(capital, px, atr_v, adx_v, rv_v)
            if size > 0:
                pos      = size
                entry_px = px
                stop_px  = px - atr_v * ATR_STOP_MUL
                trail_hi = px
                tp_done  = False
                capital -= size * px

        # 売りシグナル
        elif pos > 0 and bool(sell_s.iloc[i]):
            pnl_pct = (px - entry_px) / entry_px
            pnl_jpy = (px - entry_px) * pos
            capital += pos * px
            sizer.record(pnl_pct)
            trades.append({"date": dates[i], "pnl": pnl_jpy, "pct": pnl_pct*100, "type": "exit"})
            pos = 0.0; tp_done = False

    if pos > 0:
        px = float(close[-1])
        pnl_pct = (px - entry_px) / entry_px
        pnl_jpy = (px - entry_px) * pos
        capital += pos * px
        trades.append({"date": dates[-1], "pnl": pnl_jpy, "pct": pnl_pct*100, "type": "final"})

    eq  = np.array(equity)
    pk  = np.maximum.accumulate(eq)
    dd  = float(((eq-pk)/pk).min()*100)
    ret = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    br  = np.diff(eq)/eq[:-1]
    sh  = float(br.mean()/br.std()*np.sqrt(365)) if br.std()>0 else 0

    pnls  = [t["pnl"] for t in trades]
    wins  = [p for p in pnls if p > 0]
    loss  = [p for p in pnls if p < 0]
    wr    = len(wins)/len(pnls)*100 if pnls else 0
    pf    = sum(wins)/abs(sum(loss)) if loss else 99

    # 最大連続損失
    consec = cur = 0
    for p in pnls:
        if p < 0: cur += 1; consec = max(consec, cur)
        else: cur = 0

    # バルマー比（Calmar）
    calmar = abs(ret / dd) if dd != 0 else 0

    return {
        "label": label,
        "final": round(capital, 0),
        "ret":   round(ret, 2),
        "wr":    round(wr, 1),
        "dd":    round(dd, 2),
        "sharpe":round(sh, 2),
        "pf":    round(pf, 2),
        "n":     len(pnls),
        "calmar":round(calmar, 2),
        "max_consec_loss": consec,
        "avg_win": round(np.mean(wins),0)  if wins else 0,
        "avg_loss":round(np.mean(loss),0) if loss else 0,
        "equity": eq,
        "trades": trades,
        "sizer":  sizer,
    }

# ━━ ベンチマーク ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def benchmark_bah(df):
    c  = df["Close"].values
    u  = INITIAL_CAPITAL / float(c[0])
    eq = u * c
    pk = np.maximum.accumulate(eq)
    dd = float(((eq-pk)/pk).min()*100)
    br = np.diff(eq)/eq[:-1]
    sh = float(br.mean()/br.std()*np.sqrt(365)) if br.std()>0 else 0
    ret= (float(c[-1])-float(c[0]))/float(c[0])*100
    return {"label":"バイ&ホールド","final":round(u*float(c[-1]),0),
            "ret":round(ret,2),"dd":round(dd,2),"sharpe":round(sh,2),"equity":eq}

def benchmark_dca(df):
    monthly = df["Close"].resample("ME").last()
    budget  = INITIAL_CAPITAL / len(monthly)
    units   = sum(budget / float(p) for p in monthly.values)
    final   = units * float(df["Close"].iloc[-1])
    return {"label":"DCA月次", "final":round(final,0),
            "ret":round((final-INITIAL_CAPITAL)/INITIAL_CAPITAL*100,2),
            "n_months":len(monthly)}

# ━━ 年次ブレークダウン ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def yearly(df, *results):
    rows = []
    for year in sorted(df.index.year.unique()):
        mask = df.index.year == year
        if mask.sum() < 20: continue
        row = {"year": year}
        for res in results:
            eq = res["equity"][mask[:len(res["equity"])]]
            if len(eq) < 2: continue
            r  = (eq[-1]-eq[0])/eq[0]*100
            pk = np.maximum.accumulate(eq)
            dd = float(((eq-pk)/pk).min()*100)
            row[res["label"]] = {"ret": round(r,1), "dd": round(dd,1)}
        rows.append(row)
    return rows

# ━━ 現在シグナル ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def now_signal(df, buy_s, sell_s):
    last    = df.iloc[-1]
    px      = float(last["Close"])
    e200    = float(last["ema200"])
    e100    = float(last["ema100"])
    e50     = float(last["ema50"])
    rsi_v   = float(last["rsi"])
    adx_v   = float(last["adx"])
    atr_v   = float(last["atr"])
    rv_v    = float(last["rv20"])
    w_sma   = float(last["weekly_sma20"])
    slope   = float(last["e200_slope"])

    above_e200   = px > e200
    above_e100   = px > e100
    above_wsma   = px > w_sma
    e50_gt_e200  = float(last["ema50"]) > e200
    trend_ok     = above_e200 and (slope > 0)

    # 現在の戦略状態判定
    if trend_ok and above_e100 and above_wsma:
        state = "BULL（保有適合）"
        color = "🟢"
    elif above_e200 and not above_wsma:
        state = "混合（慎重に）"
        color = "🟡"
    elif not above_e200 and rsi_v < 30:
        state = "BEAR底圏（反発待ち）"
        color = "🟠"
    else:
        state = "BEAR（ノーポジ推奨）"
        color = "🔴"

    # エントリー待ちの条件を具体的に
    conditions = []
    if not above_e200:
        gap = (e200 - px) / px * 100
        conditions.append(f"・価格が EMA200(${e200:,.0f})を上抜ける (+{gap:.1f}%先)")
    if not e50_gt_e200:
        conditions.append(f"・EMA50(${e50:,.0f})が EMA200 を上抜ける")
    if not above_wsma:
        conditions.append(f"・週足 SMA20(${w_sma:,.0f})を上抜ける")

    # リカバリーシナリオ
    pct_to_e200 = (e200 - px) / px * 100
    pct_to_30   = 30.0
    btc_for_30  = px * 1.30

    return {
        "date": df.index[-1].date(),
        "px": px, "e200": e200, "e100": e100, "e50": e50,
        "rsi": rsi_v, "adx": adx_v, "atr": atr_v, "rv": rv_v,
        "w_sma": w_sma, "slope": slope,
        "state": state, "color": color,
        "conditions": conditions,
        "pct_to_e200": pct_to_e200,
        "btc_for_30": btc_for_30,
    }

# ━━ レポート ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def report(df, r6, r5, bah, dca, yearly_data, sig):
    W = 65
    print("\n" + "═"*W)
    print("  BTC-USD v6 真のアルファ検証 + トレンドライド")
    print(f"  期間: {df.index[0].date()} ～ {df.index[-1].date()}  ({len(df):,}日)")
    print("═"*W)

    # ── 1. 総合比較
    print("\n【1】全戦略 5年間比較（初期資金 50万円）")
    print(f"  {'戦略':<22} {'最終資産':>12} {'収益率':>8} {'MaxDD':>7} {'Sharpe':>7} {'PF':>6} {'取引':>4}")
    print("  " + "─"*(W-2))
    for r in [bah, r5, r6]:
        ar = "▲" if r["ret"]>=0 else "▼"
        pf_s = f"{r.get('pf','-'):.2f}" if "pf" in r else "  —"
        n_s  = str(r.get("n","-"))
        print(f"  {r['label']:<22} {r['final']:>12,.0f}円 "
              f"{ar}{abs(r['ret']):>6.1f}%  {r['dd']:>6.1f}%  "
              f"{r['sharpe']:>6.2f}  {pf_s:>5}  {n_s:>4}")
    print(f"  {'DCA月次':22} {dca['final']:>12,.0f}円 "
          f"▲{dca['ret']:>6.1f}%  —       —      —      {dca['n_months']:>3}ヶ月")

    # v5 → v6 改善
    if r6["ret"] > r5["ret"]:
        print(f"\n  v5→v6 改善: 収益率 {r5['ret']:+.1f}% → {r6['ret']:+.1f}%  "
              f"(+{r6['ret']-r5['ret']:.1f}pt)")
    else:
        print(f"\n  v5→v6: 収益率 {r5['ret']:+.1f}% → {r6['ret']:+.1f}%")

    # B&H との比較
    print(f"\n  バイ&ホールドとの比較:")
    print(f"  {'指標':<20} {'B&H':>12} {'v6戦略':>12} {'v6優位'}")
    print("  " + "─"*52)
    metrics = [
        ("最終資産",   f"{bah['final']:>12,.0f}円", f"{r6['final']:>12,.0f}円",
         "✗" if r6["final"] < bah["final"] else "✓"),
        ("収益率",     f"{bah['ret']:>+11.1f}%",    f"{r6['ret']:>+11.1f}%",
         "✗" if r6["ret"] < bah["ret"] else "✓"),
        ("最大DD",     f"{bah['dd']:>+11.1f}%",     f"{r6['dd']:>+11.1f}%",
         "✓" if abs(r6["dd"]) < abs(bah["dd"]) else "✗"),
        ("Sharpe",    f"{bah['sharpe']:>12.2f}",   f"{r6['sharpe']:>12.2f}",
         "✓" if r6["sharpe"] > bah["sharpe"] else "✗"),
    ]
    for name, bv, sv, mark in metrics:
        print(f"  {name:<20} {bv} {sv}  {mark}")

    # ── 2. 年次ブレークダウン
    print("\n【2】年次ブレークダウン")
    bah_k = "バイ&ホールド"
    v6_k  = r6["label"]
    v5_k  = r5["label"]
    print(f"  {'年':>4}  {'v6戦略':>8}  {'v5戦略':>8}  {'B&H':>8}  評価")
    print("  " + "─"*56)
    for row in yearly_data:
        y    = row["year"]
        v6r  = row.get(v6_k, {}).get("ret", 0)
        v5r  = row.get(v5_k, {}).get("ret", 0)
        bahr = row.get(bah_k, {}).get("ret", 0)
        v6dd = row.get(v6_k, {}).get("dd", 0)

        note = ""
        if bahr < -30:
            note = f"暴落年 → v6: {'+' if v6r>=0 else ''}{v6r:.0f}%  B&H: {bahr:.0f}%"
        elif v6r >= bahr * 0.7 and bahr > 50:
            note = "強気相場の大半を捕捉"
        elif v6r < bahr * 0.3 and bahr > 50:
            note = "強気相場で出遅れ ← 改善余地"

        a6 = "▲" if v6r>=0 else "▼"
        a5 = "▲" if v5r>=0 else "▼"
        ab = "▲" if bahr>=0 else "▼"
        print(f"  {y:>4}  {a6}{abs(v6r):>5.0f}%  {a5}{abs(v5r):>5.0f}%  "
              f"{ab}{abs(bahr):>5.0f}%  {note}")

    # ── 3. Kelly基準
    ks = r6["sizer"].stats()
    print("\n【3】Kelly基準ポジションサイジング（v6実績）")
    if ks:
        kf = r6["sizer"].kelly_fraction()
        print(f"  取引数            : {ks['n']}回")
        print(f"  勝率              : {ks['W']}%")
        print(f"  利益/損失比 R     : {ks['R']}")
        print(f"  フルKelly         : {ks['kelly_full']}%")
        print(f"  採用 Half-Kelly   : {ks['kelly_half']}%  （現在の推奨値）")
        print(f"  次トレードの推奨リスク額: {INITIAL_CAPITAL*kf:,.0f}円 "
              f"（資産の {kf*100:.1f}%）")

    # ── 4. 取引詳細
    print("\n【4】v6 取引統計")
    print(f"  総取引数          : {r6['n']:>5}回")
    print(f"  勝率              : {r6['wr']:>5.1f}%")
    print(f"  プロフィットF     : {r6['pf']:>5.2f}")
    print(f"  Calmar比率        : {r6['calmar']:>5.2f}  （収益率/最大DD）")
    print(f"  平均利益/取引     : {r6['avg_win']:>+8,.0f}円")
    print(f"  平均損失/取引     : {r6['avg_loss']:>+8,.0f}円")
    print(f"  最大連続負け      : {r6['max_consec_loss']:>5}回")

    # ── 5. 現在のシグナル
    s = sig
    print(f"\n【5】現在の市場状態と推奨行動（{s['date']}）")
    print(f"  状態: {s['color']} {s['state']}")
    print()
    print(f"  {'指標':<15} {'値':>14}  {'判定'}")
    print("  " + "─"*48)
    checks = [
        ("BTC価格",     f"${s['px']:>12,.0f}", ""),
        ("EMA200",      f"${s['e200']:>12,.0f}",
         "✓ 上" if s['px']>s['e200'] else f"✗ 下（{s['pct_to_e200']:.1f}%先）"),
        ("EMA100",      f"${s['e100']:>12,.0f}",
         "✓ 上" if s['px']>s['e100'] else "✗ 下"),
        ("週足SMA20",   f"${s['w_sma']:>12,.0f}",
         "✓ 上" if s['px']>s['w_sma'] else "✗ 下"),
        ("RSI(14)",     f"{s['rsi']:>13.1f}",
         "底圏" if s['rsi']<30 else ("過熱" if s['rsi']>70 else "正常")),
        ("ADX(14)",     f"{s['adx']:>13.1f}",
         "強トレンド" if s['adx']>30 else ("中" if s['adx']>20 else "弱/レンジ")),
        ("実現ボラ",    f"{s['rv']*100:>12.1f}%", ""),
        ("EMA200傾き",  f"{s['slope']:>+12,.0f}",
         "上昇トレンド中" if s['slope']>0 else "下降トレンド中"),
    ]
    for name, val, judge in checks:
        print(f"  {name:<15} {val}  {judge}")

    print()
    if s["conditions"]:
        print("  エントリー条件（以下が揃うまで待機）:")
        for c in s["conditions"]:
            print(f"    {c}")
        print()
        print(f"  目標価格 +30% = ${s['btc_for_30']:>10,.0f}  "
              f"（現在から +30.0%）")
        print(f"  現在の推奨: ノーポジションで上記条件を待つ")
    else:
        print("  → 全条件クリア。エントリー可能ゾーン。")

    # ── 6. 設計の核心
    print(f"\n【6】v5 → v6 改善の核心")
    print("  v5 の失敗原因:")
    print("    RSIクロス(56)条件 → 強気相場でRSIが高止まりするとシグナルなし")
    print("    結果: 2021+23.2%、2023+154%の上昇を ほぼ取り損ねた")
    print()
    print("  v6 の解決策:")
    print("    × 「エントリーポイントを探す」（RSIクロス）")
    print("    ○ 「トレンドにある間はずっと乗り続ける」（EMAクロス + 保有継続）")
    print("    ・EMA200上抜け or EMA50/200ゴールデンクロス → 乗る")
    print("    ・EMA100割れ or RSI35割れ or 週足SMA20割れ → 降りる")
    print()
    print("  本質的なトレードオフ:")
    print(f"    B&H     : 最大リターン +{bah['ret']:.0f}%  ← でも最大DD -{abs(bah['dd']):.0f}%")
    print(f"    v6戦略  : リターン    {r6['ret']:+.0f}%  ← でも最大DD  {r6['dd']:.0f}%")
    print(f"    → 元本を守りながら利益を積み上げる「守りながら攻める」設計")
    print("═"*W)

# ━━ main ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    df = load()
    df = indicators(df)

    print("\n[SIGNAL] シグナル生成中...")
    buy6, sell6 = signals_v6(df)
    buy5, sell5 = signals_v5(df)

    print("[BACKTEST] v6 トレンドライド...")
    r6 = backtest(df, buy6, sell6, label="v6 トレンドライド")

    print("[BACKTEST] v5 RSIクロス（比較用）...")
    r5 = backtest(df, buy5, sell5, label="v5 RSIクロス")

    print("[BACKTEST] バイ&ホールド...")
    bah = benchmark_bah(df)
    bah["label"] = "バイ&ホールド"

    dca = benchmark_dca(df)

    print("[YEARLY] 年次分析...")
    min_len    = min(len(r6["equity"]), len(r5["equity"]), len(bah["equity"]), len(df))
    df_tr      = df.iloc[:min_len]
    yearly_data= yearly(df_tr, r6, r5, bah)

    sig = now_signal(df, buy6, sell6)

    report(df, r6, r5, bah, dca, yearly_data, sig)
