"""
BTC-USD Strategy v7 — 実戦仕様（5つの改善を全て組み込み）

改善①: 取引コスト（手数料0.10% + スリッページ0.05%）
改善②: 押し目再エントリー（トレンド継続中に素早く乗り直す）
改善③: 相場サイクルフィルター（強気年は攻め、弱気年は守る）
改善④: 逆張りサブ戦略（RSI<22 極端な売られすぎに小ロットで仕込む）
改善⑤: 円建てP&L管理（USD/JPY実データで実際の損益を計算）
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os

INITIAL_JPY    = 500_000
FEE_RATE       = 0.001    # 片道 0.10%（取引所手数料）
SLIP_RATE      = 0.0005   # 片道 0.05%（スリッページ）
COST_ONEWAY    = FEE_RATE + SLIP_RATE   # 0.15% / 片道
COST_ROUNDTRIP = COST_ONEWAY * 2        # 0.30% / 往復

KELLY_MAIN     = 0.10     # メイン戦略 Half-Kelly（v6より少し保守的）
KELLY_REENTRY  = 0.08     # 再エントリー（やや小さく）
KELLY_CONTRA   = 0.025    # 逆張りサブ（小ロット）

ATR_STOP_MAIN  = 4.0
ATR_STOP_REENT = 3.5
ATR_STOP_CONT  = 2.5

CSV_BTC   = "/Users/hiroseren/btc_usd_daily_5y.csv"
CSV_FX    = "/Users/hiroseren/usdjpy_5y.csv"

# ━━ データ取得 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_btc():
    df = pd.read_csv(CSV_BTC, index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df.dropna(subset=["Close"])

def load_or_fetch_fx():
    if not os.path.exists(CSV_FX):
        print("[FX] USD/JPY 5年分を取得中...")
        fx = yf.download("USDJPY=X", period="5y", interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(fx.columns, pd.MultiIndex):
            fx.columns = fx.columns.get_level_values(0)
        fx.to_csv(CSV_FX)
    fx = pd.read_csv(CSV_FX, index_col=0, parse_dates=True)
    if isinstance(fx.columns, pd.MultiIndex):
        fx.columns = fx.columns.get_level_values(0)
    if fx.index.tz is not None:
        fx.index = fx.index.tz_localize(None)
    return fx["Close"].rename("usdjpy")

# ━━ 指標計算 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calc_indicators(df, fx):
    c = df["Close"].copy()
    h, l = df["High"], df["Low"]
    df = df.copy()

    # EMA
    for span, name in [(21,"ema21"),(50,"ema50"),(100,"ema100"),(200,"ema200")]:
        df[name] = c.ewm(span=span, adjust=False).mean()

    # 週足SMA20
    wsma = c.resample("W").last().rolling(20).mean()
    df["weekly_sma20"] = wsma.reindex(df.index, method="ffill")

    # RSI(14)
    d  = c.diff()
    g  = d.clip(lower=0).rolling(14).mean()
    lo = (-d.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + g / lo.replace(0, np.nan))

    # ATR(14)
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # ADX(14)
    up = h.diff(); dn = -l.diff()
    pdm = up.where((up>dn)&(up>0),0).rolling(14).mean()
    ndm = dn.where((dn>up)&(dn>0),0).rolling(14).mean()
    pdi = 100*pdm/df["atr"].replace(0,np.nan)
    ndi = 100*ndm/df["atr"].replace(0,np.nan)
    df["adx"]      = (100*(pdi-ndi).abs()/(pdi+ndi).replace(0,np.nan)).rolling(14).mean()
    df["minus_di"] = ndi

    # 実現ボラ(20日)
    df["rv20"] = c.pct_change().rolling(20).std() * np.sqrt(365)

    # EMA200 傾き（改善③: サイクル強度に使う）
    df["e200_slope_90d"] = (df["ema200"] - df["ema200"].shift(90)) / df["ema200"].shift(90) * 100

    # ATH / DD
    df["ath"]        = c.cummax()
    df["dd_from_ath"] = (c - df["ath"]) / df["ath"] * 100

    # USD/JPY を日次データに合わせる（改善⑤）
    df["usdjpy"] = fx.reindex(df.index, method="ffill").fillna(method="bfill")

    return df

# ━━ 改善③: サイクル強度スコア ━━━━━━━━━━━━━━━━━━━━━━━━━━

def cycle_strength(row):
    """
    EMA200の90日傾き & ATHからの距離 からサイクルフェーズを判定。
    強気フェーズ → ポジションサイズ最大1.5倍
    弱気フェーズ → ポジションサイズ最小0.6倍
    """
    slope = row.get("e200_slope_90d", 0) if not np.isnan(row.get("e200_slope_90d", np.nan)) else 0
    dd    = row.get("dd_from_ath", 0)    if not np.isnan(row.get("dd_from_ath", np.nan))    else 0

    # 強気: EMA上昇 + ATH近い
    if slope > 5 and dd > -20:
        return 1.5
    elif slope > 2:
        return 1.2
    elif slope < -5 or dd < -50:
        return 0.6
    elif slope < -2:
        return 0.8
    return 1.0

# ━━ ポジションサイジング（コスト込み） ━━━━━━━━━━━━━━━━━━━

def pos_size(capital_usd, price, atr_v, rv_v, kelly, cyc=1.0):
    sl   = price - atr_v * ATR_STOP_MAIN
    risk = price - sl
    if risk <= 0 or capital_usd <= 0:
        return 0.0, sl

    vol_factor = np.clip(0.30 / max(rv_v, 0.05), 0.3, 2.0)
    size_risk  = (capital_usd * kelly * cyc) / risk
    size_vol   = (capital_usd * kelly * cyc * vol_factor) / risk
    return min(size_risk, size_vol, capital_usd / price * 0.99), sl

# ━━ コスト適用 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def apply_cost(price, direction):
    """方向: 'buy'=スリッページで高く買う, 'sell'=スリッページで安く売る"""
    if direction == "buy":
        return price * (1 + COST_ONEWAY)
    else:
        return price * (1 - COST_ONEWAY)

# ━━ バックテストエンジン ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_v7(df):
    """
    ポジション管理:
      main_pos  : メイン戦略のポジション
      cont_pos  : 逆張りサブのポジション（独立）
    資金は共有: capital_usd（USD建て）
    最終的に equity_jpy に変換
    """
    close   = df["Close"].values
    atr_    = df["atr"].values
    rsi_    = df["rsi"].values
    adx_    = df["adx"].values
    mdi_    = df["minus_di"].values
    e50_    = df["ema50"].values
    e100_   = df["ema100"].values
    e200_   = df["ema200"].values
    wsma_   = df["weekly_sma20"].values
    rv_     = df["rv20"].values
    slope_  = df["e200_slope_90d"].values
    dd_     = df["dd_from_ath"].values
    fx_     = df["usdjpy"].values
    dates   = df.index

    # ── 資金（USD建て）
    capital_usd = INITIAL_JPY / fx_[0]

    # ── ポジション状態
    m = {"qty":0.0, "entry":0.0, "stop":0.0, "trail":0.0, "tp":False, "reentry_ok":False}
    s = {"qty":0.0, "entry":0.0, "stop":0.0}  # 逆張りサブ

    equity_usd = []
    trades = []
    total_cost_usd = 0.0

    def _get(arr, i):
        v = float(arr[i]) if i < len(arr) else np.nan
        return 0.0 if np.isnan(v) else v

    for i in range(len(df)):
        px     = float(close[i])
        atr_v  = _get(atr_, i)
        rsi_v  = _get(rsi_, i)
        adx_v  = _get(adx_, i)
        mdi_v  = _get(mdi_, i)
        e50_v  = _get(e50_, i)
        e100_v = _get(e100_, i)
        e200_v = _get(e200_, i)
        wsma_v = _get(wsma_, i)
        rv_v_  = _get(rv_, i) or 0.3
        cyc_v  = cycle_strength({"e200_slope_90d": _get(slope_,i), "dd_from_ath": _get(dd_,i)})

        rsi_p  = _get(rsi_, i-1) if i > 0 else 50.0
        e50_p  = _get(e50_, i-1) if i > 0 else e50_v
        e200_p = _get(e200_, i-1) if i > 0 else e200_v
        px_p   = float(close[i-1]) if i > 0 else px

        eq = capital_usd + m["qty"] * px + s["qty"] * px
        equity_usd.append(eq)

        # ── トレーリングSL 更新
        if m["qty"] > 0:
            m["trail"] = max(m["trail"], px)
            m["stop"]  = max(m["stop"], m["trail"] - atr_v * ATR_STOP_MAIN)

        # ── メイン: ストップ判定
        if m["qty"] > 0 and px <= m["stop"]:
            exec_px = apply_cost(px, "sell")
            pnl     = (exec_px - m["entry"]) * m["qty"]
            cost    = exec_px * m["qty"] * COST_ONEWAY
            capital_usd += m["qty"] * exec_px
            total_cost_usd += cost
            trades.append({"date":dates[i],"type":"MAIN_SL","pnl":pnl,"px":exec_px})
            m["reentry_ok"] = (px > e200_v)   # ストップ後: トレンド上ならすぐ再エントリー可
            m.update({"qty":0.0,"entry":0.0,"stop":0.0,"trail":0.0,"tp":False})

        # ── メイン: 部分利食い（ATR×3到達で35%確定）
        if m["qty"] > 0 and not m["tp"] and px >= m["entry"] + atr_v * 3.0:
            cut     = m["qty"] * 0.35
            exec_px = apply_cost(px, "sell")
            pnl     = (exec_px - m["entry"]) * cut
            cost    = exec_px * cut * COST_ONEWAY
            capital_usd += cut * exec_px
            total_cost_usd += cost
            m["qty"] -= cut
            m["tp"]   = True
            trades.append({"date":dates[i],"type":"PARTIAL_TP","pnl":pnl,"px":exec_px})

        # ── メイン: 売りシグナル
        cross_below_e100 = (px_p >= e100_v) and (px < e100_v)
        rsi_collapse     = (rsi_p >= 35) and (rsi_v < 35)
        macro_break      = (px < wsma_v) and (wsma_v > 0)
        sell_sig         = cross_below_e100 or rsi_collapse or macro_break

        if m["qty"] > 0 and sell_sig:
            exec_px = apply_cost(px, "sell")
            pnl     = (exec_px - m["entry"]) * m["qty"]
            cost    = exec_px * m["qty"] * COST_ONEWAY
            capital_usd += m["qty"] * exec_px
            total_cost_usd += cost
            trades.append({"date":dates[i],"type":"MAIN_EXIT","pnl":pnl,"px":exec_px})
            m["reentry_ok"] = (px > e200_v)
            m.update({"qty":0.0,"entry":0.0,"stop":0.0,"trail":0.0,"tp":False})

        # ── 改善②: 再エントリーシグナル（条件を絞って過取引を防止）
        # 条件: 前回exit後でreentry_ok=True
        #   + EMA200上 かつ EMA200の傾きが正（上昇トレンド継続中）
        #   + RSIが50を下から上抜け（モメンタム回復）
        #   + EMA50がEMA200より上（ゴールデンクロス状態を維持）
        reentry_sig = (
            m["qty"] == 0 and m["reentry_ok"] and
            px > e200_v and _get(slope_, i) > 0 and
            e50_v > e200_v and                      # ← 追加: ゴールデンクロス状態
            (rsi_p < 50) and (rsi_v >= 50)
        )
        if reentry_sig:
            size, sl = pos_size(capital_usd, px, atr_v, rv_v_, KELLY_REENTRY, cyc_v)
            if size > 0:
                exec_px = apply_cost(px, "buy")
                cost    = exec_px * size * COST_ONEWAY
                capital_usd -= size * exec_px
                total_cost_usd += cost
                m.update({"qty":size,"entry":exec_px,"stop":sl,"trail":px,"tp":False,"reentry_ok":False})
                trades.append({"date":dates[i],"type":"REENTRY_BUY","pnl":0,"px":exec_px})

        # ── メイン: 買いシグナル（EMA200上抜け or ゴールデンクロス）
        cross_above_e200 = (px_p <= e200_p) and (px > e200_v)
        golden_cross     = (e50_p <= e200_p) and (e50_v > e200_v)
        buy_sig          = cross_above_e200 or golden_cross

        if m["qty"] == 0 and buy_sig and not m["reentry_ok"]:
            size, sl = pos_size(capital_usd, px, atr_v, rv_v_, KELLY_MAIN, cyc_v)
            if size > 0:
                exec_px = apply_cost(px, "buy")
                cost    = exec_px * size * COST_ONEWAY
                capital_usd -= size * exec_px
                total_cost_usd += cost
                m.update({"qty":size,"entry":exec_px,"stop":sl,"trail":px,"tp":False,"reentry_ok":False})
                trades.append({"date":dates[i],"type":"MAIN_BUY","pnl":0,"px":exec_px})

        # ── 改善④: 逆張りサブ戦略
        # 条件: RSI<22 + ADX下降中(-DI上昇 = 売り圧力弱まり) + EMA200が90日前より上（長期上昇トレンド）
        slope_v = _get(slope_, i)
        contra_buy = (
            s["qty"] == 0 and m["qty"] == 0 and   # 両方ノーポジ
            rsi_v < 22 and
            adx_v > 15 and mdi_v < _get(mdi_, i-3) if i >= 3 else False and
            slope_v > -10                          # 長期的に崩壊していない
        )
        if s["qty"] == 0 and m["qty"] == 0 and rsi_v < 22 and slope_v > -10:
            # -DIが直近3日で減少（売り圧力が緩んできた）
            mdi_3d_ago = _get(mdi_, max(0, i-3))
            if mdi_v < mdi_3d_ago:
                size = min(
                    (capital_usd * KELLY_CONTRA) / (atr_v * ATR_STOP_CONT),
                    capital_usd / px * 0.25
                )
                if size > 0:
                    exec_px = apply_cost(px, "buy")
                    sl_c    = px - atr_v * ATR_STOP_CONT
                    cost    = exec_px * size * COST_ONEWAY
                    capital_usd -= size * exec_px
                    total_cost_usd += cost
                    s.update({"qty":size,"entry":exec_px,"stop":sl_c})
                    trades.append({"date":dates[i],"type":"CONTRA_BUY","pnl":0,"px":exec_px})

        # ── 逆張りサブ: ストップ / 利食い
        if s["qty"] > 0:
            contra_sl   = px <= s["stop"]
            contra_exit = rsi_v >= 45  # RSI 45以上で利確

            if contra_sl or contra_exit:
                exec_px = apply_cost(px, "sell")
                pnl     = (exec_px - s["entry"]) * s["qty"]
                cost    = exec_px * s["qty"] * COST_ONEWAY
                capital_usd += s["qty"] * exec_px
                total_cost_usd += cost
                ttype = "CONTRA_SL" if contra_sl else "CONTRA_EXIT"
                trades.append({"date":dates[i],"type":ttype,"pnl":pnl,"px":exec_px})
                s.update({"qty":0.0,"entry":0.0,"stop":0.0})

    # 未決済を最終値で決済
    for pos, ptype in [(m,"MAIN"),(s,"SUB")]:
        if pos["qty"] > 0:
            px_  = float(close[-1])
            exec_px = apply_cost(px_, "sell")
            pnl  = (exec_px - pos["entry"]) * pos["qty"]
            cost = exec_px * pos["qty"] * COST_ONEWAY
            capital_usd += pos["qty"] * exec_px
            total_cost_usd += cost
            trades.append({"date":dates[-1],"type":f"{ptype}_FINAL","pnl":pnl,"px":exec_px})

    # ── 円建て変換
    eq_usd = np.array(equity_usd)
    fx_arr = np.array([float(f) if not np.isnan(float(f)) else 150.0 for f in fx_])
    fx_arr[:len(eq_usd)]  # 長さを合わせる
    min_len = min(len(eq_usd), len(fx_arr))
    eq_jpy  = eq_usd[:min_len] * fx_arr[:min_len]
    final_jpy = float(capital_usd * fx_arr[-1])

    # ── 統計
    pk   = np.maximum.accumulate(eq_jpy)
    dd   = float(((eq_jpy - pk) / pk).min() * 100)
    br   = np.diff(eq_jpy) / eq_jpy[:-1]
    sh   = float(br.mean() / br.std() * np.sqrt(365)) if br.std() > 0 else 0
    ret  = (final_jpy - INITIAL_JPY) / INITIAL_JPY * 100

    pnls = [t["pnl"] * fx_arr[min(df.index.get_loc(t["date"]), len(fx_arr)-1)]
            if t["pnl"] != 0 else 0 for t in trades]
    wins  = [p for p in pnls if p > 0]
    loss_ = [p for p in pnls if p < 0]
    wr    = len(wins)/len(pnls)*100 if pnls else 0
    pf    = sum(wins)/abs(sum(loss_)) if loss_ else 99
    calmar= abs(ret/dd) if dd != 0 else 0

    n_main    = sum(1 for t in trades if "MAIN_BUY"  in t["type"])
    n_reentry = sum(1 for t in trades if "REENTRY"   in t["type"])
    n_contra  = sum(1 for t in trades if "CONTRA_BUY"in t["type"])

    return {
        "final_jpy":    round(final_jpy, 0),
        "ret":          round(ret, 2),
        "dd":           round(dd, 2),
        "sharpe":       round(sh, 2),
        "wr":           round(wr, 1),
        "pf":           round(pf, 2),
        "calmar":       round(calmar, 2),
        "n_trades":     len([t for t in trades if "BUY" in t["type"]]),
        "n_main":       n_main,
        "n_reentry":    n_reentry,
        "n_contra":     n_contra,
        "total_cost_jpy": round(total_cost_usd * float(fx_arr[-1]), 0),
        "equity_jpy":   eq_jpy.tolist(),
        "trades":       trades,
    }

# ━━ v6 を同条件で再現（コスト込み・為替込み）━━━━━━━━━━━━━━━

def run_v6_with_costs(df):
    """v6と同じシグナルだが、コストと為替を追加して公平比較"""
    close  = df["Close"].values
    atr_   = df["atr"].values
    rsi_   = df["rsi"].values
    e50_   = df["ema50"].values
    e100_  = df["ema100"].values
    e200_  = df["ema200"].values
    wsma_  = df["weekly_sma20"].values
    rv_    = df["rv20"].values
    fx_    = df["usdjpy"].values
    dates  = df.index

    capital_usd = INITIAL_JPY / fx_[0]
    pos = {"qty":0.0,"entry":0.0,"stop":0.0,"trail":0.0,"tp":False}
    equity_usd = []
    trades = []
    total_cost = 0.0

    def _g(arr, i): return float(arr[i]) if not np.isnan(float(arr[i])) else 0.0

    for i in range(len(df)):
        px    = float(close[i])
        atr_v = _g(atr_, i) or px*0.03
        rsi_v = _g(rsi_, i) or 50
        rsi_p = _g(rsi_, i-1) if i>0 else 50
        e50_v = _g(e50_, i); e50_p = _g(e50_, i-1) if i>0 else e50_v
        e100_v= _g(e100_, i)
        e200_v= _g(e200_, i); e200_p= _g(e200_, i-1) if i>0 else e200_v
        wsma_v= _g(wsma_, i)
        rv_v_ = _g(rv_, i) or 0.3
        px_p  = float(close[i-1]) if i>0 else px

        equity_usd.append(capital_usd + pos["qty"] * px)

        if pos["qty"]>0:
            pos["trail"] = max(pos["trail"], px)
            pos["stop"]  = max(pos["stop"], pos["trail"] - atr_v*4.0)

        if pos["qty"]>0 and px<=pos["stop"]:
            ep = apply_cost(px,"sell"); cost=ep*pos["qty"]*COST_ONEWAY
            capital_usd += pos["qty"]*ep; total_cost+=cost
            trades.append({"type":"SL","pnl":(ep-pos["entry"])*pos["qty"]})
            pos.update({"qty":0.0,"entry":0.0,"stop":0.0,"trail":0.0,"tp":False})

        if pos["qty"]>0 and not pos["tp"] and px>=pos["entry"]+atr_v*3.0:
            cut=pos["qty"]*0.35; ep=apply_cost(px,"sell"); cost=ep*cut*COST_ONEWAY
            capital_usd+=cut*ep; total_cost+=cost; pos["qty"]-=cut; pos["tp"]=True

        sell = ((px_p>=e100_v and px<e100_v) or
                (rsi_p>=35 and rsi_v<35) or
                (px<wsma_v and wsma_v>0))
        if pos["qty"]>0 and sell:
            ep=apply_cost(px,"sell"); cost=ep*pos["qty"]*COST_ONEWAY
            capital_usd+=pos["qty"]*ep; total_cost+=cost
            trades.append({"type":"EXIT","pnl":(ep-pos["entry"])*pos["qty"]})
            pos.update({"qty":0.0,"entry":0.0,"stop":0.0,"trail":0.0,"tp":False})

        cross_up = (px_p<=e200_p and px>e200_v)
        gc       = (e50_p<=e200_p and e50_v>e200_v)
        if pos["qty"]==0 and (cross_up or gc):
            sl=px-atr_v*4.0; risk=px-sl
            if risk>0:
                size=min((capital_usd*KELLY_MAIN)/risk, capital_usd/px*0.99)
                if size>0:
                    ep=apply_cost(px,"buy"); cost=ep*size*COST_ONEWAY
                    capital_usd-=size*ep; total_cost+=cost
                    pos.update({"qty":size,"entry":ep,"stop":sl,"trail":px,"tp":False})

    if pos["qty"]>0:
        ep=apply_cost(float(close[-1]),"sell"); cost=ep*pos["qty"]*COST_ONEWAY
        capital_usd+=pos["qty"]*ep; total_cost+=cost
        trades.append({"type":"FINAL","pnl":(ep-pos["entry"])*pos["qty"]})

    fx_arr = np.array([float(f) if not np.isnan(float(f)) else 150.0 for f in fx_])
    eq_usd = np.array(equity_usd)
    min_len = min(len(eq_usd), len(fx_arr))
    eq_jpy  = eq_usd[:min_len] * fx_arr[:min_len]
    final   = capital_usd * float(fx_arr[-1])

    pk  = np.maximum.accumulate(eq_jpy)
    dd  = float(((eq_jpy-pk)/pk).min()*100)
    br  = np.diff(eq_jpy)/eq_jpy[:-1]
    sh  = float(br.mean()/br.std()*np.sqrt(365)) if br.std()>0 else 0
    ret = (final-INITIAL_JPY)/INITIAL_JPY*100
    pnls=[t["pnl"] for t in trades if t["pnl"]!=0]
    wins=[p for p in pnls if p>0]; loss=[p for p in pnls if p<0]
    wr  = len(wins)/len(pnls)*100 if pnls else 0
    pf  = sum(wins)/abs(sum(loss)) if loss else 99
    return {"final_jpy":round(final,0),"ret":round(ret,2),"dd":round(dd,2),
            "sharpe":round(sh,2),"wr":round(wr,1),"pf":round(pf,2),
            "n_trades":len([t for t in trades if t.get("type","") not in ("SL","EXIT","FINAL")]),
            "total_cost_jpy":round(total_cost*float(fx_arr[-1]),0),
            "equity_jpy":eq_jpy.tolist()}

# ━━ 年次ブレークダウン ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def yearly(df, *results):
    rows = []
    for yr in sorted(df.index.year.unique()):
        mask = df.index.year == yr
        if mask.sum() < 20: continue
        row = {"year": yr}
        for res in results:
            eq = np.array(res["equity_jpy"])[mask[:len(res["equity_jpy"])]]
            if len(eq)<2: continue
            r = (eq[-1]-eq[0])/eq[0]*100
            pk= np.maximum.accumulate(eq)
            dd= float(((eq-pk)/pk).min()*100)
            row[res.get("label","?")] = {"ret":round(r,1),"dd":round(dd,1)}
        rows.append(row)
    return rows

# ━━ バイ&ホールド（為替込み） ━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def bah_with_fx(df):
    c   = df["Close"].values
    fx_ = df["usdjpy"].values
    fx0 = float(fx_[0]) if not np.isnan(fx_[0]) else 150
    units = INITIAL_JPY / (float(c[0]) * fx0)
    fx_arr= np.array([float(f) if not np.isnan(float(f)) else 150.0 for f in fx_])
    eq_jpy= units * c[:len(fx_arr)] * fx_arr[:len(c)]
    pk    = np.maximum.accumulate(eq_jpy)
    dd    = float(((eq_jpy-pk)/pk).min()*100)
    br    = np.diff(eq_jpy)/eq_jpy[:-1]
    sh    = float(br.mean()/br.std()*np.sqrt(365)) if br.std()>0 else 0
    final = units * float(c[-1]) * float(fx_arr[-1])
    ret   = (final - INITIAL_JPY)/INITIAL_JPY*100
    return {"label":"B&H(為替込)","final_jpy":round(final,0),"ret":round(ret,2),
            "dd":round(dd,2),"sharpe":round(sh,2),"equity_jpy":eq_jpy.tolist()}

# ━━ レポート出力 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def report(df, v7, v6c, bah, yearly_data):
    W = 72
    print("\n" + "═"*W)
    print("  BTC-USD v7 バックテスト（5つの改善を全て組み込み）")
    print(f"  期間: {df.index[0].date()} ～ {df.index[-1].date()}")
    print("═"*W)

    # ── 1. 総合比較
    print("\n【1】5年間パフォーマンス比較（円建て、初期資金50万円）")
    print(f"  {'戦略':<22} {'最終資産':>12} {'収益率':>8} {'MaxDD':>7} "
          f"{'Sharpe':>7} {'PF':>5} {'取引':>4} {'コスト':>8}")
    print("  " + "─"*70)

    for r, lbl in [(bah,"B&H(為替込)"),(v6c,"v6+コスト+為替"),(v7,"v7(全改善)")]:
        ar = "▲" if r["ret"]>=0 else "▼"
        pf_s = f"{r.get('pf',0):.2f}" if r.get('pf') else "  —"
        n_s  = str(r.get("n_trades","—"))
        ct_s = f"{r.get('total_cost_jpy',0):,.0f}円" if r.get("total_cost_jpy") else "  —"
        print(f"  {lbl:<22} {r['final_jpy']:>12,.0f}円 "
              f"{ar}{abs(r['ret']):>6.1f}% "
              f"{r['dd']:>6.1f}% "
              f"{r['sharpe']:>6.2f} "
              f"{pf_s:>5} "
              f"{n_s:>4} "
              f"{ct_s:>8}")

    # コスト影響
    if v7.get("total_cost_jpy"):
        cost_pct = v7["total_cost_jpy"] / INITIAL_JPY * 100
        print(f"\n  取引コスト合計: {v7['total_cost_jpy']:,.0f}円 ({cost_pct:.1f}%)")
        print(f"  → コスト未考慮版との差分がリアルな「摩擦コスト」")

    # v6 vs v7 改善効果
    print(f"\n  v6(コスト込) → v7 改善効果:")
    ret_diff = v7["ret"] - v6c["ret"]
    dd_diff  = abs(v6c["dd"]) - abs(v7["dd"])
    sh_diff  = v7["sharpe"] - v6c["sharpe"]
    print(f"    収益率: {v6c['ret']:+.1f}% → {v7['ret']:+.1f}%  ({ret_diff:+.1f}pt)")
    print(f"    最大DD: {v6c['dd']:.1f}% → {v7['dd']:.1f}%   ({dd_diff:+.1f}pt 改善)")
    print(f"    Sharpe: {v6c['sharpe']:.2f} → {v7['sharpe']:.2f}          ({sh_diff:+.2f})")

    # ── 2. v7 内訳
    print(f"\n【2】v7 シグナル内訳")
    print(f"  メイン買い    : {v7['n_main']:>3}回（EMA200上抜け/GC）")
    print(f"  再エントリー  : {v7['n_reentry']:>3}回  ← 改善②（押し目乗り直し）")
    print(f"  逆張りサブ    : {v7['n_contra']:>3}回  ← 改善④（RSI<22 底圏仕込み）")

    # ── 3. 年次ブレークダウン
    print(f"\n【3】年次ブレークダウン（円建て）")
    v7_k  = "v7"
    v6c_k = "v6c"
    bah_k = "B&H(為替込)"
    print(f"  {'年':>4}  {'v7':>8}  {'v6+コスト':>9}  {'B&H':>8}  改善")
    print("  " + "─"*58)
    for row in yearly_data:
        yr   = row["year"]
        v7r  = row.get(v7_k,  {}).get("ret", 0)
        v6r  = row.get(v6c_k, {}).get("ret", 0)
        bahr = row.get(bah_k, {}).get("ret", 0)
        diff = v7r - v6r
        note = ""
        if bahr < -30:   note = "← 暴落防衛"
        elif diff > 5:   note = f"← +{diff:.0f}pt改善（再エントリー効果）"
        elif diff < -3:  note = f"← {diff:.0f}pt悪化"
        a7  = "▲" if v7r>=0 else "▼"
        a6  = "▲" if v6r>=0 else "▼"
        ab  = "▲" if bahr>=0 else "▼"
        print(f"  {yr:>4}  {a7}{abs(v7r):>5.0f}%  "
              f"{a6}{abs(v6r):>6.0f}%   "
              f"{ab}{abs(bahr):>5.0f}%  {note}")

    # ── 4. 改善別の効果分解
    print(f"\n【4】改善別の効果（定性評価）")
    items = [
        ("改善①", "取引コスト組込み",
         f"実態が見えるようになった。{v7['total_cost_jpy']:,.0f}円({v7['total_cost_jpy']/INITIAL_JPY*100:.1f}%)の摩擦コスト"),
        ("改善②", "押し目再エントリー",
         f"{v7['n_reentry']}回のリエントリーを追加。強気相場での取り逃がしを削減"),
        ("改善③", "サイクルフィルター",
         "強気フェーズで1.5倍、弱気フェーズで0.6倍のポジションに自動調整"),
        ("改善④", "逆張りサブ戦略",
         f"{v7['n_contra']}回の底圏仕込み。RSI<22の極値で小ロット参戦"),
        ("改善⑤", "円建てP&L",
         f"現在のUSD/JPY≈{df['usdjpy'].iloc[-1]:.0f}円。為替で±5〜10%の実質差異が見える"),
    ]
    for code, name, effect in items:
        print(f"  {code} {name:<18} : {effect}")

    # ── 5. 現在のシグナル（v7ベース）
    last     = df.iloc[-1]
    px       = float(last["Close"])
    e200     = float(last["ema200"])
    rsi_v    = float(last["rsi"]) if not np.isnan(last["rsi"]) else 0
    slope_v  = float(last["e200_slope_90d"]) if not np.isnan(last["e200_slope_90d"]) else 0
    mdi_v    = float(last["minus_di"]) if not np.isnan(last["minus_di"]) else 0
    mdi_3    = float(df["minus_di"].iloc[-4]) if not np.isnan(df["minus_di"].iloc[-4]) else mdi_v
    usdjpy_v = float(last["usdjpy"])
    cyc      = cycle_strength({"e200_slope_90d": slope_v, "dd_from_ath": float(last["dd_from_ath"])})
    atr_v    = float(last["atr"]) if not np.isnan(last["atr"]) else 0

    contra_possible = (rsi_v < 22 and slope_v > -10 and mdi_v < mdi_3)

    print(f"\n【5】現在のv7シグナル")
    print(f"  BTC価格    : ${px:>10,.0f}  (≈ {px*usdjpy_v:,.0f}円)")
    print(f"  EMA200     : ${e200:>10,.0f}  ({'上' if px>e200 else '下 ←待機中'})")
    print(f"  RSI(14)    : {rsi_v:>10.1f}  ({'🟠 底圏' if rsi_v<25 else ''})")
    print(f"  サイクル強度: {cyc:>10.1f}x  ({'強気(1.5x)' if cyc>=1.5 else '弱気(0.6x)' if cyc<=0.6 else '中立'})")
    print(f"  USD/JPY    : {usdjpy_v:>10.2f}円")
    print()
    if contra_possible:
        size_contra = (INITIAL_JPY / usdjpy_v) * KELLY_CONTRA / (atr_v * ATR_STOP_CONT)
        jpy_val     = size_contra * px * usdjpy_v
        print(f"  🟠 改善④逆張り条件が揃いつつあります")
        print(f"     推奨サイズ: {size_contra:.4f} BTC (≈{jpy_val:,.0f}円相当)")
        print(f"     ストップ: ${px - atr_v*ATR_STOP_CONT:,.0f}")
        print(f"     利食い目標: RSI≥45")
    else:
        print(f"  🔴 全シグナルなし。ノーポジション維持。")
        if rsi_v < 30:
            print(f"     （RSI={rsi_v:.1f}は底圏だが-DIがまだ下降中）")
    print("═"*W)

# ━━ main ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("[LOAD] データ読込...")
    df = load_btc()
    fx = load_or_fetch_fx()
    df = calc_indicators(df, fx)
    print(f"  BTC: {len(df):,}日  USD/JPY: {len(fx):,}日")

    print("[BACKTEST] v7 実行中（5つの改善）...")
    v7 = run_v7(df)
    v7["label"] = "v7"

    print("[BACKTEST] v6+コスト+為替（比較用）...")
    v6c = run_v6_with_costs(df)
    v6c["label"] = "v6c"

    print("[BACKTEST] B&H+為替...")
    bah = bah_with_fx(df)

    print("[YEARLY] 年次集計...")
    min_len = min(len(v7["equity_jpy"]), len(v6c["equity_jpy"]),
                  len(bah["equity_jpy"]), len(df))
    df_t = df.iloc[:min_len]
    v7_  = {**v7,  "equity_jpy": v7["equity_jpy"][:min_len]}
    v6c_ = {**v6c, "equity_jpy": v6c["equity_jpy"][:min_len]}
    bah_ = {**bah, "equity_jpy": bah["equity_jpy"][:min_len]}
    yr   = yearly(df_t, v7_, v6c_, bah_)

    report(df, v7, v6c, bah, yr)
