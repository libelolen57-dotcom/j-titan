"""
BTC Trading Dashboard
使い方: python3 btc_dashboard.py
毎朝実行して「今日何をすべきか」を確認する

出力:
  1. 多時間足シグナルスコア（月足/週足/日足）
  2. エントリー条件トラッカー（各条件まであと何%か）
  3. 歴史的コンテキスト（今と似た状況で過去どうなったか）
  4. シナリオ分析（BTC が X 円になったら資産はいくらか）
  5. 具体的アクションプラン（今日すること）
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

CAPITAL      = 500_000    # 現在の手元資金
CSV_DAILY    = "/Users/hiroseren/btc_usd_daily_5y.csv"
REFRESH_DAYS = 1          # 1日以上古ければ再取得

# ━━ データ取得 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_data():
    need_refresh = True
    if os.path.exists(CSV_DAILY):
        age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(CSV_DAILY))
        need_refresh = age.days >= REFRESH_DAYS

    if need_refresh:
        print("[DATA] 最新データを取得中...")
        df = yf.download("BTC-USD", period="5y", interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if not df.empty:
            df.to_csv(CSV_DAILY)
    else:
        df = pd.read_csv(CSV_DAILY, index_col=0, parse_dates=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df.dropna(subset=["Close"])

# ━━ 全指標を計算 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calc_all(df):
    c = df["Close"].copy()
    h, l = df["High"], df["Low"]

    # EMA / SMA
    df["ema21"]  = c.ewm(span=21,  adjust=False).mean()
    df["ema50"]  = c.ewm(span=50,  adjust=False).mean()
    df["ema100"] = c.ewm(span=100, adjust=False).mean()
    df["ema200"] = c.ewm(span=200, adjust=False).mean()
    df["sma200"] = c.rolling(200).mean()

    # 月足 EMA12 (約250日を12ヶ月に換算)
    monthly = c.resample("ME").last()
    m_ema12 = monthly.ewm(span=12, adjust=False).mean()
    df["monthly_ema12"] = m_ema12.reindex(df.index, method="ffill")

    # 週足 SMA20
    weekly_sma = c.resample("W").last().rolling(20).mean()
    df["weekly_sma20"] = weekly_sma.reindex(df.index, method="ffill")

    # RSI
    d  = c.diff()
    g  = d.clip(lower=0).rolling(14).mean()
    lo = (-d.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + g / lo.replace(0, np.nan))

    # 週足RSI
    wc = c.resample("W").last()
    wd = wc.diff()
    wg = wd.clip(lower=0).rolling(14).mean()
    wl = (-wd.clip(upper=0)).rolling(14).mean()
    wrsi = 100 - 100 / (1 + wg / wl.replace(0, np.nan))
    df["weekly_rsi"] = wrsi.reindex(df.index, method="ffill")

    # ATR / ボラ
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["rv20"] = c.pct_change().rolling(20).std() * np.sqrt(365)

    # ADX
    up = h.diff(); dn = -l.diff()
    pdm = up.where((up>dn)&(up>0), 0).rolling(14).mean()
    ndm = dn.where((dn>up)&(dn>0), 0).rolling(14).mean()
    atr14 = df["atr"]
    pdi = 100 * pdm / atr14.replace(0, np.nan)
    ndi = 100 * ndm / atr14.replace(0, np.nan)
    df["adx"]  = (100*(pdi-ndi).abs()/(pdi+ndi).replace(0,np.nan)).rolling(14).mean()
    df["plus_di"]  = pdi
    df["minus_di"] = ndi

    # ATH / ドローダウン
    df["ath"] = c.cummax()
    df["dd_from_ath"] = (c - df["ath"]) / df["ath"] * 100

    # ボリンジャーバンド（週足 20SMA ±2σ）
    wc20 = c.rolling(20).mean()
    ws20 = c.rolling(20).std()
    df["bb_upper"] = wc20 + 2 * ws20
    df["bb_lower"] = wc20 - 2 * ws20

    return df

# ━━ 多時間足シグナルスコア ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def multi_tf_score(df):
    """
    月足/週足/日足それぞれのブル度を 0-100 で採点。
    合計スコア: 0=全力ベア, 100=全力ブル
    """
    last = df.iloc[-1]
    c    = float(last["Close"])

    scores = {}

    # ── 月足スコア (0-40点)
    m_ema12 = float(last["monthly_ema12"])
    m_above = c > m_ema12
    m_pts   = 40 if m_above else 0
    scores["monthly"] = {
        "score": m_pts, "max": 40,
        "label": "月足 EMA12",
        "value": f"${c:,.0f} vs EMA12 ${m_ema12:,.0f}",
        "bull": m_above,
    }

    # ── 週足スコア (0-35点)
    w_sma   = float(last["weekly_sma20"])
    w_rsi   = float(last["weekly_rsi"]) if not np.isnan(last["weekly_rsi"]) else 50
    w_above = c > w_sma
    w_rsi_ok = w_rsi > 45

    w_pts = 0
    if w_above:   w_pts += 20
    if w_rsi_ok:  w_pts += 15
    scores["weekly"] = {
        "score": w_pts, "max": 35,
        "label": "週足 SMA20 + RSI",
        "value": f"SMA20 ${w_sma:,.0f}  RSI {w_rsi:.1f}",
        "bull": w_above and w_rsi_ok,
    }

    # ── 日足スコア (0-25点)
    e200   = float(last["ema200"])
    e100   = float(last["ema100"])
    rsi_v  = float(last["rsi"])
    adx_v  = float(last["adx"]) if not np.isnan(last["adx"]) else 0
    slope  = float(df["ema200"].iloc[-1] - df["ema200"].iloc[-15]) if len(df) > 15 else 0

    d_pts = 0
    if c > e200:    d_pts += 10
    if c > e100:    d_pts += 8
    if rsi_v > 50:  d_pts += 7
    scores["daily"] = {
        "score": d_pts, "max": 25,
        "label": "日足 EMA200/100 + RSI",
        "value": f"EMA200 ${e200:,.0f}  RSI {rsi_v:.1f}",
        "bull": c > e200,
    }

    total = sum(s["score"] for s in scores.values())
    total_max = sum(s["max"] for s in scores.values())

    return scores, total, total_max

# ━━ エントリー条件トラッカー ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def entry_tracker(df):
    last  = df.iloc[-1]
    c     = float(last["Close"])
    e200  = float(last["ema200"])
    e100  = float(last["ema100"])
    e50   = float(last["ema50"])
    wsma  = float(last["weekly_sma20"])
    mema  = float(last["monthly_ema12"])
    rsi_v = float(last["rsi"])
    slope = float(df["ema200"].iloc[-1] - df["ema200"].iloc[-15]) if len(df)>15 else 0

    conds = [
        {
            "name": "月足 EMA12 上抜け（マクロ強気）",
            "met":  c > mema,
            "target": mema,
            "pct_needed": (mema - c) / c * 100 if c < mema else 0,
            "priority": 1,
        },
        {
            "name": "週足 SMA20 上抜け（中期転換）",
            "met":  c > wsma,
            "target": wsma,
            "pct_needed": (wsma - c) / c * 100 if c < wsma else 0,
            "priority": 2,
        },
        {
            "name": "日足 EMA200 上抜け（トレンド転換）",
            "met":  c > e200 and slope > 0,
            "target": e200,
            "pct_needed": (e200 - c) / c * 100 if c < e200 else 0,
            "priority": 3,
        },
        {
            "name": "日足 RSI > 50（モメンタム回復）",
            "met":  rsi_v > 50,
            "target": 50,
            "pct_needed": 0,
            "priority": 4,
            "current": rsi_v,
        },
    ]
    return conds, c

# ━━ 歴史的コンテキスト ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def historical_context(df):
    """
    現在に似た過去の状況（RSI<25 かつ EMA200下）での
    その後 30/60/90 日のパフォーマンスを集計。
    """
    c    = df["Close"]
    rsi_ = df["rsi"]
    e200 = df["ema200"]
    atd  = df["dd_from_ath"]

    # 現在のATHドローダウン
    current_dd = float(atd.iloc[-1])
    current_rsi = float(rsi_.iloc[-1])

    # 類似条件: RSI < 30 かつ 価格 < EMA200
    similar = (rsi_ < 30) & (c < e200) & ~df.index.isin(df.index[-90:])

    results_30, results_60, results_90 = [], [], []
    for i in range(len(df) - 90):
        if not similar.iloc[i]:
            continue
        px_now = float(c.iloc[i])
        for future, lst in [(30, results_30), (60, results_60), (90, results_90)]:
            if i + future < len(df):
                px_fut = float(c.iloc[i + future])
                lst.append((px_fut - px_now) / px_now * 100)

    stats = {}
    for days, lst in [(30, results_30), (60, results_60), (90, results_90)]:
        if lst:
            stats[days] = {
                "n":      len(lst),
                "mean":   round(np.mean(lst), 1),
                "median": round(np.median(lst), 1),
                "positive_pct": round(sum(1 for x in lst if x > 0) / len(lst) * 100, 0),
                "p25":    round(np.percentile(lst, 25), 1),
                "p75":    round(np.percentile(lst, 75), 1),
                "best":   round(max(lst), 1),
                "worst":  round(min(lst), 1),
            }
        else:
            stats[days] = None

    # ATHドローダウン別の過去統計
    dd_buckets = [
        ("軽微 (-20%以内)",  -20,   0,  []),
        ("中程度 (-20〜-40%)", -40, -20, []),
        ("深刻 (-40〜-60%)",  -60, -40, []),
        ("壊滅 (-60%以上)",   -100, -60, []),
    ]
    for i in range(len(df) - 90):
        dd_val = float(atd.iloc[i])
        for label, lo, hi, lst in dd_buckets:
            if lo < dd_val <= hi:
                if i + 90 < len(df):
                    px_now = float(c.iloc[i])
                    px_fut = float(c.iloc[i + 90])
                    lst.append((px_fut - px_now) / px_now * 100)

    return stats, current_dd, current_rsi, dd_buckets

# ━━ シナリオ分析 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def scenario_analysis(current_price, atr_v, rv_v):
    """
    BTC が様々な価格に到達した場合の試算。
    エントリーは「EMA200 上抜け確認後」と仮定。
    """
    entry_signal_px = 80_821   # 現在の EMA200 ≈ 入場シグナル想定価格
    stop_distance   = atr_v * 4.0
    stop_px         = entry_signal_px - stop_distance

    # Kelly（v6バックテスト実績から）
    kelly_half = 0.12
    risk_per_trade = CAPITAL * kelly_half
    position_btc   = risk_per_trade / stop_distance if stop_distance > 0 else 0
    position_btc   = min(position_btc, CAPITAL / entry_signal_px)
    position_cost  = position_btc * entry_signal_px

    scenarios = [
        ("最悪（-30%下落継続）", current_price * 0.70, "🔴"),
        ("横ばい（現状維持）",   current_price * 1.00, "⚪"),
        ("弱回復（EMA200到達）", entry_signal_px,       "🟡"),
        ("中回復（+30%目標）",  current_price * 1.30,  "🟠"),
        ("強回復（前ATH回帰）", 108_000,               "🟢"),
        ("超強気（新ATH +20%）",130_000,               "🚀"),
    ]

    rows = []
    for label, target_px, icon in scenarios:
        ret_pct = (target_px - current_price) / current_price * 100
        # エントリーはEMA200上抜け後（entry_signal_px）、目標がentry以下なら未エントリー
        if target_px < entry_signal_px:
            trade_pnl = 0
            trade_ret = 0
            note = "シグナル未発生、ノーポジ"
        else:
            trade_pnl = (target_px - entry_signal_px) * position_btc
            trade_ret = (target_px - entry_signal_px) / entry_signal_px * 100
            note = f"保有BTC: {position_btc:.4f}枚"
        final_cap = CAPITAL - position_cost + position_cost + trade_pnl if target_px >= entry_signal_px else CAPITAL
        # より正確に: entry時に position_cost を消費、後に target_px × btc を受取
        final_cap = (CAPITAL - position_cost) + position_btc * target_px if target_px >= entry_signal_px else CAPITAL

        rows.append({
            "icon": icon, "label": label, "target": target_px,
            "btc_ret": round(ret_pct, 1),
            "trade_pnl": round(trade_pnl, 0),
            "final_cap": round(final_cap, 0),
            "cap_ret": round((final_cap - CAPITAL) / CAPITAL * 100, 1),
            "note": note,
        })
    return rows, position_btc, entry_signal_px, stop_px

# ━━ ウォッチリスト価格 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def watchlist(df):
    last  = df.iloc[-1]
    c     = float(last["Close"])
    e50   = float(last["ema50"])
    e100  = float(last["ema100"])
    e200  = float(last["ema200"])
    wsma  = float(last["weekly_sma20"])
    mema  = float(last["monthly_ema12"])
    atr_v = float(last["atr"])
    bbl   = float(last["bb_lower"])
    bbu   = float(last["bb_upper"])
    ath   = float(last["ath"])

    levels = [
        ("📈 月足EMA12（マクロ転換）",     mema,       "ここを超えたら積極参戦"),
        ("📈 EMA200（日足転換ライン）",     e200,       "v6戦略のエントリー条件"),
        ("📈 週足SMA20（中期転換）",       wsma,       "ここを週足終値で超えると強い"),
        ("📈 EMA100",                     e100,       "日足トレンド中期"),
        ("📊 EMA50",                      e50,        "短期トレンド"),
        ("📊 現在価格",                    c,          "←── イマここ"),
        ("📉 BB下限（逆張り候補）",        bbl,        "ここで反発するか注目"),
        ("📉 ATR×3 サポート",              c - atr_v*3, "ここを割ると下値加速"),
        ("🎯 目標 +30%",                  c * 1.30,   "最終目標"),
        ("🏔 前回ATH",                    ath,        f"${ath:,.0f}"),
    ]
    return sorted(levels, key=lambda x: x[1], reverse=True)

# ━━ メイン出力 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_dashboard(df):
    W = 68
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    last = df.iloc[-1]
    c    = float(last["Close"])
    atr_v= float(last["atr"])
    rv_v = float(last["rv20"])

    bar = "━" * W
    print(f"\n{'═'*W}")
    print(f"  📊 BTC/USD Trading Dashboard  {now}")
    print(f"{'═'*W}")

    # ── セクション1: 多時間足スコア
    scores, total, total_max = multi_tf_score(df)
    pct = total / total_max * 100

    if pct >= 75:   mood = "🟢 強気（エントリー可）"
    elif pct >= 50: mood = "🟡 中立（慎重に待機）"
    elif pct >= 25: mood = "🟠 弱気（戻り待ち）"
    else:           mood = "🔴 強い弱気（待機）"

    print(f"\n{'─'*W}")
    print(f"  【1】多時間足シグナルスコア")
    print(f"{'─'*W}")
    filled = int(total / total_max * 30)
    bar_s  = "█"*filled + "░"*(30-filled)
    print(f"  総合スコア: {total}/{total_max}点  [{bar_s}] {pct:.0f}%")
    print(f"  判定: {mood}")
    print()
    for key, s in scores.items():
        icon = "✓" if s["bull"] else "✗"
        pts_bar = "▌" * s["score"] + "·" * (s["max"] - s["score"])
        print(f"  {icon} {s['label']:<22} {s['score']:>2}/{s['max']}点  {pts_bar}")
        print(f"    {s['value']}")

    # ── セクション2: エントリー条件
    conds, cur_px = entry_tracker(df)
    met_count = sum(1 for c_item in conds if c_item["met"])

    print(f"\n{'─'*W}")
    print(f"  【2】エントリー条件トラッカー  ({met_count}/{len(conds)} 達成)")
    print(f"{'─'*W}")
    print(f"  現在価格: ${cur_px:>10,.0f}")
    print()
    for c_item in conds:
        icon = "✅" if c_item["met"] else "⬜"
        if c_item["met"]:
            status = "達成"
        else:
            if "current" in c_item:
                status = f"現在 {c_item['current']:.1f} → 目標 {c_item['target']:.0f}"
            else:
                status = f"あと +{c_item['pct_needed']:.1f}% (${c_item['target']:,.0f})"
        print(f"  {icon} {c_item['name']}")
        print(f"       {status}")

    # ── セクション3: 歴史的コンテキスト
    h_stats, dd_now, rsi_now, dd_buckets = historical_context(df)
    ath_px = float(last["ath"])

    print(f"\n{'─'*W}")
    print(f"  【3】歴史的コンテキスト")
    print(f"{'─'*W}")
    print(f"  ATHからの下落 : {dd_now:+.1f}%  (ATH: ${ath_px:,.0f})")
    print(f"  現在の RSI   : {rsi_now:.1f}  （底圏: <25, 過熱: >75）")
    print()
    print(f"  過去に「RSI<30 かつ EMA200下」だった時の統計:")
    print(f"  {'期間':<8} {'N':>4} {'上昇確率':>7} {'中央値':>8} {'25%-75%範囲':>16} {'最良/最悪':>16}")
    print(f"  {'─'*64}")
    for days in [30, 60, 90]:
        s = h_stats.get(days)
        if s:
            print(f"  {days}日後   {s['n']:>4}回  {s['positive_pct']:>5.0f}%   "
                  f"{s['median']:>+7.1f}%  "
                  f"  {s['p25']:>+5.1f}〜{s['p75']:>+5.1f}%  "
                  f"  {s['best']:>+6.1f} / {s['worst']:>+6.1f}%")
        else:
            print(f"  {days}日後   データ不足")

    print()
    print(f"  ATH下落幅別の 90日後中央値リターン:")
    for label, lo, hi, lst in dd_buckets:
        if lst:
            med = np.median(lst)
            pos = sum(1 for x in lst if x > 0) / len(lst) * 100
            bar_d = "█" * min(int(abs(med)/5), 12)
            print(f"  {label:<22} → 中央値 {med:>+6.1f}%  上昇確率 {pos:.0f}%  {bar_d}")
        else:
            print(f"  {label:<22} → データなし")

    # ── セクション4: シナリオ分析
    scenarios, pos_btc, entry_px, stop_px = scenario_analysis(cur_px, atr_v, rv_v)

    print(f"\n{'─'*W}")
    print(f"  【4】シナリオ分析（手元資金: {CAPITAL:,}円）")
    print(f"{'─'*W}")
    print(f"  想定エントリー: EMA200上抜け確認後 ≈ ${entry_px:,.0f}")
    print(f"  ポジションサイズ: {pos_btc:.4f} BTC  (Half-Kelly 12%)")
    print(f"  ストップライン: ${stop_px:,.0f}  (ATR×4 = ${atr_v*4:,.0f}下)")
    print()
    print(f"  {'シナリオ':<22} {'BTC目標価格':>12} {'BTC変動':>8} {'最終資産':>12} {'資産変動':>8}")
    print(f"  {'─'*66}")
    for s in scenarios:
        cap_arrow = "▲" if s["cap_ret"] >= 0 else "▼"
        btc_arrow = "▲" if s["btc_ret"] >= 0 else "▼"
        print(f"  {s['icon']} {s['label']:<20} ${s['target']:>10,.0f} "
              f"{btc_arrow}{abs(s['btc_ret']):>6.1f}%  "
              f"{s['final_cap']:>12,.0f}円 "
              f"{cap_arrow}{abs(s['cap_ret']):>6.1f}%")

    # ── セクション5: ウォッチリスト
    levels = watchlist(df)

    print(f"\n{'─'*W}")
    print(f"  【5】価格ウォッチリスト（アクション目安）")
    print(f"{'─'*W}")
    for name, price, note in levels:
        dist = (price - cur_px) / cur_px * 100
        marker = " ←── NOW" if abs(dist) < 1.5 else f"  ({dist:+.1f}%)"
        print(f"  {name:<28} ${price:>10,.0f}{marker}")
        if note and abs(dist) < 30:
            print(f"    ↳ {note}")

    # ── セクション6: 今日のアクションプラン
    print(f"\n{'═'*W}")
    print(f"  【6】今日のアクションプラン")
    print(f"{'═'*W}")

    # スコアに応じたアクション
    if total >= 75:
        print(f"  ✅ 全条件クリア。ポジション保有 or エントリー検討。")
        print(f"     推奨サイズ: {pos_btc:.4f} BTC  (≈ {pos_btc * cur_px:,.0f}円)")
        print(f"     ストップ: ${stop_px:,.0f}  利食い目標: ${cur_px * 1.20:,.0f} (部分)")
    elif total >= 50:
        print(f"  🟡 条件部分達成。小さめポジションで様子見可能。")
        print(f"     フルサイズの50%: {pos_btc * 0.5:.4f} BTC")
        print(f"     ストップは通常より広く: ${cur_px - atr_v * 5:,.0f}")
    elif total >= 25:
        print(f"  🟠 待機推奨。以下の条件達成で準備:")
        for c_item in conds:
            if not c_item["met"]:
                print(f"     • {c_item['name']}")
    else:
        print(f"  🔴 明確な待機。ノーポジションを維持。")
        # 最初に注目すべきレベル
        unmet = [c_item for c_item in conds if not c_item["met"]]
        if unmet:
            first = unmet[0]
            print(f"\n  次のアクションポイント:")
            if "current" not in first:
                print(f"    BTC が ${first['target']:,.0f} を超えたら再確認")
                print(f"    （現在から +{first['pct_needed']:.1f}%、約 ${first['target'] - cur_px:,.0f}上昇が必要）")

    # 今週の注目イベント
    print(f"\n  今週チェックすること:")
    print(f"  • 週足の終値が週足SMA20(${float(last['weekly_sma20']):,.0f})を上抜けるか")
    print(f"  • RSI が 35 を超えてきたら短期の底打ちシグナル")
    print(f"  • 出来高を確認（回復は高出来高を伴うと信頼性が高い）")

    # ATH ドローダウンの歴史的観点
    if dd_now < -30:
        print(f"\n  💡 長期投資家の視点:")
        print(f"     現在は ATH から {dd_now:.0f}% 下落地点。")
        if dd_now < -50:
            print(f"     過去のサイクルでは -50%〜-70% 付近が長期積立の好機でした。")
        elif dd_now < -35:
            print(f"     過去2サイクルの中間底付近。分割積立（DCA）の検討余地あり。")

    print(f"\n  次回確認: 明日 or 週足確定（毎週日曜）")
    print(f"{'═'*W}\n")

# ━━ main ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    df = get_data()
    df = calc_all(df)
    print_dashboard(df)
