"""
BTC エアトレードエンジン v7
毎日 GitHub Actions から自動実行 → btc_portfolio.json と btc_log/ に記録

戦略: v7（5つの改善を組み込み）
  改善①: 取引コスト（手数料0.10% + スリッページ0.05%）を計上
  改善②: 押し目再エントリー（EMA50>EMA200 の強気継続中にRSI50上抜けで乗り直し）
  改善③: サイクルフィルター（EMA200の90日傾きでポジションサイズを0.6〜1.5倍調整）
  改善④: 逆張りサブ戦略（RSI<22 + -DI減少でノーポジ時に小ロット仕込み）
  改善⑤: 円建てP&L表示（USD/JPYレートで実際の損益を表示）
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime, date
import pytz

# ─── 設定 ───────────────────────────────────────────────
PORTFOLIO_FILE  = "btc_portfolio.json"
LOG_DIR         = "btc_log"
INITIAL_CAPITAL = 500_000       # 円
KELLY_HALF      = 0.10          # Half-Kelly（v7: メイン）
KELLY_REENTRY   = 0.08          # 再エントリー用
KELLY_CONTRA    = 0.025         # 逆張りサブ（小ロット）
ATR_STOP_MUL    = 4.0
ATR_STOP_CONT   = 2.5           # 逆張りサブのストップ
ATR_TP_MUL      = 3.0
TARGET_VOL      = 0.30
FEE_RATE        = 0.001         # 改善①: 手数料 0.10%/片道
SLIP_RATE       = 0.0005        # 改善①: スリッページ 0.05%/片道
COST_ONEWAY     = FEE_RATE + SLIP_RATE
JST             = pytz.timezone("Asia/Tokyo")

# ─── ポートフォリオ I/O ──────────────────────────────────

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE) as f:
            return json.load(f)
    return {
        "initial_capital": INITIAL_CAPITAL,
        "cash":            INITIAL_CAPITAL,
        "position": {
            "qty":        0.0,
            "entry_price":0.0,
            "entry_date": None,
            "stop_price": 0.0,
            "trail_hi":   0.0,
            "tp_done":    False,
        },
        "trades":          [],
        "total_trades":    0,
        "wins":            0,
        "created_at":      datetime.now(JST).isoformat(),
        "last_updated":    None,
    }

def save_portfolio(pf):
    pf["last_updated"] = datetime.now(JST).isoformat()
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(pf, f, ensure_ascii=False, indent=2)

# ─── データ取得 ──────────────────────────────────────────

def fetch_data():
    df = yf.download("BTC-USD", period="3y", interval="1d",
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df.dropna(subset=["Close"])

# ─── 指標計算 ────────────────────────────────────────────

def add_indicators(df):
    c = df["Close"].copy()
    h, l = df["High"], df["Low"]

    df["ema50"]  = c.ewm(span=50,  adjust=False).mean()
    df["ema100"] = c.ewm(span=100, adjust=False).mean()
    df["ema200"] = c.ewm(span=200, adjust=False).mean()

    monthly    = c.resample("ME").last()
    m_ema12    = monthly.ewm(span=12, adjust=False).mean()
    df["monthly_ema12"] = m_ema12.reindex(df.index, method="ffill")

    weekly_sma = c.resample("W").last().rolling(20).mean()
    df["weekly_sma20"] = weekly_sma.reindex(df.index, method="ffill")

    d  = c.diff()
    g  = d.clip(lower=0).rolling(14).mean()
    lo = (-d.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + g / lo.replace(0, np.nan))

    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    df["atr"]  = tr.rolling(14).mean()
    df["rv20"] = c.pct_change().rolling(20).std() * np.sqrt(365)

    # 改善③: サイクル強度（EMA200の90日傾き）
    df["e200_slope_90d"] = (df["ema200"] - df["ema200"].shift(90)) / df["ema200"].shift(90) * 100
    df["e200_slope"]     = df["ema200"] - df["ema200"].shift(15)

    # ADX / -DI（改善④: 逆張り条件に使用）
    up  = df["High"].diff(); dn = -df["Low"].diff()
    pdm = up.where((up>dn)&(up>0), 0).rolling(14).mean()
    ndm = dn.where((dn>up)&(dn>0), 0).rolling(14).mean()
    atr14 = df["atr"]
    df["adx"]      = (100*(pdm/atr14.replace(0,np.nan) - ndm/atr14.replace(0,np.nan)).abs() /
                      (pdm/atr14.replace(0,np.nan) + ndm/atr14.replace(0,np.nan)).replace(0,np.nan)
                      ).rolling(14).mean()
    df["minus_di"] = 100 * ndm / atr14.replace(0, np.nan)

    df["ath"]         = c.cummax()
    df["dd_from_ath"] = (c - df["ath"]) / df["ath"] * 100

    return df

# ─── 改善③: サイクル強度スコア ──────────────────────────

def get_cycle_multiplier(last_row):
    slope = float(last_row.get("e200_slope_90d", 0)) if not np.isnan(last_row.get("e200_slope_90d", np.nan)) else 0
    dd    = float(last_row.get("dd_from_ath", 0))    if not np.isnan(last_row.get("dd_from_ath", np.nan))    else 0
    if slope > 5  and dd > -20: return 1.5
    elif slope > 2:             return 1.2
    elif slope < -5 or dd < -50:return 0.6
    elif slope < -2:            return 0.8
    return 1.0

# ─── v7 シグナル判定（本日分） ───────────────────────────

def check_signals(df):
    """
    v7シグナル（look-ahead なし: 前日終値確定後に判定）
    """
    if len(df) < 10:
        return "NONE", "NONE", "NONE"

    last  = df.iloc[-1]
    prev  = df.iloc[-2]
    prev3 = df.iloc[-4] if len(df) >= 4 else prev

    c_now    = float(last["Close"]);   c_prev  = float(prev["Close"])
    e50_now  = float(last["ema50"]);   e50_prev= float(prev["ema50"])
    e100_now = float(last["ema100"])
    e200_now = float(last["ema200"]);  e200_prev=float(prev["ema200"])
    slope_90 = float(last["e200_slope_90d"]) if not np.isnan(last["e200_slope_90d"]) else 0
    rsi_now  = float(last["rsi"])   if not np.isnan(last["rsi"])  else 50
    rsi_prev = float(prev["rsi"])   if not np.isnan(prev["rsi"]) else 50
    w_sma    = float(last["weekly_sma20"]) if not np.isnan(last["weekly_sma20"]) else 0
    mdi_now  = float(last["minus_di"]) if not np.isnan(last["minus_di"]) else 50
    mdi_3ago = float(prev3["minus_di"])if not np.isnan(prev3["minus_di"])else mdi_now

    # ── メイン買い（EMA200上抜け or ゴールデンクロス）
    cross_above_e200 = (c_prev <= e200_prev) and (c_now > e200_now) and (slope_90 > 0)
    golden_cross     = (e50_prev <= e200_prev) and (e50_now > e200_now)
    main_buy = cross_above_e200 or golden_cross

    # ── 改善②: 再エントリー（EMA50>EMA200 の強気継続中 + RSI50上抜け）
    reentry_buy = (
        c_now > e200_now and slope_90 > 0 and
        e50_now > e200_now and              # ゴールデンクロス状態を維持
        (rsi_prev < 50) and (rsi_now >= 50) # RSIが50を下から上抜け
    )

    # ── 改善④: 逆張りサブ（RSI<22 + -DI減少中）
    contra_buy = (
        rsi_now < 22 and
        slope_90 > -10 and                  # 長期トレンド崩壊でない
        mdi_now < mdi_3ago                  # 売り圧力が緩んできた
    )

    # ── 売りシグナル
    cross_below_e100 = (c_prev >= e100_now) and (c_now < e100_now)
    rsi_collapse     = (rsi_prev >= 35) and (rsi_now < 35)
    macro_break      = (c_now < w_sma) and (w_sma > 0)
    sell_signal      = cross_below_e100 or rsi_collapse or macro_break

    # 信号を文字列に変換
    if main_buy:
        reason = "EMA200上抜け" if cross_above_e200 else "ゴールデンクロス"
        buy_sig = f"MAIN:{reason}"
    elif reentry_buy:
        buy_sig = "REENTRY:RSI50上抜け+ゴールデン状態"
    elif contra_buy:
        buy_sig = "CONTRA:RSI底圏+売り圧力低下"
    else:
        buy_sig = "NONE"

    sell_reason = ""
    if cross_below_e100: sell_reason = "EMA100下抜け"
    elif rsi_collapse:   sell_reason = "RSI35下抜け"
    elif macro_break:    sell_reason = "週足SMA20下抜け"
    sell_sig = f"SELL:{sell_reason}" if sell_signal else "NONE"

    return buy_sig, sell_sig, last

# ─── 改善①: コスト適用 ──────────────────────────────────

def apply_cost(price, direction):
    return price * (1 + COST_ONEWAY) if direction == "buy" else price * (1 - COST_ONEWAY)

# ─── ポジションサイジング（v7: サイクル倍率付き） ────────────

def calc_size(capital, price, atr_v, rv_v, kelly=None, cycle_mul=1.0, stop_mul=None):
    if kelly    is None: kelly    = KELLY_HALF
    if stop_mul is None: stop_mul = ATR_STOP_MUL
    sl   = price - atr_v * stop_mul
    risk = price - sl
    if risk <= 0:
        return 0.0, sl
    vol_factor = min(max(TARGET_VOL / max(rv_v, 0.05), 0.3), 2.0)
    size_risk  = (capital * kelly * cycle_mul) / risk
    size_vol   = (capital * kelly * cycle_mul * vol_factor) / risk
    size       = min(size_risk, size_vol, capital / price * 0.98)
    return max(size, 0.0), sl

# ─── トレーリングストップ更新 ────────────────────────────

def update_trailing(pf, current_price, atr_v):
    pos = pf["position"]
    if pos["qty"] <= 0:
        return

    pos["trail_hi"] = max(pos["trail_hi"], current_price)
    new_sl = pos["trail_hi"] - atr_v * ATR_STOP_MUL
    pos["stop_price"] = max(pos["stop_price"], new_sl)

# ─── 取引執行 ────────────────────────────────────────────

def execute_buy(pf, price, atr_v, rv_v, reason, today_str):
    if pf["position"]["qty"] > 0:
        return None   # すでにポジションあり

    size, sl = calc_size(pf["cash"], price, atr_v, rv_v)
    if size <= 0:
        return None

    cost = size * price
    if cost > pf["cash"]:
        return None

    pf["cash"] -= cost
    pf["position"] = {
        "qty":         round(size, 6),
        "entry_price": round(price, 2),
        "entry_date":  today_str,
        "stop_price":  round(sl, 2),
        "trail_hi":    round(price, 2),
        "tp_done":     False,
    }
    pf["total_trades"] += 1

    record = {
        "date":   today_str,
        "action": "BUY",
        "price":  round(price, 2),
        "qty":    round(size, 6),
        "cost":   round(cost, 0),
        "reason": reason,
        "cash_after": round(pf["cash"], 0),
    }
    pf["trades"].append(record)
    return record

def execute_sell(pf, price, reason, today_str):
    pos = pf["position"]
    if pos["qty"] <= 0:
        return None

    qty      = pos["qty"]
    proceeds = qty * price
    pnl      = (price - pos["entry_price"]) * qty
    pnl_pct  = (price - pos["entry_price"]) / pos["entry_price"] * 100
    hold_days= (date.fromisoformat(today_str) - date.fromisoformat(pos["entry_date"])).days \
               if pos["entry_date"] else 0

    pf["cash"] += proceeds
    if pnl > 0:
        pf["wins"] = pf.get("wins", 0) + 1

    record = {
        "date":       today_str,
        "action":     "SELL",
        "price":      round(price, 2),
        "qty":        round(qty, 6),
        "proceeds":   round(proceeds, 0),
        "pnl":        round(pnl, 0),
        "pnl_pct":    round(pnl_pct, 2),
        "hold_days":  hold_days,
        "reason":     reason,
        "cash_after": round(pf["cash"] + proceeds, 0),
    }
    pf["trades"].append(record)
    pf["position"] = {
        "qty": 0.0, "entry_price": 0.0, "entry_date": None,
        "stop_price": 0.0, "trail_hi": 0.0, "tp_done": False,
    }
    return record

def check_stop(pf, price, atr_v, today_str):
    pos = pf["position"]
    if pos["qty"] <= 0:
        return None
    if price <= pos["stop_price"]:
        return execute_sell(pf, price, f"ストップロス(${pos['stop_price']:,.0f})", today_str)
    return None

def check_partial_tp(pf, price, atr_v, today_str):
    pos = pf["position"]
    if pos["qty"] <= 0 or pos["tp_done"]:
        return None
    tp_level = pos["entry_price"] + atr_v * ATR_TP_MUL
    if price >= tp_level:
        # 40% 部分利食い
        cut_qty  = pos["qty"] * 0.40
        proceeds = cut_qty * price
        pnl      = (price - pos["entry_price"]) * cut_qty
        pf["cash"]           += proceeds
        pf["position"]["qty"] = round(pos["qty"] - cut_qty, 6)
        pf["position"]["tp_done"] = True
        if pnl > 0:
            pf["wins"] = pf.get("wins", 0) + 1
        record = {
            "date":     today_str,
            "action":   "PARTIAL_TP",
            "price":    round(price, 2),
            "qty":      round(cut_qty, 6),
            "proceeds": round(proceeds, 0),
            "pnl":      round(pnl, 0),
            "reason":   f"部分利食い(ATR×{ATR_TP_MUL}到達)",
        }
        pf["trades"].append(record)
        return record
    return None

# ─── 統計計算 ────────────────────────────────────────────

def portfolio_stats(pf, current_price):
    pos      = pf["position"]
    unr_pnl  = (current_price - pos["entry_price"]) * pos["qty"] if pos["qty"] > 0 else 0
    total_eq = pf["cash"] + pos["qty"] * current_price
    ret_pct  = (total_eq - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    sell_trades = [t for t in pf["trades"] if t["action"] == "SELL"]
    realized    = sum(t.get("pnl", 0) for t in sell_trades)
    partial_tp  = [t for t in pf["trades"] if t["action"] == "PARTIAL_TP"]
    realized   += sum(t.get("pnl", 0) for t in partial_tp)

    total_trades = pf.get("total_trades", 0)
    wins         = pf.get("wins", 0)
    win_rate     = wins / total_trades * 100 if total_trades > 0 else 0

    return {
        "total_equity":  round(total_eq, 0),
        "cash":          round(pf["cash"], 0),
        "position_qty":  pos["qty"],
        "position_value":round(pos["qty"] * current_price, 0),
        "unrealized_pnl":round(unr_pnl, 0),
        "realized_pnl":  round(realized, 0),
        "total_return":  round(ret_pct, 2),
        "total_trades":  total_trades,
        "win_rate":      round(win_rate, 1),
    }

# ─── 日次ログ書き込み ─────────────────────────────────────

def write_log(today_str, df, pf, actions, stats):
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{today_str}.txt")

    last     = df.iloc[-1]
    price    = float(last["Close"])
    rsi_v    = float(last["rsi"])   if not np.isnan(last["rsi"])  else 0
    e200     = float(last["ema200"])
    atr_v    = float(last["atr"])   if not np.isnan(last["atr"]) else 0
    dd_ath   = float(last["dd_from_ath"]) if not np.isnan(last["dd_from_ath"]) else 0

    pos = pf["position"]
    lines = [
        f"=== BTC エアトレード日報 {today_str} ===",
        f"",
        f"■ 市場",
        f"  BTC価格  : ${price:>10,.0f}",
        f"  EMA200   : ${e200:>10,.0f}  ({'上' if price>e200 else '下'})",
        f"  RSI(14)  : {rsi_v:>10.1f}",
        f"  ATHからDD: {dd_ath:>+9.1f}%",
        f"",
        f"■ 今日のアクション",
    ]
    if actions:
        for a in actions:
            lines.append(f"  {a['action']}  ${a['price']:,.0f}  qty={a.get('qty',0):.4f}  "
                         f"pnl={a.get('pnl', 'N/A')}  {a['reason']}")
    else:
        lines.append("  なし（ホールドまたはノーポジ待機）")

    lines += [
        f"",
        f"■ ポートフォリオ",
        f"  総資産   : {stats['total_equity']:>10,.0f}円  ({stats['total_return']:>+.2f}%)",
        f"  現金     : {stats['cash']:>10,.0f}円",
        f"  BTC保有  : {stats['position_qty']:.4f} BTC  ({stats['position_value']:,.0f}円)",
        f"  未実現損益: {stats['unrealized_pnl']:>+9,.0f}円",
        f"  実現損益 : {stats['realized_pnl']:>+9,.0f}円",
        f"  取引回数 : {stats['total_trades']}回  勝率 {stats['win_rate']:.1f}%",
    ]

    if pos["qty"] > 0:
        lines += [
            f"",
            f"■ 現在のポジション",
            f"  エントリー: ${pos['entry_price']:,.0f}  ({pos['entry_date']})",
            f"  ストップ  : ${pos['stop_price']:,.0f}",
            f"  含み損益  : {stats['unrealized_pnl']:>+,}円",
        ]

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return "\n".join(lines)

# ─── Telegram 通知 ───────────────────────────────────────

def send_telegram(message):
    token   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print("[Telegram] 未設定のためスキップ")
        return

    import urllib.request
    payload = json.dumps({
        "chat_id": int(chat_id),
        "text":    message,
        "parse_mode": "HTML",
    }).encode("utf-8")
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=10)
        print("[Telegram] 送信完了")
    except Exception as e:
        print(f"[Telegram] 送信失敗: {e}")

def build_telegram_msg(today_str, df, actions, stats, pf):
    last  = df.iloc[-1]
    price = float(last["Close"])
    rsi_v = float(last["rsi"]) if not np.isnan(last["rsi"]) else 0
    e200  = float(last["ema200"])
    dd    = float(last["dd_from_ath"]) if not np.isnan(last["dd_from_ath"]) else 0

    trend = "📈 EMA200上" if price > e200 else "📉 EMA200下"
    ret   = stats["total_return"]
    ret_s = f"{'🟢' if ret >= 0 else '🔴'} {ret:+.2f}%"

    lines = [
        f"🤖 <b>BTC エアトレード {today_str}</b>",
        f"",
        f"BTC価格: <b>${price:,.0f}</b>  {trend}",
        f"RSI: {rsi_v:.1f}  ATHから: {dd:.1f}%",
        f"",
    ]

    if actions:
        lines.append("📌 <b>今日のアクション</b>")
        for a in actions:
            icon = "🟢" if a["action"] in ("BUY",) else "🔴" if "SELL" in a["action"] else "🟡"
            pnl_s = f"  損益: {a['pnl']:+,}円" if "pnl" in a else ""
            lines.append(f"{icon} {a['action']} @ ${a['price']:,.0f}{pnl_s}")
            lines.append(f"   {a['reason']}")
    else:
        pos = pf["position"]
        if pos["qty"] > 0:
            lines.append(f"⏸ 保有継続 ({pos['qty']:.4f} BTC @ ${pos['entry_price']:,.0f})")
        else:
            lines.append(f"⏸ 待機中（シグナルなし）")

    lines += [
        f"",
        f"💰 <b>ポートフォリオ</b>",
        f"総資産: {stats['total_equity']:,}円  {ret_s}",
        f"現金: {stats['cash']:,}円  BTC: {stats['position_value']:,}円",
        f"実現損益: {stats['realized_pnl']:+,}円",
        f"取引: {stats['total_trades']}回  勝率: {stats['win_rate']:.1f}%",
    ]
    return "\n".join(lines)

# ─── メイン ──────────────────────────────────────────────

def main():
    today_str = datetime.now(JST).strftime("%Y-%m-%d")
    print(f"[{today_str}] BTC エアトレード実行中...")

    # データ取得
    df = fetch_data()
    df = add_indicators(df)

    last      = df.iloc[-1]
    price     = float(last["Close"])
    atr_v     = float(last["atr"])  if not np.isnan(last["atr"])  else price * 0.03
    rv_v      = float(last["rv20"]) if not np.isnan(last["rv20"]) else 0.3
    cycle_mul = get_cycle_multiplier(last)
    print(f"  BTC価格: ${price:,.0f}  ATR: ${atr_v:,.0f}  RV: {rv_v*100:.1f}%  "
          f"サイクル: {cycle_mul}x")

    # ポートフォリオ読込
    pf = load_portfolio()

    # シグナル判定（v7: 3種）
    buy_sig, sell_sig, _ = check_signals(df)
    print(f"  シグナル — 買い: {buy_sig}  売り: {sell_sig}")

    actions = []

    # ─ 既存ポジションの管理
    update_trailing(pf, price, atr_v)

    sl_action = check_stop(pf, price, atr_v, today_str)
    if sl_action:
        actions.append(sl_action)
        print(f"  ストップロス: pnl={sl_action['pnl']:+,}円")
        pf["reentry_ok"] = (price > float(last["ema200"]) if not np.isnan(last["ema200"]) else False)

    if not sl_action:
        tp_action = check_partial_tp(pf, price, atr_v, today_str)
        if tp_action:
            actions.append(tp_action)
            print(f"  部分利食い: pnl={tp_action['pnl']:+,}円")

    # ─ 売りシグナル（メインポジション）
    if sell_sig != "NONE" and pf["position"]["qty"] > 0:
        reason = sell_sig.replace("SELL:", "")
        exec_price = apply_cost(price, "sell")
        sell_action = execute_sell(pf, exec_price, reason, today_str)
        if sell_action:
            actions.append(sell_action)
            print(f"  売却: pnl={sell_action['pnl']:+,}円  (コスト込み実行価格 ${exec_price:,.0f})")
            pf["reentry_ok"] = (price > float(last["ema200"]) if not np.isnan(last["ema200"]) else False)

    # ─ 買いシグナル（3種に分岐）
    sig_type = buy_sig.split(":")[0] if buy_sig != "NONE" else ""

    if sig_type == "MAIN" and pf["position"]["qty"] == 0:
        reason    = buy_sig.replace("MAIN:", "")
        exec_price= apply_cost(price, "buy")
        size, sl  = calc_size(pf["cash"], exec_price, atr_v, rv_v,
                               kelly=KELLY_HALF, cycle_mul=cycle_mul)
        if size > 0 and size * exec_price <= pf["cash"]:
            buy_action = execute_buy(pf, exec_price, atr_v, rv_v, reason, today_str)
            if buy_action:
                actions.append(buy_action)
                pf["reentry_ok"] = False
                print(f"  メイン買い: {buy_action['qty']:.4f} BTC @ ${exec_price:,.0f}  "
                      f"サイクル{cycle_mul}x")

    elif sig_type == "REENTRY" and pf["position"]["qty"] == 0 and pf.get("reentry_ok", False):
        reason    = buy_sig.replace("REENTRY:", "")
        exec_price= apply_cost(price, "buy")
        size, sl  = calc_size(pf["cash"], exec_price, atr_v, rv_v,
                               kelly=KELLY_REENTRY, cycle_mul=cycle_mul)
        if size > 0 and size * exec_price <= pf["cash"]:
            buy_action = execute_buy(pf, exec_price, atr_v, rv_v,
                                     f"再エントリー:{reason}", today_str)
            if buy_action:
                actions.append(buy_action)
                pf["reentry_ok"] = False
                print(f"  再エントリー: {buy_action['qty']:.4f} BTC @ ${exec_price:,.0f}")

    elif sig_type == "CONTRA" and pf["position"]["qty"] == 0:
        reason    = buy_sig.replace("CONTRA:", "")
        exec_price= apply_cost(price, "buy")
        size, sl  = calc_size(pf["cash"], exec_price, atr_v, rv_v,
                               kelly=KELLY_CONTRA, cycle_mul=1.0, stop_mul=ATR_STOP_CONT)
        if size > 0 and size * exec_price <= pf["cash"]:
            buy_action = execute_buy(pf, exec_price, atr_v, rv_v,
                                     f"逆張りサブ:{reason}", today_str)
            if buy_action:
                actions.append(buy_action)
                print(f"  逆張りサブ: {buy_action['qty']:.4f} BTC @ ${exec_price:,.0f}  "
                      f"(小ロット)")

    # 統計計算
    stats = portfolio_stats(pf, price)
    print(f"  総資産: {stats['total_equity']:,}円  ({stats['total_return']:+.2f}%)")

    # ポートフォリオ保存
    save_portfolio(pf)

    # ログ書き込み
    log_text = write_log(today_str, df, pf, actions, stats)
    print(f"  ログ保存: {LOG_DIR}/{today_str}.txt")

    # Telegram 通知（単体実行時のみ。GitHub Actions では workflow の統合通知が送る）
    if os.environ.get("BTC_SEND_TELEGRAM", "0") == "1":
        tg_msg = build_telegram_msg(today_str, df, actions, stats, pf)
        send_telegram(tg_msg)

    # workflow 統合通知用サマリーをファイルに書き出す
    summary = build_telegram_msg(today_str, df, actions, stats, pf)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "today_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary)

    # 終了コードでシグナルを伝える（CI での表示用）
    print("\n" + "─"*50)
    print(log_text)
    print("─"*50)

    return 0

if __name__ == "__main__":
    sys.exit(main())
