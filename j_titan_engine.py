#!/usr/bin/env python3
"""
J-Titan Engine v2 — Japanese Swing Trade AI [決定版]
Integrates every feature: MACD+SMA, market filter, stop-loss, trailing stop,
TSE price limits, 2% risk rule, 4-slot portfolio, walk-forward optimisation,
and daily auto paper-trading with portfolio.json persistence.

Usage:
  python j_titan_engine.py --mode backtest   # optimise + test
  python j_titan_engine.py --mode auto       # daily paper-trade update
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from functools import reduce
from itertools import product

import numpy as np
import pandas as pd
import pytz
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════
INITIAL_CAPITAL   = 1_000_000
LOT               = 100          # 単元株
COMMISSION        = 0.0005       # 片道0.05%
MAX_SLOTS         = 4            # 同時保有上限（複利: 1枠=総資産÷4 を動的計算）
RISK_PER_TRADE    = 0.02         # 2%リスクルール
TEST_DAYS         = 252          # テスト期間（約1年）
MARKET_SMA        = 25           # 日経地合いフィルター SMA
MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9

# ── モメンタムフィルター ─────────────────────────────────────────────────────
RSI_PERIOD    = 14
ADX_PERIOD    = 14
RSI_THRESHOLD = 50.0     # RSI >= 50 で上昇モメンタムあり
ADX_THRESHOLD = 22.0     # ADX >= 22 で明確なトレンドあり

SMA_PERIODS     = [20, 25, 30]
STOP_LOSS_RATES = [-0.02, -0.025, -0.03, -0.05]
TRAILING_RATES  = [0.025, 0.03,  0.04,  0.05]

# ── 監視銘柄（東証プライム・グロース 主要40銘柄）──────────────────────────────
SYMBOLS = [
    # 自動車・輸送
    "7203", "7267", "7201", "7269", "7270",
    # 電機・精密
    "6758", "6501", "6702", "6752", "6503",
    # 半導体・電子部品
    "8035", "6857", "6723", "4063", "6526",
    # IT・インターネット
    "4689", "4755", "3659", "4307",
    # 通信
    "9432", "9433", "9434", "9984",
    # 銀行・金融
    "8306", "8411", "8316", "8604", "8591",
    # 商社
    "8058", "8001", "8002", "8031",
    # 医薬品
    "4502", "4519", "4568", "4507",
    # 消費・生活・エネルギー
    "3382", "7974", "5020", "5401",
    # 素材・機械
    "6301", "4452",
]
NAMES = {
    "7203": "トヨタ自動車",  "7267": "ホンダ",       "7201": "日産自動車",
    "7269": "スズキ",        "7270": "SUBARU",
    "6758": "ソニーG",       "6501": "日立製作所",    "6702": "富士通",
    "6752": "パナソニック",  "6503": "三菱電機",
    "8035": "東京エレクトロン","6857": "アドバンテスト", "6723": "ルネサス",
    "4063": "信越化学",      "6526": "ソシオネクスト",
    "4689": "LINEヤフー",    "4755": "楽天G",         "3659": "ネクソン",
    "4307": "野村総研",
    "9432": "NTT",           "9433": "KDDI",          "9434": "ソフトバンク",
    "9984": "SBG",
    "8306": "三菱UFJ",       "8411": "みずほFG",      "8316": "三井住友FG",
    "8604": "野村HD",        "8591": "ORIX",
    "8058": "三菱商事",      "8001": "伊藤忠商事",    "8002": "丸紅",
    "8031": "三井物産",
    "4502": "武田薬品",      "4519": "中外製薬",      "4568": "第一三共",
    "4507": "塩野義製薬",
    "3382": "セブン&アイ",   "7974": "任天堂",        "5020": "ENEOS",
    "5401": "日本製鉄",
    "6301": "コマツ",        "4452": "花王",
}
TICKER_MAP = {s: f"{s}.T" for s in SYMBOLS}
TICKER_MAP["N225"] = "^N225"

TOKYO_TZ       = pytz.timezone("Asia/Tokyo")
PORTFOLIO_PATH = "portfolio.json"


# ══════════════════════════════════════════════════════════════════════════════
# TSE Daily Price Limit (値幅制限)
# ══════════════════════════════════════════════════════════════════════════════
def tse_limit(p: float) -> float:
    if p <    100: return   30.0
    if p <    200: return   30.0
    if p <    500: return   50.0
    if p <    700: return   80.0
    if p <  1_000: return  100.0
    if p <  1_500: return  200.0
    if p <  2_000: return  300.0
    if p <  3_000: return  400.0
    if p <  5_000: return  500.0
    if p <  7_000: return  700.0
    if p < 10_000: return 1_000.0
    if p < 15_000: return 1_500.0
    if p < 20_000: return 2_000.0
    if p < 30_000: return 3_000.0
    if p < 50_000: return 4_000.0
    if p < 70_000: return 5_000.0
    if p <100_000: return 10_000.0
    return 20_000.0


# ══════════════════════════════════════════════════════════════════════════════
# Portfolio JSON — ペーパートレード永続状態
# ══════════════════════════════════════════════════════════════════════════════
def _init_portfolio() -> dict:
    return {
        "created":            str(datetime.now(TOKYO_TZ).date()),
        "last_updated":       None,
        "cash":               float(INITIAL_CAPITAL),
        "positions":          {},
        "pending_orders":     {},
        "realized_trades":    [],
        "total_realized_pnl": 0.0,
        "params": {
            "sma": 25, "stop_loss": -0.03, "trailing": 0.04,
            "source": "default — run --mode backtest to optimise",
        },
    }

def load_portfolio() -> dict:
    if os.path.exists(PORTFOLIO_PATH):
        with open(PORTFOLIO_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return _init_portfolio()

def save_portfolio(state: dict) -> None:
    with open(PORTFOLIO_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, default=str)


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════════════
def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(TOKYO_TZ)
    else:
        df.index = df.index.tz_convert(TOKYO_TZ)
    df.sort_index(inplace=True)
    df.index.name = "Date"
    return df


def load_csv(code: str):
    path = f"data/{code}.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(TOKYO_TZ)
    else:
        df.index = df.index.tz_convert(TOKYO_TZ)
    if "Volume" in df.columns:
        df = df[df["Volume"].notna() & (df["Volume"] > 0)]
    return df


def load_and_refresh(code: str):
    """Load CSV + fetch the very latest trading days from yfinance."""
    df_hist = load_csv(code)
    today   = datetime.now(TOKYO_TZ).date()
    try:
        ticker = TICKER_MAP.get(code, f"{code}.T")
        df_new = yf.download(ticker,
                             start=today - timedelta(days=7),
                             end=today + timedelta(days=1),
                             auto_adjust=True, progress=False)
        if not df_new.empty:
            df_new = _normalise(df_new)
            if "Volume" in df_new.columns:
                df_new = df_new[df_new["Volume"].notna() & (df_new["Volume"] > 0)]
            if df_hist is not None:
                df = pd.concat([df_hist, df_new])
                df = df[~df.index.duplicated(keep="last")].sort_index()
            else:
                df = df_new
            return df
    except Exception:
        pass
    return df_hist


def load_or_fetch(code: str) -> pd.DataFrame:
    """Load from CSV; if missing, fetch 5-year history from yfinance and save."""
    path = f"data/{code}.csv"
    if os.path.exists(path):
        return load_csv(code)
    ticker = TICKER_MAP.get(code, f"{code}.T")
    today  = datetime.now(TOKYO_TZ).date()
    start  = today - timedelta(days=5 * 365 + 5)
    try:
        df = yf.download(ticker, start=start, end=today,
                         auto_adjust=True, progress=False)
        if df.empty:
            return None
        df = _normalise(df)
        if "Volume" in df.columns:
            df = df[df["Volume"].notna() & (df["Volume"] > 0)]
        os.makedirs("data", exist_ok=True)
        df.to_csv(path)
        return df
    except Exception as e:
        print(f"    WARNING: {code} ({ticker}) 取得失敗: {e}")
        return None


def load_n225_series(df: pd.DataFrame):
    """Return Close price series from N225 DataFrame."""
    return df["Close"] if df is not None and "Close" in df.columns else pd.Series(dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
# Indicators
# ══════════════════════════════════════════════════════════════════════════════
def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def compute_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    ph, pl, pc = h.shift(1), l.shift(1), c.shift(1)

    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()],
                   axis=1).max(axis=1)

    up, dn     = h - ph, pl - l
    plus_dm_v  = np.where((up > dn) & (up > 0), up.values, 0.0)
    minus_dm_v = np.where((dn > up) & (dn > 0), dn.values, 0.0)
    plus_dm    = pd.Series(plus_dm_v,  index=df.index, dtype=float)
    minus_dm   = pd.Series(minus_dm_v, index=df.index, dtype=float)

    alpha   = 1.0 / period
    atr     = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr
    minus_di= 100.0 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx     = 100.0 * (plus_di - minus_di).abs() / di_sum
    return dx.ewm(alpha=alpha, adjust=False).mean()


def build_indicators(df: pd.DataFrame, sma_period: int) -> pd.DataFrame:
    c     = df["Close"]
    ema_f = c.ewm(span=MACD_FAST, adjust=False).mean()
    ema_s = c.ewm(span=MACD_SLOW, adjust=False).mean()
    macd  = ema_f - ema_s
    sig   = macd.ewm(span=MACD_SIG, adjust=False).mean()
    sma   = c.rolling(sma_period).mean()
    gc    = (macd > sig) & (macd.shift(1) <= sig.shift(1))
    dc    = (macd < sig) & (macd.shift(1) >= sig.shift(1))
    rsi   = compute_rsi(c)
    adx   = compute_adx(df)
    return pd.DataFrame({
        "close": c, "open": df["Open"], "sma": sma,
        "golden_cross": gc, "dead_cross": dc,
        "above_sma": c > sma, "below_sma": c < sma,
        "rsi": rsi, "adx": adx,
    })


def build_market_filter_arr(n225_close: pd.Series, idx: pd.DatetimeIndex) -> np.ndarray:
    sma25 = n225_close.rolling(MARKET_SMA).mean()
    above = (n225_close > sma25).reindex(idx).ffill().fillna(True)
    return above.values.astype(bool)


# ══════════════════════════════════════════════════════════════════════════════
# Portfolio Backtest Engine (バックテスト本体)
# ══════════════════════════════════════════════════════════════════════════════
def portfolio_backtest(
    ind_all:       dict,
    stop_loss_pct: float,
    trailing_pct:  float,
    mkt_above:     np.ndarray,
    common_idx:    pd.DatetimeIndex,
    track_skips:   bool = False,
) -> dict:
    syms = [s for s in SYMBOLS if s in ind_all]
    n    = len(common_idx)
    if n == 0 or not syms:
        empty_idx = common_idx[:1] if n else pd.DatetimeIndex([])
        return dict(final_asset=INITIAL_CAPITAL, total_return=0.0, total_trades=0,
                    win_rate=0.0, max_drawdown=0.0, profit_factor=0.0,
                    asset_series=pd.Series([INITIAL_CAPITAL], index=empty_idx),
                    trades=[], skips=[])

    C  = {s: ind_all[s]["close"].values.astype(float)       for s in syms}
    O  = {s: ind_all[s]["open"].values.astype(float)        for s in syms}
    GC = {s: ind_all[s]["golden_cross"].values.astype(bool) for s in syms}
    DC = {s: ind_all[s]["dead_cross"].values.astype(bool)   for s in syms}
    AV = {s: ind_all[s]["above_sma"].values.astype(bool)    for s in syms}
    BL = {s: ind_all[s]["below_sma"].values.astype(bool)    for s in syms}
    SM = {s: ind_all[s]["sma"].values.astype(float)         for s in syms}
    RS = {s: ind_all[s]["rsi"].values.astype(float)         for s in syms}
    AD = {s: ind_all[s]["adx"].values.astype(float)         for s in syms}

    cash      = float(INITIAL_CAPITAL)
    positions = {}   # sym → {entry_price, shares, peak_close, entry_date}
    p_sells   = {}   # sym → (reason, deferred_count)
    p_buys    = {}   # sym → deferred_count
    asset_arr = np.empty(n)
    trades, skips = [], []

    for i in range(n):
        prev = max(0, i - 1)

        # ── Execute pending sells at today's open ─────────────────────────
        for sym in list(p_sells.keys()):
            o, pc = O[sym][i], C[sym][prev]
            reason, dfr = p_sells[sym]
            if o <= pc - tse_limit(pc) and dfr < 3:   # ストップ安: 持越し
                p_sells[sym] = (reason, dfr + 1)
            else:
                if sym in positions:
                    pos    = positions.pop(sym)
                    sh, ep = pos["shares"], pos["entry_price"]
                    xproc  = o * sh * (1 - COMMISSION)
                    cash  += xproc
                    trades.append(dict(
                        symbol=sym, shares=sh, reason=reason,
                        entry_date=pos["entry_date"], entry_price=ep,
                        exit_date=common_idx[i], exit_price=o,
                        profit=xproc - ep * sh * (1 + COMMISSION),
                    ))
                del p_sells[sym]

        # ── Execute pending buys at today's open ──────────────────────────
        for sym in list(p_buys.keys()):
            o, pc = O[sym][i], C[sym][prev]
            dfr   = p_buys[sym]
            if o >= pc + tse_limit(pc) and dfr < 2:   # ストップ高: 持越し
                p_buys[sym] = dfr + 1
                continue
            del p_buys[sym]
            if sym in positions or len(positions) >= MAX_SLOTS:
                continue
            port_val = cash + sum(positions[s]["shares"] * C[s][prev] for s in positions)
            sl_dist  = o * abs(stop_loss_pct)
            if sl_dist <= 0:
                continue
            slot_cap = port_val / MAX_SLOTS   # 複利: 総資産÷4 を動的計算
            lots   = min(int(port_val * RISK_PER_TRADE / sl_dist / LOT),
                         int(slot_cap / (o * (1 + COMMISSION)) / LOT))
            shares = lots * LOT
            cost   = shares * o * (1 + COMMISSION)
            if shares >= LOT and cost <= cash:
                cash -= cost
                positions[sym] = dict(entry_price=o, shares=shares,
                                      peak_close=o, entry_date=common_idx[i])

        # ── Mark-to-market ────────────────────────────────────────────────
        asset_arr[i] = cash + sum(positions[s]["shares"] * C[s][i] for s in positions)

        # ── Signal check at today's close ─────────────────────────────────
        mkt_ok = bool(mkt_above[i])
        for sym in syms:
            c = C[sym][i]
            if sym in positions:
                pos = positions[sym]
                pos["peak_close"] = max(pos["peak_close"], c)
                if sym not in p_sells:
                    if c <= pos["entry_price"] * (1 + stop_loss_pct):
                        p_sells[sym] = ("stop_loss", 0)
                    elif c <= pos["peak_close"] * (1 - trailing_pct):
                        p_sells[sym] = ("trailing_stop", 0)
                    elif DC[sym][i] or BL[sym][i]:
                        p_sells[sym] = ("signal", 0)
            elif sym not in p_buys:
                if not np.isnan(SM[sym][i]) and AV[sym][i] and GC[sym][i]:
                    rsi_v  = RS[sym][i]
                    adx_v  = AD[sym][i]
                    rsi_ok = not np.isnan(rsi_v) and rsi_v >= RSI_THRESHOLD
                    adx_ok = not np.isnan(adx_v) and adx_v >= ADX_THRESHOLD
                    if mkt_ok and rsi_ok and adx_ok:
                        p_buys[sym] = 0
                    elif track_skips:
                        if not mkt_ok:
                            skips.append({"date": common_idx[i], "symbol": sym,
                                          "type": "market",
                                          "reason": "市場地合い悪化のため購入を見送りました"})
                        else:
                            skips.append({"date": common_idx[i], "symbol": sym,
                                          "type": "momentum",
                                          "reason": (f"RSI/ADX強度不足 "
                                                     f"(RSI={rsi_v:.1f}, ADX={adx_v:.1f})")})

    # ── Force-liquidate remaining positions at last close ─────────────────
    last = n - 1
    for sym in list(positions.keys()):
        pos    = positions.pop(sym)
        sh, ep = pos["shares"], pos["entry_price"]
        sp     = C[sym][last]
        xproc  = sp * sh * (1 - COMMISSION)
        cash  += xproc
        trades.append(dict(
            symbol=sym, shares=sh, reason="forced",
            entry_date=pos["entry_date"], entry_price=ep,
            exit_date=common_idx[last], exit_price=sp,
            profit=xproc - ep * sh * (1 + COMMISSION),
        ))
    asset_arr[last] = cash

    asset_s = pd.Series(asset_arr, index=common_idx, name="asset")
    n_tr    = len(trades)
    final   = float(asset_s.iloc[-1])
    ret     = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    if n_tr > 0:
        profs  = [t["profit"] for t in trades]
        wins   = [p for p in profs if p > 0]
        losses = [p for p in profs if p <= 0]
        wr     = len(wins) / n_tr * 100
        gp     = sum(wins) if wins else 0.0
        gl     = abs(sum(losses)) if losses else 0.0
        pf     = gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0)
    else:
        wr = pf = 0.0

    rm    = asset_s.cummax()
    maxdd = float(((asset_s - rm) / rm * 100).min())
    return dict(final_asset=final, total_return=ret, total_trades=n_tr,
                win_rate=wr, max_drawdown=maxdd, profit_factor=pf,
                asset_series=asset_s, trades=trades, skips=skips)


# ══════════════════════════════════════════════════════════════════════════════
# Backtest Mode
# ══════════════════════════════════════════════════════════════════════════════
def run_backtest(df_all: dict, n225_close: pd.Series) -> None:
    active = [s for s in SYMBOLS if s in df_all]
    if len(active) < 2:
        sys.exit("Error: 2銘柄以上必要です。fetch_master_data.py を実行してください。")

    # Build common date index and train/test split
    ind_ref    = {s: build_indicators(df_all[s], 25) for s in active}
    common_idx = reduce(lambda a, b: a.intersection(b),
                        [ind_ref[s].index for s in active]).sort_values()
    split_date = common_idx[-TEST_DAYS]
    train_idx  = common_idx[common_idx < split_date]
    test_idx   = common_idx[common_idx >= split_date]

    print(f"\n  共通取引日: {common_idx[0].date()} ～ {common_idx[-1].date()}  ({len(common_idx)} 日)")
    print(f"  訓練期間  : {train_idx[0].date()} ～ {train_idx[-1].date()}  ({len(train_idx)} 日)")
    print(f"  テスト期間: {test_idx[0].date()} ～ {test_idx[-1].date()}  ({len(test_idx)} 日)")

    mkt_train = build_market_filter_arr(n225_close, train_idx)
    mkt_test  = build_market_filter_arr(n225_close, test_idx)

    # Pre-compute indicators for all SMA candidates
    print("\n  インジケータ計算中 ...")
    all_ind = {
        sma: {
            s: {
                "train": build_indicators(df_all[s], sma).reindex(train_idx),
                "test":  build_indicators(df_all[s], sma).reindex(test_idx),
            }
            for s in active
        }
        for sma in SMA_PERIODS
    }

    # Grid search on training data
    n_combos = len(SMA_PERIODS) * len(STOP_LOSS_RATES) * len(TRAILING_RATES)
    print(f"\n{'─'*68}")
    print(f"  グリッドサーチ  ({n_combos} 組合せ × 訓練期間)  ※地合いフィルター込み")
    print(f"{'─'*68}")
    print(f"  {'SMA':>4} {'SL':>6} {'TS':>6}  {'最終資産(¥)':>16} {'リターン':>8} "
          f"{'取引':>6} {'勝率':>6}")
    print(f"  {'─'*4} {'─'*6} {'─'*6}  {'─'*16} {'─'*8} {'─'*6} {'─'*6}")

    best_asset, best_params = -float("inf"), None

    for sma, sl, ts in product(SMA_PERIODS, STOP_LOSS_RATES, TRAILING_RATES):
        ind_t = {s: all_ind[sma][s]["train"] for s in active}
        r     = portfolio_backtest(ind_t, sl, ts, mkt_train, train_idx)
        is_b  = r["final_asset"] > best_asset
        if is_b:
            best_asset, best_params = r["final_asset"], (sma, sl, ts)
        mark = " ◀" if is_b else ""
        print(f"  {sma:>4} {sl*100:>5.1f}% {ts*100:>5.1f}%  {r['final_asset']:>16,.0f} "
              f"{r['total_return']:>+7.1f}% {r['total_trades']:>6} "
              f"{r['win_rate']:>5.1f}%{mark}")

    best_sma, best_sl, best_ts = best_params
    print(f"\n  ★ 最適パラメータ: SMA={best_sma}, SL={best_sl*100:.1f}%, "
          f"Trailing={best_ts*100:.1f}%")
    print(f"     訓練期間 最終資産: ¥{best_asset:,.0f}")

    # Test on out-of-sample data
    ind_te = {s: all_ind[best_sma][s]["test"] for s in active}
    result  = portfolio_backtest(ind_te, best_sl, best_ts, mkt_test, test_idx,
                                 track_skips=True)

    pf_str = f"{result['profit_factor']:.2f}" if result["profit_factor"] != float("inf") else "∞"
    skips  = result["skips"]

    print(f"\n{'═'*68}")
    print(f"  テスト期間 結果  "
          f"(SMA={best_sma}, SL={best_sl*100:.0f}%, Trailing={best_ts*100:.0f}%)")
    print(f"{'─'*68}")
    print(f"  初期資金                     : ¥{INITIAL_CAPITAL:>14,.0f}")
    print(f"  最終資産額                   : ¥{result['final_asset']:>14,.0f}")
    print(f"  総利益率                     :  {result['total_return']:>+13.2f}%")
    print(f"  総トレード回数               :  {result['total_trades']:>13} 回")
    print(f"  勝率                         :  {result['win_rate']:>13.1f}%")
    print(f"  最大ドローダウン             :  {result['max_drawdown']:>+13.2f}%")
    print(f"  プロフィットファクター       :  {pf_str:>13}")
    print(f"{'─'*68}")

    if skips:
        mkt_skips = [s for s in skips if s.get("type") == "market"]
        mom_skips = [s for s in skips if s.get("type") == "momentum"]
        print(f"  地合いフィルター スキップ       : {len(mkt_skips):>3} 件")
        print(f"  RSI/ADXモメンタムフィルター スキップ: {len(mom_skips):>3} 件")
        if mom_skips:
            by_sym = {}
            for sk in mom_skips:
                by_sym.setdefault(sk["symbol"], []).append(sk["reason"])
            for sym, reasons in sorted(by_sym.items()):
                print(f"    [{sym}] {NAMES.get(sym, sym)}: {len(reasons)} 件  例: {reasons[0]}")

    if result["trades"]:
        print(f"\n  取引内訳 ({result['total_trades']} 件):")
        by_r = {}
        for t in result["trades"]:
            by_r.setdefault(t["reason"], []).append(t["profit"])
        for reason, profs in sorted(by_r.items()):
            w = sum(1 for p in profs if p > 0)
            print(f"    {reason:<16}: {len(profs):>3}件  勝率{w/len(profs)*100:5.1f}%  "
                  f"P&L ¥{sum(profs):>+,.0f}")

    print(f"{'═'*68}\n")

    # Save optimised params to portfolio.json (preserve cash/positions)
    state = load_portfolio()
    state["params"] = {
        "sma": best_sma, "stop_loss": best_sl, "trailing": best_ts,
        "source": f"backtest optimised on {datetime.now(TOKYO_TZ).date()}",
    }
    save_portfolio(state)
    print(f"  最適パラメータを {PORTFOLIO_PATH} に保存しました\n")

    # Plot
    os.makedirs("output", exist_ok=True)
    asset_s    = result["asset_series"]
    plot_dates = asset_s.index.tz_convert(None) if asset_s.index.tz else asset_s.index

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8),
                                    gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(plot_dates, asset_s.values, color="steelblue", lw=1.5, label="Portfolio")
    ax1.axhline(INITIAL_CAPITAL, color="dimgray", ls="--", lw=0.9,
                label=f"Initial  Y{INITIAL_CAPITAL:,}")
    ax1.fill_between(plot_dates, asset_s.values, INITIAL_CAPITAL,
                     where=asset_s.values >= INITIAL_CAPITAL, alpha=0.2, color="green")
    ax1.fill_between(plot_dates, asset_s.values, INITIAL_CAPITAL,
                     where=asset_s.values <  INITIAL_CAPITAL, alpha=0.2, color="red")
    ax1.set_title(
        f"J-Titan Portfolio [{', '.join(active)}]\n"
        f"SMA={best_sma} SL={best_sl*100:.0f}% TS={best_ts*100:.0f}% | "
        f"Return={result['total_return']:+.2f}% | "
        f"MaxDD={result['max_drawdown']:.2f}% | "
        f"Trades={result['total_trades']} | WinRate={result['win_rate']:.1f}% | "
        f"MktSkips={len(skips)}",
        fontsize=9,
    )
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"Y{x:,.0f}"))
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    rm = asset_s.cummax()
    dd = (asset_s - rm) / rm * 100
    ax2.fill_between(plot_dates, dd.values, 0, alpha=0.5, color="red")
    ax2.plot(plot_dates, dd.values, color="darkred", lw=0.8)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    ax2.set_xlabel("Date")
    plt.tight_layout()
    out_path = "output/j_titan_asset_curve.png"
    plt.savefig(out_path, dpi=150)
    print(f"  グラフ保存 → {out_path}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Auto Mode — Daily Paper Trade Update
# ══════════════════════════════════════════════════════════════════════════════
def run_auto(df_all: dict, n225_close: pd.Series) -> None:
    state       = load_portfolio()
    params      = state.get("params", {})
    sma_period  = int(params.get("sma", 25))
    sl_pct      = float(params.get("stop_loss", -0.03))
    ts_pct      = float(params.get("trailing",  0.04))

    active = [s for s in SYMBOLS if s in df_all]

    # Build indicators on full data (ensures proper MACD/SMA warmup)
    ind_all = {s: build_indicators(df_all[s], sma_period) for s in active}

    # Determine "today" = latest date common to all active symbols
    latest_per_sym = [ind_all[s].index[-1] for s in active if len(ind_all[s]) > 0]
    if not latest_per_sym:
        sys.exit("Error: データなし")
    today = min(latest_per_sym)   # conservative: all symbols must have today's data

    # Previous trading day
    ref_idx  = ind_all[active[0]].index
    pos_t    = ref_idx.get_loc(today) if today in ref_idx else len(ref_idx) - 1
    yesterday = ref_idx[pos_t - 1] if pos_t > 0 else today

    def price(sym, date, col):
        try:
            return float(ind_all[sym].loc[date, col])
        except (KeyError, TypeError):
            return None

    # ── Execute pending orders at today's open ──────────────────────────────
    positions = state.setdefault("positions", {})
    cash      = float(state.get("cash", INITIAL_CAPITAL))
    pending   = state.get("pending_orders", {})
    r_trades  = state.setdefault("realized_trades", [])
    total_pnl = float(state.get("total_realized_pnl", 0.0))
    exec_log  = []

    new_pending = {}

    for sym, order in list(pending.items()):
        o  = price(sym, today, "open")
        pc = price(sym, yesterday, "close")
        if o is None or pc is None:
            new_pending[sym] = order   # no data — carry forward
            continue

        lim    = tse_limit(pc)
        dfr    = int(order.get("deferred", 0))
        action = order["action"]

        if action == "sell":
            if o <= pc - lim and dfr < 3:        # ストップ安: 持越し
                order["deferred"] = dfr + 1
                new_pending[sym]  = order
                exec_log.append(f"  [{sym}] SELL 持越し (ストップ安, {dfr+1}回目)")
            else:
                if sym in positions:
                    pos    = positions.pop(sym)
                    sh     = int(pos["shares"])
                    ep     = float(pos["entry_price"])
                    xproc  = o * sh * (1 - COMMISSION)
                    profit = xproc - ep * sh * (1 + COMMISSION)
                    cash  += xproc
                    total_pnl += profit
                    r_trades.append({
                        "date": str(today.date()), "symbol": sym,
                        "shares": sh, "action": "sell",
                        "price": round(o, 1), "profit": round(profit, 0),
                        "reason": order.get("reason", "signal"),
                    })
                    exec_log.append(f"  [{sym}] {NAMES.get(sym,sym)} SELL {sh}株 "
                                    f"@ ¥{o:,.0f}  P&L ¥{profit:+,.0f}  "
                                    f"({order.get('reason','signal')})")

        elif action == "buy":
            if o >= pc + lim and dfr < 2:        # ストップ高: 持越し
                order["deferred"] = dfr + 1
                new_pending[sym]  = order
                exec_log.append(f"  [{sym}] BUY 持越し (ストップ高, {dfr+1}回目)")
            else:
                if sym not in positions and len(positions) < MAX_SLOTS:
                    port_val = cash + sum(
                        int(positions[s]["shares"]) *
                        (price(s, yesterday, "close") or float(positions[s]["entry_price"]))
                        for s in positions
                    )
                    sl_dist = o * abs(sl_pct)
                    if sl_dist > 0:
                        slot_cap = port_val / MAX_SLOTS
                        lots   = min(
                            int(port_val * RISK_PER_TRADE / sl_dist / LOT),
                            int(slot_cap / (o * (1 + COMMISSION)) / LOT),
                        )
                        shares = lots * LOT
                        cost   = shares * o * (1 + COMMISSION)
                        if shares >= LOT and cost <= cash:
                            cash -= cost
                            positions[sym] = {
                                "entry_date":  str(today.date()),
                                "entry_price": round(o, 1),
                                "shares":      shares,
                                "peak_close":  round(o, 1),
                            }
                            r_trades.append({
                                "date": str(today.date()), "symbol": sym,
                                "shares": shares, "action": "buy",
                                "price": round(o, 1), "profit": 0,
                                "reason": "entry",
                            })
                            exec_log.append(f"  [{sym}] {NAMES.get(sym,sym)} BUY {shares}株 "
                                            f"@ ¥{o:,.0f}  コスト ¥{cost:,.0f}")
                        else:
                            exec_log.append(f"  [{sym}] BUY スキップ (資金不足)")
                else:
                    exec_log.append(f"  [{sym}] BUY スキップ (スロット満杯)")

    # ── Update positions at today's close, generate new pending ─────────────
    n225_sma25 = n225_close.rolling(MARKET_SMA).mean()
    n225_above = (n225_close > n225_sma25)
    n225_ok    = bool(n225_above.get(today, True))
    n225_val   = float(n225_close.get(today, float("nan")))
    n225_sma_v = float(n225_sma25.get(today, float("nan")))

    today_close = {}
    signal_log  = []

    for sym in active:
        c = price(sym, today, "close")
        if c is None:
            continue
        today_close[sym] = c

        if sym in positions:
            pos = positions[sym]
            pos["peak_close"] = max(float(pos.get("peak_close", pos["entry_price"])), c)
            ep = float(pos["entry_price"])
            pk = float(pos["peak_close"])

            if sym not in new_pending:
                if c <= ep * (1 + sl_pct):
                    new_pending[sym] = {"action": "sell", "reason": "stop_loss", "deferred": 0}
                    signal_log.append(f"  [{sym}] {NAMES.get(sym,sym)} → 明日SELL (ストップロス -3%)")
                elif c <= pk * (1 - ts_pct):
                    new_pending[sym] = {"action": "sell", "reason": "trailing_stop", "deferred": 0}
                    signal_log.append(f"  [{sym}] {NAMES.get(sym,sym)} → 明日SELL (トレーリング利確)")
                else:
                    try:
                        row = ind_all[sym].loc[today]
                        if bool(row["dead_cross"]) or bool(row["below_sma"]):
                            new_pending[sym] = {"action": "sell", "reason": "signal", "deferred": 0}
                            signal_log.append(f"  [{sym}] {NAMES.get(sym,sym)} → 明日SELL (DC/SMA割れ)")
                        else:
                            signal_log.append(f"  [{sym}] {NAMES.get(sym,sym)} 保有継続")
                    except KeyError:
                        signal_log.append(f"  [{sym}] {NAMES.get(sym,sym)} 保有継続 (データなし)")

        elif sym not in new_pending:
            try:
                row   = ind_all[sym].loc[today]
                rsi_v = float(row.get("rsi", float("nan")))
                adx_v = float(row.get("adx", float("nan")))
                rsi_ok = not np.isnan(rsi_v) and rsi_v >= RSI_THRESHOLD
                adx_ok = not np.isnan(adx_v) and adx_v >= ADX_THRESHOLD
                if (not pd.isna(row["sma"]) and
                        bool(row["above_sma"]) and bool(row["golden_cross"])):
                    if n225_ok and rsi_ok and adx_ok:
                        new_pending[sym] = {"action": "buy", "deferred": 0}
                        signal_log.append(
                            f"  [{sym}] {NAMES.get(sym,sym)} → 明日BUY "
                            f"(GC+SMA上+地合いOK RSI={rsi_v:.1f} ADX={adx_v:.1f})")
                    elif not n225_ok:
                        signal_log.append(
                            f"  [{sym}] {NAMES.get(sym,sym)} "
                            f"【スキップ】市場地合い悪化のため購入を見送りました")
                    else:
                        signal_log.append(
                            f"  [{sym}] {NAMES.get(sym,sym)} "
                            f"【スキップ】RSI/ADX強度不足 "
                            f"(RSI={rsi_v:.1f}, ADX={adx_v:.1f})")
            except KeyError:
                pass

    # ── Compute portfolio summary ───────────────────────────────────────────
    positions_val = sum(
        int(positions[s]["shares"]) * today_close.get(s, float(positions[s]["entry_price"]))
        for s in positions
    )
    total_val = cash + positions_val

    # ── Save state ──────────────────────────────────────────────────────────
    state["cash"]               = round(cash, 0)
    state["positions"]          = positions
    state["pending_orders"]     = new_pending
    state["realized_trades"]    = r_trades[-100:]
    state["total_realized_pnl"] = round(total_pnl, 0)
    state["last_updated"]       = str(today.date())
    save_portfolio(state)

    # ══ Print formatted output ═══════════════════════════════════════════════
    W = 68
    print(f"\n{'═'*W}")
    print(f"  【J-Titan 投資AI  明日の注文指示】")
    print(f"  処理日: {today.date()}  パラメータ: SMA={sma_period}, "
          f"SL={sl_pct*100:.0f}%, TS={ts_pct*100:.0f}%")
    print(f"{'─'*W}")

    # Asset summary
    print(f"  ◆ 現在の資産状況 (本日終値ベース)")
    print(f"    現金残高          : ¥{cash:>14,.0f}")
    print(f"    保有株評価額      : ¥{positions_val:>14,.0f}")
    print(f"    確定済み損益      : ¥{total_pnl:>+14,.0f}")
    print(f"    {'─'*38}")
    print(f"    総資産 (時価)     : ¥{total_val:>14,.0f}")

    # Positions
    if positions:
        print(f"\n  ◆ 保有銘柄 ({len(positions)} 銘柄)")
        for sym, pos in positions.items():
            ep  = float(pos["entry_price"])
            sh  = int(pos["shares"])
            pk  = float(pos.get("peak_close", ep))
            c   = today_close.get(sym, ep)
            unr = (c - ep) * sh - (ep * sh + c * sh) * COMMISSION
            pct = (c / ep - 1) * 100
            sl_line = ep * (1 + sl_pct)
            ts_line = pk * (1 - ts_pct)
            print(f"    [{sym}] {NAMES.get(sym,sym):<16}: {sh}株  取得¥{ep:,.0f}  "
                  f"現在¥{c:,.0f}  含損益¥{unr:+,.0f} ({pct:+.1f}%)")
            print(f"          損切ライン ¥{sl_line:,.0f}  最高値 ¥{pk:,.0f}  "
                  f"TS発動ライン ¥{ts_line:,.0f}")
    else:
        print(f"\n  ◆ 保有銘柄: なし")

    # Tomorrow's orders
    buys  = {s: v for s, v in new_pending.items() if v["action"] == "buy"}
    sells = {s: v for s, v in new_pending.items() if v["action"] == "sell"}

    print(f"\n{'─'*W}")
    print(f"  ◆ 【明日の注文指示】  ({today.date()} 翌営業日 朝9:00 成行注文)")

    if buys:
        print(f"\n  ★ 買い注文")
        for sym, order in buys.items():
            c   = today_close.get(sym, 0)
            ep  = c  # approximate (actual open unknown)
            pv  = total_val
            sld = ep * abs(sl_pct)
            slot_cap = pv / MAX_SLOTS
            lots = min(
                int(pv * RISK_PER_TRADE / sld / LOT) if sld > 0 else 0,
                int(slot_cap / (ep * (1 + COMMISSION)) / LOT),
            )
            sh   = lots * LOT
            cost = sh * ep * (1 + COMMISSION)
            sl_p = ep * (1 + sl_pct)
            print(f"    [{sym}] {NAMES.get(sym,sym)}: {sh}株  成行買い  "
                  f"推定¥{cost:,.0f}  損切 ¥{sl_p:,.0f}")

    if sells:
        print(f"\n  ✗ 売り・損切り・利確")
        for sym, order in sells.items():
            reason_jp = {
                "stop_loss": "ストップロス", "trailing_stop": "トレーリング利確",
                "signal": "MACDシグナル", "forced": "強制決済",
            }.get(order.get("reason", ""), order.get("reason", ""))
            sh = int(positions.get(sym, {}).get("shares", 0))
            if sh > 0:
                print(f"    [{sym}] {NAMES.get(sym,sym)}: {sh}株 全株売り  ({reason_jp})")

    if not buys and not sells:
        print(f"    なし (全銘柄 様子見)")

    # Market filter
    n225_status = f"¥{n225_val:,.0f}" if not np.isnan(n225_val) else "N/A"
    sma_status  = f"¥{n225_sma_v:,.0f}" if not np.isnan(n225_sma_v) else "N/A"
    mkt_label   = "▲ 地合い良好 (買い許可)" if n225_ok else "▼ 地合い悪化 (買い全スキップ)"
    print(f"\n{'─'*W}")
    print(f"  ◆ 日経平均 地合い判定")
    print(f"    N225終値: {n225_status}  SMA{MARKET_SMA}: {sma_status}  → {mkt_label}")

    # Today's processing log
    if exec_log or signal_log:
        print(f"\n{'─'*W}")
        print(f"  ◆ 本日の処理ログ")
        for line in exec_log + signal_log:
            print(line)

    # Recent trades
    recent = [t for t in r_trades if t.get("action") == "sell"][-5:]
    if recent:
        print(f"\n{'─'*W}")
        print(f"  ◆ 直近の確定トレード")
        for t in reversed(recent):
            reason_jp = {
                "stop_loss": "損切", "trailing_stop": "利確TS",
                "signal": "シグナル", "forced": "強制",
            }.get(t.get("reason", ""), t.get("reason", ""))
            p = float(t.get("profit", 0))
            print(f"    {t['date']}  [{t['symbol']}] {NAMES.get(t['symbol'],t['symbol']):<16}"
                  f" {t['shares']}株売  ¥{p:>+,.0f}  ({reason_jp})")

    print(f"\n  portfolio.json 更新完了: {PORTFOLIO_PATH}")
    print(f"{'═'*W}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="J-Titan Engine v2")
    ap.add_argument("--mode", choices=["backtest", "auto"], default="backtest",
                    help="backtest (最適化+テスト) / auto (毎日の自動ペーパートレード)")
    args = ap.parse_args()

    print(f"\n{'═'*68}")
    print(f"  J-Titan Engine v2  —  Japanese Swing Trade AI [決定版]")
    print(f"  Mode: {args.mode.upper()}  |  {datetime.now(TOKYO_TZ).strftime('%Y-%m-%d %H:%M')}")
    print(f"{'═'*68}")

    # Load data — auto: refresh latest; backtest: load CSV or fetch if missing
    loader = load_and_refresh if args.mode == "auto" else load_or_fetch
    print("\n  データ読み込み中 ...")
    df_all = {}
    for sym in SYMBOLS:
        df = loader(sym)
        if df is not None and len(df) > 0:
            df_all[sym] = df
            suffix = " (最新データ取得済)" if args.mode == "auto" else ""
            print(f"    [{sym}] {NAMES[sym]:<16}: {len(df)} 取引日{suffix}")
        else:
            print(f"    [{sym}] {NAMES[sym]:<16}: ファイルなし — スキップ")

    n225_df    = loader("N225")
    n225_close = load_n225_series(n225_df)
    print(f"    [N225] 日経平均          : {len(n225_close)} 取引日  "
          f"(地合いフィルター SMA{MARKET_SMA} 有効)")

    if args.mode == "backtest":
        run_backtest(df_all, n225_close)
    else:
        run_auto(df_all, n225_close)
