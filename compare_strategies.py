#!/usr/bin/env python3
"""
J-Titan 戦略比較スクリプト
- 道A: 現戦略（日本株 + RSI上限75 + ROE15% + 日経52週高値85%）
- 道B: 米国株（同シグナルをNASDAQ/S&P上位40銘柄に適用）
- 道C: 道A + 決算モメンタム（各WF期間開始時点の純利益YoY成長10%以上）

Usage:
  python3 compare_strategies.py [--skip-earnings-fetch]
"""

import sys, os, json, time, random, argparse
sys.path.insert(0, os.path.dirname(__file__))

from datetime import timedelta
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

from j_titan_engine import (
    SYMBOLS, NAMES, TOKYO_TZ,
    INITIAL_CAPITAL, LOT, COMMISSION, MAX_SLOTS, LEVERAGE_FACTOR,
    MARGIN_RATE, RISK_PER_TRADE, RSI_THRESHOLD, RSI_MAX, ADX_THRESHOLD,
    VOLUME_RATIO_MIN, RS_LOOKBACK, SMA_PERIODS, ATR_STOP_MULTS,
    ATR_PERIODS, TRAILING_RATES, PARTIAL_PROFIT_R, PARTIAL_PROFIT_R2,
    MIN_HOLD_DAYS_SMA, STOP_COOLDOWN_DAYS, TIME_STOP_DAYS, TIME_STOP_MIN_PNL,
    ATR_BE_TRIGGER, MARKET_SMA, MARKET_SMA_SLOW, N225_HIGH_52W_MIN,
    build_indicators, portfolio_backtest, load_or_fetch, load_n225_series,
    _normalise,
)

# ══════════════════════════════════════════════════════════════════════════════
# 道B — 米国株ユニバース
# ══════════════════════════════════════════════════════════════════════════════
US_SYMBOLS = [
    # 半導体・ハードウェア
    "NVDA", "TSM", "AVGO", "AMD", "QCOM", "TXN", "MU", "AMAT", "LRCX",
    # ソフトウェア・クラウド
    "MSFT", "ORCL", "CRM", "ADBE", "NOW",
    # ビッグテック
    "AAPL", "AMZN", "GOOGL", "META",
    # ヘルスケア
    "UNH", "LLY", "ABBV", "MRK", "TMO",
    # 金融
    "JPM", "V", "MA", "GS", "BLK",
    # 消費財・小売
    "COST", "HD", "NKE",
    # エネルギー
    "XOM", "CVX",
    # 航空宇宙・防衛
    "RTX", "LMT",
    # 通信・メディア
    "T", "VZ",
]
US_NAMES = {
    "NVDA": "NVIDIA",    "TSM": "TSMC",        "AVGO": "Broadcom",
    "AMD": "AMD",        "QCOM": "Qualcomm",   "TXN": "TI",
    "MU": "Micron",      "AMAT": "AppMaterials","LRCX": "LamResearch",
    "MSFT": "Microsoft", "ORCL": "Oracle",     "CRM": "Salesforce",
    "ADBE": "Adobe",     "NOW": "ServiceNow",
    "AAPL": "Apple",     "AMZN": "Amazon",     "GOOGL": "Alphabet",
    "META": "Meta",
    "UNH": "UnitedHealth","LLY": "EliLilly",   "ABBV": "AbbVie",
    "MRK": "Merck",      "TMO": "ThermoFisher",
    "JPM": "JPMorgan",   "V": "Visa",          "MA": "Mastercard",
    "GS": "Goldman",     "BLK": "BlackRock",
    "COST": "Costco",    "HD": "HomeDepot",    "NKE": "Nike",
    "XOM": "ExxonMobil", "CVX": "Chevron",
    "RTX": "RTX",        "LMT": "Lockheed",
    "T": "AT&T",         "VZ": "Verizon",
}
SPXMARKET     = "^GSPC"
US_DATA_DIR   = "data/us"
EARNINGS_CACHE_PATH = "data/earnings_growth_cache.json"
EARNINGS_GROWTH_MIN = 0.10   # 純利益YoY成長10%以上

# 米国株は高モメンタムでRSIが日本株より高くなるため専用閾値を使用
US_RSI_THRESHOLD = 55.0   # 日本株より低め
US_RSI_MAX       = 82.0   # 米国株は80超えも正常
US_ADX_THRESHOLD = 20.0   # 日本株より緩め

# ══════════════════════════════════════════════════════════════════════════════
# 米国株データ取得
# ══════════════════════════════════════════════════════════════════════════════
def fetch_us_stock(code: str, start, end) -> pd.DataFrame:
    """米国株をyfinanceで取得（.Tなし）"""
    for attempt in range(3):
        try:
            df = yf.download(code, start=start, end=end,
                             auto_adjust=True, progress=False)
            if not df.empty:
                df = _normalise(df)
                if "Volume" in df.columns:
                    df = df[df["Volume"].notna() & (df["Volume"] > 0)]
                return df
        except Exception as e:
            if attempt < 2:
                time.sleep(5 + random.uniform(0, 3))
    return None


def load_us_data(min_history: int = 1400) -> dict:
    """米国株の7年データをロード（キャッシュ優先）"""
    os.makedirs(US_DATA_DIR, exist_ok=True)
    from datetime import datetime
    today = datetime.now(TOKYO_TZ).date()
    start = today - timedelta(days=7 * 365 + 5)
    df_all = {}
    print(f"  米国株データ読み込み中 ({len(US_SYMBOLS)} 銘柄)...")
    for code in US_SYMBOLS:
        path = f"{US_DATA_DIR}/{code}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC").tz_convert(TOKYO_TZ)
            else:
                df.index = df.index.tz_convert(TOKYO_TZ)
        else:
            print(f"    [{code}] 取得中...", flush=True)
            df = fetch_us_stock(code, start, today)
            if df is not None:
                df.to_csv(path)
            time.sleep(0.5 + random.uniform(0, 0.3))
        if df is not None and len(df) >= min_history:
            df_all[code] = df
    print(f"  通過: {len(df_all)} 銘柄")
    return df_all


def load_spx_series() -> pd.Series:
    """S&P500 終値系列を取得"""
    from datetime import datetime
    today = datetime.now(TOKYO_TZ).date()
    start = today - timedelta(days=7 * 365 + 5)
    path  = f"{US_DATA_DIR}/SPX.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(TOKYO_TZ)
        else:
            df.index = df.index.tz_convert(TOKYO_TZ)
    else:
        df = fetch_us_stock(SPXMARKET, start, today)
        if df is not None:
            df.to_csv(path)
    return df["Close"] if df is not None and "Close" in df.columns else pd.Series(dtype=float)


def build_us_market_filter(spx_close: pd.Series, idx: pd.DatetimeIndex) -> np.ndarray:
    """S&P500 版地合いフィルター（SMA25 & SMA75 & 52週高値85%以上）"""
    sma_fast = spx_close.rolling(MARKET_SMA).mean()
    sma_slow = spx_close.rolling(MARKET_SMA_SLOW).mean()
    high_52w = spx_close.rolling(252).max()
    near_h   = spx_close / high_52w.replace(0, np.nan) >= N225_HIGH_52W_MIN
    above    = ((spx_close > sma_fast) & (spx_close > sma_slow)
                & near_h).reindex(idx).ffill().fillna(True)
    return above.values.astype(bool)


# ══════════════════════════════════════════════════════════════════════════════
# 道C — 決算モメンタムキャッシュ
# ══════════════════════════════════════════════════════════════════════════════
def build_earnings_cache(codes: list, force: bool = False) -> dict:
    """
    各銘柄の過去年次決算データをキャッシュ。
    形式: {code: [(report_date, net_income), ...]}  新しい順
    """
    if not force and os.path.exists(EARNINGS_CACHE_PATH):
        with open(EARNINGS_CACHE_PATH, encoding="utf-8") as f:
            cache = json.load(f)
        print(f"  決算キャッシュ読み込み: {len(cache)} 銘柄")
        return cache

    print(f"  決算データ取得中 ({len(codes)} 銘柄)... ※数分かかります")
    cache = {}
    for i, code in enumerate(codes, 1):
        try:
            fin = yf.Ticker(f"{code}.T").financials
            if fin is not None and not fin.empty and "Net Income" in fin.index:
                ni = fin.loc["Net Income"]
                records = sorted(
                    [(str(d.date()), float(v))
                     for d, v in ni.items() if pd.notna(v)],
                    reverse=True
                )
                if records:
                    cache[code] = records
        except Exception:
            pass
        if i % 50 == 0:
            print(f"    {i}/{len(codes)} 完了...", flush=True)
        time.sleep(0.3 + random.uniform(0, 0.2))

    os.makedirs("data", exist_ok=True)
    with open(EARNINGS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"  決算キャッシュ保存: {len(cache)} 銘柄 → {EARNINGS_CACHE_PATH}")
    return cache


def get_earnings_growth_at(code: str, as_of: pd.Timestamp,
                            cache: dict) -> float:
    """
    as_of 時点で利用可能な最新年次報告書のYoY純利益成長率を返す。
    データなし・計算不可の場合は NaN を返す。
    """
    records = cache.get(code, [])
    if not records:
        return float("nan")
    # as_of 以前の報告書のみ使用（ルックアヘッドバイアス回避）
    as_of_naive = as_of.tz_localize(None) if as_of.tzinfo else as_of
    available = [(d, v) for d, v in records if pd.Timestamp(d) <= as_of_naive]
    if len(available) < 2:
        return float("nan")
    latest_ni = available[0][1]
    prior_ni  = available[1][1]
    if prior_ni <= 0 or latest_ni == prior_ni:
        return float("nan")
    return (latest_ni - prior_ni) / abs(prior_ni)


# ══════════════════════════════════════════════════════════════════════════════
# ウォークフォワード（共通ロジック）
# ══════════════════════════════════════════════════════════════════════════════
def run_wf_core(df_all: dict, market_close: pd.Series,
                names: dict, market_filter_fn,
                test_window: int = 252,
                min_train: int = 500,
                active_filter=None,
                rsi_lo: float = None, rsi_hi: float = None,
                adx_thr: float = None) -> dict:
    """
    共通ウォークフォワード実行。
    active_filter: (active_list, test_start_date) -> filtered_active_list
    """
    # パラメータオーバーライド（道B米国株用）
    _rsi_lo  = rsi_lo  if rsi_lo  is not None else RSI_THRESHOLD
    _rsi_hi  = rsi_hi  if rsi_hi  is not None else RSI_MAX
    _adx_thr = adx_thr if adx_thr is not None else ADX_THRESHOLD

    active = [s for s in df_all.keys()]
    ind_ref    = {s: build_indicators(df_all[s], 25) for s in active}
    common_idx = reduce(lambda a, b: a.intersection(b),
                        [ind_ref[s].index for s in active]).sort_values()
    n = len(common_idx)

    periods, pos = [], min_train
    while pos < n:
        tr = common_idx[:pos]
        te = common_idx[pos:min(pos + test_window, n)]
        if len(te) >= 20:
            periods.append((tr, te))
        pos += test_window

    if not periods:
        return {"error": "データ不足"}

    all_ind_full = {
        (sma, atr_p): {s: build_indicators(df_all[s], sma, atr_p).reindex(common_idx)
                       for s in active}
        for sma in SMA_PERIODS for atr_p in ATR_PERIODS
    }

    results, all_assets = [], []
    for pidx, (train_idx, test_idx) in enumerate(periods):
        n_yrs   = len(train_idx) / 252
        act      = active_filter(active, test_idx[0]) if active_filter else active
        mkt_tr   = market_filter_fn(market_close, train_idx)
        mkt_te   = market_filter_fn(market_close, test_idx)
        n225_tr  = market_close.reindex(train_idx).ffill().bfill().fillna(0).values.astype(float)
        n225_te  = market_close.reindex(test_idx).ffill().bfill().fillna(0).values.astype(float)

        # 米国株用パラメータ一時適用（モンキーパッチ）
        import j_titan_engine as _eng
        _orig_rsi_lo, _orig_rsi_hi, _orig_adx = _eng.RSI_THRESHOLD, _eng.RSI_MAX, _eng.ADX_THRESHOLD
        _eng.RSI_THRESHOLD = _rsi_lo
        _eng.RSI_MAX       = _rsi_hi
        _eng.ADX_THRESHOLD = _adx_thr

        best_calmar, best_params = -float("inf"), None
        best_asset_fallback, best_params_fallback = -float("inf"), (30, 2.0, 0.10, 20)
        for sma, am, ts, ap in product(SMA_PERIODS, ATR_STOP_MULTS,
                                        TRAILING_RATES, ATR_PERIODS):
            ind_tr = {s: all_ind_full[(sma, ap)][s].reindex(train_idx) for s in act}
            r = portfolio_backtest(ind_tr, am, ts, mkt_tr, train_idx,
                                   n225_arr=n225_tr)
            ann = r["total_return"] / n_yrs
            mdd = abs(r["max_drawdown"])
            cal = ann / mdd if mdd > 0 and r["total_trades"] >= 5 and r["total_return"] > 0 \
                  else -float("inf")
            if cal > best_calmar:
                best_calmar, best_params = cal, (sma, am, ts, ap)
            if r["final_asset"] > best_asset_fallback:
                best_asset_fallback = r["final_asset"]
                best_params_fallback = (sma, am, ts, ap)
        if best_params is None:
            best_params = best_params_fallback

        # パラメータ復元
        _eng.RSI_THRESHOLD = _orig_rsi_lo
        _eng.RSI_MAX       = _orig_rsi_hi
        _eng.ADX_THRESHOLD = _orig_adx

        bs, ba, bt, bp = best_params
        ind_te = {s: all_ind_full[(bs, bp)][s].reindex(test_idx) for s in act}
        _eng.RSI_THRESHOLD = _rsi_lo
        _eng.RSI_MAX       = _rsi_hi
        _eng.ADX_THRESHOLD = _adx_thr
        res = portfolio_backtest(ind_te, ba, bt, mkt_te, test_idx,
                                 n225_arr=n225_te, enable_shorts=False)
        _eng.RSI_THRESHOLD = _orig_rsi_lo
        _eng.RSI_MAX       = _orig_rsi_hi
        _eng.ADX_THRESHOLD = _orig_adx
        results.append({
            "period":  pidx + 1,
            "tr_end":  train_idx[-1].date(),
            "te_start": test_idx[0].date(),
            "te_end":  test_idx[-1].date(),
            "params":  best_params,
            "te_ret":  res["total_return"],
            "te_mdd":  res["max_drawdown"],
            "te_trades": res["total_trades"],
            "asset_series": res["asset_series"],
        })
        all_assets.append(res["asset_series"])
        print(f"  期間{pidx+1}/{len(periods)} → テスト {res['total_return']:+.1f}%  "
              f"DD{res['max_drawdown']:.1f}%  {res['total_trades']}件")

    # エクイティ連結
    cur, parts = float(INITIAL_CAPITAL), []
    for a in all_assets:
        scaled = a * (cur / INITIAL_CAPITAL)
        parts.append(scaled)
        cur = float(scaled.iloc[-1])
    combined = pd.concat(parts)
    n_yrs_tot = len(combined) / 252
    ann = ((combined.iloc[-1] / INITIAL_CAPITAL) ** (1 / n_yrs_tot) - 1) * 100
    rm  = combined.cummax()
    mdd = float(((combined - rm) / rm * 100).min())
    cal = ann / abs(mdd) if mdd < 0 else 0.0

    return {
        "periods":  results,
        "combined": combined,
        "ann_ret":  ann,
        "max_dd":   mdd,
        "calmar":   cal,
    }


# ══════════════════════════════════════════════════════════════════════════════
# メイン比較実行
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-earnings-fetch", action="store_true",
                    help="決算データを再取得せず既存キャッシュを使う")
    ap.add_argument("--min-history", type=int, default=1400)
    args = ap.parse_args()

    print(f"\n{'═'*68}")
    print(f"  J-Titan 戦略比較  道A / 道B / 道C")
    print(f"{'═'*68}")

    # ── 共通データ読み込み ──────────────────────────────────────────────────
    print("\n[1/4] 日本株データ読み込み...")
    jp_data = {}
    for s in SYMBOLS:
        df = load_or_fetch(s)
        if df is not None and len(df) >= args.min_history:
            jp_data[s] = df
    n225_df    = load_or_fetch("N225")
    n225_close = load_n225_series(n225_df)
    print(f"  日本株: {len(jp_data)} 銘柄 / N225: {len(n225_close)} 日")

    print("\n[2/4] 米国株データ読み込み...")
    us_data    = load_us_data(min_history=args.min_history)
    spx_close  = load_spx_series()
    print(f"  米国株: {len(us_data)} 銘柄 / S&P500: {len(spx_close)} 日")

    print("\n[3/4] 決算データ準備...")
    jp_codes = list(jp_data.keys())
    earnings_cache = build_earnings_cache(
        jp_codes, force=not args.skip_earnings_fetch
    )
    earnings_ok_count = sum(1 for c in jp_codes if c in earnings_cache)
    print(f"  決算データあり: {earnings_ok_count}/{len(jp_codes)} 銘柄")

    from j_titan_engine import build_market_filter_arr as jp_mkt_filter

    # ── 道A: 現戦略 ─────────────────────────────────────────────────────────
    print("\n[4/4] ウォークフォワード実行中...\n")
    print("  ── 道A: 日本株現戦略 ──")
    result_a = run_wf_core(
        jp_data, n225_close, NAMES, jp_mkt_filter,
        min_train=500,
    )

    # ── 道B: 米国株 ──────────────────────────────────────────────────────────
    print("\n  ── 道B: 米国株戦略 ──")
    result_b = run_wf_core(
        us_data, spx_close, US_NAMES, build_us_market_filter,
        min_train=500,
        rsi_lo=US_RSI_THRESHOLD, rsi_hi=US_RSI_MAX, adx_thr=US_ADX_THRESHOLD,
    )

    # ── 道C: 道A + 決算モメンタム ────────────────────────────────────────────
    print("\n  ── 道C: 日本株 + 決算モメンタム ──")

    def earnings_active_filter(active, test_start_date):
        """test_start_date 時点で決算成長10%以上の銘柄のみ返す"""
        filtered = []
        as_of = pd.Timestamp(test_start_date)
        for code in active:
            g = get_earnings_growth_at(code, as_of, earnings_cache)
            if np.isnan(g) or g >= EARNINGS_GROWTH_MIN:
                filtered.append(code)
        return filtered

    result_c = run_wf_core(
        jp_data, n225_close, NAMES, jp_mkt_filter,
        min_train=500,
        active_filter=earnings_active_filter,
    )

    # ── 道A+B 50/50 合成 ─────────────────────────────────────────────────────
    def combine_50_50(res_x: dict, res_y: dict) -> dict:
        """2戦略の日次リターンを50/50で合成してエクイティ曲線・統計を返す。"""
        cx = res_x["combined"]
        cy = res_y["combined"]
        common = cx.index.intersection(cy.index)
        cx, cy = cx.reindex(common), cy.reindex(common)
        ret = 0.5 * cx.pct_change().fillna(0) + 0.5 * cy.pct_change().fillna(0)
        equity = INITIAL_CAPITAL * (1 + ret).cumprod()
        n_yrs  = len(equity) / 252
        ann    = ((equity.iloc[-1] / INITIAL_CAPITAL) ** (1 / n_yrs) - 1) * 100
        rm     = equity.cummax()
        mdd    = float(((equity - rm) / rm * 100).min())
        cal    = ann / abs(mdd) if mdd < 0 else 0.0
        return {"combined": equity, "ann_ret": ann, "max_dd": mdd, "calmar": cal}

    result_ab = combine_50_50(result_a, result_b)

    # ── 比較サマリー ─────────────────────────────────────────────────────────
    print(f"\n{'═'*68}")
    print(f"  ウォークフォワード 比較結果 (2021〜2026  5期間)")
    print(f"{'─'*68}")
    print(f"  {'戦略':<8}  {'年率':>6}  {'最大DD':>7}  {'Calmar':>7}  期間別リターン")
    print(f"  {'─'*8}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*30}")
    for label, res in [("道A", result_a), ("道B", result_b), ("道C", result_c)]:
        if "error" in res:
            print(f"  {label:<8}  ERROR: {res['error']}")
            continue
        rets = "  ".join(f"{r['te_ret']:+.1f}%" for r in res["periods"])
        print(f"  {label:<8}  {res['ann_ret']:>+5.1f}%  {res['max_dd']:>+6.1f}%"
              f"  {res['calmar']:>7.2f}  {rets}")
    print(f"  {'─'*68}")
    ab = result_ab
    print(f"  {'道A+B(半々)':<8}  {ab['ann_ret']:>+5.1f}%  {ab['max_dd']:>+6.1f}%"
          f"  {ab['calmar']:>7.2f}  (合成のため期間別なし)")
    print(f"{'─'*68}")
    print(f"  現在(道A)  年率 {result_a['ann_ret']:+.1f}%  Calmar {result_a['calmar']:.2f}")
    print(f"  道A+B合成  年率 {ab['ann_ret']:+.1f}%  Calmar {ab['calmar']:.2f}"
          f"  MaxDD {ab['max_dd']:+.1f}%")
    print(f"{'═'*68}\n")

    # ── グラフ ───────────────────────────────────────────────────────────────
    os.makedirs("output", exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(13, 15),
                             gridspec_kw={"height_ratios": [1, 1, 1, 1]})
    colors = {"道A": "steelblue", "道B": "darkorange", "道C": "forestgreen",
              "道A+B": "purple"}

    plot_items = [("道A", result_a), ("道B", result_b), ("道C", result_c),
                  ("道A+B", result_ab)]
    for ax, (label, res) in zip(axes, plot_items):
        if "error" in res:
            ax.text(0.5, 0.5, f"{label}: {res['error']}", transform=ax.transAxes,
                    ha="center", va="center")
            continue
        combined = res["combined"]
        dates    = combined.index.tz_convert(None) if combined.index.tz else combined.index
        ax.plot(dates, combined.values, color=colors[label], lw=1.5, label=label)
        ax.axhline(INITIAL_CAPITAL, color="dimgray", ls="--", lw=0.8)
        ax.fill_between(dates, combined.values, INITIAL_CAPITAL,
                        where=combined.values >= INITIAL_CAPITAL, alpha=0.2,
                        color=colors[label])
        ax.fill_between(dates, combined.values, INITIAL_CAPITAL,
                        where=combined.values < INITIAL_CAPITAL, alpha=0.15, color="red")
        ax.set_title(
            f"{label}  AnnRet={res['ann_ret']:+.1f}%  MaxDD={res['max_dd']:.1f}%"
            f"  Calmar={res['calmar']:.2f}",
            fontsize=10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"¥{x:,.0f}"))
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

    plt.suptitle("J-Titan A / B / C / A+B  Walk-Forward Comparison", fontsize=12, y=1.01)
    plt.tight_layout()
    out = "output/strategy_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  グラフ保存 → {out}\n")

    # ── 結果をJSONで保存 ────────────────────────────────────────────────────
    summary = {
        "道A": {"ann_ret": result_a["ann_ret"], "max_dd": result_a["max_dd"],
                "calmar":  result_a["calmar"],
                "periods": [{"period":r["period"],"te_ret":r["te_ret"]}
                            for r in result_a["periods"]]},
        "道B": {"ann_ret": result_b.get("ann_ret", "N/A"),
                "max_dd":  result_b.get("max_dd", "N/A"),
                "calmar":  result_b.get("calmar", "N/A"),
                "periods": [{"period":r["period"],"te_ret":r["te_ret"]}
                            for r in result_b.get("periods",[])]},
        "道C": {"ann_ret": result_c["ann_ret"], "max_dd": result_c["max_dd"],
                "calmar":  result_c["calmar"],
                "periods": [{"period":r["period"],"te_ret":r["te_ret"]}
                            for r in result_c["periods"]]},
        "道A+B": {"ann_ret": result_ab["ann_ret"], "max_dd": result_ab["max_dd"],
                  "calmar":  result_ab["calmar"]},
    }
    with open("output/strategy_comparison.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"  結果JSON保存 → output/strategy_comparison.json\n")


if __name__ == "__main__":
    main()
