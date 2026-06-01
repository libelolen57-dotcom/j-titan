#!/usr/bin/env python3
"""
J-Titan Signal Screener
全監視銘柄を毎朝スキャンし、本日 買いシグナルが出ている銘柄を一覧表示する。

Usage:
  python3 screen_signals.py            # 全銘柄スクリーニング
  python3 screen_signals.py --fetch    # yfinance から最新データを取得してスクリーニング
  python3 screen_signals.py --top 10   # 上位10銘柄のみ表示
"""

import argparse
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz

# ── j_titan_engine の定数・関数を再利用 ─────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from j_titan_engine import (
    SYMBOLS, NAMES, TICKER_MAP, TOKYO_TZ,
    MARKET_SMA, RSI_THRESHOLD, ADX_THRESHOLD, VOLUME_RATIO_MIN,
    build_indicators, load_csv, load_or_fetch,
)
try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

# ── スクリーニング対象のSMA（portfolio.jsonがあれば読み込む） ────────────────
SMA_DEFAULT = 25

def _load_sma_from_portfolio() -> int:
    import json
    path = os.path.join(os.path.dirname(__file__), "portfolio.json")
    if os.path.exists(path):
        with open(path) as f:
            p = json.load(f)
        return int(p.get("params", {}).get("sma", SMA_DEFAULT))
    return SMA_DEFAULT


def score_signal(row: pd.Series, prev_row: pd.Series) -> float:
    """
    シグナル強度スコア（0〜100）を計算する。
    RSI・ADX・出来高比率を正規化して合成。
    """
    rsi = float(row.get("rsi", 50))
    adx = float(row.get("adx", 0))
    vr  = float(row.get("vol_ratio", 1.0))

    rsi_score = min(max((rsi - RSI_THRESHOLD) / (100 - RSI_THRESHOLD) * 40, 0), 40)
    adx_score = min(max((adx - ADX_THRESHOLD) / (60 - ADX_THRESHOLD) * 40, 0), 40)
    vr_score  = min(max((vr - VOLUME_RATIO_MIN) / (3.0 - VOLUME_RATIO_MIN) * 20, 0), 20)
    return round(rsi_score + adx_score + vr_score, 1)


def run_screener(use_fetch: bool, top_n: int) -> None:
    sma_period = _load_sma_from_portfolio()
    loader     = load_or_fetch if use_fetch else load_csv

    today_jst  = datetime.now(TOKYO_TZ)
    print(f"\n{'═'*70}")
    print(f"  J-Titan シグナルスクリーナー  {today_jst.strftime('%Y-%m-%d %H:%M')}")
    print(f"  SMA={sma_period}  RSI≥{RSI_THRESHOLD:.0f}  ADX≥{ADX_THRESHOLD:.0f}  "
          f"出来高比率≥{VOLUME_RATIO_MIN:.2f}x")
    print(f"{'─'*70}")

    # ── N225 地合いフィルター ─────────────────────────────────────────────
    n225_df = loader("N225")
    if n225_df is not None and "Close" in n225_df.columns:
        n225_c   = n225_df["Close"]
        sma25    = n225_c.rolling(MARKET_SMA).mean()
        n225_now = float(n225_c.iloc[-1])
        sma_now  = float(sma25.iloc[-1])
        mkt_ok   = n225_now > sma_now
        mkt_label = "▲ 地合い良好（買い許可）" if mkt_ok else "▼ 地合い悪化（買いスキップ推奨）"
        print(f"  日経平均: ¥{n225_now:,.0f}  SMA{MARKET_SMA}: ¥{sma_now:,.0f}  {mkt_label}")
    else:
        mkt_ok = True
        print("  日経平均: データなし（地合いフィルター無効）")

    print(f"{'─'*70}")

    results   = []
    skipped   = 0
    no_data   = 0

    for sym in SYMBOLS:
        df = loader(sym)
        if df is None or len(df) < 60:
            no_data += 1
            continue

        # 必要カラムを確認
        needed = {"Open", "High", "Low", "Close", "Volume"}
        if not needed.issubset(df.columns):
            no_data += 1
            continue

        ind = build_indicators(df, sma_period)
        if len(ind) < 2:
            no_data += 1
            continue

        today_row = ind.iloc[-1]
        prev_row  = ind.iloc[-2]
        date_str  = ind.index[-1].date()

        # ── シグナル判定 ──────────────────────────────────────────────────
        gc   = bool(today_row["golden_cross"])
        abv  = bool(today_row["above_sma"])
        sma_v = float(today_row["sma"])

        if np.isnan(sma_v) or not (gc and abv):
            skipped += 1
            continue

        rsi_v = float(today_row["rsi"])
        adx_v = float(today_row["adx"])
        vr_v  = float(today_row["vol_ratio"])

        rsi_ok = not np.isnan(rsi_v) and rsi_v >= RSI_THRESHOLD
        adx_ok = not np.isnan(adx_v) and adx_v >= ADX_THRESHOLD
        vr_ok  = not np.isnan(vr_v)  and vr_v  >= VOLUME_RATIO_MIN

        # フィルター通過チェック（地合い除く）
        pass_filters = rsi_ok and adx_ok and vr_ok

        results.append({
            "symbol":   sym,
            "name":     NAMES.get(sym, sym),
            "date":     date_str,
            "close":    float(today_row["close"]),
            "sma":      sma_v,
            "rsi":      rsi_v,
            "adx":      adx_v,
            "vr":       vr_v,
            "rsi_ok":   rsi_ok,
            "adx_ok":   adx_ok,
            "vr_ok":    vr_ok,
            "mkt_ok":   mkt_ok,
            "all_ok":   pass_filters and mkt_ok,
            "score":    score_signal(today_row, prev_row),
        })

    # ── 表示 ─────────────────────────────────────────────────────────────
    if not results:
        print("  本日 ゴールデンクロス銘柄なし\n")
        return

    # 「全フィルター通過」を上位に、次にスコア順
    results.sort(key=lambda x: (not x["all_ok"], -x["score"]))
    if top_n:
        results = results[:top_n]

    buy_list    = [r for r in results if r["all_ok"]]
    partial_list = [r for r in results if not r["all_ok"]]

    if buy_list:
        print(f"\n  ★ 買いシグナル確定（全フィルター通過）  {len(buy_list)} 銘柄")
        print(f"  {'コード':<6} {'銘柄名':<18} {'終値':>8} {'RSI':>6} {'ADX':>6} "
              f"{'出来高比':>8} {'スコア':>6}")
        print(f"  {'─'*6} {'─'*18} {'─'*8} {'─'*6} {'─'*6} {'─'*8} {'─'*6}")
        for r in buy_list:
            print(f"  {r['symbol']:<6} {r['name']:<18} ¥{r['close']:>7,.0f} "
                  f"{r['rsi']:>5.1f}  {r['adx']:>5.1f}  {r['vr']:>6.2f}x  {r['score']:>5.1f}")

    if partial_list:
        print(f"\n  △ GC成立だがフィルター未通過  {len(partial_list)} 銘柄")
        print(f"  {'コード':<6} {'銘柄名':<18} {'終値':>8} {'RSI':>6} {'ADX':>6} "
              f"{'出来高比':>8} 未通過理由")
        print(f"  {'─'*6} {'─'*18} {'─'*8} {'─'*6} {'─'*6} {'─'*8} {'─'*10}")
        for r in partial_list:
            reasons = []
            if not r["mkt_ok"]:  reasons.append("地合い悪")
            if not r["rsi_ok"]:  reasons.append(f"RSI={r['rsi']:.1f}")
            if not r["adx_ok"]:  reasons.append(f"ADX={r['adx']:.1f}")
            if not r["vr_ok"]:   reasons.append(f"VR={r['vr']:.2f}x")
            print(f"  {r['symbol']:<6} {r['name']:<18} ¥{r['close']:>7,.0f} "
                  f"{r['rsi']:>5.1f}  {r['adx']:>5.1f}  {r['vr']:>6.2f}x  "
                  f"{' / '.join(reasons)}")

    total_gc = len(buy_list) + len(partial_list)
    print(f"\n{'─'*70}")
    print(f"  スキャン: {len(SYMBOLS)} 銘柄  GC銘柄: {total_gc}  "
          f"買い確定: {len(buy_list)}  データなし: {no_data}")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="J-Titan Signal Screener")
    ap.add_argument("--fetch", action="store_true",
                    help="yfinanceから最新データを取得してスクリーニング")
    ap.add_argument("--top", type=int, default=0,
                    help="表示銘柄数の上限（0=全件）")
    args = ap.parse_args()
    run_screener(use_fetch=args.fetch, top_n=args.top)
