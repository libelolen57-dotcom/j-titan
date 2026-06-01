#!/usr/bin/env python3
"""
Fetch 5-year daily data for 8 Japanese stocks + Nikkei 225 (TSE).
Saves actual trading days only as CSV (Asia/Tokyo timezone).
"""
import os
from datetime import datetime, timedelta

import pytz
import pandas as pd
import yfinance as yf

STOCKS = {
    "7203.T": ("7203", "トヨタ自動車"),
    "6758.T": ("6758", "ソニーグループ"),
    "4689.T": ("4689", "LINEヤフー"),
    "9432.T": ("9432", "NTT"),
    "8306.T": ("8306", "三菱UFJ"),
    "8411.T": ("8411", "みずほFG"),
    "4502.T": ("4502", "武田薬品"),
    "6526.T": ("6526", "ソシオネクスト"),
}

INDICES = {
    "^N225": ("N225", "日経平均"),
}

TOKYO_TZ   = pytz.timezone("Asia/Tokyo")
END_DATE   = datetime.now(TOKYO_TZ).date()
START_DATE = END_DATE - timedelta(days=5 * 365 + 5)

os.makedirs("data", exist_ok=True)


def fetch_and_save(ticker_sym: str, code: str, name: str) -> None:
    print(f"  [{code}] {name:<16}", end=" ... ", flush=True)
    df = yf.download(ticker_sym, start=START_DATE, end=END_DATE,
                     auto_adjust=True, progress=False)
    if df.empty:
        print("WARNING: no data")
        return
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(TOKYO_TZ)
    else:
        df.index = df.index.tz_convert(TOKYO_TZ)
    df.sort_index(inplace=True)
    df.index.name = "Date"
    path = f"data/{code}.csv"
    df.to_csv(path)
    print(f"{len(df)} trading days  →  {path}")


print(f"Fetching {len(STOCKS)} stocks + {len(INDICES)} index  "
      f"({START_DATE} to {END_DATE})")
print()

print("  ── 個別銘柄 ──")
for ticker_sym, (code, name) in STOCKS.items():
    fetch_and_save(ticker_sym, code, name)

print()
print("  ── 市場インデックス ──")
for ticker_sym, (code, name) in INDICES.items():
    fetch_and_save(ticker_sym, code, name)

print("\nDone.")
