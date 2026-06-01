import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import pytz

TICKERS = {
    "7203.T": "7203",
    "6758.T": "6758",
    "4689.T": "4689",
}

TOKYO_TZ = pytz.timezone("Asia/Tokyo")
END_DATE = datetime.now(TOKYO_TZ).date()
START_DATE = END_DATE - timedelta(days=5 * 365 + 2)

os.makedirs("data", exist_ok=True)

for ticker_symbol, filename in TICKERS.items():
    print(f"Fetching {ticker_symbol} ...")
    df = yf.download(ticker_symbol, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

    if df.empty:
        print(f"  WARNING: No data for {ticker_symbol}")
        continue

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Convert index timezone to Asia/Tokyo
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(TOKYO_TZ)
    else:
        df.index = df.index.tz_convert(TOKYO_TZ)

    # Reindex to full calendar range and forward-fill weekends/holidays
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D", tz=TOKYO_TZ)
    df = df.reindex(full_index)
    df.ffill(inplace=True)

    # Drop rows that are still NaN (before first trading day)
    df.dropna(how="all", inplace=True)

    df.index.name = "Date"
    output_path = os.path.join("data", f"{filename}.csv")
    df.to_csv(output_path)
    print(f"  Saved {len(df)} rows to {output_path}")

print("Done.")
