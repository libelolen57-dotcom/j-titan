#!/usr/bin/env python3
"""
J-Titan Signal Screener — 動的ユニバース版
毎朝 JPX 公式 CSV から東証プライム銘柄を取得し、
時価総額・売買代金・株価でフィルタリング。
上位 200 銘柄に対してハイブリッドシグナルを判定し、
買いシグナル銘柄を一覧表示する。

Usage:
  python3 screen_signals.py                # 動的選定（デフォルト）
  python3 screen_signals.py --fetch        # yfinance から最新データ取得
  python3 screen_signals.py --top 10       # 上位10銘柄のみ表示
  python3 screen_signals.py --static       # 固定 SYMBOLS リスト（旧モード）
  python3 screen_signals.py --rebuild-cache  # ユニバースキャッシュ強制再構築
"""

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from j_titan_engine import (
    SYMBOLS as SYMBOLS_STATIC, NAMES as NAMES_STATIC, TOKYO_TZ,
    MARKET_SMA, RSI_THRESHOLD, ADX_THRESHOLD, VOLUME_RATIO_MIN,
    build_indicators, load_csv, load_or_fetch,
)
try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False


# ══════════════════════════════════════════════════════════════════════════════
# 定数
# ══════════════════════════════════════════════════════════════════════════════
JPX_CSV_URL         = (
    "https://www.jpx.co.jp/markets/statistics-equities"
    "/misc/tvdivq0000001vg2-att/data_j.xls"
)
UNIVERSE_CACHE_PATH = "data/universe_cache.json"
MKTCAP_MIN_JPY      = 500_0000_0000    # 500億円
PRICE_MIN_JPY       = 500              # 500円
TURNOVER_MIN_JPY    = 10_0000_0000     # 10億円/日
UNIVERSE_TOP_N      = 200
CACHE_REFRESH_DAYS  = 7

# ── レート制限・リトライ設定 ──────────────────────────────────────────────────
RETRY_MAX        = 3   # 最大リトライ回数
RETRY_BASE_WAIT  = 10  # 基本待機秒数（指数バックオフの底）
RETRY_JITTER_MAX = 5   # ジッター上限秒数（thundering herd 防止）
MKTCAP_WORKERS   = 4   # 並列スレッド数（8→4 に削減してレート制限を緩和）

SMA_DEFAULT = 25


# ══════════════════════════════════════════════════════════════════════════════
# リトライユーティリティ
# ══════════════════════════════════════════════════════════════════════════════
def _is_rate_limited(exc: Exception) -> bool:
    """yfinance / requests の 429 Too Many Requests を検出"""
    msg = str(exc).lower()
    return "429" in msg or "too many requests" in msg or "rate limit" in msg


def _wait(attempt: int, base: int = RETRY_BASE_WAIT, label: str = "") -> None:
    """指数バックオフ + ジッターで待機（429 は長め、その他は一定）"""
    secs = base * (2 ** attempt) + random.uniform(0, RETRY_JITTER_MAX)
    tag  = f" ({label})" if label else ""
    print(f"    → {secs:.0f}秒待機してリトライ {attempt + 1}/{RETRY_MAX}{tag}",
          flush=True)
    time.sleep(secs)


def _yf_download_with_retry(tickers: list, period: str = "1mo",
                             **kwargs) -> pd.DataFrame:
    """
    yf.download のラッパー。
    429 → 指数バックオフ（10 / 20 / 40 秒）でリトライ。
    空データ → 一定待機（10 秒）でリトライ。
    全リトライ失敗時は空 DataFrame を返す。
    """
    for attempt in range(RETRY_MAX):
        try:
            df = yf.download(tickers, period=period,
                             auto_adjust=True, progress=False, **kwargs)
            if not df.empty:
                return df
            # 空データは一時的な障害として扱う
            if attempt < RETRY_MAX - 1:
                print(f"    WARNING: データが空 (attempt {attempt+1})")
                _wait(attempt, base=RETRY_BASE_WAIT, label="空データ")
        except Exception as e:
            if _is_rate_limited(e):
                print(f"    WARNING: 429 レート制限 (attempt {attempt+1}): {e}")
                if attempt < RETRY_MAX - 1:
                    _wait(attempt, base=RETRY_BASE_WAIT, label="429")
            elif attempt < RETRY_MAX - 1:
                print(f"    WARNING: 取得エラー (attempt {attempt+1}): {e}")
                _wait(0, base=RETRY_BASE_WAIT, label="一時エラー")
            else:
                print(f"    ERROR: 全リトライ失敗: {e}")
    return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# Step 1: 銘柄ユニバース構築（週次キャッシュ）
# ══════════════════════════════════════════════════════════════════════════════
def fetch_prime_codes() -> list:
    """
    JPX 公式 Excel から東証プライム・内国株式の4桁コードを取得。
    ネットワーク障害に備えて最大 RETRY_MAX 回リトライする。
    """
    for attempt in range(RETRY_MAX):
        try:
            print(f"  JPX公式CSVダウンロード中 (attempt {attempt+1}) ...", flush=True)
            df = pd.read_excel(JPX_CSV_URL, header=0, dtype=str)

            df.columns = df.columns.str.strip()
            mkt_col  = next((c for c in df.columns if "市場" in c), None)
            code_col = next((c for c in df.columns if "コード" in c), None)
            if mkt_col is None or code_col is None:
                print(f"    WARNING: 期待するカラムが見つかりません: {list(df.columns)}")
                return []

            prime = df[df[mkt_col].str.contains("プライム（内国株式）", na=False)]
            codes = prime[code_col].str.strip().tolist()
            codes = [c for c in codes
                     if isinstance(c, str) and len(c) == 4 and c.isdigit()]
            print(f"    東証プライム（内国株式）: {len(codes)} 銘柄取得")
            return codes

        except Exception as e:
            print(f"    WARNING: JPX CSV 取得失敗 (attempt {attempt+1}): {e}")
            if attempt < RETRY_MAX - 1:
                _wait(0, base=RETRY_BASE_WAIT, label="JPX CSV")

    print("    ERROR: JPX CSV 全リトライ失敗")
    return []


def _fetch_single_mktcap(code: str) -> tuple:
    """
    単一銘柄の時価総額を取得（ThreadPoolExecutor 用）。
    429 時は指数バックオフでリトライ。
    """
    for attempt in range(RETRY_MAX):
        try:
            fi = yf.Ticker(f"{code}.T").fast_info
            mc = getattr(fi, "market_cap", None)
            return (code, int(mc) if mc else None)
        except Exception as e:
            if _is_rate_limited(e) and attempt < RETRY_MAX - 1:
                time.sleep(RETRY_BASE_WAIT * (2 ** attempt)
                           + random.uniform(0, RETRY_JITTER_MAX))
            else:
                break
    return (code, None)


def build_universe_cache(prime_codes: list) -> dict:
    """全プライム銘柄の時価総額を並列取得 → 500億円以上をキャッシュ（週次）"""
    print(f"  時価総額フィルター構築中 ({len(prime_codes)} 銘柄) ...")
    print(f"  ※ 初回・週次更新は数分かかります（{MKTCAP_WORKERS}並列）")
    mktcap_map = {}
    done = 0
    with ThreadPoolExecutor(max_workers=MKTCAP_WORKERS) as ex:
        futures = {ex.submit(_fetch_single_mktcap, c): c for c in prime_codes}
        for fut in as_completed(futures):
            code, mc = fut.result()
            if mc and mc >= MKTCAP_MIN_JPY:
                mktcap_map[code] = mc
            done += 1
            if done % 200 == 0:
                print(f"    進捗 {done}/{len(prime_codes)} ... "
                      f"通過 {len(mktcap_map)} 銘柄", flush=True)

    print(f"  時価総額500億円以上: {len(mktcap_map)} 銘柄")
    os.makedirs("data", exist_ok=True)
    with open(UNIVERSE_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "updated":    datetime.now(TOKYO_TZ).isoformat(),
            "mktcap_min": MKTCAP_MIN_JPY,
            "codes":      mktcap_map,
        }, f, ensure_ascii=False, indent=2)
    return mktcap_map


def load_universe_cache(force_rebuild: bool = False) -> dict:
    """
    キャッシュ読み込みロジック（優先順位順）:
      1. キャッシュ有効（7日以内）かつ force_rebuild=False → そのまま返す
      2. 期限切れ or force_rebuild → JPX CSV + 時価総額で再構築を試みる
      3. 再構築失敗 → 期限切れの旧キャッシュで代替（JPX障害時のフォールバック）
      4. キャッシュが存在しない → 固定 SYMBOLS にフォールバック
    """
    stale_codes = None  # 期限切れでも保持しておくフォールバック用

    # ── 既存キャッシュを確認 ───────────────────────────────────────────────
    if os.path.exists(UNIVERSE_CACHE_PATH):
        try:
            with open(UNIVERSE_CACHE_PATH, encoding="utf-8") as f:
                cache = json.load(f)

            stale_codes = cache.get("codes", {})  # 後段フォールバック用に退避

            if not force_rebuild:
                updated_str = cache.get("updated", "2000-01-01T00:00:00+00:00")
                updated = datetime.fromisoformat(updated_str)
                if updated.tzinfo is None:
                    updated = TOKYO_TZ.localize(updated)
                age = (datetime.now(TOKYO_TZ) - updated).days
                if age < CACHE_REFRESH_DAYS:
                    print(f"  ユニバースキャッシュ: {len(stale_codes)} 銘柄 "
                          f"({age}日前更新)")
                    return stale_codes
                print(f"  ユニバースキャッシュ期限切れ ({age}日) → 再構築試行")

        except Exception as e:
            print(f"  キャッシュ読み込みエラー ({e}) → 再構築試行")

    # ── 再構築試行 ──────────────────────────────────────────────────────────
    prime_codes = fetch_prime_codes()
    if prime_codes:
        try:
            return build_universe_cache(prime_codes)
        except Exception as e:
            print(f"  WARNING: キャッシュ再構築失敗 ({e})")

    # ── フォールバック: 旧キャッシュ（期限切れ）で代替 ──────────────────────
    if stale_codes:
        print(f"  WARNING: JPX CSV 取得失敗 → 旧キャッシュ（期限切れ）で代替 "
              f"({len(stale_codes)} 銘柄)")
        return stale_codes

    # ── 最終フォールバック: 固定 SYMBOLS ──────────────────────────────────
    print("  WARNING: キャッシュなし → 固定SYMBOLSにフォールバック")
    return {s: 0 for s in SYMBOLS_STATIC}


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: 売買代金・株価フィルター（毎日、高速）
# ══════════════════════════════════════════════════════════════════════════════
def select_dynamic_symbols(top_n: int = UNIVERSE_TOP_N,
                           force_rebuild: bool = False) -> list:
    """
    1. 時価総額キャッシュ（週次）から候補コードを取得
    2. yf.download（429リトライ付き）で1ヶ月OHLCV一括取得
    3. 株価≥500円 & 平均売買代金≥10億円/日 でフィルタリング
    4. 売買代金降順で上位 top_n 銘柄を返す
    """
    if not HAS_YF:
        return list(SYMBOLS_STATIC)

    mktcap_map = load_universe_cache(force_rebuild=force_rebuild)
    candidates = list(mktcap_map.keys())
    if not candidates:
        return list(SYMBOLS_STATIC)

    print(f"\n  売買代金・株価フィルター ({len(candidates)} 銘柄 → 1ヶ月データ一括取得) ...",
          flush=True)
    tickers = [f"{c}.T" for c in candidates]

    raw = _yf_download_with_retry(tickers, period="1mo")
    if raw.empty:
        print("  WARNING: データ取得失敗 → 固定SYMBOLSにフォールバック")
        return list(SYMBOLS_STATIC)

    is_multi = isinstance(raw.columns, pd.MultiIndex)
    scored   = []

    for code in candidates:
        t = f"{code}.T"
        try:
            if is_multi:
                close_s = raw["Close"]
                vol_s   = raw["Volume"]
                if t not in close_s.columns:
                    continue
                close = close_s[t].dropna()
                vol   = vol_s[t].dropna()
            else:
                close = raw["Close"].dropna()
                vol   = raw["Volume"].dropna()

            if len(close) < 10:
                continue
            price = float(close.iloc[-1])
            if price < PRICE_MIN_JPY:
                continue
            idx      = close.index.intersection(vol.index)
            turnover = float((close.loc[idx] * vol.loc[idx]).mean())
            if turnover < TURNOVER_MIN_JPY:
                continue
            scored.append((code, price, turnover))
        except Exception:
            pass

    scored.sort(key=lambda x: -x[2])
    selected = [code for code, _, _ in scored[:top_n]]

    print(f"  フィルター通過: {len(scored)} 銘柄 → 上位 {len(selected)} 銘柄を採用")
    if scored:
        print(f"\n  {'順位':<4} {'コード':<7} {'銘柄名':<20} {'株価':>8} {'平均売買代金/日':>14}")
        print(f"  {'─'*4} {'─'*7} {'─'*20} {'─'*8} {'─'*14}")
        for rank, (code, price, to) in enumerate(scored[:10], 1):
            name = NAMES_STATIC.get(code, code)
            print(f"  {rank:<4} {code:<7} {name:<20} ¥{price:>6,.0f}  "
                  f"¥{to/1e8:>8.1f}億円")
        if len(scored) > 10:
            print(f"  ... 他 {len(scored)-10} 銘柄")
    return selected


# ══════════════════════════════════════════════════════════════════════════════
# シグナルスコア計算
# ══════════════════════════════════════════════════════════════════════════════
def score_signal(row: pd.Series, prev_row: pd.Series) -> float:
    rsi = float(row.get("rsi", 50))
    adx = float(row.get("adx", 0))
    vr  = float(row.get("vol_ratio", 1.0))
    rsi_score = min(max((rsi - RSI_THRESHOLD) / (100 - RSI_THRESHOLD) * 40, 0), 40)
    adx_score = min(max((adx - ADX_THRESHOLD) / (60 - ADX_THRESHOLD) * 40, 0), 40)
    vr_score  = min(max((vr - VOLUME_RATIO_MIN) / (3.0 - VOLUME_RATIO_MIN) * 20, 0), 20)
    return round(rsi_score + adx_score + vr_score, 1)


# ══════════════════════════════════════════════════════════════════════════════
# スクリーナー本体
# ══════════════════════════════════════════════════════════════════════════════
def _load_sma_from_portfolio() -> int:
    path = os.path.join(os.path.dirname(__file__), "portfolio.json")
    if os.path.exists(path):
        with open(path) as f:
            p = json.load(f)
        return int(p.get("params", {}).get("sma", SMA_DEFAULT))
    return SMA_DEFAULT


def run_screener(use_fetch: bool, top_n: int,
                 static_mode: bool = False,
                 force_rebuild: bool = False) -> None:
    sma_period = _load_sma_from_portfolio()
    today_jst  = datetime.now(TOKYO_TZ)

    # ── 銘柄リスト選定 ──────────────────────────────────────────────────
    if static_mode or not HAS_YF:
        symbols    = list(SYMBOLS_STATIC)
        names      = NAMES_STATIC
        mode_label = f"固定リスト ({len(symbols)} 銘柄)"
        loader     = load_or_fetch if use_fetch else load_csv
    else:
        symbols    = select_dynamic_symbols(UNIVERSE_TOP_N,
                                            force_rebuild=force_rebuild)
        names      = {s: NAMES_STATIC.get(s, s) for s in symbols}
        mode_label = f"動的選定 上位{len(symbols)}銘柄"
        loader     = load_or_fetch   # 未取得銘柄を自動ダウンロード

    print(f"\n{'═'*72}")
    print(f"  J-Titan シグナルスクリーナー  {today_jst.strftime('%Y-%m-%d %H:%M')}")
    print(f"  モード    : {mode_label}")
    print(f"  フィルター: SMA={sma_period}  RSI≥{RSI_THRESHOLD:.0f}  "
          f"ADX≥{ADX_THRESHOLD:.0f}  出来高比率≥{VOLUME_RATIO_MIN:.2f}x")
    print(f"{'─'*72}")

    # ── N225 地合いフィルター ────────────────────────────────────────────
    n225_loader = load_or_fetch if HAS_YF else load_csv
    n225_df = n225_loader("N225")
    if n225_df is not None and "Close" in n225_df.columns:
        n225_c    = n225_df["Close"]
        sma25     = n225_c.rolling(MARKET_SMA).mean()
        n225_now  = float(n225_c.iloc[-1])
        sma_now   = float(sma25.iloc[-1])
        mkt_ok    = n225_now > sma_now
        mkt_label = "▲ 地合い良好（買い許可）" if mkt_ok else "▼ 地合い悪化（買いスキップ推奨）"
        print(f"  日経平均 : ¥{n225_now:,.0f}  SMA{MARKET_SMA}: ¥{sma_now:,.0f}  {mkt_label}")
    else:
        mkt_ok = True
        print("  日経平均 : データなし（地合いフィルター無効）")

    print(f"{'─'*72}")

    results = []
    no_data = 0
    skipped = 0

    for sym in symbols:
        df = loader(sym)
        if df is None or len(df) < 60:
            no_data += 1
            continue
        if not {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns):
            no_data += 1
            continue

        ind = build_indicators(df, sma_period)
        if len(ind) < 2:
            no_data += 1
            continue

        today_row = ind.iloc[-1]
        prev_row  = ind.iloc[-2]
        sma_v     = float(today_row["sma"])
        gc        = bool(today_row["golden_cross"])
        abv       = bool(today_row["above_sma"])

        if np.isnan(sma_v) or not (gc and abv):
            skipped += 1
            continue

        rsi_v = float(today_row["rsi"])
        adx_v = float(today_row["adx"])
        vr_v  = float(today_row["vol_ratio"])
        rsi_ok = not np.isnan(rsi_v) and rsi_v >= RSI_THRESHOLD
        adx_ok = not np.isnan(adx_v) and adx_v >= ADX_THRESHOLD
        vr_ok  = not np.isnan(vr_v)  and vr_v  >= VOLUME_RATIO_MIN

        results.append({
            "symbol":  sym,
            "name":    names.get(sym, sym),
            "date":    ind.index[-1].date(),
            "close":   float(today_row["close"]),
            "sma":     sma_v,
            "rsi":     rsi_v,
            "adx":     adx_v,
            "vr":      vr_v,
            "rsi_ok":  rsi_ok,
            "adx_ok":  adx_ok,
            "vr_ok":   vr_ok,
            "mkt_ok":  mkt_ok,
            "all_ok":  rsi_ok and adx_ok and vr_ok and mkt_ok,
            "score":   score_signal(today_row, prev_row),
        })

    # ── 表示 ─────────────────────────────────────────────────────────────
    if not results:
        print(f"\n  本日 ゴールデンクロス銘柄なし")
        print(f"\n{'─'*72}")
        print(f"  スキャン: {len(symbols)} 銘柄  GC銘柄: 0  データなし: {no_data}")
        print(f"{'═'*72}\n")
        return

    results.sort(key=lambda x: (not x["all_ok"], -x["score"]))
    if top_n:
        results = results[:top_n]

    buy_list     = [r for r in results if r["all_ok"]]
    partial_list = [r for r in results if not r["all_ok"]]

    if buy_list:
        print(f"\n  ★ 買いシグナル確定（全フィルター通過）  {len(buy_list)} 銘柄")
        print(f"  {'コード':<7} {'銘柄名':<20} {'終値':>8} {'RSI':>6} {'ADX':>6} "
              f"{'出来高比':>8} {'スコア':>6}")
        print(f"  {'─'*7} {'─'*20} {'─'*8} {'─'*6} {'─'*6} {'─'*8} {'─'*6}")
        for r in buy_list:
            print(f"  {r['symbol']:<7} {r['name']:<20} ¥{r['close']:>7,.0f} "
                  f"{r['rsi']:>5.1f}  {r['adx']:>5.1f}  {r['vr']:>6.2f}x  {r['score']:>5.1f}")

    if partial_list:
        print(f"\n  △ GC成立だがフィルター未通過  {len(partial_list)} 銘柄")
        print(f"  {'コード':<7} {'銘柄名':<20} {'終値':>8} {'RSI':>6} {'ADX':>6} "
              f"{'出来高比':>8} 未通過理由")
        print(f"  {'─'*7} {'─'*20} {'─'*8} {'─'*6} {'─'*6} {'─'*8} {'─'*10}")
        for r in partial_list:
            reasons = []
            if not r["mkt_ok"]:  reasons.append("地合い悪")
            if not r["rsi_ok"]:  reasons.append(f"RSI={r['rsi']:.1f}")
            if not r["adx_ok"]:  reasons.append(f"ADX={r['adx']:.1f}")
            if not r["vr_ok"]:   reasons.append(f"VR={r['vr']:.2f}x")
            print(f"  {r['symbol']:<7} {r['name']:<20} ¥{r['close']:>7,.0f} "
                  f"{r['rsi']:>5.1f}  {r['adx']:>5.1f}  {r['vr']:>6.2f}x  "
                  f"{' / '.join(reasons)}")

    total_gc = len(buy_list) + len(partial_list)
    print(f"\n{'─'*72}")
    print(f"  スキャン: {len(symbols)} 銘柄  GC銘柄: {total_gc}  "
          f"買い確定: {len(buy_list)}  データなし: {no_data}")
    print(f"{'═'*72}\n")


# ══════════════════════════════════════════════════════════════════════════════
# エントリーポイント
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="J-Titan Signal Screener — 動的ユニバース版"
    )
    ap.add_argument("--fetch",          action="store_true",
                    help="yfinanceから最新データを取得してスクリーニング")
    ap.add_argument("--top",            type=int, default=0,
                    help="表示銘柄数の上限（0=全件）")
    ap.add_argument("--static",         action="store_true",
                    help="固定SYMBOLSリストを使用（旧モード）")
    ap.add_argument("--rebuild-cache",  action="store_true",
                    help="ユニバースキャッシュを強制再構築（毎週月曜日に推奨）")
    args = ap.parse_args()

    run_screener(
        use_fetch=args.fetch,
        top_n=args.top,
        static_mode=args.static,
        force_rebuild=args.rebuild_cache,
    )
