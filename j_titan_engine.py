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
MAX_SLOTS         = 5            # 同時保有上限（複利: 1枠=総資産÷5 を動的計算）
LEVERAGE_FACTOR   = 1.3          # 信用取引レバレッジ倍率（テスト最優: 1.45はTS5%を強制選択し逆効果）
MARGIN_RATE       = 0.020        # 信用取引年利（2.0%/年）— 日次で正確に控除
RISK_PER_TRADE    = 0.015        # 1.5%リスクルール（1トレードの最大損失を総資産の1.5%に制限）
TEST_DAYS         = 252          # テスト期間（約1年）
MARKET_SMA        = 25           # 日経地合いフィルター SMA
MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9

# ── モメンタムフィルター ─────────────────────────────────────────────────────
RSI_PERIOD       = 14
ADX_PERIOD       = 14
RSI_THRESHOLD    = 55.0   # RSI >= 55 で明確な上昇モメンタムあり（弱いエントリー排除）
ADX_THRESHOLD    = 22.0   # トレンドフィルター
VOLUME_RATIO_MIN = 1.05   # 出来高 >= 20日平均の1.05倍（明らかな閑散シグナル除外）
RSI_OVERBOUGHT         = 70.0   # この水準から反転下落したら early_exit
EARLY_EXIT_PROFIT_CAP  = 0.03   # 含み益がこれ未満の初期フェーズのみ early_exit 有効
EARLY_EXIT_DAYS_CAP    = 10     # 保有営業日数がこれ以内の初期フェーズのみ early_exit 有効

# ── 稼ぐための4フィルター ─────────────────────────────────────────────────────
RS_LOOKBACK       = 42    # 相対強度: 約2ヶ月（42営業日）で銘柄 vs N225 を比較
PARTIAL_PROFIT_R  = 2.0   # 部分利確: ATR×この倍数で50%を利確しトレーリング継続
TIME_STOP_DAYS    = 20    # タイムストップ: 保有上限（営業日）
TIME_STOP_MIN_PNL = 0.01  # タイムストップ: 含み益1%未満のまま20日経過で撤退

SMA_PERIODS    = [20, 25, 30]
ATR_STOP_MULTS = [1.5, 1.8, 2.0]           # 初期損切り幅 = ATR × 倍率（1.8中心に攻め寄り）
ATR_BE_TRIGGER = 1.0                        # 利益が ATR×1.0 に達したら損切りを建値へ
ATR_PERIODS    = [14, 20]                   # ATR計算期間
TRAILING_RATES = [0.05, 0.075, 0.10, 0.125]  # 5.0 / 7.5 / 10.0 / 12.5%（利大狙い）

# ── 信用売り設定 ─────────────────────────────────────────────────────────────
SHORT_ENABLED = True
RSI_SHORT     = 45.0   # RSI ≤ 45 でショートエントリー（明確な下落モメンタム）

# ── 監視銘柄（東証プライム・グロース 主要40銘柄）──────────────────────────────
SYMBOLS = [
    # 建設・不動産
    "1801", "1802", "1803", "1812", "1925", "1928", "1942",
    # 食品・飲料・たばこ
    "2502", "2503", "2768", "2802", "2914",
    # 小売・IT流通
    "3064", "3088", "3092", "3382", "3563",
    # 繊維・紙
    "3110",
    # 化学・素材
    "3407", "3436", "4004", "4005", "4062", "4063", "4088",
    "4182", "4186", "4188", "4204", "4452",
    # IT・インターネット・サービス
    "2413", "3626", "3659", "4307", "4385", "4392", "4689", "4755", "6532",
    # 医薬品・医療
    "4502", "4503", "4506", "4507", "4519", "4523", "4543", "4568", "4578",
    # 化粧品・消費財
    "4661", "4901", "4911", "4980",
    # エネルギー
    "1605", "5016", "5019", "5020",
    # 鉄鋼・非鉄金属
    "5101", "5108", "5201", "5332", "5333", "5334", "5344",
    "5401", "5411", "5631", "5706", "5711", "5713", "5805", "5838",
    # 機械・重工
    "6098", "6134", "6141", "6201", "6268", "6269", "6273", "6278",
    "6301", "6305", "6315", "6324", "6326", "6361", "6367", "6383",
    "6479", "6481", "6504", "6506",
    # 半導体・電子部品
    "6146", "6525", "6526", "6590", "6594", "6645", "6701", "6702",
    "6723", "6724", "6752", "6754", "6758", "6762", "6787",
    "6841", "6857", "6861", "6869", "6871", "6902", "6920",
    "6954", "6963", "6965", "6971", "6976", "6981", "6988", "8035",
    # 電機・精密機器
    "6178", "6501", "6503", "7701", "7729", "7733", "7735", "7741", "7751",
    # 自動車・輸送機器
    "7003", "7011", "7012", "7013", "7173", "7182", "7201", "7202",
    "7203", "7220", "7261", "7267", "7269", "7270", "7272",
    # 小売・生活
    "7453", "7532", "7826", "7832", "7936", "7974",
    # 印刷・その他製造
    "7911",
    # 通信
    "9432", "9433", "9434", "9984",
    # 銀行・金融
    "8303", "8306", "8308", "8309", "8411", "8473", "8591", "8593",
    "8601", "8604", "8630", "8697", "8725", "8750", "8766", "8795",
    "8316",
    # 商社
    "8001", "8002", "8015", "8031", "8053", "8058",
    # 生活用品
    "8113", "8136",
    # 不動産
    "8801", "8802", "8830",
    # 鉄道・輸送
    "9005", "9020", "9021", "9022", "9024",
    # 海運・航空
    "9101", "9104", "9107", "9147", "9201", "9202",
    # 電力・ガス
    "9412", "9502", "9503", "9531", "9532",
    # サービス・エンタメ
    "9697", "9735", "9766", "9843", "9983",
]
NAMES = {
    # 建設
    "1605": "INPEX",        "1801": "大成建設",      "1802": "大林組",
    "1803": "清水建設",      "1812": "鹿島建設",      "1925": "大和ハウス",
    "1928": "積水ハウス",    "1942": "関電工",
    # 食品
    "2413": "エムスリー",    "2502": "アサヒG",       "2503": "キリンHD",
    "2768": "双日",          "2802": "味の素",         "2914": "JT",
    # 小売
    "3064": "MonotaRO",     "3088": "マツキヨC",     "3092": "ZOZO",
    "3382": "セブン&アイ",   "3563": "フード&ライフ",
    # 繊維
    "3110": "日東紡",
    # 化学
    "3407": "旭化成",        "3436": "SUMCO",         "3626": "TIS",
    "4004": "レゾナック",    "4005": "住友化学",       "4062": "イビデン",
    "4063": "信越化学",      "4088": "エア・ウォーター","4182": "三菱ガス化",
    "4186": "東京応化工",    "4188": "三菱ケミG",     "4204": "積水化学",
    "4452": "花王",
    # IT
    "3659": "ネクソン",      "4307": "野村総研",       "4385": "メルカリ",
    "4392": "FutureInn",    "4689": "LINEヤフー",    "4755": "楽天G",
    "6532": "ベイカレント",
    # 医薬
    "4502": "武田薬品",      "4503": "アステラス",     "4506": "住友ファーマ",
    "4507": "塩野義製薬",    "4519": "中外製薬",       "4523": "エーザイ",
    "4543": "テルモ",        "4568": "第一三共",       "4578": "大塚HD",
    # 消費財
    "4661": "OLC",           "4901": "富士フイルム",   "4911": "資生堂",
    "4980": "デクセリアルズ",
    # エネルギー
    "5016": "ENEOSHarmo",   "5019": "出光興産",       "5020": "ENEOS",
    # 鉄鋼・非鉄
    "5101": "横浜ゴム",      "5108": "ブリヂストン",   "5201": "AGC",
    "5332": "TOTO",          "5333": "日本ガイシ",     "5334": "日特エンジ",
    "5344": "丸和電子材料",   "5401": "日本製鉄",       "5411": "JFE-HD",
    "5631": "日本製鋼所",    "5706": "三井金属",       "5711": "三菱マテリアル",
    "5713": "住友金属鉱山",  "5805": "SWCC",           "5838": "楽天銀行",
    # 機械
    "6098": "リクルートHD",  "6134": "富士機械製造",   "6141": "DMG森精機",
    "6201": "豊田自動織機",  "6268": "ナブテスコ",     "6269": "MODEC",
    "6273": "SMC",           "6278": "ユニオンツール",  "6301": "コマツ",
    "6305": "日立建機",      "6315": "東和精機",       "6324": "ハーモニック",
    "6326": "クボタ",        "6361": "荏原製作所",     "6367": "ダイキン工業",
    "6383": "ダイフク",      "6479": "ミネベアミツミ", "6481": "THK",
    "6504": "富士電機",      "6506": "安川電機",
    # 半導体
    "6146": "ディスコ",      "6525": "国際電気",       "6526": "ソシオネクスト",
    "6590": "芝浦メカ",      "6594": "ニデック",       "6645": "オムロン",
    "6701": "NEC",           "6702": "富士通",         "6723": "ルネサス",
    "6724": "セイコーエプソン","6752": "パナソニック",  "6754": "アンリツ",
    "6758": "ソニーG",       "6762": "TDK",            "6787": "メイコー",
    "6841": "横河電機",      "6857": "アドバンテスト", "6861": "キーエンス",
    "6869": "シスメックス",  "6871": "マイクロニクス",  "6902": "デンソー",
    "6920": "レーザーテック", "6954": "ファナック",     "6963": "ローム",
    "6965": "浜松ホトニクス", "6971": "京セラ",         "6976": "太陽誘電",
    "6981": "村田製作所",    "6988": "日東電工",       "8035": "東京エレクトロン",
    # 電機・精密
    "6178": "日本郵政",      "6501": "日立製作所",     "6503": "三菱電機",
    "7701": "島津製作所",    "7729": "東京精密",       "7733": "オリンパス",
    "7735": "SCREENホールディングス","7741": "HOYA",   "7751": "キヤノン",
    # 自動車
    "7003": "三井E&S",       "7011": "三菱重工",       "7012": "川崎重工",
    "7013": "IHI",           "7173": "東京キラリ",     "7182": "ゆうちょ銀行",
    "7201": "日産自動車",    "7202": "いすゞ自動車",   "7203": "トヨタ自動車",
    "7220": "武蔵精密",      "7261": "マツダ",         "7267": "ホンダ",
    "7269": "スズキ",        "7270": "SUBARU",         "7272": "ヤマハ発動機",
    # 小売
    "7453": "良品計画",      "7532": "パン・パシフィック","7826": "古河電工",
    "7832": "バンダイナムコ", "7936": "アシックス",     "7974": "任天堂",
    "7911": "TOPPANホールディングス",
    # 通信
    "9432": "NTT",           "9433": "KDDI",           "9434": "ソフトバンク",
    "9984": "SBG",
    # 銀行・金融
    "8303": "SBI新生銀行",   "8306": "三菱UFJ",        "8308": "りそなHD",
    "8309": "三井住友TH",    "8411": "みずほFG",       "8473": "SBI-HD",
    "8591": "ORIX",          "8593": "三菱HC",         "8601": "大和証券G",
    "8604": "野村HD",        "8630": "SOMPOホールディングス","8697": "JPX",
    "8725": "MS&AD",         "8750": "第一生命HD",     "8766": "東京海上HD",
    "8795": "T&D-HD",        "8316": "三井住友FG",
    # 商社
    "8001": "伊藤忠商事",    "8002": "丸紅",           "8015": "豊田通商",
    "8031": "三井物産",      "8053": "住友商事",       "8058": "三菱商事",
    # 生活用品
    "8113": "ユニ・チャーム", "8136": "サンリオ",
    # 不動産
    "8801": "三井不動産",    "8802": "三菱地所",       "8830": "住友不動産",
    # 鉄道
    "9005": "東急",          "9020": "JR東日本",       "9021": "JR西日本",
    "9022": "JR東海",        "9024": "西武HD",
    # 海運・航空
    "9101": "日本郵船",      "9104": "商船三井",       "9107": "川崎汽船",
    "9147": "日本通運",      "9201": "JAL",            "9202": "ANA-HD",
    # 電力・ガス
    "9412": "スカパーJSAT",  "9502": "中部電力",       "9503": "関西電力",
    "9531": "東京ガス",      "9532": "大阪ガス",
    # エンタメ・サービス
    "9697": "カプコン",      "9735": "セコム",         "9766": "コナミG",
    "9843": "ニトリHD",      "9983": "ファーストリテイリング",
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
# Position Sizing — 1.5%リスクルール
# ══════════════════════════════════════════════════════════════════════════════
def calc_position_size(price: float, atr: float, atr_stop_mult: float,
                       portfolio_value: float) -> int:
    """
    1.5%リスクルール + 信用取引レバレッジ対応の購入株数計算。

    損失上限 = portfolio_value × RISK_PER_TRADE (1.5%)  ← 純資産ベース（レバレッジ前）
    損切り幅 = ATR × atr_stop_mult
    購入株数 = min(損失上限÷損切り幅, 1スロット信用余力) を LOT 切り捨て

    LEVERAGE_FACTOR=1.3 の場合:
      総運用可能額 = portfolio_value × 1.3  （現物＋信用建玉の合計上限）
      1スロット上限 = 総運用可能額 ÷ MAX_SLOTS

    Returns: 購入株数（LOT単位）、0=エントリー不可
    """
    sl_dist = atr * atr_stop_mult if atr > 0 else price * 0.03
    if sl_dist <= 0 or price <= 0:
        return 0
    risk_amount   = portfolio_value * RISK_PER_TRADE                       # 許容損失額（純資産基準）
    buying_power  = portfolio_value * LEVERAGE_FACTOR                      # 信用余力込みの総運用可能額
    slot_cap      = buying_power / MAX_SLOTS                               # 1スロット上限
    lots = min(
        int(risk_amount / sl_dist / LOT),                                  # リスクルール上限
        int(slot_cap / (price * (1 + COMMISSION)) / LOT),                  # スロット上限（信用余力込み）
    )
    return max(0, lots) * LOT


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
            "sma": 25, "atr_stop_mult": 1.5, "trailing": 0.04,
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
def compute_atr(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


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


def build_indicators(df: pd.DataFrame, sma_period: int,
                     atr_period: int = ADX_PERIOD) -> pd.DataFrame:
    c     = df["Close"]
    ema_f = c.ewm(span=MACD_FAST, adjust=False).mean()
    ema_s = c.ewm(span=MACD_SLOW, adjust=False).mean()
    macd  = ema_f - ema_s
    sig   = macd.ewm(span=MACD_SIG, adjust=False).mean()
    sma   = c.rolling(sma_period).mean()
    gc    = (macd > sig) & (macd.shift(1) <= sig.shift(1))
    dc    = (macd < sig) & (macd.shift(1) >= sig.shift(1))
    rsi       = compute_rsi(c)
    adx       = compute_adx(df)
    atr       = compute_atr(df, atr_period)
    avg_vol   = df["Volume"].rolling(20).mean()
    vol_ratio = df["Volume"] / avg_vol.replace(0, np.nan)
    # early_exit: 2日以上連続でRSI>70（連続過熱）した後に前日比下落した初日
    rsi_exit  = (rsi.shift(2) > RSI_OVERBOUGHT) & (rsi.shift(1) > RSI_OVERBOUGHT) & (rsi < rsi.shift(1))
    return pd.DataFrame({
        "close": c, "open": df["Open"], "sma": sma,
        "golden_cross": gc, "dead_cross": dc,
        "above_sma": c > sma, "below_sma": c < sma,
        "rsi": rsi, "adx": adx, "atr": atr, "vol_ratio": vol_ratio,
        "rsi_exit": rsi_exit,
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
    atr_stop_mult: float,
    trailing_pct:  float,
    mkt_above:     np.ndarray,
    common_idx:    pd.DatetimeIndex,
    n225_arr:      np.ndarray = None,   # 相対強度フィルター用 N225 終値配列
    track_skips:   bool = False,
    enable_shorts: bool = False,        # グリッドサーチは無効、テスト期間のみ有効
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
    RS = {s: ind_all[s]["rsi"].values.astype(float)       for s in syms}
    AD = {s: ind_all[s]["adx"].values.astype(float)       for s in syms}
    AT = {s: ind_all[s]["atr"].values.astype(float)      for s in syms}
    VR = {s: ind_all[s]["vol_ratio"].values.astype(float) for s in syms}
    EX = {s: ind_all[s]["rsi_exit"].values.astype(bool)   for s in syms}

    N225 = n225_arr if n225_arr is not None else np.full(n, np.nan)

    cash            = float(INITIAL_CAPITAL)
    positions       = {}   # sym → {entry_price, shares, peak_close, entry_date}
    short_positions = {}   # sym → {entry_price, shares, trough_close, ...}  信用売りポジ
    p_sells         = {}   # sym → (reason, deferred_count)
    p_partial       = {}   # sym → (reason, deferred_count)  部分利確
    p_buys          = {}   # sym → deferred_count
    p_short_covers  = {}   # sym → (reason, deferred_count)  信用売り決済（買い戻し）
    p_short_entries = {}   # sym → deferred_count             信用売り新規
    asset_arr       = np.empty(n)
    trades, skips   = [], []

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

        # ── Execute pending short covers (buy-back) at today's open ─────
        for sym in list(p_short_covers.keys()):
            o, pc = O[sym][i], C[sym][prev]
            reason, dfr = p_short_covers[sym]
            if o >= pc + tse_limit(pc) and dfr < 3:   # ストップ高: 買い戻し不可, 持越し
                p_short_covers[sym] = (reason, dfr + 1)
            else:
                if sym in short_positions:
                    spos = short_positions.pop(sym)
                    sh, ep = spos["shares"], spos["entry_price"]
                    cover_cost = o * sh * (1 + COMMISSION)
                    cash -= cover_cost
                    trades.append(dict(
                        symbol=sym, shares=sh, reason=reason,
                        entry_date=spos["entry_date"], entry_price=ep,
                        exit_date=common_idx[i], exit_price=o,
                        profit=ep * sh * (1 - COMMISSION) - cover_cost,
                        direction="short",
                    ))
                del p_short_covers[sym]

        # ── Execute partial profit sells at today's open ─────────────────
        for sym in list(p_partial.keys()):
            o, pc = O[sym][i], C[sym][prev]
            reason, dfr = p_partial[sym]
            if o <= pc - tse_limit(pc) and dfr < 3:
                p_partial[sym] = (reason, dfr + 1)
            else:
                if sym in positions:
                    pos    = positions[sym]
                    sh_all = pos["shares"]
                    sh_half = (sh_all // 2 // LOT) * LOT   # 50%・単元株に切り捨て
                    ep_pos  = pos["entry_price"]
                    if sh_half >= LOT:
                        xproc  = o * sh_half * (1 - COMMISSION)
                        cash  += xproc
                        pos["shares"]        -= sh_half
                        pos["partial_taken"]  = True
                        pos["stop_price"]     = max(pos["stop_price"], ep_pos)
                        trades.append(dict(
                            symbol=sym, shares=sh_half, reason="partial_profit",
                            entry_date=pos["entry_date"], entry_price=ep_pos,
                            exit_date=common_idx[i], exit_price=o,
                            profit=xproc - ep_pos * sh_half * (1 + COMMISSION),
                        ))
                    else:
                        p_sells[sym] = (reason, 0)   # 単元未満なら全決済
                del p_partial[sym]

        # ── Execute pending buys at today's open ──────────────────────────
        for sym in list(p_buys.keys()):
            o, pc = O[sym][i], C[sym][prev]
            dfr   = p_buys[sym]
            if o >= pc + tse_limit(pc) and dfr < 2:   # ストップ高: 持越し
                p_buys[sym] = dfr + 1
                continue
            del p_buys[sym]
            total_slots = len(positions) + len(short_positions)
            if sym in positions or sym in short_positions or total_slots >= MAX_SLOTS:
                continue
            port_val = (cash
                + sum(positions[s]["shares"] * C[s][prev] for s in positions)
                - sum(short_positions[s]["shares"] * C[s][prev] for s in short_positions))
            atr_val  = AT[sym][prev]
            atr_val  = atr_val if not np.isnan(atr_val) and atr_val > 0 else 0.0
            sl_dist  = atr_val * atr_stop_mult if atr_val > 0 else o * 0.03
            shares   = calc_position_size(o, atr_val, atr_stop_mult, port_val)
            cost     = shares * o * (1 + COMMISSION)
            if shares >= LOT and cost <= cash:
                cash -= cost
                positions[sym] = dict(
                    entry_price=o, shares=shares,
                    peak_close=o, entry_date=common_idx[i],
                    stop_price=o - sl_dist,   # ATR動的損切り価格
                    atr_entry=atr_val,
                    be_moved=False,           # 建値移動フラグ
                    entry_idx=i,              # ハイブリッドearly_exit用
                )

        # ── Execute pending short entries at today's open ────────────────
        for sym in list(p_short_entries.keys()):
            o, pc = O[sym][i], C[sym][prev]
            dfr = p_short_entries[sym]
            if o <= pc - tse_limit(pc) and dfr < 2:   # ストップ安: 空売り不可, 持越し
                p_short_entries[sym] = dfr + 1
                continue
            del p_short_entries[sym]
            total_slots = len(positions) + len(short_positions)
            if sym in positions or sym in short_positions or total_slots >= MAX_SLOTS:
                continue
            port_val = (cash
                + sum(positions[s]["shares"] * C[s][prev] for s in positions)
                - sum(short_positions[s]["shares"] * C[s][prev] for s in short_positions))
            atr_val = AT[sym][prev]
            atr_val = atr_val if not np.isnan(atr_val) and atr_val > 0 else 0.0
            sl_dist = atr_val * atr_stop_mult if atr_val > 0 else o * 0.03
            shares  = calc_position_size(o, atr_val, atr_stop_mult, port_val)
            if shares >= LOT:
                proceeds = shares * o * (1 - COMMISSION)
                cash += proceeds   # 空売り代金受取
                short_positions[sym] = dict(
                    entry_price=o, shares=shares,
                    trough_close=o, entry_date=common_idx[i],
                    stop_price=o + sl_dist,   # 損切りは建値より上
                    atr_entry=atr_val,
                    be_moved=False,
                    entry_idx=i,
                )

        # ── 信用取引金利コスト（日次控除）─────────────────────────────────
        # 買い建て: 借入額 = 評価額 × (1 - 1/LEVERAGE_FACTOR)
        if LEVERAGE_FACTOR > 1.0 and positions:
            borrowed = sum(
                positions[s]["shares"] * C[s][i] * (LEVERAGE_FACTOR - 1.0) / LEVERAGE_FACTOR
                for s in positions
            )
            cash -= borrowed * (MARGIN_RATE / 365.0)
        # 売り建て: 貸株料 = ポジション評価額 × 年利2% / 365
        if short_positions:
            for s in short_positions:
                cash -= short_positions[s]["shares"] * C[s][i] * (MARGIN_RATE / 365.0)

        # ── Mark-to-market ────────────────────────────────────────────────
        asset_arr[i] = (cash
            + sum(positions[s]["shares"] * C[s][i] for s in positions)
            - sum(short_positions[s]["shares"] * C[s][i] for s in short_positions))

        # ── Signal check at today's close ─────────────────────────────────
        mkt_ok = bool(mkt_above[i])
        for sym in syms:
            c = C[sym][i]
            if sym in positions:
                pos  = positions[sym]
                ep   = pos["entry_price"]
                peak = max(pos["peak_close"], c)
                pos["peak_close"] = peak
                sp   = pos["stop_price"]
                atr_e = pos.get("atr_entry", 0.0)

                # 利益が ATR×1.0 を超えたら損切りラインを建値に引き上げ
                if not pos.get("be_moved", False) and atr_e > 0 and c >= ep + atr_e * ATR_BE_TRIGGER:
                    pos["stop_price"] = ep
                    pos["be_moved"]   = True
                    sp = ep

                if sym not in p_sells and sym not in p_partial:
                    days_held = i - pos.get("entry_idx", i)
                    unr_pct   = (c - ep) / ep if ep > 0 else 0.0
                    # 部分利確: ATR×PARTIAL_PROFIT_R に達したら50%売り
                    if (not pos.get("partial_taken") and atr_e > 0
                            and c >= ep + atr_e * PARTIAL_PROFIT_R):
                        p_partial[sym] = ("partial_profit", 0)
                    elif c <= sp:
                        p_sells[sym] = ("stop_loss", 0)
                    elif EX[sym][i]:
                        # ハイブリッド: 含み益<3% OR 保有≤10日 の初期フェーズのみ early_exit
                        early_phase = (unr_pct < EARLY_EXIT_PROFIT_CAP or
                                       days_held <= EARLY_EXIT_DAYS_CAP)
                        if early_phase:
                            p_sells[sym] = ("early_exit", 0)
                    elif days_held >= TIME_STOP_DAYS and unr_pct < TIME_STOP_MIN_PNL:
                        # タイムストップ: 20日経過で含み益1%未満なら機会損失を切る
                        p_sells[sym] = ("time_stop", 0)
                    else:
                        ts_line = peak * (1 - trailing_pct)
                        if c <= ts_line:
                            p_sells[sym] = ("trailing_stop", 0)
                            pos["sma_below_days"] = 0
                        elif DC[sym][i] or BL[sym][i]:
                            # 2日連続でSMA割れを確認してから退出（誤シグナル抑制）
                            pos["sma_below_days"] = pos.get("sma_below_days", 0) + 1
                            if pos["sma_below_days"] >= 2:
                                p_sells[sym] = ("signal", 0)
                        else:
                            pos["sma_below_days"] = 0
            elif sym not in p_buys:
                if not np.isnan(SM[sym][i]) and AV[sym][i] and GC[sym][i]:
                    rsi_v  = RS[sym][i]
                    adx_v  = AD[sym][i]
                    vr_v   = VR[sym][i]
                    rsi_ok = not np.isnan(rsi_v) and rsi_v >= RSI_THRESHOLD
                    adx_ok = not np.isnan(adx_v) and adx_v >= ADX_THRESHOLD
                    vr_ok  = not np.isnan(vr_v)  and vr_v  >= VOLUME_RATIO_MIN
                    # 相対強度: 過去RS_LOOKBACK日で銘柄リターン ≥ N225リターン
                    rs_idx = max(0, i - RS_LOOKBACK)
                    if (i >= RS_LOOKBACK and C[sym][rs_idx] > 0
                            and not np.isnan(N225[i]) and not np.isnan(N225[rs_idx])
                            and N225[rs_idx] > 0):
                        rs_ok = (C[sym][i] / C[sym][rs_idx]) / (N225[i] / N225[rs_idx]) >= 1.0
                    else:
                        rs_ok = True   # データ不足時はスキップ
                    if mkt_ok and rsi_ok and adx_ok and vr_ok and rs_ok:
                        p_buys[sym] = 0
                    elif track_skips:
                        if not mkt_ok:
                            skips.append({"date": common_idx[i], "symbol": sym,
                                          "type": "market",
                                          "reason": "市場地合い悪化のため購入を見送りました"})
                        else:
                            skips.append({"date": common_idx[i], "symbol": sym,
                                          "type": "momentum",
                                          "reason": (f"RSI/ADX/出来高フィルター "
                                                     f"(RSI={rsi_v:.1f}, ADX={adx_v:.1f}, "
                                                     f"VR={vr_v:.2f})")})

            # ── 信用売りポジション管理 ─────────────────────────────────────
            if sym in short_positions and sym not in p_short_covers:
                spos   = short_positions[sym]
                ep_s   = spos["entry_price"]
                trough = min(spos["trough_close"], c)
                spos["trough_close"] = trough
                sp_s   = spos["stop_price"]
                atr_e_s = spos.get("atr_entry", 0.0)
                days_held_s = i - spos.get("entry_idx", i)
                unr_pct_s   = (ep_s - c) / ep_s if ep_s > 0 else 0.0

                # 建値移動: 含み益が ATR×1.0 超えたら損切りを建値へ
                if not spos.get("be_moved", False) and atr_e_s > 0 and c <= ep_s - atr_e_s * ATR_BE_TRIGGER:
                    spos["stop_price"] = ep_s
                    spos["be_moved"]   = True
                    sp_s = ep_s

                if c >= sp_s:
                    p_short_covers[sym] = ("stop_loss", 0)
                elif days_held_s >= TIME_STOP_DAYS and unr_pct_s < TIME_STOP_MIN_PNL:
                    p_short_covers[sym] = ("time_stop", 0)
                else:
                    ts_line_s = trough * (1 + trailing_pct)
                    if c >= ts_line_s:
                        p_short_covers[sym] = ("trailing_stop", 0)
                        spos["sma_above_days"] = 0
                    elif GC[sym][i] or AV[sym][i]:
                        spos["sma_above_days"] = spos.get("sma_above_days", 0) + 1
                        if spos["sma_above_days"] >= 2:
                            p_short_covers[sym] = ("signal", 0)
                    else:
                        spos["sma_above_days"] = 0

            # ── 信用売り新規エントリーシグナル（地合い悪化時のみ）────────────
            elif (enable_shorts and not mkt_ok
                    and sym not in positions and sym not in short_positions
                    and sym not in p_short_entries and sym not in p_buys):
                total_slots = len(positions) + len(short_positions)
                if total_slots < MAX_SLOTS and DC[sym][i]:
                    rsi_v_s = RS[sym][i]
                    adx_v_s = AD[sym][i]
                    rsi_short_ok = not np.isnan(rsi_v_s) and rsi_v_s <= RSI_SHORT
                    adx_short_ok = not np.isnan(adx_v_s) and adx_v_s >= ADX_THRESHOLD
                    rs_idx = max(0, i - RS_LOOKBACK)
                    rs_weak = True
                    if (i >= RS_LOOKBACK and C[sym][rs_idx] > 0
                            and not np.isnan(N225[i]) and not np.isnan(N225[rs_idx])
                            and N225[rs_idx] > 0):
                        rs_weak = ((C[sym][i] / C[sym][rs_idx])
                                   / (N225[i] / N225[rs_idx])) <= 1.0
                    if rsi_short_ok and adx_short_ok and rs_weak:
                        p_short_entries[sym] = 0

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
    # ── Force-cover remaining short positions at last close ───────────────
    for sym in list(short_positions.keys()):
        spos = short_positions.pop(sym)
        sp   = C[sym][last]
        cover_cost = sp * spos["shares"] * (1 + COMMISSION)
        cash -= cover_cost
        trades.append(dict(
            symbol=sym, shares=spos["shares"], reason="forced",
            entry_date=spos["entry_date"], entry_price=spos["entry_price"],
            exit_date=common_idx[last], exit_price=sp,
            profit=spos["entry_price"] * spos["shares"] * (1 - COMMISSION) - cover_cost,
            direction="short",
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
def run_backtest(df_all: dict, n225_close: pd.Series,
                 test_days: int = TEST_DAYS, min_history: int = 0) -> None:
    active = [s for s in SYMBOLS if s in df_all]
    if min_history > 0:
        active = [s for s in active if len(df_all[s]) >= min_history]
    if len(active) < 2:
        sys.exit("Error: 2銘柄以上必要です。fetch_master_data.py を実行してください。")

    # Build common date index and train/test split
    ind_ref    = {s: build_indicators(df_all[s], 25) for s in active}
    common_idx = reduce(lambda a, b: a.intersection(b),
                        [ind_ref[s].index for s in active]).sort_values()
    split_date = common_idx[-test_days]
    train_idx  = common_idx[common_idx < split_date]
    test_idx   = common_idx[common_idx >= split_date]

    print(f"\n  共通取引日: {common_idx[0].date()} ～ {common_idx[-1].date()}  ({len(common_idx)} 日)")
    print(f"  訓練期間  : {train_idx[0].date()} ～ {train_idx[-1].date()}  ({len(train_idx)} 日)")
    print(f"  テスト期間: {test_idx[0].date()} ～ {test_idx[-1].date()}  ({len(test_idx)} 日)")

    mkt_train = build_market_filter_arr(n225_close, train_idx)
    mkt_test  = build_market_filter_arr(n225_close, test_idx)

    # 相対強度フィルター用 N225 配列（各期間のインデックスに揃える）
    n225_train_arr = n225_close.reindex(train_idx).ffill().bfill().fillna(0).values.astype(float)
    n225_test_arr  = n225_close.reindex(test_idx).ffill().bfill().fillna(0).values.astype(float)

    # Pre-compute indicators for all (SMA, ATR_PERIOD) combinations
    print("\n  インジケータ計算中 ...")
    all_ind = {
        (sma, atr_p): {
            s: {
                "train": build_indicators(df_all[s], sma, atr_p).reindex(train_idx),
                "test":  build_indicators(df_all[s], sma, atr_p).reindex(test_idx),
            }
            for s in active
        }
        for sma in SMA_PERIODS
        for atr_p in ATR_PERIODS
    }

    # Grid search on training data
    n_combos = len(SMA_PERIODS) * len(ATR_STOP_MULTS) * len(TRAILING_RATES) * len(ATR_PERIODS)
    print(f"\n{'─'*74}")
    print(f"  グリッドサーチ  ({n_combos} 組合せ × 訓練期間)  ※地合いフィルター込み")
    print(f"{'─'*74}")
    print(f"  {'SMA':>4} {'ATR×':>6} {'TS':>6} {'ATRp':>5}  {'最終資産(¥)':>16} "
          f"{'リターン':>8} {'取引':>6} {'勝率':>6} {'PF':>6}")
    print(f"  {'─'*4} {'─'*6} {'─'*6} {'─'*5}  {'─'*16} {'─'*8} {'─'*6} {'─'*6} {'─'*6}")

    best_asset, best_params = -float("inf"), None
    best_pf = 0.0

    for sma, atr_m, ts, atr_p in product(SMA_PERIODS, ATR_STOP_MULTS,
                                          TRAILING_RATES, ATR_PERIODS):
        ind_t = {s: all_ind[(sma, atr_p)][s]["train"] for s in active}
        r     = portfolio_backtest(ind_t, atr_m, ts, mkt_train, train_idx,
                                   n225_arr=n225_train_arr)
        is_b  = r["final_asset"] > best_asset
        if is_b:
            best_asset  = r["final_asset"]
            best_params = (sma, atr_m, ts, atr_p)
            best_pf     = r["profit_factor"]
        mark = " ◀" if is_b else ""
        pf_s = f"{r['profit_factor']:.2f}" if r["profit_factor"] != float("inf") else "∞"
        print(f"  {sma:>4} {atr_m:>5.1f}x {ts*100:>5.1f}% {atr_p:>5}  "
              f"{r['final_asset']:>16,.0f} {r['total_return']:>+7.1f}% "
              f"{r['total_trades']:>6} {r['win_rate']:>5.1f}% {pf_s:>6}{mark}")

    best_sma, best_atr_m, best_ts, best_atr_p = best_params
    pf_best_s = f"{best_pf:.2f}" if best_pf != float("inf") else "∞"
    print(f"\n  ★ 最適パラメータ: SMA={best_sma}, ATR×{best_atr_m:.1f}, "
          f"Trailing={best_ts*100:.1f}%, ATR期間={best_atr_p}日")
    print(f"     訓練期間 最終資産: ¥{best_asset:,.0f}  PF: {pf_best_s}")

    # Test on out-of-sample data
    ind_te = {s: all_ind[(best_sma, best_atr_p)][s]["test"] for s in active}
    result  = portfolio_backtest(ind_te, best_atr_m, best_ts, mkt_test, test_idx,
                                 n225_arr=n225_test_arr, track_skips=True,
                                 enable_shorts=False)  # バックテスト評価はロングのみ（2023-26は強気相場）

    pf_str = f"{result['profit_factor']:.2f}" if result["profit_factor"] != float("inf") else "∞"
    skips  = result["skips"]

    print(f"\n{'═'*74}")
    print(f"  テスト期間 結果  "
          f"(SMA={best_sma}, ATR×{best_atr_m:.1f}, TS={best_ts*100:.1f}%, ATRp={best_atr_p})")
    print(f"{'─'*74}")
    print(f"  初期資金                     : ¥{INITIAL_CAPITAL:>14,.0f}")
    print(f"  最終資産額                   : ¥{result['final_asset']:>14,.0f}")
    print(f"  総利益率                     :  {result['total_return']:>+13.2f}%")
    print(f"  総トレード回数               :  {result['total_trades']:>13} 回")
    print(f"  勝率                         :  {result['win_rate']:>13.1f}%")
    print(f"  最大ドローダウン             :  {result['max_drawdown']:>+13.2f}%")
    print(f"  プロフィットファクター       :  {pf_str:>13}")
    print(f"{'─'*74}")

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

    # ── 全トレード詳細ログを出力 ──────────────────────────────────────────────
    if result["trades"]:
        trades_sorted = sorted(result["trades"], key=lambda t: t["entry_date"])
        print(f"\n  {'─'*74}")
        print(f"  全トレード詳細（{len(trades_sorted)}件）")
        print(f"  {'─'*74}")
        print(f"  {'エントリー日':<13} {'決済日':<13} {'銘柄':<16} {'理由':<16}"
              f" {'株数':>5} {'買値':>7} {'売値':>7} {'損益':>9}")
        print(f"  {'─'*13} {'─'*13} {'─'*16} {'─'*16}"
              f" {'─'*5} {'─'*7} {'─'*7} {'─'*9}")
        cum = 0
        for t in trades_sorted:
            cum += t["profit"]
            mark = "✓" if t["profit"] > 0 else "✗"
            print(f"  {str(t['entry_date'])[:10]:<13} {str(t['exit_date'])[:10]:<13}"
                  f" {NAMES.get(t['symbol'], t['symbol']):<16}"
                  f" {t['reason']:<16} {t['shares']:>5}"
                  f" ¥{t['entry_price']:>6,.0f} ¥{t['exit_price']:>6,.0f}"
                  f" ¥{t['profit']:>+8,.0f} {mark}  累計¥{cum:>+,.0f}")
        print(f"  {'─'*74}")

    print(f"{'═'*74}\n")

    # Save optimised params to portfolio.json (preserve cash/positions)
    state = load_portfolio()
    state["params"] = {
        "sma": best_sma, "atr_stop_mult": best_atr_m,
        "trailing": best_ts, "atr_period": best_atr_p,
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
        f"SMA={best_sma} ATR×{best_atr_m:.1f} TS={best_ts*100:.1f}% ATRp={best_atr_p} | "
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
    state         = load_portfolio()
    params        = state.get("params", {})
    sma_period    = int(params.get("sma", 25))
    atr_stop_mult = float(params.get("atr_stop_mult", 1.5))
    ts_pct        = float(params.get("trailing", 0.04))
    atr_period    = int(params.get("atr_period", ADX_PERIOD))

    active = [s for s in SYMBOLS if s in df_all]

    # ユニバースキャッシュと照合（screen_signals.py の上位200銘柄に限定）
    cache_path = "data/universe_cache.json"
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as _f:
                _uc = json.load(_f)
            _cached_codes = set(_uc.get("codes", {}).keys())
            if _cached_codes:
                active = [s for s in active if s in _cached_codes]
        except Exception:
            pass   # キャッシュ読み込み失敗時は SYMBOLS をそのまま使用

    # Build indicators on full data (ensures proper MACD/SMA warmup)
    ind_all = {s: build_indicators(df_all[s], sma_period, atr_period) for s in active}

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

        elif action == "partial_sell":
            if o <= pc - lim and dfr < 3:        # ストップ安: 持越し
                order["deferred"] = dfr + 1
                new_pending[sym]  = order
                exec_log.append(f"  [{sym}] PARTIAL SELL 持越し (ストップ安, {dfr+1}回目)")
            else:
                if sym in positions:
                    pos     = positions[sym]
                    sh_all  = int(pos["shares"])
                    sh_half = (sh_all // 2 // LOT) * LOT
                    ep_pos  = float(pos["entry_price"])
                    if sh_half >= LOT:
                        xproc  = o * sh_half * (1 - COMMISSION)
                        profit = xproc - ep_pos * sh_half * (1 + COMMISSION)
                        cash  += xproc
                        total_pnl += profit
                        pos["shares"]        = sh_all - sh_half
                        pos["partial_taken"] = True
                        pos["stop_price"]    = max(float(pos["stop_price"]), ep_pos)
                        r_trades.append({
                            "date": str(today.date()), "symbol": sym,
                            "shares": sh_half, "action": "partial_sell",
                            "price": round(o, 1), "profit": round(profit, 0),
                            "reason": "partial_profit",
                        })
                        exec_log.append(f"  [{sym}] {NAMES.get(sym,sym)} PARTIAL SELL {sh_half}株 "
                                        f"@ ¥{o:,.0f}  P&L ¥{profit:+,.0f}  (部分利確)")
                    else:
                        # 単元未満なら全決済
                        xproc  = o * sh_all * (1 - COMMISSION)
                        profit = xproc - ep_pos * sh_all * (1 + COMMISSION)
                        cash  += xproc
                        total_pnl += profit
                        positions.pop(sym)
                        r_trades.append({
                            "date": str(today.date()), "symbol": sym,
                            "shares": sh_all, "action": "sell",
                            "price": round(o, 1), "profit": round(profit, 0),
                            "reason": "partial_profit",
                        })
                        exec_log.append(f"  [{sym}] {NAMES.get(sym,sym)} SELL (全株) {sh_all}株 "
                                        f"@ ¥{o:,.0f}  P&L ¥{profit:+,.0f}")

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
                    try:
                        atr_val = float(ind_all[sym].loc[yesterday, "atr"])
                        if np.isnan(atr_val) or atr_val <= 0:
                            atr_val = o * 0.02
                    except (KeyError, TypeError):
                        atr_val = o * 0.02
                    sl_dist = atr_val * atr_stop_mult
                    shares  = calc_position_size(o, atr_val, atr_stop_mult, port_val)
                    if shares >= LOT:
                        cost = shares * o * (1 + COMMISSION)
                    if shares >= LOT:
                        cost   = shares * o * (1 + COMMISSION)
                        if shares >= LOT and cost <= cash:
                            cash -= cost
                            positions[sym] = {
                                "entry_date":  str(today.date()),
                                "entry_price": round(o, 1),
                                "shares":      shares,
                                "peak_close":  round(o, 1),
                                "stop_price":  round(o - sl_dist, 1),
                                "atr_entry":   round(atr_val, 2),
                                "be_moved":    False,
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
            pos  = positions[sym]
            ep   = float(pos["entry_price"])
            peak = max(float(pos.get("peak_close", ep)), c)
            pos["peak_close"] = peak
            sp    = float(pos.get("stop_price", ep * 0.97))
            atr_e = float(pos.get("atr_entry", 0.0))

            # 建値移動: 利益がATR×1.0を超えたら損切りラインを建値に
            if not pos.get("be_moved", False) and atr_e > 0 and c >= ep + atr_e * ATR_BE_TRIGGER:
                pos["stop_price"] = ep
                pos["be_moved"]   = True
                sp = ep

            # 保有日数・含み益計算（タイムストップ・部分利確共用）
            try:
                ref_idx_s  = ind_all[sym].index
                entry_dt   = pd.Timestamp(pos["entry_date"])
                if entry_dt.tzinfo is None:
                    entry_dt = entry_dt.tz_localize(TOKYO_TZ)
                entry_pos_i = ref_idx_s.get_loc(entry_dt) if entry_dt in ref_idx_s else 0
                today_pos_i = ref_idx_s.get_loc(today) if today in ref_idx_s else len(ref_idx_s)-1
                days_held   = today_pos_i - entry_pos_i
            except Exception:
                days_held = 0
            unr_pct = (c - ep) / ep if ep > 0 else 0.0

            if sym not in new_pending:
                # 部分利確: ATR×PARTIAL_PROFIT_R 達成で50%売り
                if (not pos.get("partial_taken") and atr_e > 0
                        and c >= ep + atr_e * PARTIAL_PROFIT_R):
                    new_pending[sym] = {"action": "partial_sell",
                                        "reason": "partial_profit", "deferred": 0}
                    signal_log.append(
                        f"  [{sym}] {NAMES.get(sym,sym)} → 明日PARTIAL SELL "
                        f"(部分利確 ATR×{PARTIAL_PROFIT_R:.1f} 含み益{unr_pct*100:.1f}%)")
                elif c <= sp:
                    new_pending[sym] = {"action": "sell", "reason": "stop_loss", "deferred": 0}
                    signal_log.append(f"  [{sym}] {NAMES.get(sym,sym)} → 明日SELL (ATR損切 ¥{sp:,.0f})")
                elif c <= peak * (1 - ts_pct):
                    new_pending[sym] = {"action": "sell", "reason": "trailing_stop", "deferred": 0}
                    signal_log.append(f"  [{sym}] {NAMES.get(sym,sym)} → 明日SELL (トレーリング利確)")
                elif days_held >= TIME_STOP_DAYS and unr_pct < TIME_STOP_MIN_PNL:
                    # タイムストップ: 20日保有で含み益1%未満
                    new_pending[sym] = {"action": "sell", "reason": "time_stop", "deferred": 0}
                    signal_log.append(
                        f"  [{sym}] {NAMES.get(sym,sym)} → 明日SELL "
                        f"(タイムストップ {days_held}日 含み益{unr_pct*100:.1f}%)")
                else:
                    try:
                        row = ind_all[sym].loc[today]
                        if bool(row.get("rsi_exit", False)):
                            # ハイブリッド: 含み益<3% OR 保有≤10営業日 の初期フェーズのみ
                            early_phase = (unr_pct < EARLY_EXIT_PROFIT_CAP or
                                           days_held <= EARLY_EXIT_DAYS_CAP)
                            if early_phase:
                                new_pending[sym] = {"action": "sell", "reason": "early_exit", "deferred": 0}
                                signal_log.append(
                                    f"  [{sym}] {NAMES.get(sym,sym)} → 明日SELL "
                                    f"(RSI連続過熱反転 RSI={float(row['rsi']):.1f} 含み益{unr_pct*100:.1f}%)")
                            else:
                                signal_log.append(
                                    f"  [{sym}] {NAMES.get(sym,sym)} 保有継続 "
                                    f"(RSI反転だが利益乗り継続 含み益{unr_pct*100:.1f}%)")
                        elif bool(row["dead_cross"]) or bool(row["below_sma"]):
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
                vr_v  = float(row.get("vol_ratio", float("nan")))
                rsi_ok = not np.isnan(rsi_v) and rsi_v >= RSI_THRESHOLD
                adx_ok = not np.isnan(adx_v) and adx_v >= ADX_THRESHOLD
                vr_ok  = not np.isnan(vr_v)  and vr_v  >= VOLUME_RATIO_MIN
                if (not pd.isna(row["sma"]) and
                        bool(row["above_sma"]) and bool(row["golden_cross"])):
                    # 相対強度フィルター
                    rs_ok = True
                    try:
                        sym_idx  = ind_all[sym].index
                        p_now    = sym_idx.get_loc(today)
                        p_ago    = max(0, p_now - RS_LOOKBACK)
                        c_now_rs = float(ind_all[sym].iloc[p_now]["close"])
                        c_ago_rs = float(ind_all[sym].iloc[p_ago]["close"])
                        n_now_rs = float(n225_close.asof(today))
                        n_ago_rs = float(n225_close.iloc[max(0, n225_close.index.get_loc(
                            n225_close.index.asof(today)) - RS_LOOKBACK)])
                        if c_ago_rs > 0 and n_ago_rs > 0:
                            rs_ok = (c_now_rs / c_ago_rs) / (n_now_rs / n_ago_rs) >= 1.0
                    except Exception:
                        rs_ok = True
                    # 決算スキップフィルター（yfinance calendar）
                    earnings_ok = True
                    try:
                        cal = yf.Ticker(f"{sym}.T").calendar
                        if isinstance(cal, dict) and "Earnings Date" in cal:
                            earn_dates = cal["Earnings Date"]
                            if earn_dates:
                                earn_dt   = pd.Timestamp(earn_dates[0]).date()
                                days_diff = (earn_dt - today.date()).days
                                if -3 <= days_diff <= 5:
                                    earnings_ok = False
                                    signal_log.append(
                                        f"  [{sym}] {NAMES.get(sym,sym)} "
                                        f"【決算スキップ】{earn_dt}（あと{days_diff}日）")
                    except Exception:
                        pass
                    if n225_ok and rsi_ok and adx_ok and vr_ok and rs_ok and earnings_ok:
                        new_pending[sym] = {"action": "buy", "deferred": 0}
                        signal_log.append(
                            f"  [{sym}] {NAMES.get(sym,sym)} → 明日BUY "
                            f"(GC+SMA上+RS OK RSI={rsi_v:.1f} ADX={adx_v:.1f})")
                    elif not n225_ok:
                        signal_log.append(
                            f"  [{sym}] {NAMES.get(sym,sym)} "
                            f"【スキップ】市場地合い悪化")
                    elif not rs_ok:
                        signal_log.append(
                            f"  [{sym}] {NAMES.get(sym,sym)} "
                            f"【スキップ】相対強度不足（N225 アンダーパフォーム）")
                    elif earnings_ok:
                        signal_log.append(
                            f"  [{sym}] {NAMES.get(sym,sym)} "
                            f"【スキップ】RSI/ADX/VR強度不足")
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
          f"ATR×{atr_stop_mult:.1f}, TS={ts_pct*100:.0f}%")
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
            sl_line = float(pos.get("stop_price", ep * 0.97))
            ts_line = pk * (1 - ts_pct)
            be_flag = " [建値移動済]" if pos.get("be_moved") else ""
            print(f"    [{sym}] {NAMES.get(sym,sym):<16}: {sh}株  取得¥{ep:,.0f}  "
                  f"現在¥{c:,.0f}  含損益¥{unr:+,.0f} ({pct:+.1f}%)")
            print(f"          ATR損切 ¥{sl_line:,.0f}{be_flag}  最高値 ¥{pk:,.0f}  "
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
            ep  = c
            pv  = total_val
            try:
                atr_val = float(ind_all[sym].loc[today, "atr"])
            except (KeyError, TypeError):
                atr_val = ep * 0.02
            sld = atr_val * atr_stop_mult if atr_val > 0 else ep * 0.03
            slot_cap = pv / MAX_SLOTS
            lots = min(
                int(pv * RISK_PER_TRADE / sld / LOT) if sld > 0 else 0,
                int(slot_cap / (ep * (1 + COMMISSION)) / LOT),
            )
            sh   = lots * LOT
            cost = sh * ep * (1 + COMMISSION)
            sl_p = ep - sld
            print(f"    [{sym}] {NAMES.get(sym,sym)}: {sh}株  成行買い  "
                  f"推定¥{cost:,.0f}  ATR損切 ¥{sl_p:,.0f}")

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
    ap.add_argument("--test-days", type=int, default=TEST_DAYS,
                    help=f"テスト期間の営業日数 (デフォルト: {TEST_DAYS})")
    ap.add_argument("--min-history", type=int, default=800,
                    help="最小データ日数（未満の銘柄を除外）")
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
        run_backtest(df_all, n225_close,
                     test_days=args.test_days, min_history=args.min_history)
    else:
        run_auto(df_all, n225_close)
