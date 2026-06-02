"""
BTC エアトレード履歴ビューア
使い方: python3 btc_history.py

btc_portfolio.json と btc_log/ の記録をまとめて表示。
取引一覧・累積損益曲線・週次サマリーを出力。
"""

import json
import os
import glob
from datetime import datetime, date
import numpy as np

PORTFOLIO_FILE = "btc_portfolio.json"
LOG_DIR        = "btc_log"
INITIAL        = 500_000

def load():
    if not os.path.exists(PORTFOLIO_FILE):
        print("btc_portfolio.json が見つかりません。btc_paper_trade.py を先に実行してください。")
        return None
    with open(PORTFOLIO_FILE) as f:
        return json.load(f)

def print_history(pf):
    trades = pf.get("trades", [])
    if not trades:
        print("取引履歴なし（運用開始待ち）")
        return

    W = 70
    print("\n" + "═"*W)
    print("  BTC エアトレード 取引履歴")
    print("  開始日:", pf.get("created_at", "不明")[:10])
    print("  最終更新:", pf.get("last_updated", "不明")[:10])
    print("═"*W)

    # ── 取引一覧
    print("\n【取引一覧】")
    print(f"  {'#':>3}  {'日付':>10}  {'種別':>10}  {'価格':>10}  {'数量':>8}  {'損益':>10}  {'理由'}")
    print("  " + "─"*66)

    capital = float(INITIAL)
    equity_curve = [capital]
    closed_trades = []

    for i, t in enumerate(trades, 1):
        action = t["action"]
        price  = t.get("price", 0)
        qty    = t.get("qty", 0)
        pnl    = t.get("pnl", None)
        date_  = t.get("date", "")
        reason = t.get("reason", "")[:20]

        pnl_s = f"{pnl:>+9,.0f}円" if pnl is not None else "          "
        icon  = "🟢" if action == "BUY" else ("🔴" if "SELL" in action else "🟡")

        print(f"  {i:>3}  {date_:>10}  {icon}{action:>9}  "
              f"${price:>9,.0f}  {qty:>8.4f}  {pnl_s}  {reason}")

        if pnl is not None:
            capital += pnl
            equity_curve.append(capital)
            closed_trades.append(pnl)

    # ── 統計
    current_equity = pf["cash"] + (
        pf["position"]["qty"] * 0  # 評価額は別途
    )

    print(f"\n【パフォーマンスサマリー】")
    print(f"  初期資金   : {INITIAL:>12,}円")
    print(f"  現在の現金 : {pf['cash']:>12,.0f}円")

    if closed_trades:
        wins  = [p for p in closed_trades if p > 0]
        loss  = [p for p in closed_trades if p < 0]
        total_n = len(closed_trades)
        win_n   = len(wins)
        wr      = win_n / total_n * 100

        realized = sum(closed_trades)
        pf_ratio = sum(wins) / abs(sum(loss)) if loss else float("inf")

        print(f"  実現損益   : {realized:>+12,.0f}円")
        print(f"  取引回数   : {total_n:>12}回")
        print(f"  勝率       : {wr:>11.1f}%")
        print(f"  PF         : {pf_ratio:>12.2f}")
        if wins:
            print(f"  平均利益   : {np.mean(wins):>+12,.0f}円/回")
        if loss:
            print(f"  平均損失   : {np.mean(loss):>+12,.0f}円/回")

        # 最大連続損失
        consec = cur = 0
        for p in closed_trades:
            if p < 0: cur += 1; consec = max(consec, cur)
            else: cur = 0
        print(f"  最大連続負け: {consec:>11}回")

    # ── 累積損益のテキストグラフ
    if len(equity_curve) > 1:
        print(f"\n【累積損益曲線（概略）】")
        eq = np.array(equity_curve)
        mn, mx = eq.min(), eq.max()
        span   = mx - mn if mx > mn else 1
        H = 6  # 高さ

        # 正規化してグラフ化
        cols = min(len(eq), 50)
        if len(eq) > cols:
            idx = np.linspace(0, len(eq)-1, cols, dtype=int)
            eq_plot = eq[idx]
        else:
            eq_plot = eq

        rows = []
        for row in range(H, 0, -1):
            line = ""
            threshold = mn + (row / H) * span
            ref_label = f"{threshold/10000:>5.0f}万" if row in (1, H//2+1, H) else "     "
            for val in eq_plot:
                if val >= threshold:
                    line += "█"
                else:
                    line += "·"
            rows.append(f"  {ref_label} |{line}")

        for r in rows:
            print(r)
        print(f"  {'':5} +" + "─"*cols)
        start_d = trades[0]["date"] if trades else ""
        end_d   = trades[-1]["date"] if trades else ""
        print(f"  {' '*7}{start_d}{' '*(cols-len(start_d)-len(end_d))}{end_d}")

    # ── 現在のポジション
    pos = pf["position"]
    if pos["qty"] > 0:
        print(f"\n【現在の保有ポジション】")
        print(f"  BTC数量   : {pos['qty']:.6f} BTC")
        print(f"  取得価格  : ${pos['entry_price']:,.0f}  ({pos['entry_date']})")
        print(f"  ストップ  : ${pos['stop_price']:,.0f}")
        print(f"  部分利食い: {'済' if pos['tp_done'] else '未'}")
    else:
        print(f"\n  現在ポジションなし（シグナル待機中）")

    # ── ログファイル一覧
    logs = sorted(glob.glob(os.path.join(LOG_DIR, "*.txt")))
    logs = [l for l in logs if "dashboard" not in l]
    if logs:
        print(f"\n【日次ログ一覧】 ({len(logs)}日分)")
        for log in logs[-10:]:  # 最新10日
            fname = os.path.basename(log)
            size  = os.path.getsize(log)
            print(f"  {fname}  ({size}bytes)")
        if len(logs) > 10:
            print(f"  ... 他 {len(logs)-10}日分")

    print("═"*W)
    print()

if __name__ == "__main__":
    pf = load()
    if pf:
        print_history(pf)
