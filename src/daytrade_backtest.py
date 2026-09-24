"""
寄り付きデイトレ(SMART v4 の日足で検証できる部分)の過去検証.

候補の条件 (前日引け時点で判定, 事前分析シートのスクリーニング条件):
  前日騰落率  : +3% 〜 +10%
  時価総額    : 100億 〜 1000億円 (前日終値 × 期末発行済株式数)
  相対出来高  : 前日出来高 ÷ その前20日平均 ≥ RELVOL_MIN
  ギャップ上限: 始値が前日終値の +5% 超なら見送り (過熱回避)

売買 (当日):
  買い        : 前日終値 ×(1+k) の指値。始値がそれ以下なら始値で約定、
                ザラ場で安値が指値に届けば指値で約定、届かなければ不成立
  売り        : 約定値 +tp% で利確 / −sl% で損切 / どちらも無ければ大引けで手仕舞い
  ※日足では高値と安値の順番がわからないため、利確と損切の両方に届いた日は「損切」とする(保守的)
  ※コスト: 往復 COST_PCT% を差し引く (手数料・スリッページの目安)

板・気配・個別材料・VIX は J-Quants に無いため検証対象外.
"""
from __future__ import annotations

import itertools
import os
from datetime import timedelta

import numpy as np
import pandas as pd

import common as c

MCAP_MIN = float((os.environ.get("MCAP_MIN_OKU") or "100")) * 1e8
MCAP_MAX = float((os.environ.get("MCAP_MAX_OKU") or "1000")) * 1e8
RELVOL_MIN = float((os.environ.get("RELVOL_MIN") or "1.0"))
GAP_MAX = float((os.environ.get("GAP_MAX_PCT") or "5")) / 100
COST = float((os.environ.get("COST_PCT") or "0.1")) / 100
YEARS = float((os.environ.get("BACKTEST_YEARS") or "2"))

LIMITS = {"前日終値": 0.0, "+0.5%": 0.005, "+1%": 0.01, "成行(始値)": np.inf}
TPS = [0.015, 0.02, 0.03]
SLS = [0.01, 0.015, 0.02]
MAIN = ("+0.5%", 0.02, 0.015)  # 表で詳しく見る代表設定


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    end = c.today() - timedelta(weeks=c.DELAY_WEEKS)
    start = c.today() - timedelta(days=int(365 * YEARS) - 3)
    raw = c.load_range("bars", start, end)
    summ = c.load_range("summary", start, end)  # Freeは2年より前を取れないため同じ期間
    if raw.empty or summ.empty:
        raise SystemExit("データが取得できませんでした")
    b = raw[["Date", "Code", "AdjO", "AdjH", "AdjL", "AdjC", "AdjVo", "C", "Va"]].copy()
    b["Date"] = pd.to_datetime(b["Date"])
    for col in b.columns[2:]:
        b[col] = pd.to_numeric(b[col], errors="coerce")
    b = b.dropna(subset=["AdjO", "AdjH", "AdjL", "AdjC"]).drop_duplicates(["Code", "Date"])
    s = summ[["Code", "DiscDate", "ShOutFY"]].copy()
    s["DiscDate"] = pd.to_datetime(s["DiscDate"], errors="coerce")
    s["ShOutFY"] = pd.to_numeric(s["ShOutFY"], errors="coerce")
    return b.sort_values(["Code", "Date"]), s.dropna().sort_values("DiscDate")


def candidates(b: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    g = b.groupby("Code")
    b["prevC"] = g["AdjC"].shift(1)
    b["prev2C"] = g["AdjC"].shift(2)
    b["prevVo"] = g["AdjVo"].shift(1)
    b["avgVo20"] = g["AdjVo"].transform(lambda x: x.shift(2).rolling(20, min_periods=15).mean())
    b["prevRawC"] = g["C"].shift(1)
    b["prevRet"] = b["prevC"] / b["prev2C"] - 1
    b["relVol"] = b["prevVo"] / b["avgVo20"]
    b["gap"] = b["AdjO"] / b["prevC"] - 1

    # 時価総額: 直近に開示された発行済株式数 × 前日終値(未調整)
    b = pd.merge_asof(b.sort_values("Date"), s.rename(columns={"DiscDate": "Date"}),
                      on="Date", by="Code", direction="backward")
    b["mcap"] = b["prevRawC"] * b["ShOutFY"]

    m = (b["prevRet"].between(0.03, 0.10) & b["mcap"].between(MCAP_MIN, MCAP_MAX)
         & (b["relVol"] >= RELVOL_MIN) & (b["gap"] <= GAP_MAX))
    return b[m].copy()


def simulate(cand: pd.DataFrame, k: float, tp: float, sl: float) -> pd.DataFrame:
    x = cand
    if np.isinf(k):
        filled = pd.Series(True, index=x.index)
        entry = x["AdjO"]
    else:
        lim = x["prevC"] * (1 + k)
        filled = x["AdjL"] <= lim
        entry = np.minimum(x["AdjO"], lim)
    tp_px, sl_px = entry * (1 + tp), entry * (1 - sl)
    hit_sl = x["AdjL"] <= sl_px
    hit_tp = x["AdjH"] >= tp_px
    exit_px = np.where(hit_sl, sl_px, np.where(hit_tp, tp_px, x["AdjC"]))
    r = exit_px / entry - 1 - COST
    out = x[["Date", "Code", "prevRet", "relVol", "gap", "mcap"]].copy()
    out["filled"] = filled
    out["result"] = np.where(hit_sl, "損切", np.where(hit_tp, "利確", "引け"))
    out["ret"] = r
    return out[out["filled"]]


def stats(t: pd.DataFrame, n_cand: int) -> dict:
    if t.empty:
        return {"候補": n_cand, "約定": 0}
    w, l = t[t["ret"] > 0]["ret"], t[t["ret"] <= 0]["ret"]
    return {
        "候補": n_cand,
        "約定": len(t),
        "約定率%": 100 * len(t) / n_cand,
        "勝率%": 100 * (t["ret"] > 0).mean(),
        "利確到達%": 100 * (t["result"] == "利確").mean(),
        "損切到達%": 100 * (t["result"] == "損切").mean(),
        "平均損益%": 100 * t["ret"].mean(),
        "平均利益%": 100 * w.mean() if len(w) else np.nan,
        "平均損失%": 100 * l.mean() if len(l) else np.nan,
        "PF": w.sum() / -l.sum() if l.sum() < 0 else np.nan,
    }


def main() -> None:
    b, s = load()
    cand = candidates(b, s)
    n = len(cand)
    print(f"候補(のべ): {n}件 / {cand['Date'].nunique()}日")
    if n == 0:
        raise SystemExit("条件に合う候補が0件でした")

    rows = []
    for (name, k), tp, sl in itertools.product(LIMITS.items(), TPS, SLS):
        st = stats(simulate(cand, k, tp, sl), n)
        rows.append({"指値": name, "利確幅%": tp * 100, "損切幅%": sl * 100, **st})
    grid = pd.DataFrame(rows).round(2)

    c.RESULT_DIR.mkdir(exist_ok=True)
    grid.to_csv(c.RESULT_DIR / "daytrade_grid.csv", index=False, encoding="utf-8-sig")
    main_t = simulate(cand, LIMITS[MAIN[0]], MAIN[1], MAIN[2])
    main_t.to_csv(c.RESULT_DIR / "daytrade_trades.csv", index=False, encoding="utf-8-sig")

    # 代表設定を条件別に分解
    def by(col, bins, labels):
        g = main_t.assign(区分=pd.cut(main_t[col], bins, labels=labels))
        return pd.DataFrame({k: stats(v, len(v)) for k, v in g.groupby("区分", observed=True)}).T

    tbl_rv = by("relVol", [0, 1.5, 3, np.inf], ["1〜1.5倍", "1.5〜3倍", "3倍以上"]).round(2)
    tbl_ret = by("prevRet", [0.03, 0.05, 0.07, 0.10], ["+3〜5%", "+5〜7%", "+7〜10%"]).round(2)
    tbl_gap = by("gap", [-1, -0.01, 0.01, 0.05], ["GD(−1%未満)", "±1%", "GU(+1〜5%)"]).round(2)
    for t in (tbl_rv, tbl_ret, tbl_gap):
        t.drop(columns=["候補", "約定率%"], errors="ignore", inplace=True)

    best = grid[grid["約定"] >= 100].sort_values("平均損益%", ascending=False).head(5)
    md = (
        f"## 寄り付きデイトレ 過去検証 (のべ候補 {n}件 / {cand['Date'].nunique()}日)\n"
        f"条件: 前日+3〜10%, 時価総額{MCAP_MIN/1e8:.0f}〜{MCAP_MAX/1e8:.0f}億, 相対出来高≥{RELVOL_MIN}, "
        f"ギャップ≤+{GAP_MAX*100:.0f}% / コスト往復{COST*100:.2f}% / 利確・損切両方到達日は損切扱い\n\n"
        f"### 全設定\n{grid.to_markdown(index=False)}\n\n"
        f"### 平均損益の上位5設定 (約定100件以上)\n{best.to_markdown(index=False)}\n\n"
        f"### 代表設定 (指値{MAIN[0]} / 利確{MAIN[1]*100:.1f}% / 損切{MAIN[2]*100:.1f}%) の内訳\n"
        f"相対出来高別\n{tbl_rv.to_markdown()}\n\n前日騰落率別\n{tbl_ret.to_markdown()}\n\n"
        f"寄りのギャップ別\n{tbl_gap.to_markdown()}\n\n"
        "見方: 平均損益%がコスト控除後でプラスか、PF(総利益÷総損失)が1を超えるかが最低ライン。"
        "上位設定だけを採用すると過剰最適化になりやすいので、周辺の設定でも同じ傾向か確認する。"
    )
    c.write_step_summary(md)
    top = best.head(3)
    c.post_slack(
        f"*寄り付きデイトレ 過去検証* のべ{n}件\n"
        + "\n".join(f"指値{r['指値']} 利確{r['利確幅%']}% 損切{r['損切幅%']}%: 約定{int(r['約定'])} "
                    f"勝率{r['勝率%']:.0f}% 平均{r['平均損益%']:+.2f}% PF{r['PF']:.2f}"
                    for _, r in top.iterrows())
    )


if __name__ == "__main__":
    main()
