"""
過去検証: 「進捗率が高い銘柄は、次の決算で上振れし株価も上がるのか」を測る.

各四半期決算(1Q〜3Q)を起点に、次の決算までを1イベントとして集計する.
  上振れ率      : 次の決算までに会社が営業利益予想を上方修正した(or 本決算で予想超え)割合
  決算日勝率    : 次の決算発表の反応日に、市場(全銘柄の中央値)より上がった割合
  窓A           : 次の決算の5営業日前の引けで買い → 反応日の引けで売り
  窓B           : 起点決算の反応日の翌日引けで買い → 次の決算の反応日の引けで売り
※ リターンはすべて「市場中央値との差(超過リターン)」, 手数料・スリッページは含まない
"""
from __future__ import annotations

import os
from datetime import timedelta

import pandas as pd

import common as c
import signals as sg

MIN_VALUE = float((os.environ.get("MIN_TRADING_VALUE") or "50000000"))  # 20日売買代金中央値の下限(円)
YEARS = float((os.environ.get("BACKTEST_YEARS") or "2"))


def main() -> None:
    end = c.today() - timedelta(weeks=c.DELAY_WEEKS)
    start = c.today() - timedelta(days=int(365 * YEARS) - 3)
    print(f"期間: {start} 〜 {end}")

    summ = c.prep_summary(c.load_range("summary", start, end))
    bars = c.prep_bars(c.load_range("bars", start, end))
    if summ.empty or bars.empty:
        raise SystemExit("データが取得できませんでした。APIキーとプランを確認してください。")

    master = c.throttled(c.client().get_eq_master)
    scale = (master.drop_duplicates("Code").set_index("Code")["ScaleCat"]
             if master is not None and not master.empty else pd.Series(dtype=str))

    # 価格テーブル
    px = bars.pivot(index="Date", columns="Code", values="AdjC").sort_index()
    va = bars.pivot(index="Date", columns="Code", values="Va").sort_index()
    tdays = px.index
    mkt = px.pct_change(fill_method=None).median(axis=1)  # 市場代理: 全銘柄の日次リターン中央値
    mkt_lvl = (1 + mkt.fillna(0)).cumprod()

    st = sg.add_progress(sg.statements(summ))
    fop_rows = summ[summ["FOP"].notna()][["Code", "DiscDate", "CurFYSt", "FOP"]]
    fop_by_code = {k: v for k, v in fop_rows.groupby("Code")}

    def ret(code, d0, d1):
        if code not in px.columns or d0 is None or d1 is None:
            return None
        p0, p1 = px.at[d0, code], px.at[d1, code]
        if pd.isna(p0) or pd.isna(p1) or p0 <= 0:
            return None
        return (p1 / p0 - 1) - (mkt_lvl[d1] / mkt_lvl[d0] - 1)

    def shift(d, n):
        i = tdays.get_loc(d) + n
        return tdays[i] if 0 <= i < len(tdays) else None

    rows = []
    for code, g in st.groupby("Code"):
        g = g.reset_index(drop=True)
        for i in range(len(g) - 1):
            a, b = g.loc[i], g.loc[i + 1]
            if a["CurPerType"] not in sg.BASE or pd.isna(a["excess_pt"]):
                continue
            if (b["DiscDate"] - a["DiscDate"]).days > 200:
                continue
            ea = c.effective_day(a["DiscDate"], a["DiscTime"], tdays)
            eb = c.effective_day(b["DiscDate"], b["DiscTime"], tdays)
            if ea is None or eb is None or eb <= ea:
                continue
            # 流動性フィルタ
            if code not in va.columns:
                continue
            liq = va[code].loc[:ea].tail(20).median()
            if pd.isna(liq) or liq < MIN_VALUE:
                continue
            # 上振れ判定
            fr = fop_by_code.get(code, fop_rows.iloc[:0])
            win = fr[
                (fr["DiscDate"] > a["DiscDate"])
                & (fr["DiscDate"] <= b["DiscDate"])
                & (fr["CurFYSt"] == a["CurFYSt"])
            ]
            up = bool((win["FOP"] > a["FOP"] * 1.01).any())
            if b["CurPerType"] == "FY" and b["CurFYSt"] == a["CurFYSt"]:
                up = up or bool(b["OP"] > a["FOP"] * 1.01)
            pre_b = shift(eb, -1)
            rows.append({
                "Code": code,
                "ScaleCat": scale.get(code, ""),
                "起点開示日": a["DiscDate"].date(),
                "四半期": a["CurPerType"],
                "進捗率": round(a["progress"] * 100, 1),
                "超過pt": round(a["excess_pt"], 1),
                "基準": a["excess_basis"],
                "区分": sg.bin_label(a["excess_pt"]),
                "次決算日": b["DiscDate"].date(),
                "上振れ": up,
                "決算日超過": ret(code, pre_b, eb),
                "窓A超過": ret(code, shift(eb, -5), eb),
                "窓B超過": ret(code, shift(ea, 1), eb),
            })

    ev = pd.DataFrame(rows)
    c.RESULT_DIR.mkdir(exist_ok=True)
    ev.to_csv(c.RESULT_DIR / "backtest_events.csv", index=False, encoding="utf-8-sig")
    if ev.empty:
        raise SystemExit("イベントが0件でした")

    ev["小型"] = ~ev["ScaleCat"].fillna("").str.contains("Core30|Large70|Mid400")

    def table(df: pd.DataFrame) -> pd.DataFrame:
        def agg(x):
            return pd.Series({
                "件数": len(x),
                "上振れ率%": 100 * x["上振れ"].mean(),
                "決算日勝率%": 100 * (x["決算日超過"].dropna() > 0).mean(),
                "決算日平均%": 100 * x["決算日超過"].mean(),
                "窓A勝率%": 100 * (x["窓A超過"].dropna() > 0).mean(),
                "窓A平均%": 100 * x["窓A超過"].mean(),
                "窓B勝率%": 100 * (x["窓B超過"].dropna() > 0).mean(),
                "窓B平均%": 100 * x["窓B超過"].mean(),
            })
        t = df.groupby("区分").apply(agg, include_groups=False)
        t.loc["全体"] = agg(df)
        t["件数"] = t["件数"].astype(int)
        return t.round(1)

    t_all, t_small = table(ev), table(ev[ev["小型"]])
    t_all.to_csv(c.RESULT_DIR / "backtest_summary_all.csv", encoding="utf-8-sig")
    t_small.to_csv(c.RESULT_DIR / "backtest_summary_small.csv", encoding="utf-8-sig")

    md = (
        f"## 進捗率シグナルの過去検証 ({start}〜{end})\n"
        f"流動性: 20日売買代金中央値 ≥ {MIN_VALUE/1e6:.0f}百万円 / リターンは市場中央値との差\n\n"
        f"### 全銘柄\n{t_all.to_markdown()}\n\n### 小型株のみ (TOPIX Mid400以上を除く)\n"
        f"{t_small.to_markdown()}\n\n"
        "見方: ④⑤(進捗が良い)の行が「全体」より勝率・平均とも高ければ、シグナルに意味がある。"
        "件数が少ない行(目安50件未満)は偶然の影響が大きい。"
    )
    c.write_step_summary(md)

    def line(t, label):
        r = t.loc[label] if label in t.index else None
        if r is None:
            return f"{label}: データなし"
        return (f"{label}: {int(r['件数'])}件 / 上振れ{r['上振れ率%']:.0f}% / "
                f"決算日勝率{r['決算日勝率%']:.0f}% 平均{r['決算日平均%']:+.2f}% / "
                f"窓A勝率{r['窓A勝率%']:.0f}% 平均{r['窓A平均%']:+.2f}%")

    c.post_slack(
        f"*進捗率シグナル 過去検証* ({start}〜{end})\n"
        + "\n".join(line(t_all, k) for k in ["全体", "④ +10〜+20pt", "⑤ +20pt以上"])
        + "\n小型株のみ\n"
        + "\n".join(line(t_small, k) for k in ["全体", "④ +10〜+20pt", "⑤ +20pt以上"])
        + "\n(詳細は GitHub Actions の実行結果ページ)"
    )


if __name__ == "__main__":
    main()
