"""
毎朝のスクリーニング: 四季報「会社比強気」リスト × 進捗率 × 決算日 × 流動性 で候補を絞り、Slackに投稿.

shikiho/shikiho_list.csv に銘柄があればそれを母集団に、空なら全銘柄を進捗率だけで絞る.
"""
from __future__ import annotations

import os
from datetime import timedelta

import pandas as pd

import common as c
import signals as sg

MIN_VALUE = float((os.environ.get("MIN_TRADING_VALUE") or "50000000"))
MAX_DAYS_TO_EARNINGS = int((os.environ.get("MAX_DAYS_TO_EARNINGS") or "28"))
TOP_N = int((os.environ.get("TOP_N") or "15"))
MARK_BONUS = {"大幅強気": 10.0, "会社比強気": 5.0}


def load_shikiho() -> pd.DataFrame:
    p = "shikiho/shikiho_list.csv"
    if not os.path.exists(p):
        return pd.DataFrame(columns=["Code", "Mark"])
    df = pd.read_csv(p, dtype=str).dropna(subset=["Code"])
    df["Code"] = df["Code"].map(c.norm_code)
    df["Mark"] = df.get("Mark", "").fillna("")
    return df[["Code", "Mark"]].drop_duplicates("Code")


def main() -> None:
    t = c.today()
    end = t - timedelta(weeks=c.DELAY_WEEKS)
    summ = c.prep_summary(c.load_range("summary", t - timedelta(days=400), end))
    bars = c.prep_bars(c.load_range("bars", end - timedelta(days=45), end))
    if summ.empty or bars.empty:
        raise SystemExit("データが取得できませんでした")

    st = sg.add_progress(sg.statements(summ))
    latest = st.sort_values("DiscDate").groupby("Code").tail(1)
    latest = latest[latest["CurPerType"].isin(list(sg.BASE)) & latest["progress"].notna()]
    latest = latest[latest["DiscDate"] >= pd.Timestamp(end - timedelta(days=120))]

    shiki = load_shikiho()
    use_list = not shiki.empty
    df = latest.merge(shiki, on="Code", how="inner" if use_list else "left")
    df["Mark"] = df["Mark"].fillna("")

    # 流動性と直近20日の上昇(織り込み度合いの目安)
    liq = bars.sort_values("Date").groupby("Code").tail(20).groupby("Code")["Va"].median()
    last = bars.sort_values("Date").groupby("Code")["AdjC"].agg(lambda x: x.iloc[-1] / x.iloc[-21] - 1
                                                                if len(x) > 20 else None)
    df["売買代金(百万)"] = df["Code"].map(liq) / 1e6
    df["20日騰落%"] = 100 * pd.to_numeric(df["Code"].map(last), errors="coerce")
    df = df[df["売買代金(百万)"] * 1e6 >= MIN_VALUE]

    # 決算発表予定日 (取得できないプランでは使わない)
    cal = c.throttled(c.client().get_eq_earnings_cal)
    has_cal = cal is not None and not cal.empty and {"Code", "Date"} <= set(cal.columns)
    if has_cal:
        cal = cal.assign(Code=cal["Code"].astype(str)).drop_duplicates("Code", keep="last")
        df["決算予定日"] = df["Code"].map(cal.set_index("Code")["Date"])
        df["決算まで日数"] = (pd.to_datetime(df["決算予定日"]) - pd.Timestamp(t)).dt.days
        df = df[df["決算まで日数"].between(1, MAX_DAYS_TO_EARNINGS)]

    master = c.throttled(c.client().get_eq_master)
    if master is not None and not master.empty:
        df["銘柄名"] = df["Code"].map(master.drop_duplicates("Code").set_index("Code")["CoName"])

    df["スコア"] = df["excess_pt"] + df["Mark"].map(MARK_BONUS).fillna(0)
    df = df.sort_values("スコア", ascending=False)

    cols = ["Code", "銘柄名", "Mark", "CurPerType", "progress", "excess_pt", "excess_basis",
            "売買代金(百万)", "20日騰落%", "スコア"] + (["決算予定日", "決算まで日数"] if has_cal else [])
    out = df[[x for x in cols if x in df.columns]].rename(columns={
        "Mark": "四季報", "CurPerType": "直近決算", "progress": "進捗率",
        "excess_pt": "超過pt", "excess_basis": "基準"})
    out["進捗率"] = (100 * out["進捗率"]).round(1)
    out = out.round(1)
    c.RESULT_DIR.mkdir(exist_ok=True)
    out.to_csv(c.RESULT_DIR / f"screen_{t:%Y%m%d}.csv", index=False, encoding="utf-8-sig")

    top = out.head(TOP_N)
    note = []
    if c.DELAY_WEEKS:
        note.append(f"⚠ データは{c.DELAY_WEEKS}週遅延(Freeプラン)。実運用は有料プランで DATA_DELAY_WEEKS=0 に")
    if not use_list:
        note.append("四季報リスト未登録のため全銘柄を進捗率のみで評価")
    if not has_cal:
        note.append("決算予定日が取得できないため日付フィルタなし")

    c.write_step_summary(f"## スクリーニング {t}\n" + "\n".join(f"- {n}" for n in note)
                         + f"\n\n{top.to_markdown(index=False)}")
    lines = []
    for _, r in top.iterrows():
        s = (f"{r['Code'][:4]} {r.get('銘柄名', '')} {r['四季報']} 進捗{r['進捗率']}% "
             f"({r['基準']}{r['超過pt']:+.0f}pt) 20日{r['20日騰落%']:+.0f}%")
        if has_cal:
            s += f" 決算{r['決算まで日数']:.0f}日後"
        lines.append(s)
    c.post_slack(f"*決算前スクリーニング {t}* 上位{len(top)}件\n" + "\n".join(lines)
                 + ("\n" + "\n".join(note) if note else ""))


if __name__ == "__main__":
    main()
