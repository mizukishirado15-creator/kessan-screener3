"""進捗率シグナルの計算 (四季報「会社比強気」の代わりに使う、決算短信ベースの指標)."""
from __future__ import annotations

import pandas as pd

BASE = {"1Q": 0.25, "2Q": 0.50, "3Q": 0.75}


def statements(s: pd.DataFrame) -> pd.DataFrame:
    """決算短信本体の行だけ (業績修正の開示などは除く)."""
    m = s["DocType"].str.contains("FinancialStatements") & s["CurPerType"].isin(
        ["1Q", "2Q", "3Q", "FY"]
    )
    return s[m].reset_index(drop=True)


def add_progress(st: pd.DataFrame) -> pd.DataFrame:
    """
    progress   : 累計営業利益 ÷ 会社の通期営業利益予想
    hist_prog  : 前年同四半期の進捗率 (前年の実績営業利益に対して)
    excess_pt  : 進捗率が「いつもの年」または「単純な按分(25/50/75%)」をどれだけ上回っているか [%pt]
    """
    st = st.copy()
    q = st["CurPerType"].isin(list(BASE))
    ok = q & (st["FOP"] > 0) & st["OP"].notna()
    st["progress"] = (st["OP"] / st["FOP"]).where(ok)

    fy_actual = (
        st[st["CurPerType"] == "FY"]
        .dropna(subset=["CurFYSt"])
        .groupby(["Code", "CurFYSt"])["OP"].last()
    )
    same_q = (
        st[q].dropna(subset=["CurFYSt"])
        .groupby(["Code", "CurPerType", "CurFYSt"])["OP"].last()
    )

    hist = []
    for r in st.itertuples():
        h = None
        if r.CurPerType in BASE and pd.notna(r.CurFYSt):
            prev = r.CurFYSt - pd.DateOffset(years=1)
            op_q = same_q.get((r.Code, r.CurPerType, prev))
            op_fy = fy_actual.get((r.Code, prev))
            if op_q is not None and op_fy is not None and op_fy > 0:
                h = op_q / op_fy
        hist.append(h)
    st["hist_prog"] = pd.to_numeric(pd.Series(hist, index=st.index), errors="coerce")
    base = st["CurPerType"].map(BASE)
    st["excess_pt"] = 100 * (st["progress"] - st["hist_prog"].fillna(base))
    st["excess_basis"] = st["hist_prog"].notna().map({True: "前年同期比", False: "単純按分比"})
    return st


def bin_label(x: float) -> str:
    if pd.isna(x):
        return "NA"
    if x < -10:
        return "① -10pt未満"
    if x < 0:
        return "② -10〜0pt"
    if x < 10:
        return "③ 0〜+10pt"
    if x < 20:
        return "④ +10〜+20pt"
    return "⑤ +20pt以上"
