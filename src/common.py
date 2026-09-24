"""共通処理: J-Quants取得(レート制限・キャッシュ付き)、数値変換、Slack投稿."""
from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
import jquantsapi

CACHE_DIR = Path(os.environ.get("CACHE_DIR", "data/cache"))
RESULT_DIR = Path("results")
# Free=5, Light=60, Standard=120, Premium=500 (リクエスト/分)
DELAY_WEEKS = int(os.environ.get("DATA_DELAY_WEEKS") or "12")  # Free=12, 有料=0
RATE_PER_MIN = int(os.environ.get("JQUANTS_RATE_PER_MIN") or "5")
_MIN_INTERVAL = 60.0 / RATE_PER_MIN * 1.1
_last_call = 0.0
_cli = None

NUM_COLS = ["Sales", "OP", "OdP", "NP", "FSales", "FOP", "FOdP", "FNP"]
JST = timezone(timedelta(hours=9))


def today() -> date:
    return datetime.now(JST).date()


def client() -> jquantsapi.ClientV2:
    global _cli
    if _cli is None:
        if not os.environ.get("JQUANTS_API_KEY"):
            raise SystemExit("JQUANTS_API_KEY が設定されていません (GitHub Secrets を確認)")
        _cli = jquantsapi.ClientV2()
    return _cli


def throttled(fn, *args, **kwargs):
    """レート制限を守って呼ぶ。失敗時は待って最大3回リトライ。取得できなければ None。"""
    global _last_call
    for attempt in range(3):
        wait = _MIN_INTERVAL - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "403" in msg or "400" in msg:  # プラン範囲外など。リトライしても無駄
                print(f"  skip ({msg[:100]})")
                return None
            print(f"  retry {attempt + 1}/3 after error: {msg[:120]}")
            time.sleep(70)
    return None


def weekdays(start: date, end: date):
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


def _fetch(kind: str, d: date) -> pd.DataFrame:
    if kind == "bars":
        return client().get_eq_bars_daily(date_yyyymmdd=f"{d:%Y-%m-%d}")
    if kind == "summary":
        return client().get_fin_summary(date_yyyymmdd=f"{d:%Y%m%d}")
    raise ValueError(kind)


def load_range(kind: str, start: date, end: date) -> pd.DataFrame:
    """日次データを1日ずつ取得し data/cache に保存 (2回目以降はキャッシュから)."""
    days = list(weekdays(start, end))
    frames = []
    for i, d in enumerate(days, 1):
        path = CACHE_DIR / kind / f"{d:%Y%m%d}.csv.gz"
        if path.exists():
            if path.stat().st_size > 0:
                frames.append(pd.read_csv(path, dtype=str))
            continue
        print(f"[{kind}] {d} ({i}/{len(days)})", flush=True)
        df = throttled(_fetch, kind, d)
        if df is None:  # エラー(プラン範囲外など)はキャッシュしない
            continue
        # 直近分(プランの遅延期間+1週間)は後からデータが増えうるのでキャッシュしない
        if (today() - d).days > DELAY_WEEKS * 7 + 7:
            path.parent.mkdir(parents=True, exist_ok=True)
            if df.empty:
                path.write_bytes(b"")  # 祝日などは空ファイルで記録
            else:
                df.to_csv(path, index=False)
        if not df.empty:
            frames.append(df.astype(str))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def prep_bars(bars: pd.DataFrame) -> pd.DataFrame:
    bars = bars[["Date", "Code", "AdjC", "Va"]].copy()
    bars["Date"] = pd.to_datetime(bars["Date"])
    for c in ["AdjC", "Va"]:
        bars[c] = pd.to_numeric(bars[c], errors="coerce")
    return bars.dropna(subset=["AdjC"]).drop_duplicates(["Code", "Date"])


def prep_summary(s: pd.DataFrame) -> pd.DataFrame:
    s = s.copy()
    for c in ["DiscDate", "CurFYSt", "CurFYEn", "CurPerEn"]:
        if c in s.columns:
            s[c] = pd.to_datetime(s[c], errors="coerce")
    for c in NUM_COLS:
        if c in s.columns:
            s[c] = pd.to_numeric(s[c], errors="coerce")
    s["DiscTime"] = s["DiscTime"].fillna("").astype(str) if "DiscTime" in s else ""
    s["DocType"] = s["DocType"].fillna("").astype(str)
    s["CurPerType"] = s["CurPerType"].fillna("").astype(str)
    return s.sort_values(["Code", "DiscDate", "DiscTime"]).reset_index(drop=True)


def effective_day(disc_date, disc_time: str, tdays: pd.DatetimeIndex):
    """開示が株価に反映される最初の営業日 (15:30以降の開示は翌営業日)."""
    after_close = disc_time >= "15:30"
    idx = tdays.searchsorted(disc_date, side="right" if after_close else "left")
    return tdays[idx] if idx < len(tdays) else None


def norm_code(code) -> str:
    code = str(code).strip().upper()
    return code + "0" if len(code) == 4 else code


def post_slack(text: str) -> None:
    url = os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        print("(SLACK_WEBHOOK_URL 未設定のため Slack 投稿はスキップ)")
        return
    r = requests.post(url, json={"text": text}, timeout=30)
    print("Slack:", r.status_code)


def write_step_summary(md: str) -> None:
    """GitHub Actions の実行結果ページに表示する."""
    p = os.environ.get("GITHUB_STEP_SUMMARY")
    if p:
        with open(p, "a", encoding="utf-8") as f:
            f.write(md + "\n")
    print(md)
