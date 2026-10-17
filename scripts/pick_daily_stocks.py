#!/usr/bin/env python3
"""每日动态选股 v2：从候选池按动量挑 12 只（A4/H3/US5）"""
import json
import os
import sys

import numpy as np
import yfinance as yf

UNIVERSE = {
    "cn": [
        "600519.SS", "000858.SZ", "601318.SS", "600036.SS", "601398.SS", "601288.SS",
        "600030.SS", "601899.SS", "600887.SS", "601012.SS", "600276.SS", "600309.SS",
        "601088.SS", "600028.SS", "601668.SS", "601857.SS", "600900.SS", "601166.SS",
        "000333.SZ", "000651.SZ", "002594.SZ", "300750.SZ", "300059.SZ", "002475.SZ",
        "002415.SZ", "000725.SZ", "002714.SZ", "300760.SZ", "300124.SZ", "002352.SZ",
    ],
    "hk": [
        "0700.HK", "9988.HK", "3690.HK", "1810.HK", "9618.HK", "9999.HK", "1024.HK",
        "2015.HK", "1211.HK", "0981.HK", "2318.HK", "0939.HK", "1398.HK", "3988.HK",
        "0883.HK", "0857.HK", "0386.HK", "2628.HK", "2388.HK", "0016.HK", "0001.HK",
        "0027.HK", "1928.HK", "0291.HK", "1109.HK", "0688.HK",
    ],
    "us": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "AMD", "NFLX",
        "CRM", "ORCL", "ADBE", "COST", "WMT", "JPM", "V", "MA", "UNH", "LLY",
        "XOM", "CVX", "KO", "PEP", "MCD",
    ],
}

NEED = {"cn": 4, "hk": 3, "us": 5}

# 兜底已经是仓库格式（hk 前缀只在 hk 市场加）
FALLBACK = {
    "cn": ["600519", "300750", "002594", "601318"],
    "hk": ["hk00700", "hk09988", "hk01810"],
    "us": ["AAPL", "NVDA", "MSFT", "TSLA", "AMZN"],
}


def to_repo_symbol(ticker: str, market: str) -> str:
    if market == "cn":
        return ticker.split(".")[0]
    if market == "hk":
        if ticker.lower().startswith("hk"):
            return ticker
        num = ticker.split(".")[0]
        return "hk" + num.zfill(4)
    return ticker


def rsi_wilder(closes, period=14):
    if len(closes) < period + 1:
        return None
    d = np.diff(closes)
    gain = np.where(d > 0, d, 0.0)
    loss = np.where(d < 0, -d, 0.0)
    ag, al = gain[:period].mean(), loss[:period].mean()
    for i in range(period, len(d)):
        ag = (ag * (period - 1) + gain[i]) / period
        al = (al * (period - 1) + loss[i]) / period
    if al == 0:
        return 100.0
    return 100 - 100 / (1 + ag / al)


def _series(frame, col):
    s = frame[col]
    if hasattr(s, "columns"):  # 2D 情况取第一列
        s = s.iloc[:, 0]
    return s.dropna()


def score_ticker(df):
    close = _series(df, "Close")
    vol = _series(df, "Volume")
    if len(close) < 70 or len(vol) < 25:
        return None, "bars"
    c = close.values.astype(float)
    px = float(c[-1])
    ma50 = float(c[-50:].mean())
    mom20 = (px / float(c[-21]) - 1) * 100
    mom60 = (px / float(c[-61]) - 1) * 100
    rsi = rsi_wilder(c)
    if rsi is None:
        return None, "rsi"
    volr = float(vol.values[-5:].mean() / max(vol.values[-20:].mean(), 1))
    rets = np.diff(c[-20:]) / c[-21:-1]
    volat = float(np.std(rets) * 100)

    if px <= ma50:
        return None, "below_ma50"
    if rsi >= 78:
        return None, "rsi_hot"
    if volat > 15:
        return None, "volatile"

    sweet = max(0.0, 1 - abs(rsi - 62) / 25)
    score = 0.45 * mom20 + 0.25 * mom60 + 0.15 * (volr - 1) * 20 + 0.15 * sweet * 20
    return {"score": score, "rsi": rsi, "mom20": mom20, "mom60": mom60, "volr": volr, "px": px}, "ok"


def fetch(pool):
    try:
        df = yf.download(
            tickers=" ".join(pool), period="6mo", interval="1d",
            group_by="ticker", auto_adjust=True, threads=True, progress=False,
        )
    except Exception as e:
        print(f"[pick] download exception: {e}", file=sys.stderr)
        return {}
    out = {}
    for t in pool:
        sub = None
        try:
            if hasattr(df.columns, "levels"):
                if t in df.columns.get_level_values(0):
                    sub = df[t]
                elif t in df.columns.get_level_values(1):
                    sub = df.xs(t, axis=1, level=1)
            else:
                sub = df
        except Exception:
            sub = None
        if sub is None or "Close" not in getattr(sub, "columns", []):
            continue
        if sub["Close"].dropna().empty:
            continue
        out[t] = sub
    return out


def main():
    picked, detail = [], {}
    for market, pool in UNIVERSE.items():
        data = fetch(pool)
        rows, reasons = [], {}
        for t, df in data.items():
            try:
                s, why = score_ticker(df)
            except Exception as e:
                s, why = None, f"err:{type(e).__name__}:{str(e)[:70]}"
            reasons[why] = reasons.get(why, 0) + 1
            if s:
                rows.append((t, s))
        rows.sort(key=lambda x: -x[1]["score"])
        take = rows[: NEED[market]]
        print(
            f"[pick] {market}: universe={len(pool)} downloaded={len(data)} pass={len(rows)} "
            f"reasons={json.dumps(reasons)} picked={[t for t, _ in take]}",
            file=sys.stderr,
        )
        if len(take) < NEED[market]:
            have = {to_repo_symbol(t, market) for t, _ in take}
            for f in FALLBACK[market]:
                if len(take) >= NEED[market]:
                    break
                if f not in have:
                    take.append((f, {"score": 0, "fallback": True}))
        picked.extend(to_repo_symbol(t, market) for t, _ in take)
        detail[market] = [
            {"symbol": to_repo_symbol(t, market), "score": round(float(s.get("score", 0)), 2),
             "rsi": round(float(s["rsi"]), 1) if "rsi" in s else None,
             "mom20": round(float(s["mom20"]), 1) if "mom20" in s else None}
            for t, s in take
        ]

    line = ",".join(picked)
    print(f"STOCK_LIST={line}")
    print("[pick] detail=" + json.dumps(detail, ensure_ascii=False), file=sys.stderr)

    gh_env = os.environ.get("GITHUB_ENV")
    if gh_env:
        with open(gh_env, "a", encoding="utf-8") as fh:
            fh.write(f"STOCK_LIST={line}\n")


if __name__ == "__main__":
    main()
