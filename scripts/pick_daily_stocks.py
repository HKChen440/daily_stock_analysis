#!/usr/bin/env python3
"""每日动态选股 v3：从候选池按动量挑 12 只（A股 4 / 港股 3 / 美股 5）

v3 改动（相对 v2）:
- 不再用 yfinance 批量下载（本机/CI 经常被 Yahoo 429 全灭，导致静默退化成 FALLBACK 固定股）
- 改为直接调 Yahoo chart API：串行 + 退避重试 + query1/query2 轮换 + 当日磁盘缓存
- A 股额外兜底：Yahoo 拿不到时用 Tushare（TUSHARE_TOKEN，或 ~/.openclaw/secrets/tushare.token）
- 每个市场的失败原因会打印出来，不再无声降级

用法:
  python scripts/pick_daily_stocks.py [--no-cache] [--debug]
输出:
  stdout 最后一行 STOCK_LIST=...（同时写入 $GITHUB_ENV，Actions 里可直接用）
"""
import datetime
import json
import os
import sys
import time
import urllib.request

import pandas as pd

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

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"
)
YF_HOSTS = ("query1", "query2")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".cache", "pick")
USE_CACHE = "--no-cache" not in sys.argv
DEBUG = "--debug" in sys.argv


def log(msg):
    print(msg, file=sys.stderr, flush=True)


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
    d = closes.diff().dropna().values
    gain = [x if x > 0 else 0.0 for x in d]
    loss = [-x if x < 0 else 0.0 for x in d]
    ag, al = sum(gain[:period]) / period, sum(loss[:period]) / period
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
    rsi = rsi_wilder(close)
    if rsi is None:
        return None, "rsi"
    volr = float(vol.values[-5:].mean() / max(vol.values[-20:].mean(), 1))
    rets = pd.Series(c[-20:]).pct_change().dropna()
    volat = float(rets.std() * 100)

    if px <= ma50:
        return None, "below_ma50"
    if rsi >= 78:
        return None, "rsi_hot"
    if volat > 15:
        return None, "volatile"

    sweet = max(0.0, 1 - abs(rsi - 62) / 25)
    score = 0.45 * mom20 + 0.25 * mom60 + 0.15 * (volr - 1) * 20 + 0.15 * sweet * 20
    return {
        "score": score, "rsi": rsi, "mom20": mom20, "mom60": mom60,
        "volr": volr, "px": px, "volat": volat,
    }, "ok"


# ---------------------------------------------------------------- 数据抓取

def _get_json(url, timeout=25):
    req = urllib.request.Request(
        url, headers={"User-Agent": UA, "Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def yahoo_fetch(ticker):
    """直接拉 Yahoo chart API，返回 Close/Volume DataFrame（升序）。"""
    last_err = None
    for attempt in range(4):
        host = YF_HOSTS[attempt % 2]
        url = (
            f"https://{host}.finance.yahoo.com/v8/finance/chart/{ticker}"
            "?range=6mo&interval=1d&events=div%2Csplits"
        )
        try:
            j = _get_json(url)
            res = ((j.get("chart") or {}).get("result") or [None])[0]
            if not res:
                raise ValueError("empty result")
            quote = ((res.get("indicators") or {}).get("quote") or [{}])[0]
            closes = quote.get("close") or []
            vols = quote.get("volume") or []
            ts = res.get("timestamp") or []
            n = min(len(closes), len(ts))
            df = pd.DataFrame(
                {"Close": closes[:n], "Volume": vols[:n] if vols else [0] * n},
                index=pd.to_datetime(ts[:n], unit="s"),
            ).dropna(subset=["Close"])
            if len(df) < 70:
                raise ValueError(f"bars={len(df)}")
            return df
        except Exception as e:  # noqa: BLE001
            last_err = e
            if DEBUG:
                log(f"[pick] {ticker} attempt {attempt + 1} via {host} failed: {e}")
            time.sleep(0.7 * (attempt + 1))
    raise last_err if last_err else RuntimeError("unknown")


def _tushare_token():
    tok = os.environ.get("TUSHARE_TOKEN", "").strip()
    if tok:
        return tok
    path = os.path.expanduser("~/.openclaw/secrets/tushare.token")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def tushare_fetch(ticker):
    """A 股兜底：Tushare daily -> Close/Volume DataFrame。"""
    token = _tushare_token()
    if not token:
        raise RuntimeError("no TUSHARE_TOKEN")
    code = ticker.replace(".SS", ".SH").replace(".SZ", ".SZ")
    end = datetime.date.today()
    start = end - datetime.timedelta(days=300)
    payload = {
        "api_name": "daily",
        "token": token,
        "params": {
            "ts_code": code,
            "start_date": start.strftime("%Y%m%d"),
            "end_date": end.strftime("%Y%m%d"),
        },
        "fields": "trade_date,close,vol",
    }
    req = urllib.request.Request(
        "http://api.tushare.pro",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": UA},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        j = json.loads(resp.read().decode("utf-8", "replace"))
    if j.get("code") != 0:
        raise RuntimeError(f"tushare code={j.get('code')} msg={j.get('msg')}")
    items = ((j.get("data") or {}).get("items") or [])
    if not items:
        raise RuntimeError("tushare empty")
    rows = sorted(items)  # trade_date 升序
    df = pd.DataFrame(
        {
            "Close": [float(r[1]) for r in rows],
            "Volume": [float(r[2]) * 100 for r in rows],
        },
        index=pd.to_datetime([str(r[0]) for r in rows], format="%Y%m%d"),
    )
    if len(df) < 70:
        raise ValueError(f"bars={len(df)}")
    return df


def _cache_file(ticker):
    day = datetime.date.today().strftime("%Y%m%d")
    name = ticker.replace(".", "_") + f"_{day}.csv"
    return os.path.join(CACHE_DIR, name)


def fetch_one(ticker, market):
    cf = _cache_file(ticker)
    if USE_CACHE and os.path.exists(cf):
        try:
            df = pd.read_csv(cf, index_col=0, parse_dates=True)
            if len(df) >= 70:
                return df, "cache"
        except Exception:  # noqa: BLE001
            pass

    errors = []
    sources = [("yahoo", yahoo_fetch)]
    if market == "cn":
        sources.append(("tushare", tushare_fetch))
    for name, fn in sources:
        try:
            df = fn(ticker)
            os.makedirs(CACHE_DIR, exist_ok=True)
            df.to_csv(cf)
            return df, name
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}:{type(e).__name__}:{str(e)[:60]}")

    # 港股代码在某些源需要 4 位补零
    if market == "hk" and not ticker[0].isdigit():
        pass
    raise RuntimeError(" | ".join(errors))


def fetch(pool, market):
    out, srcs = {}, {}
    for t in pool:
        try:
            df, src = fetch_one(t, market)
            out[t] = df
            srcs[src] = srcs.get(src, 0) + 1
        except Exception as e:  # noqa: BLE001
            log(f"[pick] {t} FAILED: {e}")
        time.sleep(0.35)  # 对 Yahoo 客气点
    log(f"[pick] {market}: sources={json.dumps(srcs)}")
    return out


def main():
    picked, detail = [], {}
    used_fallback = {}
    for market, pool in UNIVERSE.items():
        data = fetch(pool, market)
        rows, reasons = [], {}
        for t, df in data.items():
            try:
                s, why = score_ticker(df)
            except Exception as e:  # noqa: BLE001
                s, why = None, f"err:{type(e).__name__}:{str(e)[:70]}"
            reasons[why] = reasons.get(why, 0) + 1
            if s:
                rows.append((t, s))
        rows.sort(key=lambda x: -x[1]["score"])
        take = rows[: NEED[market]]
        log(
            f"[pick] {market}: universe={len(pool)} downloaded={len(data)} pass={len(rows)} "
            f"reasons={json.dumps(reasons)} picked={[t for t, _ in take]}"
        )
        if len(take) < NEED[market]:
            have = {to_repo_symbol(t, market) for t, _ in take}
            added = []
            for f in FALLBACK[market]:
                if len(take) >= NEED[market]:
                    break
                if f not in have:
                    take.append((f, {"score": 0, "fallback": True}))
                    added.append(f)
            if added:
                used_fallback[market] = added
                log(f"[pick] {market}: 数量不足，用兜底补 {added}")

        picked.extend(to_repo_symbol(t, market) for t, _ in take)
        detail[market] = [
            {
                "symbol": to_repo_symbol(t, market),
                "score": round(float(s.get("score", 0)), 2),
                "rsi": round(float(s["rsi"]), 1) if "rsi" in s else None,
                "mom20": round(float(s["mom20"]), 1) if "mom20" in s else None,
                "fallback": bool(s.get("fallback")),
            }
            for t, s in take
        ]

    line = ",".join(picked)
    print(f"STOCK_LIST={line}")
    log("[pick] detail=" + json.dumps(detail, ensure_ascii=False))
    if used_fallback:
        log(f"[pick] ⚠️ 有市场靠兜底补齐: {json.dumps(used_fallback, ensure_ascii=False)}")

    gh_env = os.environ.get("GITHUB_ENV")
    if gh_env:
        with open(gh_env, "a", encoding="utf-8") as fh:
            fh.write(f"STOCK_LIST={line}\n")


if __name__ == "__main__":
    main()
