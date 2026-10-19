#!/usr/bin/env python3
"""每日动态选股 v5：纯美股，从候选池按动量挑 8 只。

v5 改动（相对 v4）：
- 美股池从 43 只扩到 ~120 只（纳指 100 主体 + 高流动性大中盘），仍按动量挑 8 只

v4 改动（相对 v3）：
- 只保留美股池，不再覆盖 A 股 / 港股
- NEED = {"us": 8}；兜底列表也只保留美股
- 移除 Tushare / A 股 / 港股相关分叉，目录仅剩美股代码

v3 保留的能力：
- 不依赖 yfinance 批量下载（CI 上经常被 Yahoo 429 全灭）
- 直接调 Yahoo chart API：串行 + 退避重试 + query1/query2 轮换 + 当日磁盘缓存
- 失败原因逐个打印，不再无声降级

用法:
  python scripts/pick_daily_stocks.py [--no-cache] [--debug]
输出:
  stdout 最后一行 STOCK_LIST=...；同时写入 $GITHUB_ENV 与 ./pick_stock_list.txt
  （Actions 里后续步骤可直接读文件，避免被其他 STOCK_LIST 配置覆盖）
"""
import datetime
import json
import os
import sys
import time
import urllib.request

import pandas as pd

# 纯美股池：高流动性大中盘（~120 只，纳指 100 为主体）
UNIVERSE = {
    "us": [
        # 大型科技 / 互联网
        "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "AMZN", "META", "TSLA",
        "AVGO", "AMD", "NFLX", "CRM", "ORCL", "ADBE", "NOW", "INTU",
        "CSCO", "IBM", "DELL", "ANET",
        # 半导体 / 硬件
        "QCOM", "TXN", "AMAT", "LRCX", "MU", "ADI", "KLAC", "MRVL",
        "INTC", "ASML", "ARM", "SMCI",
        # 软件 / 平台 / 新经济
        "PLTR", "PANW", "CRWD", "ZS", "FTNT", "DDOG", "SNOW", "NET",
        "SHOP", "MELI", "UBER", "ABNB", "BKNG", "PYPL", "COIN", "HOOD",
        "APP", "ADSK", "WDAY", "TEAM",
        # 通讯
        "TMUS", "CMCSA", "VZ", "T",
        # 医疗
        "LLY", "UNH", "ISRG", "TMO", "AMGN", "GILD", "VRTX", "REGN",
        "ABBV", "JNJ", "PFE", "MRK", "BMY", "ABT", "DHR", "SYK",
        "MDT", "ELV", "CI", "DXCM", "IDXX",
        # 金融
        "JPM", "V", "MA", "GS", "MS", "BAC", "WFC", "C", "SCHW",
        "AXP", "BLK", "SPGI", "CME", "ICE",
        # 消费
        "COST", "WMT", "KO", "PEP", "MCD", "DIS", "SBUX", "NKE",
        "MDLZ", "PM", "MO", "TGT", "LOW", "HD", "LULU", "MAR", "ORLY",
        # 工业 / 能源 / 材料
        "XOM", "CVX", "COP", "SLB", "CAT", "DE", "BA", "GE", "RTX",
        "LMT", "NOC", "GD", "UNP", "UPS", "FDX", "ETN", "HON", "LIN", "FCX",
        # 公用 / REIT
        "NEE", "DUK", "SO", "AMT", "PLD",
    ],
}
# 去重并保持顺序（防止手工维护时重复代码导致重复请求）
UNIVERSE["us"] = list(dict.fromkeys(UNIVERSE["us"]))

NEED = {"us": 8}

# 兜底：全部为美股代码（与 NEED 数量对齐，保证报告只含美股）
FALLBACK = {
    "us": ["AAPL", "NVDA", "MSFT", "TSLA", "AMZN", "META", "GOOGL", "AVGO"],
}

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"
)
YF_HOSTS = ("query1", "query2")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".cache", "pick")
LIST_FILE = os.path.join(os.getcwd(), "pick_stock_list.txt")
USE_CACHE = "--no-cache" not in sys.argv
DEBUG = "--debug" in sys.argv


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def to_repo_symbol(ticker: str) -> str:
    """美股代码即仓库格式（裸 ticker，如 AAPL）。"""
    return ticker.strip().upper()


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
    """直接打 Yahoo chart API，返回 Close/Volume DataFrame（升序）。"""
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
    for name, fn in (("yahoo", yahoo_fetch),):
        try:
            df = fn(ticker)
            os.makedirs(CACHE_DIR, exist_ok=True)
            df.to_csv(cf)
            return df, name
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}:{type(e).__name__}:{str(e)[:60]}")
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
            have = {to_repo_symbol(t) for t, _ in take}
            added = []
            for f in FALLBACK[market]:
                if len(take) >= NEED[market]:
                    break
                if f not in have:
                    take.append((f, {"score": 0, "fallback": True}))
                    have.add(f)
                    added.append(f)
            if added:
                used_fallback[market] = added
                log(f"[pick] {market}: 数量不足，用兜底补 {len(added)} 只: {added}")

        picked.extend(to_repo_symbol(t) for t, _ in take)
        detail[market] = [
            {
                "symbol": to_repo_symbol(t),
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
        log(f"[pick] WARN 部分标的来自兜底列表: {json.dumps(used_fallback, ensure_ascii=False)}")

    gh_env = os.environ.get("GITHUB_ENV")
    if gh_env:
        with open(gh_env, "a", encoding="utf-8") as fh:
            fh.write(f"STOCK_LIST={line}\n")

    # 供 workflow 后续步骤直接读取（优先级最高，避免被 vars/secrets 覆盖）
    try:
        with open(LIST_FILE, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")
        log(f"[pick] wrote {LIST_FILE}")
    except OSError as e:  # noqa: BLE001
        log(f"[pick] WARN 无法写入 {LIST_FILE}: {e}")


if __name__ == "__main__":
    main()
