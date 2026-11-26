import os
import json
import math
import xml.etree.ElementTree as ET
from urllib.parse import quote
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone

BASE_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMD", "AVGO", "PLTR", "ARM", "SMCI", "INTC", "MU",
    "QCOM", "CRM", "NOW", "ADBE", "ORCL", "SNOW", "PANW", "CRWD", "DDOG", "MDB",
    "AMZN", "GOOGL", "META", "TSLA", "NFLX", "UBER", "ABNB", "SHOP", "SPOT", "BKNG",
    "COST", "WMT", "HD", "TGT", "NKE", "SBUX", "MCD", "DIS", "ROKU", "TTD",
    "COIN", "MSTR", "MARA", "HOOD", "SOFI", "JPM", "GS", "BAC", "V", "MA",
    "PYPL", "SQ", "AFRM", "UPST", "RIVN", "LCID", "NIO", "LI", "XPEV", "F",
    "GM", "LLY", "UNH", "JNJ", "XOM", "CVX", "OXY", "SLB", "BA", "CAT", "DE", "GE"
]

BATCH_SIZE = 18
DEFAULT_PROFIT_TARGET_PCT = 0.06
MIN_SCORE_TO_ALERT = 0.65
DEFAULT_INITIAL_CAPITAL = 10000.0
MAX_OPEN_POSITIONS = 6
DEMO_FILE = "demo_portfolio.json"
LATEST_SCAN_FILE = "latest_scan.json"
WATCHLIST_FILE = "watchlist.json"

POS_NEWS_WORDS = (
    "beat", "beats", "surge", "surges", "soar", "soars", "upgrade", "upgrades",
    "bullish", "record", "profit", "profits", "growth", "partnership", "deal",
    "buyback", "outperform", "raises guidance", "strong", "gain", "gains", "rally"
)
NEG_NEWS_WORDS = (
    "miss", "misses", "drop", "drops", "plunge", "plunges", "downgrade", "downgrades",
    "bearish", "loss", "losses", "lawsuit", "probe", "investigation", "sec",
    "dilution", "offering", "cut guidance", "weak", "warn", "warns", "fall", "falls", "recall"
)


def get_profit_target_pct() -> float:
    raw = os.getenv("INPUT_PROFIT_TARGET_PCT", "").strip()
    if not raw:
        return DEFAULT_PROFIT_TARGET_PCT
    try:
        val = float(raw)
        if 0.5 <= val <= 100.0:
            return val / 100.0
    except ValueError:
        pass
    return DEFAULT_PROFIT_TARGET_PCT


def get_initial_capital() -> float:
    raw = os.getenv("DEMO_INITIAL_CAPITAL", "").strip()
    if raw:
        try:
            v = float(raw)
            if v >= 100:
                return v
        except ValueError:
            pass
    return DEFAULT_INITIAL_CAPITAL


def get_watchlist_tickers() -> list[str]:
    env_list = os.getenv("WATCHLIST_TICKERS", "").strip()
    if env_list:
        return list(dict.fromkeys([t.strip().upper() for t in env_list.split(",") if t.strip()]))

    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return list(dict.fromkeys([str(t).strip().upper() for t in data if str(t).strip()]))
                if isinstance(data, dict) and isinstance(data.get("tickers"), list):
                    return list(dict.fromkeys([str(t).strip().upper() for t in data["tickers"] if str(t).strip()]))
        except Exception as exc:
            print(f"Error reading {WATCHLIST_FILE}: {exc}")

    return []


def get_scan_tickers(watchlist_tickers: list[str]) -> tuple[list[str], bool]:
    manual = os.getenv("INPUT_TICKERS", "").strip()
    if manual:
        tickers = list(dict.fromkeys([t.strip().upper() for t in manual.split(",") if t.strip()]))
        return tickers, True

    extra_raw = os.getenv("EXTRA_TICKERS", "").strip()
    extra_tickers = [t.strip().upper() for t in extra_raw.split(",") if t.strip()]

    pool = list(dict.fromkeys(BASE_UNIVERSE))
    slot_index = int(datetime.now(timezone.utc).timestamp() // 1800)
    total = len(pool)
    start_idx = (slot_index * BATCH_SIZE) % total

    if start_idx + BATCH_SIZE <= total:
        batch = pool[start_idx:start_idx + BATCH_SIZE]
    else:
        wrap = (start_idx + BATCH_SIZE) - total
        batch = pool[start_idx:] + pool[:wrap]

    combined = list(dict.fromkeys(watchlist_tickers + extra_tickers + batch))
    return combined, False


def fetch_stock_news(ticker: str) -> dict:
    headlines = []
    seen = set()

    # 1. Yahoo Finance RSS Feed
    try:
        rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote(ticker)}&region=US&lang=en-US"
        resp = requests.get(rss_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        if resp.status_code == 200 and resp.text:
            root = ET.fromstring(resp.text)
            for item in root.findall(".//item")[:6]:
                title_el = item.find("title")
                link_el = item.find("link")
                pub_el = item.find("pubDate")
                title = (title_el.text or "").strip() if title_el is not None else ""
                link = (link_el.text or "").strip() if link_el is not None else ""
                pub = (pub_el.text or "").strip() if pub_el is not None else ""
                if title and title.lower() not in seen:
                    seen.add(title.lower())
                    headlines.append({"title": title, "link": link, "published": pub})
    except Exception:
        pass

    # 2. yfinance .news fallback
    if len(headlines) < 3:
        try:
            raw_news = yf.Ticker(ticker).news or []
            for n in raw_news[:6]:
                content = n.get("content") if isinstance(n, dict) and isinstance(n.get("content"), dict) else n
                title = str((content or {}).get("title") or "").strip()
                link = str(((content or {}).get("canonicalUrl") or {}).get("url") or (content or {}).get("link") or "").strip()
                pub = str((content or {}).get("pubDate") or "").strip()
                if title and title.lower() not in seen:
                    seen.add(title.lower())
                    headlines.append({"title": title, "link": link, "published": pub})
        except Exception:
            pass

    pos_hits = 0
    neg_hits = 0
    for h in headlines:
        low_t = h["title"].lower()
        p_c = sum(1 for w in POS_NEWS_WORDS if w in low_t)
        n_c = sum(1 for w in NEG_NEWS_WORDS if w in low_t)
        pos_hits += p_c
        neg_hits += n_c
        if p_c > n_c:
            h["tone"] = "POSITIVE"
        elif n_c > p_c:
            h["tone"] = "NEGATIVE"
        else:
            h["tone"] = "NEUTRAL"

    if neg_hits > pos_hits + 1:
        overall_tone = "NEGATIVE"
        summary_he = f"ידיעות שליליות ({neg_hits} מול {pos_hits})"
    elif pos_hits > neg_hits + 1:
        overall_tone = "POSITIVE"
        summary_he = f"ידיעות חיוביות ({pos_hits} מול {neg_hits})"
    elif headlines:
        overall_tone = "MIXED"
        summary_he = f"ידיעות מעורבות ({len(headlines)} כותרות)"
    else:
        overall_tone = "NONE"
        summary_he = "אין ידיעות חדשות ברשת כרגע"

    return {
        "count": len(headlines),
        "tone": overall_tone,
        "summaryHe": summary_he,
        "posHits": pos_hits,
        "negHits": neg_hits,
        "items": headlines[:4],
    }


def send_telegram_message(text: str) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        print("Telegram credentials missing; skipping push.")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code == 200:
            print("Telegram message sent.")
            return True
        print(f"Telegram API error ({resp.status_code}): {resp.text}")
        return False
    except Exception as exc:
        print(f"Telegram error: {exc}")
        return False


def fetch_market_bundle(ticker: str) -> tuple[pd.DataFrame | None, dict, dict, dict]:
    news_data = fetch_stock_news(ticker)
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None, {}, {}, news_data
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).copy()
        if len(df) < 30:
            return None, {}, {}, news_data

        info = {}
        try:
            raw_info = yf.Ticker(ticker).info or {}
            if isinstance(raw_info, dict):
                info = raw_info
        except Exception:
            pass

        bench = {}
        for sym, key in [("SPY", "spy"), ("^VIX", "vix")]:
            try:
                bdf = yf.download(sym, period="3mo", interval="1d", progress=False, auto_adjust=True)
                if bdf is not None and not bdf.empty and len(bdf) >= 22:
                    if isinstance(bdf.columns, pd.MultiIndex):
                        bdf.columns = bdf.columns.get_level_values(0)
                    c = bdf["Close"].dropna()
                    if len(c) >= 22:
                        last_v = float(c.iloc[-1])
                        prev_v = float(c.iloc[-2])
                        ret20_v = (last_v - float(c.iloc[-21])) / float(c.iloc[-21])
                        sma20_v = float(c.rolling(20).mean().iloc[-1])
                        bench[key] = {
                            "last": last_v,
                            "chg_pct": ((last_v - prev_v) / max(prev_v, 1e-6)) * 100.0,
                            "ret_20d": ret20_v,
                            "above_sma20": last_v > sma20_v,
                        }
            except Exception:
                pass

        return df, info, bench, news_data
    except Exception as exc:
        print(f"Error fetching {ticker}: {exc}")
        return None, {}, {}, news_data


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def compute_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> tuple[float, float]:
    low_min = df["Low"].rolling(k_period).min()
    high_max = df["High"].rolling(k_period).max()
    denom = (high_max - low_min).replace(0, np.nan)
    k_series = ((df["Close"] - low_min) / denom) * 100.0
    d_series = k_series.rolling(d_period).mean()
    k_val = float(k_series.iloc[-1]) if not math.isnan(float(k_series.iloc[-1])) else 50.0
    d_val = float(d_series.iloc[-1]) if not math.isnan(float(d_series.iloc[-1])) else 50.0
    return k_val, d_val


def compute_adx(df: pd.DataFrame, period: int = 14) -> float:
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"] - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean().replace(0, np.nan)
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).rolling(period).mean() / atr
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr
    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx_series = dx.rolling(period).mean()
    val = float(adx_series.iloc[-1])
    return val if not math.isnan(val) else 20.0


def compute_obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["Close"].diff().fillna(0.0))
    return (sign * df["Volume"]).cumsum()


def compute_ichimoku(df: pd.DataFrame) -> dict:
    tenkan = (df["High"].rolling(9).max() + df["Low"].rolling(9).min()) / 2.0
    kijun = (df["High"].rolling(26).max() + df["Low"].rolling(26).min()) / 2.0
    span_a = ((tenkan + kijun) / 2.0).shift(26)
    span_b = ((df["High"].rolling(52).max() + df["Low"].rolling(52).min()) / 2.0).shift(26)

    a_val = float(span_a.iloc[-1]) if not math.isnan(float(span_a.iloc[-1])) else float(df["Close"].iloc[-1])
    b_val = float(span_b.iloc[-1]) if not math.isnan(float(span_b.iloc[-1])) else float(df["Close"].iloc[-1])
    cloud_top = max(a_val, b_val)
    return {"cloud_top": cloud_top, "above_cloud": float(df["Close"].iloc[-1]) > cloud_top}


def run_66_checks_before_entry_exit(df: pd.DataFrame, info: dict, bench: dict, news_data: dict, target_pct: float) -> dict:
    close = df["Close"]
    open_p = df["Open"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    last_open = float(open_p.iloc[-1])
    prev_open = float(open_p.iloc[-2])
    last_high = float(high.iloc[-1])
    prev_high = float(high.iloc[-2])
    last_low = float(low.iloc[-1])

    sma_20 = float(close.rolling(20).mean().iloc[-1])
    sma_50 = float(close.rolling(min(50, len(close))).mean().iloc[-1])
    sma_200 = float(close.rolling(min(200, len(close))).mean().iloc[-1])
    ema_20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    ema_50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])

    prev_sma50 = float(close.rolling(min(50, len(close) - 1)).mean().iloc[-2])
    prev_sma200 = float(close.rolling(min(200, len(close) - 1)).mean().iloc[-2])

    rsi_s = compute_rsi(close, 14)
    rsi = float(rsi_s.iloc[-1]) if not math.isnan(float(rsi_s.iloc[-1])) else 50.0
    prev_rsi = float(rsi_s.iloc[-2]) if not math.isnan(float(rsi_s.iloc[-2])) else 50.0

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    macd_hist = float((macd - sig).iloc[-1])
    prev_macd_hist = float((macd - sig).iloc[-2])

    stoch_k, stoch_d = compute_stochastic(df)

    std20 = float(close.rolling(20).std().iloc[-1])
    if math.isnan(std20) or std20 <= 0:
        std20 = last_close * 0.02
    bb_upper = sma_20 + 2.0 * std20
    bb_lower = max(0.01, sma_20 - 2.0 * std20)

    atr = float(compute_atr(df, 14).iloc[-1])
    if math.isnan(atr) or atr <= 0:
        atr = last_close * 0.02

    adx = compute_adx(df, 14)

    high_10 = float(high.rolling(10).max().iloc[-2])
    low_10 = float(low.rolling(10).min().iloc[-2])
    high_20 = float(high.rolling(20).max().iloc[-2])
    low_20 = float(low.rolling(20).min().iloc[-2])
    high_60 = float(high.rolling(min(60, len(high))).max().iloc[-2])
    low_60 = float(low.rolling(min(60, len(low))).min().iloc[-2])
    swing_range = max(high_60 - low_60, 1e-6)
    fib_382 = high_60 - 0.382 * swing_range
    fib_618 = high_60 - 0.618 * swing_range

    vol_avg20 = float(volume.rolling(20).mean().iloc[-1])
