import os
import json
import math
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
MIN_SCORE_TO_ALERT = 0.68
DEMO_FILE = "demo_portfolio.json"
LATEST_SCAN_FILE = "latest_scan.json"
WATCHLIST_FILE = "watchlist.json"


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


def fetch_stock_data(ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, period="3mo", interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).copy()
        if len(df) < 25:
            return None
        return df
    except Exception as exc:
        print(f"Error fetching {ticker}: {exc}")
        return None


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


def compute_indicators_pack(df: pd.DataFrame) -> dict:
    close = df["Close"]
    open_p = df["Open"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    prev_high = float(high.iloc[-2])
    prev_low = float(low.iloc[-2])
    last_open = float(open_p.iloc[-1])
    last_high = float(high.iloc[-1])
    last_low = float(low.iloc[-1])

    sma_5 = float(close.rolling(5).mean().iloc[-1])
    sma_10 = float(close.rolling(10).mean().iloc[-1])
    sma_20 = float(close.rolling(20).mean().iloc[-1])
    sma_50 = float(close.rolling(min(50, len(close))).mean().iloc[-1])
    ema_9 = float(close.ewm(span=9, adjust=False).mean().iloc[-1])
    ema_21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])

    std_20 = float(close.rolling(20).std().iloc[-1])
    if math.isnan(std_20) or std_20 <= 0:
        std_20 = last_close * 0.02
    bb_upper = sma_20 + (2.0 * std_20)
    bb_lower = max(0.01, sma_20 - (2.0 * std_20))

    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = (2.0 * pivot) - prev_low
    s1 = max(0.01, (2.0 * pivot) - prev_high)

    rsi_series = compute_rsi(close, 14)
    rsi = float(rsi_series.iloc[-1]) if not math.isnan(float(rsi_series.iloc[-1])) else 50.0
    prev_rsi = float(rsi_series.iloc[-2]) if not math.isnan(float(rsi_series.iloc[-2])) else 50.0

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_val = float(macd.iloc[-1])
    signal_val = float(signal.iloc[-1])
    macd_hist = float((macd - signal).iloc[-1])
    prev_macd_hist = float((macd - signal).iloc[-2])

    atr_series = compute_atr(df, 14)
    atr = float(atr_series.iloc[-1]) if not math.isnan(float(atr_series.iloc[-1])) else (last_close * 0.02)

    vol_avg_20 = float(volume.rolling(20).mean().iloc[-1])
    last_vol = float(volume.iloc[-1])
    vol_ratio = (last_vol / vol_avg_20) if vol_avg_20 > 0 else 1.0

    high_10 = float(high.rolling(10).max().iloc[-2])
    low_10 = float(low.rolling(10).min().iloc[-2])
    high_20 = float(high.rolling(20).max().iloc[-2])
    low_20 = float(low.rolling(20).min().iloc[-2])
    high_52 = float(high.max())
    low_52 = float(low.min())

    ret_1d = (last_close - prev_close) / prev_close if prev_close else 0.0
    ret_3d = (last_close - float(close.iloc[-4])) / float(close.iloc[-4]) if len(close) >= 4 else 0.0
    ret_5d = (last_close - float(close.iloc[-6])) / float(close.iloc[-6]) if len(close) >= 6 else 0.0
    ret_10d = (last_close - float(close.iloc[-11])) / float(close.iloc[-11]) if len(close) >= 11 else 0.0
    ret_20d = (last_close - float(close.iloc[-21])) / float(close.iloc[-21]) if len(close) >= 21 else 0.0

    candle_body = abs(last_close - last_open)
    candle_range = max(last_high - last_low, 1e-6)
    body_ratio = candle_body / candle_range

    return {
        "last_close": last_close,
        "prev_close": prev_close,
        "last_open": last_open,
        "last_high": last_high,
        "last_low": last_low,
        "sma_5": sma_5,
        "sma_10": sma_10,
        "sma_20": sma_20,
        "sma_50": sma_50,
        "ema_9": ema_9,
        "ema_21": ema_21,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "pivot": pivot,
        "r1": r1,
        "s1": s1,
        "rsi": rsi,
        "prev_rsi": prev_rsi,
        "macd_val": macd_val,
        "signal_val": signal_val,
        "macd_hist": macd_hist,
        "prev_macd_hist": prev_macd_hist,
        "atr": atr,
        "vol_ratio": vol_ratio,
        "vol_avg_20": vol_avg_20,
        "high_10": high_10,
        "low_10": low_10,
        "high_20": high_20,
        "low_20": low_20,
        "high_52": high_52,
        "low_52": low_52,
        "ret_1d": ret_1d,
        "ret_3d": ret_3d,
        "ret_5d": ret_5d,
        "ret_10d": ret_10d,
        "ret_20d": ret_20d,
        "body_ratio": body_ratio,
        "candle_range": candle_range,
        "close_series": close,
        "volume_series": volume,
    }


def run_50_agents(ind: dict, target_pct: float) -> list[dict]:
    last_close = ind["last_close"]
    prev_close = ind["prev_close"]
    last_open = ind["last_open"]
    last_high = ind["last_high"]
    last_low = ind["last_low"]
    sma_5 = ind["sma_5"]
    sma_10 = ind["sma_10"]
    sma_20 = ind["sma_20"]
    sma_50 = ind["sma_50"]
    ema_9 = ind["ema_9"]
    ema_21 = ind["ema_21"]
    rsi = ind["rsi"]
    prev_rsi = ind["prev_rsi"]
    macd_hist = ind["macd_hist"]
    prev_macd_hist = ind["prev_macd_hist"]
    atr = ind["atr"]
    vol_ratio = ind["vol_ratio"]
    vol_avg_20 = ind["vol_avg_20"]
    high_20 = ind["high_20"]
    low_20 = ind["low_20"]
    high_52 = ind["high_52"]
    low_52 = ind["low_52"]
    ret_1d = ind["ret_1d"]
    ret_3d = ind["ret_3d"]
    ret_5d = ind["ret_5d"]
    ret_10d = ind["ret_10d"]
    ret_20d = ind["ret_20d"]
    body_ratio = ind["body_ratio"]
    candle_range = ind["candle_range"]
    close = ind["close_series"]
    volume = ind["volume_series"]

    agents_def = []

    def add_agent(agent_id: int, category: str, name: str, bullish: bool, confidence: float):
        agents_def.append({
            "id": agent_id,
            "category": category,
            "name": name,
            "signal": "BULLISH" if bullish else "BEARISH",
            "score": round(max(0.05, min(0.99, confidence)), 2),
        })

    add_agent(1, "פריצות", "סוכן פריצת שיא 20 ימים", last_close > high_20, 0.82 if last_close > high_20 else 0.35)
    add_agent(2, "פריצות", "סוכן התקרבות לפריצה", last_close >= high_20 * 0.985, 0.74 if last_close >= high_20 * 0.985 else 0.42)
    add_agent(3, "פריצות", "סוכן הגנה מתחתית 20 ימים", last_close > low_20 * 1.03, 0.72 if last_close > low_20 * 1.03 else 0.30)
    add_agent(4, "מומנטום", "סוכן מומנטום יומי חיובי", ret_1d > 0.01, 0.76 if ret_1d > 0.01 else 0.38)
    add_agent(5, "מומנטום", "סוכן מומנטום 3 ימים", ret_3d > 0.02, 0.75 if ret_3d > 0.02 else 0.40)
    add_agent(6, "מומנטום", "סוכן מומנטום שבועי (5 ימים)", ret_5d > 0.025, 0.77 if ret_5d > 0.025 else 0.41)
    add_agent(7, "מומנטום", "סוכן מגמה 10 ימים", ret_10d > 0.03, 0.73 if ret_10d > 0.03 else 0.42)
    add_agent(8, "מומנטום", "סוכן מגמה חודשית (20 ימים)", ret_20d > 0.04, 0.74 if ret_20d > 0.04 else 0.40)
    add_agent(9, "נרות יפניים", "סוכן פער מחיר (Gap Up/Down)", last_open > prev_close * 1.005, 0.73 if last_open > prev_close * 1.005 else 0.44)
    add_agent(10, "נרות יפניים", "סוכן נר שורי דומיננטי", (last_close > last_open) and (body_ratio > 0.55), 0.76 if (last_close > last_open) and (body_ratio > 0.55) else 0.39)

    add_agent(11, "מגמה", "סוכן מחיר מעל SMA5", last_close > sma_5, 0.72 if last_close > sma_5 else 0.37)
    add_agent(12, "מגמה", "סוכן מחיר מעל SMA10", last_close > sma_10, 0.74 if last_close > sma_10 else 0.38)
    add_agent(13, "מגמה", "סוכן מחיר מעל SMA20", last_close > sma_20, 0.78 if last_close > sma_20 else 0.35)
    add_agent(14, "מגמה", "סוכן מחיר מעל SMA50", last_close > sma_50, 0.80 if last_close > sma_50 else 0.34)
    add_agent(15, "מגמה", "סוכן מחיר מעל EMA9", last_close > ema_9, 0.74 if last_close > ema_9 else 0.39)
    add_agent(16, "מגמה", "סוכן מחיר מעל EMA21", last_close > ema_21, 0.77 if last_close > ema_21 else 0.36)
    add_agent(17, "מגמה", "סוכן יישור ממוצעים (EMA9 > EMA21)", ema_9 > ema_21, 0.79 if ema_9 > ema_21 else 0.35)
    add_agent(18, "מגמה", "סוכן מבנה שורי (SMA20 > SMA50)", sma_20 > sma_50, 0.81 if sma_20 > sma_50 else 0.34)
    add_agent(19, "מגמה", "סוכן מיקום בטווח 52 שבועות", (last_close - low_52) / max(high_52 - low_52, 1e-6) > 0.55, 0.71 if (last_close - low_52) / max(high_52 - low_52, 1e-6) > 0.55 else 0.43)
    add_agent(20, "מגמה", "סוכן שיפוע ממוצע 20 ימים", float(close.rolling(20).mean().iloc[-1]) > float(close.rolling(20).mean().iloc[-4]), 0.75 if float(close.rolling(20).mean().iloc[-1]) > float(close.rolling(20).mean().iloc[-4]) else 0.38)

    add_agent(21, "מתנדים", "סוכן RSI אזור כוח (50-72)", 50.0 <= rsi <= 72.0, 0.78 if 50.0 <= rsi <= 72.0 else 0.42)
    add_agent(22, "מתנדים", "סוכן RSI עולה מעל אתמול", rsi > prev_rsi, 0.73 if rsi > prev_rsi else 0.40)
    add_agent(23, "מתנדים", "סוכן היפוך ממכירת יתר (RSI < 35)", rsi < 35.0 and rsi > prev_rsi, 0.76 if rsi < 35.0 and rsi > prev_rsi else 0.38)
    add_agent(24, "מתנדים", "סוכן הימנעות מקניית יתר קיצונית", rsi < 78.0, 0.70 if rsi < 78.0 else 0.25)
    add_agent(25, "מתנדים", "סוכן MACD מעל קו האות", ind["macd_val"] > ind["signal_val"], 0.79 if ind["macd_val"] > ind["signal_val"] else 0.35)
    add_agent(26, "מתנדים", "סוכן היסטוגרמת MACD חיובית", macd_hist > 0, 0.77 if macd_hist > 0 else 0.36)
    add_agent(27, "מתנדים", "סוכן האצת היסטוגרמת MACD", macd_hist > prev_macd_hist, 0.75 if macd_hist > prev_macd_hist else 0.38)
    add_agent(28, "מתנדים", "סוכן קרוס שורי ב-MACD", macd_hist > 0 and prev_macd_hist <= 0, 0.85 if (macd_hist > 0 and prev_macd_hist <= 0) else 0.40)
    add_agent(29, "מתנדים", "סוכן מומנטום ביניים (RSI > 55)", rsi > 55.0, 0.74 if rsi > 55.0 else 0.39)
    add_agent(30, "מתנדים", "סוכן איזון מתנדים", 42.0 <= rsi <= 68.0 and macd_hist >= -0.1, 0.72 if 42.0 <= rsi <= 68.0 and macd_hist >= -0.1 else 0.41)

    add_agent(31, "נפח", "סוכן זינוק נפח (מעל פי 1.3 מהממוצע)", vol_ratio >= 1.3, 0.82 if vol_ratio >= 1.3 else 0.38)
    add_agent(32, "נפח", "סוכן נפח תקין ומעלה", vol_ratio >= 1.0, 0.72 if vol_ratio >= 1.0 else 0.40)
    add_agent(33, "נפח", "סוכן נפח בעלייה יומית", (float(volume.iloc[-1]) > float(volume.iloc[-2])) and (last_close >= prev_close), 0.76 if (float(volume.iloc[-1]) > float(volume.iloc[-2])) and (last_close >= prev_close) else 0.39)
    add_agent(34, "נפח", "סוכן אישור פריצה בנפח", (last_close > sma_20) and (vol_ratio > 1.15), 0.80 if (last_close > sma_20) and (vol_ratio > 1.15) else 0.36)
    add_agent(35, "נפח", "סוכן צבירה שבועית בנפח", float(volume.tail(5).mean()) > vol_avg_20, 0.74 if float(volume.tail(5).mean()) > vol_avg_20 else 0.41)
    add_agent(36, "נפח", "סוכן נזילות גבוהה", vol_avg_20 >= 1_000_000, 0.75 if vol_avg_20 >= 1_000_000 else 0.35)
    add_agent(37, "נפח", "סוכן יחס נפח-מחיר חיובי", (ret_1d > 0) and (vol_ratio > 0.9), 0.73 if (ret_1d > 0) and (vol_ratio > 0.9) else 0.42)
    add_agent(38, "נפח", "סוכן הימנעות ממימוש בנפח חריג", not ((ret_1d < -0.02) and (vol_ratio > 1.8)), 0.71 if not ((ret_1d < -0.02) and (vol_ratio > 1.8)) else 0.25)
    add_agent(39, "נפח", "סוכן עקביות מחזור מסחר", float(volume.tail(3).min()) > vol_avg_20 * 0.6, 0.70 if float(volume.tail(3).min()) > vol_avg_20 * 0.6 else 0.44)
    add_agent(40, "נפח", "סוכן דחיפת קונים בסגירה", (last_close - last_low) / candle_range > 0.6, 0.75 if (last_close - last_low) / candle_range > 0.6 else 0.39)

    atr_pct = (atr / last_close) if last_close > 0 else 0.02
    add_agent(41, "סיכון", "סוכן תנודתיות תומכת יעד", 0.01 <= atr_pct <= 0.09, 0.78 if 0.01 <= atr_pct <= 0.09 else 0.40)
    add_agent(42, "סיכון", "סוכן מרחק סביר מ-SMA20", abs(last_close - sma_20) / sma_20 < 0.12, 0.73 if abs(last_close - sma_20) / sma_20 < 0.12 else 0.37)
    add_agent(43, "סיכון", "סוכן יחס סיכוי-סיכון ליעד", (atr * 1.5) < (last_close * target_pct), 0.77 if (atr * 1.5) < (last_close * target_pct) else 0.38)
    add_agent(44, "סיכון", "סוכן יציבות סגירה יומית", abs(ret_1d) < 0.12, 0.72 if abs(ret_1d) < 0.12 else 0.30)
    add_agent(45, "סיכון", "סוכן מרווח ביטחון מתמיכה", (last_close - low_20) / max(last_close, 1e-6) >= 0.025, 0.74 if (last_close - low_20) / max(last_close, 1e-6) >= 0.025 else 0.36)
    add_agent(46, "סיכון", "סוכן הימנעות ממתיחה קיצונית מעל EMA21", (last_close - ema_21) / max(ema_21, 1e-6) < 0.14, 0.73 if (last_close - ema_21) / max(ema_21, 1e-6) < 0.14 else 0.33)
    add_agent(47, "סיכון", "סוכן עקביות טווח קצר", ret_3d > -0.03 and ret_5d > -0.04, 0.74 if ret_3d > -0.03 and ret_5d > -0.04 else 0.35)
    add_agent(48, "סיכון", "סוכן מומנטום פוטנציאלי ליעד", (ret_5d + (rsi - 50) / 100.0) > 0.02, 0.76 if (ret_5d + (rsi - 50) / 100.0) > 0.02 else 0.40)
    add_agent(49, "סיכון", "סוכן איכות מגמה כוללת", (last_close > sma_20) and (rsi > 48), 0.78 if (last_close > sma_20) and (rsi > 48) else 0.36)
    add_agent(50, "סיכון", "סוכן אישור סופי לתרחיש לונג", (last_close > ema_9) and (macd_hist >= prev_macd_hist * 0.8), 0.77 if (last_close > ema_9) and (macd_hist >= prev_macd_hist * 0.8) else 0.38)

    return agents_def


def build_sentiment_summary(agents: list[dict], change_pct: float, rsi: float) -> dict:
    bullish = [a for a in agents if a["signal"] == "BULLISH"]
    bearish = [a for a in agents if a["signal"] == "BEARISH"]
    bull_ratio = len(bullish) / max(len(agents), 1)

    cat_stats = {}
    for cat in ["פריצות", "מומנטום", "נרות יפניים", "מגמה", "מתנדים", "נפח", "סיכון"]:
        in_cat = [a for a in agents if a["category"] == cat]
        if in_cat:
            b_cat = len([a for a in in_cat if a["signal"] == "BULLISH"])
            cat_stats[cat] = f"{b_cat}/{len(in_cat)}"

    if bull_ratio >= 0.65 and change_pct >= 0:
        label = "שורי חזק"
        code = "BULLISH"
    elif bull_ratio >= 0.52:
        label = "שורי מתון"
        code = "BULLISH_SOFT"
    elif bull_ratio <= 0.38 or change_pct <= -2.5:
        label = "דובי / לחץ מכירה"
        code = "BEARISH"
    else:
        label = "ניטרלי / מעורב"
        code = "NEUTRAL"

    rsi_note = "קניית יתר" if rsi >= 70 else ("מכירת יתר" if rsi <= 30 else "מאוזן")
    desc = f"{label} ({len(bullish)} בעד / {len(bearish)} נגד | RSI {rsi:.1f} - {rsi_note})"

    return {
        "code": code,
        "label": label,
        "description": desc,
        "categories": cat_stats,
    }


def build_plain_explanation(agents: list[dict]) -> str:
    cat_labels = {
        "פריצות": "מבנה פריצה/תמיכה טוב",
        "מומנטום": "מומנטום מחיר חיובי",
        "מגמה": "מגמה עולה מעל ממוצעים",
        "מתנדים": "איתות חיובי ב-RSI/MACD",
        "נפח": "תמיכה של נפח מסחר",
        "סיכון": "יחס סיכון סביר ליעד",
    }
    bullish = [a for a in agents if a["signal"] == "BULLISH"]
    ordered_cats = []
    for cat in ["פריצות", "מומנטום", "מגמה", "מתנדים", "נפח", "סיכון"]:
        if len([a for a in bullish if a["category"] == cat]) >= 2:
            ordered_cats.append(cat_labels[cat])
    return " • ".join(ordered_cats[:3]) if ordered_cats else "שילוב כללי חיובי של רוב הסוכנים"


def compute_entry_and_exit_levels(ind: dict, weighted_score: float, bull_count: int, target_pct: float) -> dict:
    last_close = ind["last_close"]
    sma_20 = ind["sma_20"]
    sma_50 = ind["sma_50"]
    ema_9 = ind["ema_9"]
    ema_21 = ind["ema_21"]
    high_10 = ind["high_10"]
    low_10 = ind["low_10"]
    high_20 = ind["high_20"]
    low_20 = ind["low_20"]
    r1 = ind["r1"]
    s1 = ind["s1"]
    bb_upper = ind["bb_upper"]
    atr = ind["atr"]
    rsi = ind["rsi"]
    vol_ratio = ind["vol_ratio"]
    macd_hist = ind["macd_hist"]

    # רמת פריצה אמיתית מהנתונים (ללא דחיפה מלאכותית של המחיר הנוכחי)
    resistance_ref = max(high_20, min(r1, bb_upper)) if r1 > high_20 * 0.96 else high_20
    breakout_level = round(max(0.01, resistance_ref * 1.002), 2)

    # תמיכה מרכזית משולבת (ממוצעים, S1 ושפל אחרון)
    support_candidates = [x for x in [s1, ema_21, sma_20, low_10] if 0 < x < last_close * 1.03]
    primary_support = round(max(support_candidates), 2) if support_candidates else round(max(0.01, low_20), 2)

    already_broke_out = last_close > high_20 and (vol_ratio >= 1.0 or macd_hist > 0)
    near_breakout = (high_20 * 0.988) <= last_close <= high_20
    uptrend_pullback = (last_close > sma_50) and (last_close >= sma_20 * 0.97) and (last_close <= high_20 * 0.988) and (rsi < 72)

    if already_broke_out:
        entry_price = round(last_close, 2)
        entry_type = "אישור פריצה (שוק נוכחי)"
        entry_note = "המחיר פרץ מעל שיא 20 ימים עם תמיכת אינדיקטורים"
    elif uptrend_pullback:
        # כניסה בנסיגה בריאה לתמיכה/EMA21/S1 (יכולה להיות מתחת למחיר הנוכחי או סביבו)
        pullback_zone = max(primary_support, min(last_close, max(ema_21, s1, last_close - atr * 0.5)))
        entry_price = round(max(0.01, min(last_close, pullback_zone)), 2)
        entry_type = "כניסה בנסיגה לתמיכה (Pullback)"
        entry_note = "מגמה עולה — כניסה סביב תמיכה/ממוצע קצר במקום רדיפה אחרי שיא"
    elif near_breakout:
        entry_price = breakout_level
        entry_type = "כניסה בפריצת התנגדות"
        entry_note = "המחיר קרוב להתנגדות — כניסה באישור פריצת שיא 20 ימים"
    else:
        entry_price = round(last_close, 2)
        entry_type = "מחיר נוכחי (זהירות / בדיקת תמיכה)"
        entry_note = "מבנה מעורב — כדאי לוודא החזקת תמיכה ומומנטום לפני כניסה"

    delta_to_entry_pct = round(((entry_price - last_close) / max(last_close, 0.01)) * 100, 2)

    # חישוב סטופ לוס מבוסס ATR ותמיכה (מבטיח שהסטופ מתחת לכניסה)
    atr_stop = entry_price - max(atr * 1.25, entry_price * 0.015)
    support_stop = primary_support * 0.992 if primary_support < entry_price else atr_stop
    raw_stop = max(atr_stop, support_stop)
    max_stop_allowed = entry_price * 0.94
    stop_loss = round(max(0.01, min(raw_stop, max_stop_allowed, entry_price * 0.985)), 2)
    stop_pct = round(((entry_price - stop_loss) / max(entry_price, 0.01)) * 100, 1)

    # חישוב יעד רווח ביחס לשער הכניסה
    target_price = round(entry_price * (1.0 + target_pct), 2)

    # טריילינג ויציאה מומלצת
    trail_exit = round(max(stop_loss, min(target_price, max(ema_9, last_close - atr * 1.1))), 2)

    if last_close >= target_price * 0.985:
        exit_action = "צא למימוש רווח"
        recommended_exit = round(last_close, 2)
        exit_reason = "המחיר הגיע או קרוב מאוד ליעד הרווח"
    elif last_close <= stop_loss:
        exit_action = "צא מיד (סטופ)"
        recommended_exit = round(last_close, 2)
        exit_reason = "המחיר ירד מתחת לסטופ לוס — הגבלת הפסד"
    elif last_close < ema_9 and macd_hist < 0 and bull_count < 24:
        exit_action = "צא בהחלשות מגמה"
        recommended_exit = round(last_close, 2)
        exit_reason = "שבירת EMA9 ומומנטום MACD שלילי"
    else:
        exit_action = "יעד / טריילינג"
        recommended_exit = target_price
        exit_reason = f"מימוש ביעד ${target_price:.2f}, סטופ ${stop_loss:.2f} או טריילינג ${trail_exit:.2f}"

    risk_per_share = max(entry_price - stop_loss, 0.01)
    reward_per_share = max(target_price - entry_price, 0.01)
    rr_ratio = round(reward_per_share / risk_per_share, 2)

    return {
        "breakoutLevel": breakout_level,
        "supportLevel": primary_support,
        "entry": entry_price,
        "entryType": entry_type,
        "entryNote": entry_note,
        "deltaToEntryPct": delta_to_entry_pct,
        "stopLoss": stop_loss,
        "stopPct": stop_pct,
        "targetPrice": target_price,
        "targetPct": round(target_pct * 100, 1),
        "trailExit": trail_exit,
        "recommendedExit": recommended_exit,
        "exitAction": exit_action,
        "exitReason": exit_reason,
        "rrRatio": rr_ratio,
        "indicatorsSnapshot": {
            "rsi14": round(rsi, 1),
            "macdHist": round(macd_hist, 3),
            "volRatio": round(vol_ratio, 2),
            "atrPct": round((atr / max(last_close, 0.01)) * 100, 2),
            "ema9": round(ema_9, 2),
            "ema21": round(ema_21, 2),
            "sma20": round(sma_20, 2),
            "sma50": round(sma_50, 2),
        },
    }


def orchestrator_evaluate(ticker: str, df: pd.DataFrame, agents: list[dict], target_pct: float) -> dict:
    ind = compute_indicators_pack(df)
    last_close = ind["last_close"]
    prev_close = ind["prev_close"]
    change_pct = round(((last_close - prev_close) / max(prev_close, 0.01)) * 100, 2)

    bullish_agents = [a for a in agents if a["signal"] == "BULLISH"]
    bearish_agents = [a for a in agents if a["signal"] == "BEARISH"]

    total_weight = sum(a["score"] for a in agents) or 1.0
    bull_weight = sum(a["score"] for a in bullish_agents)
    weighted_score = bull_weight / total_weight

    levels = compute_entry_and_exit_levels(ind, weighted_score, len(bullish_agents), target_pct)
    sentiment = build_sentiment_summary(agents, change_pct, ind["rsi"])

    return {
        "ticker": ticker,
        "price": round(last_close, 2),
        "changePct": change_pct,
        "bullCount": len(bullish_agents),
        "bearCount": len(bearish_agents),
        "scorePct": round(weighted_score * 100, 1),
        "weightedScore": round(weighted_score, 3),
        **levels,
        "sentiment": sentiment,
        "reason": build_plain_explanation(agents),
        "agents": agents,
    }


def save_latest_scan(evaluated: list[dict], scanned_tickers: list[str], target_pct_label: float, is_manual: bool):
    payload = {
        "updatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source": "Yahoo Finance (yfinance)",
        "mode": "manual" if is_manual else "auto",
        "targetPct": target_pct_label,
        "scannedCount": len(scanned_tickers),
        "scannedTickers": scanned_tickers,
        "results": evaluated,
    }
    with open(LATEST_SCAN_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_demo_portfolio() -> dict:
    if os.path.exists(DEMO_FILE):
        try:
            with open(DEMO_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "open_positions" in data and "closed_positions" in data:
                    return data
        except Exception as exc:
            print(f"Error loading {DEMO_FILE}: {exc}")
    return {"open_positions": [], "closed_positions": []}


def save_demo_portfolio(data: dict):
    with open(DEMO_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def update_demo_portfolio(evaluated: list[dict], qualified: list[dict], demo_mode: str) -> str:
    portfolio = load_demo_portfolio()
    price_map = {item["ticker"]: item["price"] for item in evaluated}
    now_str = datetime.now(timezone.utc).strftime("%d/%m %H:%M UTC")
    status_lines = []

    if demo_mode in ("check", "both"):
        still_open = []
        for pos in portfolio["open_positions"]:
            t = pos["ticker"]
            curr = price_map.get(t)
            if curr is None:
                df = fetch_stock_data(t)
                curr = round(float(df["Close"].iloc[-1]), 2) if df is not None else pos["entry"]

            if curr >= pos["target_price"]:
                pnl_pct = round(((curr - pos["entry"]) / pos["entry"]) * 100, 2)
                portfolio["closed_positions"].append({
                    **pos,
                    "exit_price": curr,
                    "pnl_pct": pnl_pct,
                    "result": "WIN",
                    "exit_reason": "הגיעה ליעד רווח",
                    "closed_at": now_str,
                })
                status_lines.append(f"✅ <b>{t}</b> הגיעה ליעד (+{pnl_pct}%) ב-${curr:.2f}")
            elif curr <= pos["stop_loss"]:
                pnl_pct = round(((curr - pos["entry"]) / pos["entry"]) * 100, 2)
                portfolio["closed_positions"].append({
                    **pos,
                    "exit_price": curr,
                    "pnl_pct": pnl_pct,
                    "result": "LOSS",
                    "exit_reason": "פגעה בסטופ לוס",
                    "closed_at": now_str,
                })
                status_lines.append(f"🛑 <b>{t}</b> פגעה בסטופ ({pnl_pct}%) ב-${curr:.2f}")
            else:
                pnl_pct = round(((curr - pos["entry"]) / pos["entry"]) * 100, 2)
                still_open.append(pos)
                status_lines.append(f"⏳ <b>{t}</b> פתוחה (עכשיו ${curr:.2f} | {pnl_pct:+.1f}%)")

        portfolio["open_positions"] = still_open

    if demo_mode in ("open", "both"):
        open_tickers = {p["ticker"] for p in portfolio["open_positions"]}
        for pick in qualified:
            if len(portfolio["open_positions"]) >= 6:
                break
            if pick["ticker"] in open_tickers:
                continue
            portfolio["open_positions"].append({
                "ticker": pick["ticker"],
                "entry": pick["entry"],
                "entry_type": pick["entryType"],
                "breakout_level": pick["breakoutLevel"],
                "stop_loss": pick["stopLoss"],
                "target_price": pick["targetPrice"],
                "target_pct": pick["targetPct"],
                "score": pick["weightedScore"],
                "opened_at": now_str,
            })
            open_tickers.add(pick["ticker"])
            status_lines.append(f"🆕 נפתחה עסקת דמו ב-<b>{pick['ticker']}</b> ב-${pick['entry']:.2f} (יעד ${pick['targetPrice']:.2f})")

    save_demo_portfolio(portfolio)

    closed = portfolio["closed_positions"]
    wins = len([c for c in closed if c.get("result") == "WIN"])
    total_closed = len(closed)
    win_rate = round((wins / total_closed) * 100, 1) if total_closed > 0 else 0.0

    summary_header = f"<b>תיק דמו:</b> {len(portfolio['open_positions'])} פתוחות | {total_closed} סגורות (הצלחה: {win_rate}%)"
    if not status_lines:
        return f"{summary_header}\nאין שינויים חדשים בעסקאות הדמו."
    return f"{summary_header}\n" + "\n".join(status_lines[:6])


def build_watchlist_telegram_section(watchlist_tickers: list[str], eval_map: dict[str, dict]) -> str:
    if not watchlist_tickers:
        return ""
    lines = ["<b>📌 עדכון רשימת מעקב:</b>"]
    for t in watchlist_tickers[:8]:
        item = eval_map.get(t)
        if not item:
            df = fetch_stock_data(t)
            if df is not None:
                agents = run_50_agents(compute_indicators_pack(df), DEFAULT_PROFIT_TARGET_PCT)
                item = orchestrator_evaluate(t, df, agents, DEFAULT_PROFIT_TARGET_PCT)
        if item:
            sign = "+" if item["changePct"] >= 0 else ""
            lines.append(
                f"• <b>{t}</b>: עכשיו ${item['price']:.2f} ({sign}{item['changePct']}%) | "
                f"כניסה ${item['entry']:.2f} ({item['entryType']}) | פריצה ${item['breakoutLevel']:.2f} | "
                f"יעד ${item['targetPrice']:.2f} | סטופ ${item['stopLoss']:.2f} | "
                f"יציאה: <b>{item['exitAction']}</b> (${item['recommendedExit']:.2f})"
            )
        else:
            lines.append(f"• <b>{t}</b>: אין נתון זמין כרגע")
    return "\n".join(lines)


def format_telegram_alert(picks: list[dict], scanned_tickers: list[str], target_pct_label: float, is_manual_tickers: bool, demo_summary: str, watchlist_section: str) -> str:
    now_str = datetime.now(timezone.utc).strftime("%d/%m %H:%M UTC")
    mode_label = "בדיקת מניות נבחרות" if is_manual_tickers else "סריקה אוטומטית מתחלפת"
    sample_tickers = ", ".join(scanned_tickers[:10]) + ("..." if len(scanned_tickers) > 10 else "")

    lines = [
        f"<b>50 Agents Scanner — {mode_label}</b>",
        f"נסרקו {len(scanned_tickers)} מניות | יעד רווח: <b>+{target_pct_label}%</b> | {now_str}",
        f"מניות: <code>{sample_tickers}</code>",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    for idx, item in enumerate(picks, start=1):
        sent = item.get("sentiment", {}).get("description", "")
        snap = item.get("indicatorsSnapshot", {})
        lines.extend([
            f"<b>{idx}. {item['ticker']}</b> — מחיר <b>${item['price']:.2f}</b> | התאמה <b>{item['scorePct']}%</b> ({item['bullCount']}/50)",
            f"סנטימנט: <b>{sent}</b> | נפח x{snap.get('volRatio', 1)} | ATR {snap.get('atrPct', 0)}%",
            f"פריצה: <b>${item['breakoutLevel']:.2f}</b> | תמיכה: <b>${item['supportLevel']:.2f}</b> | כניסה: <b>${item['entry']:.2f}</b> ({item['entryType']})",
            f"יעד (+{item['targetPct']}%): <b>${item['targetPrice']:.2f}</b> | סטופ: <b>${item['stopLoss']:.2f}</b> (-{item['stopPct']}%) | 1:{item['rrRatio']}",
            f"יציאה: <b>{item['exitAction']}</b> ב-<b>${item['recommendedExit']:.2f}</b> — {item['exitReason']}",
            f"סיבה: {item['reason']}",
            "────────────────────",
        ])

    if watchlist_section:
        lines.append(watchlist_section)
        lines.append("────────────────────")

    lines.append(demo_summary)
    return "\n".join(lines).strip()


def main():
    target_pct = get_profit_target_pct()
    target_pct_label = round(target_pct * 100, 1)
    demo_mode = os.getenv("INPUT_DEMO_MODE", "both").strip().lower()

    watchlist_tickers = get_watchlist_tickers()
    tickers_to_scan, is_manual_tickers = get_scan_tickers(watchlist_tickers)
    evaluated = []
    scanned_ok = []

    for ticker in tickers_to_scan:
        df = fetch_stock_data(ticker)
        if df is None:
            continue
        scanned_ok.append(ticker)
        ind = compute_indicators_pack(df)
        agents = run_50_agents(ind, target_pct)
        summary = orchestrator_evaluate(ticker, df, agents, target_pct)
        evaluated.append(summary)
        print(
            f"{ticker}: price={summary['price']} entry={summary['entry']} ({summary['entryType']}) "
            f"breakout={summary['breakoutLevel']} support={summary['supportLevel']} "
            f"target={summary['targetPrice']} stop={summary['stopLoss']} exit={summary['exitAction']}"
        )

    evaluated.sort(key=lambda x: x["weightedScore"], reverse=True)
    save_latest_scan(evaluated, scanned_ok, target_pct_label, is_manual_tickers)

    qualified = evaluated[:5] if is_manual_tickers else [x for x in evaluated if x["weightedScore"] >= MIN_SCORE_TO_ALERT][:5]
    demo_summary = update_demo_portfolio(evaluated, qualified, demo_mode)
    eval_map = {item["ticker"]: item for item in evaluated}
    watchlist_section = build_watchlist_telegram_section(watchlist_tickers, eval_map)

    print(f"Scanned {len(scanned_ok)}/{len(tickers_to_scan)}; qualified: {len(qualified)}")

    if qualified or is_manual_tickers or watchlist_section:
        msg = format_telegram_alert(qualified, scanned_ok, target_pct_label, is_manual_tickers, demo_summary, watchlist_section)
        send_telegram_message(msg)
    else:
        print("No stocks passed threshold and no watchlist items; skipping Telegram message.")


if __name__ == "__main__":
    main()
