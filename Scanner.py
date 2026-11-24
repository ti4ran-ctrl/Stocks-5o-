import os
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone

# ==========================================
# הגדרות משתמש - ניתן לערוך לפי הצורך
# ==========================================
WATCHLIST = [
    "NVDA", "AAPL", "MSFT", "TSLA", "AMZN", "META", "GOOGL",
    "AMD", "PLTR", "COIN", "NFLX", "CRM", "AVGO", "SMCI", "SOFI"
]

TARGET_PROFIT_PCT = 6.0      # יעד רווח מינימלי באחוזים (למשל 6%)
MAX_RISK_PCT = 3.0           # סיכון מקסימלי (Stop-Loss) באחוזים
MIN_MANAGER_SCORE = 68       # ציון מינימלי של הסוכן המנהל (מתוך 100) כדי לשלוח התראה
MIN_BULLISH_AGENTS = 28      # מינימום סוכנים חיוביים מתוך 50

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


# ==========================================
# שכבת חישוב אינדיקטורים מרכזית
# ==========================================
def compute_indicators(df: pd.DataFrame) -> dict:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    open_p = df["Open"]
    vol = df["Volume"]

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    def calc_rsi(series, period):
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    rsi7 = calc_rsi(close, 7)
    rsi14 = calc_rsi(close, 14)
    rsi21 = calc_rsi(close, 21)

    bb_std = close.rolling(20).std()
    bb_upper = sma20 + 2 * bb_std
    bb_lower = sma20 - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / sma20
    bb_pct_b = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    macd_line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()

    vol_sma20 = vol.rolling(20).mean()
    vol_ratio = vol / vol_sma20.replace(0, np.nan)
    typical_price = (high + low + close) / 3
    vwap20 = (typical_price * vol).rolling(20).sum() / vol.rolling(20).sum().replace(0, np.nan)

    obv = (np.sign(close.diff()).fillna(0) * vol).cumsum()
    obv_sma10 = obv.rolling(10).mean()

    resistance_20 = high.iloc[-21:-1].max()
    support_20 = low.iloc[-21:-1].min()
    resistance_50 = high.iloc[-51:-1].max() if len(high) > 51 else resistance_20

    return {
        "df": df,
        "price": float(close.iloc[-1]),
        "prev_close": float(close.iloc[-2]),
        "open": float(open_p.iloc[-1]),
        "high": float(high.iloc[-1]),
        "low": float(low.iloc[-1]),
        "sma20": float(sma20.iloc[-1]),
        "sma50": float(sma50.iloc[-1]),
        "sma200": float(sma200.iloc[-1]) if not np.isnan(sma200.iloc[-1]) else float(sma50.iloc[-1]),
        "ema9": float(ema9.iloc[-1]),
        "ema21": float(ema21.iloc[-1]),
        "ema50": float(ema50.iloc[-1]),
        "rsi7": float(rsi7.iloc[-1]),
        "rsi14": float(rsi14.iloc[-1]),
        "prev_rsi14": float(rsi14.iloc[-2]),
        "rsi21": float(rsi21.iloc[-1]),
        "bb_upper": float(bb_upper.iloc[-1]),
        "bb_lower": float(bb_lower.iloc[-1]),
        "bb_width": float(bb_width.iloc[-1]),
        "bb_width_min20": float(bb_width.iloc[-20:].min()),
        "bb_pct_b": float(bb_pct_b.iloc[-1]),
        "macd_line": float(macd_line.iloc[-1]),
        "macd_signal": float(macd_signal.iloc[-1]),
        "macd_hist": float(macd_hist.iloc[-1]),
        "prev_macd_hist": float(macd_hist.iloc[-2]),
        "atr14": float(atr14.iloc[-1]),
        "vol_ratio": float(vol_ratio.iloc[-1]),
        "vwap20": float(vwap20.iloc[-1]),
        "obv": float(obv.iloc[-1]),
        "obv_sma10": float(obv_sma10.iloc[-1]),
        "resistance_20": float(resistance_20),
        "resistance_50": float(resistance_50),
        "support_20": float(support_20),
    }


# ==========================================
# רשת 50 הסוכנים המתמחים
# ==========================================
def run_50_agents(ind: dict, info: dict, news: list) -> list:
    df = ind["df"]
    c, o, h, l = ind["price"], ind["open"], ind["high"], ind["low"]
    prev_c = ind["prev_close"]
    prev_o = float(df["Open"].iloc[-2])
    prev_h = float(df["High"].iloc[-2])
    prev_l = float(df["Low"].iloc[-2])

    body = abs(c - o)
    candle_range = max(h - l, 1e-6)
    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)

    agents = []

    def add_agent(agent_id, category, name, cond_bull, cond_bear, weight=1.0, note=""):
        if cond_bull:
            signal, score = "BULL", 85
        elif cond_bear:
            signal, score = "BEAR", 15
        else:
            signal, score = "NEUTRAL", 50
        agents.append({
            "id": agent_id,
            "category": category,
            "name": name,
            "signal": signal,
            "score": score,
            "weight": weight,
            "note": note
        })

    # --- קבוצה 1: נרות יפניים ותבניות מחיר (1-10) ---
    add_agent(1, "נרות יפניים", "סוכן Bullish/Bearish Engulfing",
              c > o and prev_c < prev_o and c >= prev_o and o <= prev_c,
              c < o and prev_c > prev_o and c <= prev_o and o >= prev_c, 1.3)
    add_agent(2, "נרות יפניים", "סוכן פטיש / כוכב נופל (Hammer/Shooting Star)",
              lower_wick > 2 * body and upper_wick < 0.3 * candle_range,
              upper_wick > 2 * body and lower_wick < 0.3 * candle_range, 1.2)
    add_agent(3, "נרות יפניים", "סוכן דוג'י היפוך (Doji Reversal)",
              body < 0.12 * candle_range and ind["rsi14"] < 40,
              body < 0.12 * candle_range and ind["rsi14"] > 65, 0.9)
    add_agent(4, "נרות יפניים", "סוכן נר מומנטום מלא (Marubozu)",
              c > o and body > 0.75 * candle_range,
              c < o and body > 0.75 * candle_range, 1.2)
    add_agent(5, "נרות יפניים", "סוכן Inside Bar Breakout",
              h < prev_h and l > prev_l and c > o,
              h < prev_h and l > prev_l and c < o, 1.0)
    add_agent(6, "נרות יפניים", "סוכן שלושה חיילים לבנים / עורבים",
              all(df["Close"].iloc[-i] > df["Open"].iloc[-i] for i in [1, 2, 3]),
              all(df["Close"].iloc[-i] < df["Open"].iloc[-i] for i in [1, 2, 3]), 1.2)
    add_agent(7, "נרות יפניים", "סוכן סגירה בחלק העליון של הנר",
              (c - l) / candle_range > 0.75,
              (c - l) / candle_range < 0.25, 1.0)
    add_agent(8, "נרות יפניים", "סוכן פריצת גבוה אתמול (Prev High Break)",
              c > prev_h,
              c < prev_l, 1.2)
    add_agent(9, "נרות יפניים
