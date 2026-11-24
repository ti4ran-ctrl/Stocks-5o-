import os
import math
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone

WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "AVGO", "PLTR",
    "COIN", "MSTR", "MARA", "SMCI", "ARM", "NFLX", "INTC", "MU", "QCOM", "CRM",
    "SHOP", "UBER", "ABNB", "SOFI", "RIVN", "LCID", "NIO", "BABA", "LLY", "UNH"
]

PROFIT_TARGET_PCT = 0.06  # יעד רווח של 6%
MIN_SCORE_TO_ALERT = 0.68  # סף ציון מינימלי להמלצה


def get_telegram_config():
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    return token, chat_id


def send_telegram_message(text: str) -> bool:
    token, chat_id = get_telegram_config()
    if not token or not chat_id:
        print("Telegram credentials missing; skipping Telegram push.")
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
            print("Telegram message sent successfully.")
            return True
        print(f"Telegram API error ({resp.status_code}): {resp.text}")
        return False
    except Exception as exc:
        print(f"Failed sending Telegram message: {exc}")
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


def run_50_agents(df: pd.DataFrame) -> list[dict]:
    close = df["Close"]
    open_p = df["Open"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    last_open = float(open_p.iloc[-1])
    last_high = float(high.iloc[-1])
    last_low = float(low.iloc[-1])

    sma_5 = float(close.rolling(5).mean().iloc[-1])
    sma_10 = float(close.rolling(10).mean().iloc[-1])
    sma_20 = float(close.rolling(20).mean().iloc[-1])
    sma_50 = float(close.rolling(min(50, len(close))).mean().iloc[-1])
    ema_9 = float(close.ewm(span=9, adjust=False).mean().iloc[-1])
    ema_21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])

    rsi_series = compute_rsi(close, 14)
    rsi = float(rsi_series.iloc[-1]) if not math.isnan(float(rsi_series.iloc[-1])) else 50.0
    prev_rsi = float(rsi_series.iloc[-2]) if not math.isnan(float(rsi_series.iloc[-2])) else 50.0

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = float((macd - signal).iloc[-1])
    prev_macd_hist = float((macd - signal).iloc[-2])

    atr_series = compute_atr(df, 14)
    atr = float(atr_series.iloc[-1]) if not math.isnan(float(atr_series.iloc[-1])) else (last_close * 0.02)

    vol_avg_20 = float(volume.rolling(20).mean().iloc[-1])
    last_vol = float(volume.iloc[-1])
    vol_ratio = (last_vol / vol_avg_20) if vol_avg_20 > 0 else 1.0

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

    agents_def = []

    def add_agent(agent_id: int, category: str, name: str, bullish: bool, confidence: float):
        agents_def.append({
            "id": agent_id,
            "category": category,
            "name": name,
            "signal": "BULLISH" if bullish else "BEARISH",
            "score": round(max(0.05, min(0.99, confidence)), 2),
        })

    # 1-10: פריצות, מומנטום ונרות יפניים
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

    # 11-20: ממוצעים נעים ומגמות
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

    # 21-30: מתנדים (RSI, MACD)
    add_agent(21, "מתנדים", "סוכן RSI אזור כוח (50-72)", 50.0 <= rsi <= 72.0, 0.78 if 50.0 <= rsi <= 72.0 else 0.42)
    add_agent(22, "מתנדים", "סוכן RSI עולה מעל אתמול", rsi > prev_rsi, 0.73 if rsi > prev_rsi else 0.40)
    add_agent(23, "מתנדים", "סוכן היפוך ממכירת יתר (RSI < 35)", rsi < 35.0 and rsi > prev_rsi, 0.76 if rsi < 35.0 and rsi > prev_rsi else 0.38)
    add_agent(24, "מתנדים", "סוכן הימנעות מקניית יתר קיצונית", rsi < 78.0, 0.70 if rsi < 78.0 else 0.25)
    add_agent(25, "מתנדים", "סוכן MACD מעל קו האות", float(macd.iloc[-1]) > float(signal.iloc[-1]), 0.79 if float(macd.iloc[-1]) > float(signal.iloc[-1]) else 0.35)
    add_agent(26, "מתנדים", "סוכן היסטוגרמת MACD חיובית", macd_hist > 0, 0.77 if macd_hist > 0 else 0.36)
    add_agent(27, "מתנדים", "סוכן האצת היסטוגרמת MACD", macd_hist > prev_macd_hist, 0.75 if macd_hist > prev_macd_hist else 0.38)
    add_agent(28, "מתנדים", "סוכן קרוס שורי ב-MACD", (float(macd.iloc[-2]) <= float(signal.iloc[-2])) and (float(macd.iloc[-1]) > float(signal.iloc[-1])), 0.85 if (float(macd.iloc[-2]) <= float(signal.iloc[-2])) and (float(macd.iloc[-1]) > float(signal.iloc[-1])) else 0.40)
    add_agent(29, "מתנדים", "סוכן מומנטום ביניים (RSI > 55)", rsi > 55.0, 0.74 if rsi > 55.0 else 0.39)
    add_agent(30, "מתנדים", "סוכן איזון מתנדים", 42.0 <= rsi <= 68.0 and macd_hist >= -0.1, 0.72 if 42.0 <= rsi <= 68.0 and macd_hist >= -0.1 else 0.41)

    # 31-40: נפח מסחר (Volume)
    add_agent(31, "נפח", "סוכן זינוק נפח (מעל פי 1.3 מהממוצע)", vol_ratio >= 1.3, 0.82 if vol_ratio >= 1.3 else 0.38)
    add_agent(32, "נפח", "סוכן נפח תקין ומעלה", vol_ratio >= 1.0, 0.72 if vol_ratio >= 1.0 else 0.40)
    add_agent(33, "נפח", "סוכן נפח בעלייה יומית", (last_vol > float(volume.iloc[-2])) and (last_close >= prev_close), 0.76 if (last_vol > float(volume.iloc[-2])) and (last_close >= prev_close) else 0.39)
    add_agent(34, "נפח", "סוכן אישור פריצה בנפח", (last_close > sma_20) and (vol_ratio > 1.15), 0.80 if (last_close > sma_20) and (vol_ratio > 1.15) else 0.36)
    add_agent(35, "נפח", "סוכן צבירה שבועית בנפח", float(volume.tail(5).mean()) > vol_avg_20, 0.74 if float(volume.tail(5).mean()) > vol_avg_20 else 0.41)
    add_agent(36, "נפח", "סוכן נזילות גבוהה", vol_avg_20 >= 1_000_000, 0.75 if vol_avg_20 >= 1_000_000 else 0.35)
    add_agent(37, "נפח", "סוכן יחס נפח-מחיר חיובי", (ret_1d > 0) and (vol_ratio > 0.9), 0.73 if (ret_1d > 0) and (vol_ratio > 0.9) else 0.42)
    add_agent(38, "נפח", "סוכן הימנעות ממימוש בנפח חריג", not ((ret_1d < -0.02) and (vol_ratio > 1.8)), 0.71 if not ((ret_1d < -0.02) and (vol_ratio > 1.8)) else 0.25)
    add_agent(39, "נפח", "סוכן עקביות מחזור מסחר", float(volume.tail(3).min()) > vol_avg_20 * 0.6, 0.70 if float(volume.tail(3).min()) > vol_avg_20 * 0.6 else 0.44)
    add_agent(40, "נפח", "סוכן דחיפת קונים בסגירה", (last_close - last_low) / candle_range > 0.6, 0.75 if (last_close - last_low) / candle_range > 0.6 else 0.39)

    # 41-50: תנודתיות, סיכון והיתכנות יעד 6%
    atr_pct = (atr / last_close) if last_close > 0 else 0.02
    add_agent(41, "סיכון", "סוכן תנודתיות תומכת יעד 6%", 0.012 <= atr_pct <= 0.08, 0.78 if 0.012 <= atr_pct <= 0.08 else 0.40)
    add_agent(42, "סיכון", "סוכן מרחק סביר מ-SMA20", abs(last_close - sma_20) / sma_20 < 0.12, 0.73 if abs(last_close - sma_20) / sma_20 < 0.12 else 0.37)
    add_agent(43, "סיכון", "סוכן יחס סיכוי-סיכון ליעד 6%", (atr * 1.5) < (last_close * PROFIT_TARGET_PCT), 0.77 if (atr * 1.5) < (last_close * PROFIT_TARGET_PCT) else 0.38)
    add_agent(44, "סיכון", "סוכן יציבות סגירה יומית", abs(ret_1d) < 0.12, 0.72 if abs(ret_1d) < 0.12 else 0.30)
    add_agent(45, "סיכון", "סוכן מרווח ביטחון מתמיכה", (last_close - low_20) / max(last_close, 1e-6) >= 0.025, 0.74 if (last_close - low_20) / max(last_close, 1e-6) >= 0.025 else 0.36)
    add_agent(46, "סיכון", "סוכן הימנעות ממתיחה קיצונית מעל EMA21", (last_close - ema_21) / max(ema_21, 1e-6) < 0.14, 0.73 if (last_close - ema_21) / max(ema_21, 1e-6) < 0.14 else 0.33)
    add_agent(47, "סיכון", "סוכן עקביות טווח קצר", ret_3d > -0.03 and ret_5d > -0.04, 0.74 if ret_3d > -0.03 and ret_5d > -0.04 else 0.35)
    add_agent(48, "סיכון", "סוכן מומנטום פוטנציאלי ל-6%", (ret_5d + (rsi - 50) / 100.0) > 0.02, 0.76 if (ret_5d + (rsi - 50) / 100.0) > 0.02 else 0.40)
    add_agent(49, "סיכון", "סוכן איכות מגמה כוללת", (last_close > sma_20) and (rsi > 48), 0.78 if (last_close > sma_20) and (rsi > 48) else 0.36)
    add_agent(50, "סיכון", "סוכן אישור סופי לתרחיש לונג", (last_close > ema_9) and (macd_hist >= prev_macd_hist * 0.8), 0.77 if (last_close > ema_9) and (macd_hist >= prev_macd_hist * 0.8) else 0.38)

    return agents_def


def orchestrator_evaluate(ticker: str, df: pd.DataFrame, agents: list[dict]) -> dict:
    last_close = float(df["Close"].iloc[-1])
    atr_series = compute_atr(df, 14)
    atr = float(atr_series.iloc[-1]) if not math.isnan(float(atr_series.iloc[-1])) else (last_close * 0.02)

    bullish_agents = [a for a in agents if a["signal"] == "BULLISH"]
    bearish_agents = [a for a in agents if a["signal"] == "BEARISH"]

    total_weight = sum(a["score"] for a in agents) or 1.0
    bull_weight = sum(a["score"] for a in bullish_agents)
    weighted_score = bull_weight / total_weight

    entry_price = round(last_close, 2)
    stop_distance = min(max(atr * 1.25, entry_price * 0.015), entry_price * 0.045)
    stop_loss = round(max(0.01, entry_price - stop_distance), 2)
    target_6_pct = round(entry_price * (1.0 + PROFIT_TARGET_PCT), 2)

    risk_per_share = max(entry_price - stop_loss, 0.01)
    reward_per_share = max(target_6_pct - entry_price, 0.01)
    rr_ratio = round(reward_per_share / risk_per_share, 2)

    top_reasons = [a["name"] for a in sorted(bullish_agents, key=lambda x: x["score"], reverse=True)[:4]]

    return {
        "ticker": ticker,
        "bull_count": len(bullish_agents),
        "bear_count": len(bearish_agents),
        "weighted_score": round(weighted_score, 3),
        "entry": entry_price,
        "stop_loss": stop_loss,
        "target_6_pct": target_6_pct,
        "rr_ratio": rr_ratio,
        "top_reasons": top_reasons,
    }


def format_telegram_alert(picks: list[dict], scanned_count: int) -> str:
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"<b>50 Agents Stock Scanner — התראת איתות</b>",
        f"נסרקו {scanned_count} מניות | {now_str}",
        "",
    ]
    for item in picks:
        reasons_txt = " • ".join(item["top_reasons"]) if item["top_reasons"] else "איתות מגמה חיובי"
        lines.extend([
            f"<b>{item['ticker']}</b> | ציון מנהל: <b>{int(item['weighted_score'] * 100)}%</b> ({item['bull_count']}/50 שוריים)",
            f"• שער כניסה: <b>${item['entry']:.2f}</b>",
            f"• Stop-Loss: <b>${item['stop_loss']:.2f}</b>",
            f"• יעד רווח (+6%): <b>${item['target_6_pct']:.2f}</b> (יחס סיכוי/סיכון 1:{item['rr_ratio']})",
            f"• מובילים: {reasons_txt}",
            "",
        ])
    return "\n".join(lines).strip()


def main():
    evaluated = []
    scanned = 0

    for ticker in WATCHLIST:
        df = fetch_stock_data(ticker)
        if df is None:
            continue
        scanned += 1
        agents = run_50_agents(df)
        summary = orchestrator_evaluate(ticker, df, agents)
        evaluated.append(summary)
        print(
            f"{ticker}: score={summary['weighted_score']} "
            f"({summary['bull_count']}/50 bull) entry={summary['entry']} target6={summary['target_6_pct']}"
        )

    evaluated.sort(key=lambda x: x["weighted_score"], reverse=True)
    qualified = [x for x in evaluated if x["weighted_score"] >= MIN_SCORE_TO_ALERT][:5]

    print(f"Total scanned: {scanned}, Qualified picks: {len(qualified)}")

    if qualified:
        msg = format_telegram_alert(qualified, scanned)
        send_telegram_message(msg)
    else:
        print("No stocks passed the threshold in this run; no Telegram alert sent.")


if __name__ == "__main__":
    main()
