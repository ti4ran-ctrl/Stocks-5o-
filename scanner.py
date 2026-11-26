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
MIN_SCORE_TO_ALERT = 0.65
DEFAULT_INITIAL_CAPITAL = 10000.0
MAX_OPEN_POSITIONS = 6
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


def fetch_market_bundle(ticker: str) -> tuple[pd.DataFrame | None, dict, dict]:
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None, {}, {}
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).copy()
        if len(df) < 30:
            return None, {}, {}

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

        return df, info, bench
    except Exception as exc:
        print(f"Error fetching {ticker}: {exc}")
        return None, {}, {}


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
    cloud_bot = min(a_val, b_val)
    return {
        "cloud_top": cloud_top,
        "cloud_bot": cloud_bot,
        "above_cloud": float(df["Close"].iloc[-1]) > cloud_top,
    }


def run_66_checks_before_entry_exit(df: pd.DataFrame, info: dict, bench: dict, target_pct: float) -> dict:
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
    fib_618 = high_60 - 0.618 * swing_range

    vol_avg20 = float(volume.rolling(20).mean().iloc[-1])
    last_vol = float(volume.iloc[-1])
    vol_ratio = (last_vol / vol_avg20) if vol_avg20 > 0 else 1.0

    obv = compute_obv(df)
    obv_rising = float(obv.iloc[-1]) > float(obv.iloc[-6]) if len(obv) >= 6 else True

    typical_price = (high + low + close) / 3.0
    vwap_20 = float((typical_price.tail(20) * volume.tail(20)).sum() / max(volume.tail(20).sum(), 1.0))

    gap_pct = ((last_open - prev_close) / max(prev_close, 1e-6)) * 100.0
    ret_1d = (last_close - prev_close) / max(prev_close, 1e-6)
    ret_5d = (last_close - float(close.iloc[-6])) / max(float(close.iloc[-6]), 1e-6)
    ret_20d = (last_close - float(close.iloc[-21])) / max(float(close.iloc[-21]), 1e-6)
    ret_60d = (last_close - float(close.iloc[-61])) / max(float(close.iloc[-61]), 1e-6) if len(close) >= 61 else ret_20d

    body = abs(last_close - last_open)
    rng = max(last_high - last_low, 1e-6)
    lower_wick = min(last_open, last_close) - last_low
    prev_body = abs(prev_close - prev_open)
    is_hammer = (lower_wick >= 2.0 * max(body, 1e-6))
    is_bull_engulf = (last_close > last_open) and (prev_close < prev_open) and (body > prev_body)

    rev_growth = info.get("revenueGrowth")
    earn_growth = info.get("earningsGrowth")
    op_margin = info.get("operatingMargins")
    net_margin = info.get("profitMargins")
    ebitda = info.get("ebitda")
    free_cashflow = info.get("freeCashflow")
    total_cash = info.get("totalCash")
    total_debt = info.get("totalDebt")
    current_ratio = info.get("currentRatio")
    pe_ratio = info.get("trailingPE")
    peg_ratio = info.get("pegRatio")
    ps_ratio = info.get("priceToSalesTrailing12Months")
    pb_ratio = info.get("priceToBook")
    ev_ebitda = info.get("enterpriseToEbitda")
    roe = info.get("returnOnEquity")
    div_yield = info.get("dividendYield")
    payout_ratio = info.get("payoutRatio")
    short_pct = info.get("shortPercentOfFloat")
    inst_pct = info.get("heldPercentInstitutions")
    insider_pct = info.get("heldPercentInsiders")
    rec_mean = info.get("recommendationMean")
    target_mean_price = info.get("targetMeanPrice")
    sector = info.get("sector") or ""
    industry = info.get("industry") or ""
    beta = info.get("beta")

    checks = []

    def add_check(num: int, section: str, title: str, status: str, weight: float = 1.0):
        checks.append({"id": num, "section": section, "title": title, "status": status, "weight": weight})

    # א. ניתוח טכני (1-19)
    add_check(1, "א. טכני", "מגמה", "PASS" if (last_close > sma_20 and ret_20d > 0) else ("WARN" if ret_20d >= -0.04 else "FAIL"))
    add_check(2, "א. טכני", "תמיכות והתנגדויות", "PASS" if last_close >= low_20 * 1.02 else "WARN")
    add_check(3, "א. טכני", "נרות יפניים", "PASS" if (is_hammer or is_bull_engulf or (last_close > last_open and body / rng > 0.5)) else "WARN")
    add_check(4, "א. טכני", "תבניות גרף", "PASS" if last_close >= high_20 * 0.97 or last_close >= fib_618 else "WARN")
    dc_recent = (prev_sma50 >= prev_sma200) and (sma_50 < sma_200)
    add_check(5, "א. טכני", "ממוצעים נעים וחציות", "FAIL" if dc_recent else ("PASS" if (last_close > sma_50 and sma_50 >= sma_200 * 0.98) else "WARN"))
    add_check(6, "א. טכני", "RSI", "WARN" if rsi >= 78 else ("PASS" if 42 <= rsi <= 72 else "WARN"))
    add_check(7, "א. טכני", "MACD", "PASS" if (float(macd.iloc[-1]) > float(sig.iloc[-1]) or macd_hist > prev_macd_hist) else "WARN")
    add_check(8, "א. טכני", "סטוכסטי", "PASS" if (stoch_k > stoch_d and stoch_k < 82) else ("WARN" if stoch_k < 88 else "FAIL"))
    add_check(9, "א. טכני", "בולינגר", "PASS" if bb_lower <= last_close <= bb_upper * 1.02 else "WARN")
    atr_pct = (atr / max(last_close, 1e-6)) * 100.0
    add_check(10, "א. טכני", "ATR", "PASS" if 0.8 <= atr_pct <= 8.5 else "WARN")
    add_check(11, "א. טכני", "ADX", "PASS" if adx >= 20 else "WARN")
    add_check(12, "א. טכני", "פיבונאצ'י", "PASS" if last_close >= fib_618 else "WARN")
    add_check(13, "א. טכני", "נפח", "PASS" if vol_ratio >= 0.95 and vol_avg20 >= 500_000 else "WARN")
    add_check(14, "א. טכני", "OBV ו-VWAP", "PASS" if (obv_rising and last_close >= vwap_20 * 0.98) else "WARN")
    add_check(15, "א. טכני", "פערי מחיר", "PASS" if abs(gap_pct) < 6.0 or gap_pct > 0 else "WARN")
    add_check(16, "א. טכני", "מסגרות זמן", "PASS" if ((last_close > ema_20) and (ema_20 > ema_50) and (ret_5d > -0.03)) else "WARN")
    bear_div = (last_close > prev_high) and (rsi < prev_rsi - 3.0)
    add_check(17, "א. טכני", "דיברגנציות", "WARN" if bear_div else "PASS")
    ichi = compute_ichimoku(df)
    add_check(18, "א. טכני", "איצ'ימוקו", "PASS" if ichi["above_cloud"] else "WARN")
    add_check(19, "א. טכני", "גל אליוט ודאו", "PASS" if (last_close > sma_50 and ret_60d > 0) else "WARN")

    # ב. דוחות כספיים (20-25)
    add_check(20, "ב. דוחות", "רווח והפסד", "PASS" if (net_margin is not None and net_margin > 0) else ("NA_DATA" if net_margin is None else "WARN"))
    add_check(21, "ב. דוחות", "מאזן", "PASS" if (total_cash is not None and total_debt is not None and total_cash >= total_debt * 0.35) else ("NA_DATA" if total_cash is None else "WARN"))
    add_check(22, "ב. דוחות", "תזרים ו-FCF", "PASS" if (free_cashflow is not None and free_cashflow > 0) else ("NA_DATA" if free_cashflow is None else "WARN"))
    add_check(23, "ב. דוחות", "צמיחה YoY/QoQ", "PASS" if (rev_growth is not None and rev_growth > 0.05) else ("NA_DATA" if rev_growth is None else "WARN"))
    add_check(24, "ב. דוחות", "שולי רווח", "PASS" if (op_margin is not None and op_margin > 0.08) else ("NA_DATA" if op_margin is None else "WARN"))
    add_check(25, "ב. דוחות", "EBITDA", "PASS" if (ebitda is not None and ebitda > 0) else ("NA_DATA" if ebitda is None else "WARN"))

    # ג. מכפילים ויחסים (26-37)
    add_check(26, "ג. מכפילים", "P/E", "PASS" if (pe_ratio is not None and 0 < pe_ratio < 65) else ("NA_DATA" if pe_ratio is None else "WARN"))
    add_check(27, "ג. מכפילים", "PEG", "PASS" if (peg_ratio is not None and 0 < peg_ratio <= 2.5) else ("NA_DATA" if peg_ratio is None else "WARN"))
    add_check(28, "ג. מכפילים", "P/S", "PASS" if (ps_ratio is not None and 0 < ps_ratio < 20) else ("NA_DATA" if ps_ratio is None else "WARN"))
    add_check(29, "ג. מכפילים", "P/B", "PASS" if (pb_ratio is not None and 0 < pb_ratio < 25) else ("NA_DATA" if pb_ratio is None else "WARN"))
    add_check(30, "ג. מכפילים", "EV/EBITDA", "PASS" if (ev_ebitda is not None and 0 < ev_ebitda < 35) else ("NA_DATA" if ev_ebitda is None else "WARN"))
    add_check(31, "ג. מכפילים", "EPS", "PASS" if (earn_growth is not None and earn_growth > 0) else ("NA_DATA" if earn_growth is None else "WARN"))
    add_check(32, "ג. מכפילים", "ROE/ROA/ROIC", "PASS" if (roe is not None and roe > 0.10) else ("NA_DATA" if roe is None else "WARN"))
    add_check(33, "ג. מכפילים", "חוב וריבית", "PASS" if (total_debt is None or total_cash is None or total_debt <= total_cash * 2.5) else "WARN")
    add_check(34, "ג. מכפילים", "נזילות", "PASS" if (current_ratio is not None and current_ratio >= 1.1) else ("NA_DATA" if current_ratio is None else "WARN"))
    add_check(35, "ג. מכפילים", "דיבידנד", "PASS" if (div_yield is not None and (payout_ratio is None or payout_ratio < 0.85)) else "NA_DATA")
    add_check(36, "ג. מכפילים", "Buyback", "PASS" if (free_cashflow is not None and free_cashflow > 0 and (payout_ratio is None or payout_ratio < 0.75)) else "NA_DATA")
    add_check(37, "ג. מכפילים", "השוואה לענף", "PASS" if (sector or industry) else "NA_DATA")

    # ד. איכותי (38-44)
    add_check(38, "ד. איכותי", "מודל עסקי", "PASS" if bool(sector) else "NA_DATA")
    add_check(39, "ד. איכותי", "יתרון תחרותי", "PASS" if (op_margin is not None and op_margin >= 0.15) else "NA_DATA")
    add_check(40, "ד. איכותי", "הנהלה", "PASS" if (roe is not None and roe >= 0.12) else "NA_DATA")
    add_check(41, "ד. איכותי", "בעלי עניין", "PASS" if insider_pct is not None else "NA_DATA")
    add_check(42, "ד. איכותי", "חדשנות", "PASS" if (rev_growth is not None and rev_growth > 0) else "NA_DATA")
    add_check(43, "ד. איכותי", "סחירות/לקוחות", "PASS" if (vol_avg20 >= 1_000_000) else "WARN")
    add_check(44, "ד. איכותי", "רגולציה ומשפטי", "WARN" if abs(gap_pct) > 10 else "PASS")

    # ה. דיווחים (45-51)
    add_check(45, "ה. דיווחים", "דיווחים מיידיים", "NA_DATA")
    add_check(46, "ה. דיווחים", "שיחות ועידה", "NA_DATA")
    add_check(47, "ה. דיווחים", "עסקאות בעלי עניין", "PASS" if insider_pct is not None else "NA_DATA")
    add_check(48, "ה. דיווחים", "מוסדיים", "PASS" if inst_pct is not None else "NA_DATA")
    add_check(49, "ה. דיווחים", "הנפקות ודילול", "WARN" if (ret_20d < -0.15 and vol_ratio > 1.5) else "PASS")
    add_check(50, "ה. דיווחים", "תקשורת וחדשות", "PASS" if vol_ratio < 2.5 else "WARN")
    add_check(51, "ה. דיווחים", "אנליסטים", "PASS" if (rec_mean is not None and rec_mean <= 2.5) else ("NA_DATA" if (rec_mean is None and target_mean_price is None) else "WARN"))

    # ו. סנטימנט ושוק (52-57)
    add_check(52, "ו. סנטימנט", "Short Interest", "WARN" if (short_pct is not None and short_pct > 0.15) else ("PASS" if short_pct is not None else "NA_DATA"))
    spy_ret20 = bench.get("spy", {}).get("ret_20d", 0.0)
    add_check(53, "ו. סנטימנט", "חוזק מול מדד", "PASS" if ((ret_20d - spy_ret20) >= 0) else "WARN")
    add_check(54, "ו. סנטימנט", "סנטימנט מומנטום", "PASS" if (rsi >= 50 and ret_5d > 0) else "WARN")
    vix_last = bench.get("vix", {}).get("last")
    add_check(55, "ו. סנטימנט", "VIX", "FAIL" if (vix_last is not None and vix_last > 30) else ("WARN" if (vix_last is not None and vix_last > 22) else ("PASS" if vix_last is not None else "NA_DATA")))
    spy_up = bench.get("spy", {}).get("above_sma20", True)
    add_check(56, "ו. סנטימנט", "מצב שוק רחב", "PASS" if spy_up else "WARN")
    add_check(57, "ו. סנטימנט", "אופציות", "WARN" if vol_ratio > 2.2 else "PASS")

    # ז. מקרו (58-61)
    env_rates = os.getenv("MACRO_RATES_TREND", "NEUTRAL").strip().upper()
    add_check(58, "ז. מקרו", "ריבית ואינפלציה", "PASS" if env_rates in ("EASING", "STABLE", "NEUTRAL") else "WARN")
    add_check(59, "ז. מקרו", "ענף ועונתיות", "PASS" if ret_20d > -0.05 else "WARN")
    env_fx_geo = os.getenv("MACRO_GEO_FX", "NEUTRAL").strip().upper()
    add_check(60, "ז. מקרו", "גיאו ומטבע", "WARN" if env_fx_geo in ("RISK_OFF", "HIGH_RISK") else "PASS")
    add_check(61, "ז. מקרו", "כלכלה כללית", "PASS" if spy_up else "WARN")

    # ח. ניהול סיכונים (62-66)
    add_check(62, "ח. סיכונים", "גודל פוזיציה", "PASS")
    add_check(63, "ח. סיכונים", "סטופ ויעד", "PASS")
    add_check(64, "ח. סיכונים", "יחס סיכון/סיכוי", "PASS")
    add_check(65, "ח. סיכונים", "פיזור תיק", "PASS")
    add_check(66, "ח. סיכונים", "אופק השקעה", "PASS")

    passed = [c for c in checks if c["status"] == "PASS"]
    warned = [c for c in checks if c["status"] == "WARN"]
    failed = [c for c in checks if c["status"] == "FAIL"]

    scored = [c for c in checks if c["status"] in ("PASS", "WARN", "FAIL")]
    weighted_sum = sum(
        (c["weight"] if c["status"] == "PASS" else (0.35 * c["weight"] if c["status"] == "WARN" else 0.0))
        for c in scored
    )
    max_sum = sum(c["weight"] for c in scored) or 1.0
    score_ratio = weighted_sum / max_sum

    if len(failed) >= 2 or score_ratio < 0.45:
        gate_verdict = "הימנע / המתן"
    elif len(failed) == 1 or score_ratio < 0.62:
        gate_verdict = "אזהרה / כניסה מותנית"
    else:
        gate_verdict = "מאושר לבחינת כניסה"

    return {
        "totalChecks": len(checks),
        "passedCount": len(passed),
        "warnedCount": len(warned),
        "failedCount": len(failed),
        "scorePct": round(score_ratio * 100, 1),
        "weightedScore": round(score_ratio, 3),
        "gateVerdict": gate_verdict,
        "levelsRaw": {
            "last_close": last_close,
            "prev_close": prev_close,
            "ret_1d": ret_1d,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "ema_20": ema_20,
            "high_10": high_10,
            "low_10": low_10,
            "high_20": high_20,
            "low_20": low_20,
            "bb_upper": bb_upper,
            "fib_618": fib_618,
            "atr": atr,
            "rsi": rsi,
            "macd_hist": macd_hist,
            "vol_ratio": vol_ratio,
            "vwap_20": vwap_20,
        },
    }


def compute_entry_exit_after_66_checks(eval_66: dict, target_pct: float) -> dict:
    r = eval_66["levelsRaw"]
    last_close = r["last_close"]
    prev_close = r["prev_close"]
    change_pct = round(((last_close - prev_close) / max(prev_close, 0.01)) * 100, 2)

    high_10 = r["high_10"]
    low_10 = r["low_10"]
    high_20 = r["high_20"]
    low_20 = r["low_20"]
    sma_20 = r["sma_20"]
    ema_20 = r["ema_20"]
    bb_upper = r["bb_upper"]
    fib_618 = r["fib_618"]
    vwap_20 = r["vwap_20"]
    atr = r["atr"]
    rsi = r["rsi"]
    macd_hist = r["macd_hist"]
    vol_ratio = r["vol_ratio"]

    near_res_candidates = [x for x in [high_10 * 1.002, high_20 * 1.002, bb_upper] if x > last_close * 0.99]
    breakout_level = round(min(near_res_candidates), 2) if near_res_candidates else round(last_close * 1.015, 2)

    near_sup_candidates = [x for x in [vwap_20, ema_20, sma_20, low_10, fib_618] if 0 < x < last_close * 1.01]
    support_level = round(max(near_sup_candidates), 2) if near_sup_candidates else round(max(0.01, low_20), 2)

    weak_short_term = (last_close < ema_20) and (macd_hist < 0) and (r["ret_1d"] < -0.015)

    if eval_66["failedCount"] >= 2 or weak_short_term:
        entry_price = round(max(last_close * 1.01, ema_20), 2) if ema_20 > last_close * 1.005 else round(breakout_level, 2)
        entry_type = "כניסה מותנית (לאחר התייצבות)"
        exit_action = "הימנע / המתן לכניסה"
        target_price = round(entry_price * (1.0 + target_pct), 2)
        max_risk_pct = min(max(target_pct / 1.5, 0.02), target_pct * 0.75)
        stop_loss = round(max(0.01, entry_price * (1.0 - max_risk_pct)), 2)
        recommended_exit = target_price
        exit_reason = f"יעד מותנה ${target_price:.2f} | סטופ ${stop_loss:.2f}"
    else:
        broke_out = last_close > high_20 and (vol_ratio >= 1.0 or macd_hist > 0)
        pullback_ok = (last_close >= sma_20 * 0.975) and (rsi < 72)

        if broke_out:
            entry_price = round(last_close, 2)
            entry_type = "כניסה בפריצה מאושרת"
        elif pullback_ok and support_level < last_close * 0.995:
            entry_price = round(max(support_level, last_close - min(atr * 0.4, last_close * 0.02)), 2)
            entry_type = "כניסה בנסיגה לתמיכה"
        else:
            entry_price = round(last_close, 2)
            entry_type = "כניסה במחיר נוכחי"

        target_price = round(entry_price * (1.0 + target_pct), 2)
        max_risk_pct = min(max(target_pct / 1.5, 0.015), target_pct * 0.75)
        stop_from_risk = entry_price * (1.0 - max_risk_pct)
        stop_from_sup = support_level * 0.992 if support_level < entry_price else stop_from_risk
        stop_loss = round(max(0.01, min(entry_price * 0.988, max(stop_from_risk, stop_from_sup))), 2)

        # שער יציאה דינמי (מתעדכן גם כשהמניה עולה תוך כדי החזקה)
        dynamic_trail = round(max(stop_loss * 1.005, min(target_price, max(ema_20, last_close - atr * 0.85))), 2)
        if last_close >= target_price * 0.985:
            exit_action = "צא למימוש רווח"
            recommended_exit = round(last_close, 2)
            exit_reason = "הגיע ליעד הרווח"
        elif last_close > entry_price * 1.025 and dynamic_trail > stop_loss:
            exit_action = "יציאה דינמית (טריילינג עולה)"
            recommended_exit = dynamic_trail
            exit_reason = f"עודכן ל-${dynamic_trail:.2f} (יעד ${target_price:.2f})"
        else:
            exit_action = "יעד רווח / סטופ"
            recommended_exit = target_price
            exit_reason = f"יעד ${target_price:.2f} | סטופ ${stop_loss:.2f}"

    stop_pct = round(((entry_price - stop_loss) / max(entry_price, 0.01)) * 100, 1)
    delta_pct = round(((entry_price - last_close) / max(last_close, 0.01)) * 100, 2)
    rr_ratio = round(max(target_price - entry_price, 0.01) / max(entry_price - stop_loss, 0.01), 2)

    sent_code = "BULLISH" if (eval_66["scorePct"] >= 65 and not weak_short_term) else ("BEARISH" if (eval_66["failedCount"] >= 2 or weak_short_term) else "NEUTRAL")
    sent_label = "שורי" if sent_code == "BULLISH" else ("דובי/נסיגה" if sent_code == "BEARISH" else "מעורב")

    return {
        "price": round(last_close, 2),
        "changePct": change_pct,
        "bullCount": eval_66["passedCount"],
        "bearCount": eval_66["failedCount"],
        "scorePct": eval_66["scorePct"],
        "weightedScore": eval_66["weightedScore"],
        "gateVerdict": "הימנע / המתן" if weak_short_term else eval_66["gateVerdict"],
        "checksSummary": {
            "total": eval_66["totalChecks"],
            "passed": eval_66["passedCount"],
            "warned": eval_66["warnedCount"],
            "failed": eval_66["failedCount"],
        },
        "breakoutLevel": breakout_level,
        "supportLevel": support_level,
        "entry": entry_price,
        "entryType": entry_type,
        "deltaToEntryPct": delta_pct,
        "stopLoss": stop_loss,
        "stopPct": stop_pct,
        "targetPrice": target_price,
        "targetPct": round(target_pct * 100, 1),
        "recommendedExit": recommended_exit,
        "exitAction": exit_action,
        "exitReason": exit_reason,
        "rrRatio": rr_ratio,
        "sentiment": {"code": sent_code, "label": sent_label},
    }


def load_demo_portfolio(initial_capital: float) -> dict:
    if os.path.exists(DEMO_FILE):
        try:
            with open(DEMO_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "open_positions" in data and "closed_positions" in data:
                    data.setdefault("initial_capital", initial_capital)
                    data.setdefault("cash", initial_capital)
                    return data
        except Exception as exc:
            print(f"Error loading {DEMO_FILE}: {exc}")
    return {
        "initial_capital": initial_capital,
        "cash": initial_capital,
        "open_positions": [],
        "closed_positions": [],
    }


def save_demo_portfolio(data: dict):
    with open(DEMO_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def update_demo_portfolio(
    evaluated: list[dict],
    qualified: list[dict],
    watchlist_tickers: list[str],
    demo_mode: str,
    initial_capital: float,
) -> tuple[dict, str]:
    portfolio = load_demo_portfolio(initial_capital)
    eval_map = {item["ticker"]: item for item in evaluated}
    now_str = datetime.now(timezone.utc).strftime("%d/%m %H:%M UTC")
    status_lines = []

    init_cap = float(portfolio.get("initial_capital", initial_capital) or initial_capital)
    cash = float(portfolio.get("cash", init_cap))

    # 1. עדכון שער יציאה דינמי וסגירת עסקאות כשמגיעים ליעד / שער יציאה / סטופ
    if demo_mode in ("check", "both"):
        still_open = []
        for pos in portfolio["open_positions"]:
            t = pos["ticker"]
            item = eval_map.get(t)
            if item:
                curr = float(item["price"])
                # עדכון שער יציאה דינמי כלפי מעלה כשהמניה מתקדמת
                prev_exit = float(pos.get("exit_price_target", pos.get("target_price", curr)))
                new_rec_exit = float(item.get("recommendedExit", prev_exit))
                if curr > float(pos["entry"]) * 1.015 and new_rec_exit > prev_exit:
                    pos["exit_price_target"] = round(new_rec_exit, 2)
                    pos["exit_action"] = item.get("exitAction", "יציאה דינמית")
                else:
                    pos.setdefault("exit_price_target", round(float(pos.get("target_price", curr)), 2))
                    pos.setdefault("exit_action", item.get("exitAction", "יעד רווח / סטופ"))
            else:
                df, _, _ = fetch_market_bundle(t)
                curr = round(float(df["Close"].iloc[-1]), 2) if df is not None else float(pos["entry"])
                pos.setdefault("exit_price_target", round(float(pos.get("target_price", curr)), 2))
                pos.setdefault("exit_action", "יעד רווח / סטופ")

            entry_p = float(pos["entry"])
            shares = float(pos.get("shares") or (float(pos.get("allocated", 1500.0)) / max(entry_p, 0.01)))
            allocated = round(shares * entry_p, 2)
            pos["shares"] = round(shares, 4)
            pos["allocated"] = allocated

            target_p = float(pos["target_price"])
            stop_p = float(pos["stop_loss"])
            active_exit_p = float(pos.get("exit_price_target", target_p))

            # תנאי יציאה מהמניה: הגעה ליעד, הגעה לשער יציאה דינמי ממומש, או פגיעה בסטופ
            hit_target = curr >= target_p
            hit_dynamic_exit = (active_exit_p != target_p) and (curr >= active_exit_p)
            hit_stop = curr <= stop_p

            if hit_target or hit_dynamic_exit or hit_stop:
                pnl_dollar = round((curr - entry_p) * shares, 2)
                pnl_pct = round(((curr - entry_p) / max(entry_p, 0.01)) * 100, 2)
                cash = round(cash + (curr * shares), 2)
                reason = "הגיעה ליעד רווח" if hit_target else ("מימוש בשער יציאה דינמי" if hit_dynamic_exit else "פגעה בסטופ לוס")
                result = "WIN" if pnl_dollar >= 0 else "LOSS"

                portfolio["closed_positions"].append({
                    **pos,
                    "exit_price": round(curr, 2),
                    "pnl_dollar": pnl_dollar,
                    "pnl_pct": pnl_pct,
                    "result": result,
                    "exit_reason": reason,
                    "closed_at": now_str,
                })
                icon = "✅" if pnl_dollar >= 0 else "🛑"
                status_lines.append(f"{icon} נמכרה <b>{t}</b> ב-${curr:.2f} ({pnl_pct:+.1f}% | {pnl_dollar:+.2f}$) — {reason}")
            else:
                unrealized_dollar = round((curr - entry_p) * shares, 2)
                unrealized_pct = round(((curr - entry_p) / max(entry_p, 0.01)) * 100, 2)
                pos["current_price"] = round(curr, 2)
                pos["unrealized_dollar"] = unrealized_dollar
                pos["unrealized_pct"] = unrealized_pct
                still_open.append(pos)
                status_lines.append(
                    f"⏳ <b>{t}</b> ב-${curr:.2f} ({unrealized_pct:+.1f}% | {unrealized_dollar:+.2f}$) | יציאה עדכנית: ${active_exit_p:.2f}"
                )

        portfolio["open_positions"] = still_open

    # 2. קנייה אוטומטית (מהרשימה האוטומטית + מניות מרשימת המעקב שבחרת)
    if demo_mode in ("open", "both"):
        open_tickers = {p["ticker"] for p in portfolio["open_positions"]}
        watch_picks = [eval_map[t] for t in watchlist_tickers if t in eval_map and "הימנע" not in eval_map[t].get("exitAction", "")]
        candidates = list(dict.fromkeys([x["ticker"] for x in (watch_picks + qualified)]))

        # חישוב גודל פוזיציה אוטומטי ביחס לתיק
        per_trade_budget = round(max(250.0, init_cap / max(MAX_OPEN_POSITIONS, 1)), 2)

        for t in candidates:
            if len(portfolio["open_positions"]) >= MAX_OPEN_POSITIONS:
                break
            if t in open_tickers:
                continue
            pick = eval_map.get(t)
            if not pick or "הימנע" in pick.get("exitAction", ""):
                continue

            entry_p = float(pick["entry"])
            if entry_p <= 0 or cash < min(250.0, per_trade_budget * 0.5):
                continue

            invest_amt = min(cash, per_trade_budget)
            shares = round(invest_amt / entry_p, 4)
            actual_cost = round(shares * entry_p, 2)
            if shares <= 0 or actual_cost > cash:
                continue

            cash = round(cash - actual_cost, 2)
            source_type = "רשימת מעקב" if t in watchlist_tickers else "בחירה אוטומטית"

            pos_obj = {
                "ticker": t,
                "source": source_type,
                "entry": round(entry_p, 2),
                "current_price": round(float(pick["price"]), 2),
                "shares": shares,
                "allocated": actual_cost,
                "stop_loss": round(float(pick["stopLoss"]), 2),
                "target_price": round(float(pick["targetPrice"]), 2),
                "exit_price_target": round(float(pick["recommendedExit"]), 2),
                "exit_action": pick.get("exitAction", "יעד רווח / סטופ"),
                "unrealized_dollar": round((float(pick["price"]) - entry_p) * shares, 2),
                "unrealized_pct": round(((float(pick["price"]) - entry_p) / max(entry_p, 0.01)) * 100, 2),
                "score": pick["weightedScore"],
                "opened_at": now_str,
            }
            portfolio["open_positions"].append(pos_obj)
            open_tickers.add(t)
            status_lines.append(
                f"🆕 נקנתה <b>{t}</b> ({source_type}) ב-${entry_p:.2f} (${actual_cost:.0f}) | יעד ${pick['targetPrice']:.2f} | יציאה ${pick['recommendedExit']:.2f}"
            )

    # חישוב רווח על כל מניה ורווח כללי על התיק
    open_market_value = 0.0
    open_unrealized_pnl = 0.0
    for p in portfolio["open_positions"]:
        c_p = float(eval_map.get(p["ticker"], {}).get("price", p.get("current_price", p["entry"])))
        sh = float(p.get("shares", 0))
        en = float(p.get("entry", c_p))
        val = round(c_p * sh, 2)
        upnl = round((c_p - en) * sh, 2)
        p["current_price"] = round(c_p, 2)
        p["unrealized_dollar"] = upnl
        p["unrealized_pct"] = round(((c_p - en) / max(en, 0.01)) * 100, 2)
        open_market_value += val
        open_unrealized_pnl += upnl

    realized_pnl = round(sum(float(c.get("pnl_dollar", 0)) for c in portfolio["closed_positions"]), 2)
    total_equity = round(cash + open_market_value, 2)
    total_pnl_dollar = round(total_equity - init_cap, 2)
    total_pnl_pct = round((total_pnl_dollar / max(init_cap, 1.0)) * 100, 2)

    portfolio["initial_capital"] = round(init_cap, 2)
    portfolio["cash"] = round(cash, 2)
    portfolio["summary"] = {
        "initial_capital": round(init_cap, 2),
        "cash": round(cash, 2),
        "open_market_value": round(open_market_value, 2),
        "total_equity": total_equity,
        "realized_pnl_dollar": realized_pnl,
        "unrealized_pnl_dollar": round(open_unrealized_pnl, 2),
        "total_pnl_dollar": total_pnl_dollar,
        "total_pnl_pct": total_pnl_pct,
    }

    save_demo_portfolio(portfolio)

    closed = portfolio["closed_positions"]
    wins = len([c for c in closed if c.get("result") == "WIN"])
    win_rate = round((wins / len(closed)) * 100, 1) if closed else 0.0

    header = (
        f"<b>תיק אוטומטי:</b> שווי ${total_equity:,.0f} | רווח כללי: <b>{total_pnl_dollar:+,.2f}$ ({total_pnl_pct:+.2f}%)</b>\n"
        f"מזומן: ${cash:,.0f} | ממומש: {realized_pnl:+,.2f}$ | פתוח: {open_unrealized_pnl:+,.2f}$ | הצלחה: {win_rate}%"
    )
    if not status_lines:
        return portfolio, f"{header}\nאין שינויים חדשים בפוזיציות."
    return portfolio, f"{header}\n" + "\n".join(status_lines[:7])


def save_latest_scan(evaluated: list[dict], scanned_tickers: list[str], target_pct_label: float, is_manual: bool):
    payload = {
        "updatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source": "Yahoo Finance (yfinance) — Universal Dynamic Exit & Auto Portfolio",
        "mode": "manual" if is_manual else "auto",
        "targetPct": target_pct_label,
        "scannedCount": len(scanned_tickers),
        "scannedTickers": scanned_tickers,
        "results": evaluated,
    }
    with open(LATEST_SCAN_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_watchlist_telegram_section(watchlist_tickers: list[str], eval_map: dict[str, dict], target_pct: float) -> str:
    if not watchlist_tickers:
        return ""
    lines = ["<b>📌 מעקב וניהול המניות שלך:</b>"]
    for t in watchlist_tickers[:8]:
        item = eval_map.get(t)
        if not item:
            df, info, bench = fetch_market_bundle(t)
            if df is not None:
                e66 = run_66_checks_before_entry_exit(df, info, bench, target_pct)
                item = {"ticker": t, **compute_entry_exit_after_66_checks(e66, target_pct)}
        if item:
            sign = "+" if item["changePct"] >= 0 else ""
            lines.append(
                f"• <b>{t}</b>: ${item['price']:.2f} ({sign}{item['changePct']}%) | "
                f"כניסה ${item['entry']:.2f} | יעד ${item['targetPrice']:.2f} | "
                f"יציאה דינמית: <b>${item['recommendedExit']:.2f}</b> ({item['exitAction']})"
            )
    return "\n".join(lines)


def format_telegram_alert(picks: list[dict], scanned_tickers: list[str], target_pct_label: float, is_manual_tickers: bool, demo_summary: str, watchlist_section: str) -> str:
    now_str = datetime.now(timezone.utc).strftime("%d/%m %H:%M UTC")
    mode_label = "בדיקת מניות נבחרות" if is_manual_tickers else "סריקה וניהול אוטומטי"
    sample_tickers = ", ".join(scanned_tickers[:10]) + ("..." if len(scanned_tickers) > 10 else "")

    lines = [
        f"<b>66-Checks Engine — {mode_label}</b>",
        f"נסרקו {len(scanned_tickers)} מניות | יעד: <b>+{target_pct_label}%</b> | {now_str}",
        f"מניות: <code>{sample_tickers}</code>",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    for idx, item in enumerate(picks, start=1):
        lines.extend([
            f"<b>{idx}. {item['ticker']}</b> — ${item['price']:.2f} | ציון <b>{item['scorePct']}%</b>",
            f"כניסה: <b>${item['entry']:.2f}</b> ({item['entryType']}) | יעד: <b>${item['targetPrice']:.2f}</b> | סטופ: <b>${item['stopLoss']:.2f}</b>",
            f"יציאה דינמית: <b>${item['recommendedExit']:.2f}</b> — <b>{item['exitAction']}</b>",
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
    initial_capital = get_initial_capital()
    demo_mode = os.getenv("INPUT_DEMO_MODE", "both").strip().lower()

    watchlist_tickers = get_watchlist_tickers()
    tickers_to_scan, is_manual_tickers = get_scan_tickers(watchlist_tickers)
    evaluated = []
    scanned_ok = []

    for ticker in tickers_to_scan:
        df, info, bench = fetch_market_bundle(ticker)
        if df is None:
            continue
        scanned_ok.append(ticker)
        eval_66 = run_66_checks_before_entry_exit(df, info, bench, target_pct)
        levels = compute_entry_exit_after_66_checks(eval_66, target_pct)
        summary = {"ticker": ticker, **levels}
        evaluated.append(summary)

    evaluated.sort(key=lambda x: x["weightedScore"], reverse=True)
    save_latest_scan(evaluated, scanned_ok, target_pct_label, is_manual_tickers)

    qualified = (
        evaluated[:5]
        if is_manual_tickers
        else [x for x in evaluated if x["weightedScore"] >= MIN_SCORE_TO_ALERT and "הימנע" not in x.get("exitAction", "")][:5]
    )

    _, demo_summary = update_demo_portfolio(evaluated, qualified, watchlist_tickers, demo_mode, initial_capital)
    eval_map = {item["ticker"]: item for item in evaluated}
    watchlist_section = build_watchlist_telegram_section(watchlist_tickers, eval_map, target_pct)

    if qualified or is_manual_tickers or watchlist_section:
        msg = format_telegram_alert(qualified, scanned_ok, target_pct_label, is_manual_tickers, demo_summary, watchlist_section)
        send_telegram_message(msg)


if __name__ == "__main__":
    main()
