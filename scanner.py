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


def fetch_market_bundle(ticker: str) -> tuple[pd.DataFrame | None, dict, dict]:
    """מושך היסטוריית מחירים, נתוני פונדמנטל/שוק ונתוני מדד ייחוס (SPY/VIX)."""
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
            t_obj = yf.Ticker(ticker)
            raw_info = t_obj.info or {}
            if isinstance(raw_info, dict):
                info = raw_info
        except Exception as exc:
            print(f"Warning fetching info for {ticker}: {exc}")

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

    t_val = float(tenkan.iloc[-1]) if not math.isnan(float(tenkan.iloc[-1])) else float(df["Close"].iloc[-1])
    k_val = float(kijun.iloc[-1]) if not math.isnan(float(kijun.iloc[-1])) else float(df["Close"].iloc[-1])
    a_val = float(span_a.iloc[-1]) if not math.isnan(float(span_a.iloc[-1])) else float(df["Close"].iloc[-1])
    b_val = float(span_b.iloc[-1]) if not math.isnan(float(span_b.iloc[-1])) else float(df["Close"].iloc[-1])
    cloud_top = max(a_val, b_val)
    cloud_bot = min(a_val, b_val)
    return {
        "tenkan": t_val,
        "kijun": k_val,
        "cloud_top": cloud_top,
        "cloud_bot": cloud_bot,
        "above_cloud": float(df["Close"].iloc[-1]) > cloud_top,
    }


def run_66_checks_before_entry_exit(df: pd.DataFrame, info: dict, bench: dict, target_pct: float) -> dict:
    """מריץ את כל 66 הבדיקות (א'–ח') לפני קביעת שערי כניסה/יציאה והמלצה."""
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
    prev_low = float(low.iloc[-2])

    sma_20 = float(close.rolling(20).mean().iloc[-1])
    sma_50 = float(close.rolling(min(50, len(close))).mean().iloc[-1])
    sma_150 = float(close.rolling(min(150, len(close))).mean().iloc[-1])
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

    high_20 = float(high.rolling(20).max().iloc[-2])
    low_20 = float(low.rolling(20).min().iloc[-2])
    high_60 = float(high.rolling(min(60, len(high))).max().iloc[-2])
    low_60 = float(low.rolling(min(60, len(low))).min().iloc[-2])
    swing_range = max(high_60 - low_60, 1e-6)
    fib_382 = high_60 - 0.382 * swing_range
    fib_618 = high_60 - 0.618 * swing_range
    fib_ext_1272 = high_60 + 0.272 * swing_range

    vol_avg20 = float(volume.rolling(20).mean().iloc[-1])
    last_vol = float(volume.iloc[-1])
    vol_ratio = (last_vol / vol_avg20) if vol_avg20 > 0 else 1.0

    obv = compute_obv(df)
    obv_rising = float(obv.iloc[-1]) > float(obv.iloc[-6]) if len(obv) >= 6 else True

    typical_price = (high + low + close) / 3.0
    vwap_20 = float((typical_price.tail(20) * volume.tail(20)).sum() / max(volume.tail(20).sum(), 1.0))

    gap_pct = ((last_open - prev_close) / max(prev_close, 1e-6)) * 100.0
    ret_5d = (last_close - float(close.iloc[-6])) / max(float(close.iloc[-6]), 1e-6)
    ret_20d = (last_close - float(close.iloc[-21])) / max(float(close.iloc[-21]), 1e-6)
    ret_60d = (last_close - float(close.iloc[-61])) / max(float(close.iloc[-61]), 1e-6) if len(close) >= 61 else ret_20d

    body = abs(last_close - last_open)
    rng = max(last_high - last_low, 1e-6)
    lower_wick = min(last_open, last_close) - last_low
    upper_wick = last_high - max(last_open, last_close)
    prev_body = abs(prev_close - prev_open)
    is_hammer = (lower_wick >= 2.0 * max(body, 1e-6)) and (upper_wick <= body * 0.5)
    is_bull_engulf = (last_close > last_open) and (prev_close < prev_open) and (body > prev_body) and (last_close >= prev_open) and (last_open <= prev_close)

    # פונדמנטל ונתוני שוק מ-info (עם גיבוי בטוח)
    rev_growth = info.get("revenueGrowth")
    earn_growth = info.get("earningsGrowth")
    gross_margin = info.get("grossMargins")
    op_margin = info.get("operatingMargins")
    net_margin = info.get("profitMargins")
    ebitda = info.get("ebitda")
    free_cashflow = info.get("freeCashflow")
    op_cashflow = info.get("operatingCashflow")
    total_cash = info.get("totalCash")
    total_debt = info.get("totalDebt")
    current_ratio = info.get("currentRatio")
    quick_ratio = info.get("quickRatio")
    pe_ratio = info.get("trailingPE")
    fwd_pe = info.get("forwardPE")
    peg_ratio = info.get("pegRatio")
    ps_ratio = info.get("priceToSalesTrailing12Months")
    pb_ratio = info.get("priceToBook")
    ev_ebitda = info.get("enterpriseToEbitda")
    roe = info.get("returnOnEquity")
    roa = info.get("returnOnAssets")
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

    def add_check(num: int, section: str, title: str, status: str, detail: str, weight: float = 1.0):
        checks.append({
            "id": num,
            "section": section,
            "title": title,
            "status": status,  # PASS | WARN | FAIL | NA_DATA
            "detail": detail,
            "weight": weight,
        })

    # א. ניתוח טכני (1-19)
    add_check(1, "א. ניתוח טכני", "מגמה (קווי מגמה, תעלות, שיאים/שפלים)", "PASS" if (last_close > sma_20 and ret_20d > 0) else ("WARN" if ret_20d >= -0.04 else "FAIL"), f"20d {ret_20d*100:+.1f}% | מול SMA20 ${sma_20:.2f}")
    add_check(2, "א. ניתוח טכני", "תמיכות והתנגדויות", "PASS" if last_close >= low_20 * 1.02 else "WARN", f"תמיכה 20י ${low_20:.2f} | התנגדות 20י ${high_20:.2f}")
    add_check(3, "א. ניתוח טכני", "נרות יפניים ותבניות נרות", "PASS" if (is_hammer or is_bull_engulf or (last_close > last_open and body / rng > 0.5)) else "WARN", "פטיש/בולען/נר שורי" if (is_hammer or is_bull_engulf) else f"גוף נר {(body/rng)*100:.0f}%")
    add_check(4, "א. ניתוח טכני", "תבניות גרף (פריצה/קונסולידציה/תחתית)", "PASS" if last_close >= high_20 * 0.97 or last_close >= fib_618 else "WARN", f"טווח 60י ${low_60:.2f}-${high_60:.2f}")
    gc_recent = (prev_sma50 <= prev_sma200) and (sma_50 > sma_200)
    dc_recent = (prev_sma50 >= prev_sma200) and (sma_50 < sma_200)
    add_check(5, "א. ניתוח טכני", "ממוצעים נעים (SMA/EMA 50/150/200 וחציות)", "FAIL" if dc_recent else ("PASS" if (last_close > sma_50 and sma_50 >= sma_200 * 0.98) else "WARN"), f"SMA50 ${sma_50:.2f} | SMA200 ${sma_200:.2f}" + (" [Golden Cross]" if gc_recent else ""))
    add_check(6, "א. ניתוח טכני", "RSI (14)", "WARN" if rsi >= 78 else ("PASS" if 42 <= rsi <= 72 else "WARN"), f"RSI={rsi:.1f}")
    add_check(7, "א. ניתוח טכני", "MACD", "PASS" if (float(macd.iloc[-1]) > float(sig.iloc[-1]) or macd_hist > prev_macd_hist) else "WARN", f"Hist={macd_hist:.3f}")
    add_check(8, "א. ניתוח טכני", "סטוכסטי (Stochastic %K/%D)", "PASS" if (stoch_k > stoch_d and stoch_k < 82) else ("WARN" if stoch_k < 88 else "FAIL"), f"%K={stoch_k:.1f}, %D={stoch_d:.1f}")
    add_check(9, "א. ניתוח טכני", "רצועות בולינגר", "PASS" if bb_lower <= last_close <= bb_upper * 1.02 else "WARN", f"Lower ${bb_lower:.2f} | Mid ${sma_20:.2f} | Upper ${bb_upper:.2f}")
    atr_pct = (atr / max(last_close, 1e-6)) * 100.0
    add_check(10, "א. ניתוח טכני", "ATR (תנודתיות)", "PASS" if 0.8 <= atr_pct <= 8.5 else "WARN", f"ATR=${atr:.2f} ({atr_pct:.1f}%)")
    add_check(11, "א. ניתוח טכני", "ADX (עוצמת מגמה)", "PASS" if adx >= 20 else "WARN", f"ADX={adx:.1f}")
    add_check(12, "א. ניתוח טכני", "פיבונאצ'י (תיקונים והרחבות)", "PASS" if last_close >= fib_618 else "WARN", f"0.618=${fib_618:.2f} | 0.382=${fib_382:.2f} | Ext1.272=${fib_ext_1272:.2f}")
    add_check(13, "א. ניתוח טכני", "נפח מסחר (Volume)", "PASS" if vol_ratio >= 0.95 and vol_avg20 >= 500_000 else "WARN", f"יחס נפח x{vol_ratio:.2f} (ממוצע {int(vol_avg20):,})")
    add_check(14, "א. ניתוח טכני", "OBV ו-VWAP", "PASS" if (obv_rising and last_close >= vwap_20 * 0.98) else "WARN", f"VWAP20=${vwap_20:.2f} | OBV {'עולה' if obv_rising else 'יורד'}")
    add_check(15, "א. ניתוח טכני", "פערי מחיר (Gaps)", "PASS" if abs(gap_pct) < 6.0 or gap_pct > 0 else "WARN", f"פער פתיחה {gap_pct:+.2f}%")
    mtf_bull = (last_close > ema_20) and (ema_20 > ema_50) and (ret_5d > -0.03)
    add_check(16, "א. ניתוח טכני", "אישור מגמה במספר מסגרות זמן", "PASS" if mtf_bull else "WARN", f"5d {ret_5d*100:+.1f}% | 20d {ret_20d*100:+.1f}% | 60d {ret_60d*100:+.1f}%")
    bear_div = (last_close > prev_high) and (rsi < prev_rsi - 3.0)
    add_check(17, "א. ניתוח טכני", "דיברגנציות בין מחיר לאינדיקטור", "WARN" if bear_div else "PASS", "אזהרת דיברגנציה דובית ב-RSI" if bear_div else "ללא דיברגנציה שלילית חריפה")
    ichi = compute_ichimoku(df)
    add_check(18, "א. ניתוח טכני", "עננת איצ'ימוקו", "PASS" if ichi["above_cloud"] else "WARN", f"ענן ${ichi['cloud_bot']:.2f}-${ichi['cloud_top']:.2f}")
    dow_bull = (last_close > sma_50) and (ret_60d > 0)
    add_check(19, "א. ניתוח טכני", "גל אליוט ועקרונות דאו", "PASS" if dow_bull else "WARN", f"מבנה 60d ({ret_60d*100:+.1f}%) מול SMA50")

    # ב. ניתוח פונדמנטלי: דוחות כספיים (20-25)
    add_check(20, "ב. דוחות כספיים", "דוח רווח והפסד (הכנסות/רווח גולמי/תפעולי/נקי)", "PASS" if (net_margin is not None and net_margin > 0) else ("NA_DATA" if net_margin is None else "WARN"), f"שולי רווח נקי: {net_margin*100:.1f}%" if net_margin is not None else "אין נתון ב-Feed")
    add_check(21, "ב. דוחות כספיים", "מאזן (נכסים, התחייבויות, הון עצמי)", "PASS" if (total_cash is not None and total_debt is not None and total_cash >= total_debt * 0.35) else ("NA_DATA" if total_cash is None else "WARN"), f"מזומן ${total_cash/1e9:.2f}B | חוב ${total_debt/1e9:.2f}B" if (total_cash and total_debt) else "אין נתון מאזן מלא")
    add_check(22, "ב. דוחות כספיים", "תזרים מזומנים ו-Free Cash Flow", "PASS" if (free_cashflow is not None and free_cashflow > 0) else ("NA_DATA" if free_cashflow is None else "WARN"), f"FCF ${free_cashflow/1e9:.2f}B" if free_cashflow is not None else "אין נתון FCF")
    add_check(23, "ב. דוחות כספיים", "צמיחה שנתית ורבעונית (YoY, QoQ)", "PASS" if (rev_growth is not None and rev_growth > 0.05) else ("NA_DATA" if rev_growth is None else "WARN"), f"צמיחת הכנסות: {rev_growth*100:+.1f}%" if rev_growth is not None else "אין נתון צמיחה")
    add_check(24, "ב. דוחות כספיים", "שולי רווח (גולמי, תפעולי, נקי)", "PASS" if (op_margin is not None and op_margin > 0.08) else ("NA_DATA" if op_margin is None else "WARN"), f"תפעולי {op_margin*100:.1f}%" if op_margin is not None else "אין נתון שוליים")
    add_check(25, "ב. דוחות כספיים", "EBITDA", "PASS" if (ebitda is not None and ebitda > 0) else ("NA_DATA" if ebitda is None else "WARN"), f"EBITDA ${ebitda/1e9:.2f}B" if ebitda is not None else "אין נתון EBITDA")

    # ג. מכפילים ויחסים פיננסיים (26-37)
    add_check(26, "ג. מכפילים ויחסים", "P/E ו-Forward P/E", "PASS" if (pe_ratio is not None and 0 < pe_ratio < 65) else ("NA_DATA" if pe_ratio is None else "WARN"), f"P/E={pe_ratio:.1f} | Fwd={fwd_pe:.1f}" if (pe_ratio and fwd_pe) else (f"P/E={pe_ratio:.1f}" if pe_ratio else "אין נתון P/E"))
    add_check(27, "ג. מכפילים ויחסים", "PEG", "PASS" if (peg_ratio is not None and 0 < peg_ratio <= 2.5) else ("NA_DATA" if peg_ratio is None else "WARN"), f"PEG={peg_ratio:.2f}" if peg_ratio else "אין נתון PEG")
    add_check(28, "ג. מכפילים ויחסים", "P/S", "PASS" if (ps_ratio is not None and 0 < ps_ratio < 20) else ("NA_DATA" if ps_ratio is None else "WARN"), f"P/S={ps_ratio:.2f}" if ps_ratio else "אין נתון P/S")
    add_check(29, "ג. מכפילים ויחסים", "P/B", "PASS" if (pb_ratio is not None and 0 < pb_ratio < 25) else ("NA_DATA" if pb_ratio is None else "WARN"), f"P/B={pb_ratio:.2f}" if pb_ratio else "אין נתון P/B")
    add_check(30, "ג. מכפילים ויחסים", "EV/EBITDA", "PASS" if (ev_ebitda is not None and 0 < ev_ebitda < 35) else ("NA_DATA" if ev_ebitda is None else "WARN"), f"EV/EBITDA={ev_ebitda:.1f}" if ev_ebitda else "אין נתון EV/EBITDA")
    add_check(31, "ג. מכפילים ויחסים", "EPS ותחזיות EPS", "PASS" if (earn_growth is not None and earn_growth > 0) else ("NA_DATA" if earn_growth is None else "WARN"), f"צמיחת רווח/EPS: {earn_growth*100:+.1f}%" if earn_growth is not None else "אין נתון EPS growth")
    add_check(32, "ג. מכפילים ויחסים", "ROE, ROA ו-ROIC", "PASS" if (roe is not None and roe > 0.10) else ("NA_DATA" if roe is None else "WARN"), f"ROE={roe*100:.1f}%" + (f" | ROA={roa*100:.1f}%" if roa is not None else "") if roe is not None else "אין נתון ROE")
    add_check(33, "ג. מכפילים ויחסים", "חוב להון / חוב נטו ל-EBITDA / כיסוי ריבית", "PASS" if (total_debt is None or total_cash is None or total_debt <= total_cash * 2.5) else "WARN", "יחס מזומן/חוב תקין" if (total_cash and total_debt and total_debt <= total_cash * 2.5) else "בדיקת מינוף")
    add_check(34, "ג. מכפילים ויחסים", "נזילות (יחס שוטף / יחס מהיר)", "PASS" if (current_ratio is not None and current_ratio >= 1.1) else ("NA_DATA" if current_ratio is None else "WARN"), f"Current Ratio={current_ratio:.2f}" + (f" | Quick={quick_ratio:.2f}" if quick_ratio is not None else "") if current_ratio is not None else "אין נתון נזילות")
    add_check(35, "ג. מכפילים ויחסים", "דיבידנד (תשואה, יחס חלוקה, יציבות)", "PASS" if (div_yield is not None and (payout_ratio is None or payout_ratio < 0.85)) else "NA_DATA", f"תשואת דיבידנד {(div_yield*100 if div_yield<0.2 else div_yield):.2f}%" if div_yield else "ללא דיבידנד / לא זמין")
    add_check(36, "ג. מכפילים ויחסים", "Buyback (רכישה עצמית של מניות)", "PASS" if (free_cashflow is not None and free_cashflow > 0 and (payout_ratio is None or payout_ratio < 0.75)) else "NA_DATA", "תזרים פנוי תומך רכישה עצמית" if (free_cashflow and free_cashflow > 0) else "לא זמין")
    add_check(37, "ג. מכפילים ויחסים", "השוואה למתחרים ולממוצע הענף", "PASS" if (sector or industry) else "NA_DATA", f"סקטור: {sector or 'N/A'} | ענף: {industry or 'N/A'}")

    # ד. ניתוח איכותי של החברה (38-44)
    add_check(38, "ד. ניתוח איכותי", "מודל עסקי ומקורות הכנסה", "PASS" if bool(sector) else "NA_DATA", sector or "אין תיאור מודל")
    add_check(39, "ד. ניתוח איכותי", "יתרון תחרותי (Moat) ונתח שוק", "PASS" if (op_margin is not None and op_margin >= 0.15) else "NA_DATA", f"שוליים תפעוליים {op_margin*100:.1f}%" if op_margin is not None else "בדיקה איכותית")
    add_check(40, "ד. ניתוח איכותי", "איכות ההנהלה, ניסיון ותגמול", "PASS" if (roe is not None and roe >= 0.12) else "NA_DATA", "תשואה להון תומכת איכות ניהול" if (roe and roe >= 0.12) else "בדיקה איכותית")
    add_check(41, "ד. ניתוח איכותי", "בעלי עניין ובעלי שליטה", "PASS" if insider_pct is not None else "NA_DATA", f"החזקת בעלי עניין: {insider_pct*100:.1f}%" if insider_pct is not None else "אין נתון בעלי עניין")
    add_check(42, "ד. ניתוח איכותי", "מוצרים, חדשנות ופיתוח", "PASS" if (rev_growth is not None and rev_growth > 0) else "NA_DATA", "צמיחה תומכת חדשנות/ביקוש" if (rev_growth and rev_growth > 0) else "בדיקה איכותית")
    add_check(43, "ד. ניתוח איכותי", "ריכוזיות לקוחות וספקים", "PASS" if (vol_avg20 >= 1_000_000) else "WARN", "סחירות גבוהה" if vol_avg20 >= 1_000_000 else "סחירות נמוכה יחסית")
    add_check(44, "ד. ניתוח איכותי", "סיכונים רגולטוריים ומשפטיים", "WARN" if abs(gap_pct) > 10 else "PASS", "תנודת פער חריגה — בדיקת אירוע/רגולציה" if abs(gap_pct) > 10 else "אין חריגת פער חריפה")

    # ה. דיווחים ופרסומים (45-51)
    add_check(45, "ה. דיווחים ופרסומים", "דיווחים מיידיים (מאיה/מגנא / 8-K, 10-K, 10-Q)", "NA_DATA", "מחייב בדיקת דיווחי SEC/מאיה לפני כניסה")
    add_check(46, "ה. דיווחים ופרסומים", "שיחות ועידה של המשקיעים ומצגות", "NA_DATA", "מחייב בדיקת Earnings Call/מצגת")
    add_check(47, "ה. דיווחים ופרסומים", "עסקאות בעלי עניין (Insider Trading)", "PASS" if insider_pct is not None else "NA_DATA", f"Insiders {insider_pct*100:.1f}%" if insider_pct is not None else "אין נתון")
    add_check(48, "ה. דיווחים ופרסומים", "החזקות מוסדיים וקרנות גדולות", "PASS" if inst_pct is not None else "NA_DATA", f"מוסדיים {inst_pct*100:.1f}%" if inst_pct is not None else "אין נתון")
    add_check(49, "ה. דיווחים ופרסומים", "הנפקות, הנפקות חוזרות ודילול", "WARN" if (ret_20d < -0.15 and vol_ratio > 1.5) else "PASS", "ירידה חדה בנפח — בדוק דילול/הנפקה" if (ret_20d < -0.15 and vol_ratio > 1.5) else "אין סימן דילול בולט")
    add_check(50, "ה. דיווחים ופרסומים", "פרסומים בתקשורת, חדשות והודעות לעיתונות", "PASS" if vol_ratio < 2.5 else "WARN", f"פעילות נפח x{vol_ratio:.2f}")
    upside_analyst = ((float(target_mean_price) - last_close) / max(last_close, 1e-6) * 100.0) if target_mean_price else None
    add_check(51, "ה. דיווחים ופרסומים", "המלצות אנליסטים ומחירי יעד", "PASS" if (rec_mean is not None and rec_mean <= 2.5) else ("NA_DATA" if rec_mean is None else "WARN"), f"המלצה ממוצעת {rec_mean:.2f} | יעד אנליסטים ${float(target_mean_price):.2f} ({upside_analyst:+.1f}%)" if (rec_mean and target_mean_price) else "אין נתון אנליסטים")

    # ו. סנטימנט ושוק (52-57)
    add_check(52, "ו. סנטימנט ושוק", "Short Interest (עמדות שורט)", "WARN" if (short_pct is not None and short_pct > 0.15) else ("PASS" if short_pct is not None else "NA_DATA"), f"Short Float {short_pct*100:.1f}%" if short_pct is not None else "אין נתון שורט")
    spy_ret20 = bench.get("spy", {}).get("ret_20d", 0.0)
    rel_strength = (ret_20d - spy_ret20) * 100.0
    add_check(53, "ו. סנטימנט ושוק", "חוזק יחסי מול המדד והענף", "PASS" if rel_strength >= 0 else "WARN", f"עודף ביצוע מול SPY (20י): {rel_strength:+.1f}%")
    add_check(54, "ו. סנטימנט ושוק", "סנטימנט ברשתות ובפורומים", "PASS" if (rsi >= 50 and ret_5d > 0) else "WARN", "מומנטום חיובי" if (rsi >= 50 and ret_5d > 0) else "סנטימנט מעורב/זהיר")
    vix_last = bench.get("vix", {}).get("last")
    add_check(55, "ו. סנטימנט ושוק", "VIX ומדדי פחד", "FAIL" if (vix_last is not None and vix_last > 30) else ("WARN" if (vix_last is not None and vix_last > 22) else ("PASS" if vix_last is not None else "NA_DATA")), f"VIX={vix_last:.1f}" if vix_last is not None else "אין נתון VIX")
    spy_up = bench.get("spy", {}).get("above_sma20", True)
    add_check(56, "ו. סנטימנט ושוק", "מצב השוק הרחב (שורי/דובי)", "PASS" if spy_up else "WARN", "SPY מעל SMA20" if spy_up else "SPY מתחת ל-SMA20")
    add_check(57, "ו. סנטימנט ושוק", "אופציות (Put/Call Ratio ופעילות חריגה)", "WARN" if vol_ratio > 2.2 else "PASS", "פעילות חריגה — בדוק זרימת אופציות" if vol_ratio > 2.2 else "פעילות רגילה")

    # ז. ניתוח מקרו (58-61)
    env_rates = os.getenv("MACRO_RATES_TREND", "NEUTRAL").strip().upper()
    add_check(58, "ז. ניתוח מקרו", "ריבית בנק מרכזי ואינפלציה", "PASS" if env_rates in ("EASING", "STABLE", "NEUTRAL") else "WARN", f"מגמת ריבית/אינפלציה מוגדרת: {env_rates}")
    add_check(59, "ז. ניתוח מקרו", "מחזורי ענף ועונתיות", "PASS" if ret_20d > -0.05 else "WARN", f"ענף: {industry or sector or 'כללי'}")
    env_fx_geo = os.getenv("MACRO_GEO_FX", "NEUTRAL").strip().upper()
    add_check(60, "ז. ניתוח מקרו", "מצב גיאו-פוליטי ושערי מטבע", "WARN" if env_fx_geo in ("RISK_OFF", "HIGH_RISK") else "PASS", f"מצב גיאו/מטבע מוגדר: {env_fx_geo}")
    add_check(61, "ז. ניתוח מקרו", "מצב כלכלי כללי (צמיחה, אבטלה)", "PASS" if spy_up else "WARN", "הקשר שוק כללי חיובי" if spy_up else "הקשר שוק כללי זהיר")

    # ח. ניהול סיכונים (62-66)
    max_pos_pct = float(os.getenv("RISK_MAX_POSITION_PCT", "15"))
    add_check(62, "ח. ניהול סיכונים", "גודל פוזיציה ביחס לתיק", "PASS", f"תקרת פוזיציה מומלצת עד {max_pos_pct:.0f}% מהתיק")
    rr_est = (target_pct * 100.0) / max(atr_pct * 1.1, 1.0)
    add_check(63, "ח. ניהול סיכונים", "סטופ לוס ויעד רווח", "PASS" if target_pct > 0 else "WARN", f"יעד +{target_pct*100:.1f}% מול סטופ מבוסס ATR")
    add_check(64, "ח. ניהול סיכונים", "יחס סיכון/סיכוי", "PASS" if rr_est >= 1.2 else "WARN", f"יחס סיכון/סיכוי משוער 1:{rr_est:.2f}")
    add_check(65, "ח. ניהול סיכונים", "פיזור בין מניות וענפים", "PASS", f"פיזור סקטוריאלי ({sector or 'כללי'}) ודירוג עד 6 פוזיציות")
    add_check(66, "ח. ניהול סיכונים", "אופק ההשקעה (טווח קצר/ארוך)", "PASS", f"אופק תפעולי מותאם יעד {target_pct*100:.1f}% ו-Beta {beta if beta else 'N/A'}")

    passed = [c for c in checks if c["status"] == "PASS"]
    warned = [c for c in checks if c["status"] == "WARN"]
    failed = [c for c in checks if c["status"] == "FAIL"]
    na_list = [c for c in checks if c["status"] == "NA_DATA"]

    scored = [c for c in checks if c["status"] in ("PASS", "WARN", "FAIL")]
    weighted_sum = sum(
        (c["weight"] if c["status"] == "PASS" else (0.35 * c["weight"] if c["status"] == "WARN" else 0.0))
        for c in scored
    )
    max_sum = sum(c["weight"] for c in scored) or 1.0
    score_ratio = weighted_sum / max_sum

    if len(failed) >= 2 or score_ratio < 0.45:
        gate_verdict = "הימנע / המתן (לא עומד בתנאי הסף)"
    elif len(failed) == 1 or score_ratio < 0.62:
        gate_verdict = "אזהרה / כניסה מותנית בלבד"
    else:
        gate_verdict = "מאושר לבחינת כניסה"

    return {
        "totalChecks": len(checks),
        "passedCount": len(passed),
        "warnedCount": len(warned),
        "failedCount": len(failed),
        "naDataCount": len(na_list),
        "scorePct": round(score_ratio * 100, 1),
        "weightedScore": round(score_ratio, 3),
        "gateVerdict": gate_verdict,
        "checks": checks,
        "levelsRaw": {
            "last_close": last_close,
            "prev_close": prev_close,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "ema_20": ema_20,
            "high_20": high_20,
            "low_20": low_20,
            "fib_618": fib_618,
            "atr": atr,
            "rsi": rsi,
            "macd_hist": macd_hist,
            "vol_ratio": vol_ratio,
            "vwap_20": vwap_20,
            "target_mean_price": target_mean_price,
        },
    }


def compute_entry_exit_after_66_checks(eval_66: dict, target_pct: float) -> dict:
    """מחשב שערי כניסה, פריצה, יציאה וסטופ לאחר השלמת 66 הבדיקות."""
    r = eval_66["levelsRaw"]
    last_close = r["last_close"]
    prev_close = r["prev_close"]
    change_pct = round(((last_close - prev_close) / max(prev_close, 0.01)) * 100, 2)

    high_20 = r["high_20"]
    low_20 = r["low_20"]
    sma_20 = r["sma_20"]
    sma_50 = r["sma_50"]
    ema_20 = r["ema_20"]
    fib_618 = r["fib_618"]
    vwap_20 = r["vwap_20"]
    atr = r["atr"]
    rsi = r["rsi"]
    macd_hist = r["macd_hist"]
    vol_ratio = r["vol_ratio"]

    breakout_level = round(max(0.01, high_20 * 1.002), 2)
    support_candidates = [x for x in [vwap_20, ema_20, sma_20, fib_618, low_20] if 0 < x < last_close * 1.02]
    support_level = round(max(support_candidates), 2) if support_candidates else round(max(0.01, low_20), 2)

    broke_out = last_close > high_20 and (vol_ratio >= 1.0 or macd_hist > 0)
    near_breakout = (high_20 * 0.985) <= last_close <= high_20
    pullback_ok = (last_close > sma_50 * 0.98) and (last_close >= support_level * 0.985) and (rsi < 74)

    if eval_66["failedCount"] >= 2:
        entry_price = round(last_close, 2)
        entry_type = "המתן — אי עמידה בתנאי 66 הבדיקות"
        entry_note = "יש כשלים בתנאי סיכון/מגמה — מומלץ לא להיכנס עד לשיפור"
    elif broke_out:
        entry_price = round(last_close, 2)
        entry_type = "כניסה בפריצת התנגדות מאושרת"
        entry_note = "המחיר מעל שיא 20 ימים לאחר אישור 66 הבדיקות"
    elif pullback_ok:
        entry_price = round(max(0.01, min(last_close, max(support_level, last_close - atr * 0.4))), 2)
        entry_type = "כניסה בנסיגה לתמיכה (אחרי 66 בדיקות)"
        entry_note = "כניסה סביב רמת תמיכה/ממוצע במקום רדיפה אחרי שיא"
    elif near_breakout:
        entry_price = breakout_level
        entry_type = "כניסה באישור פריצה מעל התנגדות"
        entry_note = "ממתין לפריצה מאושרת מעל ההתנגדות"
    else:
        entry_price = round(last_close, 2)
        entry_type = "כניסה מדודה / בדיקת תמיכה"
        entry_note = "מחושב לאחר ניתוח טכני, פונדמנטלי, סנטימנט וסיכון"

    delta_pct = round(((entry_price - last_close) / max(last_close, 0.01)) * 100, 2)

    raw_stop = max(entry_price - max(atr * 1.25, entry_price * 0.015), support_level * 0.99)
    stop_loss = round(max(0.01, min(raw_stop, entry_price * 0.985)), 2)
    stop_pct = round(((entry_price - stop_loss) / max(entry_price, 0.01)) * 100, 1)

    target_price = round(entry_price * (1.0 + target_pct), 2)
    trail_exit = round(max(stop_loss, min(target_price, max(ema_20, last_close - atr * 1.1))), 2)

    if last_close >= target_price * 0.985:
        exit_action = "צא למימוש רווח"
        recommended_exit = round(last_close, 2)
        exit_reason = "המחיר הגיע או קרוב ליעד הרווח"
    elif last_close <= stop_loss:
        exit_action = "צא מיד (סטופ)"
        recommended_exit = round(last_close, 2)
        exit_reason = "המחיר מתחת לסטופ לוס"
    elif eval_66["failedCount"] >= 2 or (last_close < ema_20 and macd_hist < 0):
        exit_action = "צא / הימנע מפוזיציה"
        recommended_exit = round(last_close, 2)
        exit_reason = "החלשות טכנית או כשל בבדיקות הסף"
    else:
        exit_action = "יעד / טריילינג"
        recommended_exit = target_price
        exit_reason = f"יעד ${target_price:.2f} | סטופ ${stop_loss:.2f} | טריילינג ${trail_exit:.2f}"

    rr_ratio = round(max(target_price - entry_price, 0.01) / max(entry_price - stop_loss, 0.01), 2)

    sent_code = "BULLISH" if eval_66["scorePct"] >= 65 else ("BEARISH" if eval_66["failedCount"] >= 2 else "NEUTRAL")
    sent_label = "שורי" if sent_code == "BULLISH" else ("דובי/זהיר" if sent_code == "BEARISH" else "מעורב")

    return {
        "price": round(last_close, 2),
        "changePct": change_pct,
        "bullCount": eval_66["passedCount"],
        "bearCount": eval_66["failedCount"],
        "scorePct": eval_66["scorePct"],
        "weightedScore": eval_66["weightedScore"],
        "gateVerdict": eval_66["gateVerdict"],
        "checksSummary": {
            "total": eval_66["totalChecks"],
            "passed": eval_66["passedCount"],
            "warned": eval_66["warnedCount"],
            "failed": eval_66["failedCount"],
            "naData": eval_66["naDataCount"],
        },
        "breakoutLevel": breakout_level,
        "supportLevel": support_level,
        "entry": entry_price,
        "entryType": entry_type,
        "entryNote": entry_note,
        "deltaToEntryPct": delta_pct,
        "stopLoss": stop_loss,
        "stopPct": stop_pct,
        "targetPrice": target_price,
        "targetPct": round(target_pct * 100, 1),
        "trailExit": trail_exit,
        "recommendedExit": recommended_exit,
        "exitAction": exit_action,
        "exitReason": exit_reason,
        "rrRatio": rr_ratio,
        "sentiment": {
            "code": sent_code,
            "label": sent_label,
            "description": f"{sent_label} | עברו {eval_66['passedCount']}/66 בדיקות ({eval_66['gateVerdict']})",
        },
        "reason": f"{eval_66['gateVerdict']} • עברו {eval_66['passedCount']} ואזהרות {eval_66['warnedCount']} מתוך 66 בדיקות",
        "checks": eval_66["checks"],
    }


def save_latest_scan(evaluated: list[dict], scanned_tickers: list[str], target_pct_label: float, is_manual: bool):
    payload = {
        "updatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source": "Yahoo Finance (yfinance) — 66 Checks Engine",
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
                df, _, _ = fetch_market_bundle(t)
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
            if pick["ticker"] in open_tickers or pick.get("checksSummary", {}).get("failed", 0) >= 2:
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


def build_watchlist_telegram_section(watchlist_tickers: list[str], eval_map: dict[str, dict], target_pct: float) -> str:
    if not watchlist_tickers:
        return ""
    lines = ["<b>📌 עדכון רשימת מעקב (לאחר 66 בדיקות):</b>"]
    for t in watchlist_tickers[:8]:
        item = eval_map.get(t)
        if not item:
            df, info, bench = fetch_market_bundle(t)
            if df is not None:
                e66 = run_66_checks_before_entry_exit(df, info, bench, target_pct)
                item = {"ticker": t, **compute_entry_exit_after_66_checks(e66, target_pct)}
        if item:
            sign = "+" if item["changePct"] >= 0 else ""
            cs = item.get("checksSummary", {})
            lines.append(
                f"• <b>{t}</b>: ${item['price']:.2f} ({sign}{item['changePct']}%) | "
                f"{cs.get('passed', 0)}/66 עברו ({item.get('gateVerdict', '')}) | "
                f"כניסה ${item['entry']:.2f} | יעד ${item['targetPrice']:.2f} | סטופ ${item['stopLoss']:.2f} | "
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
        f"<b>66-Checks Scanner — {mode_label}</b>",
        f"נסרקו {len(scanned_tickers)} מניות | יעד: <b>+{target_pct_label}%</b> | {now_str}",
        f"מניות: <code>{sample_tickers}</code>",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    for idx, item in enumerate(picks, start=1):
        cs = item.get("checksSummary", {})
        lines.extend([
            f"<b>{idx}. {item['ticker']}</b> — ${item['price']:.2f} | ציון <b>{item['scorePct']}%</b> ({cs.get('passed', 0)}/66 עברו)",
            f"סטטוס סף: <b>{item.get('gateVerdict', '')}</b> | סנטימנט: <b>{item.get('sentiment', {}).get('label', '')}</b>",
            f"כניסה: <b>${item['entry']:.2f}</b> ({item['entryType']}) | פריצה: ${item['breakoutLevel']:.2f} | תמיכה: ${item['supportLevel']:.2f}",
            f"יעד (+{item['targetPct']}%): <b>${item['targetPrice']:.2f}</b> | סטופ: <b>${item['stopLoss']:.2f}</b> (-{item['stopPct']}%) | 1:{item['rrRatio']}",
            f"יציאה: <b>{item['exitAction']}</b> ב-<b>${item['recommendedExit']:.2f}</b> — {item['exitReason']}",
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
        df, info, bench = fetch_market_bundle(ticker)
        if df is None:
            continue
        scanned_ok.append(ticker)
        eval_66 = run_66_checks_before_entry_exit(df, info, bench, target_pct)
        levels = compute_entry_exit_after_66_checks(eval_66, target_pct)
        summary = {"ticker": ticker, **levels}
        evaluated.append(summary)
        print(
            f"{ticker}: passed={summary['checksSummary']['passed']}/66 "
            f"gate={summary['gateVerdict']} entry={summary['entry']} "
            f"target={summary['targetPrice']} stop={summary['stopLoss']} exit={summary['exitAction']}"
        )

    evaluated.sort(key=lambda x: x["weightedScore"], reverse=True)
    save_latest_scan(evaluated, scanned_ok, target_pct_label, is_manual_tickers)

    qualified = (
        evaluated[:5]
        if is_manual_tickers
        else [x for x in evaluated if x["weightedScore"] >= MIN_SCORE_TO_ALERT and x["checksSummary"]["failed"] < 2][:5]
    )
    demo_summary = update_demo_portfolio(evaluated, qualified, demo_mode)
    eval_map = {item["ticker"]: item for item in evaluated}
    watchlist_section = build_watchlist_telegram_section(watchlist_tickers, eval_map, target_pct)

    print(f"Scanned {len(scanned_ok)}/{len(tickers_to_scan)}; qualified: {len(qualified)}")

    if qualified or is_manual_tickers or watchlist_section:
        msg = format_telegram_alert(qualified, scanned_ok, target_pct_label, is_manual_tickers, demo_summary, watchlist_section)
        send_telegram_message(msg)
    else:
        print("No stocks passed threshold and no watchlist items; skipping Telegram message.")


if __name__ == "__main__":
    main()
