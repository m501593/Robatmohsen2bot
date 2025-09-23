# main.py
import time
import threading
import logging
import math
import json
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import ccxt
import requests

from telegram import Bot

# -------------------------
# CONFIG (put your values)
# -------------------------
TELEGRAM_TOKEN = "7993216439:AAHKSHJMHrcnfEcedw54aetp1JPxZ83Ks4M"
ADMIN_CHAT_ID = 84544682  # integer

WATCHLIST = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "XRP/USDT",
]

TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]

SCAN_INTERVAL_SECONDS = 60
PROBABILITY_THRESHOLD = 50  # example threshold
AUTO_SCAN_TOP_N = 100

DB_PATH = "signals.db"  # optional for future
LOG_FILE = "bot.log"

# create bot
bot = Bot(token=TELEGRAM_TOKEN)

# Logging
logging.basicConfig(level=logging.INFO, filename=LOG_FILE,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("scanner")

# -------------------------
# EXCHANGE CLIENTS (ccxt)
# -------------------------
exchanges = {}
for name in ["binance", "bybit", "kucoin"]:
    try:
        ex = getattr(ccxt, name)({
            'enableRateLimit': True,
            # replay protection / options can be added
        })
        # for futures symbols mapping we may need exchange-specific adjustments
        exchanges[name] = ex
    except Exception as e:
        logger.warning(f"can't init {name}: {e}")

def fetch_ohlcv(symbol, timeframe, limit=200):
    """Try Binance first, fallback to other exchanges via ccxt."""
    # ccxt symbol format sometimes uses 'BTC/USDT' same as WATCHLIST
    for name in ["binance", "bybit", "kucoin"]:
        ex = exchanges.get(name)
        if not ex:
            continue
        try:
            # Some exchanges require different market id for futures. We'll try a general approach.
            if timeframe not in ex.timeframes and hasattr(ex, 'timeframes'):
                # if exchange doesn't support timeframe, skip
                pass
            ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
            df['ts'] = pd.to_datetime(df['ts'], unit='ms')
            return df
        except Exception as e:
            logger.debug(f"fetch_ohlcv failed {name} {symbol} {timeframe}: {e}")
            continue
    raise RuntimeError(f"All exchanges failed for {symbol} {timeframe}")

# -------------------------
# INDICATORS (lightweight)
# -------------------------
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(window=period).mean()
    down = -delta.clip(upper=0).rolling(window=period).mean()
    rs = up / (down + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# -------------------------
# SCORER (simple mapping of rules)
# -------------------------
def score_for_symbol(symbol):
    """Compute score across multiple timeframes and indicators; return details dict."""
    details = {}
    total_score = 0
    max_score = 0

    for tf in TIMEFRAMES:
        try:
            df = fetch_ohlcv(symbol, tf, limit=200)
        except Exception as e:
            logger.warning(f"ohlcv fail {symbol} {tf}: {e}")
            continue

        close = df['close']
        high = df['high']
        low = df['low']

        # Indicators
        ema50 = ema(close, 50).iloc[-1]
        ema200 = ema(close, 200).iloc[-1] if len(close) >= 200 else ema(close, min(200, len(close))).iloc[-1]
        rsi_val = rsi(close).iloc[-1]
        macd_line, signal_line, hist = macd(close)
        macd_cross = 1 if macd_line.iloc[-1] > signal_line.iloc[-1] else -1
        atr_val = atr(df).iloc[-1] if len(df) > 14 else np.nan
        vol = df['volume'].iloc[-1]
        vol_mean20 = df['volume'].rolling(20).mean().iloc[-1] if len(df) > 20 else df['volume'].mean()

        # scoring weights per timeframe (example)
        # Trend: ema50 vs ema200
        trend_score = 0
        if close.iloc[-1] >= ema200:
            trend_score = 15
        elif abs(close.iloc[-1] - ema200)/ema200 < 0.02:
            trend_score = 7
        else:
            trend_score = 0

        # Structure (simple HH/HL)
        structure_score = 0
        lookback = 6
        if len(high) >= lookback:
            highs = high[-lookback:]
            lows = low[-lookback:]
            if highs.iloc[-1] > highs.iloc[:-1].max() and lows.iloc[-1] > lows.iloc[:-1].min():
                structure_score = 12
            else:
                structure_score = 6

        # Volume
        volume_score = 10 if vol > (vol_mean20 if not np.isnan(vol_mean20) else 0) else 3

        # RSI
        rsi_score = 0
        if rsi_val > 55:
            rsi_score = 8
        elif rsi_val < 45:
            rsi_score = 8  # for shorts
        else:
            rsi_score = 3

        # MACD
        macd_score = 8 if macd_cross == 1 else 0

        # Multi-timeframe alignment simple bonus
        multi_tf_bonus = 0
        # here we just add small bonus for higher TF bullishness (ex: if ema50>ema200)
        if ema50 > ema200:
            multi_tf_bonus = 4

        # sum up for this timeframe
        tf_score = trend_score + structure_score + volume_score + rsi_score + macd_score + multi_tf_bonus
        tf_max = 15 + 12 + 10 + 8 + 8 + 4

        details[tf] = {
            "score": int(tf_score),
            "max": int(tf_max),
            "trend": float(close.iloc[-1] / ema200) if ema200 != 0 else 0,
            "rsi": float(round(rsi_val, 2)),
            "macd_cross": int(macd_cross),
            "atr": float(atr_val) if not np.isnan(atr_val) else None,
            "last_price": float(close.iloc[-1]),
        }

        total_score += tf_score
        max_score += tf_max

    # normalize to 0-100
    normalized = int((total_score / max_score) * 100) if max_score > 0 else 0

    # decide category
    if normalized >= 75:
        category = "STRONG"
    elif normalized >= 60:
        category = "GOOD"
    elif normalized >= 45:
        category = "WEAK"
    else:
        category = "NO_TRADE"

    result = {
        "symbol": symbol,
        "score": normalized,
        "category": category,
        "details": details,
        "total_raw": int(total_score),
        "max_raw": int(max_score)
    }
    return result

# -------------------------
# Message formatting (Persian)
# -------------------------
def format_message(result):
    sym = result['symbol'].replace('/', '')
    score = result['score']
    cat = result['category']
    total_raw = result['total_raw']
    max_raw = result['max_raw']

    # choose primary timeframe summary (1h if exists else first)
    primary_tf = "1h" if "1h" in result['details'] else list(result['details'].keys())[0]
    det = result['details'].get(primary_tf, {})
    last_price = det.get('last_price', None)
    rsi_val = det.get('rsi', None)
    atr_val = det.get('atr', None)

    # Compose Persian message similar to sample
    lines = []
    lines.append(f"جفت ارز: {sym}  | دسته: {cat}")
    lines.append(f"امتیاز کل: {score}% ({total_raw}/{max_raw})")
    lines.append(f"تایم‌فریم: {primary_tf}  | قیمت: {last_price}")
    lines.append(f"اندیکاتورها:")
    lines.append(f" - RSI: {rsi_val}")
    lines.append(f" - ATR: {atr_val}")
    lines.append("")
    lines.append("خلاصه تحلیلی:")
    lines.append(f" وضعیت کلی: {cat}")
    lines.append("")
    lines.append("برای جزئیات بیشتر /status SYMBOL را ارسال کن.")
    return "\n".join(lines)

# -------------------------
# Worker: scan list and send if above threshold
# -------------------------
def scan_and_alert():
    logger.info("scan_and_alert started")
    for symbol in WATCHLIST:
        try:
            result = score_for_symbol(symbol)
        except Exception as e:
            logger.exception(f"scoring failed for {symbol}: {e}")
            continue

        # send if above threshold or category strong/good
        if result['score'] >= PROBABILITY_THRESHOLD or result['category'] in ("STRONG", "GOOD"):
            msg = format_message(result)
            try:
                bot.send_message(chat_id=ADMIN_CHAT_ID, text=msg)
                logger.info(f"alert sent for {symbol}")
            except Exception as e:
                logger.exception(f"telegram send failed: {e}")

# -------------------------
# Scheduler loop
# -------------------------
def scheduler_loop():
    while True:
        try:
            scan_and_alert()
        except Exception as e:
            logger.exception(f"scan loop error: {e}")
            try:
                bot.send_message(chat_id=ADMIN_CHAT_ID, text=f"خطای اسکن: {e}")
            except:
                pass
        time.sleep(SCAN_INTERVAL_SECONDS)

if __name__ == "__main__":
    logger.info("starting main scheduler")
    t = threading.Thread(target=scheduler_loop, daemon=True)
    t.start()
    # keep main alive
    while True:
        time.sleep(10)
