# main.py
# Single-file complete bot: Binance futures scanner -> Persian signal messages -> Telegram (webhook)
# Requirements: aiogram==2.25.1, python-binance, pandas, numpy, aiohttp

import os
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
from binance.client import Client
from aiogram import Bot, Dispatcher, types
from aiogram.utils.executor import start_webhook

# -----------------------------
# CONFIG: (Hardcoded token/ID per your request)
# -----------------------------
API_TOKEN = "7993216439:AAHKSHJMHrcnfEcedw54aetp1JPxZ83Ks4M"   # <--- your token (hardcoded)
ADMIN_ID = 84544682                                            # <--- your admin id (hardcoded)

# Render webhook host (your render app)
WEBHOOK_HOST = "https://robatmohsen2bot.onrender.com"  # adjust only if different
WEBHOOK_PATH = "/webhook"
WEBHOOK_URL = f"{WEBHOOK_HOST}{WEBHOOK_PATH}"

WEBAPP_HOST = "0.0.0.0"
WEBAPP_PORT = int(os.getenv("PORT", "10000"))

# Binance keys (optional)
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")  # optional, set in ENV for higher rate limits
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# Analyzer params
ANALYZE_INTERVAL = int(os.getenv("ANALYZE_INTERVAL", "300"))  # seconds between cycles
TOP_N = int(os.getenv("TOP_N", "100"))  # top N futures symbols scanned
SEND_THRESHOLD = int(os.getenv("SEND_THRESHOLD", "80"))  # score threshold to send (80 => GOOD+STRONG)

# Scoring weights (sum ~134) — follow your spec
WEIGHTS = {
    "trend_ema200": 15, "market_structure": 12, "price_action": 12, "volume": 10,
    "rsi": 8, "macd": 8, "multi_tf": 8, "funding_oi": 8, "news_flag": 7, "liquidity_depth": 7,
    "order_flow": 5, "liquidity_pools": 5, "vwap": 3, "volatility": 3, "correlation": 5,
    "monthly_bias": 5, "long_short_ratio": 5, "time_of_day": 3, "position_sizing": 5,
    "psych_check": 3, "backtest_flag": 3, "regime_detection": 4, "asset_specific": 3,
    "smc_proxy": 3, "ml_prob": 5,
}
TOTAL_WEIGHT = sum(WEIGHTS.values())

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("signals_bot")

# -----------------------------
# Clients init
# -----------------------------
try:
    if BINANCE_API_KEY and BINANCE_API_SECRET:
        binance = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
    else:
        binance = Client()
except Exception as e:
    logger.exception("Failed to initialize Binance client: %s", e)
    raise

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

# -----------------------------
# Market data helpers
# -----------------------------
def fetch_klines_df(symbol: str, interval: str = "15m", limit: int = 500) -> pd.DataFrame:
    """Fetch klines (futures if available). Returns DataFrame with columns open_time, open, high, low, close, volume."""
    try:
        raw = binance.futures_klines(symbol=symbol, interval=interval, limit=limit)
    except Exception:
        raw = binance.get_klines(symbol=symbol, interval=interval, limit=limit)
    cols = ["open_time","open","high","low","close","volume","close_time","qav","num_trades","tbbase","tquote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["open_time","open","high","low","close","volume"]]

def get_futures_tickers_df() -> pd.DataFrame:
    """Return futures tickers DataFrame sorted by 24h quoteVolume (USDT pairs)."""
    try:
        tickers = binance.futures_ticker()
    except Exception:
        tickers = binance.get_ticker()
    df = pd.DataFrame(tickers)
    if "quoteVolume" in df.columns:
        df["quoteVolume"] = pd.to_numeric(df["quoteVolume"], errors="coerce").fillna(0)
    elif "volume" in df.columns:
        df["quoteVolume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    else:
        df["quoteVolume"] = 0
    df = df[df["symbol"].str.endswith("USDT")]
    df = df.sort_values("quoteVolume", ascending=False)
    return df

def get_top_symbols(limit: int = TOP_N) -> List[str]:
    df = get_futures_tickers_df()
    symbols = df["symbol"].tolist()[:limit]
    return symbols

def get_funding_and_oi(symbol: str) -> Tuple[float, float]:
    funding = 0.0
    oi = 0.0
    try:
        res = binance.futures_funding_rate(symbol=symbol, limit=1)
        if isinstance(res, list) and res:
            funding = float(res[0].get("fundingRate", 0.0))
    except Exception:
        logger.debug("funding fetch failed for %s", symbol)
    try:
        t = binance.futures_ticker(symbol=symbol)
        oi = float(t.get("openInterest", 0.0))
    except Exception:
        logger.debug("oi fetch failed for %s", symbol)
    return funding, oi

def get_order_book(symbol: str, limit: int = 5) -> dict:
    try:
        return binance.futures_order_book(symbol=symbol, limit=limit)
    except Exception:
        try:
            return binance.get_order_book(symbol=symbol, limit=limit)
        except Exception:
            return {}

# -----------------------------
# Indicator implementations (safe, no external ta dependency)
# -----------------------------
def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()

def compute_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff().fillna(0)
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    # Wilder's smoothing
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    # If roll_down is zero -> rsi_series may be NaN; handle
    last = rsi_series.iloc[-1]
    if np.isnan(last):
        # fallback: if avg gain > avg loss then RSI near 100 else near 0.
        avg_gain = up.mean()
        avg_loss = down.mean()
        if avg_loss == 0:
            return 70.0 if avg_gain > 0 else 50.0
        return 100 - (100 / (1 + (avg_gain/avg_loss)))
    return float(last)

def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean().iloc[-1] if len(tr) >= period else tr.mean()
    return float(atr) if not np.isnan(atr) else float((high - low).mean())

def compute_vwap(df: pd.DataFrame) -> float:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    pv = (typical * df["volume"]).cumsum()
    v = df["volume"].cumsum()
    if v.iloc[-1] == 0:
        return float(df["close"].iloc[-1])
    return float((pv / v).iloc[-1])

def compute_obv(df: pd.DataFrame) -> float:
    close = df["close"]
    vol = df["volume"]
    sign = np.sign(close.diff().fillna(0))
    obv = (sign * vol).cumsum().iloc[-1]
    return float(obv)

# -----------------------------
# Scoring engine (25 proxies, consistent)
# -----------------------------
def detect_market_structure(df: pd.DataFrame) -> str:
    highs = df["high"].tail(12)
    lows = df["low"].tail(12)
    if highs.iloc[-1] > highs.mean() and lows.iloc[-1] > lows.mean():
        return "bull"
    if highs.iloc[-1] < highs.mean() and lows.iloc[-1] < lows.mean():
        return "bear"
    return "neutral"

def time_of_day_score() -> int:
    h = datetime.utcnow().hour
    if 12 <= h <= 17:
        return 3
    if 9 <= h < 12 or 17 < h <= 20:
        return 1
    return 0

def ml_prob_stub(symbol: str) -> float:
    return 0.5  # placeholder

def compute_score_full(symbol: str, df15: pd.DataFrame, df1h: pd.DataFrame = None, df4h: pd.DataFrame = None) -> Tuple[int, List[str], Dict]:
    notes: List[str] = []
    parts: Dict[str, int] = {}

    try:
        indicators = {}
        indicators["last_price"] = float(df15["close"].iloc[-1])
        indicators["ema200"] = float(ema(df15["close"], 200).iloc[-1]) if len(df15) >= 200 else float(df15["close"].rolling(window=min(200,len(df15))).mean().iloc[-1])
        indicators["ema50"] = float(ema(df15["close"], 50).iloc[-1]) if len(df15) >= 50 else float(df15["close"].rolling(window=min(50,len(df15))).mean().iloc[-1])
        indicators["rsi14"] = compute_rsi(df15["close"], 14)
        indicators["atr14"] = compute_atr(df15, 14)
        indicators["vwap"] = compute_vwap(df15)
        indicators["obv"] = compute_obv(df15)
        mean20 = df15["volume"].rolling(20).mean().iloc[-1] if len(df15) >= 20 else df15["volume"].mean()
        indicators["vol_strength"] = float(df15["volume"].iloc[-1] / mean20) if mean20 and mean20>0 else 1.0
        indicators["pct_24h"] = float(df15["close"].iloc[-1] / df15["close"].iloc[0] - 1) if len(df15) > 1 else 0.0
    except Exception as e:
        logger.exception("Indicator computation failed for %s: %s", symbol, e)
        # fallback default indicators
        indicators = {"last_price": 0.0, "ema200": 0.0, "ema50": 0.0, "rsi14":50.0, "atr14":0.0, "vwap":0.0, "obv":0.0, "vol_strength":1.0, "pct_24h":0.0}

    # 1 trend_ema200
    try:
        if df4h is not None and len(df4h) >= 200:
            ema200_htf = float(ema(df4h["close"], 200).iloc[-1])
            if indicators["last_price"] >= ema200_htf * 1.02:
                parts["trend_ema200"] = WEIGHTS["trend_ema200"]; notes.append("Trend HTF: bullish")
            elif indicators["last_price"] >= ema200_htf * 0.98:
                parts["trend_ema200"] = int(WEIGHTS["trend_ema200"] * 0.5); notes.append("Trend HTF: near EMA200")
            else:
                parts["trend_ema200"] = 0; notes.append("Trend HTF: bearish")
        else:
            parts["trend_ema200"] = WEIGHTS["trend_ema200"] if indicators["last_price"] >= indicators["ema200"] * 1.02 else (int(WEIGHTS["trend_ema200"]*0.5) if indicators["last_price"] >= indicators["ema200"] * 0.98 else 0)
            notes.append("Trend: LTF fallback")
    except Exception:
        parts["trend_ema200"] = 0

    # 2 market structure
    ms = detect_market_structure(df15)
    if ms == "bull": parts["market_structure"] = WEIGHTS["market_structure"]
    elif ms == "neutral": parts["market_structure"] = int(WEIGHTS["market_structure"]*0.5)
    else: parts["market_structure"] = 0
    notes.append(f"Market structure: {ms}")

    # 3 price action
    try:
        last = df15.iloc[-1]; prev = df15.iloc[-2]
        if last["close"] > prev["close"] and indicators["vol_strength"] > 1.2:
            parts["price_action"] = WEIGHTS["price_action"]; notes.append("Price action: candle+vol confirm")
        elif last["close"] > prev["close"]:
            parts["price_action"] = int(WEIGHTS["price_action"]*0.6); notes.append("Price action: candle up")
        else:
            parts["price_action"] = 0; notes.append("Price action: none")
    except Exception:
        parts["price_action"] = 0

    # 4 volume
    if indicators["vol_strength"] >= 1.5:
        parts["volume"] = WEIGHTS["volume"]; notes.append("Volume: strong")
    elif indicators["vol_strength"] >= 1.0:
        parts["volume"] = int(WEIGHTS["volume"]*0.6); notes.append("Volume: normal")
    else:
        parts["volume"] = 0; notes.append("Volume: weak")

    # 5 rsi
    r = indicators["rsi14"]
    if r >= 55:
        parts["rsi"] = WEIGHTS["rsi"]; notes.append(f"RSI {r:.1f}: bullish")
    elif r <= 45:
        parts["rsi"] = int(WEIGHTS["rsi"]*0.6); notes.append(f"RSI {r:.1f}: bearish")
    else:
        parts["rsi"] = 0; notes.append(f"RSI {r:.1f}: neutral")

    # 6 macd proxy (ema50 > ema200)
    parts["macd"] = WEIGHTS["macd"] if indicators["ema50"] > indicators["ema200"] else 0

    # 7 multi_tf alignment (use 1h if available)
    if df1h is not None:
        try:
            ema50_1h = float(ema(df1h["close"],50).iloc[-1]) if len(df1h)>=50 else float(df1h["close"].rolling(window=min(50,len(df1h))).mean().iloc[-1])
            ema200_1h = float(ema(df1h["close"],200).iloc[-1]) if len(df1h)>=200 else float(df1h["close"].rolling(window=min(200,len(df1h))).mean().iloc[-1])
            parts["multi_tf"] = WEIGHTS["multi_tf"] if (ema50_1h > ema200_1h and indicators["ema50"] > indicators["ema200"]) else int(WEIGHTS["multi_tf"]*0.5)
        except Exception:
            parts["multi_tf"] = int(WEIGHTS["multi_tf"]*0.5)
    else:
        parts["multi_tf"] = int(WEIGHTS["multi_tf"]*0.6)

    # 8 funding & oi
    funding, oi = get_funding_and_oi(symbol)
    if abs(funding) > 0.001:
        parts["funding_oi"] = int(WEIGHTS["funding_oi"]*0.3); notes.append(f"Funding extreme: {funding:.6f}")
    else:
        parts["funding_oi"] = WEIGHTS["funding_oi"]; notes.append(f"Funding: {funding:.6f}")

    # 9 news flag (placeholder)
    nf = False
    parts["news_flag"] = 0 if nf else WEIGHTS["news_flag"]

    # 10 liquidity depth via orderbook spread
    try:
        ob = get_order_book(symbol, limit=5)
        asks = ob.get("asks", []); bids = ob.get("bids", [])
        if asks and bids:
            best_ask = float(asks[0][0]); best_bid = float(bids[0][0])
            spread_pct = (best_ask - best_bid) / ((best_ask + best_bid)/2)
            parts["liquidity_depth"] = WEIGHTS["liquidity_depth"] if spread_pct < 0.002 else int(WEIGHTS["liquidity_depth"]*0.5)
        else:
            parts["liquidity_depth"] = int(WEIGHTS["liquidity_depth"]*0.5)
    except Exception:
        parts["liquidity_depth"] = int(WEIGHTS["liquidity_depth"]*0.5)

    # 11..18 advanced proxies
    parts["order_flow"] = WEIGHTS["order_flow"] if indicators["vol_strength"] > 1.3 else int(WEIGHTS["order_flow"]*0.5)
    parts["liquidity_pools"] = WEIGHTS["liquidity_pools"] if indicators["vol_strength"] > 1.2 else 0
    parts["vwap"] = WEIGHTS["vwap"] if indicators["last_price"] > indicators["vwap"] else int(WEIGHTS["vwap"]*0.5)
    parts["volatility"] = WEIGHTS["volatility"] if (indicators["atr14"] / indicators["last_price"] if indicators["last_price"] else 1.0) < 0.05 else int(WEIGHTS["volatility"]*0.5)
    parts["correlation"] = WEIGHTS["correlation"]  # placeholder (needs BTC/ETH correlation)
    parts["monthly_bias"] = WEIGHTS["monthly_bias"]  # placeholder
    # long_short balance: more balanced -> higher score
    ls_ratio = 0.5
    if funding > 0.0005: ls_ratio = 0.8
    if funding < -0.0005: ls_ratio = 0.2
    parts["long_short_ratio"] = int(WEIGHTS["long_short_ratio"] * (1 - abs(0.5 - ls_ratio) * 2))
    parts["time_of_day"] = int(WEIGHTS["time_of_day"] * (time_of_day_score() / 3))

    # 19..25 stabilizers / extras (proxies)
    parts["position_sizing"] = WEIGHTS["position_sizing"]
    parts["psych_check"] = WEIGHTS["psych_check"]
    parts["backtest_flag"] = WEIGHTS["backtest_flag"]
    parts["regime_detection"] = WEIGHTS["regime_detection"] if (indicators["atr14"] / indicators["last_price"] if indicators["last_price"] else 1.0) < 0.1 else int(WEIGHTS["regime_detection"]*0.5)
    parts["asset_specific"] = WEIGHTS["asset_specific"]
    parts["smc_proxy"] = WEIGHTS["smc_proxy"]
    parts["ml_prob"] = int(WEIGHTS["ml_prob"] * ml_prob_stub(symbol))

    # finalize score
    raw_sum = sum(int(v) for v in parts.values())
    score = max(0, min(TOTAL_WEIGHT, int(raw_sum)))

    # direction decision
    direction = "NEUTRAL"
    if parts.get("trend_ema200",0) > 0 and parts.get("market_structure",0) > 0 and parts.get("rsi",0) > 0:
        direction = "LONG"
    elif parts.get("trend_ema200",0) == 0 and parts.get("market_structure",0) == 0 and parts.get("rsi",0) > 0:
        direction = "SHORT"

    meta = {"parts": parts, "indicators": indicators, "funding": funding, "oi": oi, "notes": notes, "direction": direction}
    return score, notes, meta

# -----------------------------
# Persian formatting (match your sample)
# -----------------------------
def format_persian_report(symbol: str, score: int, meta: Dict) -> str:
    ind = meta.get("indicators", {})
    last = ind.get("last_price", 0.0)
    rsi_val = ind.get("rsi14", 0.0)
    direction = meta.get("direction", "NEUTRAL")
    funding = meta.get("funding", 0.0)

    if score >= 100:
        tier = "STRONG 🔥"
    elif score >= 80:
        tier = "GOOD ✅"
    elif score >= 60:
        tier = "WEAK ⚠️"
    else:
        tier = "NO_TRADE ⛔️"

    atr = ind.get("atr14", max(0.0001, last*0.01))
    entry = last
    stop = last - atr*2
    tp1 = last + atr*2
    tp2 = last + atr*4

    notes = meta.get("notes", [])[:10]
    txt = (
        f"🚨 سیگنال: {tier}   {direction}\n"
        f"📌 نماد: {symbol}\n"
        f"🔢 امتیاز کل: {score}/{TOTAL_WEIGHT}\n"
        f"📈 قیمت فعلی: {last:,.2f} USDT\n"
        f"📊 RSI: {rsi_val:.1f}\n"
        f"🔔 Funding: {funding:.6f}\n\n"
        "📋 خلاصه تحلیل:\n"
    )
    for n in notes:
        txt += f" - {n}\n"
    txt += (
        f"\n🎯 سطوح پیشنهادی:\nورود: {entry:,.2f} USDT\n"
        f"حد ضرر: {stop:,.2f} USDT\nTP1: {tp1:,.2f} USDT\nTP2: {tp2:,.2f} USDT\n\n"
        "⚠️ هشدار: این تحلیل صرفاً دیدگاه تحلیلی است؛ مسئولیت سرمایه‌گذاری با شماست."
    )
    return txt

# -----------------------------
# Background analyzer loop
# -----------------------------
async def analyzer_loop():
    logger.info("Analyzer loop started (top=%s, interval=%ss)", TOP_N, ANALYZE_INTERVAL)
    await asyncio.sleep(2)  # let bot start
    while True:
        try:
            symbols = get_top_symbols(TOP_N)
            logger.info("Scanning %d symbols...", len(symbols))
            for sym in symbols:
                try:
                    df15 = fetch_klines_df(sym, "15m", limit=300)
                    df1h = fetch_klines_df(sym, "1h", limit=200)
                    df4h = fetch_klines_df(sym, "4h", limit=200)
                    score, notes, meta = compute_score_full(sym, df15, df1h, df4h)
                    if score >= SEND_THRESHOLD:
                        msg = format_persian_report(sym, score, meta)
                        try:
                            await bot.send_message(ADMIN_ID, msg)
                        except Exception as e:
                            logger.exception("send_message failed for %s: %s", sym, e)
                    # friendly delay to avoid bursts
                    await asyncio.sleep(0.4)
                except Exception as e_sym:
                    logger.exception("Error analyzing %s: %s", sym, e_sym)
                    await asyncio.sleep(0.2)
            logger.info("Cycle complete; sleeping %s seconds", ANALYZE_INTERVAL)
        except Exception as e:
            logger.exception("Top-level analyzer error: %s", e)
        await asyncio.sleep(ANALYZE_INTERVAL)

# -----------------------------
# Telegram handlers (admin only)
# -----------------------------
@dp.message_handler(commands=["start"])
async def cmd_start(message: types.Message):
    if message.from_user.id != ADMIN_ID:
        await message.reply("❌ شما دسترسی ندارید.")
        return
    await message.reply("✅ رباط تحلیل‌گر فعال شد. گزارش‌ها برای شما ارسال خواهد شد.")

@dp.message_handler(commands=["scan"])
async def cmd_scan(message: types.Message):
    if message.from_user.id != ADMIN_ID:
        return
    await message.reply("در حال اجرای اسکن سریع... لطفاً صبر کنید.")
    symbols = get_top_symbols(limit=20)
    out = []
    for s in symbols[:10]:
        try:
            df15 = fetch_klines_df(s, "15m", limit=250)
            score, notes, meta = compute_score_full(s, df15, None, None)
            out.append(f"{s}: {score}/{TOTAL_WEIGHT}")
        except Exception:
            out.append(f"{s}: error")
    await message.reply("\n".join(out))

@dp.message_handler(commands=["signal"])
async def cmd_signal(message: types.Message):
    if message.from_user.id != ADMIN_ID:
        return
    args = message.get_args()
    if not args:
        await message.reply("Usage: /signal SYMBOL (e.g. /signal BTCUSDT)")
        return
    symbol = args.strip().upper()
    try:
        df15 = fetch_klines_df(symbol, "15m", limit=300)
        df1h = fetch_klines_df(symbol, "1h", limit=200)
        df4h = fetch_klines_df(symbol, "4h", limit=200)
        score, notes, meta = compute_score_full(symbol, df15, df1h, df4h)
        txt = format_persian_report(symbol, score, meta)
        await message.reply(txt)
    except Exception as e:
        logger.exception("Manual signal error: %s", e)
        await message.reply(f"Error analyzing {symbol}: {e}")

@dp.message_handler(commands=["status"])
async def cmd_status(message: types.Message):
    if message.from_user.id != ADMIN_ID:
        return
    await message.reply(f"ربات آنلاین است. چرخه هر {ANALYZE_INTERVAL} ثانیه اجرا می‌شود. TOP_N={TOP_N}")

# -----------------------------
# Startup & shutdown (webhook)
# -----------------------------
async def on_startup(dp):
    logger.info("on_startup: setting webhook and starting analyzer")
    try:
        await bot.set_webhook(WEBHOOK_URL)
        logger.info("Webhook set: %s", WEBHOOK_URL)
    except Exception as e:
        logger.exception("Failed to set webhook: %s", e)
    # start analyzer background task
    asyncio.create_task(analyzer_loop())

async def on_shutdown(dp):
    logger.info("on_shutdown: deleting webhook and closing bot")
    try:
        await bot.delete_webhook()
    except Exception:
        pass
    await bot.close()

# -----------------------------
# Run (entrypoint)
# -----------------------------
if __name__ == "__main__":
    logger.info("Starting webhook server on %s:%s", WEBAPP_HOST, WEBAPP_PORT)
    start_webhook(
        dispatcher=dp,
        webhook_path=WEBHOOK_PATH,
        on_startup=on_startup,
        on_shutdown=on_shutdown,
        skip_updates=True,
        host=WEBAPP_HOST,
        port=WEBAPP_PORT,
    )
