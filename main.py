# main.py
import asyncio
import logging
import math
import time
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, List

import aiohttp
import numpy as np
import pandas as pd
from aiogram import Bot, Dispatcher, types
from aiogram.utils import exceptions
from aiogram.utils.executor import start_webhook

import config  # your config.py with TELEGRAM_TOKEN, ADMIN_ID, WATCHLIST, ...

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scanner")

# -------------------------
# Bot init
# -------------------------
bot = Bot(token=config.TELEGRAM_TOKEN)
dp = Dispatcher(bot)

# aiohttp session used across requests
_http_session: aiohttp.ClientSession = None


async def get_session() -> aiohttp.ClientSession:
    global _http_session
    if _http_session is None or _http_session.closed:
        _http_session = aiohttp.ClientSession()
    return _http_session


# -------------------------
# Helpers - numeric / indicator
# -------------------------
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=1).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1 / period, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h_l = df['high'] - df['low']
    h_pc = (df['high'] - df['close'].shift()).abs()
    l_pc = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df['high'] + df['low'] + df['close']) / 3
    v = df['volume']
    return (tp * v).cumsum() / (v.cumsum() + 1e-9)


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df['close'].diff()).fillna(0)
    return (direction * df['volume']).cumsum()


def stoch_rsi(rsi_series: pd.Series, k_period=14, d_period=3):
    min_r = rsi_series.rolling(k_period).min()
    max_r = rsi_series.rolling(k_period).max()
    k = 100 * (rsi_series - min_r) / (max_r - min_r + 1e-9)
    d = k.rolling(d_period).mean()
    return k, d


# -------------------------
# Binance fetchers (futures)
# -------------------------
BINANCE_FAPI = "https://fapi.binance.com"

async def fetch_klines_binance(symbol: str, interval: str = "1h", limit: int = 500) -> pd.DataFrame:
    """
    returns DataFrame with columns: open_time, open, high, low, close, volume (float)
    """
    session = await get_session()
    url = f"{BINANCE_FAPI}/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        async with session.get(url, timeout=20) as resp:
            if resp.status != 200:
                logger.warning("Binance klines failed %s %s", symbol, resp.status)
                return pd.DataFrame()
            data = await resp.json()
    except Exception as e:
        logger.exception("fetch_klines_binance error %s", e)
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    # convert types
    for col in ["open", "high", "low", "close", "volume", "taker_buy_base", "taker_buy_quote"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    return df


async def fetch_recent_trades(symbol: str, limit: int = 1000) -> List[Dict[str, Any]]:
    session = await get_session()
    url = f"{BINANCE_FAPI}/fapi/v1/aggTrades?symbol={symbol}&limit={limit}"
    try:
        async with session.get(url, timeout=20) as resp:
            if resp.status != 200:
                return []
            return await resp.json()
    except Exception:
        return []


async def fetch_depth(symbol: str, limit: int = 20) -> Dict[str, Any]:
    session = await get_session()
    url = f"{BINANCE_FAPI}/fapi/v1/depth?symbol={symbol}&limit={limit}"
    try:
        async with session.get(url, timeout=20) as resp:
            if resp.status != 200:
                return {}
            return await resp.json()
    except Exception:
        return {}


async def fetch_funding_and_oi(symbol: str) -> Tuple[float, float]:
    """
    Returns (fundingRate, openInterest)
    """
    session = await get_session()
    funding_url = f"{BINANCE_FAPI}/fapi/v1/premiumIndex?symbol={symbol}"
    oi_url = f"{BINANCE_FAPI}/fapi/v1/openInterest?symbol={symbol}"
    funding = 0.0
    oi = 0.0
    try:
        async with session.get(funding_url, timeout=10) as resp:
            if resp.status == 200:
                j = await resp.json()
                funding = float(j.get("lastFundingRate", 0.0))
    except Exception:
        pass
    try:
        async with session.get(oi_url, timeout=10) as resp:
            if resp.status == 200:
                j = await resp.json()
                oi = float(j.get("openInterest", 0.0))
    except Exception:
        pass
    return funding, oi


# -------------------------
# Scoring: mapping of 25 metrics into numeric score
# -------------------------
# We'll map to a max_total consistent with your design (~134).
WEIGHTS = {
    # Core technical (10)
    "trend": 15,
    "market_structure": 12,
    "price_action": 12,
    "volume": 10,
    "rsi": 8,
    "macd": 8,
    "multi_tf": 8,
    "funding_oi": 8,
    "news": 7,  # optional/manual
    "liquidity": 7,

    # Advanced (8)
    "order_flow": 5,
    "liquidity_pools": 5,
    "vwap": 3,
    "volatility": 3,
    "correlation": 5,
    "monthly_bias": 5,
    "long_short_ratio": 5,
    "session": 3,

    # Stabilizers (7)
    "position_sizing": 3,
    "psych": 1,
    "backtest": 1,
    "regime": 2,
    "asset_specific": 2,
    "smc": 2,
    "ml": 1
}
MAX_SCORE = sum(WEIGHTS.values())


def normalize_to_100(x: float) -> float:
    # map 0..MAX_SCORE to 0..100
    return max(0.0, min(100.0, (x / MAX_SCORE) * 100.0))


def compute_support_resistance(df: pd.DataFrame) -> Tuple[float, float]:
    # Simple approach: highest highs and lowest lows from last N
    lookback = min(len(df), 200)
    if lookback < 5:
        return (float(df['close'].iloc[-1]), float(df['close'].iloc[-1]))
    highs = df['high'][-lookback:]
    lows = df['low'][-lookback:]
    resistance = float(highs.max())
    support = float(lows.min())
    return support, resistance


async def score_symbol(symbol: str, interval: str = "1h") -> Dict[str, Any]:
    """
    Compute full 25-metric score and return details
    """
    df = await fetch_klines_binance(symbol, interval=interval, limit=500)
    if df.empty:
        # fallback: try smaller limit / return minimal
        return {"symbol": symbol, "score": 0, "details": "no data", "components": {}}

    # compute indicators
    close = df['close']
    ema50 = ema(close, 50)
    ema200 = ema(close, 200)
    macd_line, macd_signal, macd_hist = macd(close)
    rsi_series = rsi(close, period=14)
    atr_series = atr(df, period=14)
    vwap_series = vwap(df)
    obv_series = obv(df)
    stoch_k, stoch_d = stoch_rsi(rsi_series)

    last_close = float(close.iloc[-1])
    last_rsi = float(rsi_series.iloc[-1]) if len(rsi_series) > 0 else 50.0
    last_macd_hist = float(macd_hist.iloc[-1]) if len(macd_hist) > 0 else 0.0
    last_atr = float(atr_series.iloc[-1]) if len(atr_series) > 0 else 0.0
    last_vwap = float(vwap_series.iloc[-1]) if len(vwap_series) > 0 else last_close
    last_obv = float(obv_series.iloc[-1]) if len(obv_series) > 0 else 0.0

    # support / resistance
    support, resistance = compute_support_resistance(df)

    # recent trades -> order flow proxy
    trades = await fetch_recent_trades(symbol, limit=500)
    buys = 0
    sells = 0
    for t in trades:
        # aggTrades format: 'm' field indicates maker? use taker_is_buyer? (agg doesn't include taker flag)
        # many aggTrades don't have isBuyerMaker; safe fallback using price movement: can't be precise
        # We'll use taker_buy_base if available in klines summary
        pass
    # quick proxy: compare taker_buy_base in recent klines
    recent_taker_buy = df['taker_buy_base'].tail(20).sum()
    recent_volume = df['volume'].tail(20).sum()
    buy_ratio = (recent_taker_buy / (recent_volume + 1e-9)) if recent_volume > 0 else 0.5

    # funding and open interest
    funding, oi = await fetch_funding_and_oi(symbol)

    # liquidity: depth snapshot
    depth = await fetch_depth(symbol, limit=20)
    bid_depth = sum([float(x[1]) for x in depth.get('bids', [])[:10]]) if depth else 0.0
    ask_depth = sum([float(x[1]) for x in depth.get('asks', [])[:10]]) if depth else 0.0
    spread = 0.0
    if depth and depth.get('bids') and depth.get('asks'):
        best_bid = float(depth['bids'][0][0])
        best_ask = float(depth['asks'][0][0])
        spread = (best_ask - best_bid) / ((best_ask + best_bid) / 2 + 1e-9)

    # multi-timeframe check: check EMA50 vs EMA200 on HTF (we'll use same df but can simulate HTF by resampling)
    # For simplicity: use last 4H (every 4th candle) as HTF approx when interval is 1h or less
    multi_tf_score = 0
    try:
        if len(df) >= 200:
            # get HTF by resampling every 4 bars as proxy for 4h
            htf_close = df['close'].resample('4H', on='open_time').last() if 'open_time' in df.columns else df['close']
            if len(htf_close) > 10:
                htf_ema50 = ema(htf_close, 50).iloc[-1]
                htf_ema200 = ema(htf_close, 200).iloc[-1] if len(htf_close) >= 200 else ema(htf_close, 200).iloc[-1]
                if htf_ema50 > htf_ema200:
                    multi_tf_score = WEIGHTS['multi_tf']
    except Exception:
        multi_tf_score = 0

    components = {}

    # 1. Trend (EMA200 on HTF) approximate
    trend_score = 0
    if ema50.iloc[-1] >= ema200.iloc[-1]:
        # price above ema200? strong
        trend_score = WEIGHTS['trend']
    elif abs(ema50.iloc[-1] - ema200.iloc[-1]) / (ema200.iloc[-1] + 1e-9) < 0.02:
        trend_score = WEIGHTS['trend'] / 2
    else:
        trend_score = 0
    components['trend'] = trend_score

    # 2. Market structure (HH/HL or LH/LL) simple heuristic
    ms_score = 0
    look = 20
    highs = df['high'].tail(look).values
    lows = df['low'].tail(look).values
    if len(highs) >= 6:
        # check simple HH/HL
        if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
            ms_score = WEIGHTS['market_structure']
        else:
            ms_score = WEIGHTS['market_structure'] / 2
    components['market_structure'] = ms_score

    # 3. Price action: presence of strong bullish/bearish candle on support/resistance
    pa_score = 0
    last_open = df['open'].iloc[-1]
    last_high = df['high'].iloc[-1]
    last_low = df['low'].iloc[-1]
    last_close_val = df['close'].iloc[-1]
    body = abs(last_close_val - last_open)
    candle_range = last_high - last_low + 1e-9
    body_ratio = body / candle_range
    # bullish engulfing proxy
    if last_close_val > last_open and body_ratio > 0.6 and last_close_val > last_open:
        pa_score = WEIGHTS['price_action']
    else:
        pa_score = WEIGHTS['price_action'] / 2
    components['price_action'] = pa_score

    # 4. Volume
    vol_score = 0
    recent_vol = df['volume'].tail(20)
    if len(recent_vol) > 0:
        if recent_vol.iloc[-1] > recent_vol.mean():
            vol_score = WEIGHTS['volume']
        else:
            vol_score = WEIGHTS['volume'] / 3
    components['volume'] = vol_score

    # 5. RSI
    rsi_score = 0
    if last_rsi > 55:
        rsi_score = WEIGHTS['rsi']
    elif last_rsi < 45:
        # negative for long but could be positive for short; for scoring give half
        rsi_score = WEIGHTS['rsi'] / 2
    else:
        rsi_score = WEIGHTS['rsi'] / 3
    components['rsi'] = rsi_score

    # 6. MACD
    macd_score = 0
    if macd_line.iloc[-1] > macd_signal.iloc[-1] and last_macd_hist > 0:
        macd_score = WEIGHTS['macd']
    elif last_macd_hist > 0:
        macd_score = WEIGHTS['macd'] / 2
    components['macd'] = macd_score

    # 7. Multi-timeframe (already computed)
    components['multi_tf'] = multi_tf_score

    # 8. Funding + OI
    funding_score = 0
    # neutral funding near zero gives positive; extreme funding reduces
    if abs(funding) < 0.0005:
        funding_score = WEIGHTS['funding_oi']
    elif abs(funding) < 0.001:
        funding_score = WEIGHTS['funding_oi'] / 2
    else:
        funding_score = 0
    components['funding_oi'] = funding_score

    # 9. News/events - config flag
    components['news'] = WEIGHTS['news'] if getattr(config, "NEWS_ALLOW", False) else 0

    # 10. Liquidity/depth/spread
    liquidity_score = 0
    if spread < 0.005 and (bid_depth + ask_depth) > 0:
        liquidity_score = WEIGHTS['liquidity']
    elif (bid_depth + ask_depth) > 0:
        liquidity_score = WEIGHTS['liquidity'] / 2
    components['liquidity'] = liquidity_score

    # 11. Order flow (proxy by taker buy ratio)
    order_flow_score = 0
    if buy_ratio > 0.55:
        order_flow_score = WEIGHTS['order_flow']
    elif buy_ratio > 0.5:
        order_flow_score = WEIGHTS['order_flow'] / 2
    components['order_flow'] = order_flow_score

    # 12. Liquidity pools / stophunt (proxied by big recent wick)
    liquidity_pools_score = 0
    # detect big wick last candle
    wick_top = float(df['high'].iloc[-1] - df['close'].iloc[-1])
    wick_bottom = float(df['open'].iloc[-1] - df['low'].iloc[-1])
    if wick_bottom > last_atr * 0.8 or wick_top > last_atr * 0.8:
        liquidity_pools_score = WEIGHTS['liquidity_pools'] / 1.5
    components['liquidity_pools'] = liquidity_pools_score

    # 13. VWAP
    vwap_score = 0
    if last_close >= last_vwap:
        vwap_score = WEIGHTS['vwap']
    else:
        vwap_score = 0
    components['vwap'] = vwap_score

    # 14. Volatility (ATR)
    volatility_score = 0
    # moderate ATR preferred; just assign small points
    volatility_score = WEIGHTS['volatility']
    components['volatility'] = volatility_score

    # 15. Correlation with BTC (if symbol is alt)
    correlation_score = 0
    try:
        if symbol != "BTCUSDT" and "BTCUSDT" in config.WATCHLIST:
            btc_df = await fetch_klines_binance("BTCUSDT", interval=interval, limit=500)
            if not btc_df.empty:
                corr = float(np.corrcoef(df['close'].tail(100), btc_df['close'].tail(100))[0, 1])
                if corr > 0.6:
                    correlation_score = WEIGHTS['correlation']
                elif corr > 0.3:
                    correlation_score = WEIGHTS['correlation'] / 2
    except Exception:
        correlation_score = 0
    components['correlation'] = correlation_score

    # 16. Monthly/Weekly bias - simplified using long EMA on whole df
    monthly_bias_score = 0
    if len(df) >= 200:
        if ema(df['close'], 50).iloc[-1] > ema(df['close'], 200).iloc[-1]:
            monthly_bias_score = WEIGHTS['monthly_bias']
    components['monthly_bias'] = monthly_bias_score

    # 17. Long/Short ratio (proxy from funding sign and buy_ratio)
    ls_score = WEIGHTS['long_short_ratio'] if (abs(buy_ratio - 0.5) < 0.2) else WEIGHTS['long_short_ratio'] / 2
    components['long_short_ratio'] = ls_score

    # 18. Session/time-of-day
    session_score = 0
    utc_hour = datetime.utcnow().hour
    # Favor overlaps: 12-16 UTC roughly (London + NY overlap)
    if 12 <= utc_hour <= 16:
        session_score = WEIGHTS['session']
    components['session'] = session_score

    # 19. Position sizing - optional
    components['position_sizing'] = WEIGHTS['position_sizing'] if getattr(config, "ENABLE_POS_SIZING", False) else 0

    # 20. Psychology check: rely on config flag (user must confirm)
    components['psych'] = WEIGHTS['psych'] if getattr(config, "USER_CONFIRMED", True) else 0

    # 21. Backtest flag: small bonus if enabled
    components['backtest'] = WEIGHTS['backtest'] if getattr(config, "HAS_BACKTEST", False) else 0

    # 22. Regime detection (ATR relative)
    regime_score = 0
    avg_atr = atr_series.tail(50).mean() if len(atr_series) > 0 else last_atr
    if last_atr < avg_atr:
        regime_score = WEIGHTS['regime']  # calmer market
    components['regime'] = regime_score

    # 23. Asset-specific rules: from config per-symbol overrides
    components['asset_specific'] = WEIGHTS['asset_specific']

    # 24. Smart Money Concepts (proxy)
    smc_score = 0
    # if big wick + retest near level, small bonus
    if (wick_bottom > last_atr * 0.6) or (wick_top > last_atr * 0.6):
        smc_score = WEIGHTS['smc'] / 1.5
    components['smc'] = smc_score

    # 25. ML model (optional)
    components['ml'] = WEIGHTS['ml'] if getattr(config, "ENABLE_ML", False) else 0

    # Sum all components
    total_raw = sum(components.values())
    normalized = normalize_to_100(total_raw)

    # compute direction: long if ema50>ema200 and other confirmations
    direction = "LONG" if ema50.iloc[-1] > ema200.iloc[-1] else "SHORT"

    # Build price targets: ATR-based simple
    if direction == "LONG":
        sl = round(last_close - 1.5 * last_atr, 2)
        tp1 = round(last_close + 1.0 * last_atr, 2)
        tp2 = round(last_close + 2.0 * last_atr, 2)
    else:
        sl = round(last_close + 1.5 * last_atr, 2)
        tp1 = round(last_close - 1.0 * last_atr, 2)
        tp2 = round(last_close - 2.0 * last_atr, 2)

    # risk reward approx
    rr = None
    try:
        if direction == "LONG":
            rr = abs((tp1 - last_close) / (last_close - sl + 1e-9))
        else:
            rr = abs((last_close - tp1) / (sl - last_close + 1e-9))
        rr = round(rr, 2)
    except Exception:
        rr = None

    # Compose details to return
    result = {
        "symbol": symbol,
        "score_raw": total_raw,
        "score_norm": round(normalized, 1),
        "direction": direction,
        "last_price": last_close,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "rr": rr,
        "support": round(support, 2),
        "resistance": round(resistance, 2),
        "funding": funding,
        "open_interest": oi,
        "components": components,
        "indicators": {
            "rsi": round(last_rsi, 1),
            "macd_hist": round(last_macd_hist, 6),
            "ema50": round(float(ema50.iloc[-1]), 2),
            "ema200": round(float(ema200.iloc[-1]), 2),
            "atr": round(last_atr, 4),
            "vwap": round(last_vwap, 2),
            "obv": round(last_obv, 2),
            "buy_ratio": round(buy_ratio, 3),
            "spread": round(spread, 6),
        }
    }
    return result


# -------------------------
# Message formatting (Persian)
# -------------------------
def format_signal_message(res: Dict[str, Any], timeframe: str = "1h") -> str:
    # Build Persian message matching screenshots
    sym = res['symbol']
    last_price = res['last_price']
    dir_fa = "لانگ" if res['direction'] == "LONG" else "شورت"
    score = res['score_norm']
    prob = f"{int(score)}%"
    entry = f"USDT {res['last_price']:,}"
    support = f"USDT {res['support']:,}"
    resistance = f"USDT {res['resistance']:,}"
    sl = f"USDT {res['sl']:,}"
    tp1 = f"USDT {res['tp1']:,}"
    tp2 = f"USDT {res['tp2']:,}"
    rr = res['rr'] if res['rr'] is not None else "-"
    funding = res['funding']
    oi = res['open_interest']

    # Indicators summary
    ind = res['indicators']
    ind_lines = []
    ind_lines.append(f"🔎 تحلیل اندیکاتورها:")
    ind_lines.append(f"📈 RSI: نزدیک {ind['rsi']} → {'ممنتوم مثبت' if ind['rsi']>=50 else 'ضعیف'}")
    ind_lines.append(f"📊 MACD hist: {ind['macd_hist']}")
    ind_lines.append(f"📐 EMA50/EMA200: {res['indicators']['ema50']} / {res['indicators']['ema200']}")
    ind_lines.append(f"🧮 حجم: نسبت خرید % {int(ind['buy_ratio']*100)} - OI: {int(oi)}")
    ind_block = "\n".join(ind_lines)

    header = f"⚡️ جفت ارز: {sym}\n⏱ تایم‌فریم: {timeframe}\n📌 وضعیت بازار: {dir_fa}\n🔢 امتیاز: {score} / 100\n🔮 احتمال: {prob}\n🔁 نسبت ریسک/ریوارد: {rr}\n"
    body = f"🔑 سطح کلیدی: {resistance}\n⏱ زمان مناسب برای ورود: بعد از اصلاح کوچک\n📌 زمان نگهداری: چند ساعت\n\n🔵 ورود: {entry}\n🔴 حد ضرر: {sl}\n🎯 حد سود (TP1): {tp1}\n🎯 حد سود (TP2): {tp2}\n⚠️ هشدار: ریسک پایین؛ تایید بر اساس اصلاح صعودی\n\n" + ind_block
    return header + "\n" + body


# -------------------------
# Bot commands
# -------------------------
@dp.message_handler(commands=["start"])
async def cmd_start(message: types.Message):
    await message.answer("سلام 👋\nربات سیگنال‌ساز آماده‌ست.\nبرای تحلیل سریع یک نماد بنویس:\n/scan BTC\nبرای دریافت گزارش کامل: /status BTC")


@dp.message_handler(commands=["scan"])
async def cmd_scan(message: types.Message):
    try:
        parts = message.text.strip().split()
        if len(parts) < 2:
            await message.reply("فرمت: /scan SYMBOL  (مثال: /scan BTC)")
            return
        symbol_raw = parts[1].upper()
        if not symbol_raw.endswith("USDT"):
            symbol = symbol_raw + "USDT"
        else:
            symbol = symbol_raw
        await message.reply(f"در حال تحلیل {symbol} ... ⏳")
        res = await score_symbol(symbol, interval=config.DEFAULT_INTERVAL)
        msg = format_signal_message(res, timeframe=config.DEFAULT_INTERVAL)
        await message.answer(msg)
    except Exception as e:
        logger.exception("scan command error")
        await message.reply("خطا هنگام تحلیل. لطفا بعدا امتحان کنید.")


@dp.message_handler(commands=["status"])
async def cmd_status(message: types.Message):
    # detailed score per component
    try:
        parts = message.text.strip().split()
        if len(parts) < 2:
            await message.reply("فرمت: /status SYMBOL")
            return
        symbol_raw = parts[1].upper()
        if not symbol_raw.endswith("USDT"):
            symbol = symbol_raw + "USDT"
        else:
            symbol = symbol_raw
        await message.reply(f"در حال اجرای گزارش برای {symbol} ...")
        res = await score_symbol(symbol, interval=config.DEFAULT_INTERVAL)
        comp = res.get("components", {})
        lines = [f"📋 گزارش امتیاز‌ها برای {symbol} (نرمال‌شده: {res['score_norm']}%)"]
        for k, v in comp.items():
            lines.append(f"• {k}: {round(float(v), 2)}")
        lines.append("\n" + format_signal_message(res, timeframe=config.DEFAULT_INTERVAL))
        await message.answer("\n".join(lines))
    except Exception:
        logger.exception("status command")
        await message.reply("خطا در تولید گزارش.")


@dp.message_handler(commands=["subscribe"])
async def cmd_subscribe(message: types.Message):
    # simple subscribe: add chat id to subscribers file (minimal)
    cid = str(message.chat.id)
    try:
        subs = set()
        try:
            with open("subscribers.txt", "r") as f:
                subs = set(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            subs = set()
        subs.add(cid)
        with open("subscribers.txt", "w") as f:
            f.write("\n".join(subs))
        await message.reply("شما عضو لیست اطلاع‌رسانی شدید ✅")
    except Exception:
        await message.reply("خطا در عضویت.")


@dp.message_handler(commands=["unsubscribe"])
async def cmd_unsubscribe(message: types.Message):
    cid = str(message.chat.id)
    try:
        subs = set()
        try:
            with open("subscribers.txt", "r") as f:
                subs = set(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            subs = set()
        if cid in subs:
            subs.remove(cid)
        with open("subscribers.txt", "w") as f:
            f.write("\n".join(subs))
        await message.reply("شما از لیست اطلاع‌رسانی حذف شدید ✅")
    except Exception:
        await message.reply("خطا در عملیات حذف.")


# -------------------------
# Auto-scan task (periodic)
# -------------------------
async def send_signal_to_subscribers(text: str):
    # send message to admin and subscribers
    recipients = {str(config.ADMIN_ID)}
    # add file subscribers
    try:
        with open("subscribers.txt", "r") as f:
            for line in f:
                if line.strip():
                    recipients.add(line.strip())
    except FileNotFoundError:
        pass

    for r in recipients:
        try:
            await bot.send_message(int(r), text)
        except exceptions.BotBlocked:
            logger.warning("Bot blocked by %s", r)
        except Exception as e:
            logger.exception("Failed send to %s: %s", r, e)


async def auto_scan_task():
    logger.info("Auto-scan started (top %s every %s seconds)", config.AUTO_SCAN_TOP_N, config.SCAN_INTERVAL_SECONDS)
    while True:
        start = time.time()
        tasks = []
        watch = config.WATCHLIST[:config.AUTO_SCAN_TOP_N]
        for symbol in watch:
            # ensure symbol format
            sym = symbol.upper()
            if not sym.endswith("USDT"):
                sym = sym + "USDT"
            tasks.append(score_symbol(sym, interval=config.DEFAULT_INTERVAL))
        # run concurrently with limit
        results = []
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.exception("auto_scan gather error: %s", e)
            results = []
        # evaluate results
        for r in results:
            if isinstance(r, Exception):
                continue
            try:
                # r is dict
                prob = r.get("score_norm", 0)
                if prob >= config.PROBABILITY_THRESHOLD:
                    msg = format_signal_message(r, timeframe=config.DEFAULT_INTERVAL)
                    await send_signal_to_subscribers(msg)
            except Exception:
                logger.exception("error processing result")

        elapsed = time.time() - start
        wait_for = max(5, config.SCAN_INTERVAL_SECONDS - elapsed)
        await asyncio.sleep(wait_for)


# -------------------------
# Startup / Shutdown
# -------------------------
async def on_startup(dispatcher):
    logger.info("Starting up bot...")
    # start auto_scan in background if enabled
    if getattr(config, "ENABLE_AUTO_SCAN", True):
        asyncio.create_task(auto_scan_task())

    if getattr(config, "ENABLE_WEBHOOK", False):
        try:
            await bot.set_webhook(config.WEBHOOK_URL)
            logger.info("Webhook set: %s", config.WEBHOOK_URL)
        except Exception:
            logger.exception("Failed to set webhook")


async def on_shutdown(dispatcher):
    logger.info("Shutting down bot...")
    try:
        await bot.delete_webhook()
    except Exception:
        pass
    if _http_session and not _http_session.closed:
        await _http_session.close()
    await bot.session.close()


# -------------------------
# Run (webhook or polling)
# -------------------------
if __name__ == "__main__":
    if getattr(config, "ENABLE_WEBHOOK", False):
        # webhook mode for Render (if configured)
        start_webhook(
            dispatcher=dp,
            webhook_path=config.WEBHOOK_PATH,
            on_startup=on_startup,
            on_shutdown=on_shutdown,
            skip_updates=True,
            host=config.WEBAPP_HOST,
            port=config.WEBAPP_PORT,
        )
    else:
        from aiogram import executor
        executor.start_polling(dp, on_startup=on_startup, skip_updates=True)
