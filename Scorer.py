# scorer.py
import math
import logging
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

import config
import exchange
import indicators
import utils

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

# Weights for 25 metrics (sum defines MAX_RAW)
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
    "news": 7,
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
MAX_RAW = sum(WEIGHTS.values())


def normalize_score(raw: float) -> float:
    """Normalize raw score (0..MAX_RAW) to 0..100"""
    try:
        norm = max(0.0, min(100.0, (raw / MAX_RAW) * 100.0))
        return round(norm, 2)
    except Exception:
        return 0.0


def compute_support_resistance(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Simple support/resistance approximation:
    support = recent low (lookback window)
    resistance = recent high
    """
    if df.empty or "low" not in df.columns or "high" not in df.columns:
        return (0.0, 0.0)
    look = min(len(df), 200)
    support = float(df['low'].tail(look).min())
    resistance = float(df['high'].tail(look).max())
    return support, resistance


def detect_market_structure(df: pd.DataFrame, lookback: int = 20) -> str:
    """
    Very simple HH/HL or LH/LL detection using local highs/lows.
    Returns 'bull', 'bear', or 'neutral'
    """
    if df.empty or len(df) < max(lookback, 5):
        return "neutral"
    highs = df['high'].tail(lookback).values
    lows = df['low'].tail(lookback).values
    try:
        # check last two swings
        if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
            return "bull"
        if highs[-1] < highs[-2] and lows[-1] < lows[-2]:
            return "bear"
    except Exception:
        pass
    return "neutral"


def compute_order_flow_proxy(df: pd.DataFrame) -> float:
    """
    Proxy for order flow using taker buy volume ratio from klines
    (uses taker_buy_base column if available)
    Returns ratio in [0..1]
    """
    if df.empty:
        return 0.5
    if 'taker_buy_base' in df.columns:
        recent = df['taker_buy_base'].tail(50).sum()
        total = df['volume'].tail(50).sum()
        if total <= 0:
            return 0.5
        return float(min(1.0, max(0.0, recent / (total + 1e-9))))
    # fallback: assume neutral
    return 0.5


def score_components_from_df(symbol: str, df: pd.DataFrame, interval: str = "1h") -> Dict[str, float]:
    """
    Compute component scores (raw per weight) based on df and symbol data.
    Returns dict of component_name -> score (0..weight)
    """
    comp = {k: 0.0 for k in WEIGHTS.keys()}

    if df.empty or len(df) < 5:
        # no data -> return zeros
        return comp

    close = df['close']
    last_close = float(close.iloc[-1])
    # Indicators
    try:
        ema50 = indicators.ema(close, 50).iloc[-1]
        ema200 = indicators.ema(close, 200).iloc[-1]
    except Exception:
        ema50 = float(close.iloc[-1])
        ema200 = float(close.iloc[-1])

    macd_line, macd_signal, macd_hist = indicators.macd(close)
    last_macd_hist = float(macd_hist.iloc[-1]) if len(macd_hist) > 0 else 0.0
    rsi_series = indicators.rsi(close)
    last_rsi = float(rsi_series.iloc[-1]) if len(rsi_series) > 0 else 50.0
    atr_series = indicators.bollinger_bands(close)[1]  # reuse sma from bollinger as simple volatility proxy
    # But compute a simple ATR using rolling range
    tr = (df['high'] - df['low']).rolling(14).mean()
    last_atr = float(tr.iloc[-1]) if len(tr) > 0 else max(0.0, last_close * 0.001)

    # support/resistance
    support, resistance = compute_support_resistance(df)

    # 1. trend (EMA50 vs EMA200)
    if ema50 >= ema200:
        comp['trend'] = WEIGHTS['trend']
    elif abs(ema50 - ema200) / (ema200 + 1e-9) < 0.02:
        comp['trend'] = WEIGHTS['trend'] / 2
    else:
        comp['trend'] = 0.0

    # 2. market structure
    ms = detect_market_structure(df)
    if ms == "bull":
        comp['market_structure'] = WEIGHTS['market_structure']
    elif ms == "bear":
        comp['market_structure'] = 0.0
    else:
        comp['market_structure'] = WEIGHTS['market_structure'] / 2

    # 3. price action (simple candle body strength)
    last_open = float(df['open'].iloc[-1])
    last_high = float(df['high'].iloc[-1])
    last_low = float(df['low'].iloc[-1])
    last_body = abs(last_close - last_open)
    candle_range = (last_high - last_low) + 1e-9
    body_ratio = last_body / candle_range
    if last_close > last_open and body_ratio > 0.6:
        comp['price_action'] = WEIGHTS['price_action']
    elif body_ratio > 0.3:
        comp['price_action'] = WEIGHTS['price_action'] / 2
    else:
        comp['price_action'] = WEIGHTS['price_action'] / 4

    # 4. volume
    recent_vol = df['volume'].tail(20)
    if len(recent_vol) > 0 and recent_vol.iloc[-1] > recent_vol.mean():
        comp['volume'] = WEIGHTS['volume']
    else:
        comp['volume'] = WEIGHTS['volume'] / 3

    # 5. rsi
    if last_rsi > 65:
        comp['rsi'] = WEIGHTS['rsi'] * 0.6  # overbought -> penalize for long
    elif last_rsi > 55:
        comp['rsi'] = WEIGHTS['rsi']
    elif last_rsi < 35:
        comp['rsi'] = WEIGHTS['rsi'] * 0.5
    else:
        comp['rsi'] = WEIGHTS['rsi'] / 3

    # 6. macd
    if macd_line.iloc[-1] > macd_signal.iloc[-1] and last_macd_hist > 0:
        comp['macd'] = WEIGHTS['macd']
    elif last_macd_hist > 0:
        comp['macd'] = WEIGHTS['macd'] / 2
    else:
        comp['macd'] = 0.0

    # 7. multi_tf - approximate using resampled data if possible
    try:
        # resample to 4h if index is timestamp
        if 'timestamp' in df.columns:
            htf = df.set_index('timestamp').resample('4H').last()
            if len(htf) > 10:
                htf_close = htf['close'].dropna()
                htf_ema50 = indicators.ema(htf_close, 50).iloc[-1] if len(htf_close) > 50 else indicators.ema(htf_close, 10).iloc[-1]
                htf_ema200 = indicators.ema(htf_close, 200).iloc[-1] if len(htf_close) > 200 else indicators.ema(htf_close, 50).iloc[-1]
                if htf_ema50 > htf_ema200:
                    comp['multi_tf'] = WEIGHTS['multi_tf']
                else:
                    comp['multi_tf'] = 0.0
            else:
                comp['multi_tf'] = WEIGHTS['multi_tf'] / 2
        else:
            comp['multi_tf'] = WEIGHTS['multi_tf'] / 2
    except Exception:
        comp['multi_tf'] = WEIGHTS['multi_tf'] / 2

    # 8. funding + open interest
    try:
        funding = exchange.get_funding_rate(symbol)
        oi = exchange.get_open_interest(symbol)
    except Exception:
        funding = 0.0
        oi = 0.0
    # neutral small funding is ok
    if funding is None:
        comp['funding_oi'] = WEIGHTS['funding_oi'] / 2
    elif abs(funding) < 0.0005:
        comp['funding_oi'] = WEIGHTS['funding_oi']
    elif abs(funding) < 0.001:
        comp['funding_oi'] = WEIGHTS['funding_oi'] / 2
    else:
        comp['funding_oi'] = 0.0

    # 9. news (manual flag)
    comp['news'] = WEIGHTS['news'] if getattr(config, "NEWS_ALLOW", False) else 0.0

    # 10. liquidity - use order book spread and depth
    depth = exchange.get_order_book(symbol, limit=10)
    bid_depth = sum([float(x[1]) for x in depth.get('bids', [])[:10]]) if depth else 0.0
    ask_depth = sum([float(x[1]) for x in depth.get('asks', [])[:10]]) if depth else 0.0
    spread = 0.0
    try:
        if depth and depth.get('bids') and depth.get('asks'):
            best_bid = float(depth['bids'][0][0])
            best_ask = float(depth['asks'][0][0])
            spread = (best_ask - best_bid) / ((best_ask + best_bid) / 2 + 1e-9)
    except Exception:
        spread = 0.0
    if spread < 0.005 and (bid_depth + ask_depth) > 0:
        comp['liquidity'] = WEIGHTS['liquidity']
    elif (bid_depth + ask_depth) > 0:
        comp['liquidity'] = WEIGHTS['liquidity'] / 2
    else:
        comp['liquidity'] = 0.0

    # 11. order flow proxy
    buy_ratio = compute_order_flow_proxy(df)
    if buy_ratio > 0.55:
        comp['order_flow'] = WEIGHTS['order_flow']
    elif buy_ratio > 0.5:
        comp['order_flow'] = WEIGHTS['order_flow'] / 2
    else:
        comp['order_flow'] = 0.0

    # 12. liquidity pools / stophunt proxy via large wicks
    wick_top = last_high - max(last_close, last_open)
    wick_bottom = min(last_close, last_open) - last_low
    if wick_bottom > last_atr * 0.8 or wick_top > last_atr * 0.8:
        comp['liquidity_pools'] = WEIGHTS['liquidity_pools'] / 1.5
    else:
        comp['liquidity_pools'] = 0.0

    # 13. vwap
    try:
        vwap_series = indicators.vwap(df)
        last_vwap = float(vwap_series.iloc[-1])
        comp['vwap'] = WEIGHTS['vwap'] if last_close >= last_vwap else 0.0
    except Exception:
        comp['vwap'] = 0.0

    # 14. volatility
    comp['volatility'] = WEIGHTS['volatility']  # small constant allocation (could be refined)

    # 15. correlation with BTC (if symbol != BTC)
    corr_score = 0.0
    if symbol != "BTCUSDT":
        try:
            btc_df = exchange.get_klines("BTCUSDT", interval=interval, limit=200)
            if not btc_df.empty:
                corr = float(np.corrcoef(df['close'].tail(100), btc_df['close'].tail(100))[0, 1])
                if corr > 0.6:
                    corr_score = WEIGHTS['correlation']
                elif corr > 0.3:
                    corr_score = WEIGHTS['correlation'] / 2
        except Exception:
            corr_score = 0.0
    comp['correlation'] = corr_score

    # 16. monthly/weekly bias (approx)
    try:
        if len(df) >= 200:
            long_ema50 = indicators.ema(df['close'], 50).iloc[-1]
            long_ema200 = indicators.ema(df['close'], 200).iloc[-1]
            comp['monthly_bias'] = WEIGHTS['monthly_bias'] if long_ema50 > long_ema200 else 0.0
        else:
            comp['monthly_bias'] = WEIGHTS['monthly_bias'] / 2
    except Exception:
        comp['monthly_bias'] = 0.0

    # 17. long/short ratio (proxy)
    comp['long_short_ratio'] = WEIGHTS['long_short_ratio'] if abs(buy_ratio - 0.5) < 0.2 else WEIGHTS['long_short_ratio'] / 2

    # 18. session/time-of-day
    utc_hour = pd.Timestamp.utcnow().hour
    comp['session'] = WEIGHTS['session'] if 12 <= utc_hour <= 16 else 0.0

    # 19. position sizing (optional flag)
    comp['position_sizing'] = WEIGHTS['position_sizing'] if getattr(config, "ENABLE_POS_SIZING", False) else 0.0

    # 20. psychology (user confirmed)
    comp['psych'] = WEIGHTS['psych'] if getattr(config, "USER_CONFIRMED", True) else 0.0

    # 21. backtest flag
    comp['backtest'] = WEIGHTS['backtest'] if getattr(config, "HAS_BACKTEST", False) else 0.0

    # 22. regime detection (ATR relative)
    avg_tr = tr.tail(50).mean() if len(tr) > 0 else last_atr
    comp['regime'] = WEIGHTS['regime'] if last_atr < (avg_tr * 1.1) else 0.0

    # 23. asset-specific rules
    comp['asset_specific'] = WEIGHTS['asset_specific']

    # 24. smc proxy
    comp['smc'] = WEIGHTS['smc'] / 2 if (wick_bottom > last_atr * 0.6 or wick_top > last_atr * 0.6) else 0.0

    # 25. ml placeholder
    comp['ml'] = WEIGHTS['ml'] if getattr(config, "ENABLE_ML", False) else 0.0

    return comp


def build_signal_from_components(symbol: str, df: pd.DataFrame, comp: Dict[str, float]) -> Dict[str, Any]:
    """
    Compose final result dictionary from components and df.
    """
    total_raw = sum(comp.values())
    score_norm = normalize_score(total_raw)

    last_price = float(df['close'].iloc[-1]) if not df.empty else None
    support, resistance = compute_support_resistance(df)

    # Determine direction (simple)
    ema50 = indicators.ema(df['close'], 50).iloc[-1] if len(df) > 0 else last_price
    ema200 = indicators.ema(df['close'], 200).iloc[-1] if len(df) > 0 else last_price
    direction = "LONG" if ema50 >= ema200 else "SHORT"

    # ATR-based SL/TP
    tr = (df['high'] - df['low']).rolling(14).mean()
    last_atr = float(tr.iloc[-1]) if len(tr) > 0 else max(0.0, last_price * 0.001)

    if direction == "LONG":
        sl = round(last_price - 1.5 * last_atr, 6)
        tp1 = round(last_price + 1.0 * last_atr, 6)
        tp2 = round(last_price + 2.0 * last_atr, 6)
    else:
        sl = round(last_price + 1.5 * last_atr, 6)
        tp1 = round(last_price - 1.0 * last_atr, 6)
        tp2 = round(last_price - 2.0 * last_atr, 6)

    # risk-reward
    try:
        if direction == "LONG":
            rr = abs((tp1 - last_price) / (last_price - sl + 1e-9))
        else:
            rr = abs((last_price - tp1) / (sl - last_price + 1e-9))
        rr = round(rr, 2)
    except Exception:
        rr = None

    indicators_summary = {
        "rsi": round(float(indicators.rsi(df['close']).iloc[-1]) if len(df) > 0 else 0.0, 2),
        "macd_hist": round(float(indicators.macd(df['close'])[2].iloc[-1]) if len(df) > 0 else 0.0, 6),
        "ema50": round(float(indicators.ema(df['close'], 50).iloc[-1]) if len(df) > 0 else 0.0, 6),
        "ema200": round(float(indicators.ema(df['close'], 200).iloc[-1]) if len(df) > 0 else 0.0, 6),
        "atr": round(last_atr, 6)
    }

    result = {
        "symbol": symbol,
        "score_raw": round(total_raw, 2),
        "score_norm": score_norm,
        "direction": direction,
        "last_price": round(last_price, 6) if last_price is not None else None,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "rr": rr,
        "support": round(support, 6),
        "resistance": round(resistance, 6),
        "funding": exchange.get_funding_rate(symbol),
        "open_interest": exchange.get_open_interest(symbol),
        "components": comp,
        "indicators": indicators_summary
    }
    return result


def score_symbol(symbol: str, interval: str = "1h") -> Dict[str, Any]:
    """
    Top-level function:
    - fetch klines
    - compute components
    - build and return final signal dict
    """
    try:
        df = exchange.get_klines(symbol, interval=interval, limit=500)
    except Exception as e:
        LOG.exception("Error fetching klines for %s: %s", symbol, e)
        df = pd.DataFrame()

    if df.empty:
        # return minimal structure
        return {
            "symbol": symbol,
            "score_raw": 0.0,
            "score_norm": 0.0,
            "direction": "NEUTRAL",
            "last_price": None,
            "sl": None,
            "tp1": None,
            "tp2": None,
            "rr": None,
            "support": None,
            "resistance": None,
            "funding": None,
            "open_interest": None,
            "components": {},
            "indicators": {}
        }

    comp = score_components_from_df(symbol, df, interval=interval)
    res = build_signal_from_components(symbol, df, comp)
    return res
