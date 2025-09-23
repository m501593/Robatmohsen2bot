# main.py
# Single-file Binance futures scanner + Telegram notifier (webhook-ready)
# Code uses English identifiers/comments. Telegram messages in Persian.
# Inserted TELEGRAM_TOKEN and ADMIN_ID as provided by user.

import os
import time
import math
import json
import sqlite3
import logging
import threading
from datetime import datetime
from typing import Dict, Any, List

import requests
import numpy as np
import pandas as pd
import ta
from flask import Flask, request, jsonify

# ================== CONFIG (user-provided token & chat id inserted) ==================
TELEGRAM_TOKEN = "7993216439:AAHKSHJMHrcnfEcedw54aetp1JPxZ83Ks4M"  # provided by user
ADMIN_ID = 84544682  # provided by user (numeric chat id)

# Webhook / hosting config
# Best: set WEBHOOK_HOST as environment variable to your service URL (e.g. https://your-render-app.onrender.com)
WEBHOOK_HOST = os.getenv("WEBHOOK_HOST", "")  # if empty, webhook won't be set automatically
WEBHOOK_PATH = "/webhook"
WEBHOOK_URL = f"{WEBHOOK_HOST}{WEBHOOK_PATH}" if WEBHOOK_HOST else ""

# App host/port (Render provides PORT env variable)
WEBAPP_HOST = "0.0.0.0"
WEBAPP_PORT = int(os.getenv("PORT", "10000"))

# Binance endpoints
BINANCE_SPOT = "https://api.binance.com"
BINANCE_FUT = "https://fapi.binance.com"

# runtime params
TIMEFRAMES = ["15m", "1h", "4h", "1d"]
AUTO_SCAN_TOP_N = int(os.getenv("AUTO_SCAN_TOP_N", "100"))
PROBABILITY_THRESHOLD = int(os.getenv("PROBABILITY_THRESHOLD", "50"))
SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", "60"))
WATCHLIST_ENV = os.getenv("WATCHLIST", "")
WATCHLIST = [s.strip().upper() for s in WATCHLIST_ENV.split(",") if s.strip()] or ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","LINKUSDT"]

MAX_SCORE = 134
SCORE_STRONG = 100
SCORE_GOOD = 80
SCORE_WEAK = 60

DB_PATH = os.getenv("DB_PATH", "signals.db")
ENABLE_AUTO_SCAN = os.getenv("ENABLE_AUTO_SCAN", "True").lower() in ("1","true","yes")
ENABLE_WEBHOOK = os.getenv("ENABLE_WEBHOOK", "True").lower() in ("1","true","yes")

# Weight map (approximate; sum near MAX_SCORE)
WEIGHTS = {
    "trend": 15, "structure": 12, "price_action": 12, "volume": 10, "rsi": 8,
    "macd": 8, "multi_tf": 8, "funding_oi": 8, "news": 7, "liquidity": 7,
    "order_flow": 5, "liquidity_pool": 5, "vwap": 3, "volatility": 3, "correlation": 5,
    "monthly_bias": 5, "longshort_ratio": 5, "time_of_day": 3,
    "position_sizing": 3, "psych": 2, "backtest": 2, "regime": 3, "asset_specific": 2,
    "smart_money": 3, "ml": 2
}

# ---------------- logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("scanner")

# ---------------- DB ----------------
def init_db(path: str = DB_PATH):
    conn = sqlite3.connect(path, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            symbol TEXT,
            score REAL,
            category TEXT,
            direction TEXT,
            probability INTEGER,
            details TEXT
        )
    """)
    conn.commit()
    return conn

DB_CONN = init_db(DB_PATH)

def save_signal(symbol: str, score: float, category: str, direction: str, probability: int, details: Dict[str,Any]):
    try:
        cur = DB_CONN.cursor()
        cur.execute(
            "INSERT INTO signals (ts, symbol, score, category, direction, probability, details) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (datetime.utcnow().isoformat(), symbol, score, category, direction, probability, json.dumps(details, ensure_ascii=False))
        )
        DB_CONN.commit()
    except Exception:
        logger.exception("Failed to save signal")

# ---------------- HTTP helper ----------------
def safe_get(url: str, params: dict = None, timeout: int = 15):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        logger.debug("HTTP GET failed: %s %s", url, params, exc_info=True)
        return None

# ---------------- Binance data fetchers ----------------
def get_klines(symbol: str, interval: str="15m", limit: int=200, futures: bool = True) -> pd.DataFrame:
    base = BINANCE_FUT if futures else BINANCE_SPOT
    url = f"{base}/api/v3/klines"
    res = safe_get(url, params={"symbol": symbol, "interval": interval, "limit": limit})
    if not res:
        raise RuntimeError(f"No klines for {symbol}@{interval}")
    df = pd.DataFrame(res, columns=[
        "open_time","open","high","low","close","volume","close_time","quote_av","trades","taker_base","taker_quote","ignore"
    ])
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df

def get_24hr_tickers():
    url = f"{BINANCE_SPOT}/api/v3/ticker/24hr"
    res = safe_get(url)
    return res or []

def get_funding_rate(symbol: str):
    url = f"{BINANCE_FUT}/fapi/v1/premiumIndex"
    res = safe_get(url, params={"symbol": symbol})
    if res and isinstance(res, dict):
        return float(res.get("lastFundingRate", 0.0))
    return None

def get_open_interest(symbol: str):
    url = f"{BINANCE_FUT}/fapi/v1/openInterest"
    res = safe_get(url, params={"symbol": symbol})
    if res and "openInterest" in res:
        return float(res["openInterest"])
    return None

def get_order_book(symbol: str, limit: int = 20):
    url = f"{BINANCE_SPOT}/api/v3/depth"
    res = safe_get(url, params={"symbol": symbol, "limit": limit})
    return res

# ---------------- indicators ----------------
def ema(series: pd.Series, window: int):
    return ta.trend.EMAIndicator(series, window=window).ema_indicator()

def rsi(series: pd.Series, window: int=14):
    return ta.momentum.RSIIndicator(series, window=window).rsi()

def macd_diff(series: pd.Series):
    return ta.trend.MACD(series).macd_diff()

def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int=14):
    return ta.volatility.AverageTrueRange(high, low, close, window=window).average_true_range()

def vwap(df: pd.DataFrame):
    typical = (df["high"] + df["low"] + df["close"]) / 3
    cumtyp = (typical * df["volume"]).cumsum()
    cumvol = df["volume"].cumsum()
    return cumtyp / cumvol

# ---------------- scoring (25 metrics simplified) ----------------
def score_symbol(symbol: str) -> Dict[str,Any]:
    """
    Compute composite score across multiple heuristics.
    Returns dict:
      {score, category, direction, probability, entry, sl, tp1, tp2, details}
    """
    try:
        # fetch base timeframe
        base_tf = TIMEFRAMES[0]
        df = get_klines(symbol, interval=base_tf, limit=200, futures=True)
        if df is None or df.empty:
            return {"error": "no data"}

        last_price = float(df["close"].iloc[-1])

        # collect higher timeframe dfs
        dfs = {tf: None for tf in TIMEFRAMES}
        dfs[base_tf] = df
        for tf in TIMEFRAMES[1:]:
            try:
                dfs[tf] = get_klines(symbol, interval=tf, limit=200, futures=True)
            except Exception:
                dfs[tf] = None

        total = 0.0
        components = {}

        # metric 1: trend (HTF EMA200 vs price)
        trend_score = 0.0
        htf = dfs.get("4h") or dfs.get("1d") or dfs.get("1h")
        if htf is not None:
            try:
                ema50 = ema(htf["close"], 50).iloc[-1]
                ema200 = ema(htf["close"], 200).iloc[-1]
                if ema50 > ema200 and htf["close"].iloc[-1] > ema200:
                    trend_score = WEIGHTS["trend"]
                elif abs(htf["close"].iloc[-1] - ema200) / ema200 < 0.02:
                    trend_score = WEIGHTS["trend"] * 0.5
                else:
                    trend_score = 0.0
            except Exception:
                trend_score = WEIGHTS["trend"] * 0.3
        else:
            trend_score = WEIGHTS["trend"] * 0.3
        components["trend"] = round(trend_score,3)
        total += trend_score

        # metric 2: structure (HH/HL or LH/LL simple)
        structure_score = 0.0
        try:
            highs = htf["high"].tail(6) if htf is not None else df["high"].tail(6)
            lows = htf["low"].tail(6) if htf is not None else df["low"].tail(6)
            if len(highs) >= 3:
                if highs.iloc[-1] > highs.iloc[-2] and lows.iloc[-1] > lows.iloc[-2]:
                    structure_score = WEIGHTS["structure"]
                elif highs.iloc[-1] < highs.iloc[-2] and lows.iloc[-1] < lows.iloc[-2]:
                    structure_score = 0.0
                else:
                    structure_score = WEIGHTS["structure"] * 0.5
            else:
                structure_score = WEIGHTS["structure"] * 0.4
        except Exception:
            structure_score = WEIGHTS["structure"] * 0.4
        components["structure"] = round(structure_score,3)
        total += structure_score

        # metric 3: price action
        pa_score = 0.0
        try:
            o1 = df["open"].iloc[-1]; c1 = df["close"].iloc[-1]
            o2 = df["open"].iloc[-2]; c2 = df["close"].iloc[-2]
            # bullish engulf
            if (c1 > o1) and (c2 < o2) and (c1 > o2) and (o1 < c2):
                pa_score += WEIGHTS["price_action"] * 0.9
            # proximity to recent support
            recent_low = df["low"].rolling(window=20).min().iloc[-2]
            recent_high = df["high"].rolling(window=20).max().iloc[-2]
            if recent_low and abs(last_price - recent_low) / recent_low < 0.01:
                pa_score += WEIGHTS["price_action"] * 0.6
            if recent_high and abs(last_price - recent_high) / recent_high < 0.01:
                pa_score += 0.0
        except Exception:
            pa_score = WEIGHTS["price_action"] * 0.3
        components["price_action"] = round(pa_score,3)
        total += pa_score

        # metric 4: volume
        vol_score = 0.0
        try:
            vol = df["volume"]
            sma20 = vol.rolling(window=20).mean().iloc[-1]
            if sma20 and vol.iloc[-1] > sma20 * 1.2:
                vol_score = WEIGHTS["volume"]
            else:
                vol_score = WEIGHTS["volume"] * 0.6
        except Exception:
            vol_score = WEIGHTS["volume"] * 0.5
        components["volume"] = round(vol_score,3)
        total += vol_score

        # metric 5: RSI
        rsi_score = 0.0
        try:
            rsi_val = rsi(df["close"], window=14).iloc[-1]
            if rsi_val > 70:
                rsi_score = -WEIGHTS["rsi"] * 0.5
            elif 55 <= rsi_val <= 65:
                rsi_score = WEIGHTS["rsi"]
            elif rsi_val < 30:
                rsi_score = WEIGHTS["rsi"] * 1.2
            else:
                rsi_score = WEIGHTS["rsi"] * 0.3
        except Exception:
            rsi_score = WEIGHTS["rsi"] * 0.2
            rsi_val = None
        components["rsi"] = round(rsi_score,3)
        components["rsi_value"] = round(float(rsi_val) if rsi_val is not None else 0,2)
        total += max(0, rsi_score)  # negative rsi_score handled later

        # metric 6: MACD
        macd_score = 0.0
        try:
            md = macd_diff(df["close"]).iloc[-1]
            macd_score = WEIGHTS["macd"] if md > 0 else WEIGHTS["macd"] * 0.2
        except Exception:
            macd_score = WEIGHTS["macd"] * 0.3
        components["macd"] = round(macd_score,3)
        total += macd_score

        # metric 7: multi-timeframe alignment
        mtf_score = 0.0
        try:
            aligns = 0; checks = 0
            for tf in ["1h","4h"]:
                d = dfs.get(tf)
                if d is None: continue
                checks += 1
                e50 = ema(d["close"], 50).iloc[-1]
                e200 = ema(d["close"], 200).iloc[-1]
                if e50 > e200: aligns += 1
            if checks > 0:
                mtf_score = WEIGHTS["multi_tf"] * (aligns / checks)
            else:
                mtf_score = WEIGHTS["multi_tf"] * 0.5
        except Exception:
            mtf_score = WEIGHTS["multi_tf"] * 0.4
        components["multi_tf"] = round(mtf_score,3)
        total += mtf_score

        # metric 8: funding + open interest
        f_score = 0.0
        try:
            fr = get_funding_rate(symbol)
            if fr is None:
                f_score = WEIGHTS["funding_oi"] * 0.6
            else:
                if abs(fr) < 0.0005:
                    f_score = WEIGHTS["funding_oi"]
                elif abs(fr) >= 0.001:
                    f_score = WEIGHTS["funding_oi"] * 0.3
                else:
                    f_score = WEIGHTS["funding_oi"] * 0.6
        except Exception:
            f_score = WEIGHTS["funding_oi"] * 0.5
        components["funding_oi"] = round(f_score,4)
        total += f_score

        # metric 9: news placeholder
        news_score = WEIGHTS["news"] * 0.6
        components["news"] = round(news_score,3)
        total += news_score

        # metric 10: liquidity / order book
        liq_score = 0.0
        try:
            ob = get_order_book(symbol, limit=20)
            if ob and "bids" in ob and "asks" in ob:
                bid_depth = sum(float(b[1]) for b in ob["bids"][:10])
                ask_depth = sum(float(a[1]) for a in ob["asks"][:10])
                depth_ratio = min(bid_depth, ask_depth) / max(1.0, (bid_depth + ask_depth))
                liq_score = WEIGHTS["liquidity"] * (0.5 + depth_ratio)
            else:
                liq_score = WEIGHTS["liquidity"] * 0.6
        except Exception:
            liq_score = WEIGHTS["liquidity"] * 0.5
        components["liquidity"] = round(liq_score,3)
        total += liq_score

        # Section B (11..18) simplified implementations (order_flow, liquidity_pool, vwap, volatility, correlation, monthly_bias, longshort_ratio, time_of_day)
        # 11. order_flow proxy
        of_score = 0.0
        try:
            df1m = get_klines(symbol, interval="1m", limit=60, futures=True)
            green = (df1m["close"] > df1m["open"]).sum()
            red = (df1m["close"] < df1m["open"]).sum()
            of_score = WEIGHTS["order_flow"] if green > red else WEIGHTS["order_flow"] * 0.4
        except Exception:
            of_score = WEIGHTS["order_flow"] * 0.4
        components["order_flow"] = round(of_score,3)
        total += of_score

        # 12. liquidity_pool / stop-hunt proxy
        lp_score = 0.0
        try:
            body = abs(df["close"].iloc[-1] - df["open"].iloc[-1])
            wick_bottom = min(df["close"].iloc[-1], df["open"].iloc[-1]) - df["low"].iloc[-1]
            wick_top = df["high"].iloc[-1] - max(df["close"].iloc[-1], df["open"].iloc[-1])
            if wick_bottom > body * 2:
                lp_score = WEIGHTS["liquidity_pool"] * 0.8
            elif wick_top > body * 2:
                lp_score = WEIGHTS["liquidity_pool"] * 0.2
            else:
                lp_score = WEIGHTS["liquidity_pool"] * 0.5
        except Exception:
            lp_score = WEIGHTS["liquidity_pool"] * 0.4
        components["liquidity_pool"] = round(lp_score,3)
        total += lp_score

        # 13. VWAP
        vwap_score = 0.0
        try:
            dfh = get_klines(symbol, interval="1h", limit=24, futures=True)
            vw = vwap(dfh).iloc[-1]
            vwap_score = WEIGHTS["vwap"] if last_price > vw else WEIGHTS["vwap"] * 0.3
        except Exception:
            vwap_score = WEIGHTS["vwap"] * 0.5
        components["vwap"] = round(vwap_score,3)
        total += vwap_score

        # 14. volatility (ATR relative)
        vol_score = 0.0
        try:
            atr_val = atr(df["high"], df["low"], df["close"], window=14).iloc[-1]
            rel = atr_val / last_price if last_price else 0
            vol_score = WEIGHTS["volatility"] if rel > 0.02 else WEIGHTS["volatility"] * 0.6
        except Exception:
            vol_score = WEIGHTS["volatility"] * 0.5
        components["volatility"] = round(vol_score,4)
        total += vol_score

        # 15. correlation to BTC (for alts)
        corr_score = 0.0
        try:
            if not symbol.startswith("BTC"):
                btc = get_klines("BTCUSDT", interval=base_tf, limit=len(df), futures=True)
                ret_sym = df["close"].pct_change().dropna()
                ret_btc = btc["close"].pct_change().dropna()
                mn = min(len(ret_sym), len(ret_btc))
                if mn > 5:
                    corr = ret_sym.tail(mn).corr(ret_btc.tail(mn))
                    corr_score = WEIGHTS["correlation"] if corr > 0.6 else WEIGHTS["correlation"] * 0.5
                else:
                    corr_score = WEIGHTS["correlation"] * 0.3
            else:
                corr_score = WEIGHTS["correlation"] * 0.6
        except Exception:
            corr_score = WEIGHTS["correlation"] * 0.4
        components["correlation"] = round(corr_score,3)
        total += corr_score

        # 16. monthly/weekly bias
        mb_score = WEIGHTS["monthly_bias"] * 0.5
        try:
            if dfs.get("1d") is not None:
                ma30 = dfs["1d"]["close"].rolling(window=30).mean().iloc[-1]
                mb_score = WEIGHTS["monthly_bias"] if dfs["1d"]["close"].iloc[-1] > ma30 else WEIGHTS["monthly_bias"] * 0.3
        except Exception:
            mb_score = WEIGHTS["monthly_bias"] * 0.4
        components["monthly_bias"] = round(mb_score,3)
        total += mb_score

        # 17. long/short ratio (proxy)
        lsr = WEIGHTS["longshort_ratio"] * 0.6
        components["longshort_ratio"] = round(lsr,3)
        total += lsr

        # 18. time-of-day
        tod = WEIGHTS["time_of_day"] if 13 <= datetime.utcnow().hour <= 17 else WEIGHTS["time_of_day"] * 0.4
        components["time_of_day"] = round(tod,3)
        total += tod

        # Section C (19..25) simplified
        components["position_sizing"] = round(WEIGHTS["position_sizing"] * 0.6,3); total += WEIGHTS["position_sizing"] * 0.6
        components["psych"] = round(WEIGHTS["psych"] * 0.8,3); total += WEIGHTS["psych"] * 0.8
        components["backtest"] = round(WEIGHTS["backtest"] * 0.2,3); total += WEIGHTS["backtest"] * 0.2
        components["regime"] = round(WEIGHTS["regime"] * 0.6,3); total += WEIGHTS["regime"] * 0.6
        components["asset_specific"] = round(WEIGHTS["asset_specific"] * 0.7,3); total += WEIGHTS["asset_specific"] * 0.7
        components["smart_money"] = round(WEIGHTS["smart_money"] * 0.6,3); total += WEIGHTS["smart_money"] * 0.6
        components["ml"] = round(WEIGHTS["ml"] * 0.2,3); total += WEIGHTS["ml"] * 0.2

        # clamp
        total = max(0.0, min(total, MAX_SCORE))
        probability = int(min(100, round((total / MAX_SCORE) * 100)))
        if total >= SCORE_STRONG:
            category = "STRONG"
        elif total >= SCORE_GOOD:
            category = "GOOD"
        elif total >= SCORE_WEAK:
            category = "WEAK"
        else:
            category = "NO_TRADE"

        # direction: heuristics from components
        long_votes = 0; short_votes = 0
        # use several indicators to decide direction
        try:
            if components.get("rsi_value",0) >= 55: long_votes += 1
            if components.get("macd",0) >= WEIGHTS["macd"] * 0.8: long_votes += 1
            if components.get("vwap",0) >= WEIGHTS["vwap"] * 0.8: long_votes += 1
        except Exception:
            pass
        # quick fallback using price vs EMA20
        try:
            ema20 = ema(df["close"], 20).iloc[-1]
            if last_price > ema20: long_votes += 1
            else: short_votes += 1
        except Exception:
            pass
        direction = "NONE"
        if long_votes > short_votes: direction = "LONG"
        elif short_votes > long_votes: direction = "SHORT"

        # entry/SL/TP based on ATR
        try:
            atr_val = atr(df["high"], df["low"], df["close"], window=14).iloc[-1]
            entry = round(last_price, 8)
            if direction == "LONG":
                sl = round(entry - 2 * atr_val, 8)
                tp1 = round(entry + 1 * atr_val, 8)
                tp2 = round(entry + 2 * atr_val, 8)
            elif direction == "SHORT":
                sl = round(entry + 2 * atr_val, 8)
                tp1 = round(entry - 1 * atr_val, 8)
                tp2 = round(entry - 2 * atr_val, 8)
            else:
                sl = round(entry - 2 * atr_val, 8)
                tp1 = round(entry + 1 * atr_val, 8)
                tp2 = round(entry + 2 * atr_val, 8)
        except Exception:
            entry = round(last_price, 8)
            sl = round(last_price * 0.98, 8)
            tp1 = round(last_price * 1.01, 8)
            tp2 = round(last_price * 1.02, 8)

        details = {
            "components": components,
            "last_price": last_price
        }

        # save and return
        save_signal(symbol, total, category, direction, probability, details)

        return {
            "symbol": symbol,
            "score": round(total,2),
            "probability": probability,
            "category": category,
            "direction": direction,
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "details": details
        }

    except Exception as e:
        logger.exception("score_symbol error: %s", e)
        return {"error": str(e)}

# ---------------- message formatting (Persian) ----------------
def format_message(symbol: str, out: Dict[str,Any]) -> str:
    if "error" in out:
        return f"⚠️ خطا در بررسی {symbol}: {out['error']}"
    dir_label = "بی‌تصمیم"
    if out["direction"] == "LONG": dir_label = "لانگ (خرید)"
    if out["direction"] == "SHORT": dir_label = "شورت (فروش)"
    header = f"📡 سیگنال برای: {symbol}\n⏰ تایم‌فریم: {TIMEFRAMES[0]}\n⚖️ وضعیت بازار: {dir_label}\n📊 امتیاز: {out['score']}/{MAX_SCORE}\n💯 درصد احتمال: {out['probability']}%\n"
    body = f"➡️ ورود کلی: {out['entry']}\n🛑 حد ضرر کلی: {out['sl']}\n🎯 حد سود کلی (TP1): {out['tp1']}\n🎯 حد سود کلی (TP2): {out['tp2']}\n"
    components = out.get("details", {}).get("components", {})
    comp_lines = "\n".join([f"{k}: {v}" for k,v in list(components.items())[:6]])
    footer = "\n⚠️ این سیگنال صرفاً تحلیلی است؛ مسئولیت معامله با شماست."
    return header + body + "\n📊 خلاصه اندیکاتورها:\n" + comp_lines + footer

# ---------------- Telegram helpers ----------------
TG_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

def tg_send(chat_id: int, text: str):
    try:
        url = f"{TG_API}/sendMessage"
        res = requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}, timeout=15)
        if not res.ok:
            logger.error("Telegram send failed: %s %s", res.status_code, res.text)
        return res
    except Exception:
        logger.exception("Failed to send tg message")
        return None

def set_telegram_webhook():
    if not WEBHOOK_URL:
        logger.warning("WEBHOOK_URL not set; skipping setWebhook")
        return False
    try:
        url = f"{TG_API}/setWebhook"
        r = requests.post(url, json={"url": WEBHOOK_URL}, timeout=15)
        logger.info("setWebhook response: %s", r.text)
        return r.ok
    except Exception:
        logger.exception("setWebhook failed")
        return False

# ---------------- auto-scan loop ----------------
def get_top_pairs(quote="USDT", top_n=100) -> List[str]:
    tickers = get_24hr_tickers()
    if not tickers:
        return WATCHLIST
    filt = [t for t in tickers if t.get("symbol","").endswith(quote)]
    sorted_pairs = sorted(filt, key=lambda x: float(x.get("quoteVolume",0.0)), reverse=True)
    return [p["symbol"] for p in sorted_pairs[:top_n]]

def auto_scan_worker():
    logger.info("Auto-scan worker started (top_n=%s interval=%s)", AUTO_SCAN_TOP_N, SCAN_INTERVAL_SECONDS)
    while True:
        try:
            pairs = get_top_pairs(top_n=AUTO_SCAN_TOP_N) if AUTO_SCAN_TOP_N > 0 else WATCHLIST
            for s in pairs:
                try:
                    out = score_symbol(s)
                    if isinstance(out, dict) and "error" not in out:
                        if out["category"] in ("STRONG","GOOD") or out["probability"] >= PROBABILITY_THRESHOLD:
                            msg = format_message(s, out)
                            tg_send(ADMIN_ID, msg)
                except Exception:
                    logger.exception("Error scanning %s", s)
                time.sleep(0.25)
        except Exception:
            logger.exception("Auto-scan outer loop error")
        time.sleep(SCAN_INTERVAL_SECONDS)

# ---------------- Flask webhook server ----------------
app = Flask(__name__)

@app.route(WEBHOOK_PATH, methods=["POST"])
def webhook():
    try:
        data = request.get_json(force=True)
        if "message" in data:
            msg = data["message"]
            chat_id = msg["chat"]["id"]
            text = msg.get("text","").strip()
            if text.startswith("/scan"):
                parts = text.split()
                if len(parts) >= 2:
                    sym = parts[1].upper()
                    tg_send(chat_id, f"🔎 در حال بررسی {sym} ...")
                    threading.Thread(target=lambda: tg_send(chat_id, format_message(sym, score_symbol(sym))), daemon=True).start()
                else:
                    tg_send(chat_id, "فرمت: /scan SYMBOL  — مثال: /scan BTCUSDT")
            elif text.startswith("/status"):
                parts = text.split()
                if len(parts) >= 2:
                    sym = parts[1].upper()
                    tg_send(chat_id, f"🔎 وضعیت {sym} ...")
                    threading.Thread(target=lambda: tg_send(chat_id, format_message(sym, score_symbol(sym))), daemon=True).start()
                else:
                    tg_send(chat_id, "فرمت: /status SYMBOL")
            elif text.startswith("/start"):
                tg_send(chat_id, "ربات آنالیز بازار فعال است. از /scan برای دریافت سیگنال استفاده کنید.")
            else:
                # if user writes symbol directly like "BTC" or "BTCUSDT"
                s = text.upper().strip()
                sym = s if s.endswith("USDT") else (s + "USDT")
                if len(sym) > 4:
                    tg_send(chat_id, f"🔎 در حال بررسی {sym} ...")
                    threading.Thread(target=lambda: tg_send(chat_id, format_message(sym, score_symbol(sym))), daemon=True).start()
                else:
                    tg_send(chat_id, "دستور نامشخص. از /scan SYMBOL استفاده کنید.")
    except Exception:
        logger.exception("webhook handler error")
    return jsonify(ok=True)

@app.route("/health", methods=["GET"])
def health():
    return "ok", 200

# ---------------- main entry ----------------
def main():
    # set webhook on Telegram if WEBHOOK_URL provided
    if ENABLE_WEBHOOK and WEBHOOK_URL:
        ok = set_telegram_webhook()
        if not ok:
            logger.warning("setWebhook returned not-OK. Ensure WEBHOOK_URL reachable by Telegram.")
    # start auto-scan thread
    if ENABLE_AUTO_SCAN:
        t = threading.Thread(target=auto_scan_worker, daemon=True)
        t.start()
    logger.info("Starting Flask server on %s:%s (webhook_path=%s)", WEBAPP_HOST, WEBAPP_PORT, WEBHOOK_PATH)
    app.run(host=WEBAPP_HOST, port=WEBAPP_PORT, threaded=True)

if __name__ == "__main__":
    main()
