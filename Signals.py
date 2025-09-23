# exchange.py
import time
import logging
from typing import Optional, List, Dict, Any

import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException

import config  # your config.py should define BINANCE_API_KEY and BINANCE_API_SECRET

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def _make_client() -> Client:
    """
    Create Binance client using keys from config.
    Public endpoints work without keys, but private endpoints need keys.
    """
    api_key = getattr(config, "BINANCE_API_KEY", "") or None
    api_secret = getattr(config, "BINANCE_API_SECRET", "") or None
    return Client(api_key, api_secret)


_client = _make_client()


def _retry(func, *args, retries: int = 3, delay: float = 1.0, **kwargs):
    """
    Simple retry wrapper for API calls.
    """
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return func(*args, **kwargs)
        except BinanceAPIException as e:
            last_exc = e
            LOG.warning("BinanceAPIException attempt %s/%s: %s", attempt, retries, e)
        except Exception as e:
            last_exc = e
            LOG.warning("Exception attempt %s/%s: %s", attempt, retries, e)
        time.sleep(delay * attempt)
    LOG.error("All retries failed for %s: %s", func.__name__, last_exc)
    raise last_exc


def get_klines(symbol: str, interval: str = "15m", limit: int = 500) -> pd.DataFrame:
    """
    Return OHLCV klines as a pandas DataFrame (timestamp, open, high, low, close, volume).
    Tries futures klines first; falls back to spot klines.
    """
    df = pd.DataFrame()
    try:
        # Try futures endpoint first (suitable for futures symbols)
        raw = _retry(_client.futures_klines, symbol=symbol, interval=interval, limit=limit)
    except Exception:
        try:
            raw = _retry(_client.get_klines, symbol=symbol, interval=interval, limit=limit)
        except Exception as e:
            LOG.error("Failed to fetch klines for %s: %s", symbol, e)
            return df

    if not raw:
        return df

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "trade_count",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    try:
        df = pd.DataFrame(raw, columns=cols)
    except Exception:
        # When Binance returns slightly different structure, create DataFrame generically
        df = pd.DataFrame(raw)

    # Convert types
    if "open_time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
    elif 0 in df.columns:
        df["timestamp"] = pd.to_datetime(df[0], unit="ms")
    else:
        df["timestamp"] = pd.NaT

    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            # fallback by position in returned list
            try:
                idx = {"open": 1, "high": 2, "low": 3, "close": 4, "volume": 5}[col]
                df[col] = pd.to_numeric(df[idx], errors="coerce")
            except Exception:
                df[col] = pd.NA

    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
    return df


def get_last_price(symbol: str) -> Optional[float]:
    """
    Return the last price for the symbol.
    Tries futures ticker then spot ticker.
    """
    try:
        res = _retry(_client.futures_symbol_ticker, symbol=symbol)
        price = float(res.get("price"))
        return price
    except Exception:
        try:
            res = _retry(_client.get_symbol_ticker, symbol=symbol)
            price = float(res.get("price"))
            return price
        except Exception as e:
            LOG.error("Failed to fetch last price for %s: %s", symbol, e)
            return None


def get_all_symbols(quote_asset: str = "USDT") -> List[str]:
    """
    Return list of tradable symbols that have quoteAsset == quote_asset.
    """
    try:
        info = _retry(_client.get_exchange_info)
        symbols = [s["symbol"] for s in info.get("symbols", []) if s.get("quoteAsset") == quote_asset and s.get("status") == "TRADING"]
        return symbols
    except Exception as e:
        LOG.error("Failed to fetch exchange info: %s", e)
        return []


def get_order_book(symbol: str, limit: int = 5) -> Dict[str, Any]:
    """
    Return a dict with bids and asks (list of [price, qty]) using futures order book first then spot.
    """
    try:
        res = _retry(_client.futures_order_book, symbol=symbol, limit=limit)
    except Exception:
        try:
            res = _retry(_client.get_order_book, symbol=symbol, limit=limit)
        except Exception as e:
            LOG.error("Failed to fetch order book for %s: %s", symbol, e)
            return {"bids": [], "asks": []}
    return {"bids": res.get("bids", []), "asks": res.get("asks", [])}


def get_funding_rate(symbol: str) -> Optional[float]:
    """
    Return latest funding rate for a futures symbol (as float). Returns None if not available.
    """
    try:
        # futures_funding_rate returns list of funding rate records (most recent last)
        res = _retry(_client.futures_funding_rate, symbol=symbol, limit=1)
        if isinstance(res, list) and len(res) > 0:
            fr = res[0].get("fundingRate") or res[0].get("fundingRate")
            return float(fr)
        elif isinstance(res, dict) and "fundingRate" in res:
            return float(res.get("fundingRate"))
    except Exception:
        LOG.debug("Failed to fetch funding rate for %s (maybe not a futures symbol)", symbol)
    return None


def get_open_interest(symbol: str) -> Optional[float]:
    """
    Return open interest for a futures symbol. Returns None on failure.
    """
    try:
        res = _retry(_client.futures_open_interest, symbol=symbol)
        oi = res.get("openInterest")
        return float(oi)
    except Exception:
        LOG.debug("Failed to fetch open interest for %s", symbol)
    return None
