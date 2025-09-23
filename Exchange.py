import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
import time
import logging
import config

# تنظیم کلاینت بایننس
client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)

def get_klines(symbol: str, interval: str = "15m", limit: int = 200) -> pd.DataFrame:
    """
    گرفتن کندل (OHLCV) از بایننس
    symbol: مثل BTCUSDT
    interval: تایم‌فریم (مثل "15m", "1h", "4h", "1d")
    limit: تعداد کندل‌ها
    """
    try:
        raw = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    except BinanceAPIException as e:
        logging.error(f"Binance API error: {e}")
        time.sleep(2)
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return pd.DataFrame()

    # تبدیل دیتا به DataFrame
    df = pd.DataFrame(raw, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])

    # تبدیل انواع داده
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)

    return df[["timestamp", "open", "high", "low", "close", "volume"]]

def get_last_price(symbol: str) -> float:
    """ آخرین قیمت """
    try:
        ticker = client.get_symbol_ticker(symbol=symbol)
        return float(ticker["price"])
    except Exception as e:
        logging.error(f"Error fetching last price for {symbol}: {e}")
        return None

def get_all_symbols(quote: str = "USDT") -> list:
    """ گرفتن لیست کل ارزهای بازار که به USDT جفت شدن """
    try:
        info = client.get_exchange_info()
        symbols = [
            s["symbol"] for s in info["symbols"]
            if s["quoteAsset"] == quote and s["status"] == "TRADING"
        ]
        return symbols
    except Exception as e:
        logging.error(f"Error fetching symbols: {e}")
        return []
