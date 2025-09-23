import pandas as pd
import numpy as np

# ==============================
# Moving Averages
# ==============================
def ema(data: pd.Series, period: int = 14) -> pd.Series:
    """Exponential Moving Average"""
    return data.ewm(span=period, adjust=False).mean()

def sma(data: pd.Series, period: int = 14) -> pd.Series:
    """Simple Moving Average"""
    return data.rolling(window=period).mean()

# ==============================
# Relative Strength Index (RSI)
# ==============================
def rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index"""
    delta = data.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    gain = pd.Series(gain).rolling(window=period).mean()
    loss = pd.Series(loss).rolling(window=period).mean()

    rs = gain / (loss + 1e-10)  # avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ==============================
# MACD (Moving Average Convergence Divergence)
# ==============================
def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD Indicator"""
    ema_fast = ema(data, fast)
    ema_slow = ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

# ==============================
# Volume-based Indicator (On-Balance Volume)
# ==============================
def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume"""
    obv = [0]
    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            obv.append(obv[-1] + volume[i])
        elif close[i] < close[i - 1]:
            obv.append(obv[-1] - volume[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=close.index)

# ==============================
# Bollinger Bands
# ==============================
def bollinger_bands(data: pd.Series, period: int = 20, std_factor: float = 2.0):
    """Bollinger Bands"""
    sma_line = sma(data, period)
    std = data.rolling(window=period).std()
    upper = sma_line + (std_factor * std)
    lower = sma_line - (std_factor * std)
    return upper, sma_line, lower
