# utils.py
import logging
from datetime import datetime

# Logging configuration
logging.basicConfig(
    filename="bot.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log(message: str):
    """Write log to file and print on console"""
    print(message)
    logging.info(message)

def format_signal(symbol: str, signal_data: dict) -> str:
    """
    Format signal message for Telegram
    """
    text = f"📊 New Signal\n"
    text += f"Symbol: {symbol}\n"
    text += f"Score: {signal_data.get('score', '?')}\n"
    text += f"Probability: {signal_data.get('probability', '?')}%\n"
    text += f"Entry: {signal_data.get('entry', '?')}\n"
    text += f"Stop Loss: {signal_data.get('stop_loss', '?')}\n"
    text += f"Target 1: {signal_data.get('tp1', '?')}\n"
    text += f"Target 2: {signal_data.get('tp2', '?')}\n"
    text += f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    return text

def calculate_risk_reward(entry: float, stop_loss: float, take_profit: float) -> float:
    """
    Calculate Risk/Reward ratio
    """
    if entry == stop_loss:
        return 0
    risk = abs(entry - stop_loss)
    reward = abs(take_profit - entry)
    return round(reward / risk, 2) if risk > 0 else 0
