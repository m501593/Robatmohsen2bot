import os

# ====== Telegram Bot Config ======
TELEGRAM_TOKEN = "7993216439:AAHKSHJMHrcnfEcedw"   # توکن ربات
ADMIN_ID = 84544682                                # آیدی عددی ادمین

# ====== Binance API Keys ======
# اگر فقط داده‌ی عمومی میخوای، میتونی خالی بذاری
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# ====== Webhook / Hosting ======
WEBHOOK_HOST = os.getenv("WEBHOOK_HOST", "https://robottest.onrender.com")
WEBHOOK_PATH = "/webhook"
WEBHOOK_URL = f"{WEBHOOK_HOST}{WEBHOOK_PATH}"

WEBAPP_HOST = "0.0.0.0"
WEBAPP_PORT = int(os.getenv("PORT", "10000"))   # Render خودش PORT رو میده

# ====== Watchlist Symbols ======
WATCHLIST = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
]

# ====== Timeframes ======
TIMEFRAMES = ["15m", "1h", "4h", "1d"]

# ====== Auto-scan Settings ======
AUTO_SCAN_TOP_N = 100
PROBABILITY_THRESHOLD = 50     # سیگنال‌های بالای ۵۰٪
SCAN_INTERVAL_SECONDS = 60     # هر ۶۰ ثانیه اسکن

# ====== Signal Scoring ======
SCORE_STRONG = 100
SCORE_GOOD = 80
SCORE_WEAK = 60

# ====== Storage / Logging ======
DB_PATH = "signals.db"
LOG_FILE = "bot.log"

# ====== Flags ======
ENABLE_AUTO_SCAN = True   # اسکن خودکار ۱۰۰ ارز برتر
ENABLE_WEBHOOK = False    # اول تست کن با polling
USE_BINANCE_WS = True     # استفاده از websocket برای سرعت بیشتر
