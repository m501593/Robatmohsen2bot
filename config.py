# config.py
# Telegram and runtime configuration (no Persian text - only English)

# Telegram bot token (from BotFather)
TELEGRAM_TOKEN = "7993216439:AAHKSHJMHrcnfEcedw54a..."  # <- your full token (replace ... with the rest if trimmed)

# Numeric admin/chat id (use integer)
ADMIN_ID = 84544682  # your numeric id

# Binance API keys (optional - leave empty for read-only/public endpoints)
BINANCE_API_KEY = ""
BINANCE_API_SECRET = ""

# Webhook / hosting (for Render or similar)
# Set WEBHOOK_HOST to your render primary URL (https://<service>.onrender.com)
WEBHOOK_HOST = "https://robatmohsen2bot.onrender.com"
WEBHOOK_PATH = "/webhook"
WEBHOOK_URL = f"{WEBHOOK_HOST}{WEBHOOK_PATH}"

# Webapp server bind (internal)
WEBAPP_HOST = "0.0.0.0"
WEBAPP_PORT = 10000

# Watchlist symbols (uppercase, use futures symbols like BTCUSDT)
WATCHLIST = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
]

# Timeframes used for analysis
TIMEFRAMES = ["15m", "1h", "4h", "1d"]

# Auto-scan / limits
AUTO_SCAN_TOP_N = 100            # scan top N tickers automatically
PROBABILITY_THRESHOLD = 50       # minimum probability (%) to send auto-signal
SCAN_INTERVAL_SECONDS = 60       # polling interval in seconds (adjust as needed)

# Signal scoring thresholds (example)
SCORE_STRONG = 100
SCORE_GOOD = 80
SCORE_WEAK = 60

# Database / storage file names (can be adjusted)
DB_PATH = "signals.db"
LOG_FILE = "bot.log"

# Other flags
ENABLE_AUTO_SCAN = True          # scan top N automatically
ENABLE_WEBHOOK = True            # use webhook (set False to use long-polling)
USE_BINANCE_WS = True            # prefer websocket data if implemented
