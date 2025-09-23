import logging
import os
from aiogram import Bot, Dispatcher, types
from aiogram.utils.executor import start_webhook

# ==============================
# Bot Token and Admin ID
API_TOKEN = "7993216439:AAHKSHJMHrcnfEcedw54aetp1JPxZ83Ks4M"
ADMIN_ID = 84544682

# ==============================
# Webhook settings for Render
WEBHOOK_HOST = f"https://{os.getenv('RENDER_EXTERNAL_HOSTNAME')}"
WEBHOOK_PATH = f"/webhook/{API_TOKEN}"
WEBHOOK_URL = f"{WEBHOOK_HOST}{WEBHOOK_PATH}"

WEBAPP_HOST = "0.0.0.0"
WEBAPP_PORT = int(os.getenv("PORT", 5000))

# ==============================
logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

# ==============================
# /start command
@dp.message_handler(commands=["start"])
async def start_cmd(message: types.Message):
    if message.from_user.id != ADMIN_ID:
        await message.answer("Access denied ❌")
        return
    await message.answer("Bot is running with webhook ✅")

# /signal command
@dp.message_handler(commands=["signal"])
async def signal_cmd(message: types.Message):
    if message.from_user.id != ADMIN_ID:
        await message.answer("Access denied ❌")
        return
    
    response = (
        "📊 Pair: BTC/USDT\n"
        "⏱ Timeframe: Intraday\n"
        "🎯 Entry: 27,450 USDT\n"
        "🛑 Stop Loss: 27,100 USDT\n"
        "✅ Take Profit (TP1): 27,900 USDT\n"
        "✅ Take Profit (TP2): 28,300 USDT\n"
        "⚠️ Low risk, based on bullish correction\n"
    )
    await message.answer(response)

# ==============================
# Webhook events
async def on_startup(dp):
    logging.warning("Starting webhook...")
    await bot.set_webhook(WEBHOOK_URL)

async def on_shutdown(dp):
    logging.warning("Shutting down webhook...")
    await bot.delete_webhook()
    await bot.close()

# ==============================
if __name__ == "__main__":
    start_webhook(
        dispatcher=dp,
        webhook_path=WEBHOOK_PATH,
        on_startup=on_startup,
        on_shutdown=on_shutdown,
        skip_updates=True,
        host=WEBAPP_HOST,
        port=WEBAPP_PORT,
    )
