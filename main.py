import logging
from aiogram import Bot, Dispatcher, types
from aiogram.utils.executor import start_webhook

API_TOKEN = "7993216439:AAHKSHJMHrcnfEcedw54aetp1JPxZ83Ks4M"
ADMIN_ID = 84544682

# Webhook settings
WEBHOOK_HOST = "https://robatmohsen2bot.onrender.com"
WEBHOOK_PATH = "/webhook"
WEBHOOK_URL = f"{WEBHOOK_HOST}{WEBHOOK_PATH}"

WEBAPP_HOST = "0.0.0.0"
WEBAPP_PORT = 10000

logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

# start command
@dp.message_handler(commands=["start"])
async def start_cmd(message: types.Message):
    if message.from_user.id != ADMIN_ID:
        await message.answer("⛔ Access denied")
        return
    await message.answer("✅ Bot is running")

# handle messages
@dp.message_handler()
async def handle_msg(message: types.Message):
    if message.from_user.id != ADMIN_ID:
        return

    response = f"""
📊 Pair: BTC/USDT
⏰ Timeframe: Intraday
🎯 Entry: 27,450 USDT
🛑 Stop Loss: 27,100 USDT
🎯 Take Profit 1: 27,900 USDT
🎯 Take Profit 2: 28,300 USDT
⚠️ Low risk setup based on bullish correction
"""
    await message.answer(response)

# startup webhook
async def on_startup(dp):
    await bot.set_webhook(WEBHOOK_URL)

# shutdown webhook
async def on_shutdown(dp):
    logging.warning("Shutting down..")
    await bot.delete_webhook()
    await bot.close()

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
