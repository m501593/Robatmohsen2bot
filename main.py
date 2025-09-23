import logging
from aiogram import Bot, Dispatcher, types
from aiogram.utils.executor import start_webhook

API_TOKEN = "7993216439:AAHKSHJMHrcnfEcedw54aetp1JPxZ83Ks4M"
ADMIN_ID = 84544682

# تنظیمات وبهوک
WEBHOOK_HOST = "https://robatmohsen2bot.onrender.com"
WEBHOOK_PATH = "/webhook"
WEBHOOK_URL = f"{WEBHOOK_HOST}{WEBHOOK_PATH}"

WEBAPP_HOST = "0.0.0.0"
WEBAPP_PORT = 10000

logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

# دستور start
@dp.message_handler(commands=["start"])
async def start_cmd(message: types.Message):
    if message.from_user.id != ADMIN_ID:
        await message.answer("⛔ دسترسی محدود است")
        return
    await message.answer("✅ ربات روشن است و آماده دریافت تحلیل‌ها")

# گرفتن پیام و جواب دادن تحلیل
@dp.message_handler()
async def handle_msg(message: types.Message):
    if message.from_user.id != ADMIN_ID:
        return
    
    response = f"""
📊 جفت ارز: BTC/USDT
⏰ تایم‌فریم: اینترادی
🎯 ورود: 27,450 USDT
🛑 حد ضرر: 27,100 USDT
🎯 حد سود (TP1): 27,900 USDT
🎯 حد سود (TP2): 28,300 USDT
⚠️ هشدار: ریسک پایین، تایید بر اساس اصلاح صعودی
"""
    await message.answer(response)

async def on_startup(dp):
    await bot.set_webhook(WEBHOOK_URL)

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
  
