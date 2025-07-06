import os
import torch

from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv
from PIL import Image

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from telegram.constants import ChatAction

# Загружаем .env файл и получаем токен бота
load_dotenv()
BOT_TOKEN = os.getenv('BOT_TOKEN')

SAVE_PATH = Path.cwd() / 'inputs'
SAVE_PATH.mkdir(parents=True, exist_ok=True)

# Загружаем модель
model = torch.hub.load("ultralytics/yolov5", "custom", "model.pt")

# Параметры модели
model.conf = 0.25
model.iou = 0.45
model.agnostic = False
model.multi_label = False
model.max_det = 1000

def get_current_time():
    return int(datetime.now(timezone.utc).timestamp())

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(f'Привет, {update.effective_user.first_name}!\nПришли мне фотографию, и я определю есть ли на ней автомобильные номера =)')

async def photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_chat_action(action=ChatAction.TYPING)
    
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    data = await file.download_as_bytearray()

    image_file = SAVE_PATH / f'{get_current_time()}.jpg'
    image_file.write_bytes(data)

    with BytesIO(data) as td:
        img = Image.open(td)
    
        results = model(img, size=640)
    
        results.print()
        results.save()

        annotated_images = results.crop()

        if not annotated_images:
            await update.message.reply_text(f'Автомобильные номера не обнаружены =(')
            return

        for annotation in annotated_images:
            buffered = BytesIO()

            annotation_data = annotation["im"][:, :, ::-1] # bgr -> rgb
            
            annotation_image = Image.fromarray(annotation_data)
            annotation_image.save(buffered, format="JPEG", quality=100, subsampling=0)

            conf = annotation["conf"]

            await update.message.reply_photo(buffered.getvalue(), f'Уверенность: {conf:.4f}')

# Инициализируем бота
app = ApplicationBuilder().token(BOT_TOKEN).build()

app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.PHOTO, photo))

print('Инициализация завершена.')

# Запуск
app.run_polling()
