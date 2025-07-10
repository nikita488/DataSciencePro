import os
import torch
import json
from enum import Enum

from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv
from PIL import Image

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from telegram.constants import ChatAction

class ImageOrientation(Enum):
    LANDSCAPE = 0
    PORTRAIT = 1
    SQUARE = 2

    @classmethod
    def from_size(cls, width: int, height: int) -> 'ImageOrientation':
        if height > width:
            return cls.PORTRAIT
        elif width > height:
            return cls.LANDSCAPE
        else:
            return cls.SQUARE

    def min_width(self) -> int:
        match self:
            case ImageOrientation.LANDSCAPE:
                return 640
            case ImageOrientation.PORTRAIT | ImageOrientation.SQUARE:
                return 480
        
    def min_height(self) -> int:
        match self:
            case ImageOrientation.LANDSCAPE | ImageOrientation.SQUARE:
                return 480
            case ImageOrientation.PORTRAIT:
                return 640

debug = False

init_time = datetime.now()

print('Инициализация...')

# Загружаем .env файл и получаем токен бота
load_dotenv()
BOT_TOKEN = os.getenv('BOT_TOKEN')

LOG_PATH = Path.cwd() / 'logs' / 'log.json'
IMAGE_LOG_PATH = LOG_PATH.parent / 'images'
IMAGE_LOG_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = 'model.pt'

# Загружаем модель
model = torch.hub.load('ultralytics/yolov5', 'custom', MODEL_NAME)

# Параметры модели
model.conf = 0.75 # минимальный порог confidence, при котором результат будет допустимым
model.iou = 0.45
model.agnostic = False
model.multi_label = False
model.max_det = 1000

scale = 1.0
size = int(640 * scale)
augment = True

image_min_width = 640
image_min_height = 480

plate_min_width = 16
plate_min_height = 16

def get_current_time():
    return int(datetime.now(timezone.utc).timestamp())

def save_log(annotation, cur_time, image_path, user_id, user_name, valid):
    key = str(cur_time)
    
    if LOG_PATH.exists():
        data = json.loads(LOG_PATH.read_text(encoding='utf-8'))
    else:
        data = {}

    plates = data.get(key, [])

    height, width, _ = annotation['im'].shape

    plate = {
        'valid': valid,
        'time': cur_time,
        'date': str(datetime.fromtimestamp(cur_time)),
        'conf': annotation['conf'].item(),
        'image': str(image_path),
        'width': width,
        'height': height,
        'user_id': user_id,
        'user_name': user_name,
    }

    plates.append(plate)
    data[key] = plates

    with open(LOG_PATH, 'w', encoding='utf-8') as log_file:
        json.dump(data, log_file, indent=4)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = [
        f'Привет, {update.effective_user.first_name}!',
        '',
        'Пришли мне фотографию транспортного средства, и я автоматически определю его номерные знаки =)',
        '',
        'Требования к фотографии:',
        '- фотография должна содержать транспортное средство целиком или его часть (фотография только номерного знака крупным планом не подойдет);',
        '- транспортное средство на фотографии должно содержать хотя бы один номерной знак;'
        '- номерные знаки должны располагаться в привычных местах на транспортном средстве;',
        '- фотография должна быть хорошего качества и высокого разрешения (не менее 640x480, 480x640, 480x480 пикселей);',
    ]
    await update.message.reply_text('\n'.join(message))

async def photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    
    await update.message.reply_chat_action(action=ChatAction.TYPING)
    
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    data = await file.download_as_bytearray()

    cur_time = get_current_time()

    image_file = IMAGE_LOG_PATH / f'{cur_time}.jpg'
    image_file.write_bytes(data)

    with BytesIO(data) as td:
        img = Image.open(td)

        width, height = img.size
        orientation = ImageOrientation.from_size(width, height)

        if width < orientation.min_width() or height < orientation.min_height():
            await update.message.reply_text(f'Разрешение фотографии слишком низкое для обнаружения...\nТребования к фотографиям описаны в команде /start')
            return
    
        results = model(img, size=size, augment=augment)

        if debug:
            last_ims = results.ims.copy()
            rendered = results.render()

            results.ims = last_ims

            for render_data in rendered:
                buffered = BytesIO()

                annotation_image = Image.fromarray(render_data)
                annotation_image.save(buffered, format='JPEG', quality=100, subsampling=0)
                await update.message.reply_photo(buffered.getvalue())

            results.print()
            results.save()

        annotated_images = results.crop(save=False)
        processed = 0

        for annotation in annotated_images:
            valid = True

            conf = annotation['conf']
            im = annotation['im']
            height, width, _ = im.shape

            if width < plate_min_width or height < plate_min_height:
                valid = False
            
            save_log(annotation, cur_time, image_file, user.id, user.username, valid)

            if valid:
                buffered = BytesIO()

                annotation_data = im[:, :, ::-1] # bgr -> rgb
                annotation_image = Image.fromarray(annotation_data)
                annotation_image.save(buffered, format='JPEG', quality=100, subsampling=0)
    
                await update.message.reply_photo(buffered.getvalue(), f'Уверенность: {(conf * 100):.1f}%')
                processed += 1

        if processed <= 0:
            await update.message.reply_text(f'Номерные знаки не обнаружены =(\nТребования к фотографии описаны в команде /start')

# Инициализируем бота
app = ApplicationBuilder().token(BOT_TOKEN).build()

app.add_handler(CommandHandler('start', start))
app.add_handler(MessageHandler(filters.PHOTO, photo))

init_duration = (datetime.now() - init_time).total_seconds()
print(f'Инициализация завершена. Длительность: {init_duration:.1f} сек.')

# Запуск
app.run_polling()
