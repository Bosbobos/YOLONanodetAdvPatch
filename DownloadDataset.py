import os
import numpy as np
from PIL import Image
from torchvision import transforms
import requests
import zipfile
from io import BytesIO
from tqdm import tqdm  # Для прогресс-бара
import json

# Конфигурация
DATA_DIR = "coco_data"  # Директория для данных
VAL_IMAGE_DIR = os.path.join(DATA_DIR, "val2017")
ANNOTATIONS_DIR = os.path.join(DATA_DIR, "annotations")
VAL_ANNOTATIONS_PATH = os.path.join(ANNOTATIONS_DIR, "instances_val2017.json")
NUMBER_CHANNELS = 3
INPUT_SHAPE = (NUMBER_CHANNELS, 640, 640)
BATCH_SIZE = 32  # Для оптимизации использования памяти

# Создаем директории
os.makedirs(VAL_IMAGE_DIR, exist_ok=True)
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

# Скачиваем и распаковываем данные (если ещё не скачаны)
# ---------------------------------------------------------------------
# 1. Изображения val2017
if not os.path.exists(VAL_IMAGE_DIR) or len(os.listdir(VAL_IMAGE_DIR)) == 0:
    print("Downloading val2017 images...")
    val_url = "http://images.cocodataset.org/zips/val2017.zip"
    response = requests.get(val_url, stream=True)
    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print("Images extracted to:", VAL_IMAGE_DIR)

# 2. Аннотации
if not os.path.exists(VAL_ANNOTATIONS_PATH):
    print("Downloading annotations...")
    ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    response = requests.get(ann_url, stream=True)
    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print("Annotations extracted to:", ANNOTATIONS_DIR)

# Получаем список файлов изображений из аннотаций
# ---------------------------------------------------------------------
with open(VAL_ANNOTATIONS_PATH, "r") as f:
    annotations = json.load(f)

# Собираем все image_id и соответствующие имена файлов
image_id_to_file = {img["id"]: img["file_name"] for img in annotations["images"]}
image_files = [os.path.join(VAL_IMAGE_DIR, image_id_to_file[id]) for id in image_id_to_file]

# Проверяем существование файлов
valid_image_files = [f for f in image_files if os.path.exists(f)]
print(f"Found {len(valid_image_files)} valid images out of {len(image_files)}")
