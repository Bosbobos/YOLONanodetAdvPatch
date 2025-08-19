import os
import json
import numpy as np

DATASET_DIR = './second_dataset'

# Загрузка аннотаций
with open(os.path.join(DATASET_DIR, 'ann', 'annotations.json')) as f:
    data = json.load(f)

# Фильтрация и обработка
filtered_anns = {}
for ann in data['annotations']:
    if ann['category_id'] == 2:  # ID класса 'gun' :cite[1]
        img_id = ann['image_id']
        current_conf = ann.get('confidence', 0.5)  # Используем confidence если есть

        # Оставляем только один bbox с максимальным confidence
        if img_id not in filtered_anns or current_conf > filtered_anns[img_id]['confidence']:
            filtered_anns[img_id] = {
                'bbox': ann['bbox'],
                'confidence': current_conf
            }

# Конвертация в YOLO формат
for img in data['images']:
    img_id = img['id']
    if img_id in filtered_anns:
        bbox = filtered_anns[img_id]['bbox']
        x_center = (bbox[0] + bbox[2] / 2) / img['width']
        y_center = (bbox[1] + bbox[3] / 2) / img['height']
        width = bbox[2] / img['width']
        height = bbox[3] / img['height']

        # Сохранение в файл
        with open(f"{DATASET_DIR}/labels/{img['file_name'].replace('.jpg', '.txt')}", 'w') as f:
            f.write(f"0 {x_center:.8f} {y_center:.8f} {width:.8f} {height:.8f}")