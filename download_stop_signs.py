from pycocotools.coco import COCO
import os
import shutil

# Пути
json_file = "coco_data/annotations/instances_val2017.json"  # COCO JSON
images_dir = "coco_data/val2017/"  # Папка с изображениями
output_dir = "coco_stop_signs/"  # Выходная папка
os.makedirs(output_dir, exist_ok=True)

# Загружаем COCO
coco = COCO(json_file)

# ID категории "stop sign" в COCO
stop_sign_id = coco.getCatIds(catNms=['stop sign'])[0]  # Обычно 12

# Получаем все изображения с "stop sign"
img_ids = coco.getImgIds(catIds=[stop_sign_id])

# Обрабатываем каждое изображение
for img_id in img_ids:
    img = coco.loadImgs(img_id)[0]
    anns = coco.loadAnns(coco.getAnnIds(img_id, catIds=[stop_sign_id]))

    # Копируем изображение в новую папку
    src_image_path = os.path.join(images_dir, img["file_name"])
    dst_image_path = os.path.join(output_dir, img["file_name"])
    shutil.copy(src_image_path, dst_image_path)

    # Создаем YOLO-файл
    label_file = os.path.join(output_dir, os.path.splitext(img["file_name"])[0] + ".txt")
    with open(label_file, "w") as f:
        for ann in anns:
            # Конвертируем COCO bbox в YOLO-формат
            x, y, w, h = ann["bbox"]
            x_center = (x + w / 2) / img["width"]
            y_center = (y + h / 2) / img["height"]
            w_norm = w / img["width"]
            h_norm = h / img["height"]
            # Класс "stop sign" в YOLO будет 0 (если это единственный класс)
            f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")